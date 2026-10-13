"""HTTP/SSE transport and independent, read-only reference viewers."""
from contextlib import asynccontextmanager
import asyncio
import html
import ipaddress
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, quote, urlencode, urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from .contracts import AssistantId, SubmitTask, TaskSnapshot, WorkspaceCommand, WorkspaceSnapshot
from .auth import (AuthStore, LoginRateLimiter, SESSION_COOKIE_NAME, default_auth_db_path)
from .task_service import BusyError, TaskService

ROOT = Path(__file__).resolve().parents[2]
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_origin(value: str):
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if (scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            return None
        port = parsed.port
        if port == (443 if scheme == "https" else 80):
            port = None
        host = parsed.hostname.rstrip(".").lower()
        return scheme, host, port
    except ValueError:
        return None


def _referer_origin(value: str):
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if (scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            return None
        port = parsed.port
        if port == (443 if scheme == "https" else 80):
            port = None
        return scheme, parsed.hostname.rstrip(".").lower(), port
    except ValueError:
        return None


def _trusted_proxy_peer(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    configured = {item.strip() for item in os.environ.get("RAG_ARCH_AUTH_TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if item.strip()}
    if str(address) in configured:
        return True
    return address.is_loopback


def _login_client_key(request: Request) -> str:
    """Use the proxy-appended client address without trusting spoofed prefixes."""
    peer = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if _trusted_proxy_peer(request) and forwarded_for:
        candidate = forwarded_for.rsplit(",", 1)[-1].strip()
        try:
            return ipaddress.ip_address(candidate).compressed
        except ValueError:
            pass
    return peer


def _origin_is_allowed(request: Request, allowed_origins: set[tuple[str, str, int | None]]) -> bool:
    supplied = request.headers.get("origin")
    if supplied is not None:
        actual = _canonical_origin(supplied)
    elif request.headers.get("sec-fetch-site", "").strip().lower() == "same-origin":
        return True
    else:
        actual = _referer_origin(request.headers.get("referer", ""))
    if actual is None:
        return False
    if allowed_origins:
        return actual in allowed_origins

    scheme = request.url.scheme
    host = request.headers.get("host", request.url.netloc)
    if _trusted_proxy_peer(request):
        scheme = request.headers.get("x-forwarded-proto", scheme).split(",", 1)[0].strip().lower()
        host = request.headers.get("x-forwarded-host", host).split(",", 1)[0].strip()
    expected = _canonical_origin(f"{scheme}://{host}")
    return expected is not None and actual == expected


def _safe_next_path(value: str | None) -> str:
    if not value:
        return "/"
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "/"
    if (not value.startswith("/") or value.startswith("//") or "\\" in value
            or parsed.scheme or parsed.netloc or parsed.path.startswith("//")):
        return "/"
    return value


def _login_page(message: str = "", next_path: str = "/", *, status_code: int = 200) -> HTMLResponse:
    error = f'<p class="error" role="alert">{html.escape(message)}</p>' if message else ""
    target = html.escape(_safe_next_path(next_path), quote=True)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录 · 建筑资料助手</title><style>
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#f4f7f2;color:#354638;font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{width:min(390px,calc(100% - 32px));padding:32px;border:1px solid #e0e9df;border-radius:14px;background:#fff;box-shadow:0 14px 40px #243d2410}}
h1{{margin:0 0 5px;font-size:22px}}.sub{{margin:0 0 24px;color:#7b897c;font-size:13px}}label{{display:block;margin:15px 0 6px;color:#596c5b;font-size:13px}}
input{{width:100%;height:42px;padding:8px 10px;border:1px solid #dbe5da;border-radius:7px;font:inherit}}input:focus{{outline:2px solid #9fc6a4;outline-offset:1px}}
button{{width:100%;margin-top:22px;height:43px;border:0;border-radius:7px;background:#397d5c;color:#fff;font:600 14px system-ui,-apple-system,"Segoe UI",sans-serif;cursor:pointer}}button:hover{{background:#306d50}}
.note{{margin:20px 0 0;padding-top:14px;border-top:1px solid #edf1ec;color:#7d897d;font-size:12px}}.error{{margin:0 0 12px;color:#b34d42;font-size:13px}}
</style></head><body><main><h1>建筑资料助手</h1><p class="sub">请使用管理员预先创建的账号登录</p>{error}
<form method="post" action="/login"><input type="hidden" name="next" value="{target}">
<label for="username">账号</label><input id="username" name="username" autocomplete="username" required maxlength="64">
<label for="password">密码</label><input id="password" name="password" type="password" autocomplete="current-password" required maxlength="1024">
<button type="submit">登录</button></form><p class="note">每个账号单独登录；当前工作区和已上传资料由所有登录用户共同使用，请勿上传只应由个人查看的内容。</p></main></body></html>"""
    return HTMLResponse(document, status_code=status_code, headers={
        "Cache-Control": "no-store", "Clear-Site-Data": '"cache"'})


def _private_response(response: Response) -> Response:
    """Keep authenticated HTML/API responses out of browser and proxy caches."""
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Pragma"] = "no-cache"
    vary = {item.strip() for item in response.headers.get("Vary", "").split(",") if item.strip()}
    vary.add("Cookie")
    response.headers["Vary"] = ", ".join(sorted(vary))
    return response


def resume_cursor(after: int, last_event_id: str | None, instance: str):
    """A refreshed EventSource uses ?after; its own reconnect header wins later."""
    if not last_event_id:
        return after, None
    try:
        previous_instance, number = last_event_id.rsplit(":", 1)
        if previous_instance != instance:
            return after, "server_restarted"
        value = int(number)
        return (value, None) if value >= 0 else (after, "invalid_cursor")
    except ValueError:
        return after, "invalid_cursor"


async def event_stream(service, after, *, last_event_id=None):
    cursor, reason = resume_cursor(after, last_event_id, service.server_instance_id)
    if reason:
        yield ServerSentEvent(event="snapshot_required", data={
            "reason": reason, "server_instance_id": service.server_instance_id})
        return
    signal = asyncio.Event()
    unsubscribe = service.events.subscribe(asyncio.get_running_loop(), signal.set)
    try:
        while True:
            # Clear BEFORE reading: a worker publishing between read and await
            # schedules set() on this loop, so the wakeup cannot be lost.
            signal.clear()
            events, reason = service.events.read_after(cursor)
            if reason:
                yield ServerSentEvent(event="snapshot_required", data={
                    "reason": reason, "server_instance_id": service.server_instance_id})
                return
            for item in events:
                cursor = item["seq"]
                yield ServerSentEvent(event=item["type"],
                    id=f"{service.server_instance_id}:{cursor}", data=item["data"])
            if not events:
                await signal.wait()
    finally:
        unsubscribe()


def create_app(service=None, *, frontend_dir=None, resources=None, documents=None, root=ROOT,
               auth_enabled=True, auth_store=None, auth_db_path=None, auth_cookie_secure=None,
               auth_session_ttl_seconds=None, auth_rate_limiter=None):
    from .standards_resources import StandardsResources
    from .standards_routes import create_standards_router
    from .document_routes import create_document_router, validate_task_documents
    materials = resources if resources is not None else StandardsResources(root)
    if auth_enabled:
        ttl_seconds = (auth_session_ttl_seconds if auth_session_ttl_seconds is not None
                       else _env_positive_int("RAG_ARCH_AUTH_SESSION_HOURS", 24 * 14) * 60 * 60)
        secure_cookie = (auth_cookie_secure if auth_cookie_secure is not None
                         else _env_bool("RAG_ARCH_AUTH_COOKIE_SECURE", True))
        authentication = auth_store or AuthStore(
            default_auth_db_path(root, auth_db_path), session_ttl_seconds=ttl_seconds)
        limiter = auth_rate_limiter or LoginRateLimiter()
        raw_allowed_origins = os.environ.get("RAG_ARCH_AUTH_ALLOWED_ORIGINS", "")
        allowed_origins = set()
        for item in raw_allowed_origins.split(","):
            if not item.strip():
                continue
            parsed = _canonical_origin(item.strip())
            if parsed is None:
                raise ValueError("RAG_ARCH_AUTH_ALLOWED_ORIGINS contains an invalid origin")
            allowed_origins.add(parsed)
    else:
        authentication = None
        limiter = None
        secure_cookie = False
        allowed_origins = set()

    @asynccontextmanager
    async def lifespan(app):
        app.state.standards = materials
        if service is not None:
            app.state.tasks = service
        else:
            from rag_arch.config import load_settings
            from .chat_runner import run_chat
            from .standards_runner import run_standards
            from .plan_runner import run_plan
            settings = load_settings()
            materials.configure_catalog(settings)
            materials.configure_embeddings(settings)

            def execute(request, progress, cancel_event):
                if request["assistant_id"] == "chat":
                    return run_chat(request, progress, cancel_event, settings=settings,
                                    documents=app.state.documents)
                if request["assistant_id"] == "standards":
                    return run_standards(request, progress, cancel_event, settings=settings, resources=materials)
                return run_plan(request, progress, cancel_event, settings=settings,
                                resources=materials, documents=app.state.documents)

            app.state.tasks = TaskService(runner=execute, execution_modes={"chat": "model", "standards": "model", "plan": "model"},
                                          model_name=settings.chat_model or None)
        from .documents import DocumentStore
        app.state.documents = documents if documents is not None else DocumentStore(
            root, app.state.tasks.workspace_id, materials.pdf_backend)
        try:
            yield
        finally:
            app.state.tasks.close()
            app.state.documents.close()
            materials.close()

    app = FastAPI(title="建筑资料助手 · Web工作台", version="0.2.0", lifespan=lifespan,
                  docs_url=None if auth_enabled else "/docs",
                  redoc_url=None if auth_enabled else "/redoc",
                  openapi_url=None if auth_enabled else "/openapi.json")
    app.state.auth_enabled = auth_enabled
    app.state.auth_store = authentication
    app.state.auth_cookie_secure = secure_cookie
    app.state.auth_rate_limiter = limiter

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if not auth_enabled:
            return await call_next(request)

        if (request.method in UNSAFE_METHODS
                and not _origin_is_allowed(request, allowed_origins)):
            if request.url.path == "/login":
                return _private_response(_login_page("登录请求来源异常，请刷新页面后重试。", status_code=403))
            return _private_response(JSONResponse({"detail": "请求来源校验失败。"}, status_code=403))

        username = authentication.resolve_session(request.cookies.get(SESSION_COOKIE_NAME))
        request.state.auth_user = username
        is_public = ((request.url.path == "/api/health" and request.method == "GET")
                     or (request.url.path == "/login" and request.method in {"GET", "POST"})
                     or (request.url.path == "/logout" and request.method == "POST"))
        if is_public:
            if request.url.path == "/login" and request.method == "GET" and username:
                return _private_response(RedirectResponse("/", status_code=303))
            return _private_response(await call_next(request))

        if username is None:
            if request.url.path.startswith("/api/") or request.url.path.startswith("/assets/"):
                response = JSONResponse({"detail": "请先登录。"}, status_code=401,
                                        headers={"Clear-Site-Data": '"cache"'})
                return _private_response(response)
            requested = request.url.path
            if request.url.query:
                requested += "?" + request.url.query
            location = "/login?" + urlencode({"next": requested})
            return _private_response(RedirectResponse(location, status_code=303))
        return _private_response(await call_next(request))

    dist = Path(frontend_dir) if frontend_dir is not None else ROOT / "frontend/dist"
    app.include_router(create_standards_router())
    app.include_router(create_document_router())
    from .history import create_history_router
    app.include_router(create_history_router(root))

    def current(request):
        return request.app.state.tasks

    def check_workspace(tasks, workspace_id):
        if workspace_id != tasks.workspace_id:
            raise HTTPException(409, "工作区已失效，请重新获取当前工作区。")

    def get_task(tasks, task_id):
        try:
            return tasks.get_task(task_id)
        except KeyError:
            raise HTTPException(404, "未找到本次后端运行中的任务。") from None

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "mode": current(request).snapshot()["mode"],
                "server_instance_id": current(request).server_instance_id}

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request):
        return _login_page(next_path=request.query_params.get("next", "/"))

    @app.post("/login", include_in_schema=False)
    async def login(request: Request):
        client_host = _login_client_key(request)
        limited, retry_after = limiter.is_limited(client_host)
        if limited:
            page = _login_page("登录尝试过多，请稍后再试。", status_code=429)
            page.headers["Retry-After"] = str(retry_after)
            return page

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > 8192:
                    return Response("登录表单过大。", status_code=413)
            except ValueError:
                return Response("无效的 Content-Length。", status_code=400)
        body = await request.body()
        if len(body) > 8192:
            return Response("登录表单过大。", status_code=413)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/x-www-form-urlencoded":
            return Response("登录表单格式无效。", status_code=415)
        try:
            values = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=16)
        except (UnicodeDecodeError, ValueError):
            return Response("登录表单格式无效。", status_code=400)
        username = values.get("username", [""])[0]
        password = values.get("password", [""])[0]
        next_path = _safe_next_path(values.get("next", ["/"])[0])
        verified = authentication.verify_user(username, password)
        if verified is None:
            limiter.failed(client_host)
            return _login_page("账号不存在或密码不正确。", next_path, status_code=401)
        limiter.succeeded(client_host)
        token, expires_at = authentication.create_session(verified)
        response = RedirectResponse(next_path, status_code=303)
        response.set_cookie(SESSION_COOKIE_NAME, token, max_age=max(0, expires_at - int(time.time())),
                            httponly=True, secure=secure_cookie, samesite="lax", path="/")
        return response

    @app.post("/logout", include_in_schema=False)
    def logout(request: Request):
        authentication.revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/", secure=secure_cookie,
                              httponly=True, samesite="lax")
        response.headers["Clear-Site-Data"] = '"cache", "storage"'
        return response

    @app.get("/api/workspace", response_model=WorkspaceSnapshot)
    def workspace(request: Request):
        return current(request).snapshot()

    @app.post("/api/tasks", response_model=TaskSnapshot, status_code=202)
    def submit(body: SubmitTask, request: Request):
        tasks = current(request)
        check_workspace(tasks, body.workspace_id)
        try:
            if tasks.execution_mode(body.assistant_id) == "model":
                validate_task_documents(body, request.app.state.documents)
            return tasks.submit(body.assistant_id, body.request_id, body.prompt,
                                duration_seconds=body.duration_seconds, fail_at_end=body.fail_at_end,
                                attachment_ids=body.attachment_ids, registered_plan_id=body.registered_plan_id,
                                plan_action=body.plan_action, use_embeddings=body.use_embeddings,
                                max_steps=body.max_steps, max_tool_calls=body.max_tool_calls)
        except BusyError:
            raise HTTPException(409, "这个助手已有任务在运行，请等待完成或停止后再提交。") from None
        except KeyError:
            raise HTTPException(404, "所选附件已失效，请重新上传。") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/tasks/{task_id}", response_model=TaskSnapshot)
    def task(task_id: str, workspace_id: str, request: Request):
        tasks = current(request)
        check_workspace(tasks, workspace_id)
        return get_task(tasks, task_id)

    @app.post("/api/tasks/{task_id}/cancel", response_model=TaskSnapshot)
    def cancel(task_id: str, body: WorkspaceCommand, request: Request):
        tasks = current(request)
        check_workspace(tasks, body.workspace_id)
        get_task(tasks, task_id)
        return tasks.cancel(task_id)

    @app.post("/api/assistants/{assistant_id}/reset", response_model=WorkspaceSnapshot)
    def reset(assistant_id: AssistantId, body: WorkspaceCommand, request: Request):
        tasks = current(request)
        check_workspace(tasks, body.workspace_id)
        try:
            tasks.reset(assistant_id)
        except BusyError:
            raise HTTPException(409, "请先完成或停止这个助手的任务，再清空对话。") from None
        return tasks.snapshot()

    def sse_service(request: Request, workspace_id: str):
        tasks = current(request)
        check_workspace(tasks, workspace_id)
        return tasks

    @app.get("/api/events", response_class=EventSourceResponse)
    async def events(tasks: TaskService = Depends(sse_service),
                     server_instance_id: str | None = None, after: int = Query(default=0, ge=0),
                     last_event_id: str | None = Header(default=None)):
        if server_instance_id is not None and server_instance_id != tasks.server_instance_id:
            yield ServerSentEvent(event="snapshot_required", data={
                "reason": "server_restarted", "server_instance_id": tasks.server_instance_id})
            return
        async for item in event_stream(tasks, after, last_event_id=last_event_id):
            yield item

    @app.get("/api/tasks/{task_id}/download")
    def download(task_id: str, workspace_id: str, request: Request, format: str = Query(pattern="^(json|md|debug)$")):
        tasks = current(request)
        check_workspace(tasks, workspace_id)
        result = get_task(tasks, task_id)
        answer = result.get("result") or {}
        if format == "debug":
            audit = answer.get("audit_log") or {}
            directory = Path(audit.get("directory") or ".").resolve()
            audit_root = (Path(root) / "data/runtime/conversations").resolve()
            path = directory / "result.json"
            if not directory.is_relative_to(audit_root) or not path.is_file():
                raise HTTPException(404, "本次没有可下载的调试记录；异常日志仍保留在本地。")
            return FileResponse(path, media_type="application/json", filename=f"调试记录_{task_id}.json")
        label = "演示记录" if result["simulation"] else "核查报告" if answer.get("report_markdown") else "对话记录"
        filename = f"{label}_{task_id}.{format}"
        if format == "json":
            content = json.dumps(result, ensure_ascii=False, indent=2)
            mime = "application/json"
        else:
            description = ("未调用真实模型、未读取资料。" if result["simulation"]
                           else "真实规范库问答；引用和工具记录见本轮 JSON 与本地审计日志。"
                           if result["assistant_id"] == "standards"
                           else "方案资料核查与追问；处理范围及未完成事项见本轮记录。"
                           if result["assistant_id"] == "plan"
                           else "基于本次选择文件的模型问答；来源和提取限制见本轮记录。"
                           if result.get("attachment_ids")
                           else "真实模型对话；本轮未上传或读取本地资料。")
            validation_labels = {
                "citation_rejected": "引用未通过核验；以下保留模型原文，仅供检查。",
                "invalid_result": "输出格式未通过核验；以下保留模型原文，仅供检查。",
                "insufficient_evidence": "资料不足以支持结论。",
                "model_output_limit": "模型输出未完整返回，回答尚未通过核验。",
            }
            validation = validation_labels.get(answer.get("termination_reason"))
            if validation is None and answer.get("status") not in {None, "answered", "grounded"}:
                validation = "回答尚未通过核验；以下原文不能作为已验证结论。"
            warnings = [warning for warning in answer.get("warnings", []) if isinstance(warning, str)]
            notices = ([validation] if validation else []) + warnings
            notice_text = "".join(
                "> " + notice.replace("\n", "\n> ") + "\n\n" for notice in dict.fromkeys(notices))
            content = answer.get("report_markdown") or (f"# {label}\n\n{description}\n\n"
                       f"任务：{task_id}\n\n任务状态：{result['status']}\n\n"
                       f"输入：{result['prompt']}\n\n" + notice_text
                       + (answer.get("text") or result.get("error") or result["detail"]))
            mime = "text/markdown"
        return Response(content, media_type=mime,
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"})

    @app.get("/", include_in_schema=False)
    @app.get("/viewer/pages", include_in_schema=False)
    def home():
        index = dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse("<h1>Web 后端已启动</h1><p>前端尚未构建，请在 frontend 目录运行 npm run build。</p>")

    app.mount("/assets", StaticFiles(directory=dist / "assets", check_dir=False), name="assets")
    return app


app = create_app()
