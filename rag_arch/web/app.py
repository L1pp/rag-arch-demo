"""HTTP/SSE transport and independent, read-only reference viewers."""
from contextlib import asynccontextmanager
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
from starlette.concurrency import run_in_threadpool

from .contracts import (AssistantId, SubmitTask, TaskSnapshot, WorkspaceCommand, WorkspaceSnapshot,
                        AuthLogin, AuthSessionSnapshot, AuthLogout)
from .auth import (LoginRateLimiter, SESSION_COOKIE_NAME, create_auth_store, ascii_username_key)
from .events import SSEConnectionLimiter, event_stream, resume_cursor
from .task_service import BusyError, TaskCapacityError, TaskService
from .tenancy import current_documents, current_tasks, TenantRegistry, TenantCapacityError

ROOT = Path(__file__).resolve().parents[2]
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _idempotency_conflict_type():
    # The durable repository is imported only when a task needs this error
    # mapping; legacy injected test services do not initialize a database.
    from .task_repository import IdempotencyConflict
    return IdempotencyConflict


def _event_session_check(authentication, token, expected_user_id, service_user_id):
    """Re-resolve the original cookie identity without rechecking a password."""
    async def check():
        if (not token or expected_user_id is None
                or (service_user_id is not None
                    and service_user_id != expected_user_id)):
            return False
        try:
            identity = await run_in_threadpool(authentication.resolve_identity, token)
        except Exception:
            return None
        return identity is not None and identity.user_id == expected_user_id
    return check


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


def _origin_string(origin):
    scheme, host, port = origin
    host = f"[{host}]" if ":" in host else host
    return f"{scheme}://{host}" + (f":{port}" if port is not None else "")


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


def _login_page(message: str = "", next_path: str = "/", *, status_code: int = 200,
                shared_workspace: bool = True) -> HTMLResponse:
    error = f'<p class="error" role="alert">{html.escape(message)}</p>' if message else ""
    target = html.escape(_safe_next_path(next_path), quote=True)
    privacy_note = ("每个账号单独登录；当前工作区和已上传资料由所有登录用户共同使用，请勿上传只应由个人查看的内容。"
                    if shared_workspace else "每个账号使用独立工作区；个人附件、任务和对话仅限本人访问。")
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
<button type="submit">登录</button></form><p class="note">{privacy_note}</p></main></body></html>"""
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


def create_app(service=None, *, frontend_dir=None, resources=None, documents=None, root=ROOT,
               auth_enabled=True, auth_store=None, auth_db_path=None, auth_cookie_secure=None,
               auth_session_ttl_seconds=None, auth_rate_limiter=None, database=None,
               task_runner=None, execution_modes=None, tenant_max_workers=6,
               tenant_max_users=32, tenant_max_tasks=128, task_backend=None,
               sse_max_connections=128, sse_max_connections_per_user=4,
               serve_frontend=True, frontend_origins=None, auth_cookie_samesite=None):
    if type(serve_frontend) is not bool:
        raise ValueError("serve_frontend must be a boolean")
    backend = task_backend or ("durable" if auth_enabled and service is None else "memory")
    if backend not in {"durable", "memory"}:
        raise ValueError("task_backend must be durable or memory")
    if backend == "durable" and (not auth_enabled or service is not None):
        raise ValueError("持久任务需要认证，不能注入共享任务服务。")
    from .standards_resources import StandardsResources
    from .standards_routes import create_standards_router
    from .document_routes import create_document_router, validate_task_documents
    materials = resources if resources is not None else StandardsResources(root)
    if auth_enabled:
        ttl_seconds = (auth_session_ttl_seconds if auth_session_ttl_seconds is not None
                       else _env_positive_int("RAG_ARCH_AUTH_SESSION_HOURS", 24 * 14) * 60 * 60)
        secure_cookie = (auth_cookie_secure if auth_cookie_secure is not None
                         else _env_bool("RAG_ARCH_AUTH_COOKIE_SECURE", True))
        if auth_store is not None:
            authentication = auth_store
        elif database is not None:
            from .auth import PostgresAuthStore
            authentication = PostgresAuthStore(database, session_ttl_seconds=ttl_seconds)
        else:
            authentication = create_auth_store(root, auth_db_path, session_ttl_seconds=ttl_seconds)
        auth_database = getattr(authentication, "database", None)
        if database is not None and (auth_database is None or
                (database.url, database.schema) != (auth_database.url, auth_database.schema)):
            raise ValueError("认证与用户数据必须使用同一数据库和schema。")
        database = database or auth_database
        if auth_rate_limiter is not None:
            limiter = auth_rate_limiter
        elif database is not None:
            from .auth import PostgresLoginRateLimiter
            limiter = PostgresLoginRateLimiter(database)
        else:
            limiter = LoginRateLimiter()
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

    same_site = (auth_cookie_samesite or os.environ.get("RAG_ARCH_AUTH_COOKIE_SAMESITE", "lax")).lower()
    if same_site not in {"lax", "strict", "none"} or (same_site == "none" and not secure_cookie):
        raise ValueError("Cookie SameSite must be lax/strict, or none with Secure enabled")
    raw_frontends = (os.environ.get("RAG_ARCH_FRONTEND_ORIGINS", "").split(",")
                     if frontend_origins is None else frontend_origins)
    if not isinstance(raw_frontends, (list, tuple)):
        raise ValueError("frontend_origins must be an explicit origin list")
    cors_origins = set()
    for item in raw_frontends:
        if not isinstance(item, str):
            raise ValueError("invalid frontend origin")
        if not item.strip():
            continue
        parsed = _canonical_origin(item.strip())
        if parsed is None:
            raise ValueError("invalid frontend origin")
        cors_origins.add(parsed)
    allowed_origins |= cors_origins

    def auth_page(*args, **kwargs):
        return _login_page(*args, **kwargs, shared_workspace=service is not None)

    @asynccontextmanager
    async def lifespan(app):
        app.state.standards = materials
        app.state.tenants = None
        if service is not None:
            app.state.tasks = service
        elif backend == "durable":
            if database is None:
                raise RuntimeError("多用户服务需要先配置并显式迁移 RAG_ARCH_DATABASE_URL。")
            if task_runner is not None or documents is not None:
                raise ValueError("持久任务API不能注入任务runner或共享DocumentStore；请独立启动worker。")
            from .tenancy import PersistentTenantRegistry
            from rag_arch.config import load_settings
            modes = execution_modes or {key: "model" for key in ("chat", "standards", "plan")}
            settings = load_settings() if "model" in modes.values() else None
            if settings is not None:
                materials.configure_catalog(settings)
                # This process schedules tasks; its read-only standard routes
                # use BM25. Full document vectors belong to the AI worker.
            app.state.tenants = PersistentTenantRegistry(
                database, root=root, pdf_backend=materials.pdf_backend,
                execution_modes=modes, model_name=getattr(settings, "chat_model", None),
                max_cached_users=tenant_max_users)
            app.state.tasks = app.state.documents = None
        else:
            if auth_enabled and database is None:
                raise RuntimeError("多用户服务需要先配置并显式迁移 RAG_ARCH_DATABASE_URL。")
            from rag_arch.config import load_settings
            from .chat_runner import run_chat
            from .standards_runner import run_standards
            from .plan_runner import run_plan
            settings = load_settings() if task_runner is None else None
            if settings is not None:
                materials.configure_catalog(settings)
                materials.configure_embeddings(settings)

            def execute(request, progress, cancel_event, store):
                if request["assistant_id"] == "chat":
                    return run_chat(request, progress, cancel_event, settings=settings,
                                    documents=store)
                if request["assistant_id"] == "standards":
                    return run_standards(request, progress, cancel_event, settings=settings, resources=materials)
                return run_plan(request, progress, cancel_event, settings=settings,
                                resources=materials, documents=store)

            modes = execution_modes or {key: "model" if task_runner is None else "simulation"
                                        for key in ("chat", "standards", "plan")}
            if auth_enabled:
                if database is None:
                    raise RuntimeError("多用户服务需要先配置并显式迁移 RAG_ARCH_DATABASE_URL。")
                if documents is not None:
                    raise ValueError("多用户服务不能注入共享DocumentStore。")
                def runner_factory(store):
                    return task_runner or (lambda request, progress, cancel_event:
                                           execute(request, progress, cancel_event, store))
                app.state.tenants = TenantRegistry(
                    database, root=root, pdf_backend=materials.pdf_backend,
                    runner_factory=runner_factory, execution_modes=modes,
                    model_name=getattr(settings, "chat_model", None),
                    max_workers=tenant_max_workers, max_users=tenant_max_users,
                    max_tasks_per_user=tenant_max_tasks)
                app.state.tasks = app.state.documents = None
            else:
                def legacy_execute(request, progress, cancel_event):
                    return execute(request, progress, cancel_event, app.state.documents)
                app.state.tasks = TaskService(runner=task_runner or legacy_execute,
                                             execution_modes=modes,
                                             model_name=getattr(settings, "chat_model", None))
        from .documents import DocumentStore
        if app.state.tenants is None:
            app.state.documents = documents if documents is not None else DocumentStore(
                root, app.state.tasks.workspace_id, materials.pdf_backend)
        try:
            yield
        finally:
            if app.state.tenants is not None:
                app.state.tenants.close()
            else:
                app.state.tasks.close()
                app.state.documents.close()
            materials.close()

    app = FastAPI(title="建筑资料助手 · Web工作台", version="0.2.0", lifespan=lifespan,
                  docs_url=None if auth_enabled or not serve_frontend else "/docs",
                  redoc_url=None if auth_enabled or not serve_frontend else "/redoc",
                  openapi_url=None if auth_enabled or not serve_frontend else "/openapi.json")
    app.state.auth_enabled = auth_enabled
    app.state.auth_store = authentication
    app.state.auth_cookie_secure = secure_cookie
    app.state.auth_rate_limiter = limiter
    app.state.serve_frontend = serve_frontend
    app.state.sse_connection_limiter = SSEConnectionLimiter(
        max_total=sse_max_connections, max_per_owner=sse_max_connections_per_user)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        request.state.principal = None
        if not serve_frontend and not request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if request.method == "POST" and request.url.path in {"/login", "/api/auth/login"}:
            # Bound chunked requests too, before FastAPI's JSON decoder reads
            # the body. Starlette's cached request replays these verified bytes.
            content = bytearray()
            async for chunk in request.stream():
                if len(content) + len(chunk) > 8192:
                    return _private_response(JSONResponse({"detail": "登录请求过大。"}, status_code=413))
                content.extend(chunk)
            request._body = bytes(content)
        if not auth_enabled:
            return await call_next(request)

        # Health/readiness probes describe the service itself, not the caller's
        # session. Bypass session lookup so a stale cookie cannot mask their
        # own liveness or generic dependency status.
        if (request.method == "GET"
                and request.url.path in {"/api/health", "/api/readiness"}):
            return _private_response(await call_next(request))

        if (request.method in UNSAFE_METHODS
                and not _origin_is_allowed(request, allowed_origins)):
            if request.url.path == "/login":
                return _private_response(auth_page("登录请求来源异常，请刷新页面后重试。", status_code=403))
            return _private_response(JSONResponse({"detail": "请求来源校验失败。"}, status_code=403))

        try:
            principal = await run_in_threadpool(
                authentication.resolve_identity, request.cookies.get(SESSION_COOKIE_NAME))
        except Exception:
            return _private_response(JSONResponse({"detail": "认证服务暂时不可用。"}, status_code=503))
        request.state.principal = principal
        username = principal.username if principal is not None else None
        request.state.auth_user = username
        is_public = ((request.url.path in {"/api/auth/login", "/api/auth/logout"} and request.method == "POST")
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
        if request.app.state.tenants is not None:
            try:
                request.state.tenant = await run_in_threadpool(
                    request.app.state.tenants.get, principal.user_id)
            except TenantCapacityError as exc:
                return _private_response(JSONResponse({"detail": str(exc)}, status_code=503))
        return _private_response(await call_next(request))

    dist = Path(frontend_dir) if frontend_dir is not None else ROOT / "frontend/dist"
    app.include_router(create_standards_router())
    app.include_router(create_document_router())
    from .history import create_history_router
    if serve_frontend:
        app.include_router(create_history_router(root))

    def current(request):
        return current_tasks(request)

    def check_workspace(tasks, workspace_id):
        if workspace_id != tasks.workspace_id:
            if getattr(tasks, "user_id", None) is not None:
                raise HTTPException(404, "未找到工作区。")
            raise HTTPException(409, "工作区已失效，请重新获取当前工作区。")

    def get_task(tasks, task_id):
        try:
            return tasks.get_task(task_id)
        except KeyError:
            raise HTTPException(404, "未找到本次后端运行中的任务。") from None

    @app.get("/api/health")
    def health(request: Request):
        if request.app.state.tenants is not None:
            return {"status": "ok", "mode": "multi_user", "task_storage":
                    "postgresql" if backend == "durable" else "process_memory"}
        return {"status": "ok", "mode": current(request).snapshot()["mode"],
                "server_instance_id": current(request).server_instance_id}

    from .readiness import ReadinessSnapshot

    @app.get("/api/readiness", response_model=ReadinessSnapshot,
             responses={503: {"model": ReadinessSnapshot, "description": "Service is not ready."}})
    async def readiness():
        from .readiness import check_readiness

        unavailable = {"database": False, "schema": False, "worker": False, "ready": False}
        try:
            state = await run_in_threadpool(
                check_readiness, database,
                auth_enabled=auth_enabled, task_backend=backend,
            )
        except Exception:
            state = unavailable
        if not state.get("ready"):
            return JSONResponse(content=state, status_code=503)
        return JSONResponse(content=state, status_code=200)

    async def verify_login(username, password, client_key):
        if authentication is None:
            raise HTTPException(503, "认证服务未启用。")
        try:
            account = ascii_username_key(username)
        except ValueError:
            account = "<invalid-username>"
        account_key = "account-client:" + json.dumps([client_key, account], ensure_ascii=True,
                                                      separators=(",", ":"))
        reservations = []
        try:
            for key in (client_key, account_key):
                attempt_id, retry_after = await run_in_threadpool(limiter.begin_attempt, key)
                if attempt_id is None:
                    raise HTTPException(429, "登录尝试过多，请稍后再试。",
                                        headers={"Retry-After": str(retry_after)})
                reservations.append((key, attempt_id))
        except Exception as exc:
            release_failed = False
            for key, ticket in reservations:
                try:
                    await run_in_threadpool(limiter.finish_attempt, key, ticket, None)
                except Exception:
                    release_failed = True
            if isinstance(exc, HTTPException) and not release_failed:
                raise
            raise HTTPException(503, "登录限流服务暂时不可用。") from None
        outcome = None
        try:
            try:
                verified = await run_in_threadpool(authentication.verify_identity, username, password)
                outcome = verified is not None
            finally:
                release_failed = False
                for key, ticket in reservations:
                    try:
                        await run_in_threadpool(limiter.finish_attempt, key, ticket, outcome)
                    except Exception:
                        release_failed = True
                if release_failed:
                    raise RuntimeError("login reservation settlement failed")
        except Exception:
            raise HTTPException(503, "认证服务暂时不可用。") from None
        if verified is None:
            raise HTTPException(401, "账号不存在或密码不正确。")
        try:
            token, expires_at = await run_in_threadpool(authentication.create_session, verified)
        except ValueError:
            raise HTTPException(401, "账号状态已变化，请重新登录。") from None
        except Exception:
            raise HTTPException(503, "认证服务暂时不可用。") from None
        return verified, token, expires_at

    def set_session_cookie(response, token, expires_at):
        response.set_cookie(SESSION_COOKIE_NAME, token, max_age=max(0, expires_at - int(time.time())),
                            httponly=True, secure=secure_cookie, samesite=same_site, path="/")

    @app.post("/api/auth/login", response_model=AuthSessionSnapshot)
    async def json_login(body: AuthLogin, request: Request):
        verified, token, expires_at = await verify_login(body.username, body.password, _login_client_key(request))
        response = JSONResponse({"authenticated": True, "owner_user_id": verified.user_id,
                                 "username": verified.username})
        set_session_cookie(response, token, expires_at)
        return _private_response(response)

    @app.get("/api/auth/session", response_model=AuthSessionSnapshot)
    def json_session(request: Request):
        principal = request.state.principal
        if principal is None:
            raise HTTPException(401, "请先登录。")
        return {"authenticated": True, "owner_user_id": principal.user_id, "username": principal.username}

    @app.post("/api/auth/logout", response_model=AuthLogout)
    async def json_logout(request: Request):
        if authentication is not None:
            try:
                await run_in_threadpool(authentication.revoke_session, request.cookies.get(SESSION_COOKIE_NAME))
            except Exception:
                raise HTTPException(503, "认证服务暂时不可用。") from None
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(SESSION_COOKIE_NAME, path="/", secure=secure_cookie,
                               httponly=True, samesite=same_site)
        response.headers["Clear-Site-Data"] = '"cache", "storage"'
        return _private_response(response)

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request):
        return auth_page(next_path=request.query_params.get("next", "/"))

    @app.post("/login", include_in_schema=False)
    async def login(request: Request):
        client_host = _login_client_key(request)
        limited, retry_after = await run_in_threadpool(limiter.is_limited, client_host)
        if limited:
            page = auth_page("登录尝试过多，请稍后再试。", status_code=429)
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
        attempt_id, retry_after = await run_in_threadpool(limiter.begin_attempt, client_host)
        if attempt_id is None:
            page = auth_page("登录尝试过多，请稍后再试。", status_code=429)
            page.headers["Retry-After"] = str(retry_after)
            return page
        outcome = None
        try:
            verified = await run_in_threadpool(authentication.verify_identity, username, password)
            outcome = verified is not None
        finally:
            await run_in_threadpool(limiter.finish_attempt, client_host, attempt_id, outcome)
        if verified is None:
            return auth_page("账号不存在或密码不正确。", next_path, status_code=401)
        try:
            token, expires_at = await run_in_threadpool(authentication.create_session, verified)
        except ValueError:
            # A password change or account disable may win after verification.
            return auth_page("账号状态已变化，请重新登录。", next_path, status_code=401)
        response = RedirectResponse(next_path, status_code=303)
        response.set_cookie(SESSION_COOKIE_NAME, token, max_age=max(0, expires_at - int(time.time())),
                            httponly=True, secure=secure_cookie, samesite=same_site, path="/")
        return response

    @app.post("/logout", include_in_schema=False)
    def logout(request: Request):
        authentication.revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/", secure=secure_cookie,
                              httponly=True, samesite=same_site)
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
            if tasks.execution_mode(body.assistant_id) == "model" or getattr(tasks, "user_id", None) is not None:
                validate_task_documents(body, current_documents(request))
            return tasks.submit(body.assistant_id, body.request_id, body.prompt,
                                duration_seconds=body.duration_seconds, fail_at_end=body.fail_at_end,
                                attachment_ids=body.attachment_ids, registered_plan_id=body.registered_plan_id,
                                plan_action=body.plan_action, use_embeddings=body.use_embeddings,
                                max_steps=body.max_steps, max_tool_calls=body.max_tool_calls)
        except BusyError:
            raise HTTPException(409, "这个助手已有任务在运行，请等待完成或停止后再提交。") from None
        except TaskCapacityError as exc:
            raise HTTPException(429, str(exc)) from None
        except _idempotency_conflict_type() as exc:
            raise HTTPException(409, str(exc)) from None
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
        principal = getattr(request.state, "principal", None)
        owner_key = getattr(principal, "user_id", None)
        if owner_key is None:
            owner_key = getattr(tasks, "user_id", None)
        if owner_key is None:
            owner_key = "anonymous"
        lease = app.state.sse_connection_limiter.reserve(owner_key)
        if lease is None:
            raise HTTPException(
                429, "事件连接数已达上限，请关闭重复页面后重试。",
                headers={"Retry-After": "2"})
        try:
            yield tasks
        finally:
            app.state.sse_connection_limiter.release(lease)

    @app.get("/api/events", response_class=EventSourceResponse)
    async def events(request: Request, tasks: TaskService = Depends(sse_service),
                     server_instance_id: str | None = None, after: str = Query(default="0"),
                     last_event_id: str | None = Header(default=None)):
        if server_instance_id is not None and server_instance_id != tasks.server_instance_id:
            yield ServerSentEvent(event="snapshot_required", data={
                "reason": "server_restarted", "server_instance_id": tasks.server_instance_id})
            return

        session_check = None
        if auth_enabled:
            token = request.cookies.get(SESSION_COOKIE_NAME)
            initial_principal = getattr(request.state, "principal", None)
            expected_user_id = getattr(initial_principal, "user_id", None)
            service_user_id = getattr(tasks, "user_id", None)
            session_check = _event_session_check(
                authentication, token, expected_user_id, service_user_id)

        async for item in event_stream(
                tasks, after, last_event_id=last_event_id,
                session_check=session_check):
            yield item

    @app.get("/api/tasks/{task_id}/download")
    def download(task_id: str, workspace_id: str, request: Request, format: str = Query(pattern="^(json|md|debug)$")):
        tasks = current(request)
        check_workspace(tasks, workspace_id)
        result = get_task(tasks, task_id)
        answer = result.get("result") or {}
        if format == "debug":
            audit = answer.get("audit_log") or {}
            if not isinstance(audit, dict) or not audit:
                raise HTTPException(404, "未找到可下载的调试记录。")
            if getattr(tasks, "user_id", None) is not None:
                from ..runtime_audit import debug_result_json
                try:
                    content = debug_result_json(
                        {"runtime": {"audit_log": audit}}, user_id=tasks.user_id,
                        workspace_id=tasks.workspace_id,
                        root=Path(root) / "data/runtime/conversations")
                except (ValueError, FileNotFoundError):
                    raise HTTPException(404, "未找到可下载的调试记录。") from None
                return Response(content, media_type="application/json",
                                headers={"Content-Disposition": f'attachment; filename="debug_{task_id}.json"'})
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
                "insufficient_evidence": "资料不足以支持结论；以下内容未得到充分证据支持。",
            }
            validation = validation_labels.get(answer.get("termination_reason"))
            if validation is None and answer.get("status") not in {None, "answered", "grounded"}:
                validation = "回答尚未通过核验；以下原文不能作为已验证结论。"
            warnings = [warning for warning in answer.get("warnings", []) if isinstance(warning, str)]
            notices = ([validation] if validation else []) + warnings
            notice_text = "".join(f"> {notice}\n\n" for notice in dict.fromkeys(notices))
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
    if not serve_frontend:
        # Kept only as an explicitly selected compatibility UI. The production
        # ASGI app below serves JSON APIs; static hosting owns all HTML.
        html_paths = {"/", "/viewer/pages", "/login", "/logout", "/assets"}
        app.router.routes = [route for route in app.router.routes
                             if getattr(route, "path", None) not in html_paths]
    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=sorted(_origin_string(item) for item in cors_origins),
                           allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"],
                           allow_headers=["Content-Type", "Last-Event-ID"],
                           expose_headers=["Retry-After", "Content-Disposition"])
    return app


app = create_app(serve_frontend=_env_bool("RAG_ARCH_SERVE_FRONTEND", False))
