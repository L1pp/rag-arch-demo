"""Per-question local records. Never send audit history back to a model."""
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .config import ROOT
from .multimodal import decode_image_data_url, image_dimensions
from .request_control import check_request_control

AUDIT_ROOT = ROOT / "data/runtime/conversations"
_current = ContextVar("rag_runtime_audit", default=None)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_url(value):
    parts = urlsplit(str(value))
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


_SAFE_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_RUN_ID = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{12}$")
_SAFE_ATTEMPT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _valid_attempt(value):
    return ((isinstance(value, int) and not isinstance(value, bool) and value > 0)
            or (isinstance(value, str) and bool(_SAFE_ATTEMPT.fullmatch(value))))


def tenant_audit_metadata(request):
    """Return trusted server-side identity fields for one Web audit run.

    Existing local CLI callers may omit tenant fields and continue to use the
    legacy audit directory. Once either tenant field is present, all identity
    fields are required so a partially scoped Web run cannot fall back to the
    shared legacy directory.
    """
    if not isinstance(request, dict):
        raise ValueError("任务审计请求须为对象。")
    task_id = request.get("task_id")
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("任务审计缺少有效 task_id。")
    user_id = request.get("user_id")
    workspace_id = request.get("workspace_id")
    if user_id is None and workspace_id is None:
        return {"task_id": task_id} if task_id is not None else {}
    if (isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1
            or not isinstance(workspace_id, str)
            or not _SAFE_WORKSPACE_ID.fullmatch(workspace_id)
            or task_id is None):
        raise ValueError("租户审计身份不完整或无效。")
    metadata = {"user_id": user_id, "workspace_id": workspace_id, "task_id": task_id}
    attempt = request.get("attempt")
    if attempt is not None:
        if not _valid_attempt(attempt):
            raise ValueError("任务审计 attempt 无效。")
        metadata["attempt"] = attempt
    return metadata


class RunAudit:
    def __init__(self, scope, settings, question, *, root=None, user_id=None,
                 workspace_id=None, task_id=None, attempt=None, **metadata):
        tenant_owned = user_id is not None or workspace_id is not None
        if tenant_owned:
            if (isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1
                    or not isinstance(workspace_id, str)
                    or not _SAFE_WORKSPACE_ID.fullmatch(workspace_id)
                    or not isinstance(task_id, str) or not task_id.strip()
                    or (attempt is not None and not _valid_attempt(attempt))):
                raise ValueError("租户审计身份不完整或无效。")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]
        audit_root = Path(root or AUDIT_ROOT)
        if tenant_owned:
            self.directory = audit_root / "users" / str(user_id) / workspace_id / run_id
        else:
            self.directory = audit_root / run_id
        self.events = self.directory / "events.jsonl"
        self.result_path = self.directory / "result.json"
        self.secrets = tuple(str(value) for value in (
            getattr(settings, "chat_api_key", ""), getattr(settings, "embedding_api_key", ""),
            getattr(settings, "reranker_api_key", "")) if value)
        self.lock, self.sequence, self.closed, self.result = Lock(), 0, False, None
        self._image_lock, self._image_refs = Lock(), {}
        self.info = {"run_id": run_id, "error": None}
        if tenant_owned:
            # Do not expose a server filesystem path through task results.
            self.info.update(user_id=user_id, workspace_id=workspace_id, task_id=task_id)
            if attempt is not None:
                self.info["attempt"] = attempt
        else:
            # Preserve the established path for local CLI and legacy tests.
            self.info["directory"] = str(self.directory)
        started_metadata = dict(metadata)
        if tenant_owned:
            started_metadata.update(user_id=user_id, workspace_id=workspace_id,
                                    task_id=task_id, run_id=run_id)
            if attempt is not None:
                started_metadata["attempt"] = attempt
        elif task_id is not None:
            started_metadata["task_id"] = task_id
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            self._failed(exc)
        self.write("run_started", scope=scope, question=question, model=settings.chat_model,
                   endpoint=_safe_url(settings.chat_base_url), **started_metadata)

    def _failed(self, exc):
        self.info["error"] = type(exc).__name__
        logging.getLogger(__name__).error("本地调用记录未完整写入（%s）：%s", type(exc).__name__, self.directory)

    def _image_reference(self, data_url):
        """Store an inline image once in this run; never alter the HTTP request."""
        with self._image_lock:
            if data_url in self._image_refs:
                return dict(self._image_refs[data_url])
            try:
                decoded = decode_image_data_url(data_url)
            except ValueError:
                return {"image_ref": None, "error": "invalid_inline_image"}
            if decoded is None:
                return None
            data, mime = decoded
            width, height = image_dimensions(data)
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
                      "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".img")
            relative = Path("images") / f"image_{len(self._image_refs) + 1:04d}{suffix}"
            target = self.directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            reference = {"image_ref": relative.as_posix(), "mime_type": mime,
                         "width": width, "height": height, "byte_count": len(data)}
            self._image_refs[data_url] = reference
            return dict(reference)

    def clean(self, value):
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            # Flatten a data-URL's local reference inside the existing image_url
            # object. Other request/response fields keep their original shape.
            inline_url = value.get("url")
            if isinstance(inline_url, str) and inline_url.startswith("data:image/"):
                reference = self._image_reference(inline_url)
                if reference is not None:
                    return {**self.clean({k: v for k, v in value.items() if k != "url"}), **reference}
            if value.get("type") == "image" and isinstance(value.get("base64"), str):
                reference = self._image_reference(f"data:{value.get('mime_type', 'image/png')};base64,{value['base64']}")
                if reference is not None:
                    return {**self.clean({k: v for k, v in value.items() if k != "base64"}), **reference}
            return {str(k): ("[REDACTED]" if str(k).lower() in {
                "api_key", "authorization", "proxy-authorization", "cookie", "set-cookie"}
                else self.clean(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            if value.startswith("data:image/"):
                reference = self._image_reference(value)
                if reference is not None:
                    return reference
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return f"<{type(value).__name__}>"

    def write(self, kind, **fields):
        if self.closed or self.info["error"]:
            return
        with self.lock:
            try:
                self.sequence += 1
                row = self.clean({"sequence": self.sequence, "at": _now(), "kind": kind, **fields})
                with self.events.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            except (OSError, ValueError, TypeError) as exc:
                self._failed(exc)

    def set_result(self, result):
        self.result = result

    def finish(self, error=None):
        self.write("run_finished", status="failed" if error else "returned", error_type=error)
        if self.result is not None and not self.info["error"]:
            try:
                self.result_path.write_text(json.dumps(self.clean(self.result), ensure_ascii=False,
                                                       indent=2, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError, TypeError) as exc:
                self._failed(exc)
        self.closed = True


@contextmanager
def record_run(scope, settings, question, *, root=None, user_id=None,
                workspace_id=None, task_id=None, attempt=None, **metadata):
    audit = RunAudit(scope, settings, question, root=root, user_id=user_id,
                     workspace_id=workspace_id, task_id=task_id, attempt=attempt, **metadata)
    token = _current.set(audit)
    error = None
    try:
        yield audit
    except BaseException as exc:
        error = type(exc).__name__
        from .generation_experiments import safe_error
        audit.write("run_exception", error_type=error, error=safe_error(exc, component="runtime"))
        raise
    finally:
        audit.finish(error)
        _current.reset(token)


def _record_async_run_exception(audit, exc):
    from .generation_experiments import safe_error
    audit.write("run_exception", error_type=type(exc).__name__,
                error=safe_error(exc, component="runtime"))


@asynccontextmanager
async def arecord_run(scope, settings, question, *, root=None, user_id=None,
                      workspace_id=None, task_id=None, attempt=None, **metadata):
    """Async-run audit scope; construction and persistence stay off the loop."""
    from .async_support import run_local
    audit = await run_local(RunAudit, scope, settings, question, root=root,
        user_id=user_id, workspace_id=workspace_id, task_id=task_id,
        attempt=attempt, **metadata)
    # ContextVar lifetime must match this async task, not the executor thread
    # used for local audit I/O.
    token = _current.set(audit)
    error = None
    try:
        yield audit
    except BaseException as exc:
        error = type(exc).__name__
        await run_local(_record_async_run_exception, audit, exc)
        raise
    finally:
        try:
            await run_local(audit.finish, error)
        finally:
            _current.reset(token)


def record_event(kind, **fields):
    audit = _current.get()
    if audit is not None:
        audit.write(kind, **fields)


def _request_hook(request):
    # This is the last cancellable boundary before an SDK or httpx transport
    # sends bytes. It applies even when a run is not being audited.
    check_request_control()
    audit = _current.get()
    if audit is None or audit.closed:
        return
    call_id = uuid4().hex
    # LangChain caches HTTP clients. Bind each request to its own run, never
    # capture one user's recorder in hooks attached to a shared client.
    request.extensions["rag_audit_owner"] = (audit, call_id)
    try:
        payload = json.loads(request.content)
    except (ValueError, UnicodeDecodeError):
        payload = {"unparsed_text": request.content.decode("utf-8", errors="replace")}
    audit.write("model_http_request", call_id=call_id, method=request.method,
                url=_safe_url(request.url), body=payload)


async def _async_request_hook(request):
    """Run request audit serialization and file I/O outside the async loop."""
    from .async_support import run_local
    await run_local(_request_hook, request)


_async_request_hook._rag_async_audit_hook = True
_async_request_hook._rag_audit_hook = True


def _response_hook(response):
    owner = response.request.extensions.get("rag_audit_owner")
    if owner is None:
        return
    audit, call_id = owner
    # Record status before reading the body. A later read timeout must not
    # erase headers that were already available. Streaming skips this hook.
    headers = {"http_status": response.status_code,
               "request_id": response.headers.get("x-request-id"),
               "content_type": response.headers.get("content-type")}
    audit.write("model_http_headers", call_id=call_id, **headers)
    try:
        response.read()
    except Exception as exc:
        audit.write("model_http_response_error", call_id=call_id, error_type=type(exc).__name__,
                    **headers)
        raise
    try:
        payload = response.json()
    except ValueError:
        payload = {"unparsed_text": response.text}
    audit.write("model_http_response", call_id=call_id, body=payload, **headers)


async def _async_response_hook(response):
    """Read async SDK responses without blocking or losing request ownership."""
    owner = response.request.extensions.get("rag_audit_owner")
    if owner is None:
        return
    audit, call_id = owner
    headers = {"http_status": response.status_code,
               "request_id": response.headers.get("x-request-id"),
               "content_type": response.headers.get("content-type")}
    from .async_support import run_local
    await run_local(audit.write, "model_http_headers", call_id=call_id, **headers)
    try:
        await response.aread()
    except Exception as exc:
        await run_local(audit.write, "model_http_response_error", call_id=call_id,
                         error_type=type(exc).__name__, **headers)
        raise
    await run_local(record_async_response, response, include_headers=False)


_async_response_hook._rag_async_audit_hook = True
_async_response_hook._rag_audit_hook = True


def record_async_response(response, *, include_headers=True):
    """Record a fully received async response using its request-bound owner."""
    owner = response.request.extensions.get("rag_audit_owner")
    if owner is None:
        return
    audit, call_id = owner
    headers = {"http_status": response.status_code,
               "request_id": response.headers.get("x-request-id"),
               "content_type": response.headers.get("content-type")}
    if include_headers:
        audit.write("model_http_headers", call_id=call_id, **headers)
    try:
        payload = response.json()
    except ValueError:
        payload = {"unparsed_text": response.text}
    audit.write("model_http_response", call_id=call_id, body=payload, **headers)


async def arecord_async_response(response):
    """Serialize and persist a fully read async response off the event loop."""
    from .async_support import run_local
    await run_local(record_async_response, response)


async def arecord_event(kind, **fields):
    """Persist one async-run event without blocking its event loop."""
    from .async_support import run_local
    await run_local(record_event, kind, **fields)


def ensure_async_request_audit(client):
    """Install one context-aware request hook on a shared AsyncClient.

    The hook binds the active audit to this request's extensions; it never
    closes over a tenant's recorder, so concurrent tasks cannot cross owners.
    Responses streaming uses this request-only hook and records its body while
    consuming SSE itself.
    """
    hook = _async_request_hook
    hooks = client.event_hooks["request"]
    hooks[:] = [item for item in hooks
                if not getattr(item, "_rag_audit_hook", False)
                or getattr(item, "_rag_async_audit_hook", False)]
    if not any(getattr(item, "_rag_async_audit_hook", False) for item in hooks):
        hooks.append(hook)
    return client


def attach_chat_audit(model):
    """Observe sync and async SDK clients; preserve their transport and auth."""
    audit = _current.get()
    for root_name, request_hook, response_hook in (("root_client", _request_hook, _response_hook),
                                     ("root_async_client", _async_request_hook, _async_response_hook)):
        try:
            client = getattr(model, root_name)._client
        except AttributeError as exc:
            if audit is not None and root_name == "root_client":
                audit._failed(exc)
            continue
        request_hooks = client.event_hooks["request"]
        request_hook._rag_audit_hook = True
        if root_name == "root_async_client":
            request_hook._rag_async_audit_hook = True
        request_hooks[:] = [hook for hook in request_hooks
                            if not getattr(hook, "_rag_audit_hook", False)
                            or bool(getattr(hook, "_rag_async_audit_hook", False)) ==
                            bool(getattr(request_hook, "_rag_async_audit_hook", False))]
        if not any(getattr(hook, "_rag_audit_hook", False) for hook in request_hooks):
            request_hooks.append(request_hook)
        response_hook._rag_audit_hook = True
        if root_name == "root_async_client":
            response_hook._rag_async_audit_hook = True
        response_hooks = client.event_hooks["response"]
        response_hooks[:] = [hook for hook in response_hooks
                             if not getattr(hook, "_rag_audit_hook", False)
                             or bool(getattr(hook, "_rag_async_audit_hook", False)) ==
                             bool(getattr(response_hook, "_rag_async_audit_hook", False))]
        if not any(getattr(hook, "_rag_audit_hook", False) for hook in response_hooks):
            response_hooks.append(response_hook)
    return model


def debug_result_json(result, *, user_id=None, workspace_id=None, root=None):
    """Add a run's local call log, enforcing tenant ownership when present.

    Tenant-owned records are resolved from their opaque run ID under the
    current audit root; their absolute filesystem paths never come from the
    result payload. Ownerless records retain the legacy local CLI behavior.
    """
    audit_root = Path(root or AUDIT_ROOT).resolve()
    row = result.to_dict() if hasattr(result, "to_dict") else deepcopy(result)
    runtime = row.get("runtime")
    info = runtime.get("audit_log") if isinstance(runtime, dict) else None
    if not info:
        info = row.get("audit_log")
    if info:
        if not isinstance(info, dict):
            raise ValueError("调用记录元数据无效。")
        tenant_owned = "user_id" in info or "workspace_id" in info
        if tenant_owned:
            stored_user = info.get("user_id")
            stored_workspace = info.get("workspace_id")
            run_id = info.get("run_id")
            if (user_id is None or workspace_id is None
                    or user_id != stored_user or workspace_id != stored_workspace):
                raise ValueError("当前用户无权下载这份调用记录。")
            if (isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1
                    or not isinstance(workspace_id, str)
                    or not _SAFE_WORKSPACE_ID.fullmatch(workspace_id)
                    or not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id)):
                raise ValueError("调用记录身份无效。")
            users_root = (audit_root / "users").resolve()
            directory = (users_root / str(user_id) / workspace_id / run_id).resolve()
            if not directory.is_relative_to(users_root):
                raise ValueError("调用记录路径不在当前用户目录内。")
            if not directory.is_dir():
                raise FileNotFoundError("本次调用记录目录不存在。")
        else:
            if user_id is not None or workspace_id is not None:
                # A legacy shared audit has no trustworthy owner. It may still
                # be opened by local CLI callers that omit identity arguments.
                raise ValueError("旧调用记录没有用户归属，不能通过多用户接口下载。")
            directory = Path(info["directory"]).resolve()
            if not directory.is_relative_to(audit_root):
                raise ValueError("调用记录路径不在本项目记录目录内。")
        saved = directory / "result.json"
        if saved.is_file():
            row = json.loads(saved.read_text(encoding="utf-8"))
        elif tenant_owned:
            raise FileNotFoundError("本次调用结果记录不存在。")
        events = directory / "events.jsonl"
        row["call_audit"] = {
            "note": "供调试用；请求与响应文字完整记录，图片按 image_ref 保存于本次记录目录，重复图片只存一份；鉴权头不记录，已配置密钥已脱敏。",
            "recording_error": info.get("error"),
            "events": [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
                      if events.is_file() else [],
        }
    return json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)
