"""Per-question local records. Never send audit history back to a model."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .config import ROOT
from .multimodal import decode_image_data_url, image_dimensions

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


class RunAudit:
    def __init__(self, scope, settings, question, *, root=None, **metadata):
        self.directory = Path(root or AUDIT_ROOT) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12])
        self.events = self.directory / "events.jsonl"
        self.result_path = self.directory / "result.json"
        self.secrets = tuple(str(value) for value in (
            getattr(settings, "chat_api_key", ""), getattr(settings, "embedding_api_key", ""),
            getattr(settings, "reranker_api_key", "")) if value)
        self.lock, self.sequence, self.closed, self.result = Lock(), 0, False, None
        self._image_lock, self._image_refs = Lock(), {}
        self.info = {"run_id": self.directory.name, "directory": str(self.directory), "error": None}
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            self._failed(exc)
        self.write("run_started", scope=scope, question=question, model=settings.chat_model,
                   endpoint=_safe_url(settings.chat_base_url), **metadata)

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
def record_run(scope, settings, question, *, root=None, **metadata):
    audit = RunAudit(scope, settings, question, root=root, **metadata)
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


def record_event(kind, **fields):
    audit = _current.get()
    if audit is not None:
        audit.write(kind, **fields)


def _request_hook(request):
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


def attach_chat_audit(model):
    """Observe the existing sync HTTP client; preserve its transport and auth."""
    audit = _current.get()
    if audit is None:
        return model
    try:
        client = model.root_client._client
    except AttributeError as exc:
        audit._failed(exc)
        return model
    for event, hook in (("request", _request_hook), ("response", _response_hook)):
        hook._rag_audit_hook = True
        client.event_hooks[event] = [h for h in client.event_hooks[event]
                                    if not getattr(h, "_rag_audit_hook", False)] + [hook]
    return model


def debug_result_json(result):
    """Reuse the existing result download, adding this run's local call log."""
    row = result.to_dict() if hasattr(result, "to_dict") else deepcopy(result)
    info = row.get("runtime", {}).get("audit_log")
    if info:
        directory = Path(info["directory"]).resolve()
        if not directory.is_relative_to(AUDIT_ROOT.resolve()):
            raise ValueError("调用记录路径不在本项目记录目录内。")
        saved = directory / "result.json"
        if saved.is_file():
            row = json.loads(saved.read_text(encoding="utf-8"))
        events = directory / "events.jsonl"
        row["call_audit"] = {
            "note": "供调试用；请求与响应文字完整记录，图片按 image_ref 保存于本次记录目录，重复图片只存一份；鉴权头不记录，已配置密钥已脱敏。",
            "recording_error": info.get("error"),
            "events": [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
                      if events.is_file() else [],
        }
    return json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)
