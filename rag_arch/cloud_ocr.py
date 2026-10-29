"""Bounded, per-page VL-OCR requests for explicitly enabled worker callers.

The caller owns tenant/asset authorization and supplies an asset-specific cache
directory. This module never enables OCR on an API or preview request by itself.
Unknown or interrupted requests remain in the cache and are never resent
automatically because the remote service may already have billed them.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from decimal import Decimal
import html
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
from time import monotonic
from typing import Callable
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from uuid import uuid4

from dotenv import dotenv_values
import httpx
from PIL import Image, UnidentifiedImageError

from .request_control import check_request_control


MODEL = "qwen-vl-ocr-2025-11-20"
MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_EDGE = 2400
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CACHE_BYTES = 5 * 1024 * 1024
MAX_OCR_TEXT_CHARS = 64_000
DEFAULT_TIMEOUT_SECONDS = 120.0
_SCHEMA = "rag-arch-vl-ocr-page-v1"
_HOST = re.compile(r"(?:ws-[a-z0-9][a-z0-9-]*\.cn-beijing\.maas\.aliyuncs\.com|dashscope\.aliyuncs\.com)\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_URL = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
_DATA_URL = re.compile(r"data:[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_\r\n-]+", re.IGNORECASE)
_SENSITIVE_QUERY = frozenset({"security-token", "security_token", "signature", "ossaccesskeyid",
                              "x-oss-signature", "x-oss-security-token", "x-oss-credential",
                              "x-amz-signature", "x-amz-credential", "x-amz-security-token",
                              "token", "access_token", "api_key", "apikey", "api-key", "key",
                              "access_key", "accesskey", "secret", "client_secret", "password",
                              "credential", "authorization", "sig"})
_UNVERIFIED = "本页为云端 OCR 识别文本，数字、单位、表格和公式可能有误或遗漏，请对照原页核实。"


def _value(values, name: str) -> str:
    value = values.get(name)
    return value.strip() if isinstance(value, str) else ""


def _endpoint(base_url: str) -> str:
    """Change only the API path on an explicitly configured official host."""
    try:
        parsed = urlsplit(base_url.rstrip("/"))
        port = parsed.port
    except ValueError:
        raise ValueError("OCR 普通按量 API 地址无效。") from None
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or not _HOST.fullmatch(host)
            or port not in (None, 443) or parsed.username or parsed.password
            or parsed.query or parsed.fragment
            or parsed.path not in ("", "/api/v1", "/compatible-mode/v1")):
        raise ValueError("OCR 需要已配置的阿里云北京普通按量 HTTPS API 地址。")
    return urlunsplit(("https", host, "/api/v1/services/aigc/multimodal-generation/generation", "", ""))


@dataclass(frozen=True)
class OCRSettings:
    enabled: bool = False
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_root(cls, root: str | Path, *, environ=None) -> "OCRSettings":
        """Load only the ordinary API credential; never consult Token Plan."""
        path = Path(root) / ".env"
        values = dict(dotenv_values(path, interpolate=False)) if path.is_file() else {}
        source = os.environ if environ is None else environ
        for name in ("RAG_ARCH_OCR_ENABLED", "RAG_ARCH_OCR_TIMEOUT_SECONDS",
                     "RAG_OCR_BASE_URL", "RAG_OCR_API_KEY", "RAG_CHAT_BASE_URL",
                     "RAG_CHAT_API_KEY"):
            if name in source:
                values[name] = source[name]
        enabled_text = _value(values, "RAG_ARCH_OCR_ENABLED").lower() or "false"
        if enabled_text not in {"true", "false"}:
            raise ValueError("RAG_ARCH_OCR_ENABLED 只支持 true 或 false。")
        # A disabled optional OCR service must not reject unrelated tasks
        # because an unused OCR-specific timeout is malformed.
        timeout = DEFAULT_TIMEOUT_SECONDS
        if enabled_text == "true":
            timeout_text = _value(values, "RAG_ARCH_OCR_TIMEOUT_SECONDS")
            try:
                timeout = float(timeout_text) if timeout_text else DEFAULT_TIMEOUT_SECONDS
            except (TypeError, ValueError):
                raise ValueError("OCR 超时时间必须是有限正数。") from None
            if not math.isfinite(timeout) or not 1 <= timeout <= 300:
                raise ValueError("OCR 超时时间须为 1 至 300 秒。")
        base_url = _value(values, "RAG_OCR_BASE_URL") or _value(values, "RAG_CHAT_BASE_URL")
        api_key = _value(values, "RAG_OCR_API_KEY") or _value(values, "RAG_CHAT_API_KEY")
        if enabled_text == "true":
            _endpoint(base_url)
            if not api_key:
                raise ValueError("OCR 普通按量 API 密钥尚未配置。")
        return cls(enabled_text == "true", base_url, api_key, timeout)


@dataclass(frozen=True)
class OCRPageResult:
    status: str
    page: int
    text: str = ""
    model: str = MODEL
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_cny: str | None = None
    warnings: tuple[str, ...] = ()
    from_cache: bool = False
    error_code: str | None = None
    recognition_verified: bool = field(default=False, init=False)

    def to_dict(self) -> dict:
        return {"status": self.status, "page": self.page, "text": self.text,
                "model": self.model, "request_id": self.request_id,
                "usage": dict(self.usage), "cost_cny": self.cost_cny,
                "warnings": list(self.warnings), "from_cache": self.from_cache,
                "error_code": self.error_code, "recognition_verified": False}


def _not_link(path: Path) -> bool:
    return not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction())


def _private_cache_directory(root: Path, supplied: str | Path) -> Path:
    """Keep cache data in an owned, non-link directory inside upload/page roots."""
    root = Path(os.path.abspath(root))
    path = Path(supplied)
    path = Path(os.path.abspath(path if path.is_absolute() else root / path))
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise ValueError("OCR 缓存目录不在项目运行目录内。") from None
    allowed = (("data", "runtime", "web_uploads"), ("data", "runtime", "document_pages"))
    if not any(parts[:3] == prefix and len(parts) > 3 for prefix in allowed):
        raise ValueError("OCR 缓存目录不在允许的文档运行目录内。")
    for ancestor in (*reversed(root.parents), root):
        if not ancestor.exists() or not _not_link(ancestor) or not ancestor.is_dir():
            raise ValueError("OCR 项目路径包含不安全的链接或非目录。")
    current = root
    for part in parts:
        current = current / part
        try:
            current.mkdir(mode=0o700, exist_ok=True)
            info = current.lstat()
        except OSError:
            raise ValueError("OCR 缓存目录无法访问。") from None
        if not _not_link(current) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("OCR 缓存目录包含不安全的链接或非目录。")
    if os.name == "posix":
        if current.stat().st_uid != os.getuid():
            raise ValueError("OCR 缓存目录不属于当前服务账户。")
        current.chmod(0o700)
    return current


def _read_record(path: Path) -> dict | None:
    if not _not_link(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CACHE_BYTES
                    or (os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077))):
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                fd = -1
                result = json.load(stream)
        finally:
            if fd >= 0:
                os.close(fd)
        return result if isinstance(result, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _create_record(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def _replace_record(path: Path, value: dict) -> None:
    if not _not_link(path):
        raise ValueError("OCR 缓存状态不能是链接。")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_unstarted(path: Path, attempt_id: str) -> None:
    """Only remove our marker if control failed before opening an HTTP request."""
    record = _read_record(path)
    if record and record.get("status") == "started" and record.get("attempt_id") == attempt_id:
        if _not_link(path):
            path.unlink(missing_ok=True)


def _validate_image(png: bytes) -> None:
    if not isinstance(png, bytes) or not png or len(png) > MAX_IMAGE_BYTES:
        raise ValueError("OCR 页图为空或过大。")
    try:
        with Image.open(BytesIO(png)) as image:
            width, height = image.size
            if (image.format != "PNG" or width < 1 or height < 1 or width > MAX_IMAGE_EDGE
                    or height > MAX_IMAGE_EDGE or width * height > MAX_IMAGE_PIXELS):
                raise ValueError("OCR 页图格式或尺寸不符合限制。")
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError("OCR 页图无法读取。") from None


def _safe_text(text: str, key: str) -> str:
    """Discard provider-signed image URLs, while keeping ordinary source URLs."""
    text = text.replace(key, "[已移除敏感内容]") if key else text
    text = _DATA_URL.sub("[图片数据已移除，请查看原页]", text)

    def redact(match):
        candidate = match.group(0)
        try:
            query = urlsplit(html.unescape(candidate)).query
            keys = {name.lower() for name, _ in parse_qsl(query.replace(";", "&"), keep_blank_values=True)}
        except ValueError:
            return "[链接待核]"
        if keys & _SENSITIVE_QUERY:
            return "[临时图片链接已移除，请查看原页]"
        return candidate

    return _URL.sub(redact, text)


def _result_from_record(record: dict | None, page: int) -> OCRPageResult:
    if (not record or record.get("schema") != _SCHEMA or record.get("page") != page
            or record.get("model") != MODEL):
        return OCRPageResult("unknown", page, warnings=("OCR 状态无法核实，未自动重复请求；请查看原页。",),
                             from_cache=True, error_code="cache_unreadable")
    status = record.get("status")
    if status == "started" or status == "unknown":
        return OCRPageResult("unknown", page, warnings=("OCR 请求结果未确认，未自动重复请求；请查看原页。",),
                             from_cache=True, error_code="request_unconfirmed")
    if status not in {"completed", "failed"}:
        return OCRPageResult("unknown", page, warnings=("OCR 状态无法核实，未自动重复请求；请查看原页。",),
                             from_cache=True, error_code="cache_unrecognized")
    text = record.get("text")
    usage = record.get("usage")
    warnings = record.get("warnings")
    request_id = record.get("request_id")
    cost = record.get("cost_cny")
    error_code = record.get("error_code")
    if (not isinstance(text, str) or (status == "completed" and not text.strip())
            or not isinstance(usage, dict) or any(k not in {"input_tokens", "output_tokens"}
                                                or type(v) is not int or v < 0 for k, v in usage.items())
            or not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings)
            or (request_id is not None and (not isinstance(request_id, str) or not _ID.fullmatch(request_id)))
            or (cost is not None and not isinstance(cost, str))
            or (error_code is not None and (not isinstance(error_code, str) or not _ID.fullmatch(error_code)))
            or record.get("recognition_verified") is not False):
        return OCRPageResult("unknown", page, warnings=("OCR 缓存内容不完整，未自动重复请求；请查看原页。",),
                             from_cache=True, error_code="cache_invalid")
    if status == "completed" and len(text) > MAX_OCR_TEXT_CHARS:
        return OCRPageResult("failed", page, request_id=request_id, usage=usage, cost_cny=cost,
                             warnings=("OCR 本页文字过长，未纳入识别结果；请查看原页。",),
                             from_cache=True, error_code="ocr_text_too_large")
    return OCRPageResult(status, page, text, MODEL, request_id, usage, cost,
                         tuple(warnings), True, error_code)


def _cached_result(path: Path, page: int, key: str) -> OCRPageResult:
    record = _read_record(path)
    result = _result_from_record(record, page)
    if result.error_code == "ocr_text_too_large" and record and record.get("status") == "completed":
        # Earlier caches may predate the text limit. Preserve the measured usage
        # and terminal outcome but discard the oversized body without a new call.
        _replace_record(path, {"schema": _SCHEMA, **result.to_dict()})
    elif result.status == "completed":
        # Older successful results were produced before some query-parameter
        # forms were classified as sensitive. Clean them when next reused.
        cleaned = _safe_text(result.text, key)
        if cleaned != result.text:
            result = replace(result, text=cleaned)
            _replace_record(path, {"schema": _SCHEMA, **result.to_dict()})
    return result


def _safe_identifier(value, key: str) -> str | None:
    return value if isinstance(value, str) and _ID.fullmatch(value) and (not key or key not in value) else None


def _usage_and_cost(value) -> tuple[dict[str, int], str | None]:
    if not isinstance(value, dict):
        return {}, None
    incoming, outgoing = value.get("input_tokens"), value.get("output_tokens")
    if any(type(n) is not int or n < 0 for n in (incoming, outgoing)):
        return {}, None
    usage = {"input_tokens": incoming, "output_tokens": outgoing}
    cost = (Decimal(incoming) * Decimal("0.3") + Decimal(outgoing) * Decimal("0.5")) / Decimal(1_000_000)
    return usage, str(cost)


def _parse_response(data: object, page: int, key: str) -> OCRPageResult:
    if not isinstance(data, dict):
        return OCRPageResult("failed", page, warnings=("OCR 返回内容无法解析，请查看原页。",),
                             error_code="invalid_response")
    request_id = _safe_identifier(data.get("request_id"), key)
    usage, cost = _usage_and_cost(data.get("usage"))
    output = data.get("output")
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return OCRPageResult("failed", page, request_id=request_id, usage=usage, cost_cny=cost,
                             warnings=("OCR 返回内容不完整，请查看原页。",), error_code="invalid_response")
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(item["text"] for item in content
                         if isinstance(item, dict) and isinstance(item.get("text"), str))
    else:
        text = ""
    if choice.get("finish_reason") != "stop" or not text.strip():
        return OCRPageResult("failed", page, request_id=request_id, usage=usage, cost_cny=cost,
                             warnings=("OCR 本页未完整返回可用文字，请查看原页。",),
                             error_code="truncated_or_empty")
    if len(text) > MAX_OCR_TEXT_CHARS:
        return OCRPageResult("failed", page, request_id=request_id, usage=usage, cost_cny=cost,
                             warnings=("OCR 本页文字过长，未纳入识别结果；请查看原页。",),
                             error_code="ocr_text_too_large")
    text = _safe_text(text, key)
    return OCRPageResult("completed", page, text=text, request_id=request_id, usage=usage,
                         cost_cny=cost, warnings=(_UNVERIFIED,))


class VLOCRClient:
    """Synchronous client; worker must own and drain its enclosing task thread."""

    def __init__(self, root: str | Path, settings: OCRSettings | None = None, transport=None):
        self.root = Path(os.path.abspath(root))
        self.settings = settings if settings is not None else OCRSettings.from_root(self.root)
        if not isinstance(self.settings, OCRSettings):
            raise TypeError("OCR settings 无效。")
        if (type(self.settings.enabled) is not bool
                or isinstance(self.settings.timeout_seconds, bool)
                or not isinstance(self.settings.timeout_seconds, (int, float))
                or not math.isfinite(self.settings.timeout_seconds)
                or not 1 <= self.settings.timeout_seconds <= 300):
            raise ValueError("OCR 配置中的开关或超时时间无效。")
        self.transport = transport

    def peek_cached_page(self, *, page: int, cache_dir: Path) -> OCRPageResult | None:
        """Read an existing validated page record without rendering or sending a request."""
        if type(page) is not int or not 1 <= page <= 1_000_000:
            raise ValueError("OCR 页码必须是正整数。")
        if not self.settings.enabled:
            return None
        _endpoint(self.settings.base_url)
        key = self.settings.api_key
        if not key:
            raise ValueError("OCR 普通按量 API 密钥尚未配置。")
        directory = _private_cache_directory(self.root, cache_dir)
        marker = directory / f"page-{page:06d}.json"
        if marker.exists() or not _not_link(marker):
            return _cached_result(marker, page, key)
        return None

    def recognize_page(self, png: bytes, *, page: int, cache_dir: Path,
                       before_request: Callable | None = None) -> OCRPageResult:
        if type(page) is not int or not 1 <= page <= 1_000_000:
            raise ValueError("OCR 页码必须是正整数。")
        if before_request is not None and not callable(before_request):
            raise TypeError("before_request 必须是可调用对象。")
        if not self.settings.enabled:
            return OCRPageResult("failed", page, warnings=("在线 OCR 未启用，请查看原页。",),
                                 error_code="ocr_disabled")
        endpoint = _endpoint(self.settings.base_url)
        key = self.settings.api_key
        if not key:
            raise ValueError("OCR 普通按量 API 密钥尚未配置。")
        _validate_image(png)
        directory = _private_cache_directory(self.root, cache_dir)
        marker = directory / f"page-{page:06d}.json"
        if marker.exists() or not _not_link(marker):
            return _cached_result(marker, page, key)

        payload = {
            "model": MODEL,
            "input": {"messages": [{"role": "user", "content": [{
                "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                "min_pixels": 3072, "max_pixels": 8_388_608, "enable_rotate": False,
            }]}]},
            "parameters": {"max_tokens": 8192, "ocr_options": {"task": "document_parsing"}},
        }
        check_request_control()
        if before_request is not None:
            before_request()
        attempt_id = uuid4().hex
        try:
            _create_record(marker, {"schema": _SCHEMA, "status": "started", "page": page,
                                    "model": MODEL, "attempt_id": attempt_id})
        except FileExistsError:
            return _cached_result(marker, page, key)
        try:
            check_request_control()
        except BaseException:
            try:
                _clear_unstarted(marker, attempt_id)
            except OSError:
                # A marker that could not be removed remains unconfirmed and
                # prevents another paid call; cancellation must still escape.
                pass
            raise

        published = False
        try:
            deadline = monotonic() + self.settings.timeout_seconds
            timeout = httpx.Timeout(self.settings.timeout_seconds,
                                    connect=min(15.0, self.settings.timeout_seconds),
                                    write=min(60.0, self.settings.timeout_seconds),
                                    pool=min(15.0, self.settings.timeout_seconds))
            with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False,
                              transport=self.transport) as client:
                with client.stream("POST", endpoint, json=payload,
                                   headers={"Authorization": "Bearer " + key,
                                            "Accept-Encoding": "identity"}) as response:
                    body = bytearray()
                    encoding = response.headers.get("content-encoding", "").strip().lower()
                    if encoding not in ("", "identity"):
                        # No implicit decompression: a compressed response can
                        # expand past the bounded worker memory budget.
                        raise ValueError("OCR 响应编码不受支持。")
                    for chunk in response.iter_bytes():
                        try:
                            check_request_control()
                        except BaseException as exc:
                            # Do not let a control error be mistaken for a
                            # transport/JSON failure and silently swallowed.
                            raise _ControlAbort(exc) from None
                        if monotonic() >= deadline:
                            raise ValueError("OCR 请求超过时限。")
                        if len(chunk) > MAX_RESPONSE_BYTES - len(body):
                            raise ValueError("OCR 响应过大。")
                        body.extend(chunk)
                    status = response.status_code
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                data = None
            if status == 200:
                result = _parse_response(data, page, key)
            else:
                request_id = _safe_identifier(data.get("request_id"), key) if isinstance(data, dict) else None
                code = _safe_identifier(data.get("code"), key) if isinstance(data, dict) else None
                usage, cost = _usage_and_cost(data.get("usage") if isinstance(data, dict) else None)
                result = OCRPageResult("failed", page, request_id=request_id, usage=usage, cost_cny=cost,
                                       warnings=("OCR 服务未成功读取本页，请查看原页。",),
                                       error_code=code or f"http_{status}")
            value = {"schema": _SCHEMA, **result.to_dict()}
            _replace_record(marker, value)
            published = True
        except _ControlAbort as exc:
            if not published:
                _replace_record(marker, {"schema": _SCHEMA, **OCRPageResult(
                    "unknown", page, warnings=("OCR 请求结果未确认，未自动重复请求；请查看原页。",),
                    error_code="request_unconfirmed").to_dict()})
            raise exc.original
        except (httpx.HTTPError, OSError, ValueError):
            result = OCRPageResult("unknown", page,
                                   warnings=("OCR 请求结果未确认，未自动重复请求；请查看原页。",),
                                   error_code="request_unconfirmed")
            if not published:
                _replace_record(marker, {"schema": _SCHEMA, **result.to_dict()})
                published = True
        except BaseException:
            if not published:
                _replace_record(marker, {"schema": _SCHEMA, **OCRPageResult(
                    "unknown", page, warnings=("OCR 请求结果未确认，未自动重复请求；请查看原页。",),
                    error_code="request_unconfirmed").to_dict()})
            raise
        check_request_control()
        return result


class _ControlAbort(BaseException):
    def __init__(self, original: BaseException):
        self.original = original
