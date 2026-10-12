"""Loopback-only reranking client; model files stay in the separate local service.

Construction has no network or model-loading side effects. ``prepare`` reads the
service identity before an experiment snapshot; regular calls prepare lazily.
No API key, environment proxy, redirect, retry, or cloud fallback is used.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import re
from urllib.parse import urlsplit

import httpx
from .config import DEFAULT_REQUEST_TIMEOUT_SECONDS

from .reranking import MAX_DOCUMENTS, RerankError, RerankResult, _parse_results

LOCAL_MODEL = "BAAI/bge-reranker-v2-m3"
_IDENTITY_FIELDS = ("revision", "device", "dtype", "max_length", "batch_size")


def validate_local_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        valid = (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                 and not parsed.username and not parsed.password and not parsed.query
                 and not parsed.fragment and parsed.path in {"", "/"})
        _ = parsed.port
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("本地重排地址须为 HTTP loopback 根地址，例如 http://127.0.0.1:8766。")
    return value.strip().rstrip("/")


class LocalRerankError(RerankError):
    """A bounded local status, without service payloads or source text."""
    def __init__(self, code: str, status_code: int | None = None, detail: str = ""):
        self.code, self.status_code = code, status_code
        RuntimeError.__init__(self, f"本地重排失败（code={code}，HTTP={status_code}）。{detail}")


def _identity(payload) -> dict:
    if not isinstance(payload, dict):
        raise LocalRerankError("InvalidResponse", detail="缺少本地模型身份。")
    result = {name: payload.get(name) for name in _IDENTITY_FIELDS}
    if (not isinstance(result["revision"], str)
            or not re.fullmatch(r"[\w./-]{1,200}", result["revision"])
            or not isinstance(result["device"], str)
            or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", result["device"])
            or not isinstance(result["dtype"], str)
            or result["dtype"] not in {"float32", "float16", "bfloat16"}
            or any(type(result[name]) is not int or result[name] < 1
                   for name in ("max_length", "batch_size"))):
        raise LocalRerankError("InvalidResponse", detail="本地模型身份字段无效。")
    return result


@dataclass
class LocalReranker:
    base_url: str = "http://127.0.0.1:8766"
    model: str = LOCAL_MODEL
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    provider: str = field(default="local", init=False)
    _identity: dict | None = field(default=None, init=False, repr=False)
    last_metadata: dict | None = field(default=None, init=False)

    def __post_init__(self):
        self.base_url = validate_local_base_url(self.base_url)
        if self.model != LOCAL_MODEL:
            raise ValueError(f"本地客户端当前支持 {LOCAL_MODEL}；不会自动下载或切换模型。")
        if (type(self.timeout) not in {int, float} or not math.isfinite(self.timeout)
                or self.timeout <= 0):
            raise ValueError("本地重排超时须为正的有限秒数。")

    @property
    def revision(self):
        return (self._identity or {}).get("revision")

    @property
    def device(self):
        return (self._identity or {}).get("device")

    @property
    def dtype(self):
        return (self._identity or {}).get("dtype")

    @property
    def max_length(self):
        return (self._identity or {}).get("max_length")

    @property
    def batch_size(self):
        return (self._identity or {}).get("batch_size")

    def _request(self, method, path, payload=None):
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                response = client.request(method, self.base_url + path, json=payload)
        except httpx.TimeoutException:
            raise LocalRerankError("Timeout") from None
        except httpx.RequestError:
            raise LocalRerankError("TransportError", detail="请确认本地服务已经启动。") from None
        try:
            body = response.json()
        except ValueError:
            raise LocalRerankError("InvalidResponse", response.status_code) from None
        if not response.is_success:
            raise LocalRerankError("LocalServiceError", response.status_code)
        if not isinstance(body, dict) or body.get("error"):
            raise LocalRerankError("InvalidResponse", response.status_code)
        return body, response.status_code

    def prepare(self) -> dict:
        """Explicit health request once; return only public, actual model settings."""
        if self._identity is None:
            body, status = self._request("GET", "/health")
            if body.get("status") not in ("ok", "ready") or body.get("model") != self.model:
                raise LocalRerankError("ModelMismatch", status, "本地服务未就绪或模型与配置不一致。")
            self._identity = _identity(body)
        return {"provider": self.provider, "model": self.model, **self._identity}

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("重排问题须为非空文本。")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("重排文档须为文本序列。")
        items = list(documents)
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("重排文档须全部为非空文本。")
        if len(items) > MAX_DOCUMENTS:
            raise ValueError("每次最多重排500份文档，请在召回层限制候选数。")
        if top_n is not None and (type(top_n) is not int or top_n < 1):
            raise ValueError("top_n 须为正整数或 None。")
        if not items:
            return []
        expected_count = len(items) if top_n is None else min(top_n, len(items))
        self.prepare()
        self.last_metadata = None
        body, status = self._request("POST", "/rerank", {
            "model": self.model, "query": query, "documents": items, "top_n": expected_count})
        metadata = body.get("metadata")
        if body.get("model") != self.model or _identity(metadata) != self._identity:
            raise LocalRerankError("IdentityChanged", status,
                                   "服务模型或推理配置发生变化，请重新建立客户端并另开实验。")
        truncated = metadata.get("truncated_pairs")
        if type(truncated) is not int or not 0 <= truncated <= len(items):
            raise LocalRerankError("InvalidResponse", status, "缺少有效的截断计数。")
        try:
            results = _parse_results(body, len(items), expected_count, status, compatible=True)
        except RerankError:
            raise LocalRerankError("InvalidResponse", status, "重排结果数量、索引或分数无效。") from None
        self.last_metadata = {**self._identity, "truncated_pairs": truncated,
                              "document_count": len(items)}
        # Preserve source indexes and make equal scores deterministic.
        return sorted(results, key=lambda result: (-result.score, result.index))
