"""Bailian's explicit text rerank protocols; no retries or fallback models.

Supports gte-rerank-v2, qwen3-rerank, qwen3.7-text-rerank and text-only qwen3-vl-rerank. Scores compare
documents within one request; oversized sets are rejected, not split/merged.
"""
from dataclasses import dataclass, field
import math
import re
from collections.abc import Sequence
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

import httpx
from .config import DEFAULT_REQUEST_TIMEOUT_SECONDS


GTE_RERANK_V2 = "gte-rerank-v2"
QWEN3_RERANK = "qwen3-rerank"
QWEN3_7_TEXT_RERANK = "qwen3.7-text-rerank"
QWEN3_VL_RERANK = "qwen3-vl-rerank"
SUPPORTED_RERANK_MODELS = (GTE_RERANK_V2, QWEN3_RERANK, QWEN3_7_TEXT_RERANK, QWEN3_VL_RERANK)
RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
COMPATIBLE_RERANK_PATH = "/compatible-api/v1/reranks"
MAX_DOCUMENTS = 500
_MODEL_DOCUMENT_LIMITS = {GTE_RERANK_V2: MAX_DOCUMENTS,
                          QWEN3_RERANK: MAX_DOCUMENTS,
                          QWEN3_7_TEXT_RERANK: MAX_DOCUMENTS,
                          QWEN3_VL_RERANK: 100}
# Bailian documents qwen3.7-text-rerank at 500 documents per request and
# 30,000 tokens per query/document (120,000 request tokens recommended).
# No local token estimates are enforced because there is no matching tokenizer;
# document count is the only provider limit enforced here.
# These exported values are legacy GTE references, not qwen3.7 limits.
MAX_ITEM_TOKENS = 4_000
MAX_REQUEST_TOKENS = 30_000


def _validate_url(value: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("重排接口地址不能为空。")
    try:
        parsed = urlsplit(value.strip())
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("重排地址须为不含账号密码、查询参数或片段的 HTTP(S) 地址。")
    return parsed


def derive_rerank_endpoint(base_url: str, model: str = GTE_RERANK_V2) -> str:
    """Resolve a workspace origin, compatible embedding base, or native base.

    Unrecognized paths require an explicit endpoint instead of a guessed URL.
    The origin is retained, so no workspace ID is copied into project code.
    Omitting model preserves the legacy GTE path; clients pass their model.
    """
    if model not in SUPPORTED_RERANK_MODELS:
        raise ValueError("不支持此重排模型，无法确定协议路径。")
    parsed = _validate_url(base_url)
    path = parsed.path.rstrip("/")
    if path not in {"", "/compatible-mode/v1", "/compatible-api/v1", "/api/v1",
                    RERANK_PATH, COMPATIBLE_RERANK_PATH}:
        raise ValueError("无法从此根地址确定重排路径，请显式提供 endpoint。")
    target = COMPATIBLE_RERANK_PATH if model == QWEN3_RERANK else RERANK_PATH
    return urlunsplit((parsed.scheme, parsed.netloc, target, "", ""))


@dataclass(frozen=True)
class RerankResult:
    """Index in the original document list and within-request relevance score."""
    index: int
    score: float


class RerankError(RuntimeError):
    """Safe status/code only; never retain response text, request or credentials."""

    def __init__(self, code: str, status_code: int | None = None,
                 detail: str = ""):
        self.code = code
        self.status_code = status_code
        status = str(status_code) if status_code is not None else "未收到响应"
        super().__init__(f"百炼重排失败（HTTP {status}，code={code}）。{detail}")


def _service_code(payload, secret: str) -> str:
    code = payload.get("code") if isinstance(payload, dict) else None
    if not code and isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        code = payload["error"].get("code")
    if (isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,95}", code)
            and secret not in code):
        return code
    return "RemoteError"


def _parse_results(payload, document_count: int, expected_count: int,
                   status_code: int, *, compatible: bool = False) -> list[RerankResult]:
    def invalid(detail):
        return RerankError("InvalidResponse", status_code, detail)

    output = payload if compatible else (payload.get("output") if isinstance(payload, dict) else None)
    raw = output.get("results") if isinstance(output, dict) else None
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise invalid("返回结果数量与请求不符。")
    results: list[RerankResult] = []
    seen: set[int] = set()
    previous_score = math.inf
    for item in raw:
        if not isinstance(item, dict):
            raise invalid("结果条目格式错误。")
        index, score = item.get("index"), item.get("relevance_score")
        if (isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < document_count or index in seen):
            raise invalid("结果索引重复、越界或不是整数。")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= score <= 1 or not math.isfinite(score)):
            raise invalid("相关性分数须为 0 到 1 之间的有限数值。")
        if score > previous_score:
            raise invalid("结果未按相关性分数从高到低排列。")
        results.append(RerankResult(index, float(score)))
        seen.add(index)
        previous_score = score
    return results


@dataclass
class BailianReranker:
    """Explicit, credential-safe configuration for one rerank request.

    endpoint overrides URL derivation and supports a separately configured
    endpoint for the selected model's protocol. Construction never performs
    network requests; VL accepts only plain text through this adapter.
    """
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    model: str = QWEN3_RERANK
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    endpoint: str | None = field(default=None, repr=False)
    _resolved_endpoint: str = field(init=False, repr=False)
    last_metadata: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("重排 API Key 未配置。")
        self.api_key = self.api_key.strip()
        if not self.api_key.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in self.api_key):
            raise ValueError("重排 API Key 格式无效。")
        if self.model not in SUPPORTED_RERANK_MODELS:
            raise ValueError("重排适配仅支持 gte-rerank-v2、qwen3-rerank、qwen3.7-text-rerank 和 qwen3-vl-rerank；不会自动换模型。")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("重排超时须为正的有限秒数。")
        if self.endpoint is not None:
            _validate_url(self.endpoint)
            self._resolved_endpoint = self.endpoint.strip()
        else:
            self._resolved_endpoint = derive_rerank_endpoint(self.base_url, self.model)

    def rerank(self, query: str, documents: Sequence[str],
               top_n: int | None = None) -> list[RerankResult]:
        self.last_metadata = None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("重排问题须为非空文本。")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("重排文档须为文本序列。")
        items = list(documents)
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("重排文档须全部为非空文本。")
        document_limit = _MODEL_DOCUMENT_LIMITS[self.model]
        if len(items) > document_limit:
            raise ValueError(f"{self.model} 每次最多重排 {document_limit} 份文档；请在召回层限制候选数。")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
            raise ValueError("top_n 须为正整数或 None。")
        if not items:
            return []
        expected_count = len(items) if top_n is None else min(top_n, len(items))
        compatible = self.model == QWEN3_RERANK
        if compatible:
            payload = {"model": self.model, "query": query,
                       "documents": items, "top_n": expected_count}
        else:
            parameters = {"top_n": expected_count}
            # Bailian documents return_documents for GTE/VL only; qwen3.7
            # returns scores in the native envelope without that parameter.
            if self.model in {GTE_RERANK_V2, QWEN3_VL_RERANK}:
                parameters["return_documents"] = False
            payload = {"model": self.model,
                       "input": {"query": query, "documents": items},
                       "parameters": parameters}
        started = perf_counter()
        from .request_control import check_request_control
        check_request_control()
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.post(self._resolved_endpoint, json=payload,
                                       headers={"Authorization": f"Bearer {self.api_key}",
                                                "Content-Type": "application/json"})
        except httpx.TimeoutException:
            raise RerankError("Timeout") from None
        except httpx.RequestError:
            raise RerankError("TransportError") from None
        try:
            body = response.json()
        except ValueError:
            code = "InvalidResponse" if response.is_success else "RemoteError"
            raise RerankError(code, response.status_code, "服务未返回有效 JSON。") from None
        if not response.is_success or (isinstance(body, dict) and (body.get("code") or body.get("error"))):
            raise RerankError(_service_code(body, self.api_key), response.status_code)
        results = _parse_results(body, len(items), expected_count, response.status_code,
                                 compatible=compatible)
        request_id = body.get("request_id") or body.get("id") if isinstance(body, dict) else None
        usage = (body.get("usage") or {}).get("total_tokens") if isinstance(body, dict) else None
        if compatible and (not isinstance(request_id, str) or not request_id
                           or not isinstance(usage, int) or usage <= 0):
            raise RerankError("InvalidResponse", response.status_code,
                              "qwen3响应缺少request_id或usage.total_tokens。")
        latency_ms = round((perf_counter() - started) * 1000, 3)
        self.last_metadata = {
            "provider": "bailian",
            "model": self.model,
            "request_id": request_id if isinstance(request_id, str) and request_id else None,
            "usage_total_tokens": usage if isinstance(usage, int) and usage > 0 else None,
            "latency_ms": latency_ms,
            "actual_cost_cny": (round(usage / 2_000_000, 9)
                                if self.model == QWEN3_RERANK and isinstance(usage, int) and usage > 0
                                else None),
            "document_count": len(items),
            "top_n": expected_count,
        }
        return results

    async def arerank(self, query: str, documents: Sequence[str],
                      top_n: int | None = None) -> list[RerankResult]:
        """Run the same protocol through HTTPX's native asynchronous client."""
        self.last_metadata = None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("重排问题须为非空文本。")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("重排文档须为文本序列。")
        items = list(documents)
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("重排文档须全部为非空文本。")
        document_limit = _MODEL_DOCUMENT_LIMITS[self.model]
        if len(items) > document_limit:
            raise ValueError(f"{self.model} 每次最多重排 {document_limit} 份文档；请在召回层限制候选数。")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
            raise ValueError("top_n 须为正整数或 None。")
        if not items:
            return []
        expected_count = len(items) if top_n is None else min(top_n, len(items))
        compatible = self.model == QWEN3_RERANK
        if compatible:
            payload = {"model": self.model, "query": query,
                       "documents": items, "top_n": expected_count}
        else:
            parameters = {"top_n": expected_count}
            if self.model in {GTE_RERANK_V2, QWEN3_VL_RERANK}:
                parameters["return_documents"] = False
            payload = {"model": self.model,
                       "input": {"query": query, "documents": items},
                       "parameters": parameters}
        started = perf_counter()
        from .request_control import check_request_control
        check_request_control()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.post(
                    self._resolved_endpoint, json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"})
        except httpx.TimeoutException:
            raise RerankError("Timeout") from None
        except httpx.RequestError:
            raise RerankError("TransportError") from None
        try:
            body = response.json()
        except ValueError:
            code = "InvalidResponse" if response.is_success else "RemoteError"
            raise RerankError(code, response.status_code, "服务未返回有效 JSON。") from None
        if not response.is_success or (isinstance(body, dict) and (body.get("code") or body.get("error"))):
            raise RerankError(_service_code(body, self.api_key), response.status_code)
        results = _parse_results(body, len(items), expected_count, response.status_code,
                                 compatible=compatible)
        request_id = body.get("request_id") or body.get("id") if isinstance(body, dict) else None
        usage = (body.get("usage") or {}).get("total_tokens") if isinstance(body, dict) else None
        if compatible and (not isinstance(request_id, str) or not request_id
                           or not isinstance(usage, int) or usage <= 0):
            raise RerankError("InvalidResponse", response.status_code,
                              "qwen3响应缺少request_id或usage.total_tokens。")
        latency_ms = round((perf_counter() - started) * 1000, 3)
        self.last_metadata = {
            "provider": "bailian",
            "model": self.model,
            "request_id": request_id if isinstance(request_id, str) and request_id else None,
            "usage_total_tokens": usage if isinstance(usage, int) and usage > 0 else None,
            "latency_ms": latency_ms,
            "actual_cost_cny": (round(usage / 2_000_000, 9)
                                if self.model == QWEN3_RERANK and isinstance(usage, int) and usage > 0
                                else None),
            "document_count": len(items),
            "top_n": expected_count,
        }
        return results
