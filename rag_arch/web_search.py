"""Native web search via a separate provider; main chat stays Chat Completions.

Docs checked 2026-09-13:
https://help.aliyun.com/zh/model-studio/web-search
https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

Chat Completions documents enable_search/forced_search, but not guaranteed
structured source return (enable_source/enable_citation are DashScope-only).
Do not infer performed searches or verified facts from the requested switch.
"""
from copy import deepcopy
import re

import httpx
from urllib.parse import urlsplit

from langchain_openai import ChatOpenAI

# Provider capability data; business code does not depend on a Qwen model name.
_BAILIAN_SEARCH_FAMILIES = (
    "qwen3.8-max", "qwen3.8-flash", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash", "qwen3.6-max-preview",
    "qwen3.6-plus", "qwen3.6-flash", "qwen3.6-27b", "qwen3.6-35b-a3b",
    "qwen3.5-plus", "qwen3.5-flash", "qwen3.5-27b", "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "qwen3-max", "qwen-max",
    "qwen-plus", "qwen-flash", "qwen-turbo", "qwq-plus",
    "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v3.2",
)
_SOURCE_FIELDS = ("search_info", "search_results", "citations", "annotations", "sources")


def chat_web_search_status(settings):
    """Configuration support, never evidence that a particular turn searched."""
    requested = bool(getattr(settings, "chat_web_search_enabled", False))
    address = urlsplit(settings.chat_base_url)
    host = (address.hostname or "").lower()
    provider = "bailian" if (host.endswith(".maas.aliyuncs.com") or host in {
        "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}) else "unknown"
    supported = provider == "bailian"
    reason = "provider_not_supported"
    if host.startswith("token-plan."):
        supported, reason = False, "tokenplan_requires_separate_protocol_validation"
    elif supported and not address.path.rstrip("/").endswith("/compatible-mode/v1"):
        supported, reason = False, "endpoint_protocol_not_supported"
    elif supported:
        name = settings.chat_model.lower()
        supported = any(name == family or (name.startswith(family + "-") and
                        re.fullmatch(r"(?:\d{4}|\d{4}-\d{2}-\d{2}|latest|preview|character)", name[len(family) + 1:]))
                        for family in _BAILIAN_SEARCH_FAMILIES)
        reason = "provider_documented" if supported else "model_not_documented_for_chat_search"
    return {"requested": requested, "enabled": requested and supported, "supported": bool(supported),
            "provider": provider,
            "protocol": "chat_completions", "mode": "auto" if requested and supported else "off",
            "reason": reason if requested else "disabled_by_setting",
            "structured_sources_guaranteed": False, "performed": None,
            "compatibility_note": "文档确认搜索参数；非流式、多模态、函数工具与结构化输出的组合需以实际服务返回核验。"}


def web_search_extra_body(settings):
    """Only documented parameters; leave the decision and default strategy to AI."""
    if not chat_web_search_status(settings)["enabled"]:
        return {}
    return {"enable_search": True, "search_options": {"forced_search": False}}


def _http_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username else None
    except ValueError:
        return None


def _source_rows(metadata):
    """Read known provider source containers, never parse answer prose as facts."""
    if not isinstance(metadata, dict):
        return []
    rows = []
    info = metadata.get("search_info")
    if isinstance(info, dict):
        rows.extend(_source_rows(info))
    for key in ("search_results", "sources", "citations", "annotations"):
        value = metadata.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                item = {"url": item}
            if not isinstance(item, dict):
                continue
            citation = item.get("url_citation", item)
            if not isinstance(citation, dict):
                continue
            url = _http_url(citation.get("url"))
            if url:
                row = {"url": url, "title": citation.get("title") or url,
                       "source_kind": "web", "source_origin": "provider_metadata",
                       "verified": False}
                for field in ("index", "id", "citation", "start_index", "end_index", "snippet"):
                    if field in citation:
                        row[field] = deepcopy(citation[field])
                rows.append(row)
    return rows


def extract_web_sources(message):
    """Return actual provider-supplied links; absent metadata yields no sources."""
    rows = []
    for metadata in (getattr(message, "response_metadata", {}), getattr(message, "additional_kwargs", {})):
        rows.extend(_source_rows(metadata))
        rows.extend(_source_rows({"sources": metadata.get("native_web_sources", [])}))
        web = metadata.get("web_search", {})
        if isinstance(web, dict):
            for payload in web.get("provider_metadata", {}).values():
                rows.extend(_source_rows(payload))
    unique = {}
    for row in rows:
        unique.setdefault(row["url"], row)
    return list(unique.values())


class WebSearchChat(ChatOpenAI):
    """Preserve optional provider search metadata through non-streaming invoke."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for choice, generation in zip(raw.get("choices", []), result.generations):
            containers = {"response": raw, "choice": choice, "message": choice.get("message", {})}
            metadata = {name: {key: deepcopy(value[key]) for key in _SOURCE_FIELDS if key in value}
                        for name, value in containers.items() if isinstance(value, dict)}
            metadata = {name: value for name, value in metadata.items() if value}
            if metadata:
                generation.message.response_metadata["web_search"] = {"provider_metadata": metadata}
        return result


# Responses exposes search events and actual source URLs, unlike hidden Chat search.
_RESPONSES_SEARCH_FAMILIES = tuple(name for name in _BAILIAN_SEARCH_FAMILIES if name not in {
    "qwen3.6-max-preview", "qwen3.6-27b", "qwen-max", "qwen-plus", "qwen-flash", "qwen-turbo", "qwq-plus", "deepseek-v3.2"
}) + ("glm-5.2", "kimi-k3")


def native_web_search_status(settings):
    """Capability of the explicit search_web tool, independent of main chat."""
    requested = bool(getattr(settings, "chat_web_search_enabled", False))
    address = urlsplit(settings.chat_base_url)
    host = (address.hostname or "").lower()
    provider = ("bailian_tokenplan" if host == "token-plan.cn-beijing.maas.aliyuncs.com" else
                "bailian" if host.endswith(".maas.aliyuncs.com") or host in {
                    "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"} else "unknown")
    name = settings.chat_model.lower()
    supported = (provider != "unknown" and address.path.rstrip("/").endswith("/compatible-mode/v1")
                 and any(name == family or (name.startswith(family + "-") and
                    re.fullmatch(r"(?:\d{4}|\d{4}-\d{2}-\d{2}|preview)", name[len(family) + 1:]))
                    for family in _RESPONSES_SEARCH_FAMILIES))
    enabled = requested and supported and settings.chat_ready
    reason = ("disabled_by_setting" if not requested else "provider_or_model_not_supported" if not supported
              else "missing_chat_configuration" if not settings.chat_ready else "provider_documented")
    return {"requested": requested, "supported": bool(supported), "enabled": bool(enabled),
            "provider": provider, "protocol": "responses", "mode": "tool" if enabled else "off",
            "reason": reason, "performed": None, "structured_sources_guaranteed": False,
            "compatibility_note": "search_web按需调用独立Responses；只按返回的搜索事件和来源登记，不保证每次找到网页。"}


def _responses_search_result(payload, query):
    events, rows, paragraphs = [], [], []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            events.append({key: deepcopy(item[key]) for key in ("id", "status", "action") if key in item})
            rows.extend(_source_rows(item.get("action", {})))
        elif item.get("type") == "message":
            for block in item.get("content", []):
                if isinstance(block, dict):
                    if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                        paragraphs.append(block["text"])
                    rows.extend(_source_rows(block))
    usage = deepcopy(payload.get("usage") or {})
    statistics = [payload.get("x_tools") or {}, usage.get("x_tools") or {}]
    statistics.extend(item.get("plugins") or {} for item in usage.get("x_details", []) if isinstance(item, dict))
    reported_counts = [value.get("web_search", {}).get("count", 0) for value in statistics if isinstance(value, dict)]
    count = max([len(events)] + [value for value in reported_counts if isinstance(value, int) and value >= 0])
    unique = {}
    for row in rows:
        current = unique.setdefault(row["url"], row)
        # Search events often contain only URL; answer annotations may add a title.
        if current.get("title") == row["url"] and row.get("title") != row["url"]:
            current["title"] = row["title"]
        if row.get("snippet") and not current.get("snippet"):
            current["snippet"] = row["snippet"]
    status = payload.get("status")
    result = {"ok": status in {"completed", "incomplete"} and not payload.get("error"),
              "performed": bool(events or count or rows), "query": query, "summary": "\n\n".join(paragraphs),
              "sources": list(unique.values()), "search_calls": count, "search_events": events,
              "status": status, "response_id": payload.get("id"), "usage": usage}
    if payload.get("error"):
        result["error"] = deepcopy(payload["error"])
    if payload.get("incomplete_details"):
        result["incomplete_details"] = deepcopy(payload["incomplete_details"])
    return result


class NativeWebSearchProvider:
    """One Responses request per search; no hidden retries or document upload."""

    def __init__(self, settings, client=None, *, async_client=None,
                 timeout_seconds=90, max_output_tokens=1500):
        self.settings, self.client = settings, client
        self.async_client = async_client
        self._owns_async_client = False
        self.timeout_seconds = min(90, max(1, timeout_seconds))
        self.max_output_tokens = max(16, max_output_tokens)
        self.status = native_web_search_status(settings)
        self.available = self.status["enabled"]

    def search(self, query: str) -> dict:
        from .runtime_audit import _request_hook, _response_hook, record_event
        base = {"ok": False, "performed": False, "query": query, "summary": "", "sources": [], "search_calls": 0}
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            return {**base, "error": "query须为1至2000字符的公开搜索问题。"}
        if not self.available:
            return {**base, "error": self.status["reason"]}
        body = {"model": self.settings.chat_model, "input": query.strip(),
                "tools": [{"type": "web_search"}], "tool_choice": "auto", "store": False,
                "max_output_tokens": self.max_output_tokens, "reasoning": {"effort": "low"}}
        owned = self.client is None
        client = self.client or httpx.Client(timeout=self.timeout_seconds, follow_redirects=False)
        # Reuse the context-aware hooks without stealing an injected client's owner.
        for event, hook in (("request", _request_hook), ("response", _response_hook)):
            if hook not in client.event_hooks[event]:
                client.event_hooks[event].append(hook)
        try:
            from .request_control import check_request_control
            check_request_control()
            response = client.post(self.settings.chat_base_url.rstrip("/") + "/responses",
                headers={"Authorization": "Bearer " + self.settings.chat_api_key}, json=body,
                timeout=self.timeout_seconds)
            if not response.is_success:
                result = {**base, "error": f"native_search_http_{response.status_code}", "http_status": response.status_code}
            else:
                payload = response.json()
                result = _responses_search_result(payload, query)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            result = {**base, "error": type(exc).__name__}
        finally:
            if owned:
                client.close()
        record_event("native_web_search_result", component="native_web_search", query=query,
                     result=result, independent_provider_call=True)
        return result

    def _get_async_client(self):
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False)
            self._owns_async_client = True
        from .runtime_audit import ensure_async_request_audit
        return ensure_async_request_audit(self.async_client)

    async def asearch(self, query: str) -> dict:
        """Perform the same Responses search through a native AsyncClient path."""
        from .runtime_audit import arecord_async_response, arecord_event
        base = {"ok": False, "performed": False, "query": query, "summary": "", "sources": [], "search_calls": 0}
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            return {**base, "error": "query须为1至2000字符的公开搜索问题。"}
        if not self.available:
            return {**base, "error": self.status["reason"]}
        body = {"model": self.settings.chat_model, "input": query.strip(),
                "tools": [{"type": "web_search"}], "tool_choice": "auto", "store": False,
                "max_output_tokens": self.max_output_tokens, "reasoning": {"effort": "low"}}
        try:
            from .request_control import check_request_control
            check_request_control()
            client = self._get_async_client()
            response = await client.post(self.settings.chat_base_url.rstrip("/") + "/responses",
                headers={"Authorization": "Bearer " + self.settings.chat_api_key}, json=body,
                timeout=self.timeout_seconds)
            await arecord_async_response(response)
            if not response.is_success:
                result = {**base, "error": f"native_search_http_{response.status_code}",
                          "http_status": response.status_code}
            else:
                result = _responses_search_result(response.json(), query)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            result = {**base, "error": type(exc).__name__}
        await arecord_event("native_web_search_result", component="native_web_search", query=query,
                            result=result, independent_provider_call=True)
        return result

    def as_langchain_tool(self):
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(self.search, coroutine=self.asearch,
            name="search_web", description=(
            "按需通过模型服务原生搜索公开网页。query仅写独立公开查询，不传本地文件或完整历史。"
            "返回真实搜索事件与外部网页来源；网页内容不自动成为已核实的本地规范证据。"))


def create_web_search_provider(settings, client=None, **kwargs):
    return NativeWebSearchProvider(settings, client=client, **kwargs)



def select_chat_protocol(settings):
    requested = getattr(settings, "chat_api_protocol", "auto")
    supported = native_web_search_status(settings)["supported"]
    if requested == "responses" and not supported:
        raise ValueError("此Provider/模型尚未登记Responses适配；可使用chat_completions。")
    return "responses" if requested == "responses" or requested == "auto" and supported else "chat_completions"


def main_native_tools(settings):
    """Project capability switches become provider tools, not API boolean flags."""
    if select_chat_protocol(settings) != "responses":
        return []
    result = []
    if settings.chat_web_search_enabled:
        result.append({"type": "web_search"})
        if settings.chat_web_extractor_enabled:
            result.append({"type": "web_extractor"})
    if settings.chat_code_interpreter_enabled:
        result.append({"type": "code_interpreter"})
    return result


def create_responses_chat(settings, *, max_tokens, thinking_budget=None, http_client=None,
                          http_async_client=None, json_output=False, stream_responses=False):
    from .responses_chat import ResponsesChat
    effort = None
    if thinking_budget is not None:
        if settings.chat_model.lower().startswith("qwen3.8"):
            # Project approximation using published effort levels, not an exact token cap.
            effort = "low" if thinking_budget <= 4096 else "medium" if thinking_budget <= 16384 else "xhigh"
        else:
            raise ValueError("此模型Responses不支持thinking_budget；请移除该专属参数，使用模型默认思考配置。")
    return ResponsesChat(model_name=settings.chat_model, api_key=settings.chat_api_key,
        base_url=settings.chat_base_url, timeout_seconds=settings.effective_chat_timeout_seconds,
        max_tokens=max_tokens, reasoning_effort=effort, native_tools=main_native_tools(settings),
        extra_body={"thinking_budget": thinking_budget} if thinking_budget is not None else {},
        http_client=http_client, http_async_client=http_async_client,
        json_output=json_output, stream_responses=bool(stream_responses))
