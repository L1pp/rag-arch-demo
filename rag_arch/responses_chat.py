"""Responses adapter for native and local tools.

Plan review receives SSE internally, then assembles one complete Responses
object for the existing Agent. Other assistants stay non-streaming unless an
explicit process opt-in is set. The UI is not a token renderer. Provider raw
output is retained for history; only function_call items become LangChain
tool_calls. API keys and raw image payloads stay out of application logs.
"""
from copy import deepcopy
import asyncio
import json
import os
import time
from typing import Any, ClassVar

import httpx
from pydantic import Field, PrivateAttr, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .model_input_view import MODEL_INPUT_VIEW_VERSION, model_message_content
from .config import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .request_control import check_request_control
from .web_search import _http_url, _source_rows

NATIVE_NAMES = ("web_search", "web_extractor", "code_interpreter")
PLAN_RESPONSES_STREAM_ENV = "RAG_PLAN_RESPONSES_STREAM"
_STREAM_PREVIEW_CHARS = 240
_ERROR_MESSAGE_CHARS = 320


def plan_responses_stream_enabled():
    """Process opt-in. Plan review also sets ResponsesChat.stream_responses."""
    return os.environ.get(PLAN_RESPONSES_STREAM_ENV, "").strip().lower() in {"1", "true", "yes"}


def _timeout_seconds(timeout, default):
    if isinstance(timeout, (int, float)):
        return float(timeout)
    read = getattr(timeout, "read", None)
    if isinstance(read, (int, float)):
        return float(read)
    return float(default)


def _provider_error_summary(payload):
    """Extract bounded business error fields from either Responses error shape."""
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("response")
    candidates = (payload.get("error"), nested.get("error") if isinstance(nested, dict) else None,
                  payload)
    for detail in candidates:
        if not isinstance(detail, dict):
            continue
        code = detail.get("code")
        message = detail.get("message")
        if isinstance(code, str) or isinstance(message, str):
            return {**({"code": code[:128]} if isinstance(code, str) else {}),
                    **({"message": message[:_ERROR_MESSAGE_CHARS]} if isinstance(message, str) else {})}
    return {}


class ResponsesServiceError(RuntimeError):
    """Keep bounded provider fields for classification without exposing text in str()."""
    def __init__(self, response, payload):
        self.response = response
        self.status_code = response.status_code
        self.error_body = {"error": _provider_error_summary(payload)}
        self.code = self.error_body["error"].get("code")
        super().__init__(f"Responses service failed (HTTP {self.status_code})")


def _input_content(content, role):
    if isinstance(content, str):
        return content
    result = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("不支持的Responses消息内容。")
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": "output_text" if role == "assistant" else "input_text", "text": block["text"]})
        elif kind == "image_url":
            image = block["image_url"]
            result.append({"type": "input_image", "image_url": image if isinstance(image, str) else image["url"]})
        elif kind == "input_image":
            result.append(deepcopy(block))
        elif kind == "image" and block.get("base64"):
            result.append({"type": "input_image", "image_url": f"data:{block.get('mime_type', 'image/png')};base64,{block['base64']}"})
        else:
            raise ValueError(f"未适配的Responses内容类型：{kind}")
    return result


def _invalid_function_call_names(message: AIMessage) -> dict[str, str]:
    """Return only ordinary invalid calls with provider-owned ids and names."""
    result = {}
    for call in message.invalid_tool_calls or []:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id", call.get("call_id"))
        name = call.get("name")
        if (isinstance(call_id, str) and call_id.strip()
                and isinstance(name, str) and name.strip()
                and name != "submit_answer"):
            result[call_id] = name
    return result


def _invalid_function_call_rows(message: AIMessage) -> list[dict]:
    """Recreate no raw Responses output using the original invalid argument text."""
    rows = []
    for call_id, name in _invalid_function_call_names(message).items():
        call = next((item for item in message.invalid_tool_calls
                     if isinstance(item, dict)
                     and item.get("id", item.get("call_id")) == call_id), None)
        arguments = call.get("args") if isinstance(call, dict) else None
        if not isinstance(arguments, str):
            continue
        rows.append({"type": "function_call", "call_id": call_id,
                     "name": name, "arguments": arguments})
    return rows


def responses_input(messages):
    rows = []
    for message in messages:
        if isinstance(message, ToolMessage):
            rows.append({"type": "function_call_output", "call_id": message.tool_call_id,
                         "output": model_message_content("tool", message.content)})
            continue
        role = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
        content = model_message_content(message.type, message.content)
        if isinstance(message, AIMessage):
            raw = message.additional_kwargs.get("responses_output")
            if isinstance(raw, list):
                active = {call.get("id") for call in message.tool_calls
                          if isinstance(call, dict) and isinstance(call.get("id"), str)}
                invalid = _invalid_function_call_names(message)
                # TerminalOutputChat turns submit_answer into visible JSON. Do
                # not replay its now-completed call without a function result.
                adapted = bool(message.additional_kwargs.get("structured_output_delivery")) and not active
                for item in raw:
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id")
                        if call_id not in active and (
                                call_id not in invalid or item.get("name") != invalid[call_id]):
                            continue
                    if adapted and item.get("type") == "message":
                        continue
                    rows.append(deepcopy(item))
                if adapted and content:
                    rows.append({"role": "assistant", "content": _input_content(content, "assistant")})
                continue
            if content:
                rows.append({"role": role, "content": _input_content(content, role)})
            for call in message.tool_calls:
                rows.append({"type": "function_call", "call_id": call["id"], "name": call["name"],
                             "arguments": json.dumps(call["args"], ensure_ascii=False)})
            rows.extend(_invalid_function_call_rows(message))
            continue
        rows.append({"role": role, "content": _input_content(content, role)})
    return rows


def native_observations(payload):
    events = [deepcopy(item) for item in payload.get("output", [])
              if item.get("type") in {name + "_call" for name in NATIVE_NAMES}]
    usage = deepcopy(payload.get("usage") or {})
    stats = [payload.get("x_tools") or {}, usage.get("x_tools") or {}]
    stats.extend(item.get("plugins") or {} for item in usage.get("x_details", []) if isinstance(item, dict))
    counts = {}
    for name in NATIVE_NAMES:
        values = [sum(item.get("type") == name + "_call" for item in events)]
        values.extend(row.get(name, {}).get("count", 0) for row in stats if isinstance(row.get(name), dict))
        count = max(value for value in values if isinstance(value, int))
        if count:
            counts[name] = count
    sources = []
    for item in events:
        sources.extend(_source_rows(item.get("action", {})))
        if item.get("type") == "web_extractor_call" and item.get("status") == "completed" and item.get("output"):
            urls = item.get("urls")
            for url in (urls if isinstance(urls, list) else []):
                if _http_url(url):
                    sources.extend(_source_rows({"sources": [{"url": url}]}))
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                sources.extend(_source_rows(block))
    unique = {}
    for source in sources:
        current = unique.setdefault(source["url"], source)
        if current["title"] == source["url"] and source["title"] != source["url"]:
            current["title"] = source["title"]
    return {"native_tool_observations": events,
            "native_tool_usage": {"counts": counts, "provider_usage": usage},
            "native_source_urls": list(unique), "native_web_sources": list(unique.values())}


def responses_message(payload):
    content, reasoning, calls, invalid, raw_calls = [], [], [], [], []
    for item in payload.get("output", []):
        kind = item.get("type")
        if kind == "message":
            content.extend(block["text"] for block in item.get("content", [])
                           if block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str))
        elif kind == "reasoning":
            reasoning.extend(block["text"] for block in item.get("summary", []) if isinstance(block.get("text"), str))
        elif kind == "function_call":
            name, call_id, arguments = item.get("name"), item.get("call_id"), item.get("arguments")
            raw_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            try:
                args = json.loads(arguments)
                if not isinstance(args, dict):
                    raise ValueError("function arguments须为对象")
                calls.append({"id": call_id, "name": name, "args": args, "type": "tool_call"})
            except (ValueError, TypeError):
                invalid.append({"id": call_id, "name": name, "args": arguments if isinstance(arguments, str) else "",
                                "error": "invalid_function_arguments", "type": "invalid_tool_call"})
    usage = deepcopy(payload.get("usage") or {})
    status = payload.get("status")
    # An interrupted SSE request is an incomplete response, but it did not
    # exhaust the model's output budget. Keep the provider's reason available
    # to the Agent so a wall-clock timeout is reported as a transport failure.
    details = payload.get("incomplete_details")
    incomplete_reason = payload.get("incomplete_reason")
    if not isinstance(incomplete_reason, str) and isinstance(details, dict):
        incomplete_reason = details.get("reason")
    if status == "incomplete" and incomplete_reason == "stream_timeout":
        finish = "timeout"
    elif status == "incomplete" and incomplete_reason == "missing_terminal_event":
        finish = "incomplete"
    else:
        finish = "length" if status == "incomplete" else "tool_calls" if raw_calls else "stop"
    extra = {**native_observations(payload), "responses_output": deepcopy(payload.get("output", [])),
             "provider_raw_tool_calls": raw_calls, "reasoning_content": "\n".join(reasoning)}
    metadata = {"model_name": payload.get("model"), "response_id": payload.get("id"), "response_status": status,
                "finish_reason": finish, "token_usage": usage, "model_input_view_version": MODEL_INPUT_VIEW_VERSION}
    if isinstance(incomplete_reason, str):
        metadata["incomplete_reason"] = incomplete_reason
    if type(payload.get("request_accepted")) is bool:
        metadata["request_accepted"] = payload["request_accepted"]
    normalized_usage = None
    if all(isinstance(usage.get(k), int) for k in ("input_tokens", "output_tokens", "total_tokens")):
        normalized_usage = {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")}
        if isinstance(usage.get("input_tokens_details", {}).get("cached_tokens"), int):
            normalized_usage["input_token_details"] = {"cache_read": usage["input_tokens_details"]["cached_tokens"]}
        if isinstance(usage.get("output_tokens_details", {}).get("reasoning_tokens"), int):
            normalized_usage["output_token_details"] = {"reasoning": usage["output_tokens_details"]["reasoning_tokens"]}
    return AIMessage(content="\n".join(content), tool_calls=calls, invalid_tool_calls=invalid,
                     additional_kwargs=extra, response_metadata=metadata, usage_metadata=normalized_usage)


def _sse_line_text(line):
    if line is None:
        return ""
    return line.decode("utf-8") if isinstance(line, bytes) else line


async def iter_responses_sse(response):
    """Yield (event_name, payload) from a Responses SSE body."""
    event_type = "message"
    data_lines = []
    async for raw in response.aiter_lines():
        line = _sse_line_text(raw).rstrip("\r")
        if line == "":
            if data_lines:
                data = "\n".join(data_lines)
                if data.strip() and data.strip() != "[DONE]":
                    try:
                        yield event_type, json.loads(data)
                    except ValueError:
                        yield event_type, {"type": "unparsed_sse", "unparsed_text": data[:_STREAM_PREVIEW_CHARS]}
            event_type = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            yield "comment", {"type": "sse_comment"}
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        try:
            yield "message", json.loads(line)
        except ValueError:
            continue
    if data_lines:
        data = "\n".join(data_lines)
        if data.strip() and data.strip() != "[DONE]":
            try:
                yield event_type, json.loads(data)
            except ValueError:
                yield event_type, {"type": "unparsed_sse", "unparsed_text": data[:_STREAM_PREVIEW_CHARS]}


def _is_delta_event(data):
    kind = data.get("type") if isinstance(data, dict) else ""
    return isinstance(kind, str) and "delta" in kind


def _stream_event_audit(event_type, data):
    row = {"event": event_type}
    if not isinstance(data, dict):
        return row
    row["type"] = data.get("type")
    if (data.get("type") or event_type) in {"response.failed", "error"}:
        response = data.get("response")
        if isinstance(response, dict):
            for key, target in (("id", "response_id"), ("status", "response_status")):
                if isinstance(response.get(key), str):
                    row[target] = response[key]
        row["error_summary"] = _provider_error_summary(data)
    for key in ("status", "name", "item_id"):
        if isinstance(data.get(key), str):
            row[key] = data[key]
    item = data.get("item")
    if isinstance(item, dict):
        if isinstance(item.get("type"), str):
            row["item_type"] = item["type"]
        if isinstance(item.get("name"), str):
            row["item_name"] = item["name"]
        if isinstance(item.get("status"), str):
            row["item_status"] = item["status"]
        if isinstance(item.get("call_id"), str):
            row["call_id_tool"] = item["call_id"]
    delta = data.get("delta")
    if isinstance(delta, str):
        row["delta_chars"] = len(delta)
        row["delta_preview"] = delta[:_STREAM_PREVIEW_CHARS]
    elif isinstance(delta, dict) and isinstance(delta.get("text"), str):
        row["delta_chars"] = len(delta["text"])
        row["delta_preview"] = delta["text"][:_STREAM_PREVIEW_CHARS]
    return {key: value for key, value in row.items() if value not in (None, "")}


def _merge_delta_audit(pending, row):
    if pending and pending.get("type") == row.get("type"):
        pending["delta_count"] = pending.get("delta_count", 1) + 1
        pending["delta_chars"] = pending.get("delta_chars", 0) + row.get("delta_chars", 0)
        preview = (pending.get("delta_preview") or "") + (row.get("delta_preview") or "")
        pending["delta_preview"] = preview[:_STREAM_PREVIEW_CHARS]
        return pending
    merged = dict(row)
    merged["delta_count"] = 1
    return merged


def assemble_streamed_response(events):
    """Prefer the provider terminal object; missing completed stays incomplete."""
    completed = None
    items = []
    error = None
    for event_type, data in events:
        if not isinstance(data, dict):
            continue
        kind = data.get("type") or event_type
        if kind in {"error", "response.failed"}:
            error = data
        elif kind in {"response.completed", "response.incomplete"}:
            completed = data.get("response") if isinstance(data.get("response"), dict) else data
        elif kind == "response.output_item.done" and isinstance(data.get("item"), dict):
            items.append(deepcopy(data["item"]))
    if error:
        return None, items, error
    if isinstance(completed, dict) and (
            completed.get("output") is not None
            or completed.get("status") in {"completed", "incomplete"}):
        # Some providers omit output from the terminal event even though they
        # emitted completed output items. Preserve the terminal status/usage
        # and recover those items only when output is absent, not when the
        # provider explicitly returned an empty output list.
        if completed.get("output") is None and items:
            completed = {**completed, "output": items}
        return completed, items, None
    if items:
        return {"output": items, "status": "incomplete",
                "incomplete_reason": "missing_terminal_event"}, items, None
    return None, items, None


class ResponsesChat(BaseChatModel):
    model_name: str
    api_key: SecretStr = Field(repr=False, exclude=True)
    base_url: str
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_tokens: int = 131072
    reasoning_effort: str | None = None
    reasoning_configuration: dict | None = None
    native_tools: list[dict] = Field(default_factory=list)
    # Compatibility metadata for Qwen's locally mapped thinking budget; this
    # field is not merged into the Responses request body.
    extra_body: dict = Field(default_factory=dict)
    json_output: bool = False
    stream_responses: bool = False
    enforce_stream_deadline: bool = True
    enforce_stream_read_timeout: bool = True
    http_client: Any = Field(default=None, repr=False, exclude=True)
    http_async_client: Any = Field(default=None, repr=False, exclude=True)
    root_async_client: Any = Field(default=None, repr=False, exclude=True)
    _owns_root_async_client: bool = PrivateAttr(default=False)
    api_protocol: ClassVar[str] = "responses"
    preserves_reasoning_content: ClassVar[bool] = True
    model_input_view_version: ClassVar[str] = MODEL_INPUT_VIEW_VERSION
    # Bailian Responses reserves about 20% of the model window server-side.
    # Expose this through the provider instead of hard-coding a model name in Agents.
    input_context_ratio: ClassVar[float] = 0.8

    @property
    def _llm_type(self):
        return "compatible-responses"

    @property
    def request_timeout(self):
        return self.timeout_seconds

    @property
    def native_tool_names(self):
        return tuple(tool["type"] for tool in self.native_tools)

    @property
    def native_tools_enabled(self):
        return bool(self.native_tools)

    def _use_stream(self):
        return bool(self.stream_responses) or plan_responses_stream_enabled()

    def plain(self):
        return self.model_copy(update={"native_tools": [], "json_output": False})

    def with_native_tools(self, names):
        """Narrow provider capabilities for a workflow stage without mutation."""
        allowed = set(names)
        return self.model_copy(update={"native_tools": [deepcopy(tool) for tool in self.native_tools
                                                       if tool["type"] in allowed]})

    def bind_tools(self, tools, **kwargs):
        formatted = []
        for tool in tools:
            converted = convert_to_openai_tool(tool)
            if converted.get("type") != "function":
                raise ValueError("本地bind_tools仅接收function；原生工具由Provider能力开关控制。")
            formatted.append({"type": "function", **converted["function"]})
        return self.bind(tools=formatted, **kwargs)

    def _request_data(self, messages, stop=None, **kwargs):
        if stop:
            raise ValueError("当前Responses端点未列stop参数，不静默忽略。")
        allowed_metadata = {"thinking_budget"} if self.model_name.lower().startswith("qwen3.8") else set()
        unsupported_metadata = set(self.extra_body) - allowed_metadata
        if unsupported_metadata:
            raise ValueError("Responses extra_body不支持的参数：" + ", ".join(sorted(unsupported_metadata)))
        local = deepcopy(kwargs.pop("tools", []))
        choice = kwargs.pop("tool_choice", "auto")
        native_enabled = kwargs.pop("native_tools_enabled", True)
        timeout = kwargs.pop("timeout", self.timeout_seconds)
        if kwargs:
            raise ValueError("未支持的Responses调用选项：" + ", ".join(sorted(kwargs)))
        native = deepcopy(self.native_tools) if native_enabled and choice != "none" else []
        # A Responses session already exposes web_search directly.
        if any(tool["type"] == "web_search" for tool in native):
            local = [tool for tool in local if tool.get("name") != "search_web"]
        prepared = list(messages)
        if self.json_output:
            prepared = [SystemMessage(content="最终答复只输出一个完整JSON对象。")]+prepared
        body = {"model": self.model_name, "input": responses_input(prepared), "store": False,
                "max_output_tokens": self.max_tokens, "stream": self._use_stream()}
        if native or local:
            body.update(tools=native+local, tool_choice=choice)
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        url = self.base_url.rstrip("/") + "/responses"
        headers = {"Authorization": "Bearer " + self.api_key.get_secret_value()}
        return body, url, headers, timeout

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        body, url, headers, timeout = self._request_data(messages, stop, **kwargs)
        if body["stream"]:
            if self.http_client is not None and not isinstance(self.http_client, httpx.AsyncClient):
                raise ValueError("流式Responses测试注入须为httpx.AsyncClient；不同步复用同步http_client。")
            payload = asyncio.run(self._post_stream_async(self.http_client, url, body, timeout, headers))
            response = None
        else:
            from .runtime_audit import _request_hook, _response_hook
            if isinstance(self.http_client, httpx.AsyncClient):
                raise ValueError("非流式Responses测试注入须为httpx.Client。")
            owned = self.http_client is None
            client = self.http_client or httpx.Client(timeout=timeout, follow_redirects=False)
            if _request_hook not in client.event_hooks["request"]:
                client.event_hooks["request"].append(_request_hook)
            saved_response_hooks = list(client.event_hooks["response"])
            if _response_hook not in client.event_hooks["response"]:
                client.event_hooks["response"].append(_response_hook)
            response = None
            try:
                response = client.post(url, json=body, timeout=timeout, headers=headers)
                try:
                    payload = response.json()
                except ValueError:
                    if not response.is_success:
                        raise ResponsesServiceError(response, {}) from None
                    raise
                if not response.is_success:
                    raise ResponsesServiceError(response, payload)
            finally:
                client.event_hooks["response"] = saved_response_hooks
                if owned:
                    client.close()
        if payload.get("error") or payload.get("status") not in {"completed", "incomplete"}:
            raise ResponsesServiceError(response if response is not None else _stream_error_response(payload), payload)
        return ChatResult(generations=[ChatGeneration(message=responses_message(payload))])

    def _get_async_client(self, timeout):
        if isinstance(self.http_async_client, httpx.AsyncClient):
            client = self.http_async_client
        elif self.http_async_client is not None:
            raise ValueError("异步Responses调用注入须为httpx.AsyncClient。")
        elif isinstance(self.http_client, httpx.AsyncClient):
            client = self.http_client
        elif self.http_client is not None:
            raise ValueError("异步Responses调用注入须为httpx.AsyncClient；不同步复用同步http_client。")
        else:
            client = self.root_async_client
            if client is None:
                client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
                self.root_async_client = client
                self._owns_root_async_client = True
        from .runtime_audit import ensure_async_request_audit
        return ensure_async_request_audit(client)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Native async Responses transport; synchronous `_generate` stays intact."""
        body, url, headers, timeout = self._request_data(messages, stop, **kwargs)
        check_request_control()
        client = self._get_async_client(timeout)
        if body["stream"]:
            payload = await self._post_stream_async(client, url, body, timeout, headers)
            response = None
        else:
            from .runtime_audit import arecord_async_response
            response = await client.post(url, json=body, timeout=timeout, headers=headers)
            await arecord_async_response(response)
            try:
                payload = response.json()
            except ValueError:
                if not response.is_success:
                    raise ResponsesServiceError(response, {}) from None
                raise
            if not response.is_success:
                raise ResponsesServiceError(response, payload)
        if payload.get("error") or payload.get("status") not in {"completed", "incomplete"}:
            raise ResponsesServiceError(response if response is not None else _stream_error_response(payload), payload)
        return ChatResult(generations=[ChatGeneration(message=responses_message(payload))])

    async def _post_stream_async(self, client, url, body, timeout, headers):
        from .runtime_audit import arecord_event, ensure_async_request_audit
        started = time.monotonic()
        total_timeout = _timeout_seconds(timeout, self.timeout_seconds)
        stream_deadline = total_timeout if self.enforce_stream_deadline else None
        stream_timeout = httpx.Timeout(timeout)
        if not self.enforce_stream_read_timeout:
            stream_timeout.read = None
        idle_timeout = stream_timeout.read
        events = []
        heartbeat_count = 0
        heartbeat_recorded = False
        first_event = None
        pending = None
        owned = client is None
        client = client or httpx.AsyncClient(timeout=stream_timeout, follow_redirects=False)
        ensure_async_request_audit(client)
        response = None
        call_id = None
        request_accepted = False
        stream_stage = "connecting"
        deadline_scope = asyncio.timeout(stream_deadline)
        try:
            check_request_control()
            async with deadline_scope:
                async with client.stream("POST", url, json=body, timeout=stream_timeout, headers=headers) as response:
                    stream_stage = "connected"
                    owner = response.request.extensions.get("rag_audit_owner")
                    call_id = owner[1] if isinstance(owner, tuple) and len(owner) == 2 else None
                    await arecord_event("model_http_connected", call_id=call_id,
                                 http_status=response.status_code,
                                 request_id=response.headers.get("x-request-id"), stream=True,
                                 stream_deadline_seconds=stream_deadline, idle_timeout_seconds=idle_timeout)
                    if not response.is_success:
                        stream_stage = "reading_error_response"
                        payload = await _read_error_payload_async(response)
                        raise ResponsesServiceError(response, payload)

                    async def _flush_pending():
                        nonlocal pending
                        if pending:
                            await arecord_event("model_http_stream", call_id=call_id, **pending)
                            pending = None

                    async def _record_row(row, *, first=False):
                        await arecord_event("model_http_stream", call_id=call_id,
                                     **({"first_event": True} if first else {}), **row)

                    async def _finish(payload, *, timed_out=False):
                        await arecord_event("model_http_response", call_id=call_id,
                                     http_status=response.status_code,
                                     request_id=response.headers.get("x-request-id"), body=payload, stream=True,
                                     request_accepted=request_accepted,
                                     heartbeat_count=heartbeat_count,
                                     elapsed_seconds=round(time.monotonic() - started, 3),
                                     **({"timed_out": True} if timed_out else {}))
                        return {**payload, "request_accepted": request_accepted}

                    stream_stage = "receiving_events"
                    async for event_type, data in iter_responses_sse(response):
                        if event_type == "comment" or (isinstance(data, dict) and data.get("type") == "sse_comment"):
                            heartbeat_count += 1
                            continue
                        # HTTP success alone (including a 200 error stream) does
                        # not prove that the service accepted the model input.
                        kind = data.get("type") or event_type
                        event_response = data.get("response")
                        event_error = bool(data.get("error")) or (isinstance(event_response, dict) and
                            bool(event_response.get("error") or event_response.get("status") == "failed"))
                        if not event_error and isinstance(kind, str) and (kind in {
                                "response.created", "response.in_progress", "response.completed",
                                "response.incomplete"} or kind.startswith((
                                "response.output_", "response.content_part.",
                                "response.function_call_arguments.", "response.reasoning_",
                                "response.web_search_call.", "response.web_extractor_call.",
                                "response.code_interpreter_call."))):
                            request_accepted = True
                        events.append((event_type, data))
                        row = _stream_event_audit(event_type, data)
                        if first_event is None:
                            first_event = row
                            if _is_delta_event(data):
                                pending = _merge_delta_audit(None, {**row, "first_event": True})
                                continue
                            await _record_row(row, first=True)
                        elif _is_delta_event(data):
                            if pending and pending.get("type") != row.get("type"):
                                await _flush_pending()
                            pending = _merge_delta_audit(pending, row)
                            if pending.get("delta_count", 1) >= 20 or pending.get("delta_chars", 0) >= 400:
                                await _flush_pending()
                            continue
                        else:
                            await _flush_pending()
                            await _record_row(row)
                        if (data.get("type") or event_type) in {
                                "response.completed", "response.incomplete", "response.failed", "error"}:
                            # A Responses terminal event completes this call even
                            # when the HTTP server leaves its SSE connection open.
                            break
                    stream_stage = "assembling_response"
                    await _flush_pending()
                    if heartbeat_count:
                        await arecord_event("model_http_stream", call_id=call_id, type="sse_comment",
                                     heartbeat_count=heartbeat_count)
                        heartbeat_recorded = True
                    payload, _, error = assemble_streamed_response(events)
                    if error:
                        raise ResponsesServiceError(response, error)
                    if not isinstance(payload, dict):
                        failure = RuntimeError("Responses流式结束但没有completed载荷。")
                        failure.request_accepted = request_accepted
                        raise failure
                    stream_stage = "recording_response"
                    return await _finish(payload)
        except (TimeoutError, httpx.TimeoutException) as exc:
            timeout_kind = ("stream_deadline" if deadline_scope.expired() else
                            "read_idle" if isinstance(exc, httpx.ReadTimeout) else
                            "connect" if isinstance(exc, httpx.ConnectTimeout) else
                            "write" if isinstance(exc, httpx.WriteTimeout) else
                            "pool" if isinstance(exc, httpx.PoolTimeout) else "transport_timeout")
            audit_rows = []
            if pending:
                audit_rows.append(("model_http_stream", {"call_id": call_id, **pending}))
                pending = None
            if heartbeat_count and not heartbeat_recorded:
                audit_rows.append(("model_http_stream", {"call_id": call_id, "type": "sse_comment",
                                                        "heartbeat_count": heartbeat_count}))
            audit_rows.append(("model_http_stream_error", {
                "call_id": call_id, "error_type": "TimeoutException", "heartbeat_count": heartbeat_count,
                "timeout_kind": timeout_kind, "stream_deadline_seconds": stream_deadline,
                "idle_timeout_seconds": idle_timeout, "request_accepted": request_accepted,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }))
            audit_errors = []
            for kind, fields in audit_rows:
                if kind == "model_http_stream_error" and audit_errors:
                    fields = {**fields, "audit_error_types": list(audit_errors)}
                try:
                    await arecord_event(kind, **fields)
                except Exception as audit_exc:
                    # An unavailable audit sink must not mask the timeout.
                    audit_errors.append(type(audit_exc).__name__)
            payload, _, error = assemble_streamed_response(events)
            if error and response is not None:
                raise ResponsesServiceError(response, error) from None
            if isinstance(payload, dict) and response is not None:
                payload = dict(payload)
                if payload.get("status") != "completed":
                    payload["status"] = "incomplete"
                    payload["incomplete_reason"] = "stream_timeout"
                if audit_errors:
                    payload["stream_audit_error_types"] = list(audit_errors)
                try:
                    return await _finish(payload, timed_out=True)
                except Exception as audit_exc:
                    # Retain the original timeout if recording the recovered
                    # response fails, with the audit failure type available to
                    # the caller without exposing provider-sensitive messages.
                    audit_errors.append(type(audit_exc).__name__)
            if isinstance(exc, TimeoutError):
                request = response.request if response is not None else httpx.Request("POST", url)
                message = ("Responses stream exceeded wall-clock timeout" if deadline_scope.expired()
                           else "Responses stream transport timed out")
                failure = httpx.ReadTimeout(message, request=request)
                failure.request_accepted = request_accepted
                if audit_errors:
                    failure.audit_error_types = audit_errors
                raise failure from exc
            exc.request_accepted = request_accepted
            if audit_errors:
                exc.audit_error_types = audit_errors
            raise
        except Exception as exc:
            # Retain bounded previews and safe classification for disconnects.
            # Never log the exception message: it may contain provider data.
            exc.request_accepted = request_accepted
            rows = []
            if pending:
                rows.append(("model_http_stream", {"call_id": call_id, **pending}))
            if heartbeat_count and not heartbeat_recorded:
                rows.append(("model_http_stream", {"call_id": call_id, "type": "sse_comment",
                                                    "heartbeat_count": heartbeat_count}))
            rows.append(("model_http_stream_error", {
                "call_id": call_id, "error_type": type(exc).__name__, "stage": stream_stage,
                "heartbeat_count": heartbeat_count, "request_accepted": request_accepted,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }))
            for kind, fields in rows:
                try:
                    await arecord_event(kind, **fields)
                except Exception:
                    # Audit failures must not replace the original transport error.
                    pass
            raise
        finally:
            if owned:
                await client.aclose()


def _read_error_payload(response):
    try:
        text = response.read().decode("utf-8")
    except Exception:
        return {}
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {"unparsed_text": text}


async def _read_error_payload_async(response):
    try:
        text = (await response.aread()).decode("utf-8")
    except Exception:
        return {}
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {"unparsed_text": text}


def _stream_error_response(payload):
    class _Response:
        status_code = None
    return _Response()
