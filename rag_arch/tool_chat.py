"""Minimal non-streaming reasoning transport for configured compatible models.

ChatOpenAI 1.6.0 intentionally ignores provider-specific reasoning_content.
DeepSeek and Qwen3.8 require that field in assistant-message round trips when
preserving thinking. This adapter preserves it without displaying it, changing the
model, enabling/disabling thinking, or installing another provider package.
"""
from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

from langchain_core.messages import AIMessage
from .web_search import WebSearchChat, web_search_extra_body, select_chat_protocol, create_responses_chat

from .config import resolve_chat_budgets
from .runtime_audit import attach_chat_audit
from .model_input_view import MODEL_INPUT_VIEW_VERSION, model_wire_messages


JSON_SCHEMA_NAME = "reference_answer"


class ReasoningContentChat(WebSearchChat):
    """Chat Completions only; stream() uses the non-streaming invoke path."""

    preserves_reasoning_content: ClassVar[bool] = True
    model_input_view_version: ClassVar[str] = MODEL_INPUT_VIEW_VERSION
    disable_streaming: bool = True

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for choice, generation in zip(raw["choices"], result.generations):
            generation.message.response_metadata["model_input_view_version"] = self.model_input_view_version
            original = choice.get("message", {})
            if "reasoning_content" in original:
                value = original["reasoning_content"]
                if value is not None and not isinstance(value, str):
                    raise ValueError("reasoning_content须为字符串或null；未转换或伪造。")
                generation.message.additional_kwargs["reasoning_content"] = value
            if "tool_calls" in original:
                # Keep the provider's exact arguments before LangChain decodes
                # them; terminal_output consumes this without reserializing it.
                generation.message.additional_kwargs["provider_raw_tool_calls"] = deepcopy(original["tool_calls"])
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "messages" not in payload:
            raise ValueError("当前推理消息适配仅支持Chat Completions接口。")
        if len(payload["messages"]) != len(messages):
            raise ValueError("消息数量发生变化，不能安全对应reasoning_content。")
        for message, outgoing in zip(messages, payload["messages"]):
            if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
                outgoing["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        payload["messages"] = model_wire_messages(payload["messages"])
        return payload


class Qwen38Chat(ReasoningContentChat):
    """Preserve Qwen3.8 reasoning; retain max_completion_tokens including thinking."""


class DeepSeekToolChat(ReasoningContentChat):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # ChatOpenAI renames this to an OpenAI-only field; DeepSeek documents
        # max_tokens. Preserve the supplied cap rather than changing its value.
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


def create_tool_chat_model(settings, *, max_tokens: int | None = None,
                           thinking_budget: int | None = None, http_client=None,
                           http_async_client=None,
                           json_output: bool = False, response_schema: dict | None = None,
                           output_method: str = "json", native_web_search: bool = False,
                           stream_responses: bool = False):
    """Construct only; credentials/model/URL come exclusively from settings.

    ``max_tokens`` is an optional explicit per-call output budget; when omitted,
    the configured ``settings.chat_max_tokens`` is used so normal and tool
    chat share one budget. ``thinking_budget`` is optional and otherwise uses
    ``settings.chat_thinking_budget``. It is sent as provider-specific
    ``extra_body`` metadata; no reasoning_effort, temperature, or beta endpoint
    is selected here. json_output requests the provider's JSON-object response
    format, without repairing invalid responses or changing thinking behavior.
    ``response_schema`` selects the provider's strict JSON-Schema response
    format and takes precedence over ``json_output`` when supplied.  With
    ``output_method="terminal_tool"``, the schema is exposed as one
    ``submit_answer`` function and no response_format is sent.
    The caller owns client lifetime when injecting http_client.
    """
    if not settings.chat_ready:
        return None
    if type(json_output) is not bool:
        raise ValueError("json_output须为布尔值。")
    if type(stream_responses) is not bool:
        raise ValueError("stream_responses须为布尔值。")
    if output_method not in {"json", "terminal_tool", "text"}:
        raise ValueError("output_method仅支持json、terminal_tool或text。")
    if output_method == "text":
        json_output, response_schema = False, None
    if response_schema is not None and not isinstance(response_schema, dict):
        raise ValueError("response_schema须为对象或None。")
    if output_method == "terminal_tool" and response_schema is None:
        raise ValueError("terminal_tool必须提供response_schema。")
    resolved_max_tokens, resolved_thinking_budget = resolve_chat_budgets(
        settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
    if select_chat_protocol(settings) == "responses":
        model = create_responses_chat(settings, max_tokens=resolved_max_tokens,
            thinking_budget=resolved_thinking_budget, http_client=http_client,
            http_async_client=http_async_client,
            json_output=json_output and response_schema is None,
            stream_responses=stream_responses)
        if response_schema is not None:
            # Responses does not document response_format/json_schema. Reuse
            # the project's existing explicit terminal-function delivery.
            from .terminal_output import TerminalOutputChat
            return TerminalOutputChat(model, response_schema)
        return model
    kwargs = {"model": settings.chat_model, "api_key": settings.chat_api_key,
              "base_url": settings.chat_base_url.rstrip("/"),
              "timeout": settings.effective_chat_timeout_seconds,
              "max_retries": 0, "use_responses_api": False, "stream_usage": False,
              "streaming": False, "disable_streaming": True}
    if http_client is not None:
        kwargs["http_client"] = http_client
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    kwargs["max_tokens"] = resolved_max_tokens
    extra_body = web_search_extra_body(settings) if native_web_search else {}
    if resolved_thinking_budget is not None:
        extra_body["thinking_budget"] = resolved_thinking_budget
    if output_method == "json" and response_schema is not None:
        extra_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": JSON_SCHEMA_NAME,
                "strict": True,
                "schema": deepcopy(response_schema),
            },
        }
    elif output_method == "json" and json_output:
        # A top-level response_format makes ChatOpenAI use SDK auto-parsing,
        # which rejects ordinary non-strict tools before any network request.
        # extra_body merges into the same provider request JSON while keeping
        # the normal create path and our own final schema validation.
        extra_body["response_format"] = {"type": "json_object"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    name = settings.chat_model.lower()
    if name.startswith("deepseek"):
        cls = DeepSeekToolChat
    elif name.startswith(("qwen3.8-flash", "qwen3.8-max")):
        cls = Qwen38Chat
    elif output_method == "terminal_tool":
        # Generic compatible endpoints need the same raw tool-call capture for terminal delivery.
        # The ordinary JSON path remains the plain ChatOpenAI transport.
        cls = ReasoningContentChat
    else:
        cls = WebSearchChat
    model = attach_chat_audit(cls(**kwargs))
    if output_method == "terminal_tool":
        from .terminal_output import TerminalOutputChat
        return TerminalOutputChat(model, response_schema)
    return model


def create_async_tool_chat_model(settings, *, max_tokens: int | None = None,
                                 thinking_budget: int | None = None, http_client=None,
                                 http_async_client=None, json_output: bool = False,
                                 response_schema: dict | None = None, output_method: str = "json",
                                 native_web_search: bool = False, stream_responses: bool = False):
    """Construct a tool-capable model with an owned per-model async transport."""
    if not settings.chat_ready:
        return None
    if type(json_output) is not bool:
        raise ValueError("json_output须为布尔值。")
    if type(stream_responses) is not bool:
        raise ValueError("stream_responses须为布尔值。")
    if output_method not in {"json", "terminal_tool", "text"}:
        raise ValueError("output_method仅支持json、terminal_tool或text。")
    if response_schema is not None and not isinstance(response_schema, dict):
        raise ValueError("response_schema须为对象或None。")
    if output_method == "text":
        response_schema = None
    if output_method == "terminal_tool" and response_schema is None:
        raise ValueError("terminal_tool必须提供response_schema。")
    resolve_chat_budgets(settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
    select_chat_protocol(settings)
    owns_client = http_async_client is None
    if owns_client:
        from .providers import (_dispose_unattached_async_client,
                               _mark_owned_async_http_client,
                               _new_owned_async_http_client)
        http_async_client = _new_owned_async_http_client(settings.effective_chat_timeout_seconds)
    try:
        model = create_tool_chat_model(settings, max_tokens=max_tokens,
            thinking_budget=thinking_budget, http_client=http_client,
            http_async_client=http_async_client, json_output=json_output,
            response_schema=response_schema, output_method=output_method,
            native_web_search=native_web_search, stream_responses=stream_responses)
        if owns_client:
            _mark_owned_async_http_client(model, http_async_client)
    except BaseException:
        if owns_client:
            _dispose_unattached_async_client(http_async_client)
        raise
    return model
