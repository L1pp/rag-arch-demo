"""Single terminal-tool delivery for tool-using chat models.

The adapter keeps ordinary evidence tools available during exploration and
converts exactly one ``submit_answer`` call, or one complete visible JSON
response at ``finish=stop``, into the JSON text expected by the existing agent
result validators. It never repairs a payload or executes a terminal call as a
normal tool.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, SystemMessage

from .json_output import parse_single_json_object

TERMINAL_TOOL_NAME = "submit_answer"
TERMINAL_OUTPUT_POLICY_VARIANT = "terminal_call_or_single_visible_json_v1"
TERMINAL_PROMPT_VARIANT = "terminal-output-v2"
_PROVIDER_INVALID_TOOL_CALL_LIMIT = 4
_PROVIDER_INVALID_TOOL_TEXT_LIMIT = 512
TERMINAL_SYSTEM_SUFFIX = (
    "\n\n传输设置（仅最终交付）：最终结果应优先调用submit_answer工具提交；兼容性兜底允许服务在finish=stop、"
    "没有任何工具调用且正文为完整单个JSON对象时直接交付该正文。可见正文路径不计为submit_answer，"
    "它只是同一响应的答案传输封装，不属于取证工具、不新增模型轮次，也不执行取证。该设置覆盖上文关于"
    "最终只输出JSON文本、不使用额外工具包装答案及最后一轮禁止调用工具的收尾格式要求；普通轮次只能按需"
    "调用本轮已绑定的取证工具。"
)
TERMINAL_ONLY_SYSTEM_SUFFIX = (
    "\n\n本轮是最终收尾：当前只允许提交submit_answer最终对象，或在finish=stop且无工具调用时直接返回一个"
    "完整JSON正文；不能继续调用取证工具。"
)


def _visible_text_raw(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict)
            and block.get("type") in {"text", "output_text"}
            and isinstance(block.get("text"), str)
        )
    return ""


def _visible_text(message: AIMessage) -> str:
    return _visible_text_raw(message).strip()


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            value = function.get("name")
            return value if isinstance(value, str) else None
        value = tool.get("name")
        return value if isinstance(value, str) else None
    value = getattr(tool, "name", None)
    return value if isinstance(value, str) else None


def _raw_parts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    parts = []
    for call in value:
        if not isinstance(call, dict):
            parts.append({"name": None, "id": None, "arguments": None})
            continue
        function = call.get("function")
        if isinstance(function, dict):
            name, arguments = function.get("name"), function.get("arguments")
        else:
            name = call.get("name")
            arguments = call.get("arguments", call.get("args"))
        parts.append({"name": name if isinstance(name, str) else None,
                      "id": call.get("id"), "arguments": deepcopy(arguments)})
    return parts


def _native_parts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    parts = []
    for call in value:
        if not isinstance(call, dict):
            parts.append({"name": None, "id": None, "arguments": None})
            continue
        arguments = call.get("args")
        parts.append({"name": call.get("name") if isinstance(call.get("name"), str) else None,
                      "id": call.get("id"), "arguments": deepcopy(arguments)})
    return parts


def _ordinary_invalid_batch(response: AIMessage, raw_parts: list[dict]) -> bool:
    """Accept provider-invalid ordinary calls without inventing a terminal call.

    LangChain keeps malformed ordinary calls in ``invalid_tool_calls``.  They
    are still useful to the Agent: it can return a tool error using the real
    call id and let the model retry.  Only a batch whose every call has a
    unique id/name and excludes ``submit_answer`` is eligible here.
    """
    if not response.invalid_tool_calls:
        return False
    seen = set()
    for call in [*(response.tool_calls or []), *(response.invalid_tool_calls or [])]:
        if not isinstance(call, dict):
            return False
        name = _tool_name(call)
        call_id = call.get("id", call.get("call_id"))
        if (not isinstance(name, str) or not name.strip() or name == TERMINAL_TOOL_NAME
                or not isinstance(call_id, str) or not call_id.strip() or call_id in seen):
            return False
        seen.add(call_id)
    if raw_parts:
        raw_seen = set()
        for part in raw_parts:
            name, call_id = part.get("name"), part.get("id")
            if (not isinstance(name, str) or not name.strip() or name == TERMINAL_TOOL_NAME
                    or not isinstance(call_id, str) or not call_id.strip() or call_id in raw_seen):
                return False
            raw_seen.add(call_id)
        if raw_seen != seen:
            return False
    return True


def _bounded_provider_invalid_tool_calls(value: Any) -> tuple[list[dict] | None, int]:
    """Keep a bounded, provider-owned diagnostic without exporting raw calls."""
    if not isinstance(value, list) or not value:
        return None, 0
    summary = []
    for raw in value[:_PROVIDER_INVALID_TOOL_CALL_LIMIT]:
        if not isinstance(raw, dict):
            summary.append({"entry_type": type(raw).__name__})
            continue
        item = {}
        for key in ("name", "id", "type"):
            current = raw.get(key)
            if isinstance(current, str):
                item[key] = current[:_PROVIDER_INVALID_TOOL_TEXT_LIMIT]
        for key in ("error", "args"):
            current = raw.get(key)
            if isinstance(current, str):
                item[key + "_preview"] = current[:_PROVIDER_INVALID_TOOL_TEXT_LIMIT]
                if len(current) > _PROVIDER_INVALID_TOOL_TEXT_LIMIT:
                    item[key + "_truncated"] = True
            elif current is not None:
                item[key + "_type"] = type(current).__name__
        summary.append(item)
    return summary, max(0, len(value) - _PROVIDER_INVALID_TOOL_CALL_LIMIT)


def _copy_additional(message: AIMessage) -> dict:
    extra = deepcopy(message.additional_kwargs or {})
    # This provider-only raw field is consumed by the adapter and is never
    # placed on a message that may be sent back in a later request.
    extra.pop("provider_raw_tool_calls", None)
    return extra


def _message_kwargs(message: AIMessage, additional_kwargs: dict) -> dict:
    return {
        "additional_kwargs": additional_kwargs,
        "response_metadata": deepcopy(message.response_metadata or {}),
        "usage_metadata": deepcopy(message.usage_metadata),
        "id": message.id,
        "name": message.name,
    }


class TerminalOutputChat:
    """Proxy a ChatOpenAI-compatible model with terminal-tool delivery."""

    def __init__(self, model, response_schema: dict):
        if not isinstance(response_schema, dict):
            raise ValueError("terminal_tool需要对象型response_schema。")
        self._model = model
        self._schema = deepcopy(response_schema)
        self._terminal_tool = {
            "type": "function",
            "function": {
                "name": TERMINAL_TOOL_NAME,
                "description": "提交最终回答对象；不得在此工具外输出最终回答。",
                "parameters": deepcopy(response_schema),
            },
        }

    def with_output_schema(self, response_schema: dict):
        """Reuse the same provider/settings with another internal output contract."""
        return TerminalOutputChat(self._model, response_schema)

    @property
    def output_delivery(self) -> dict:
        return {"method": "terminal_tool", "tool_name": TERMINAL_TOOL_NAME,
                "schema": deepcopy(self._schema),
                "acceptance_policy": TERMINAL_OUTPUT_POLICY_VARIANT,
                "prompt_variant": TERMINAL_PROMPT_VARIANT}

    @property
    def model_name(self):
        return self._model.model_name

    @property
    def extra_body(self):
        return self._model.extra_body

    @property
    def max_tokens(self):
        return self._model.max_tokens

    @property
    def thinking_budget(self):
        extra = self.extra_body
        return extra.get("thinking_budget") if isinstance(extra, dict) else None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def bind_tools(self, tools: Sequence[Any], **kwargs):
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"terminal_tool不支持在bind_tools时强制工具选择：{unsupported}")
        return _TerminalOutputBound(self, list(tools))

    def invoke(self, messages, **kwargs):
        return self._invoke(messages, [], kwargs)

    def _prepare_messages(self, messages, *, terminal_only: bool = False):
        prepared = list(messages)
        suffix = TERMINAL_SYSTEM_SUFFIX + (TERMINAL_ONLY_SYSTEM_SUFFIX if terminal_only else "")
        for index, message in enumerate(prepared):
            if not isinstance(message, SystemMessage):
                continue
            content = deepcopy(message.content)
            if isinstance(content, str):
                if TERMINAL_SYSTEM_SUFFIX not in content:
                    content += TERMINAL_SYSTEM_SUFFIX
                if terminal_only and TERMINAL_ONLY_SYSTEM_SUFFIX not in content:
                    content += TERMINAL_ONLY_SYSTEM_SUFFIX
            elif isinstance(content, list):
                content.append({"type": "text", "text": suffix})
            else:
                content = str(content) + suffix
            prepared[index] = SystemMessage(content=content,
                                             additional_kwargs=deepcopy(message.additional_kwargs or {}),
                                             response_metadata=deepcopy(message.response_metadata or {}),
                                             name=message.name, id=message.id)
            return prepared
        return [SystemMessage(content=suffix.strip())] + prepared

    def _invoke(self, messages, ordinary_tools: list[Any], invoke_kwargs: dict):
        options = dict(invoke_kwargs)
        caller_choice = options.pop("tool_choice", None)
        if (isinstance(caller_choice, dict) or caller_choice is True
                or (isinstance(caller_choice, str) and caller_choice in {"required", "any"})):
            raise ValueError("terminal_tool不支持required或强制工具选择；请使用tool_choice=none收尾。")
        terminal_only = caller_choice == "none"
        request_tools = [self._terminal_tool] if terminal_only else list(ordinary_tools) + [self._terminal_tool]
        options["tool_choice"] = "auto"
        if terminal_only and getattr(self._model, "api_protocol", None) == "responses":
            options["native_tools_enabled"] = False
        bound = self._model.bind_tools(request_tools)
        response = bound.invoke(self._prepare_messages(messages, terminal_only=terminal_only), **options)
        if not isinstance(response, AIMessage):
            raise TypeError("聊天模型响应不是AIMessage。")
        return self._adapt(response, terminal_only=terminal_only)

    def _delivery(self, response: AIMessage, *, status: str, raw_calls, raw_arguments,
                  decoded_arguments, code: str | None, acceptance_path: str | None = None,
                  terminal_submission_count: int | None = None) -> dict:
        provider_invalid_tool_calls, provider_invalid_tool_calls_truncated = (
            _bounded_provider_invalid_tool_calls(response.invalid_tool_calls)
        )
        delivery = {
            "method": "terminal_tool",
            "status": status,
            "provider_finish_reason": (response.response_metadata or {}).get("finish_reason"),
            "provider_visible_text": _visible_text_raw(response),
            "raw_tool_calls": deepcopy(raw_calls),
            "raw_arguments": deepcopy(raw_arguments),
            "decoded_arguments": deepcopy(decoded_arguments),
            "code": code,
        }
        if provider_invalid_tool_calls is not None:
            delivery["provider_invalid_tool_calls"] = provider_invalid_tool_calls
            if provider_invalid_tool_calls_truncated:
                delivery["provider_invalid_tool_calls_truncated_count"] = (
                    provider_invalid_tool_calls_truncated
                )
        if acceptance_path is not None:
            delivery["acceptance_path"] = acceptance_path
        if terminal_submission_count is not None:
            delivery["terminal_submission_count"] = terminal_submission_count
        return delivery

    def _with_delivery(self, response: AIMessage, delivery: dict, *, content=None,
                       tool_calls=None, invalid_tool_calls=None) -> AIMessage:
        additional = _copy_additional(response)
        additional["structured_output_delivery"] = delivery
        kwargs = _message_kwargs(response, additional)
        return AIMessage(content=deepcopy(response.content if content is None else content),
                         tool_calls=deepcopy(response.tool_calls if tool_calls is None else tool_calls),
                         invalid_tool_calls=deepcopy(response.invalid_tool_calls if invalid_tool_calls is None else invalid_tool_calls),
                         **kwargs)

    def _invalid(self, response: AIMessage, delivery: dict) -> AIMessage:
        invalid = [{"name": TERMINAL_TOOL_NAME, "args": "", "id": "invalid-terminal",
                    "error": delivery["code"] or "invalid_tool_call", "type": "invalid_tool_call"}]
        return self._with_delivery(response, delivery, content=_visible_text(response),
                                   tool_calls=[], invalid_tool_calls=invalid)

    def _adapt(self, response: AIMessage, *, terminal_only: bool) -> AIMessage:
        extra = response.additional_kwargs or {}
        raw_present = "provider_raw_tool_calls" in extra
        raw_calls = deepcopy(extra.get("provider_raw_tool_calls")) if raw_present else None
        native_parts = _native_parts(response.tool_calls or [])
        # Some compatible Chat Completions transports emit a null raw
        # ``tool_calls`` field for an ordinary text response. Treat that as
        # absence only when the native response also has no call or invalid
        # call; malformed/invalid tool responses still fail closed below.
        if raw_present and raw_calls is None and not response.tool_calls and not response.invalid_tool_calls:
            raw_present = False
        raw_parts = _raw_parts(raw_calls) if raw_present else []
        parts = raw_parts if raw_parts else native_parts
        raw_arguments = None
        if raw_present and raw_parts:
            values = [deepcopy(part.get("arguments")) for part in raw_parts]
            raw_arguments = values[0] if len(values) == 1 else values
        finish_reason = (response.response_metadata or {}).get("finish_reason")
        if raw_present and not isinstance(raw_calls, list):
            delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="invalid_tool_call")
            return self._invalid(response, delivery)
        if finish_reason == "length":
            delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="model_output_limit")
            return self._invalid(response, delivery)
        if response.invalid_tool_calls:
            if (not terminal_only
                    and _ordinary_invalid_batch(response, raw_parts)):
                delivery = self._delivery(response, status="passthrough", raw_calls=raw_calls,
                                          raw_arguments=raw_arguments, decoded_arguments=None,
                                          code="ordinary_tool_calls")
                return self._with_delivery(response, delivery)
            delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="invalid_tool_call")
            return self._invalid(response, delivery)
        if raw_present and raw_calls and len(native_parts) != len(raw_parts):
            delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="invalid_tool_call")
            return self._invalid(response, delivery)
        malformed = any(part.get("name") is None for part in parts)
        terminals = [part for part in parts if part.get("name") == TERMINAL_TOOL_NAME]
        ordinary = [part for part in parts if part.get("name") != TERMINAL_TOOL_NAME]
        if len(terminals) == 1 and not ordinary and len(parts) == 1:
            part = terminals[0]
            arguments = part.get("arguments")
            try:
                if raw_present and raw_parts:
                    if not isinstance(arguments, str):
                        raise TypeError("terminal arguments须为JSON文本。")
                    decoded = parse_single_json_object(arguments)
                elif isinstance(arguments, dict):
                    decoded = deepcopy(arguments)
                else:
                    raise TypeError("terminal arguments缺失。")
                content = json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError, json.JSONDecodeError):
                delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                          raw_arguments=raw_arguments, decoded_arguments=None,
                                          code="invalid_tool_call")
                return self._invalid(response, delivery)
            delivery = self._delivery(response, status="accepted", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=decoded,
                                      code="ok", acceptance_path="terminal_tool_arguments",
                                      terminal_submission_count=1)
            return self._with_delivery(response, delivery, content=content,
                                       tool_calls=[], invalid_tool_calls=[])
        if finish_reason == "stop" and not parts:
            visible_text = _visible_text_raw(response)
            try:
                # Syntax-only gate. The receiving generation/plan parser keeps
                # ownership of the response schema and field validation.
                parse_single_json_object(visible_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                delivery = self._delivery(
                    response, status="accepted_text", raw_calls=raw_calls,
                    raw_arguments=None, decoded_arguments=None, code="ok",
                    acceptance_path="provider_visible_json", terminal_submission_count=0)
                return self._with_delivery(response, delivery, content=visible_text,
                                           tool_calls=[], invalid_tool_calls=[])
            delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="invalid_visible_json")
            return self._invalid(response, delivery)
        if parts and not terminals and not malformed and not terminal_only:
            delivery = self._delivery(response, status="passthrough", raw_calls=raw_calls,
                                      raw_arguments=raw_arguments, decoded_arguments=None,
                                      code="ordinary_tool_calls")
            return self._with_delivery(response, delivery)
        delivery = self._delivery(response, status="invalid", raw_calls=raw_calls,
                                  raw_arguments=raw_arguments, decoded_arguments=None,
                                  code="invalid_tool_call")
        return self._invalid(response, delivery)


class _TerminalOutputBound:
    def __init__(self, owner: TerminalOutputChat, tools: list[Any]):
        self._owner = owner
        self._tools = tools

    @property
    def output_delivery(self):
        return self._owner.output_delivery

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def invoke(self, messages, **kwargs):
        return self._owner._invoke(messages, self._tools, kwargs)
