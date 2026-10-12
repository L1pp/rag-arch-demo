"""Trusted final-delivery instructions for bounded tool agents.

This module does not invoke a model or execute a tool.  The caller decides
when to use the returned phase and binds the terminal output adapter itself.
It deliberately keeps the final phase fail-closed: ordinary evidence tools
are disabled, and an invalid final response is recorded as a technical
failure rather than repaired into an answer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from langchain_core.messages import SystemMessage


TERMINAL_COMPLETION_POLICY = "terminal-completion-v2"
TERMINAL_COMPLETION_MARKER = "[terminal-completion-v2]"


@dataclass(frozen=True)
class CompletionWindow:
    """The caller's remaining budget before a model invocation."""

    phase: str
    remaining_model_calls: int
    remaining_tool_calls: int
    ordinary_tools_allowed: bool
    terminal_tool_only: bool
    hidden_retry_allowed: bool = False
    synthetic_answer_allowed: bool = False


def completion_window(*, model_calls: int, max_model_calls: int,
                      tool_calls: int, max_tool_calls: int,
                      evidence_budget_exhausted: bool = False) -> CompletionWindow:
    """Classify the next invocation without consuming a budget slot.

    ``pre_final`` is an early warning: the current call may still collect
    evidence, but the following call must be reserved for delivery.  ``final``
    disables ordinary tools and is the only phase that should receive the
    terminal-only instruction.
    """
    if any(type(value) is not int for value in
           (model_calls, max_model_calls, tool_calls, max_tool_calls)):
        raise ValueError("模型和工具预算必须是整数。")
    if model_calls < 0 or max_model_calls < 1 or tool_calls < 0 or max_tool_calls < 1:
        raise ValueError("模型和工具预算必须为正数，已用量不能为负。")
    remaining_models = max(0, max_model_calls - model_calls)
    remaining_tools = max(0, max_tool_calls - tool_calls)
    if evidence_budget_exhausted or remaining_tools == 0 or remaining_models <= 1:
        phase = "final"
    elif remaining_models == 2:
        phase = "pre_final"
    else:
        phase = "explore"
    final = phase == "final"
    return CompletionWindow(
        phase=phase,
        remaining_model_calls=remaining_models,
        remaining_tool_calls=remaining_tools,
        ordinary_tools_allowed=not final,
        terminal_tool_only=final,
    )


def build_completion_instruction(window: CompletionWindow, *, reason: str = "", evidence_ids=None) -> str:
    """Build a trusted instruction for the caller's next model request."""
    if not isinstance(window, CompletionWindow):
        raise TypeError("window必须是CompletionWindow。")
    if evidence_ids is not None and (not isinstance(evidence_ids, (list, tuple))
            or any(not isinstance(eid, str) for eid in evidence_ids)):
        raise ValueError("evidence_ids必须是字符串列表或None。")
    ids_line = ("本轮已取得且完整的可引用ID（仍需符合所问对象与条件）："
        + json.dumps(list(dict.fromkeys(evidence_ids)), ensure_ascii=False) + "\n"
        if evidence_ids is not None else "")
    reason_line = f"触发原因：{reason.strip()}。\n" if isinstance(reason, str) and reason.strip() else ""
    budget_line = (f"剩余模型调用{window.remaining_model_calls}次，剩余工具尝试"
                   f"{window.remaining_tool_calls}次。\n")
    citation_rule = ("引用字段只能列本轮实际取得且完整的相关证据ID。未读图像/未校准表格的ID可以帮助定位资料缺口，"
        "但不能填入evidence_ids或required_evidence_ids来充当完整正文证据。若只说明无法判断且没有可引用的完整依据，"
        "按当前schema保留不足结论，引用列表可为空；不要为了凑引用补写未完整/未取得的ID。"
        "资料未读不等于原件不存在。\n" + ids_line) if evidence_ids is not None else ""
    if window.phase == "pre_final":
        return (
            f"{TERMINAL_COMPLETION_MARKER}-preflight\n"
            "这是最终交付前的预算提醒。当前调用仍可按需取得必要证据；"
            "下一次调用必须完成最终交付，不要把预算耗尽在新的探索上。\n"
            + budget_line + reason_line + citation_rule
            + "已取得证据和条件优先于继续扩展检索；不要把参考答案或未取得内容当作依据。"
        )
    if window.phase != "final":
        return (TERMINAL_COMPLETION_MARKER + "-citations\n" + citation_rule) if citation_rule else ""
    return (
        f"{TERMINAL_COMPLETION_MARKER}\n"
        "这是本轮最终交付轮。现在关闭普通取证工具：不得调用search、get_clause、"
        "check_standard_status或任何其他普通工具，也不得请求下一轮。\n"
        "若当前提供submit_answer则用它提交；否则直接输出符合当前schema的一个完整JSON对象。\n"
        "证据不足时按当前schema如实给出无法判断或证据不足，不添加schema之外的状态字段；"
        "引用仅选上述已取得且完整的相关ID，不从旧历史或记忆补写ID。"
        "不得编造事实、引用、数值或用伪造答案填补缺口。这个收尾不允许隐藏重试、"
        "自动搜索、模型切换或合成答案。\n"
        + budget_line + reason_line + citation_rule
    )


def append_completion_instruction(messages, window: CompletionWindow, *, reason: str = "", evidence_ids=None) -> list:
    """Return messages with one idempotent trusted completion instruction.

    The original list and message objects are not mutated.  A caller may call
    this when composing a final request after a pre-final warning; the prior
    marker is replaced rather than accumulated.
    """
    if not isinstance(messages, list):
        raise TypeError("messages必须是列表。")
    instruction = build_completion_instruction(window, reason=reason, evidence_ids=evidence_ids)
    output = []
    for message in deepcopy(messages):
        content = getattr(message, "content", None)
        if isinstance(message, SystemMessage) and isinstance(content, str) and content.startswith(TERMINAL_COMPLETION_MARKER):
            continue
        output.append(message)
    if instruction:
        output.append(SystemMessage(content=instruction))
    return output


def terminal_completion_policy() -> dict:
    """Return explicit integration policy; no hidden retry or answer synthesis."""
    return {
        "policy": TERMINAL_COMPLETION_POLICY,
        "ordinary_tools_allowed": False,
        "terminal_tool_only": True,
        "hidden_retry_allowed": False,
        "synthetic_answer_allowed": False,
        "invalid_delivery": "technical_failure_preserve_raw_diagnostic",
    }


__all__ = [
    "CompletionWindow",
    "TERMINAL_COMPLETION_MARKER",
    "TERMINAL_COMPLETION_POLICY",
    "append_completion_instruction",
    "build_completion_instruction",
    "completion_window",
    "terminal_completion_policy",
]
