"""Deterministic bounded context and exact source-range acquisition.

This estimator is deliberately approximate, not the Qwen tokenizer. The budget
is an input soft limit; output tokens are separately configured by the caller.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from .model_input_view import MODEL_INPUT_VIEW_VERSION, model_message_content
from .multimodal import content_for_token_estimate

CONTEXT_POLICY_VERSION = "plan-navigation-evidence-window-v4"
NO_COMPACTION_POLICY_VERSION = "plan-token-budget-no-compaction-v1"
ESTIMATOR_VERSION = "model-view-text-v3-visual-tiles-v1"
_EVICTION_HEADROOM_CAP = 8192


class ContextBudgetError(ValueError):
    pass


def estimate_tokens(text):
    ascii_chars = sum(ord(char) < 128 for char in text)
    return math.ceil(ascii_chars / 4 + (len(text) - ascii_chars) * 1.5)


def estimate_messages(messages, schema_tokens=0, *, project_metadata=False, image_token_estimator=None):
    total = schema_tokens
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.additional_kwargs.get("responses_output"), list):
            from .responses_chat import responses_input
            for item in responses_input([message]):
                item = deepcopy(item)
                visuals = 0
                if "content" in item:
                    item["content"], visuals = content_for_token_estimate(item["content"], image_token_estimator)
                total += 12 + visuals + estimate_tokens(json.dumps(item, ensure_ascii=False, default=str))
            continue
        content = (model_message_content(message.type, message.content)
                   if project_metadata else message.content)
        content, visual_tokens = content_for_token_estimate(content, image_token_estimator)
        payload = {"role": message.type, "content": content}
        if isinstance(message, AIMessage):
            payload["tool_calls"] = message.tool_calls
            if message.invalid_tool_calls:
                payload["invalid_tool_calls"] = message.invalid_tool_calls
            # Audit snapshots stay local. ChatOpenAI does not send delivery
            # traces/provider_raw_tool_calls back; counting those can multiply
            # a report's apparent size and evict the very report being repaired.
            forwarded = {key: value for key, value in message.additional_kwargs.items()
                         if key in {"reasoning_content", "function_call"}}
            if not (message.tool_calls or message.invalid_tool_calls) and "tool_calls" in message.additional_kwargs:
                forwarded["tool_calls"] = message.additional_kwargs["tool_calls"]
            if forwarded:
                payload["additional_kwargs"] = forwarded
        if isinstance(message, ToolMessage):
            payload["tool_call_id"] = message.tool_call_id
        total += 12 + visual_tokens + estimate_tokens(json.dumps(payload, ensure_ascii=False, default=str))
    return total


def merge_ranges(ranges, unit_id, start, end):
    intervals = [list(pair) for pair in ranges.get(unit_id, [])] + [[start, end]]
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    ranges[unit_id] = merged


def register_fragments(ranges, units):
    for unit in units:
        start = unit.get("text_start", 0)
        end = unit.get("text_end", start + len(unit["text"]))
        merge_ranges(ranges, unit["unit_id"], start, end)


def contains_span(ranges, unit_id, span):
    return any(left <= span["start"] and span["end"] <= right
               for left, right in ranges.get(unit_id, []))


def source_coverage(document, ranges):
    full, partial, missing = [], [], []
    for unit in document.units:
        uid = unit["unit_id"]
        if contains_span(ranges, uid, {"start": 0, "end": len(unit["text"])}):
            full.append(uid)
        elif uid in ranges:
            partial.append(uid)
        else:
            missing.append(uid)
    return {"provided_units": len(full), "total_units": len(document.units),
            "provided_unit_ids": full, "fully_provided_unit_ids": full,
            "partially_provided_units": len(partial), "partially_provided_unit_ids": partial,
            "not_provided_unit_ids": missing, "provided_char_ranges": deepcopy(ranges),
            "provided_text_chars": sum(right - left for pairs in ranges.values() for left, right in pairs),
            "all_readable_text_provided": len(full) == len(document.units),
            "unread_pages": list(document.unread_pages),
            "non_text_review": ("not_performed" if document.filename.lower().endswith((".pdf", ".docx", ".doc")) else "not_applicable"),
            "page_images_reviewed": [],
            "meaning": "按实际发送的原文字符范围记账，导航预览不计入。完整提供不证明逐句核查或全部问题已发现；旧正文可能已退出当前上下文，可回读。"}


def _tool_pointer(message):
    """A navigation pointer, never a semantic summary or evidence substitute."""
    result = {"context_compacted": True, "tool": message.name,
              "note": "旧工具正文已退出上下文；下列仅是定位索引，不是原文或证据。需要判断时用read_plan/get_clause/search重新读取。"}
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return result
    if payload.get("context_compacted"):
        return payload
    status = payload.get("status_info")
    if isinstance(status, dict):
        result["status_lookup"] = {key: status[key] for key in ("std_no", "clause_no") if key in status}
    result["plan_ranges"] = [{key: unit[key] for key in (
        "ordinal", "unit_id", "text_start", "text_end", "complete_unit", "unit_text_chars", "locator") if key in unit}
        for unit in (
            (getattr(message, "additional_kwargs", None) or {}).get("plan_source_units")
            or payload.get("units", [])
        ) if isinstance(unit, dict)]
    result["evidence_ids"] = [unit.get("evidence_id") for unit in payload.get("evidence", [])
        if isinstance(unit, dict) and unit.get("evidence_id")]
    result["evidence_index"] = [{key: deepcopy(unit[key]) for key in (
        "evidence_id", "kind", "std_no", "clause_no", "table_id", "complete", "text_truncated",
        "content_status", "content_readable", "review_warning_code", "linked_table_ids", "owner_clause_id") if key in unit}
        for unit in payload.get("evidence", []) if isinstance(unit, dict) and unit.get("evidence_id")]
    for key in ("ok", "status", "found", "error", "missing", "context_complete", "complete_context_ids",
                "service_error", "available_modes", "mode_disabled_for_run", "disabled_service"):
        if key in payload:
            result[key] = deepcopy(payload[key])
    result["pointer_is_evidence"] = False
    return result


def prepare_messages(messages, *, budget_tokens=48000, schema_tokens=3000, project_metadata=False,
                     allow_compaction=True, image_token_estimator=None):
    """Enforce the estimated input budget; optionally run legacy compaction.

    The normal Agent passes allow_compaction=False: input is returned unchanged
    or rejected, without rewriting current-turn messages. Explicit legacy
    experiments may shrink old tool bodies and evict complete exchanges.
    """
    if not allow_compaction:
        size = estimate_messages(messages, schema_tokens, project_metadata=project_metadata,
                                 image_token_estimator=image_token_estimator)
        info = {"policy": NO_COMPACTION_POLICY_VERSION, "estimator": ESTIMATOR_VERSION,
                "model_input_view": MODEL_INPUT_VIEW_VERSION if project_metadata else "unprojected",
                "input_budget_tokens": budget_tokens, "schema_estimated_tokens": schema_tokens,
                "estimated_tokens_before": size, "estimated_tokens_after": size,
                "adjustments": [], "within_estimated_budget": size <= budget_tokens,
                "remaining_budget_tokens": max(0, budget_tokens - size), "exact_model_tokenizer": False}
        if size > budget_tokens:
            error = ContextBudgetError("本轮任务和资料超过可用输入预算；自动上下文压缩已关闭，未删改正文或发送超限请求。")
            error.context_info = info
            raise error
        return list(messages), info
    marker_text = "上下文已按预算缩减，旧工具原文及部分旧回复可能不在本次输入中。定位索引不是证据，原文仍在本地，可按需回读。不得把退出上下文的内容当作当前可见全文；缺证时保留限制。"
    rows = [row for row in messages if not (isinstance(row, HumanMessage) and row.content == marker_text)]
    had_marker = len(rows) != len(messages)
    def cost(message):
        return estimate_messages([message], project_metadata=project_metadata,
                                 image_token_estimator=image_token_estimator)
    costs = [cost(message) for message in rows]
    current = before = schema_tokens + sum(costs)
    changes = []
    # A prior marker must be restored even without a new adjustment. Reserve
    # its cost too, including when the marker-free input already fits.
    marker_cost = cost(HumanMessage(content=marker_text))
    target = budget_tokens - marker_cost if had_marker or before > budget_tokens else budget_tokens
    # Protect the newest complete exchange so freshly acquired text reaches the
    # next model call. Older exchanges can leave the window as whole pairs.
    newest_exchange = None
    for index, message in enumerate(rows):
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            end = index + 1
            expected = {call.get("id") for call in [*message.tool_calls, *message.invalid_tool_calls]}
            found = set()
            while end < len(rows) and isinstance(rows[end], ToolMessage):
                found.add(rows[end].tool_call_id)
                end += 1
            if expected == found:
                newest_exchange = (index, end)
    protected_ids = {id(row) for row in rows[slice(*newest_exchange)]} if newest_exchange else set()
    for index, message in enumerate(rows):
        if current <= target:
            break
        if isinstance(message, ToolMessage) and id(message) not in protected_ids:
            pointer = json.dumps(_tool_pointer(message), ensure_ascii=False)
            replacement = ToolMessage(content=pointer, tool_call_id=message.tool_call_id, name=message.name)
            replacement_cost = cost(replacement)
            if replacement_cost < costs[index]:
                current += replacement_cost - costs[index]
                rows[index], costs[index] = replacement, replacement_cost
                changes.append({"action": "replace_tool_body_with_pointer", "tool_call_id": message.tool_call_id})
    # Only request headroom when rewriting old tool bodies was insufficient.
    # It is best effort: required input/newest evidence may use the full budget.
    needs_eviction = current > target
    headroom = min(_EVICTION_HEADROOM_CAP, budget_tokens // 6) if needs_eviction else 0
    eviction_target = target - headroom
    index = 0
    while needs_eviction and current > eviction_target and index < len(rows) - 1:
        message = rows[index]
        if id(message) in protected_ids:
            index += 1
            continue
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            end = index + 1
            expected = {call.get("id") for call in [*message.tool_calls, *message.invalid_tool_calls]}
            found = set()
            while end < len(rows) and isinstance(rows[end], ToolMessage):
                found.add(rows[end].tool_call_id)
                end += 1
            if expected == found:
                # Extra eviction for headroom must not retire still-visible
                # source bodies merely to improve cache statistics.
                if current <= target:
                    try:
                        payloads = [json.loads(row.content) for row in rows[index + 1:end]]
                        pointers_only = all(payload.get("context_compacted") is True
                                            and payload.get("pointer_is_evidence") is False
                                            and not any(key in payload for key in ("units", "evidence", "runtime_status_evidence"))
                                            for payload in payloads)
                    except (ValueError, TypeError, AttributeError):
                        pointers_only = False
                    if not pointers_only:
                        index = end
                        continue
                current -= sum(costs[index:end])
                rows[index:end] = []
                costs[index:end] = []
                changes.append({"action": "evict_complete_tool_exchange", "tool_call_ids": sorted(expected)})
                continue
        if isinstance(message, HumanMessage) and str(message.content).startswith("此前对话"):
            if current <= target:
                index += 1
                continue
            current -= costs.pop(index)
            rows.pop(index)
            changes.append({"action": "evict_previous_turn_context"})
            continue
        if isinstance(message, AIMessage) and not (message.tool_calls or message.invalid_tool_calls):
            if current <= target:
                index += 1
                continue
            current -= costs.pop(index)
            rows.pop(index)
            changes.append({"action": "evict_old_assistant_text"})
            continue
        index += 1
    if changes or had_marker:
        rows.append(HumanMessage(content=marker_text))
        current += marker_cost
    after = current
    info = {"policy": CONTEXT_POLICY_VERSION, "estimator": ESTIMATOR_VERSION,
            "model_input_view": MODEL_INPUT_VIEW_VERSION if project_metadata else "unprojected",
            "input_budget_tokens": budget_tokens, "schema_estimated_tokens": schema_tokens,
            "estimated_tokens_before": before, "estimated_tokens_after": after,
            "adjustments": changes, "within_estimated_budget": after <= budget_tokens,
            "eviction_headroom_requested": headroom,
            "eviction_target_tokens": eviction_target if needs_eviction else None,
            "remaining_budget_tokens": max(0, budget_tokens - after),
            "exact_model_tokenizer": False}
    if after > budget_tokens:
        error = ContextBudgetError("当前任务、导航或候选报告超过上下文软预算，请缩小本轮范围或提高预算。")
        error.context_info = info
        raise error
    return rows, info
