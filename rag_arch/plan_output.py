"""Local, loss-preserving parsing of plan answers and review reports.

Only the existing single-object JSON syntax is accepted.  Valid fields and
complete findings are retained without changing the model's values; malformed
parts are returned separately instead of asking a model to rewrite the report.
Format diagnostics are not professional findings or answer limitations.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .json_output import parse_single_json_object


# Keep these checks aligned with the original _parse_final_report contract.
# Importing the Agent here would create a cycle when it uses this parser.
_FINDING_CATEGORIES = {"standard_comparison", "document_consistency", "source_gap"}
_FINDING_VERDICTS = {"符合", "不符合", "无法判断", "需要补充资料"}
_FINDING_FIELDS = (
    "title", "category", "verdict", "description", "plan_evidence",
    "standard_evidence_ids", "conditions", "recommendation",
)
_STRING_LIST_FIELDS = ("evidence_ids", "limitations", "reviewed_topics")
_OUTPUT_FIELDS = ("answer", "findings", *_STRING_LIST_FIELDS, "plan_evidence", "context_facts")


def natural_plan_output(text: str, evidence_ids=()) -> dict[str, Any]:
    """Preserve the report; only link explicit known IDs, never infer verdicts."""
    output = _empty_output()
    output["answer"] = text if isinstance(text, str) else ""
    mentioned = set(re.findall(r"\[([^\[\]\r\n]+)\]", output["answer"]))
    selected = []
    for eid in evidence_ids:
        explicit = eid in mentioned
        record = evidence_ids.get(eid, {}) if isinstance(evidence_ids, dict) else {}
        standard, clause = record.get("std_no"), record.get("clause_no")
        # Also recognize ordinary prose such as “JGJ 130-2011第6.3.2条”.
        # Require the explicit edition and adjacent clause, not two unrelated
        # mentions elsewhere in the report. Other forms remain plain text.
        if isinstance(standard, str) and standard.strip() and isinstance(clause, str) and re.fullmatch(r"\d+(?:\.\d+)+", clause):
            std_pattern = r"\s*".join(re.escape(c) for c in standard if not c.isspace())
            pattern = std_pattern + r"[\s*`]*(?:第[\s*`]*)?" + re.escape(clause) + r"(?![\d.])"
            explicit = explicit or bool(re.search(pattern, output["answer"], re.I))
        if explicit:
            selected.append(eid)
    output["evidence_ids"] = selected
    return {"output": output, "format_issues": [], "unparsed": []}


def _empty_output() -> dict[str, Any]:
    return {
        "answer": "", "findings": [], "evidence_ids": [], "limitations": [],
        "reviewed_topics": [], "plan_evidence": [],
    }


def _string_list_issues(value: object, path: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{path}必须是字符串列表。"]
    return [f"{path}[{index}]必须是字符串。"
            for index, item in enumerate(value) if not isinstance(item, str)]


def _quote_issues(value: object, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}必须是对象。"]
    issues = []
    if not isinstance(value.get("quote"), str):
        issues.append(f"{path}.quote必须是字符串。")
    for key in ("source_ref", "unit_id"):
        if key in value and not isinstance(value.get(key), str):
            issues.append(f"{path}.{key}必须是字符串。")
    return issues


def _finding_issues(value: object, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}必须是对象。"]
    issues = []
    missing = [key for key in _FINDING_FIELDS if key not in value]
    if missing:
        issues.append(f"{path}缺少必需字段：" + "、".join(missing) + "。")
    if "title" in value and (
        not isinstance(value["title"], str) or not value["title"].strip()
    ):
        issues.append(f"{path}.title必须是非空字符串。")
    for key, allowed in (("category", _FINDING_CATEGORIES), ("verdict", _FINDING_VERDICTS)):
        if key in value and (not isinstance(value[key], str) or value[key] not in allowed):
            issues.append(f"{path}.{key}无效。")
    for key in ("description", "recommendation"):
        if key in value and not isinstance(value[key], str):
            issues.append(f"{path}.{key}必须是字符串。")
    for key in ("standard_evidence_ids", "conditions"):
        if key in value:
            issues.extend(_string_list_issues(value[key], f"{path}.{key}"))
    if "plan_evidence" in value:
        refs = value["plan_evidence"]
        if not isinstance(refs, list):
            issues.append(f"{path}.plan_evidence必须是列表。")
        else:
            for index, ref in enumerate(refs):
                issues.extend(_quote_issues(ref, f"{path}.plan_evidence[{index}]"))
    return issues


def _context_fact_issues(value: object, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}必须是对象。"]
    issues = []
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        issues.append(f"{path}.summary必须是非空字符串。")
    refs = value.get("plan_evidence")
    if not isinstance(refs, list):
        issues.append(f"{path}.plan_evidence必须是列表。")
    else:
        for index, ref in enumerate(refs):
            issues.extend(_quote_issues(ref, f"{path}.plan_evidence[{index}]"))
    return issues


def parse_plan_output(text: str, *, mode: str = "review") -> dict[str, Any]:
    """Return validated output, format issues, and untouched rejected values.

    ``review`` requires the existing report fields.  ``question`` requires only
    answer, plan_evidence, evidence_ids, and limitations; supplied legacy report
    fields are still accepted and checked.  A missing required list is empty in
    the normalized output and is reported as a format issue.  No finding is
    synthesized from the answer, an empty list, or a malformed finding.

    Finding validation is atomic: one invalid field retains the entire original
    finding in ``unparsed``.  Top-level lists retain their valid individual items.
    Only findings equal in every field can be deduplicated; quotes, citation IDs,
    numbers, wording, and extra finding/reference fields are never rewritten.
    Optional ``context_facts`` retain valid cited summaries for later batches;
    they never become findings and are not added when the field is absent.
    """
    if mode not in ("review", "question"):
        raise ValueError("mode必须是review或question。")

    output = _empty_output()
    issues: list[str] = []
    unparsed: list[object] = []
    result = {"output": output, "format_issues": issues, "unparsed": unparsed}
    try:
        payload = parse_single_json_object(text)
    except (TypeError, ValueError) as exc:
        issues.append(f"无法解析为单个JSON对象：{exc}；完整原文已保留。")
        unparsed.append(text)
        return result

    required = {"answer", "evidence_ids", "limitations"}
    required.update(("findings", "reviewed_topics") if mode == "review" else ("plan_evidence",))
    for key in _OUTPUT_FIELDS:
        if key not in payload and key in required:
            issues.append(f"缺少必需字段：{key}。")

    if "answer" in payload:
        if isinstance(payload["answer"], str):
            output["answer"] = payload["answer"]
        else:
            issues.append("answer必须是字符串。")
            unparsed.append(payload["answer"])

    for key in _STRING_LIST_FIELDS:
        if key not in payload:
            continue
        values = payload[key]
        if not isinstance(values, list):
            issues.append(f"{key}必须是字符串列表。")
            unparsed.append(values)
            continue
        for index, value in enumerate(values):
            if isinstance(value, str):
                output[key].append(value)
            else:
                issues.append(f"{key}[{index}]必须是字符串。")
                unparsed.append(value)

    if "plan_evidence" in payload:
        refs = payload["plan_evidence"]
        if not isinstance(refs, list):
            issues.append("plan_evidence必须是列表。")
            unparsed.append(refs)
        else:
            for index, ref in enumerate(refs):
                ref_issues = _quote_issues(ref, f"plan_evidence[{index}]")
                if ref_issues:
                    issues.extend(ref_issues)
                    unparsed.append(ref)
                else:
                    output["plan_evidence"].append(ref)

    if "findings" in payload:
        findings = payload["findings"]
        if not isinstance(findings, list):
            issues.append("findings必须是列表。")
            unparsed.append(findings)
        else:
            seen = set()
            for index, finding in enumerate(findings):
                finding_issues = _finding_issues(finding, f"findings[{index}]")
                if finding_issues:
                    issues.extend(finding_issues)
                    unparsed.append(finding)
                    continue
                # The complete object participates, including extra fields and
                # list order.  JSON serialization also keeps True distinct from
                # 1, unlike Python's ordinary container equality.
                identity = json.dumps(finding, ensure_ascii=False, sort_keys=True)
                if identity not in seen:
                    seen.add(identity)
                    output["findings"].append(finding)

    if "context_facts" in payload:
        facts = payload["context_facts"]
        if not isinstance(facts, list):
            issues.append("context_facts必须是列表。")
            unparsed.append(facts)
        else:
            valid_facts = []
            for index, fact in enumerate(facts):
                fact_issues = _context_fact_issues(fact, f"context_facts[{index}]")
                if fact_issues:
                    issues.extend(fact_issues)
                    unparsed.append(fact)
                else:
                    valid_facts.append(fact)
            if valid_facts or not facts:
                output["context_facts"] = valid_facts

    extra = {key: value for key, value in payload.items() if key not in _OUTPUT_FIELDS}
    if extra:
        issues.append("未识别顶层字段：" + "、".join(extra) + "；原值已保留。")
        unparsed.append(extra)
    return result
