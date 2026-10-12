"""Bounded historical context for whole-plan review follow-ups.

History helps the model resolve references such as "上一轮那个问题".  It is
never a source of norm evidence.  The helper keeps the document identity so
the caller can apply its existing document/revision scope filter, and keeps
the risk-bearing parts of a prior result without retaining the large runtime
payload, messages, or raw evidence objects.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


HISTORY_CHAR_LIMIT = 24000
HISTORY_NON_EVIDENCE_NOTE = "历史回答仅供理解指代，不能作为本轮规范证据。"
HISTORY_TRUNCATION_NOTE = "历史已截短；仅供理解指代，不能作为本轮规范证据。"

_TURN_STRING_FIELDS = ("question", "answer", "status")
_TURN_LIST_FIELDS = ("limitations", "validation_notes")
_FINDING_STRING_FIELDS = (
    "finding_id", "title", "category", "verdict", "model_verdict",
    "description", "recommendation",
)
_CITATION_LIST_HINTS = ("issue", "incomplete", "unresolved", "warning")
_CITATION_SCALAR_FIELDS = ("status", "semantic_support_verified", "passed")
_QUOTE_FIELDS = ("plan_evidence", "condition_evidence", "condition_citations")
_FINAL_REVIEW_SCALAR_FIELDS = (
    "status", "code", "error_type", "requested", "attempt", "attempts",
    "format_retry_used", "model_retry_used",
)


def _strings(value: Any) -> list[str]:
    """Return non-empty strings without coercing untrusted values."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Copy small JSON-shaped metadata and drop non-serializable values.

    This is used for source-coverage metadata only.  It deliberately does not
    copy arbitrary runtime payloads, model messages, traces, or evidence.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if depth >= 4:
        return None
    if isinstance(value, list):
        return [item for item in (_json_safe(v, depth=depth + 1) for v in value)
                if item is not None]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            safe = _json_safe(item, depth=depth + 1)
            if safe is not None:
                output[key] = safe
        return output
    return None


def _quote_refs(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    refs = []
    for ref in value:
        if not isinstance(ref, dict):
            continue
        unit_id = ref.get("unit_id")
        quote = ref.get("quote")
        if isinstance(unit_id, str) and isinstance(quote, str):
            refs.append({"unit_id": unit_id, "quote": quote})
    return refs


def _sanitize_citation_check(value: Any) -> dict | None:
    """Keep citation risk diagnostics while excluding copied evidence bodies."""
    if not isinstance(value, dict):
        return None
    output = {}
    for key in _CITATION_SCALAR_FIELDS:
        if key in value and isinstance(value[key], (str, bool)):
            output[key] = value[key]
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        if key in _CITATION_SCALAR_FIELDS:
            continue
        if (key == "issues" or any(hint in key.lower() for hint in _CITATION_LIST_HINTS)):
            if isinstance(raw, list):
                output[key] = _strings(raw)
    return output or None


def _sanitize_finding(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    output = {}
    for key in _FINDING_STRING_FIELDS:
        if isinstance(value.get(key), str):
            output[key] = value[key]
    for key in ("standard_evidence_ids", "conditions"):
        if isinstance(value.get(key), list):
            output[key] = _strings(value[key])
    for key in _QUOTE_FIELDS:
        refs = _quote_refs(value.get(key))
        if refs is not None:
            output[key] = refs
    citation_check = _sanitize_citation_check(value.get("citation_check"))
    if citation_check is not None:
        output["citation_check"] = citation_check
    return output or None


def _final_review_status(value: dict) -> dict | None:
    raw = value.get("final_review")
    if raw is None and isinstance(value.get("runtime"), dict):
        raw = value["runtime"].get("final_review")
    if isinstance(raw, str):
        return {"status": raw}
    if isinstance(raw, dict) and isinstance(raw.get("status"), str):
        output = {}
        for key in _FINAL_REVIEW_SCALAR_FIELDS:
            item = raw.get(key)
            if isinstance(item, (str, bool, int, float)):
                output[key] = item
        # Some callers keep retry counters beside, rather than inside, the
        # final-review record.  Copy only the short scalar diagnostics.
        for key in ("format_retry_used", "model_retry_used", "attempts"):
            if key not in output and isinstance(value.get(key), (bool, int, float)):
                output[key] = value[key]
            runtime = value.get("runtime")
            if key not in output and isinstance(runtime, dict) and isinstance(runtime.get(key), (bool, int, float)):
                output[key] = runtime[key]
        error = raw.get("error")
        if "error_type" not in output and isinstance(error, dict):
            exception_types = error.get("exception_types")
            if isinstance(exception_types, list):
                first = next((item for item in exception_types if isinstance(item, str) and item), None)
                if first is not None:
                    output["error_type"] = first
        return output
    return None


def _sanitize_body(value: dict, *, include_identity: bool) -> dict:
    output = {}
    if include_identity:
        output["document_id"] = value["document_id"]
        output["source_revision"] = value["source_revision"]

    # This marker is always present, including when the history is not yet
    # truncated.  It prevents a later prompt from treating old prose as proof.
    output["history_note"] = HISTORY_NON_EVIDENCE_NOTE
    for key in _TURN_STRING_FIELDS:
        if isinstance(value.get(key), str):
            output[key] = value[key]
    if isinstance(value.get("page_sources"), list):
        output["page_sources"] = _json_safe(value["page_sources"])
    if isinstance(value.get("web_sources"), list):
        output["web_sources"] = _json_safe(value["web_sources"])
    if isinstance(value.get("source_coverage"), dict):
        output["source_coverage"] = _json_safe(value["source_coverage"])
    if isinstance(value.get("evidence_ids"), list):
        # These are historical reference IDs only; the history note prevents
        # them from being treated as evidence in the new turn.
        output["evidence_ids"] = _strings(value["evidence_ids"])
    for key in _TURN_LIST_FIELDS:
        if isinstance(value.get(key), list):
            output[key] = _strings(value[key])

    final_review = _final_review_status(value)
    if final_review is not None:
        output["final_review"] = final_review

    findings = []
    if isinstance(value.get("findings"), list):
        for raw in value["findings"]:
            item = _sanitize_finding(raw)
            if item is not None:
                findings.append(item)
    output["findings"] = findings
    return output


def sanitize_history_turn(value: Any) -> dict | None:
    """Reduce one prior result to safe, risk-aware conversational context.

    A valid document id and source revision are required.  Scope filtering is
    intentionally left to the caller; this helper does not merge documents.
    """
    if not isinstance(value, dict):
        return None
    document_id = value.get("document_id")
    source_revision = value.get("source_revision")
    if not isinstance(document_id, str) or not isinstance(source_revision, str):
        return None
    return _sanitize_body(value, include_identity=True)


def _history_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _prepare_rows(rows: Any) -> list[dict]:
    """Sanitize complete turns while accepting already scope-filtered bodies."""
    if not isinstance(rows, (list, tuple)):
        return []
    prepared = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("document_id"), str) and isinstance(row.get("source_revision"), str):
            clean = sanitize_history_turn(row)
        else:
            # The existing agent filters document scope first and then passes
            # only question/answer/findings.  Keep that compatibility path;
            # scope ownership remains with the caller.
            clean = _sanitize_body(row, include_identity=False)
        prepared.append(clean)
    return prepared


def _mark_history_truncated(rows: list[dict], *, findings_omitted: bool = False) -> list[dict]:
    marker = {"history_truncated": True, "history_note": HISTORY_TRUNCATION_NOTE}
    if findings_omitted:
        marker["history_findings_omitted"] = True
    if not rows:
        return [marker]
    marked = deepcopy(rows)
    # Marker wins over a copied note so the truncation state cannot be hidden.
    marked[0] = {**marked[0], **marker}
    return marked


def _trim_field(row: dict, field_name: str, max_chars: int) -> None:
    value = row.get(field_name)
    if not isinstance(value, str) or len(_history_json([row])) <= max_chars:
        return
    suffix = "…[已截短]"

    def candidate(length: int) -> str:
        return value if length >= len(value) else value[:length] + suffix

    low, high, best = 0, len(value), None
    while low <= high:
        middle = (low + high) // 2
        row[field_name] = candidate(middle)
        if len(_history_json([row])) <= max_chars:
            best = row[field_name]
            low = middle + 1
        else:
            high = middle - 1
    row[field_name] = best if best is not None else suffix


def _minimal_row(row: dict) -> dict:
    """Keep a compact risk summary if even question/answer trimming is tight."""
    minimal = {
        "history_truncated": True,
        "history_note": HISTORY_TRUNCATION_NOTE,
        "history_findings_omitted": True,
    }
    for key in ("document_id", "source_revision", "status"):
        if isinstance(row.get(key), str):
            minimal[key] = row[key]
    if isinstance(row.get("final_review"), dict) and isinstance(row["final_review"].get("status"), str):
        minimal["final_review"] = {
            key: row["final_review"][key]
            for key in _FINAL_REVIEW_SCALAR_FIELDS
            if key in row["final_review"]
            and isinstance(row["final_review"][key], (str, bool, int, float))
        }
    if isinstance(row.get("source_coverage"), dict):
        coverage = {}
        for key in ("provided_units", "total_units", "all_readable_text_provided"):
            if key in row["source_coverage"] and isinstance(row["source_coverage"][key], (str, bool, int, float)):
                coverage[key] = row["source_coverage"][key]
        if coverage:
            minimal["source_coverage"] = coverage
    if row.get("limitations"):
        minimal["limitations"] = ["历史已截短；limitations未完整保留。"]
    if row.get("validation_notes"):
        minimal["validation_notes"] = ["历史已截短；validation_notes未完整保留。"]
    return minimal


def bounded_history_json(rows: Any, max_chars: int = HISTORY_CHAR_LIMIT) -> tuple[list[dict], str]:
    """Return sanitized history and valid JSON no longer than ``max_chars``.

    When the bound is exceeded, only the newest turn is retained.  Its whole
    findings list is removed together, rather than retaining a conclusion
    while silently dropping that finding's conditions or citation risks.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars必须是正整数。")
    prepared = _prepare_rows(rows)
    serialized = _history_json(prepared)
    if len(serialized) <= max_chars:
        return prepared, serialized

    retained = [deepcopy(prepared[-1])] if prepared else []
    findings_omitted = bool(retained and retained[0].get("findings"))
    if retained:
        retained[0].pop("findings", None)
    retained = _mark_history_truncated(retained, findings_omitted=findings_omitted)
    if len(_history_json(retained)) <= max_chars:
        return retained, _history_json(retained)

    row = retained[0]
    for field_name in ("answer", "question"):
        _trim_field(row, field_name, max_chars)
        if len(_history_json(retained)) <= max_chars:
            return retained, _history_json(retained)
    if len(_history_json(retained)) <= max_chars:
        return retained, _history_json(retained)

    minimal = _minimal_row(row)
    serialized = _history_json([minimal])
    if len(serialized) > max_chars:
        raise ValueError("max_chars过小，无法同时保留历史截短标识。")
    return [minimal], serialized


__all__ = [
    "HISTORY_CHAR_LIMIT",
    "HISTORY_NON_EVIDENCE_NOTE",
    "HISTORY_TRUNCATION_NOTE",
    "sanitize_history_turn",
    "bounded_history_json",
]
