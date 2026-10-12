"""Offline six-cell retrieval experiments.

The runner deliberately stops at retrieval.  It does not construct a chat
model, score an answer, or replace a failed retrieval cell with another mode.
An :class:`~rag_arch.evidence_index.EvidenceIndex` (or a compatible fake) is
injected by the caller so local tests can exercise the complete protocol
without network access.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from typing import Any

from .benchmark_metrics import score_evidence, validate_item
from .experiment_provenance import describe_retrieval, retrieval_for_cell, sanitize_retrieval_config


RETRIEVAL_MODES = ("bm25", "dense", "rrf")
TOP_K = 8
_METRIC_KEYS = ("Recall8", "nDCG8", "Coverage")
_MISSING = object()
_SAFE_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
TABLE_COMPLETENESS_VERSION = "retrieval-table-completeness-v1"


def _as_modes(modes: Sequence[str] | str | None) -> tuple[str, ...]:
    if modes is None:
        return RETRIEVAL_MODES
    if isinstance(modes, str):
        values = [value.strip() for value in modes.split(",") if value.strip()]
    else:
        values = list(modes)
    if not values:
        raise ValueError("modes不能为空；可选bm25、dense、rrf。")
    if any(not isinstance(value, str) or value not in RETRIEVAL_MODES for value in values):
        raise ValueError("检索模式仅支持bm25、dense、rrf；hybrid不属于本六格实验。")
    return tuple(dict.fromkeys(values))


def _as_rerank_options(options: Sequence[bool] | bool | None,
                        rerank: bool | None) -> tuple[bool, ...]:
    if rerank is not None:
        options = (False, True) if rerank else (False,)
    if options is None:
        options = (False, True)
    if isinstance(options, bool):
        options = (False, True) if options else (False,)
    values = list(options)
    if not values or any(type(value) is not bool for value in values):
        raise ValueError("rerank_options须为布尔值序列；至少保留无重排单元。")
    values = list(dict.fromkeys(values))
    if True in values and False not in values:
        # A reranked-only run is useful for diagnosis, but the default six-cell
        # protocol always includes the unranked counterpart.  Do not silently
        # add it when a caller explicitly supplied a sequence.
        return tuple(values)
    return tuple(values)


def _unique_ids(values: Iterable[Any]) -> list[str]:
    """Convert an ID iterable to first-seen, non-empty strings."""
    result: list[str] = []
    seen: set[str] = set()
    if values is None:
        return result
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _item_id(item: Mapping[str, Any], position: int) -> str:
    value = item.get("item_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"line-{position}"


def _hit_ids(value: Any) -> list[str]:
    """Read IDs from SearchHit-like objects, mappings, or plain strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("evidence_id", "chunk_id", "id"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return [candidate]
        return []
    candidate = getattr(value, "chunk_id", None)
    if isinstance(candidate, str):
        return [candidate]
    chunk = getattr(value, "chunk", None)
    candidate = getattr(chunk, "chunk_id", None)
    if isinstance(candidate, str):
        return [candidate]
    if isinstance(chunk, Mapping):
        candidate = chunk.get("chunk_id") or chunk.get("evidence_id")
        if isinstance(candidate, str):
            return [candidate]
    return []


def _sequence_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return _unique_ids([value])
    if not isinstance(value, Iterable):
        return []
    result: list[str] = []
    for entry in value:
        result.extend(_hit_ids(entry))
    return _unique_ids(result)


def _result_value(result: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(result, Mapping):
        return result[name] if name in result else default
    return getattr(result, name, default)


def _extract_result(result: Any, top_k: int) -> dict[str, Any]:
    """Extract the raw and expanded trace from EvidenceResult-compatible data."""
    raw = _sequence_ids(_result_value(result, "raw_ids"))
    if not raw:
        raw = _sequence_ids(_result_value(result, "raw_hits"))
    # D2 counts independent IDs before top-k.  EvidenceIndex normally already
    # returns exactly top_k distinct chunks; this second guard protects injected
    # fakes and future index implementations from changing the metric.
    raw = raw[:top_k]

    all_context_value = _result_value(result, "all_context_ids")
    if all_context_value is _MISSING:
        all_context_value = _result_value(result, "context_hits")
    all_context = [] if all_context_value is _MISSING else _sequence_ids(all_context_value)

    # An explicitly empty complete/context ID list means that the retrieved
    # content is incomplete.  It must not be backfilled from all_context: the
    # real EvidenceResult intentionally exposes context_ids=[] in that case.
    complete_value = _result_value(result, "complete_context_ids")
    if complete_value is _MISSING:
        complete_value = _result_value(result, "context_ids")
    complete_is_explicit = complete_value is not _MISSING
    complete = [] if complete_value is _MISSING else _sequence_ids(complete_value)
    if not all_context and complete_is_explicit:
        all_context = list(complete)
    if not complete_is_explicit:
        # Presence alone cannot establish completeness, even for an injected
        # adapter. The real EvidenceResult supplies an explicit context_ids.
        complete = []

    missing = _result_value(result, "missing_evidence")
    if missing is _MISSING:
        missing = _result_value(result, "missing", [])
    if missing is _MISSING:
        missing = []
    if not isinstance(missing, list):
        missing = list(missing) if isinstance(missing, Iterable) and not isinstance(missing, (str, bytes, Mapping)) else [missing]
    return {
        "raw_ids": raw,
        "all_context_ids": _unique_ids(all_context),
        "context_ids": _unique_ids(complete),
        "complete_context_declared": complete_is_explicit,
        "missing": missing,
        "actual_mode": None if _result_value(result, "mode") is _MISSING else _result_value(result, "mode"),
        "reranked": None if _result_value(result, "reranked") is _MISSING else _result_value(result, "reranked"),
    }


def _error_chain(exc: Exception):
    current = exc
    seen: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _raw_error_code(exc: Exception) -> str | None:
    for current in _error_chain(exc):
        code = getattr(current, "code", None)
        if isinstance(code, str) and code:
            return code
    return None


def _http_status(exc: Exception) -> int | None:
    for current in _error_chain(exc):
        for name in ("status_code", "http_status"):
            value = getattr(current, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
    return None


def _safe_error(exc: Exception, *, fallback_code: str | None = None) -> dict[str, Any]:
    """Persist only exception type, HTTP status, and a safe provider code."""
    result: dict[str, Any] = {"type": type(exc).__name__}
    status = _http_status(exc)
    if status is not None:
        result["http_status"] = status
    code = _raw_error_code(exc)
    if code is not None and _SAFE_CODE.fullmatch(code):
        result["code"] = code
    elif fallback_code is not None and _SAFE_CODE.fullmatch(fallback_code):
        result["code"] = fallback_code
    return result


def _service_kind(mode: str, use_rerank: bool, exc: Exception) -> str | None:
    """Classify only quota/auth failures that should stop later service calls."""
    status = _http_status(exc)
    code = (_raw_error_code(exc) or "").lower()
    type_names = " ".join(type(current).__name__.lower() for current in _error_chain(exc))
    # This text is inspected only in memory for classification and is never
    # persisted.  SDK wrappers may retain the original safe exception as cause.
    detail = " ".join(str(getattr(current, "args", ())) for current in _error_chain(exc)).lower()
    auth_or_quota = (
        status in {401, 403, 407, 429}
        or any(token in code for token in ("auth", "unauthor", "forbidden", "apikey", "api_key",
                                           "credential", "quota", "rate", "limit", "billing", "allocation"))
        or any(token in type_names for token in ("auth", "credential", "quota", "ratelimit", "permission"))
        or any(token in detail for token in ("401", "403", "429", "quota", "rate limit", "unauthorized",
                                             "forbidden", "invalid api key", "free tier", "allocationquota"))
    )
    if not auth_or_quota:
        return None
    if use_rerank and any("rerank" in type(current).__name__.lower() for current in _error_chain(exc)):
        return "reranker"
    if mode != "bm25":
        return "embedding"
    return "reranker" if use_rerank else None


def _item_diagnostics(item: Mapping[str, Any], *, evidence_catalog=None,
                      plan_splits=None, known_source_ids=None) -> dict[str, Any]:
    try:
        report = validate_item(
            item,
            evidence_catalog=evidence_catalog,
            plan_splits=plan_splits,
            known_source_ids=known_source_ids,
        )
    except Exception as exc:  # structural checks must not discard a candidate run
        return {
            "item_id": item.get("item_id"),
            "errors": [{"code": "validation_failed", **_safe_error(exc)}],
            "warnings": [], "structurally_valid": False,
            "evaluation_scope": "candidate_trial",
            "assessment_scope": "structural_check_only_not_gold_certification",
        }
    # This is an ordinary integrity report.  A candidate without a human record
    # is intentionally still executable.
    return report


def _gold_sets(item: Mapping[str, Any]) -> tuple[list[list[str]], list[str]]:
    gold = item.get("gold")
    if not isinstance(gold, Mapping):
        return [], []
    sufficient = gold.get("sufficient", [])
    supporting = gold.get("supporting", [])
    if not isinstance(sufficient, (list, tuple)):
        sufficient = []
    groups = [list(group) for group in sufficient if isinstance(group, (list, tuple))]
    if not isinstance(supporting, (list, tuple)):
        supporting = []
    return groups, list(supporting)


def _evidence_kind(evidence_id: str, record: Any) -> str | None:
    """Read declared record types, never ID spelling or the question category."""
    if not isinstance(record, Mapping):
        return None
    kinds = set()
    if record.get("kind") in ("clause", "table", "status"):
        kinds.add(record["kind"])
    for kind in ("clause", "table", "status"):
        if record.get(f"{kind}_id") == evidence_id:
            kinds.add(kind)
    return next(iter(kinds)) if len(kinds) == 1 else None


def table_evidence_completeness(*, sufficient_evidence_sets,
                                complete_context_ids=None, evidence_catalog=None,
                                execution_status="ok", completeness_declared=True) -> dict[str, Any]:
    """Score the plan's table-dependent retrieval subset without changing C or G.

    Every alternative sufficient group must contain a known table. A confirmed
    table-free alternative rules the item out; an unresolved alternative leaves
    applicability unknown. Failed/skipped runs have unknown completeness, not a
    successful empty result. Existing catalog gap checks only remove IDs from
    caller-declared complete context and never promote image or OCR evidence.
    """
    result = {
        "schema_version": TABLE_COMPLETENESS_VERSION,
        "applicable": None, "complete": None, "outcome": "unknown",
        "reason": None, "unknown_type_ids": [], "complete_context_ids": [],
        "matched_sufficient_group_indices": [], "execution_status": execution_status,
        "context_basis": "declared_complete_context_with_existing_catalog_gap_checks",
    }
    if (not isinstance(sufficient_evidence_sets, (list, tuple))
            or any(not isinstance(group, (list, tuple)) or not group
                   or any(not isinstance(eid, str) or not eid or eid != eid.strip() for eid in group)
                   for group in sufficient_evidence_sets)):
        result["reason"] = "invalid_sufficient_groups"
        return result
    groups = [set(group) for group in sufficient_evidence_sets]
    if not groups:
        result.update(applicable=False, outcome="not_applicable", reason="no_sufficient_groups")
        return result
    catalog = evidence_catalog if isinstance(evidence_catalog, Mapping) else {}
    ids = set().union(*groups)
    kinds = {eid: _evidence_kind(eid, catalog.get(eid)) for eid in ids}
    result["unknown_type_ids"] = sorted(eid for eid, kind in kinds.items() if kind is None)
    table_free = [i for i, group in enumerate(groups)
                  if all(kinds[eid] is not None and kinds[eid] != "table" for eid in group)]
    if table_free:
        result.update(applicable=False, outcome="not_applicable", reason="table_free_alternative")
        return result
    if any(not any(kinds[eid] == "table" for eid in group) for group in groups):
        result["reason"] = "unknown_evidence_kind"
        return result
    result["applicable"] = True
    if execution_status != "ok":
        result["reason"] = "execution_unavailable"
        return result
    if completeness_declared is not True or complete_context_ids is None:
        result["reason"] = "complete_context_not_recorded"
        return result
    # Normalize explicit kind-only table records for the existing table gap
    # checker. This does not invent table contents or a completeness declaration.
    typed_catalog = {
        eid: {**record, "table_id": eid}
        if _evidence_kind(eid, record) == "table" else record
        for eid, record in catalog.items()
    }
    try:
        filtered = score_evidence(
            sufficient_evidence_sets=[], supporting_evidence_ids=[], raw_ranking=[],
            complete_context_ids=complete_context_ids, cited_ids=[],
            predicted_conclusion=None, gold_conclusion=None, evidence_catalog=typed_catalog,
        )
    except (ValueError, TypeError):
        result["reason"] = "invalid_complete_context"
        return result
    context = set(filtered["complete_context_ids"])
    result["complete_context_ids"] = sorted(context)
    result["diagnostics"] = filtered["diagnostics"]
    matched = [i for i, group in enumerate(groups)
               if group <= context and all(kinds[eid] is not None for eid in group)]
    unresolved = [i for i, group in enumerate(groups)
                  if group <= context and any(kinds[eid] is None for eid in group)]
    if not matched and unresolved:
        result.update(reason="unknown_evidence_kind_in_complete_group",
                      unresolved_sufficient_group_indices=unresolved)
        return result
    result.update(complete=bool(matched), outcome="complete" if matched else "incomplete",
                  reason="complete_sufficient_group" if matched else "no_complete_sufficient_group",
                  matched_sufficient_group_indices=matched)
    return result


def summarize_table_completeness(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep unknown and unavailable cells visible instead of shrinking the rate."""
    values = list(metrics)
    applicable = [value for value in values if value.get("applicable") is True]
    complete = sum(value.get("complete") is True for value in applicable)
    incomplete = sum(value.get("complete") is False for value in applicable)
    unknown = len(applicable) - complete - incomplete
    pending = sum(value.get("applicable") is None for value in values)
    known = complete + incomplete
    return {
        "schema_version": TABLE_COMPLETENESS_VERSION, "item_count": len(values),
        "applicable_count": len(applicable),
        "not_applicable_count": sum(value.get("applicable") is False for value in values),
        "applicability_unknown_count": pending,
        "complete_count": complete, "incomplete_count": incomplete, "unknown_count": unknown,
        "known_count": known,
        "execution_error_count": sum(value.get("execution_status") == "error" for value in applicable),
        "skipped_count": sum(value.get("execution_status") in {"skipped", "not_run"} for value in applicable),
        "rate": complete / len(applicable) if applicable and not unknown and not pending else None,
        "evaluated_rate": complete / known if known else None,
        "rate_basis": "all_applicable_items; unresolved_applicability_or_completeness_keeps_rate_null",
    }


def rescore_table_completeness(rows: Iterable[Mapping[str, Any]],
                              items: Iterable[Mapping[str, Any]], *, evidence_catalog,
                              expected_cells: Sequence[tuple[str, bool]] | None = None) -> dict[str, Any]:
    """Annotate saved retrieval rows offline; do not retrieve or rewrite old metrics.

    Join by split/category/item_id, checking the saved query when present. The
    caller supplies the exact intended item definitions and catalog; this is an
    annotation using those inputs, not a new retrieval run or Gold review.
    Supply the saved run's expected cells to register absent rows as not_run.
    No split is hard-coded, so future Test inputs use the same metric.
    """
    key_fields = ("split", "category", "item_id")
    item_index = {}
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in item_index:
            raise ValueError("题目定义包含重复 split/category/item_id，不能选择某次版本。")
        item_index[key] = item
    source_rows = list(rows)
    missing_cells = []
    if expected_cells is not None:
        cells = list(dict.fromkeys(expected_cells))
        if any(mode not in RETRIEVAL_MODES or type(rerank) is not bool for mode, rerank in cells):
            raise ValueError("expected_cells须为已知检索模式与布尔重排值。")
        present = {(*(row.get(field) for field in key_fields), row.get("mode"), row.get("rerank"))
                   for row in source_rows}
        for key, item in item_index.items():
            for mode, rerank in cells:
                if (*key, mode, rerank) not in present:
                    identity = dict(zip(key_fields, key))
                    missing_cells.append({**identity, "mode": mode, "rerank": rerank})
                    scopes = {row.get("evaluation_scope") for row in source_rows
                              if tuple(row.get(field) for field in key_fields) == key}
                    scope = next(iter(scopes)) if len(scopes) == 1 else _item_diagnostics(
                        item, evidence_catalog=evidence_catalog)["evaluation_scope"]
                    source_rows.append({**identity, "record_type": "missing_retrieval_cell",
                                        "evaluation_scope": scope,
                                        "mode": mode, "rerank": rerank, "cell": f"{mode}_{'rerank' if rerank else 'no_rerank'}",
                                        "status": "not_run", "source_row_present": False})
    annotated = []
    seen = set()
    for source in source_rows:
        key = tuple(source.get(field) for field in key_fields)
        cell_key = (*key, source.get("mode"), source.get("rerank"))
        if cell_key in seen:
            raise ValueError("保存结果包含重复题目/检索单元；请按运行分别复算。")
        seen.add(cell_key)
        row = deepcopy(dict(source))
        item = item_index.get(key)
        reason = None
        if item is None:
            reason = "item_definition_unavailable"
        elif "query" in source and source["query"] != str(item.get("question", item.get("query", ""))).strip():
            reason = "saved_query_item_mismatch"
        gold = item.get("gold") if item is not None else None
        groups = gold.get("sufficient") if isinstance(gold, Mapping) else None
        context_field = "complete_context_ids" if "complete_context_ids" in source else "context_ids"
        metric = table_evidence_completeness(
            sufficient_evidence_sets=groups if reason is None else None,
            complete_context_ids=source.get(context_field), evidence_catalog=evidence_catalog,
            execution_status=source.get("status"),
            completeness_declared=source.get("complete_context_declared", context_field in source),
        )
        if reason is not None:
            metric["reason"] = reason
        row["table_completeness"] = metric
        annotated.append(row)
    return {
        "schema_version": TABLE_COMPLETENESS_VERSION,
        "operation": "offline_annotation_of_saved_retrieval_rows",
        "context_basis": "saved_complete_context_ids_with_supplied_catalog_gap_checks",
        "new_retrieval_performed": False, "formal_gold_promoted": False,
        "missing_cells": missing_cells,
        "result_count": len(annotated), "rows": annotated, "groups": _group_summary(annotated),
    }


def _base_row(item: Mapping[str, Any], position: int, mode: str,
              use_rerank: bool, diagnostics: Mapping[str, Any], query: str) -> dict[str, Any]:
    return {
        "record_type": "retrieval_cell",
        "item_id": _item_id(item, position),
        "split": item.get("split"),
        "category": item.get("category"),
        "benchmark_track": deepcopy(item.get("benchmark_track", "clause_table")),
        "evaluation_scope": diagnostics.get("evaluation_scope", "candidate_trial"),
        "query": query,
        "verification_date": deepcopy(item.get("verification_date")),
        "verification_source_ids": deepcopy(item.get("verification_source_ids")),
        "reference_status_note": "题目登记的收录核验日期与来源；不表示本次运行重新查询了规范现行状态。",
        "mode": mode,
        "rerank": use_rerank,
        "cell": f"{mode}_{'rerank' if use_rerank else 'no_rerank'}",
        "actual_mode": None,
        "reranked": None,
        "raw_ids": [],
        "context_ids": [],
        "all_context_ids": [],
        "complete_context_declared": False,
        "missing": [],
        "Recall8": None,
        "nDCG8": None,
        "Coverage": None,
        # Lowercase aliases mirror benchmark_metrics while the three requested
        # display names remain stable in JSONL and terminal output.
        "recall_at_8": None,
        "ndcg_at_8": None,
        "evidence_coverage": None,
        "latency_ms": None,
        "status": "error",
        "item_diagnostics": dict(diagnostics),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("split"), row.get("category"), row.get("evaluation_scope"),
                 row.get("mode"), bool(row.get("rerank")), row.get("benchmark_track", "clause_table"))].append(row)
    groups: list[dict[str, Any]] = []
    for key, members in sorted(buckets.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        split, category, scope, mode, rerank, track = key
        separate_ids = _unique_ids(row["item_id"] for row in members
                                   if row.get("benchmark_track") == "standard_status")
        eligible = [row for row in members if row.get("benchmark_track") != "standard_status"]
        group: dict[str, Any] = {
            "split": split, "category": category, "evaluation_scope": scope,
            "benchmark_track": track,
            "mode": mode, "rerank": rerank, "cell": f"{mode}_{'rerank' if rerank else 'no_rerank'}",
            "sample_count": len(members),
            "success_count": sum(row.get("status") == "ok" for row in members),
            "error_count": sum(row.get("status") == "error" for row in members),
            "skipped_count": sum(row.get("status") == "skipped" for row in members),
            "separate_status_item_ids": separate_ids,
            "separate_status_item_count": len(separate_ids),
            "separate_status_cell_count": sum(row.get("status") == "skipped"
                                               and row.get("skip_reason") == "separate_status_benchmark"
                                               for row in members if row.get("benchmark_track") == "standard_status"),
        }
        for field in _METRIC_KEYS:
            values = [row[field] for row in eligible if isinstance(row.get(field), (int, float))]
            group[field] = {"mean": sum(values) / len(values) if values else None,
                            "valid_count": len(values)}
        group["table_completeness"] = summarize_table_completeness(
            row.get("table_completeness", {}) for row in eligible)
        groups.append(group)
    return groups


def run_retrieval_grid(
    index,
    items: Iterable[Mapping[str, Any]],
    *,
    modes: Sequence[str] | str | None = None,
    rerank_options: Sequence[bool] | bool | None = None,
    rerank: bool | None = None,
    top_k: int = TOP_K,
    callback: Callable[[dict[str, Any]], Any] | None = None,
    output_path: str | Path | None = None,
    append: bool = False,
    overwrite: bool = False,
    evidence_catalog: Mapping[str, Mapping] | None = None,
    plan_splits: Mapping[str, str] | None = None,
    known_source_ids: Sequence[str] | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run each item through the requested retrieval cells.

    The default is the full ``BM25/Dense/RRF × no-rerank/rerank`` grid.  Each
    cell is persisted immediately when ``output_path`` is supplied and is sent
    to ``callback`` after persistence.  A retrieval exception creates an error
    row for that question/cell and does not trigger a fallback mode.

    ``index`` only needs a compatible ``retrieve(query, mode, top_k,
    use_reranker)`` method.  This makes the function suitable for fake indexes
    in tests and for the real :class:`EvidenceIndex` in a later run.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != TOP_K:
        raise ValueError("D2检索实验固定top_k=8；Recall8不接受其他top_k。")
    selected_modes = _as_modes(modes)
    selected_rerank = _as_rerank_options(rerank_options, rerank)
    retrieval_snapshot = (describe_retrieval(index) if retrieval_config is None
                          else sanitize_retrieval_config(retrieval_config))
    source_items = list(items)
    if any(not isinstance(item, Mapping) for item in source_items):
        # Keep malformed rows visible rather than silently filtering them.
        source_items = [item if isinstance(item, Mapping) else {"_input_error": "item必须为对象", "item_id": f"line-{n}"}
                        for n, item in enumerate(source_items, 1)]

    cells = [(mode, use_rerank) for mode in selected_modes for use_rerank in selected_rerank]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    stopped_services: dict[str, dict[str, Any]] = {}
    writer = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nonempty = path.exists() and path.stat().st_size > 0
        if nonempty and not overwrite:
            raise FileExistsError("输出JSONL已存在且非空；请改用新路径或显式overwrite=True，避免重复聚合。")
        # append is retained only for compatibility with an empty path.  A
        # non-empty checkpoint is never silently extended or deduplicated.
        writer = path.open("a" if append and not overwrite else "w", encoding="utf-8", newline="")

    def emit(row: dict[str, Any]) -> None:
        gold = item.get("gold")
        row["table_completeness"] = table_evidence_completeness(
            sufficient_evidence_sets=[] if row.get("benchmark_track") == "standard_status"
            else gold.get("sufficient") if isinstance(gold, Mapping) else None,
            complete_context_ids=row["context_ids"], evidence_catalog=evidence_catalog,
            execution_status=row["status"], completeness_declared=row["complete_context_declared"],
        )
        if row.get("benchmark_track") == "standard_status":
            row["table_completeness"]["reason"] = "separate_status_benchmark"
        if writer is not None:
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            writer.flush()
        if callback is not None:
            callback(row)

    try:
        for position, item in enumerate(source_items, 1):
            diagnostics_report = _item_diagnostics(
                item, evidence_catalog=evidence_catalog, plan_splits=plan_splits,
                known_source_ids=known_source_ids,
            )
            diagnostics.append({"item_id": _item_id(item, position), **diagnostics_report})
            raw_query = item.get("question")
            if not isinstance(raw_query, str):
                # The Dev JSONL uses question; accepting query keeps the
                # injected runner useful for small retrieval-only fixtures.
                raw_query = item.get("query")
            query = raw_query if isinstance(raw_query, str) else ""
            query = query.strip()
            groups, supporting = _gold_sets(item)
            for mode, use_rerank in cells:
                started = time.perf_counter()
                row = _base_row(item, position, mode, use_rerank, diagnostics_report, query)
                row["config"] = {"retrieval": retrieval_for_cell(retrieval_snapshot, mode, use_reranker=use_rerank),
                                 "item_variant": deepcopy(item.get("item_variant"))}
                if item.get("benchmark_track") == "standard_status":
                    if diagnostics_report.get("structurally_valid") is not True:
                        row.update(status="error", error={"component": "validation", "code": "invalid_standard_status_item"})
                        errors.append({"item_id": row["item_id"], "cell": row["cell"], **row["error"]})
                    else:
                        row.update(status="skipped", skip_reason="separate_status_benchmark")
                    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    rows.append(row)
                    emit(row)
                    continue
                required_services = []
                if mode != "bm25":
                    required_services.append("embedding")
                if use_rerank:
                    required_services.append("reranker")
                blocked_service = next((service for service in required_services if service in stopped_services), None)
                if blocked_service is not None:
                    row.update({
                        "status": "skipped",
                        "skipped_service": blocked_service,
                        "error": dict(stopped_services[blocked_service]),
                    })
                    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    rows.append(row)
                    emit(row)
                    continue
                try:
                    if item.get("_input_error"):
                        raise ValueError(str(item["_input_error"]))
                    if not query:
                        raise ValueError("题目缺少非空question，未执行检索。")
                    result = index.retrieve(query, mode=mode, top_k=top_k,
                                           use_reranker=use_rerank)
                    extracted = _extract_result(result, top_k)
                    actual_mode = extracted["actual_mode"]
                    if actual_mode is None:
                        actual_mode = mode
                    if actual_mode != mode:
                        raise RuntimeError(f"索引实际模式{actual_mode!r}与请求模式{mode!r}不一致；未采用替代模式。")
                    actual_reranked = extracted["reranked"]
                    if actual_reranked is not None and bool(actual_reranked) != use_rerank:
                        raise RuntimeError("索引实际重排状态与请求单元不一致；未采用替代结果。")
                    score = score_evidence(
                        sufficient_evidence_sets=groups,
                        supporting_evidence_ids=supporting,
                        raw_ranking=extracted["raw_ids"],
                        complete_context_ids=extracted["context_ids"],
                        cited_ids=[],
                        predicted_conclusion=None,
                        gold_conclusion=None,
                        evidence_catalog=evidence_catalog,
                    )
                    row.update({
                        "status": "ok", "actual_mode": actual_mode,
                        "reranked": bool(actual_reranked) if actual_reranked is not None else use_rerank,
                        "raw_ids": extracted["raw_ids"],
                        "context_ids": extracted["context_ids"],
                        "complete_context_declared": extracted["complete_context_declared"],
                        "all_context_ids": extracted["all_context_ids"],
                        "missing": extracted["missing"],
                        "Recall8": score["recall_at_8"],
                        "nDCG8": score["ndcg_at_8"],
                        "Coverage": score["evidence_coverage"],
                        "recall_at_8": score["recall_at_8"],
                        "ndcg_at_8": score["ndcg_at_8"],
                        "evidence_coverage": score["evidence_coverage"],
                        "metric_diagnostics": score.get("diagnostics", []),
                    })
                except Exception as exc:
                    service = _service_kind(mode, use_rerank, exc)
                    row["error"] = _safe_error(exc)
                    if service is not None:
                        stop_code = f"{service}_stopped"
                        stopped_services[service] = _safe_error(exc, fallback_code=stop_code)
                        row["error"] = dict(stopped_services[service])
                        row["service_stopped"] = service
                    errors.append({"item_id": row["item_id"], "cell": row["cell"], **row["error"]})
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                rows.append(row)
                emit(row)
    finally:
        if writer is not None:
            writer.close()

    return {
        "schema_version": "retrieval-grid-v1",
        "protocol": "retrieval_only_no_generation_metrics",
        "top_k": top_k,
        "retrieval_config": retrieval_snapshot,
        "modes": list(selected_modes),
        "rerank_options": list(selected_rerank),
        "cells": [{"mode": mode, "rerank": use_rerank,
                   "cell": f"{mode}_{'rerank' if use_rerank else 'no_rerank'}"}
                  for mode, use_rerank in cells],
        "item_count": len(source_items),
        "result_count": len(rows),
        "error_count": len(errors),
        "skipped_count": sum(row.get("status") == "skipped" for row in rows),
        "separate_status_item_ids": _unique_ids(row["item_id"] for row in rows
                                                 if row.get("benchmark_track") == "standard_status"),
        "separate_status_item_count": len({_item_id(item, position)
                                            for position, item in enumerate(source_items, 1)
                                            if item.get("benchmark_track") == "standard_status"}),
        "separate_status_cell_count": sum(row.get("benchmark_track") == "standard_status"
                                           and row.get("status") == "skipped"
                                           and row.get("skip_reason") == "separate_status_benchmark" for row in rows),
        "stopped_services": stopped_services,
        "rows": rows,
        "errors": errors,
        "diagnostics": diagnostics,
        "groups": _group_summary(rows),
        "output_path": str(output_path) if output_path is not None else None,
    }


def load_items_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load candidate items while keeping malformed lines as visible failures."""
    path = Path(path)
    items: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError("JSONL行必须为对象")
            items.append(dict(value))
        except (json.JSONDecodeError, ValueError) as exc:
            items.append({"item_id": f"line-{line_number}", "question": "",
                          "_input_error": f"JSONL第{line_number}行无效：{exc}"})
    return items


__all__ = ["RETRIEVAL_MODES", "TOP_K", "load_items_jsonl", "run_retrieval_grid",
           "table_evidence_completeness", "summarize_table_completeness", "rescore_table_completeness"]
