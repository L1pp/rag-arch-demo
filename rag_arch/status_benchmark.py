"""Offline structural scoring of the explicit standard-status benchmark track.

The caller supplies the actual ordinary registry result. No lookup, model call,
answer-text grading, or conversion from registry status to engineering four
states occurs here. Saved source records are candidate references, not Gold.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


STATUS_TRACK = "standard_status"
SCORING_VARIANT = "status-record-match-v1"
_STATE_FIELDS = (
    "found", "std_no", "registered_status", "standard_status",
    "standard_status_evidence_level", "current_effective_date", "superseded_by",
    "original_mandatory", "current_mandatory_status",
    "full_text_mandatory_at_publication", "clause_found", "clause_lookup_status",
)
_SOURCE_FIELDS = (
    "source_url", "evidence_level", "observed_status", "supports_current_status",
    "checked_at", "all_body_clauses_mandatory_at_publication",
)
_DATE_FIELDS = ("verification_date", "status_checked_at", "relation_verified_at")


def _state_fields(query: Mapping) -> tuple[str, ...]:
    # A whole-standard relation list is not a whole-standard repeal conclusion.
    # Clause queries also need the exact/parent/subclause relation boundaries.
    return _STATE_FIELDS + (("matched_relations", "related_subclause_relations")
                            if query["clause_no"] is not None else ())


def is_status_benchmark(item: Mapping) -> bool:
    """Recognize only explicitly marked S items; other types get no exemption."""
    return (isinstance(item, Mapping) and item.get("benchmark_track") == STATUS_TRACK
            and item.get("category") in ("E-real", "E-injected")
            and item.get("item_type") == "S")


def _query(record: Mapping) -> dict:
    return {"std_no": record.get("query_std_no"), "clause_no": record.get("clause_no")}


def _has_query(record: Mapping) -> bool:
    return "query_std_no" in record and "clause_no" in record


def _ids(values) -> bool:
    return (isinstance(values, list)
            and all(isinstance(v, str) and bool(v.strip()) and v == v.strip() for v in values)
            and len(set(values)) == len(values))


def validate_status_reference(item: Mapping) -> list[dict]:
    """Validate only this track's metadata, without inspecting answer strings."""
    errors = []

    def add(code, field, message):
        errors.append({"code": code, "field": field, "message": message})

    if not is_status_benchmark(item):
        add("invalid_status_track_type", "benchmark_track", "状态轨道只适用于显式 E 类 S 题")
        return errors
    query = item.get("status_query")
    valid_query = (isinstance(query, Mapping) and set(query) == {"std_no", "clause_no"}
                   and isinstance(query.get("std_no"), str) and bool(query["std_no"].strip())
                   and (query.get("clause_no") is None
                        or isinstance(query["clause_no"], str) and bool(query["clause_no"].strip())))
    if not valid_query:
        add("invalid_status_query", "status_query", "保留原查询 std_no 和显式 clause_no（可为 null），不自动纠正编号")
    assertion = item.get("assertion")
    if (valid_query and (not isinstance(assertion, Mapping)
                         or assertion.get("citation_query") != [query["std_no"]])):
        add("status_query_assertion_mismatch", "assertion.citation_query", "状态原查询须与断言保留的单个 citation_query 编号逐字一致")
    if not isinstance(item.get("item_variant"), str) or not item["item_variant"].strip():
        add("missing_status_item_variant", "item_variant", "状态轨道需记录题目口径版本")
    gold = item.get("gold")
    if not isinstance(gold, Mapping) or gold.get("sufficient") != [] or gold.get("supporting") != []:
        add("status_track_has_clause_evidence", "gold", "状态题的条表 sufficient/supporting 必须均为空")
    groups = item.get("status_evidence_groups")
    valid_groups = (isinstance(groups, list) and all(_ids(group) and group for group in groups))
    if not valid_groups:
        add("invalid_status_evidence_groups", "status_evidence_groups", "需来源 ID 组；未知无来源用 []，不能用 [[]]")
    records = item.get("source_status_records")
    if not isinstance(records, list) or not records or not all(isinstance(r, Mapping) for r in records):
        add("invalid_source_status_records", "source_status_records", "需保存原状态查询结果作为候选参照")
        return errors
    matched = [r for r in records if valid_query and _has_query(r) and _query(r) == query]
    if len(matched) != 1:
        add("status_reference_query_mismatch", "source_status_records", "原始查询须唯一对应一条候选状态记录")
        return errors
    reference = matched[0]
    if any(field not in reference for field in _state_fields(query)) or type(reference.get("found")) is not bool:
        add("incomplete_status_reference", "source_status_records", "候选状态记录缺少完整状态字段或 found 布尔值")
    if not _ids(reference.get("source_ids")):
        add("invalid_status_source_ids", "source_status_records.source_ids", "来源 ID 需明确数组，未命中时可为空")
    elif valid_groups:
        expected_ids = set().union(*(set(group) for group in groups))
        if not expected_ids <= set(reference["source_ids"]):
            add("status_group_not_in_reference", "status_evidence_groups", "组内来源须实际存在于对应候选状态记录")
        verification_ids = item.get("verification_source_ids")
        if _ids(verification_ids) and not expected_ids <= set(verification_ids):
            add("status_group_not_in_verification_sources", "verification_source_ids", "来源组须保留在本题核验来源中")
    if reference.get("found") is False:
        unknown_fields = ("registered_status", "standard_status", "standard_status_evidence_level", "current_mandatory_status")
        null_fields = ("current_effective_date", "superseded_by", "original_mandatory",
                       "full_text_mandatory_at_publication", "clause_found")
        empty_fields = ("source_ids", "source_urls", "sources", "matched_relations", "related_subclause_relations")
        expected_lookup = "not_requested" if query["clause_no"] is None else "unknown"
        if (reference.get("std_no") != query["std_no"]
                or any(reference.get(f) != "unknown" for f in unknown_fields)
                or any(reference.get(f) is not None for f in null_fields)
                or any(reference.get(f) != [] for f in empty_fields)
                or reference.get("clause_lookup_status") != expected_lookup or groups != []):
            add("inconsistent_status_no_hit", "source_status_records", "未命中参照须保留原查询、unknown 与空来源/关系，不能夹带现行或废止结论")
    return errors


def _same(left, right) -> bool:
    """Keep missing/null and bool/number distinctions meaningful."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _sources(record: Mapping) -> dict[str, list[Mapping]]:
    result = {}
    for source in record.get("sources", []) if isinstance(record.get("sources"), list) else []:
        if isinstance(source, Mapping) and isinstance(source.get("source_id"), str):
            result.setdefault(source["source_id"], []).append(source)
    return result


def score_status_result(item: Mapping, status_result: Mapping | None) -> dict:
    """Compare actual status data to the saved candidate reference, independently.

    Missing result/fields stay None. Explicit unequal values are False. Source
    matching requires a complete alternative source group and the saved URL /
    evidence-level / observation metadata, not only IDs. Empty expected source
    groups are not applicable. Extra returned sources are reported, not graded
    as semantic citation errors. Inputs and nested output snapshots are detached.
    """
    errors = validate_status_reference(item)
    if errors:
        raise ValueError("状态题参照无效：" + "; ".join(e["code"] for e in errors))
    if status_result is not None and not isinstance(status_result, Mapping):
        raise ValueError("status_result 需为实际状态查询对象或 None")
    query = item["status_query"]
    state_fields = _state_fields(query)
    reference = next(r for r in item["source_status_records"] if _has_query(r) and _query(r) == query)
    observed = status_result
    unavailable = observed is None or observed.get("ok") is False or observed.get("status") in ("error", "skipped")
    gaps = []
    result = {
        "item_id": item.get("item_id"), "split": item.get("split"), "category": item.get("category"),
        "benchmark_track": STATUS_TRACK, "item_variant": item.get("item_variant"),
        "scoring_variant": SCORING_VARIANT,
        "scoring_scope": "candidate_status_record_structural_match",
        "status_conclusion_match": None, "query_fidelity": None, "source_match": None,
        "expected_query": deepcopy(dict(query)),
        "observed_query": deepcopy(_query(observed)) if observed is not None else None,
        "expected_status": {f: deepcopy(reference[f]) for f in state_fields},
        "observed_status": {f: deepcopy(observed[f]) for f in state_fields if f in observed} if observed is not None else None,
        "expected_dates": {f: deepcopy(reference.get(f)) for f in _DATE_FIELDS},
        "observed_dates": {f: deepcopy(observed.get(f)) for f in _DATE_FIELDS} if observed is not None else None,
        "status_field_matches": {}, "source_groups": [], "unreferenced_source_ids": [], "gaps": gaps,
        "clause_table_metrics_applicable": False, "four_state_metrics_applicable": False,
        "semantic_correctness": None, "source_semantic_support": None,
        "semantic_review_pending": True, "formal_gold": False,
        "note": "仅比较候选登记快照；未命中仅指本地有限登记未唯一命中。来源结构匹配不是来源语义认证，整本现行不等于工程符合。",
    }
    if unavailable:
        gaps.append({"code": "status_result_unavailable"})
        return result
    if _has_query(observed):
        result["query_fidelity"] = _same(_query(observed), dict(query))
    else:
        gaps.append({"code": "observed_query_missing"})
    missing_fields = [f for f in state_fields if f not in observed]
    result["status_field_matches"] = {f: _same(reference[f], observed[f]) if f in observed else None for f in state_fields}
    if missing_fields:
        gaps.append({"code": "observed_status_fields_missing", "fields": missing_fields})
    else:
        result["status_conclusion_match"] = all(result["status_field_matches"].values())
    groups = item["status_evidence_groups"]
    if not groups:
        gaps.append({"code": "source_matching_not_applicable_no_reference_group"})
        return result
    if not _ids(observed.get("source_ids")):
        gaps.append({"code": "observed_source_ids_missing_or_invalid"})
        return result
    actual_ids = set(observed["source_ids"])
    expected_ids = set().union(*(set(group) for group in groups))
    result["unreferenced_source_ids"] = sorted(actual_ids - expected_ids)
    expected_sources, actual_sources = _sources(reference), _sources(observed)
    for group in groups:
        missing_ids = sorted(set(group) - actual_ids)
        mismatched, unavailable_ids = [], []
        for source_id in group:
            if source_id in missing_ids:
                continue
            expected = expected_sources.get(source_id, [])
            actual = actual_sources.get(source_id, [])
            if len(expected) != 1 or len(actual) != 1:
                unavailable_ids.append(source_id)
                continue
            fields = [f for f in _SOURCE_FIELDS if f in expected[0]]
            # A bare ID or only a prose description cannot verify source identity.
            if not expected[0].get("source_url") or "evidence_level" not in fields:
                unavailable_ids.append(source_id)
            elif any(f not in actual[0] for f in fields):
                unavailable_ids.append(source_id)
            elif not all(_same(expected[0][f], actual[0][f]) for f in fields):
                mismatched.append(source_id)
        matched = False if missing_ids or mismatched else None if unavailable_ids else True
        result["source_groups"].append({"source_ids": deepcopy(group), "match": matched,
                                         "missing_ids": missing_ids, "mismatched_ids": mismatched,
                                         "unavailable_record_ids": unavailable_ids})
    matches = [group["match"] for group in result["source_groups"]]
    result["source_match"] = True if True in matches else None if None in matches else False
    if any(group["unavailable_record_ids"] for group in result["source_groups"]):
        gaps.append({"code": "source_record_metadata_unavailable"})
    return result
