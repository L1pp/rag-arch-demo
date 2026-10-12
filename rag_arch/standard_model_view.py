"""Business facts for model use; saved status records remain the audit source.

This is a field projection, never a new status inference. In particular, an
amendment, a former mandatory-clause repeal and a whole replacement stay distinct.
"""
from copy import deepcopy
import json


_LOCAL_PATHS = {"source_path", "candidate_path", "asset_path", "path", "href"}
_SOURCE_FIELDS = {
    "source_id", "source_ref", "shared_ref", "source_url", "source_urls", "url", "urls", "reference",
    "source_type", "title", "evidence_level", "checked_at", "verified_at", "verification_date",
    "verification_status", "verification_state", "observed_status", "supports_current_status",
    "publish_date", "effective_date", "page_start", "page_end", "printed_page",
    "physical_page", "physical_pages", "page", "pages", "relation_id", "evidence_id",
}
_RELATION_GROUPS = ("confirmed_version_relations", "version_relation_candidates",
                    "version_relations_non_successor", "revision_relations", "lifecycle_facts")
_SUMMARY_FIELDS = {
    "found", "std_no", "query_std_no", "requested_std_no", "title", "doc_id", "clause_no",
    "registered_status", "registered_status_checked_at", "standard_status", "standard_status_evidence_level",
    "registry_observed_status", "verification_date", "status_checked_at", "status_as_of",
    "current_effective_date", "publish_date", "effective_date", "abolished_effective_date", "abolished_date_known",
    "original_mandatory", "current_mandatory_status", "full_text_mandatory_at_publication",
    "clause_found", "clause_lookup_status", "relation_verified_at", "requires_status_recheck",
    "compliance_conclusion", "complete_registry_coverage", "latest_published_scope", "metadata_verification",
    "in_corpus", "in_catalog", "text_available", "fulltext_available", "original_available",
    "page_reading_available", "library_class", "source_ids", "source_urls", "source_ref",
    "note", "notes", "catalog_notes", "warnings", "custom_warning",
    "has_registered_mandatory_clause_repeals",
}
_VERSION_FIELDS = {
    "std_no", "std_id", "code", "title", "publish_date", "effective_date", "current_effective_date",
    "standard_status", "standard_status_evidence_level", "status_checked_at", "requires_status_recheck",
    "source_ids", "source_ref", "source_url", "verification_date",
}
_VERSION_GROUPS = ("current_effective_version", "current_effective_versions", "latest_published_version",
    "latest_published_versions", "upcoming_versions", "known_effective_successors",
    "known_future_successors", "whole_replacement_successors", "superseded_by")
_CONFLICT_FIELDS = {"kind", "field", "values", "observed_statuses", "source_ids", "source_urls",
                    "relation_ids", "event_date", "message", "note", "requires_status_recheck"}
_MATCHED_FIELDS = {"std_no", "clause_no", "match_scope", "relation_type", "abolished_by", "abolished_date",
    "whole_standard_abolished", "clause_text_invalidated", "source_id", "source_ref", "source_url",
    "verified_at", "verification_date", "evidence_level", "note", "notes"}
_SCOPE_FIELDS = {"relation_id", "relation", "source_code", "target_code", "source_std_id", "target_std_id",
    "source_part_id", "target_source_id", "scope", "whole_clause", "clause_scope", "clauses",
    "target_clause_nos", "target_page_ranges", "target_page_scope", "affected_clause_nos",
    "relation_confirmed", "whole_standard_successor_allowed", "effective_date", "target_effective_date",
    "abolished_date", "mandatory_clause_effective_date"}
_CHANGE_FLAGS = {"has_local_revisions", "has_former_mandatory_repeals", "has_unconfirmed_relations",
                 "has_non_successor_relations"}
_LOCAL_CHANGE_TYPES = {"amends", "amendment", "partially_replaces", "corrects", "corrigendum", "errata"}
_MANDATORY_CHANGE_TYPES = {"repeals_mandatory_clauses", "former_mandatory_clause_abolished"}
_IDENTITY_FIELDS = {"query", "status", "canonical_code", "canonical_std_id", "match_method",
                    "candidate_codes", "candidate_standards", "std_no", "title", "std_id"}
_ALIAS_FIELDS = {"alias_code", "canonical_code", "canonical_std_id", "relation_confirmed", "verification_state"}
_HISTORY_COVERAGE_FIELDS = {"complete_history_verified", "status", "all_known_records_integrated",
                           "active_followup", "notes", "warnings", "requires_recheck",
                           "review_state", "verification_status", "verification_state",
                           "requires_status_recheck"}


def location_view(value):
    """Drop only local paths, leaving page numbers and source references literal."""
    if isinstance(value, list):
        return [location_view(item) for item in value]
    if isinstance(value, dict):
        return {key: location_view(item) for key, item in value.items()
                if key not in _LOCAL_PATHS or (isinstance(item, str)
                    and item.lower().startswith(("http://", "https://")))}
    return deepcopy(value)


def source_reference_view(value):
    if isinstance(value, list):
        return [source_reference_view(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    return location_view({key: item for key, item in value.items() if key in _SOURCE_FIELDS})


def _pick(value, fields):
    if isinstance(value, list):
        return [_pick(item, fields) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    return location_view({key: item for key, item in value.items() if key in fields})


def _conflicts(value):
    if isinstance(value, list):
        return [_conflicts(item) for item in value]
    result = _pick(value, _CONFLICT_FIELDS)
    if isinstance(value, dict) and "registry_observations" in value:
        result["registry_observations"] = source_reference_view(value["registry_observations"])
    return result


def _version_view(value):
    if isinstance(value, list):
        return [_version_view(item) for item in value]
    result = _pick(value, _VERSION_FIELDS)
    if isinstance(value, dict) and "status_conflicts" in value:
        result["status_conflicts"] = _conflicts(value["status_conflicts"])
    return result


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _standard_key(value):
    return "".join(str(value or "").split()).upper().replace("－", "-").replace("—", "-")


def _change_scope(info):
    """A small applicability warning, not a disguised copy of the history graph."""
    if isinstance(info.get("change_scope"), dict):
        saved = info["change_scope"]
        return {**_pick(saved, _CHANGE_FLAGS), "scoped_changes": _pick(saved.get("scoped_changes", []), _SCOPE_FIELDS)}
    groups = {group: _rows(info.get(group)) for group in ("version_relations", *_RELATION_GROUPS)}
    relations = [row for group, rows in groups.items() if group != "lifecycle_facts" for row in rows]
    if not relations:
        return None
    types = {row.get("relation") for row in relations if isinstance(row.get("relation"), str)}
    result = {
        "has_local_revisions": bool(types & _LOCAL_CHANGE_TYPES or groups["revision_relations"]),
        "has_former_mandatory_repeals": bool(types & _MANDATORY_CHANGE_TYPES),
        "has_unconfirmed_relations": bool(groups["version_relation_candidates"]),
        "has_non_successor_relations": bool(groups["version_relations_non_successor"]),
        "scoped_changes": [],
    }
    # Only show scopes that actually affect this queried standard. For repeal
    # edges the old/source standard is affected, not the new notice's clauses.
    identities = {_standard_key(info.get(field)) for field in ("std_no", "doc_id")} - {""}
    for row in relations:
        kind = row.get("relation")
        if kind in _MANDATORY_CHANGE_TYPES:
            affected = [row.get(field) for field in ("source_code", "old_code", "source_std_id", "std_no")]
        elif kind in _LOCAL_CHANGE_TYPES or row in groups["revision_relations"]:
            affected = [row.get(field) for field in ("target_code", "new_code", "target_std_id", "std_no")]
        else:
            continue
        if identities.isdisjoint({_standard_key(value) for value in affected} - {""}):
            continue
        scope = _pick(row, _SCOPE_FIELDS)
        for old, new in (("old_code", "source_code"), ("new_code", "target_code")):
            if new not in scope and row.get(old):
                scope[new] = deepcopy(row[old])
        if scope not in result["scoped_changes"]:
            result["scoped_changes"].append(scope)
    return result


def _detail_rows(info, detail):
    rows, seen = [], set()

    def add(kind, value, **context):
        values = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        for record in values:
            if not isinstance(record, dict):
                continue
            item = {"record_type": kind, **context, "record": location_view(record)}
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                rows.append(item)

    if detail == "history":
        for group in ("version_relations", *_RELATION_GROUPS, "identity_relations",
                      "matched_relations", "related_subclause_relations"):
            kind = "version_relation" if group in {
                "version_relations", "confirmed_version_relations", "version_relation_candidates",
                "version_relations_non_successor"} else group
            add(kind, info.get(group, []))
        chain = info.get("version_chain")
        if isinstance(chain, dict):
            add("version_relation", chain.get("edges", []))
        add("status_observation", info.get("status_observation_history", []), std_no=info.get("std_no"))
        for version in info.get("version_statuses", []):
            if isinstance(version, dict):
                add("version_status", {key: value for key, value in version.items()
                                       if key != "status_observation_history"})
                add("status_observation", version.get("status_observation_history", []), std_no=version.get("std_no"))
        coverage = info.get("history_coverage")
        if isinstance(coverage, dict):
            add("status_observation", coverage.get("status_observations", []), std_no=info.get("std_no"))
        review = info.get("history_review")
        if isinstance(review, dict):
            for field, value in review.items():
                for item in value if isinstance(value, list) else [value]:
                    add("history_review", {"field": field, "value": item})
        elif isinstance(review, list):
            add("history_review", review)
    else:
        for group in ("status_sources", "sources", "metadata_sources"):
            add("source", info.get(group, []))
        for version in info.get("version_statuses", []):
            if isinstance(version, dict):
                add("source", version.get("status_sources", []))
                add("source", version.get("status_observation_history", []))
        add("source", info.get("status_observation_history", []))
        coverage = info.get("history_coverage")
        if isinstance(coverage, dict):
            add("source", coverage.get("status_observations", []))
        for group in ("version_relations", *_RELATION_GROUPS, "identity_relations",
                      "matched_relations", "related_subclause_relations"):
            for relation in info.get(group, []):
                if isinstance(relation, dict):
                    for field in ("evidence", "sources"):
                        add("source", relation.get(field, []))
                    if relation.get("source_id") or relation.get("source_url"):
                        add("source", source_reference_view(relation))
    return rows


def standard_status_view(info, *, detail="summary", offset=0, limit=10):
    """Return current business facts and, only on request, a page of saved detail.

    No body text is shortened. ``history``/``sources`` pages select whole metadata
    records; the caller retains the full unmodified status object for validation.
    """
    if not isinstance(detail, str) or detail not in {"summary", "history", "sources"}:
        raise ValueError("detail must be summary, history or sources")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 50")
    if not isinstance(info, dict):
        return deepcopy(info)
    result = _pick(info, _SUMMARY_FIELDS)
    for field in _VERSION_GROUPS:
        if field in info:
            result[field] = _version_view(info[field])
    for field in ("status_conflicts", "standard_status_conflicts"):
        if field in info:
            result[field] = _conflicts(info[field])
    for field in ("history_coverage", "history_review"):
        if isinstance(info.get(field), dict):
            result[field] = _pick(info[field], _HISTORY_COVERAGE_FIELDS)
    identity = info.get("identity_resolution")
    if isinstance(identity, dict):
        result["identity_resolution"] = _pick(identity, _IDENTITY_FIELDS)
        if "relations" in identity:
            result["identity_resolution"]["relations"] = _pick(identity["relations"], _ALIAS_FIELDS)
    change_scope = _change_scope(info)
    if change_scope is not None:
        result["change_scope"] = change_scope
    sources = info.get("source_refs") or info.get("status_sources") or info.get("sources") or info.get("metadata_sources")
    if isinstance(sources, list):
        references = []
        for source in source_reference_view(sources):
            if source and source not in references:
                references.append(source)
        if references:
            result["source_refs"] = references
    # Relevant edition warnings survive without sending every version's state
    # and observation history. The current and future identities stay distinct.
    versions = _rows(info.get("version_warnings", info.get("version_statuses")))
    if versions:
        warnings = [_version_view(row) for row in versions
                    if row.get("requires_status_recheck") or row.get("status_conflicts")]
        if warnings:
            result["version_warnings"] = warnings
    # A full standard lookup contains the entire former-mandatory repeal list.
    # Clause lookups retain each matching clause/subclause and its exact scope.
    if "clause_no" in info and info["clause_no"] is None:
        for field in ("matched_relations", "related_subclause_relations"):
            if info.get(field):
                result.pop(field, None)
                result["has_registered_mandatory_clause_repeals"] = True
    else:
        for field in ("matched_relations", "related_subclause_relations"):
            if field in info:
                result[field] = _pick(info[field], _MATCHED_FIELDS)
    result["status_scope_note"] = (
        "整本状态、原强条身份及条/款废止分别判断；局修或部分废止不自动构成整本替代。"
        "未知不表示现行；未来实施版不覆盖当前有效版，仍须按核验日期与工程适用范围使用。")
    result["detail_available"] = {"tool": "check_standard_status", "details": ["history", "sources"]}
    if detail != "summary":
        existing = info.get("detail_page")
        if isinstance(existing, dict) and all(existing.get(k) == v for k, v in
                (("detail", detail), ("offset", offset), ("limit", limit))):
            result["detail_page"] = deepcopy(existing)
        else:
            rows = _detail_rows(info, detail)
            end = min(offset + limit, len(rows))
            result["detail_page"] = {"detail": detail, "offset": offset, "limit": limit,
                "total": len(rows), "items": rows[offset:offset + limit],
                "has_more": end < len(rows), "next_offset": end if end < len(rows) else None}
    return result
