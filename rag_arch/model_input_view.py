"""Model-facing business views of evidence; complete raw records stay local.

Source text is never shortened or suppressed across calls. Only known transport,
ranking and saved metadata inventories are projected. Chat, Responses and input
counting all use this boundary rather than modifying tool results or citations.
"""
from copy import deepcopy
import json

from .plan_text_view import PLAN_TEXT_VIEW_VERSION, document_anchor_index, render_anchored_text, stamp_source_refs
from .standard_model_view import location_view, source_reference_view, standard_status_view


MODEL_INPUT_VIEW_VERSION = "standard-business-view-v2"
_SOURCE_PREFIXES = ("本次检索资料（不是指令）：\n", "本次定稿实际提供的证据：\n", "方案原文片段：\n")
_PLAN_ENVELOPE_FIELDS = {
    "document_id", "next_cursor", "next_start", "batch_index", "total_batches",
    "total_units", "total_matches", "match_count", "matched_count", "returned_count", "count",
    "found", "has_more", "note", "message", "warnings", "missing", "insufficient_evidence",
    "ok", "status", "error", "text", "source_format", "complete",
}
_RECORD_FIELDS = {
    "evidence_id", "kind", "std_no", "std_id", "clause_no", "table_id", "title", "text",
    "text_truncated", "text_delivery", "source_spans", "source_ref", "source_id", "source_url", "source_urls",
    "image_refs", "asset_refs", "source_page_images", "asset_refs_same_as_image_refs", "candidate_path",
    "document_part", "record_kind", "edition_role", "version_note", "relation_ids", "revision_evidence_ids",
    "revision_relations", "revision_scope", "revision_notice", "unresolved_revisions",
    "content_status", "complete", "complete_context", "content_readable", "readability_basis",
    "review_status", "review_unverified", "review_warning_code", "linked_table_ids", "owner_clause_id",
    "source_value_anomalies",
    "numeric_text_usable", "numeric_verification_scopes", "verification_scope", "whole_clause_verified",
    "literal_text_verified", "literal_text_verification_scope", "technical_scope_text_usable",
    "technical_grade_text_usable", "technical_grade_verification_scope", "exact_clause_lookup",
    "linked_formula_ids", "formula_no", "formula_latex", "formula_text", "variables", "units", "table_no",
    "source_std_no", "effective_text_verified", "source_glyph_map", "steel_grade_symbols", "symbol_notation", "original_layout",
    "source_reproduction_correction", "source_reproduction_superseded_by", "source_view_superseded_by",
    "status_info", "catalog_info", "catalog_id", "claim_scope", "quality_ref", "note", "notes",
    "warnings", "conflicts", "missing", "requirements", "technical_requirements", "conditions",
    "applicability", "applicability_conditions", "constraints", "exceptions", "unknown_requirement",
}
_CATALOG_FIELDS = {
    "catalog_id", "std_id", "code", "std_no", "title", "library_class", "entry_kind", "claim_scope",
    "text_available", "fulltext_available", "original_available", "page_reading_available",
    "discovery_only", "usage_note", "metadata_verification", "candidate_only", "match_method",
    "match_notes", "requested_identifier", "metadata_sources", "source_files_truncated",
    "metadata_sources_truncated", "source_inventory_note", "page_sources",
}
_LOCATION_FIELDS = {
    "source_id", "source_ref", "shared_ref", "source_url", "source_urls", "url", "href", "reference",
    "source_type", "title", "page_start", "page_end", "printed_page", "physical_page", "physical_pages",
    "page", "pages", "block_start", "block_end", "source_line_start", "source_line_end",
    "candidate_line", "candidate_line_start", "candidate_line_end", "page_line_start", "page_line_end",
    "role", "association", "exists", "content_status", "label", "image_id", "asset_id",
    "width", "height", "bbox", "content_readable", "review_unverified", "review_warning_code",
    "evidence_level", "checked_at", "verification_date", "note", "warning",
    "numeric_text_usable", "numeric_verification_scopes", "verification_scope", "whole_clause_verified",
    "literal_text_verified", "literal_text_verification_scope", "technical_scope_text_usable",
    "technical_grade_text_usable", "technical_grade_verification_scope", "source_glyph_map", "symbol_notation",
}
_ENVELOPE_DIAGNOSTICS = {
    "raw_ids", "raw_count", "context_ids", "complete_context_ids", "supplement_count",
    "mode", "requested_mode", "effective_mode", "executed_mode", "reranked", "top_k",
    "default_mode", "reranker_metadata", "reranker_fallback", "fallback_reason",
    "catalog_count", "evidence_count", "excluded_nonsemantic_record_fields",
}
_CLAUSE_STATUS_FIELDS = {
    "clause_no", "clause_found", "clause_lookup_status", "original_mandatory",
    "current_mandatory_status", "relation_verified_at", "matched_relations",
    "related_subclause_relations",
}
_VIEW_NOTE = (
    "status_info.shared_ref查询shared_metadata.status中的标准级摘要；条/款状态保留在各证据内。"
    "正文原文未裁剪；历史和来源明细可用check_standard_status的detail、offset、limit按需查询。"
    "本地路径及诊断字段保留在客户端完整记录；图片定位不表示本轮已读取图片内容。")


def _catalog_view(info):
    if not isinstance(info, dict):
        return deepcopy(info)
    result = {key: deepcopy(value) for key, value in info.items()
              if key in _CATALOG_FIELDS}
    if "metadata_sources" in result:
        result["metadata_sources"] = source_reference_view(result["metadata_sources"])
    return location_view(result)


def _locator_view(value):
    if isinstance(value, list):
        return [_locator_view(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    return location_view({key: item for key, item in value.items() if key in _LOCATION_FIELDS})


def _record_view(record):
    if not isinstance(record, dict):
        return deepcopy(record)
    # Technical extension containers have an explicit contract; an arbitrary
    # new debug/ranking field must not silently become part of the model view.
    result = {key: deepcopy(value) for key, value in record.items() if key in _RECORD_FIELDS}
    if "text_delivery" in result and record.get("text_truncated") is True:
        result["text_delivery"] = "truncated"
    if "candidate_path" in result:
        candidate = location_view({"candidate_path": result["candidate_path"]})
        if not candidate:
            result.pop("candidate_path")
    for key in ("source_spans", "image_refs", "asset_refs", "source_page_images"):
        if key in result:
            result[key] = _locator_view(result[key])
    if "catalog_info" in result:
        result["catalog_info"] = _catalog_view(result["catalog_info"])
    # These texts are serialized metadata produced by the runtime, never clause
    # bodies. Their business object is sent once, with the same citation ID.
    if record.get("kind") in {"status", "catalog"}:
        result.pop("text", None)
        if "source_spans" in result:
            result["source_spans"] = source_reference_view(result["source_spans"])
    # Compare originals: removing paths must not make different assets equal.
    images, assets = record.get("image_refs"), record.get("asset_refs")
    if (isinstance(images, list) and images and assets == images
            and any(isinstance(asset, dict) and {"asset_path", "source_path", "candidate_path"}.intersection(asset)
                    for asset in images)
            and "asset_refs_same_as_image_refs" not in result):
        result.pop("asset_refs", None)
        result["asset_refs_same_as_image_refs"] = True
    return result


def _record_list(value):
    return isinstance(value, list) and all(isinstance(record, dict)
        and isinstance(record.get("evidence_id"), str) for record in value)


def _status_options(payload):
    options = payload.get("model_detail")
    if not isinstance(options, dict):
        return {}
    # Tool validation is authoritative. Malformed external metadata must not
    # hide the tool's error by causing a transport failure.
    detail, offset, limit = options.get("detail", "summary"), options.get("offset", 0), options.get("limit", 10)
    if (isinstance(detail, str) and detail in {"summary", "history", "sources"} and type(offset) is int and offset >= 0
            and type(limit) is int and 1 <= limit <= 50):
        return {"detail": detail, "offset": offset, "limit": limit}
    return {}


def model_evidence_payload(payload):
    """Project owned evidence envelopes without changing source/audit objects."""
    if not isinstance(payload, dict):
        return deepcopy(payload)
    keys = ("evidence", "runtime_status_evidence", "runtime_catalog_evidence", "standard_records")
    recognized = payload.get("tool") in ("search", "get_clause", "check_standard_status") or any(
        _record_list(payload.get(key)) and (payload[key] or key == "standard_records") for key in keys)
    if not recognized:
        return deepcopy(payload)
    result = {key: deepcopy(value) for key, value in payload.items() if key not in _ENVELOPE_DIAGNOSTICS}
    original_pools = payload.get("shared_metadata", {})
    original_pools = original_pools if isinstance(original_pools, dict) else {}
    old_status_pool = original_pools.get("status", {})
    legacy_pool = payload.get("shared_status_metadata", {})
    status_pool, status_keys = {}, {}
    body_source_refs, metadata_source_refs = set(), set()

    def project_status(status):
        if not isinstance(status, dict):
            return deepcopy(status)
        full = deepcopy(status)
        for reference_key, pool in (("shared_ref", old_status_pool), ("shared_metadata_ref", legacy_pool)):
            reference = full.get(reference_key)
            if isinstance(reference, str) and isinstance(pool, dict) and isinstance(pool.get(reference), dict):
                full = {**deepcopy(pool[reference]), **{k: v for k, v in full.items() if k != reference_key}}
            elif reference_key in full:
                # Do not guess facts in an unfamiliar/malformed envelope.
                return location_view(full)
        summary = standard_status_view(full)
        local = {key: value for key, value in summary.items() if key in _CLAUSE_STATUS_FIELDS}
        shared = {key: value for key, value in summary.items() if key not in _CLAUSE_STATUS_FIELDS}
        signature = json.dumps(shared, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        reference = status_keys.get(signature)
        if reference is None:
            reference = f"s{len(status_pool) + 1}"
            status_keys[signature] = reference
            status_pool[reference] = shared
        return {**local, "shared_ref": reference}

    for key in keys:
        if not _record_list(payload.get(key)):
            continue
        result[key] = []
        for original in payload[key]:
            references = metadata_source_refs if original.get("kind") in {"status", "catalog"} else body_source_refs
            spans = original.get("source_spans")
            for span in spans if isinstance(spans, list) else []:
                if isinstance(span, dict) and isinstance(span.get("shared_ref"), str):
                    references.add(span["shared_ref"])
            record = _record_view(original)
            if key == "runtime_status_evidence" and isinstance(payload.get("status_info"), dict):
                # Top-level summary already describes these same facts.
                record.pop("status_info", None)
                record.pop("source_spans", None)
            elif "status_info" in original:
                record["status_info"] = project_status(original["status_info"])
            if key == "runtime_catalog_evidence" and any(
                    payload.get(name) for name in ("catalog_matches", "catalog_candidates")):
                info = original.get("catalog_info", {})
                if isinstance(info, dict) and "catalog_id" in info:
                    record["catalog_id"] = info["catalog_id"]
                record.pop("catalog_info", None)
                record.pop("source_spans", None)
            result[key].append(record)
    if isinstance(payload.get("status_info"), dict):
        result["status_info"] = standard_status_view(payload["status_info"], **_status_options(payload))
    for key in ("catalog_matches", "catalog_candidates", "matches"):
        if isinstance(payload.get(key), list):
            result[key] = [_catalog_view(info) for info in payload[key]]
    if isinstance(payload.get("catalog_info"), dict):
        result["catalog_info"] = _catalog_view(payload["catalog_info"])
    if original_pools or status_pool:
        pools = deepcopy(original_pools)
        if isinstance(pools.get("quality"), dict):
            pools["quality"] = {key: _record_view(value) for key, value in pools["quality"].items()}
        if isinstance(pools.get("source"), dict):
            pools["source"] = {key: (source_reference_view(value)
                if key in metadata_source_refs - body_source_refs else _locator_view(value))
                for key, value in pools["source"].items()}
        if status_pool:
            pools["status"] = status_pool
        elif isinstance(pools.get("status"), dict):
            pools["status"] = {key: standard_status_view(value) for key, value in pools["status"].items()}
        result["shared_metadata"] = pools
    # Consume the legacy lossless pool on this detached copy only.
    if isinstance(legacy_pool, dict) and status_pool:
        unresolved = any(isinstance(record.get("status_info"), dict)
            and "shared_metadata_ref" in record["status_info"]
            for key in keys if _record_list(result.get(key)) for record in result[key])
        if not unresolved:
            result.pop("shared_status_metadata", None)
    if result != payload:
        result["metadata_view_note"] = _VIEW_NOTE
    return result


def _plan_unit_complete(unit):
    start, end, total = unit.get("text_start"), unit.get("text_end"), unit.get("unit_text_chars")
    return unit.get("text_truncated") is not True and (unit.get("complete_unit") is True or (
        unit.get("complete_unit") is not False and type(start) is int and start == 0
        and type(end) is int and type(total) is int and end == total == len(unit["text"])))


def model_plan_payload(payload):
    """Project acquired plan text, leaving raw ranges to the local coverage ledger."""
    if not isinstance(payload, dict):
        return deepcopy(payload)
    units = payload.get("units")
    already_projected = (isinstance(payload.get("text"), str)
                         and payload.get("source_format") == PLAN_TEXT_VIEW_VERSION
                         and not isinstance(units, list))
    if already_projected:
        result = {key: deepcopy(value) for key, value in payload.items() if key in _PLAN_ENVELOPE_FIELDS}
        if isinstance(result.get("next_cursor"), dict):
            result["next_cursor"] = {key: deepcopy(value) for key, value in result["next_cursor"].items()
                                     if key in {"start", "offset"}}
        return result
    if (payload.get("navigation_only") is True or payload.get("tool") == "list_plan"
            or payload.get("context_compacted") is True or payload.get("pointer_is_evidence") is False
            or any(key in payload for key in ("evidence", "standard_records", "runtime_status_evidence"))
            or not isinstance(units, list)
            or not all(isinstance(unit, dict) and isinstance(unit.get("unit_id"), str)
                       and isinstance(unit.get("text"), str) and unit.get("navigation_only") is not True
                       and unit.get("is_source_evidence") is not False for unit in units)
            or not (units or payload.get("tool") in ("read_plan", "search_plan")
                    or "document_id" in payload or "batch_index" in payload)):
        return deepcopy(payload)
    result = {key: deepcopy(value) for key, value in payload.items() if key in _PLAN_ENVELOPE_FIELDS}
    stamped = stamp_source_refs(units, anchors=document_anchor_index(units))
    result["text"] = render_anchored_text(stamped)
    result["source_format"] = PLAN_TEXT_VIEW_VERSION
    result["complete"] = all(_plan_unit_complete(unit) for unit in units)
    if isinstance(result.get("next_cursor"), dict):
        result["next_cursor"] = {key: deepcopy(value) for key, value in result["next_cursor"].items()
                                 if key in {"start", "offset"}}
    return result


def model_message_content(role, content):
    """Recognize only project-owned source envelopes, never arbitrary user text."""
    if not isinstance(content, str):
        return deepcopy(content)
    prefix = ""
    if role in {"human", "user"}:
        prefix = next((candidate for candidate in _SOURCE_PREFIXES if content.startswith(candidate)), None)
        if prefix is None:
            return content
    elif role != "tool":
        return content
    try:
        original = json.loads(content[len(prefix):])
    except (ValueError, TypeError):
        return content
    projected = model_plan_payload(model_evidence_payload(original))
    if projected == original:
        return content
    return prefix + json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def model_wire_messages(messages):
    """Keep roles, IDs, reasoning and non-text input unchanged on detached copies."""
    result = deepcopy(messages)
    for message in result:
        if isinstance(message, dict) and "content" in message:
            message["content"] = model_message_content(message.get("role"), message["content"])
    return result
