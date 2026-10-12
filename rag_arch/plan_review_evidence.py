"""Bounded rehydration of acquired evidence for plan drafting and review.

This module performs no model calls, reads no files and changes no acquisition
state. Character budgets include the compact JSON envelope. Source records and
acquired plan intervals are atomic: omissions never masquerade as complete text.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unicodedata

EVIDENCE_ASSEMBLY_VERSION = "plan-review-evidence-v1"


class EvidenceAssemblyBudgetError(ValueError):
    """Even the reference/omission manifest cannot fit without hiding data."""

    def __init__(self, message, *, minimum_chars=None):
        super().__init__(message)
        self.minimum_chars = minimum_chars


def evidence_payload_json(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _merge(pairs):
    result = []
    for start, end in sorted(pairs):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _acquired_ranges(document, provided_ranges):
    units = document.by_id
    result = {}
    if not isinstance(provided_ranges, dict):
        raise ValueError("provided_ranges must be a mapping")
    for uid, pairs in provided_ranges.items():
        if uid not in units or not isinstance(pairs, list):
            raise ValueError("Unknown plan unit or invalid acquired ranges")
        checked = []
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int for value in pair)
                    or not 0 <= pair[0] <= pair[1] <= len(units[uid]["text"])):
                raise ValueError("Acquired character range is outside the real source")
            if pair[0] < pair[1]:
                checked.append(list(pair))
        if checked:
            result[uid] = _merge(checked)
    return result


def _normalized(text):
    chars, positions = [], []
    for pos, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char):
            if not normalized.isspace():
                chars.append(normalized)
                positions.append(pos)
    return "".join(chars), positions


def _matches(text, quote):
    if not isinstance(quote, str) or not quote.strip():
        return []
    body, positions = _normalized(text)
    wanted, _ = _normalized(quote)
    if not wanted:
        return []
    result, offset = [], 0
    while True:
        found = body.find(wanted, offset)
        if found < 0:
            break
        span = [positions[found], positions[found + len(wanted) - 1] + 1]
        if span not in result:
            result.append(span)
        offset = found + 1
    return result


def _provided_matches(document, ranges, quote, uid=None):
    matches = []
    for unit in document.units:
        if uid is not None and unit["unit_id"] != uid:
            continue
        for start, end in ranges.get(unit["unit_id"], []):
            for left, right in _matches(unit["text"][start:end], quote):
                matches.append({"unit_id": unit["unit_id"], "text_start": start + left, "text_end": start + right})
    return matches


def _unique(values):
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _priority_ids(values, name):
    if values is None:
        return []
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in values)):
        raise ValueError(name + " must be a list or tuple of nonempty source IDs")
    return _unique(values)


def select_acquired_plan_units(document, provided_ranges, *, recent_units=(),
                               pending_units=(), limit=24):
    """Choose a bounded, source-spread replay set from acquired plan units.

    This helper is only a selection aid for rebuilding a model input window. It
    never expands ``provided_ranges`` and never treats the returned sample as a
    full-document reread or as proof that uncovered findings do not exist.
    Pending units are mandatory when they are real, non-empty acquired ranges.
    A small tail of recent units is retained, while the remaining slots are
    spread over the front, middle and end of the acquired source order. The
    returned order follows the document source order so packing is
    deterministic.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")

    ranges = _acquired_ranges(document, provided_ranges)
    ordered = [unit["unit_id"] for unit in document.units
               if unit["unit_id"] in ranges and any(
                   start < end for start, end in ranges[unit["unit_id"]])]
    if not ordered:
        return []

    acquired = set(ordered)
    pending = [uid for uid in _priority_ids(pending_units, "pending_units")
               if uid in acquired]
    recent = [uid for uid in _priority_ids(recent_units, "recent_units")
              if uid in acquired]

    # Keep only a small recent tail.  The source-spread pass below remains the
    # main selector, so a long recent list cannot monopolize the replay window.
    recent_quota = max(1, min(4, limit // 6))
    recent_tail = recent[-recent_quota:]
    mandatory = list(dict.fromkeys(pending + recent_tail))

    # Pending evidence is a hard retention requirement.  If callers give a
    # limit smaller than that set, returning the mandatory set is safer than
    # silently dropping newly acquired context; the limit still bounds the
    # representative remainder.
    selected = set(mandatory)
    if len(selected) < limit:
        remaining = [uid for uid in ordered if uid not in selected]
        slots = min(limit - len(selected), len(remaining))
        if slots:
            positions = []
            for number in range(slots):
                if slots == 1:
                    target = (len(ordered) - 1) // 2
                else:
                    target = round(number * (len(ordered) - 1) / (slots - 1))
                positions.append(target)
            index = {uid: position for position, uid in enumerate(ordered)}
            for target in positions:
                candidates = [uid for uid in remaining if uid not in selected]
                if not candidates:
                    break
                chosen = min(candidates,
                             key=lambda uid: (abs(index[uid] - target), index[uid]))
                selected.add(chosen)

    pending_set = set(pending)
    pending_order = [uid for uid in ordered if uid in pending_set]
    replay_candidates = [uid for uid in ordered
                         if uid in selected and uid not in pending_set]
    if len(replay_candidates) <= 2:
        replay_order = replay_candidates
    else:
        # Choose the first three replay items against the full source order,
        # rather than against the recent tail.  This keeps a short prefix
        # visibly spread across the document even when recent units include
        # several late sections.
        positions = {uid: index for index, uid in enumerate(ordered)}
        targets = [0, (len(ordered) - 1) // 2, len(ordered) - 1]
        replay_order, used = [], set()
        for target in targets:
            available = [uid for uid in replay_candidates if uid not in used]
            if not available:
                break
            chosen = min(available,
                         key=lambda uid: (abs(positions[uid] - target), positions[uid]))
            replay_order.append(chosen)
            used.add(chosen)
        replay_order.extend(uid for uid in replay_candidates if uid not in used)

    # Pending units are intentionally at the head: assemble_review_evidence
    # packs plan entries in this order and may omit a later atomic entry when
    # the character envelope is tight. Recent units are present in the
    # selected set but cannot monopolize this prefix.
    return pending_order + replay_order


def _related_standard_ids(ids, records):
    ordered = _unique(ids)
    for eid in ordered:
        record = records.get(eid, {})
        linked = record.get("linked_table_ids", [])
        if not isinstance(linked, list):
            raise ValueError("Linked table IDs must be an array")
        for related in _unique(linked + [record.get("owner_clause_id")]):
            if related not in ordered:
                ordered.append(related)
    return ordered


def _references(candidate, working_notes):
    standard_ids, unit_ids, refs = [], [], []
    if candidate is not None:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be the original report object")
        overview_ids = candidate.get("evidence_ids", [])
        if not isinstance(overview_ids, list):
            raise ValueError("Overview evidence IDs must be an array")
        standard_ids.extend(overview_ids)
        rows, origin = candidate.get("findings", []), "candidate"
    else:
        rows = working_notes.get("notes", []) if isinstance(working_notes, dict) else working_notes or []
        origin = "working_note"
    if not isinstance(rows, list):
        raise ValueError("findings/working notes must be an array")
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Each finding/working note must be an object")
        for key in ("standard_evidence_ids", "evidence_ids"):
            values = row.get(key, [])
            if not isinstance(values, list):
                raise ValueError("Evidence IDs must be arrays")
            standard_ids.extend(values)
        for key in ("plan_unit_ids", "context_unit_ids"):
            values = row.get(key, [])
            if not isinstance(values, list):
                raise ValueError("Plan unit IDs must be arrays")
            unit_ids.extend(values)
        raw_refs = row.get("plan_evidence", [])
        if not isinstance(raw_refs, list):
            raise ValueError("Plan references must be an array")
        for index, ref in enumerate(raw_refs, 1):
            if not isinstance(ref, dict):
                raise ValueError("Plan reference must be an object")
            uid, quote = ref.get("unit_id"), ref.get("quote")
            if not isinstance(uid, str) or not isinstance(quote, str):
                raise ValueError("Original unit ID and quote must be strings")
            refs.append({"reference_id": f"{origin}/{number}/{index}", "origin": origin,
                         "declared_unit_id": uid, "original_quote": quote,
                         "is_source_evidence": False})
            unit_ids.append(uid)
            context_ids = ref.get("context_unit_ids", [])
            if not isinstance(context_ids, list):
                raise ValueError("Context unit IDs must be an array")
            unit_ids.extend(context_ids)
    return _unique(standard_ids), _unique(unit_ids), refs


def _pool(payload, category, value):
    table = payload["shared_metadata"][category]
    for reference, existing in table.items():
        if evidence_payload_json(existing) == evidence_payload_json(value):
            return reference
    reference = f"{category[0]}{len(table) + 1}"
    table[reference] = deepcopy(value)
    return reference


_IDENTITY = {"kind", "std_no", "clause_no", "table_id", "owner_clause_id", "linked_table_ids"}
# Retrieval ranks and duplicated asset locations do not express requirements.
_EXCLUDED = {"score", "raw_rank", "origin", "candidate_path", "asset_refs", "source_page_images"}
_POSITION = {"page_start", "page_end", "printed_page", "block_start", "block_end",
             "source_line_start", "source_line_end", "candidate_line_start", "candidate_line_end",
             "page_line_start", "page_line_end", "role"}


def _pack_standard(payload, record, complete_ids):
    packed = {"evidence_id": record["evidence_id"], "text": record.get("text", ""),
              **{key: deepcopy(record[key]) for key in sorted(_IDENTITY) if key in record},
              "complete_context": record["evidence_id"] in complete_ids if complete_ids is not None else None}
    quality = {key: deepcopy(value) for key, value in record.items()
               if key not in _IDENTITY | _EXCLUDED | {"evidence_id", "text", "status_info", "source_spans"}}
    if quality:
        packed["quality_ref"] = _pool(payload, "quality", quality)
    if "status_info" in record:
        status = record["status_info"]
        if not isinstance(status, dict):
            raise ValueError("status_info must be an object")
        packed["status_info"] = {key: deepcopy(status[key]) for key in ("clause_no",) if key in status}
        packed["status_info"]["shared_ref"] = _pool(payload, "status", {key: value for key, value in status.items() if key != "clause_no"})
    if "source_spans" in record:
        if not isinstance(record["source_spans"], list):
            raise ValueError("source_spans must be an array")
        packed["source_spans"] = []
        for span in record["source_spans"]:
            if not isinstance(span, dict):
                raise ValueError("Source span must be an object")
            # Physical/printed pages suffice for model-facing PDF provenance;
            # derived block/line coordinates stay in the unchanged local record.
            # Non-paged sources retain line coordinates for precise retrieval.
            projected = {key: value for key, value in span.items()
                         if not (span.get("page_start") is not None and key in {
                             "block_start", "block_end", "source_line_start", "source_line_end",
                             "candidate_line_start", "candidate_line_end", "page_line_start", "page_line_end"})}
            packed["source_spans"].append({**{key: deepcopy(value) for key, value in projected.items() if key in _POSITION},
                "shared_ref": _pool(payload, "source", {key: value for key, value in projected.items() if key not in _POSITION})})
    return packed


def expand_standard_records(payload):
    """Reconstruct the documented model projection for local audit/testing."""
    result = []
    pools = payload["shared_metadata"]
    for packed in payload["standard_records"]:
        row = {key: deepcopy(value) for key, value in packed.items()
               if key not in {"complete_context", "quality_ref", "status_info", "source_spans"}}
        if "quality_ref" in packed:
            row.update(deepcopy(pools["quality"][packed["quality_ref"]]))
        if "status_info" in packed:
            status = packed["status_info"]
            row["status_info"] = {**deepcopy(pools["status"][status["shared_ref"]]),
                                  **{key: deepcopy(value) for key, value in status.items() if key != "shared_ref"}}
        if "source_spans" in packed:
            row["source_spans"] = [{**deepcopy(pools["source"][span["shared_ref"]]),
                **{key: deepcopy(value) for key, value in span.items() if key != "shared_ref"}}
                for span in packed["source_spans"]]
        result.append(row)
    return result


def assemble_review_evidence(document, evidence, provided_ranges, *, candidate=None,
                             max_chars=32000, working_notes=None, complete_ids=None,
                             priority_plan_unit_ids=None, priority_standard_ids=None):
    """Return compact-JSON-bounded source data; never mark requested text acquired.

    Pass the *raw* candidate, not validate_findings' surviving references. Notes
    can name source units and IDs, but their assertions never become evidence.
    complete_ids is the agent's authoritative complete-context set; omission
    leaves completeness unknown. The outer message/token budget remains the
    caller's responsibility. Metadata that cannot fit raises rather than lies.
    Explicit priority IDs (and their acquired standard dependencies) are packed
    before candidate-selected sources without adding/reordering candidate refs.
    Priority is only scheduling: it neither supplies missing text nor validates
    a model claim. None/empty priorities preserve the default ordering.
    """
    if type(max_chars) is not int or not 1000 <= max_chars <= 256000:
        raise ValueError("max_chars must be an integer between 1000 and 256000")
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be the current run's ID-to-record mapping")
    priority_units = _priority_ids(priority_plan_unit_ids, "priority_plan_unit_ids")
    priority_standards = _priority_ids(priority_standard_ids, "priority_standard_ids")
    records = deepcopy(evidence)
    for eid, record in records.items():
        if not isinstance(record, dict) or record.get("evidence_id") != eid or not isinstance(record.get("text", ""), str):
            raise ValueError("Invalid acquired evidence record")
    complete = set(complete_ids) if complete_ids is not None else None
    if complete is not None and not complete.issubset(records):
        raise ValueError("Complete-context IDs must have acquired records")
    ranges = _acquired_ranges(document, provided_ranges)
    units = document.by_id
    ordinals = {unit["unit_id"]: pos for pos, unit in enumerate(document.units, 1)}
    requested_standards, requested_units, refs = _references(candidate, working_notes)
    unresolved = []
    for ref in refs:
        uid, quote = ref["declared_unit_id"], ref["original_quote"]
        found = _provided_matches(document, ranges, quote, uid) if uid in units else []
        if found:
            ref["location_status"] = "matched_declared_acquired_range"
            ref["matches"] = found
        else:
            matches = _provided_matches(document, ranges, quote)
            if len(matches) == 1:
                ref["location_status"] = "unique_match_in_other_acquired_unit"
                ref["matches"] = matches
                requested_units.append(matches[0]["unit_id"])
            else:
                ref["location_status"] = "ambiguous_acquired_matches" if matches else "not_matched_in_acquired_ranges"
                ref["match_count"] = len(matches)
                unresolved.append({"kind": "plan_quote", "reference_id": ref["reference_id"],
                    "unit_id": uid, "reason": ref["location_status"],
                    "meaning": "定位失败不证明该事实或原文不存在；对照真实源段，必要时定点补读。"})
    requested_units = _unique(priority_units + requested_units)
    # Follow only explicit relationships in records actually acquired this run.
    # Dependencies of new counterevidence share its priority, so an old draft
    # cannot consume their space first. Unknown IDs remain diagnostic entries.
    priority_standard_order = _related_standard_ids(priority_standards, records)
    standard_order = _related_standard_ids(
        priority_standard_order + requested_standards + (list(records) if candidate is None else []), records)
    for eid in standard_order:
        if eid not in records:
            unresolved.append({"kind": "standard", "evidence_id": eid, "reason": "not_acquired_this_run"})
    standard_order = [eid for eid in standard_order if eid in records]
    if candidate is None:
        standard_order.extend(eid for eid in records if eid not in standard_order)
    selected_units = requested_units
    read_requests, plan_entries = [], []
    for uid in selected_units:
        if uid not in units:
            unresolved.append({"kind": "plan_unit", "unit_id": uid, "reason": "unknown_source_unit"})
            continue
        body, acquired = units[uid]["text"], ranges.get(uid, [])
        cursor, gaps = 0, []
        for start, end in acquired:
            if start > cursor: gaps.append([cursor, start])
            cursor = end
        if cursor < len(body): gaps.append([cursor, len(body)])
        for start, end in gaps:
            read_requests.append({"tool": "read_plan", "unit_id": uid,
                "arguments": {"start": ordinals[uid], "offset": start, "max_chars": 16000},
                "needed_range": [start, end], "reason": "referenced_unit_context_not_yet_provided"})
        if acquired:
            plan_entries.append({"unit_id": uid, "ordinal": ordinals[uid], "locator": deepcopy(units[uid]["locator"]),
                "source_path": units[uid]["source_path"], "unit_text_chars": len(body),
                "complete_unit_context": not gaps,
                "segments": [{"text_start": start, "text_end": end, "text": body[start:end]} for start, end in acquired],
                "unprovided_ranges": gaps,
                "meaning": "仅恢复已取得范围；整单元保留可见表头及条件，跨单元或图片表格上下文完整性仍未确认。"})
    source_index = {"standards": [{"evidence_id": eid, "text_chars": len(record.get("text", "")),
                    "complete_context": eid in complete if complete is not None else None}
                    for eid, record in records.items()],
                    "plan_acquired_unit_ids": list(ranges), "plan_total_units": len(document.units)}
    omissions = ([{"kind": "standard", "evidence_id": eid, "reason": "payload_budget", "already_acquired": True} for eid in standard_order]
                 + [{"kind": "plan", "unit_id": entry["unit_id"], "reason": "payload_budget", "already_acquired": True} for entry in plan_entries])
    payload = {"policy": EVIDENCE_ASSEMBLY_VERSION, "document_id": document.document_id,
        "source_revision": document.source_revision, "mode": "candidate_review" if candidate is not None else "candidate_drafting",
        "original_plan_references": refs, "standard_records": [], "plan_segments": [],
        "shared_metadata": {"quality": {}, "status": {}, "source": {}}, "source_index": source_index,
        "unresolved_references": unresolved, "read_requests": read_requests, "omitted": omissions,
        "budget_info": {"max_chars": max_chars, "serialized_chars": max_chars, "serialization": "compact_json_ensure_ascii_false",
                        "text_truncated_by_assembler": False, "complete_payload": False},
        "meaning": "本轮已取得资料的有界恢复，不是核查结论。quality_ref、status_info.shared_ref、source_spans.shared_ref分别查shared_metadata同名表。原始候选引句和工作笔记不当原文。遗漏资料不能视为不存在；完整记录不等于适用性或语义已核准。",
        "excluded_nonsemantic_record_fields": sorted(_EXCLUDED),
        "locator_projection": "有页码的来源保留原页/印刷页及来源标识；重复的派生块号行号仅在本地完整记录保留。无页码来源继续保留行号。"}
    if priority_units or priority_standards:
        payload["priority_selection"] = {"requested_plan_unit_ids": priority_units,
            "requested_standard_ids": priority_standards,
            "meaning": "仅决定本次装配顺序，不增加已取得范围，不改变候选引文编号或证明模型判断。"}
    manifest_chars = len(evidence_payload_json(payload))
    if manifest_chars > max_chars:
        raise EvidenceAssemblyBudgetError("引用与遗漏清单本身超过证据预算；请缩小候选范围，不得静默删除失败引文。",
                                          minimum_chars=manifest_chars)
    # Interleave requested plan units and standard records so one kind does not
    # consume the whole envelope before the other has a chance to fit.
    pending = []
    priority_unit_set, priority_standard_set = set(priority_units), set(priority_standard_order)
    for priority_tier in (True, False):
        tier_plans = [entry for entry in plan_entries if (entry["unit_id"] in priority_unit_set) == priority_tier]
        tier_standards = [eid for eid in standard_order if (eid in priority_standard_set) == priority_tier]
        for index in range(max(len(tier_standards), len(tier_plans))):
            if index < len(tier_plans): pending.append(("plan", tier_plans[index]))
            if index < len(tier_standards): pending.append(("standard", records[tier_standards[index]]))
    for kind, record in pending:
        trial = deepcopy(payload)
        identity = record["evidence_id"] if kind == "standard" else record["unit_id"]
        key = "evidence_id" if kind == "standard" else "unit_id"
        trial["omitted"] = [row for row in trial["omitted"] if not (row["kind"] == kind and row.get(key) == identity)]
        if kind == "standard": trial["standard_records"].append(_pack_standard(trial, record, complete))
        else: trial["plan_segments"].append(deepcopy(record))
        if len(evidence_payload_json(trial)) <= max_chars:
            payload = trial
    payload["budget_info"]["complete_payload"] = not payload["omitted"] and not payload["read_requests"] and not payload["unresolved_references"]
    payload["budget_info"]["serialized_chars"] = 0
    while True:
        size = len(evidence_payload_json(payload))
        if payload["budget_info"]["serialized_chars"] == size: break
        payload["budget_info"]["serialized_chars"] = size
    if size > max_chars:
        raise EvidenceAssemblyBudgetError("证据装配后的元数据超过预算")
    return payload
