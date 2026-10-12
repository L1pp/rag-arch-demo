"""Summarize received benchmark reviews without providers or Gold promotion.

Human question-quality review is separate from answer/evidence review. Legacy
human source reviews remain readable. External AI records describe what was
actually received; neither the number of models nor rounds is a pass threshold.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy


EVALUATION_SCOPES = ("candidate_trial", "human_review_recorded", "external_ai_review_recorded")
EXTERNAL_REVIEW_STATUSES = ("pending", "completed", "failed")
EVIDENCE_GROUP_APPLICABILITIES = ("applicable", "not_applicable", "unknown")


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _binary_status(values):
    values = list(values)
    if not values:
        return "not_reviewed"
    if True in values and False in values:
        return "conflict"
    if False in values:
        return "changes_requested"
    if None in values:
        return "unknown"
    return "supported"


def _judgment_summary(records, invalid=False):
    """Positive clause/table-group applicability is separate from answer support.

    Legacy receipts without an applicability field retain their historical
    applicable-group interpretation. Only an explicit N/A marker changes it;
    no bool/null value is rewritten to make a record pass.
    """
    answer_status = _binary_status(r["answer_correct"] for r in records)
    applicability_values = {r.get("evidence_group_applicability", "applicable") for r in records}
    applicability = (next(iter(applicability_values)) if len(applicability_values) == 1 else
                     "conflict" if applicability_values else "not_reviewed")
    if applicability == "not_applicable":
        evidence_status = "not_applicable"
    elif applicability in {"unknown", "conflict", "not_reviewed"}:
        evidence_status = applicability
    else:
        evidence_status = _binary_status(r["evidence_complete"] for r in records)
    if invalid:
        if answer_status == "supported":
            answer_status = "unknown"
        if evidence_status == "supported":
            evidence_status = "unknown"
    conflict = "conflict" in {answer_status, evidence_status, applicability}
    opposition = any(r["answer_correct"] is False or
                     (r.get("evidence_group_applicability", "applicable") == "applicable"
                      and r["evidence_complete"] is False) for r in records)
    unresolved = bool(invalid or "unknown" in {answer_status, evidence_status, applicability})
    answer_supported = bool(records and not invalid and answer_status == "supported")
    evidence_supported = (None if evidence_status == "not_applicable" else
                          bool(records and not invalid and evidence_status == "supported"))
    supported = answer_supported and evidence_status in {"supported", "not_applicable"} and not invalid
    status = ("conflict" if conflict else "changes_requested" if opposition else "unresolved" if unresolved
              else "supported" if supported else "not_reviewed")
    return {"supported": supported, "support_status": status,
            "answer_supported": answer_supported, "answer_support_status": answer_status,
            "evidence_supported": evidence_supported, "evidence_support_status": evidence_status,
            "evidence_group_applicability": applicability,
            "support_scope": "answer_and_applicable_positive_clause_table_groups",
            "has_conflict": conflict, "has_opposition": opposition, "has_unresolved_judgments": unresolved}


def _summary_adjudication(item, reviews):
    candidate = reviews.get("summary_adjudication") if isinstance(reviews, Mapping) else None
    if candidate is None and isinstance(reviews, list):
        candidates = [record for record in reviews if isinstance(record, Mapping)
                      and record.get("reviewer_type") == "synthesis"]
        candidate = candidates[-1] if len(candidates) == 1 else None
    if candidate is None:
        return None, None
    required_match = all(candidate.get(key) == item.get(key) for key in ("item_id", "split", "item_variant")) if isinstance(candidate, Mapping) else False
    valid = (isinstance(candidate, Mapping) and required_match
             and all(_nonempty(candidate.get(key)) for key in ("item_id", "split", "item_variant"))
             and candidate.get("status") == "completed" and candidate.get("reviewer_type") == "synthesis"
             and candidate.get("independent_review") is False and _nonempty(candidate.get("reviewer"))
             and candidate.get("evidence_group_applicability") in EVIDENCE_GROUP_APPLICABILITIES
             and all(key in candidate and (candidate[key] is None or type(candidate[key]) is bool)
                     for key in ("answer_correct", "evidence_complete"))
             and isinstance(candidate.get("basis"), list) and bool(candidate["basis"]))
    if not valid:
        return None, {"code": "invalid_summary_adjudication", "field": "review_results.summary_adjudication",
                      "message": "综合裁决身份、当前题目关联或维度字段无效；保留原回执，不作为当前采用结论"}
    summary = _judgment_summary([candidate])
    revisions = deepcopy(candidate.get("required_revisions", []))
    revision_status = candidate.get("revision_status", "pending" if revisions else "not_required")
    if revisions and revision_status != "completed" and summary["supported"]:
        summary["support_status"] = "supported_with_revisions"
    summary.update(source="same_round_summary_adjudication", independent_review=False,
                   current_variant=item.get("item_variant"), required_revisions=revisions, revision_status=revision_status,
                   record=deepcopy(candidate), meaning="采用层为同轮综合复核，原模型回执仍保留在外审历史；不是新增独立模型或审阅轮次。")
    return summary, None


def assess_benchmark_reviews(item: Mapping) -> dict:
    """Inspect review_results; do not edit the item or certify its semantics.

    A list identifies external records with reviewer_type='external_ai'. A
    mapping may instead use external_ai=[...]. Required fields are reviewer,
    model, round (positive integer), status, answer_correct, evidence_complete,
    and item_variant. Judgments are bool or null; null means unresolved. Optional
    metadata and opinions are retained unchanged. Only exact current-variant
    completed records count as received reviews. Per reviewer/model, the highest
    completed round supplies the current judgment; pending/failed calls cannot
    erase a previously received result. Conflicting records at that same round
    remain visible rather than being resolved by input order.
    """
    reviews = item.get("review_results")
    diagnostics = []
    if isinstance(reviews, Mapping):
        human = reviews.get("human")
        external = reviews.get("external_ai", [])
        if not isinstance(external, list):
            diagnostics.append({"code": "invalid_external_ai_reviews", "field": "review_results.external_ai",
                                "message": "external_ai 需为实际审阅记录数组"})
            external = []
    elif isinstance(reviews, list):
        human = next((r for r in reversed(reviews) if isinstance(r, Mapping)
                      and r.get("reviewer_type") == "human"), None)
        external = [r for r in reviews if isinstance(r, Mapping) and r.get("reviewer_type") == "external_ai"]
    else:
        human, external = None, []
    human_recorded = (isinstance(human, Mapping) and human.get("status") == "approved"
                      and _nonempty(human.get("reviewer")) and _nonempty(human.get("reviewed_at"))
                      and human.get("source_checked") is True)
    variant = item.get("item_variant")
    current, invalid_current, stale_count = [], [], 0
    for index, record in enumerate(external):
        field = f"review_results.external_ai[{index}]"
        if not isinstance(record, Mapping):
            diagnostics.append({"code": "invalid_external_ai_review", "field": field,
                                "message": "审阅记录需为对象"})
            continue
        invalid = [key for key in ("reviewer", "model", "item_variant") if not _nonempty(record.get(key))]
        if type(record.get("round")) is not int or record["round"] < 1:
            invalid.append("round")
        if record.get("status") not in EXTERNAL_REVIEW_STATUSES:
            invalid.append("status")
        for key in ("answer_correct", "evidence_complete"):
            if key not in record or (record[key] is not None and type(record[key]) is not bool):
                invalid.append(key)
        if ("evidence_group_applicability" in record
                and record["evidence_group_applicability"] not in EVIDENCE_GROUP_APPLICABILITIES):
            invalid.append("evidence_group_applicability")
        if record.get("reviewer_type", "external_ai") != "external_ai":
            invalid.append("reviewer_type")
        matches = _nonempty(variant) and record.get("item_variant") == variant
        if invalid:
            diagnostics.append({"code": "invalid_external_ai_review", "field": field,
                                "message": "外部AI审阅字段无效", "invalid_fields": invalid})
            if matches:
                invalid_current.append(record)
            continue
        if not matches:
            stale_count += 1
            continue
        current.append(record)
    completed = [r for r in current if r["status"] == "completed"]
    by_reviewer = defaultdict(list)
    for record in completed:
        by_reviewer[(record["reviewer"], record["model"])].append(record)
    latest = []
    for records in by_reviewer.values():
        last_round = max(r["round"] for r in records)
        latest.extend(r for r in records if r["round"] == last_round)
    # Pending/failed placeholders are not received opinions. A later valid
    # completed round may also supersede a malformed older receipt while its
    # diagnostic remains in the history.
    unresolved_invalid = [r for r in invalid_current if r.get("status") == "completed" and not any(
        r.get("reviewer") == valid["reviewer"] and r.get("model") == valid["model"]
        and type(r.get("round")) is int and r["round"] < valid["round"] for valid in latest)]
    judgments = _judgment_summary(latest, bool(unresolved_invalid))
    adopted, adoption_diagnostic = _summary_adjudication(item, reviews)
    if adoption_diagnostic:
        diagnostics.append(adoption_diagnostic)
    if judgments["has_conflict"] or judgments["has_opposition"] or judgments["has_unresolved_judgments"]:
        diagnostics.append({"code": "external_ai_review_unresolved", "field": "review_results.external_ai",
                            "message": ("原模型回执存在反对、分歧或未定项；已由同轮综合裁决给出采用结论，原意见保留追查"
                                        if adopted else "最新外审仍有反对、分歧或未定项；保留实际记录，不能据此宣称已通过"),
                            "support_status": judgments["support_status"],
                            "resolved_for_adoption_by_summary": adopted is not None})
    external_recorded = bool(completed)
    scope = ("external_ai_review_recorded" if external_recorded else
             "human_review_recorded" if human_recorded else "candidate_trial")
    summary = {
        "record_count": len(external), "records": deepcopy(external),
        "current_variant": variant, "current_variant_record_count": len(current),
        "other_variant_record_count": stale_count, "invalid_current_record_count": len(invalid_current),
        "unresolved_invalid_current_record_count": len(unresolved_invalid),
        "current_status_counts": dict(Counter(r["status"] for r in current)),
        "completed_record_count": len(completed), "reviewer_count": len({r["reviewer"] for r in completed}),
        "model_count": len({r["model"] for r in completed}), "reviewer_model_count": len(by_reviewer),
        "models": sorted({r["model"] for r in completed}), "rounds": sorted({r["round"] for r in completed}),
        "latest_reviews": deepcopy(latest), **judgments,
    }
    return {"evaluation_scope": scope, "human_review_recorded": human_recorded,
            "external_ai_review_recorded": external_recorded, "external_review_summary": summary,
            "adopted_review_summary": adopted or {"source": "external_receipts", **deepcopy(judgments)},
            "summary_adjudication_recorded": adopted is not None,
            "diagnostics": diagnostics, "gold_promoted": False,
            "policy": "human_question_quality_external_ai_answer_evidence_no_automatic_gold"}
