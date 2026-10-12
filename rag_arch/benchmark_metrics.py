"""Pure local benchmark metrics and candidate validation for plan v1.4 D2.

This module reads no files, calls no providers, and does not create or approve
Gold. A successful structural check is not a human evidence review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import math
import re

from .status_benchmark import STATUS_TRACK, is_status_benchmark, validate_status_reference
from .benchmark_review_policy import EVALUATION_SCOPES, assess_benchmark_reviews

FOUR_STATES = ("符合", "不符合", "无法判断", "依据已废止")
CATEGORIES = ("B", "C", "D", "E-real", "E-injected", "G")
TOP_K = 8


def _table_has_structure(record: Mapping) -> bool:
    """Check actual rows/headers, not mathematical correctness or applicability."""
    markdown = record.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        cells = lambda line: [c.strip() for c in line.strip("|").split("|")]
        if len(lines) < 3 or any("|" not in line for line in lines):
            return False
        header, separator = cells(lines[0]), cells(lines[1])
        return (bool(header) and any(header) and len(separator) == len(header)
                and all(re.fullmatch(r":?-+:?", c) for c in separator)
                and all(len(cells(row)) == len(header) for row in lines[2:]))
    html = record.get("html")
    if isinstance(html, str):
        return (bool(re.search(r"<table\b", html, re.I)) and bool(re.search(r"<th\b", html, re.I))
                and len(re.findall(r"<tr\b", html, re.I)) >= 2 and bool(re.search(r"<td\b", html, re.I)))
    return False


def _unique_ids(values, field: str) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} 必须为证据 ID 数组")
    if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values):
        raise ValueError(f"{field} 中证据 ID 必须为无首尾空白的非空字符串")
    return list(dict.fromkeys(values))


def _groups(values) -> list[set[str]]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("sufficient_evidence_sets 必须为二维数组")
    result = []
    for index, group in enumerate(values):
        normalized = set(_unique_ids(group, f"sufficient_evidence_sets[{index}]"))
        if not normalized:
            raise ValueError("充分证据组不可为空；没有充分组时使用 []，不能使用 [[]]")
        result.append(normalized)
    return result


def _catalog_complete(record: Mapping) -> bool | None:
    """Check source completeness separately from an explicit review warning.

    ``content_readable`` is a narrow builder assertion for plain prose whose
    text layer and locator are present but whose OCR has not been fully proofread.
    It is usable for retrieval/evidence metrics while ``review_status`` remains
    visible to candidate and Gold review code. Tables, images and numeric gaps
    retain their stricter requirements.
    """
    if record.get("numeric_text_usable") is False or record.get("image_refs"):
        return False
    # An explicit incomplete marker from the caller remains authoritative, even
    # when a stale/optimistic readability hint is also present.
    if record.get("content_complete") is False:
        return False
    status = record.get("content_status")
    if record.get("table_id"):
        return status == "text_available" and _table_has_structure(record)
    if status in {
        "image_only", "source_page_only", "image_with_unreviewed_ocr",
        "unsupported", "ocr_candidate", "unreviewed_ocr", "text_with_page_images",
        "text_plus_image_needs_review",
    }:
        return False
    if record.get("content_readable") is True:
        return isinstance(record.get("text"), str) and bool(record["text"].strip())
    if record.get("content_complete") is False:
        return False
    if status == "needs_review":
        return False
    if record.get("content_complete") is True:
        return True
    if status == "text_available":
        return isinstance(record.get("text"), str) and bool(record["text"].strip())
    return None


def score_evidence(*, sufficient_evidence_sets, supporting_evidence_ids,
                   raw_ranking, complete_context_ids, cited_ids,
                   predicted_conclusion: str | None, gold_conclusion: str | None,
                   evidence_catalog: Mapping[str, Mapping] | None = None,
                   incomplete_evidence_ids=()) -> dict:
    """Score one query: deduplicate its raw ranking before taking exactly top 8.

    C must describe actual complete context, not merely retrieved IDs. Known
    incomplete records can be excluded defensively through the optional catalog
    or incomplete_evidence_ids. Unknown raw IDs keep their rank positions.
    Empty G yields None for retrieval metrics; no sufficient groups yields None
    for coverage/strict correctness, as required for no-evidence G questions.
    """
    groups = _groups(sufficient_evidence_sets)
    supporting = set(_unique_ids(supporting_evidence_ids, "supporting_evidence_ids"))
    raw = _unique_ids(raw_ranking, "raw_ranking")
    context = set(_unique_ids(complete_context_ids, "complete_context_ids"))
    cited = set(_unique_ids(cited_ids, "cited_ids"))
    incomplete = set(_unique_ids(incomplete_evidence_ids, "incomplete_evidence_ids"))
    relevant = supporting.union(*groups)
    diagnostics = []
    for field, source, normalized in (("raw_ranking", raw_ranking, raw),
                                      ("complete_context_ids", complete_context_ids, context),
                                      ("cited_ids", cited_ids, cited)):
        if len(source) != len(normalized):
            diagnostics.append({"code": "duplicate_ids_normalized", "field": field})
    if evidence_catalog is not None:
        unknown_gold = relevant - set(evidence_catalog)
        if unknown_gold:
            raise ValueError("Gold 引用了目录中不存在的证据：" + ", ".join(sorted(unknown_gold)))
        for field, ids in (("raw_ranking", set(raw)), ("complete_context_ids", context), ("cited_ids", cited)):
            unknown = ids - set(evidence_catalog)
            if unknown:
                diagnostics.append({"code": "unknown_evidence_ids", "field": field, "ids": sorted(unknown)})
        incomplete.update(eid for eid, record in evidence_catalog.items()
                          if isinstance(record, Mapping) and _catalog_complete(record) is False)
        unknown_context = context - set(evidence_catalog)
    else:
        unknown_context = set()
    excluded = context & (incomplete | unknown_context)
    context -= excluded
    if excluded:
        diagnostics.append({"code": "incomplete_or_unknown_context_excluded", "ids": sorted(excluded)})
    r8 = raw[:TOP_K]
    recall = len(set(r8) & relevant) / len(relevant) if relevant else None
    dcg = sum(1 / math.log2(rank + 2) for rank, eid in enumerate(r8) if eid in relevant)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(TOP_K, len(relevant))))
    coverage = max((len(context & group) / len(group) for group in groups), default=None)
    conclusion_correct = (predicted_conclusion == gold_conclusion
                          if gold_conclusion is not None and str(gold_conclusion).strip() else None)
    context_complete = any(group <= context for group in groups) if groups else None
    cited_complete = any(group <= (context & cited) for group in groups) if groups else None
    strict = (bool(conclusion_correct and cited_complete)
              if groups and conclusion_correct is not None else None)
    return {
        "raw_top8_ids": r8, "relevant_evidence_ids": sorted(relevant),
        "complete_context_ids": sorted(context), "cited_ids": sorted(cited),
        "recall_at_8": recall, "ndcg_at_8": dcg / idcg if idcg else None,
        "evidence_coverage": coverage, "conclusion_correct": conclusion_correct,
        "has_complete_sufficient_context": context_complete,
        "has_complete_cited_sufficient_group": cited_complete, "strict_correct": strict,
        "predicted_conclusion": predicted_conclusion, "gold_conclusion": gold_conclusion,
        "context_completeness_basis": "caller_declared_complete_context_with_catalog_gap_checks"
            if evidence_catalog is not None else "caller_declared_complete_context",
        "diagnostics": diagnostics,
    }


def four_state_metrics(gold_conclusions: Sequence[str], predicted_conclusions: Sequence[str | None]) -> dict:
    """Rows=gold, columns=prediction; missing predictions count as wrong.

    Macro-F1 uses all four fixed labels, with zero-denominator class F1=0.
    Empty evaluation sets return None accuracy/macro-F1. Unknown non-null labels
    are input errors, rather than being silently converted to '无法判断'.
    """
    if isinstance(gold_conclusions, str) or isinstance(predicted_conclusions, str):
        raise ValueError("四态输入必须为序列，不能是单个字符串")
    if len(gold_conclusions) != len(predicted_conclusions):
        raise ValueError("Gold 与预测数量不同")
    if any(v not in FOUR_STATES for v in gold_conclusions):
        raise ValueError("Gold 包含未知四态标签")
    if any(v is not None and v not in FOUR_STATES for v in predicted_conclusions):
        raise ValueError("预测包含未知四态标签；缺失预测请用 None")
    matrix = [[0 for _ in FOUR_STATES] for _ in FOUR_STATES]
    missing = [0 for _ in FOUR_STATES]
    for truth, prediction in zip(gold_conclusions, predicted_conclusions):
        row = FOUR_STATES.index(truth)
        if prediction is None:
            missing[row] += 1
        else:
            matrix[row][FOUR_STATES.index(prediction)] += 1
    per_class = {}
    for i, label in enumerate(FOUR_STATES):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(4)) - tp
        fn = sum(matrix[i]) + missing[i] - tp
        per_class[label] = {"support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    count = len(gold_conclusions)
    return {"labels": list(FOUR_STATES), "confusion_matrix": matrix, "row_axis": "gold",
            "column_axis": "prediction", "missing_predictions_by_gold": dict(zip(FOUR_STATES, missing)),
            "sample_count": count, "accuracy": sum(matrix[i][i] for i in range(4)) / count if count else None,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 4 if count else None,
            "per_class": per_class, "macro_policy": "fixed_four_labels_zero_division_0"}


def aggregate_results(results: Sequence[Mapping]) -> dict:
    """Average applicable metrics separately by split, category, and review scope.

    Each input is a score_evidence result plus item_id, split, category, and an
    optional evaluation_scope. There is deliberately no pooled E accuracy.
    Call separately for each system/run; repeated item IDs are reported as errors.
    """
    buckets = defaultdict(list)
    seen, errors = set(), []
    for row in results:
        if not isinstance(row, Mapping):
            errors.append({"code": "invalid_result_object"})
            continue
        key = (row.get("split"), row.get("category"), row.get("item_id"))
        if not all(isinstance(value, str) for value in key) or not key[2] or key[0] not in {"dev", "test"} or key[1] not in CATEGORIES:
            errors.append({"code": "invalid_aggregation_identity", "item_id": row.get("item_id")})
            continue
        if key[2] in seen:
            errors.append({"code": "duplicate_result", "item_id": row["item_id"]})
            continue
        seen.add(key[2])
        scope = row.get("evaluation_scope", "candidate_trial")
        if scope not in EVALUATION_SCOPES:
            errors.append({"code": "invalid_evaluation_scope", "item_id": row["item_id"]})
            continue
        buckets[(*key[:2], scope)].append(row)
    groups = []
    for (split, category, scope), rows in sorted(buckets.items()):
        group = {"split": split, "category": category, "sample_count": len(rows),
                 "evaluation_scope": scope,
                 "scope_counts": dict(Counter(r.get("evaluation_scope", "candidate_trial") for r in rows))}
        for metric in ("recall_at_8", "ndcg_at_8", "evidence_coverage", "strict_correct", "conclusion_correct"):
            values = [r[metric] for r in rows if r.get(metric) is not None]
            group[metric] = {"mean": sum(values) / len(values) if values else None, "valid_count": len(values)}
        if category in {"E-real", "E-injected"}:
            try:
                group["four_state"] = four_state_metrics(
                    [r.get("gold_conclusion") for r in rows], [r.get("predicted_conclusion") for r in rows])
            except ValueError as exc:
                group["four_state"] = None
                errors.append({"code": "invalid_four_state_labels", "split": split, "category": category, "evaluation_scope": scope,
                               "message": str(exc)})
        groups.append(group)
    return {"groups": groups, "errors": errors,
            "aggregation_policy": "separate_split_category_review_scope_no_pooled_E_accuracy"}


def validate_item(item: Mapping, *, evidence_catalog: Mapping[str, Mapping] | None = None,
                  plan_splits: Mapping[str, str] | None = None,
                  known_source_ids: Sequence[str] | None = None) -> dict:
    """Report schema/provenance gaps; never promote candidates into formal Gold.

    Canonical candidate evidence fields are gold.sufficient / gold.supporting.
    review_results may contain actual external AI answer/evidence reviews or
    legacy human source-review records. Human question-quality feedback alone
    does not certify the answer. Review identity is separate from Gold status.
    """
    errors, warnings = [], []

    def add(code, field, message, *, warning=False, **details):
        (warnings if warning else errors).append({"code": code, "field": field, "message": message, **details})

    if not isinstance(item, Mapping):
        return {"item_id": None, "errors": [{"code": "invalid_item_object", "field": "item", "message": "题目必须为 JSON 对象"}],
                "warnings": [], "structurally_valid": False, "evaluation_scope": "candidate_trial",
                "assessment_scope": "structural_check_only_not_gold_certification"}

    for field in ("item_id", "split", "category", "question", "gold_answer", "verification_date"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            add("missing_or_invalid_field", field, "需要非空字符串")
    if item.get("split") not in ("dev", "test"):
        add("invalid_split", "split", "仅使用 dev/test，不改变既定分集")
    category = item.get("category")
    track = item.get("benchmark_track", "clause_table")
    status_item = is_status_benchmark(item)
    status_no_hit = False
    if track not in ("clause_table", STATUS_TRACK):
        add("invalid_benchmark_track", "benchmark_track", "仅支持 clause_table / standard_status；旧题缺省为 clause_table")
    elif track == STATUS_TRACK:
        status_errors = validate_status_reference(item)
        errors.extend(status_errors)
        if not status_errors:
            query = item["status_query"]
            status_no_hit = any(r.get("found") is False and r.get("query_std_no") == query["std_no"]
                                and r.get("clause_no") == query["clause_no"] for r in item["source_status_records"])
    if "item_type" not in item:
        add("missing_or_invalid_field", "item_type", "需显式记录题型；非 E 类可为 null")
    if category not in CATEGORIES:
        add("invalid_category", "category", "需要既定 B/C/D/E-real/E-injected/G 类别")
    for field in ("acceptable_answers", "applicable_standards", "verification_source_ids"):
        try:
            ids = _unique_ids(item.get(field), field)
            if len(ids) != len(item[field]):
                add("duplicate_ids", field, "重复值按集合不增加证据数", warning=True)
            empty_allowed = ((field == "applicable_standards" and category == "G")
                             or (field in ("applicable_standards", "verification_source_ids") and status_no_hit))
            if field != "acceptable_answers" and not ids and not empty_allowed:
                add("empty_provenance_field", field, "核验范围及来源不能缺失")
        except ValueError as exc:
            add("missing_or_invalid_field", field, str(exc))
    try:
        raw_date = item.get("verification_date")
        if not isinstance(raw_date, str) or len(raw_date) != 10:
            raise ValueError()
        date.fromisoformat(raw_date)
    except (ValueError, TypeError):
        add("invalid_verification_date", "verification_date", "需要 YYYY-MM-DD 日期")
    if known_source_ids is not None and isinstance(item.get("verification_source_ids"), list):
        unknown = [v for v in item["verification_source_ids"] if isinstance(v, str) and v not in known_source_ids]
        if unknown:
            add("unknown_verification_sources", "verification_source_ids", "来源 ID 未登记", ids=unknown)
    gold = item.get("gold")
    groups, supporting = [], set()
    if not isinstance(gold, Mapping):
        add("missing_gold_structure", "gold", "需要 sufficient 二维数组和 supporting 数组；候选也应显式记录空缺")
    else:
        try:
            groups = _groups(gold.get("sufficient"))
            for index, source in enumerate(gold["sufficient"]):
                if len(source) != len(groups[index]):
                    add("duplicate_ids", f"gold.sufficient[{index}]", "组内重复 ID 不增加充分性", warning=True)
            keys = [frozenset(g) for g in groups]
            if len(set(keys)) != len(keys):
                add("duplicate_sufficient_groups", "gold.sufficient", "重复充分组不是额外替代依据", warning=True)
            if any(a < b for a in groups for b in groups):
                add("nonminimal_sufficient_group", "gold.sufficient", "某组包含另一完整组，须复核最小性", warning=True)
        except ValueError as exc:
            add("invalid_sufficient_groups", "gold.sufficient", str(exc))
        try:
            supporting = set(_unique_ids(gold.get("supporting"), "gold.supporting"))
            if len(supporting) != len(gold["supporting"]):
                add("duplicate_ids", "gold.supporting", "重复 ID 不增加相关证据数", warning=True)
        except ValueError as exc:
            add("invalid_supporting_ids", "gold.supporting", str(exc))
    sufficient = set().union(*groups)
    if not groups and category != "G" and not status_item:
        add("missing_sufficient_evidence", "gold.sufficient", "非 G 类候选尚未提供最小充分证据组")
    if sufficient & supporting:
        add("supporting_overlaps_sufficient", "gold.supporting", "必要证据与仅补充证据标记矛盾", ids=sorted(sufficient & supporting))
    if evidence_catalog is None and not status_item:
        add("evidence_catalog_not_checked", "gold", "未提供目录，未核对引用存在性或表格完整性", warning=True)
    elif evidence_catalog is not None:
        for eid in sorted(sufficient | supporting):
            record = evidence_catalog.get(eid)
            if not isinstance(record, Mapping):
                add("unknown_gold_evidence", "gold", "证据 ID 不存在或目录记录无效", evidence_id=eid)
                continue
            if _catalog_complete(record) is False:
                add("sufficient_evidence_incomplete" if eid in sufficient else "supporting_evidence_incomplete",
                    "gold", "图片、缺表或未校 OCR 不能充当已完整取得的证据", warning=eid not in sufficient, evidence_id=eid)
            elif (record.get("content_readable") is True
                  and (record.get("review_unverified") is True
                       or record.get("review_status") in {"unreviewed_ocr", "needs_review"})):
                add("evidence_content_not_fully_proofread", "gold",
                    "文字层可用但仍未逐字校勘；不等同正式Gold", warning=True, evidence_id=eid)
            spans = record.get("source_spans", [])
            if not isinstance(spans, (list, tuple)):
                spans = []
            located = any(isinstance(s, Mapping) and s.get("source_id") and s.get("source_path")
                          and (s.get("page_start") is not None or s.get("block_start") is not None) for s in spans)
            if not located:
                add("missing_evidence_locator", "gold", "证据缺少文件身份和页/网页块定位", evidence_id=eid)
            if record.get("table_id") and not record.get("owner_clause_id"):
                add("table_owner_missing", "gold", "表格缺少所属条文", evidence_id=eid)
    if category in ("E-real", "E-injected"):
        if not isinstance(item.get("plan_id"), str) or not item["plan_id"].strip():
            add("missing_plan_field", "plan_id", "方案核查题需要真实方案")
        assertion = item.get("assertion")
        if isinstance(assertion, Mapping):
            for field in ("original_text", "claim_text"):
                if not isinstance(assertion.get(field), str) or not assertion[field].strip():
                    add("missing_assertion_text", f"assertion.{field}", "结构化断言需保留原句和本题断言")
            if assertion.get("plan_id") and assertion["plan_id"] != item.get("plan_id"):
                add("assertion_plan_mismatch", "assertion.plan_id", "断言与题目方案 ID 不一致")
        elif not isinstance(assertion, str) or not assertion.strip():
            add("missing_plan_field", "assertion", "方案核查题需要断言文本或结构化断言")
        if "plan_compiled_date" not in item:
            add("missing_plan_compiled_date_field", "plan_compiled_date", "未知编制日期可留 null，但应显式记录")
        if item.get("item_type") not in ("N", "P", "S", "R"):
            add("invalid_check_item_type", "item_type", "方案核查仅采用 N/P/S/R")
        if item.get("gold_answer") not in FOUR_STATES:
            add("invalid_four_state_gold", "gold_answer", "方案核查结论需要四态标签")
        if category == "E-real" and (item.get("injection") or item.get("injected_from")):
            add("real_item_has_injection", "injection", "E-real 不能包含主动注入的修改")
        if category == "E-injected":
            injection = item.get("injection")
            origin = item.get("injected_from") if isinstance(item.get("injected_from"), Mapping) else {}
            if not isinstance(injection, Mapping) and isinstance(item.get("injected_from"), Mapping) and isinstance(assertion, Mapping):
                injection = {"original_text": assertion.get("original_text"), "modified_text": assertion.get("claim_text")}
            if (not isinstance(injection, Mapping) or not isinstance(injection.get("original_text"), str)
                    or not injection.get("original_text", "").strip() or not isinstance(injection.get("modified_text"), str)
                    or not injection.get("modified_text", "").strip()):
                add("missing_injection_provenance", "injection", "注入题需要 original_text 和 modified_text")
            elif injection["original_text"] == injection["modified_text"]:
                add("injection_has_no_change", "injection", "原句与修改句相同，与注入标签矛盾")
            else:
                original_text, modified_text = injection["original_text"], injection["modified_text"]
                assertion_map = assertion if isinstance(assertion, Mapping) else {}
                source_unit = origin.get("source_unit_id") or assertion_map.get("source_unit_id") or injection.get("source_unit_id")
                if not source_unit and not assertion_map.get("locator") and not injection.get("locator"):
                    add("missing_injection_source_locator", "injected_from", "注入原句须保留源单元 ID 或原方案定位")
                # Numeric-only edits preserve every surrounding character/unit.
                number_pattern = r"[+-]?\d+(?:\.\d+)?"
                numeric_only = (re.sub(number_pattern, "<NUMBER>", original_text)
                                == re.sub(number_pattern, "<NUMBER>", modified_text))
                if numeric_only:
                    before_numbers = [Decimal(value) for value in re.findall(number_pattern, original_text)]
                    after_numbers = [Decimal(value) for value in re.findall(number_pattern, modified_text)]
                    if before_numbers == after_numbers:
                        add("injection_value_unchanged", "injection", "仅改变数值格式，没有实际改变参数值")
                before = origin.get("original_value", injection.get("original_value"))
                after = origin.get("injected_value", injection.get("injected_value"))
                change_type = origin.get("change_type", injection.get("change_type"))
                state_edit = (change_type in ("state", "explicit_state_only") and isinstance(before, str) and before
                              and isinstance(after, str) and after and original_text.replace(before, after, 1) == modified_text)
                if not numeric_only and not state_edit:
                    add("injection_changes_other_wording", "injection", "不是仅改数值；明确状态修改需提供 change_type=state 和可重建的原值/改值")
                elif state_edit:
                    add("state_edit_requires_source_review", "injection", "程序仅核对替换范围；状态修改是否明确仍须回到原文复核", warning=True)
                plan_records = item.get("plan_evidence_records", [])
                if not isinstance(plan_records, (list, tuple)):
                    plan_records = []
                source_record = next((r for r in plan_records if isinstance(r, Mapping)
                                      and source_unit and source_unit in (r.get("unit_id"), r.get("source_unit_id"), r.get("id"))), None)
                if source_record:
                    source_text = source_record.get("text", source_record.get("original_text"))
                    if source_text != original_text:
                        add("injection_original_text_mismatch", "assertion.original_text", "原句与已提供方案源单元不一致")
                else:
                    add("injection_source_text_not_checked", "injected_from", "已保留位置，但未在本次输入中核对该源单元原文", warning=True)
    if plan_splits is not None and isinstance(item.get("plan_id"), str) and item["plan_id"]:
        registered = plan_splits.get(item["plan_id"])
        if registered is None:
            add("unknown_plan_split", "plan_id", "方案尚未登记 Dev/Test 分配")
        elif registered != item.get("split"):
            add("plan_split_mismatch", "split", "题目分集与方案分配不同，可能混用 Dev/Test")
    reviews = item.get("review_results")
    if not isinstance(reviews, (list, Mapping)):
        add("missing_review_results", "review_results", "需要明确记录候选审阅状态")
        reviews = {}
    policy = assess_benchmark_reviews(item)
    warnings.extend(policy["diagnostics"])
    external = policy["external_review_summary"]
    adopted = policy["adopted_review_summary"]
    human_recorded = policy["human_review_recorded"]
    if policy["evaluation_scope"] == "candidate_trial":
        add("answer_evidence_review_not_recorded", "review_results", "程序可试跑；尚无当前题目版本的答案/证据审阅记录", warning=True)
    if item.get("gold_status") == "formal":
        if adopted["support_status"] in {"conflict", "changes_requested", "unresolved", "supported_with_revisions"}:
            add("formal_claim_with_unresolved_external_reviews", "gold_status", "正式标签与当前采用层的反对、分歧、未定项或待落实修订矛盾")
        elif not (human_recorded or adopted["supported"]):
            add("formal_claim_without_answer_evidence_review", "gold_status", "正式标签缺少对应答案/证据审阅支持")
    if item.get("gold_status") == "human_reviewed" and not human_recorded:
        add("formal_claim_without_human_review", "gold_status", "人审标签缺少对应人工原文复核记录")
    if item.get("human_reviewed") is True and not human_recorded:
        add("formal_claim_without_human_review", "human_reviewed", "布尔人审标记缺少对应原文复核记录")
    return {"item_id": item.get("item_id"), "errors": errors, "warnings": warnings,
            "structurally_valid": not errors,
            "evaluation_scope": policy["evaluation_scope"],
            "review_policy": policy, "external_review_summary": external,
            "adopted_review_summary": adopted,
            "assessment_scope": "structural_check_only_not_gold_certification"}


def validate_items(items: Sequence[Mapping], **kwargs) -> dict:
    reports = [validate_item(item, **kwargs) for item in items]
    objects = [item for item in items if isinstance(item, Mapping)]
    counts = Counter(item.get("item_id") for item in objects if isinstance(item.get("item_id"), str))
    errors = [{"code": "duplicate_item_id", "item_id": eid, "count": count}
              for eid, count in counts.items() if eid and count > 1]
    plan_splits = defaultdict(set)
    for item in objects:
        if isinstance(item.get("plan_id"), str) and item["plan_id"] and isinstance(item.get("split"), str):
            plan_splits[item["plan_id"]].add(item.get("split"))
    errors += [{"code": "plan_used_in_both_splits", "plan_id": pid, "splits": sorted(splits)}
               for pid, splits in plan_splits.items() if {"dev", "test"} <= splits]
    return {"item_count": len(items), "items": reports, "errors": errors,
            "structurally_valid": not errors and all(r["structurally_valid"] for r in reports),
            "assessment_scope": "structural_check_only_not_gold_certification"}


def validate_predictions(items: Sequence[Mapping], predictions: Sequence[Mapping]) -> dict:
    """Report missing, extra and duplicate run IDs for one system/run."""
    expected = {item["item_id"] for item in items
                if isinstance(item, Mapping) and isinstance(item.get("item_id"), str) and item["item_id"]}
    counts = Counter(row.get("item_id") if isinstance(row, Mapping) and isinstance(row.get("item_id"), str) else None
                     for row in predictions)
    actual = {eid for eid in counts if eid}
    errors = [{"code": "prediction_missing", "item_id": eid} for eid in sorted(expected - actual)]
    errors += [{"code": "unknown_prediction_item", "item_id": eid} for eid in sorted(actual - expected)]
    errors += [{"code": "duplicate_prediction", "item_id": eid, "count": count}
               for eid, count in counts.items() if count > 1]
    if None in counts or "" in counts:
        errors.append({"code": "prediction_item_id_missing"})
    return {"expected_count": len(expected), "prediction_count": len(predictions), "errors": errors,
            "coverage_complete": not errors}
