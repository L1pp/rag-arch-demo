"""Atomic clause/table retrieval and post-ranking evidence expansion (plan D2)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import math

from .models import Chunk, SearchHit
from .retrieval import LocalIndex
from .standards import StandardCorpus, load_corpus, _standard_key


@dataclass
class EvidenceResult:
    query: str
    mode: str
    raw_hits: list[SearchHit]
    context_hits: list[SearchHit]
    missing_evidence: list[dict] = field(default_factory=list)
    reranked: bool = False

    @property
    def raw_ids(self) -> list[str]:
        return [hit.chunk.chunk_id for hit in self.raw_hits]

    @property
    def context_ids(self) -> list[str]:
        # A caption or image link is not the actual table/formula content.
        return [hit.chunk.chunk_id for hit in self.context_hits
                if hit.chunk.locator.get("content_complete", False)]


def _readable_table(record: dict) -> bool:
    return (record.get("content_status") == "text_available"
            and bool((record.get("markdown") or record.get("html") or "").strip()))


def _readable_clause(record: dict) -> bool:
    """Legacy whole-evidence readiness, not a measure of returned prose length."""
    body = record.get("text", "")
    if not isinstance(body, str) or not body.strip() or record.get("image_refs"):
        return False
    if record.get("numeric_text_usable") is False:
        return False
    if record.get("content_status") in {
        "image_only", "source_page_only", "image_with_unreviewed_ocr",
        "text_plus_image_needs_review", "unsupported", "ocr_candidate",
        "unreviewed_ocr",
    }:
        return False
    return (record.get("content_status", "text_available") == "text_available"
            or record.get("content_readable") is True)


def _review_metadata(record: dict) -> tuple[str | None, bool, str | None]:
    status = record.get("review_status")
    if status is None and record.get("content_status") in {"needs_review", "text_plus_image_needs_review"}:
        status = "unreviewed_ocr"
    unverified = bool(record.get("review_required")) or status in {"unreviewed_ocr", "needs_review"}
    warning = "text_not_fully_proofread" if unverified else None
    return status, unverified, warning


def _record_chunk(record: dict, kind: str, collection_id: str) -> Chunk:
    std_no = record["std_no"]
    evidence_id = record["clause_id"] if kind == "clause" else record["table_id"]
    if kind == "clause":
        body = record.get("text", "")
        title = f"{std_no} {record['clause_no']}"
        if record.get("record_kind") == "formula":
            title = f"{std_no} 公式 {record.get('formula_no', record['clause_no'])}"
            formula_parts = [body]
            for field, label in (("formula_latex", "公式（LaTeX）"), ("formula_text", "公式"),
                                 ("variables", "变量"), ("units", "单位")):
                value = record.get(field)
                if value:
                    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if rendered not in body:
                        formula_parts.append(f"{label}：{rendered}")
            body = "\n".join(part for part in formula_parts if part)
        complete = _readable_clause(record)
        linked = list(dict.fromkeys(record.get("table_ids", []) + record.get("table_refs", [])
                                   + record.get("unresolved_table_refs", [])))
    else:
        title = f"{std_no} {record.get('caption', evidence_id)}"
        body = "\n".join(part for part in [record.get("markdown") or record.get("html") or "",
                                          *record.get("notes", [])] if part)
        complete = _readable_table(record)
        linked = []
    review_status, review_unverified, review_warning_code = _review_metadata(record)
    stored_body = record.get("text", "") if kind == "clause" else (record.get("markdown") or record.get("html") or "")
    parts = [title, " / ".join(record.get("heading_path", [])), body]
    if not complete:
        parts.append("此项含未完整读取或未核准的内容，请查看原资料；不得据此补造图片、公式或表格数值。")
    locator = {
        "std_no": std_no, "evidence_id": evidence_id, "evidence_kind": kind,
        "clause_no": record.get("clause_no"), "table_id": record.get("table_id"),
        "source_spans": deepcopy(record.get("source_spans", [])),
        "candidate_path": record.get("candidate_path"),
        "image_refs": deepcopy(record.get("image_refs", [])),
        "asset_refs": deepcopy(record.get("asset_refs", [])),
        "source_page_images": deepcopy(record.get("source_page_images", [])),
        "document_part": record.get("document_part", "body"),
        "content_complete": complete,
        "stored_text_chars": len(stored_body) if isinstance(stored_body, str) and stored_body.strip() else 0,
        "content_readable": bool(record.get("content_readable",
                                           record.get("content_status", "text_available") == "text_available")),
        "content_status": record.get("content_status", "text_available"),
        "numeric_text_usable": record.get("numeric_text_usable"),
        "readability_basis": record.get("readability_basis"),
        "review_status": review_status, "review_unverified": review_unverified,
        "review_warning_code": review_warning_code,
        "standard_status": record.get("standard_status", "unknown"),
        "current_mandatory_status": record.get("current_mandatory_status", "unknown"),
        "verification_date": record.get("verification_date"),
        "status_checked_at": record.get("status_checked_at"),
        "status_source_ids": deepcopy(record.get("status_source_ids", [])),
        "linked_table_ids": linked, "owner_clause_id": record.get("owner_clause_id"),
        "linked_formula_ids": list(dict.fromkeys(record.get("formula_ids", []) + record.get("formula_refs", []))),
    }
    for field in ("std_id", "import_part_id", "record_kind", "relation_ids", "version_note",
                  "superseded_by", "deleted_by", "clause_fragment", "source_value_anomalies",
                  "formula_no", "formula_latex", "formula_text", "variables", "units",
                  "headers", "rows", "header_rows", "column_paths", "notes", "table_no",
                  "verification_scope", "verification_evidence", "numeric_verification_scopes",
                  "revision_notation_operations", "correction_ids", "edition_role", "source_std_no",
                  "numeric_values", "whole_clause_verified", "verified_fragment_of", "exact_clause_lookup",
                  "effective_date", "effective_text_verified", "consolidation_source_ids",
                  "merged_cells", "original_layout", "merged_groups", "merged_header",
                  "blank_record_cells_preserved", "is_record_template", "source_glyph_map",
                  "steel_grade_symbols", "symbol_notation", "applicability_conditions", "source_object_id",
                  "printed_table_no_verified", "literal_text_verified", "literal_text_verification_scope",
                  "technical_scope_text_usable", "technical_grade_text_usable", "technical_grade_verification_scope",
                  "source_reproduction_correction", "source_reproduction_superseded_by", "source_view_superseded_by",
                  "historical_source_table_ids"):
        if field in record:
            locator[field] = deepcopy(record[field])
    locator["excluded_from_current_retrieval"] = bool(
        record.get("superseded_by") or record.get("deleted_by")
        or record.get("source_reproduction_superseded_by") or record.get("source_view_superseded_by")
        or record.get("edition_role") in {"verified_2015_historical_baseline", "historical_2015_ocr_baseline"})
    return Chunk("standard:" + _standard_key(std_no), std_no, evidence_id,
                 "\n\n".join(part for part in parts if part), locator, "standard", collection_id)


class EvidenceIndex:
    """Raw ranking treats each clause/table as a separate ID, then expands tables.

    No exact-ID shortcut is applied to benchmark retrieval. Tools may explicitly
    use get_clause/get_table. Missing table text affects evidence completeness,
    not whether unrelated readable clauses can be loaded and queried.
    """

    def __init__(self, corpus: StandardCorpus, embeddings=None, reranker=None,
                 candidate_pool: int = 30, rrf_constant: int = 60):
        if candidate_pool < 1 or rrf_constant < 1:
            raise ValueError("候选池和 RRF 常数须为正数。")
        self.corpus = corpus
        self.collection_id = "standards:" + uuid4().hex
        self.candidate_pool = candidate_pool
        self.rrf_constant = rrf_constant
        self.reranker = reranker
        self.embeddings = embeddings
        self.trace_candidates = False
        self.chunks = [_record_chunk(record, kind, self.collection_id)
                       for kind, records in (("clause", corpus.clauses), ("table", corpus.tables))
                       for record in records if record.get("document_part", "body") != "commentary"]
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("规范条文与表格证据 ID 不唯一。")
        self._by_import_part = {}
        self._revision_relations = {}
        for chunk in self.chunks:
            locator = chunk.locator
            part = locator.get("import_part_id")
            if part:
                self._by_import_part.setdefault((locator.get("std_id"), part), []).append(chunk)
        for relation in getattr(corpus, "relations", []):
            if (relation.get("source_part_id") and relation.get("target_std_id")
                    and relation.get("relation") in {"amends", "amendment", "errata", "supersedes", "deletes"}):
                self._revision_relations.setdefault(relation["target_std_id"], []).append(relation)

        self.local = LocalIndex(self.chunks, embeddings)

    def _candidate_trace_active(self) -> bool:
        if not self.trace_candidates:
            return False
        try:
            from .runtime_audit import _current
            audit = _current.get()
            return audit is not None and not audit.closed
        except Exception:
            return False

    def _record_candidate_trace(self, **fields) -> None:
        """Best-effort diagnostic event; never participates in retrieval output."""
        try:
            from .runtime_audit import record_event
            record_event("retrieval_candidate_audit", **fields)
        except Exception:
            return None

    def _record_preflight_failure(self, query, mode, top_k, filters, stage, failure_type) -> None:
        if not self._candidate_trace_active():
            return
        try:
            saved_filters = deepcopy(filters)
        except Exception:
            saved_filters = None
        self._record_candidate_trace(
            status="failed", stage=stage, failure_type=failure_type,
            query=query, filters=saved_filters, mode=mode,
            corpus_id=self.collection_id, corpus_chunk_count=len(self.chunks),
            top_k=top_k, candidate_pool=self.candidate_pool,
            effective_pool=None, trace_candidate_limit=30,
            bm25_candidates=None, dense_candidates=None,
            pre_rerank_candidates=None, pre_rerank_candidate_count=0,
            candidate_lists_truncated=False, reranker_status="not_reached")

    def _trace_candidate(self, chunk_index: int, **scores) -> dict:
        chunk = self.chunks[chunk_index]
        locator = chunk.locator
        return {
            "evidence_id": chunk.chunk_id,
            "kind": locator.get("evidence_kind"),
            "std_no": locator.get("std_no"),
            "clause_no": locator.get("clause_no"),
            "table_id": locator.get("table_id"),
            **scores,
        }

    def get_clause(self, std_no: str, clause_no: str) -> Chunk | None:
        record = self.corpus.get_clause(std_no, clause_no)
        return self.by_id.get(record["clause_id"]) if record else None

    def get_table(self, table_id: str) -> Chunk | None:
        chunk = self.by_id.get(table_id)
        return chunk if chunk and chunk.locator["evidence_kind"] == "table" else None


    def _revision_targets(self, chunk):
        """Declared clause/ID/page relations only; never semantic rule matching."""
        locator = chunk.locator
        sid = locator.get("std_id") or self.corpus.resolve_standard_id(locator["std_no"])
        ids, details, unresolved = [], [], []
        for relation in self._revision_relations.get(sid, []):
            if locator.get("import_part_id", locator.get("document_part", "body")) != relation.get("target_part_id", "body"):
                continue
            targets = relation.get("target_clause_ids", [])
            numbers = relation.get("target_clause_nos", [])
            page_ranges = relation.get("target_page_ranges", [])
            matched = chunk.chunk_id in targets or bool(locator.get("clause_no") and locator["clause_no"] in numbers)
            if not matched and page_ranges:
                for span in locator.get("source_spans", []):
                    if relation.get("target_source_id") and span.get("source_id") != relation["target_source_id"]:
                        continue
                    start, end = span.get("page_start"), span.get("page_end", span.get("page_start"))
                    if type(start) is int and type(end) is int and any(start <= b and end >= a for a, b in page_ranges):
                        matched = True
                        break
            # Unscoped whole-document amendment links remain discovery metadata.
            if not matched:
                continue
            candidates = self._by_import_part.get((relation.get("source_std_id"), relation["source_part_id"]), [])
            source_ids = relation.get("source_evidence_ids", [])
            source_numbers = relation.get("source_clause_nos", [])
            source_pages = relation.get("source_physical_pages", [])
            selected = [c for c in candidates if c.chunk_id in source_ids or c.locator.get("clause_no") in source_numbers]
            if not source_ids and not source_numbers:
                selected = candidates
            if source_pages and (not selected or not source_ids and not source_numbers):
                selected = [c for c in candidates if any(
                    type(span.get("page_start")) is int and any(
                        span["page_start"] <= p <= span.get("page_end", span["page_start"]) for p in source_pages)
                    for span in c.locator.get("source_spans", []))]
            rid = relation.get("relation_id", relation.get("id"))
            # Do not send an entire 32-page amendment just because a source_part
            # is named. Scoped clauses/pages above are returned without truncation.
            if not source_ids and not source_numbers and not source_pages and sum(len(c.text) for c in selected) > 16000:
                selected = []
                unresolved.append({"evidence_id": rid, "owner": chunk.chunk_id,
                                   "reason": "revision_requires_scoped_read",
                                   "source_std_id": relation.get("source_std_id"),
                                   "source_part_id": relation["source_part_id"]})
            elif not selected:
                unresolved.append({"evidence_id": rid, "owner": chunk.chunk_id,
                                   "reason": "revision_evidence_missing"})
            ids.extend(c.chunk_id for c in selected)
            details.append({key: deepcopy(relation.get(key)) for key in (
                "relation_id", "relation", "scope", "whole_clause", "source_std_id", "source_part_id",
                "target_clause_nos", "target_page_ranges", "target_page_scope") if key in relation})
        return list(dict.fromkeys(ids)), details, unresolved

    def _with_revision_notice(self, hit):
        chunk = hit.chunk
        ids, details, unresolved = self._revision_targets(chunk)
        locator = deepcopy(chunk.locator)
        if ids or details:
            locator["revision_evidence_ids"] = ids
            locator["revision_relations"] = details
            locator["edition_role"] = ("deleted_base_text" if locator.get("deleted_by")
                                      else "superseded_base_text" if locator.get("superseded_by")
                                      else "base_text_with_amendment")
            locator["version_note"] = (
                "这是保留的底本文字，存在对应修订/勘误；应结合补入的新文本读取。"
                "按原页范围关联只定位相关旧材料，不表示整页其他条文全部失效。")
        elif locator.get("document_part") in {"amendment", "errata"}:
            locator["edition_role"] = "errata_text" if locator["document_part"] == "errata" else "amendment_text"
            locator["version_note"] = "这是单列的修订/勘误来源；仅按其明确范围修改底本，不能扩大为整条或整本替代。"
        elif locator.get("deleted_by") or locator.get("superseded_by"):
            locator["edition_role"] = "deleted_base_text" if locator.get("deleted_by") else "superseded_base_text"
            locator["version_note"] = "这是明确删除/替换后保留的历史原文，不作为当前检索要求。"
        elif locator.get("edition_role") == "verified_2024_effective_clause_view":
            locator.setdefault("version_note", "这是按已核2015底本与2024局修原页逐条重建的有效文字视图；只核准本条及所列结构对象，不代表整本清洁合本或全部数值已核。")
        elif locator.get("edition_role") in {"verified_2015_historical_baseline", "historical_2015_ocr_baseline"}:
            locator.setdefault("version_note", "这是独立保留的2015历史底本；当前适用文字须结合2024局修及经核准的有效视图，不能直接把历史表值当作现用要求。")
        elif locator.get("edition_role") == "verified_original_scoped_clause":
            locator.setdefault("version_note", "本对象仅按所列原页核准字句、公式或表格范围；通用规范效力关系及工程适用条件另行核对，未核内容不会因此自动升为可用。")
        elif locator.get("edition_role") == "historical_source_reproduction_with_known_errors":
            locator.setdefault("version_note", "这是保留的同版旧来源转写；已由所列可靠原书来源纠正。此来源修补不代表规范改版，旧文本不作为当前数值依据。")
        text = chunk.text
        if locator.get("edition_role") and locator.get("version_note"):
            text = f"[资料版本说明：{locator['version_note']}]\n" + text
        return SearchHit(replace(chunk, text=text, locator=locator), hit.score), unresolved


    def expand(self, raw_hits: list[SearchHit]) -> tuple[list[SearchHit], list[dict]]:
        context, missing, seen = [], [], set()

        def add(hit):
            hit, revision_missing = self._with_revision_notice(hit)
            for entry in revision_missing:
                if entry not in missing:
                    missing.append(entry)
            chunk = hit.chunk
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                context.append(hit)
                if not chunk.locator["content_complete"]:
                    missing.append({"evidence_id": chunk.chunk_id, "reason": "content_incomplete"})

        # Keep original ranking visibly first, then append supplements.
        for hit in raw_hits:
            add(hit)
        def supplement(evidence_ids, owner, score):
            added = []
            for evidence_id in dict.fromkeys(evidence_ids):
                chunk = self.by_id.get(evidence_id)
                if chunk is None:
                    entry = {"evidence_id": evidence_id, "owner": owner, "reason": "record_missing"}
                    if entry not in missing:
                        missing.append(entry)
                else:
                    add(SearchHit(chunk, score))
                    added.append(chunk)
            return added

        for hit in raw_hits:
            locator = hit.chunk.locator
            supplement(locator.get("linked_table_ids", []), hit.chunk.chunk_id, hit.score)
            supplement(locator.get("linked_formula_ids", []), hit.chunk.chunk_id, hit.score)
            # Also include the owner's directly required tables when a table
            # was retrieved first. This is bounded to one owner, not recursive.
            owner_id = locator.get("owner_clause_id")
            if owner_id:
                for owner in supplement([owner_id], hit.chunk.chunk_id, hit.score):
                    supplement(owner.locator.get("linked_table_ids", []), owner_id, hit.score)
                    supplement(owner.locator.get("linked_formula_ids", []), owner_id, hit.score)
        for hit in list(context):
            supplement(hit.chunk.locator.get("revision_evidence_ids", []), hit.chunk.chunk_id, hit.score)
        return context, missing

    def retrieve(self, query: str, mode: str = "bm25", top_k: int = 8,
                 filters: dict | None = None, use_reranker: bool = False) -> EvidenceResult:
        if mode not in {"bm25", "dense", "rrf", "hybrid"}:
            self._record_preflight_failure(query, mode, top_k, filters,
                                           "argument_validation", "ValueError")
            raise ValueError("检索模式须为 bm25、dense、rrf 或 hybrid。")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            self._record_preflight_failure(query, mode, top_k, filters,
                                           "argument_validation", "ValueError")
            raise ValueError("top_k 须为正整数。")
        if not isinstance(query, str) or not query.strip():
            return EvidenceResult(str(query or ""), mode, [], [])
        query = query.strip()
        rerank = use_reranker or mode == "hybrid"
        if mode != "bm25" and self.embeddings is None:
            self._record_preflight_failure(query, mode, top_k, filters,
                                           "retrieval_configuration", "ValueError")
            raise ValueError("此检索模式需要文档向量索引；请先配置并加载向量。")
        if rerank and self.reranker is None:
            self._record_preflight_failure(query, mode, top_k, filters,
                                           "retrieval_configuration", "ValueError")
            raise ValueError("已选择重排，但重排客户端尚未配置。")
        filters = filters or {}
        unknown = set(filters) - {"std_no", "std_nos", "kind", "document_part"}
        if unknown:
            self._record_preflight_failure(query, mode, top_k, filters,
                                           "filter_validation", "ValueError")
            raise ValueError("不支持的规范过滤字段：" + ", ".join(sorted(unknown)))
        standards = filters.get("std_nos", [filters["std_no"]] if filters.get("std_no") else [])
        if isinstance(standards, str):
            standards = [standards]
        keys = {self.corpus.resolve_standard_id(value) for value in standards}
        allowed = [i for i, chunk in enumerate(self.chunks)
                   if not chunk.locator.get("excluded_from_current_retrieval", False)
                   and (not standards or (self.corpus.resolve_standard_id(chunk.locator["std_no"]) in keys - {None}))
                   and (not filters.get("kind") or chunk.locator["evidence_kind"] == filters["kind"])
                   and (not filters.get("document_part") or chunk.locator["document_part"] == filters["document_part"])]
        pool = max(top_k, self.candidate_pool)
        trace = self._candidate_trace_active()
        trace_base = {
            "query": query, "filters": deepcopy(filters), "mode": mode,
            "corpus_id": self.collection_id, "corpus_chunk_count": len(self.chunks),
            "eligible_chunk_count": len(allowed), "top_k": top_k,
            "candidate_pool": self.candidate_pool, "effective_pool": pool,
            "trace_candidate_limit": 30,
        }
        if not allowed:
            if trace:
                self._record_candidate_trace(
                    **trace_base, status="complete", stage="empty_filter_result",
                    bm25_candidates=None if mode == "dense" else [],
                    dense_candidates=None if mode == "bm25" else [],
                    pre_rerank_candidates=[], pre_rerank_candidate_count=0,
                    candidate_lists_truncated=False, reranker_status="not_needed",
                    final_top_k=[], expanded_ids=[])
            return EvidenceResult(query, mode, [], [], reranked=rerank)

        def order(scores, positive=False):
            return sorted((i for i in allowed if not positive or scores[i] > 0),
                          key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:pool]

        bm25 = dense = scores = None
        bm25_ranked: list[int] | None = None
        dense_ranked: list[int] | None = None
        ranked: list[int] = []
        hits: list[SearchHit] = []
        context: list[SearchHit] = []
        missing: list[dict] = []
        stage = "score_generation"
        reranker_status = "not_reached" if rerank else "not_used"
        reranker_positions: dict[int, tuple[int, float]] = {}
        final_positions: dict[int, int] = {}

        def ranked_rows(indices, score_values):
            if indices is None or score_values is None:
                return None
            limit = min(pool, 30)
            return [self._trace_candidate(i, rank=rank,
                    score=float(score_values[i]))
                    for rank, i in enumerate(indices[:limit], 1)]

        def trace_fields(*, status, stage, failure_type=None):
            lanes = {
                "bm25_candidates": ranked_rows(bm25_ranked, bm25),
                "dense_candidates": ranked_rows(dense_ranked, dense),
            }
            fused = []
            bm25_ranks = {i: rank for rank, i in enumerate(bm25_ranked or [], 1)}
            dense_ranks = {i: rank for rank, i in enumerate(dense_ranked or [], 1)}
            is_fused = mode in {"rrf", "hybrid"}
            for candidate_rank, i in enumerate(ranked[:min(pool, 30)], 1):
                row = self._trace_candidate(
                    i, candidate_rank=candidate_rank,
                    candidate_score=float(scores[i]) if scores is not None else None,
                    fused_rank=candidate_rank if is_fused else None,
                    fused_score=float(scores[i]) if is_fused and scores is not None else None,
                    bm25_rank=bm25_ranks.get(i),
                    bm25_score=float(bm25[i]) if bm25 is not None else None,
                    dense_rank=dense_ranks.get(i),
                    dense_score=float(dense[i]) if dense is not None else None,
                    reranker_rank=None, reranker_score=None, final_rank=None)
                if i in reranker_positions:
                    row["reranker_rank"], row["reranker_score"] = reranker_positions[i]
                if i in final_positions:
                    row["final_rank"] = final_positions[i]
                fused.append(row)
            fields = {**trace_base, **lanes, "pre_rerank_candidates": fused,
                      "pre_rerank_candidate_count": len(ranked),
                      "candidate_lists_truncated": ((bm25_ranked is not None and len(bm25_ranked) > 30) or
                          (dense_ranked is not None and len(dense_ranked) > 30) or len(ranked) > 30),
                      "status": status, "stage": stage,
                      "reranker_status": reranker_status}
            if failure_type is not None:
                fields["failure_type"] = failure_type
            if status == "complete" or hits:
                fields["final_top_k"] = [
                    {"rank": rank, "evidence_id": hit.chunk.chunk_id,
                     "score": float(hit.score),
                     "score_kind": "reranker" if rerank and ranked else mode}
                    for rank, hit in enumerate(hits, 1)]
            if status == "complete":
                fields["expanded_ids"] = [hit.chunk.chunk_id for hit in context]
            return fields

        def emit_trace(**kwargs):
            if not trace:
                return
            try:
                self._record_candidate_trace(**trace_fields(**kwargs))
            except Exception:
                # Diagnostics are best effort and must never change retrieval.
                return None

        try:
            bm25 = self.local._bm25_scores(query) if mode != "dense" else None
            dense = self.local._cosine_scores(query) if mode != "bm25" else None
            bm25_ranked = order(bm25, True) if bm25 is not None else None
            dense_ranked = order(dense) if dense is not None else None

            if mode == "bm25":
                ranked, scores = bm25_ranked, bm25
            elif mode == "dense":
                ranked, scores = dense_ranked, dense
            else:
                fused_scores = [0.0] * len(self.chunks)
                for sequence in (bm25_ranked or [], dense_ranked or []):
                    for rank, i in enumerate(sequence, 1):
                        fused_scores[i] += 1.0 / (self.rrf_constant + rank)
                scores = fused_scores
                ranked = order(scores, True)

            stage = "reranking"
            if rerank and ranked:
                reranker_status = "running"
                outputs = self.reranker.rerank(query, [self.chunks[i].text for i in ranked], top_n=len(ranked))
                positions = [item.index for item in outputs]
                if (len(positions) != len(ranked) or set(positions) != set(range(len(ranked)))
                        or any(not math.isfinite(float(item.score)) for item in outputs)):
                    raise ValueError("重排返回的候选数、位置或得分不完整。")
                reranker_positions = {
                    ranked[item.index]: (rank, float(item.score))
                    for rank, item in enumerate(outputs, 1)
                }
                hits = [SearchHit(self.chunks[ranked[item.index]], float(item.score)) for item in outputs[:top_k]]
                final_positions = {ranked[item.index]: rank
                                   for rank, item in enumerate(outputs[:top_k], 1)}
                reranker_status = "complete"
            else:
                if rerank:
                    reranker_status = "empty_candidates"
                hits = [SearchHit(self.chunks[i], float(scores[i])) for i in ranked[:top_k]]
                final_positions = {i: rank for rank, i in enumerate(ranked[:top_k], 1)}

            stage = "evidence_expansion"
            context, missing = self.expand(hits)
        except Exception as exc:
            if stage == "reranking":
                reranker_status = "failed"
            emit_trace(status="failed", stage=stage, failure_type=type(exc).__name__)
            raise

        emit_trace(status="complete", stage="complete")
        return EvidenceResult(query, mode, hits, context, missing, rerank)


class StandardsQAIndex:
    """Compatibility view for shared multi-turn QA; retains the raw evidence trace."""

    def __init__(self, evidence: EvidenceIndex, mode="bm25", filters=None, use_reranker=False,
                 *, status_registry=None, catalog_index=None):
        self.evidence = evidence
        self.chunks = evidence.chunks
        self.mode = mode
        self.filters = filters or {}
        self.use_reranker = use_reranker
        self.status_registry = status_registry
        from .catalog_index import CatalogIndex
        self.catalog_index = catalog_index or CatalogIndex(evidence.corpus.catalog)
        self.last_result: EvidenceResult | None = None

    def discover(self, query: str, limit: int = 10) -> list[dict]:
        from .catalog_runtime import discover_catalog
        return discover_catalog(self.catalog_index, query, self.filters, limit)

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        from .catalog_runtime import explicit_standard_requests
        filters = deepcopy(self.filters)
        requested = explicit_standard_requests(query)
        if requested and not (filters.get("std_no") or filters.get("std_nos")):
            filters["std_nos"] = requested
        selected = filters.get("std_nos", [filters["std_no"]] if filters.get("std_no") else [])
        selected = [selected] if isinstance(selected, str) else selected
        bound_codes = selected or requested
        if bound_codes and self.catalog_index.records and all(
                not self.catalog_index.get(name)["matches"] or
                all(not row["text_available"] for row in self.catalog_index.get(name)["matches"])
                for name in bound_codes):
            self.last_result = EvidenceResult(query, self.mode, [], [])
            return []
        self.last_result = self.evidence.retrieve(query, self.mode, top_k, filters, self.use_reranker)
        if self.status_registry is not None:
            # Only the current QA context gets fresh saved status metadata.
            # Preserve indexed text, vectors, raw ranks, corpus and old runs.
            current_hits = []
            statuses = {}
            for hit in self.last_result.context_hits:
                locator = deepcopy(hit.chunk.locator)
                key = locator["std_no"], locator.get("clause_no")
                if key not in statuses:
                    statuses[key] = self.status_registry.check(*key)
                status = statuses[key]
                for field in ("standard_status", "current_mandatory_status", "verification_date",
                              "status_checked_at"):
                    locator[field] = status[field]
                locator["status_source_ids"] = deepcopy(status["source_ids"])
                current_hits.append(SearchHit(replace(hit.chunk, locator=locator), hit.score))
            self.last_result = replace(self.last_result, context_hits=current_hits)
        return self.last_result.context_hits


def load_evidence_index(output_dir: str | Path, **kwargs) -> EvidenceIndex:
    return EvidenceIndex(load_corpus(output_dir), **kwargs)
