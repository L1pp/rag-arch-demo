"""Candidate directory discovery; no clause, model or embedding evidence is created."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from contextvars import ContextVar
import math
from pathlib import Path
import re
from threading import Condition, RLock
import unicodedata

from .models import Chunk
from .async_support import run_local
from .retrieval import LocalIndex, tokenize
from .standard_identifiers import parse_standard_references


def _key(value):
    return re.sub(r"[^A-Z0-9_.\-\u3400-\u9fff]", "", unicodedata.normalize("NFKC", str(value)).upper().replace("—", "-").replace("–", "-"))


def _strings(value):
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


class CatalogIndex:
    """Accept a catalog list, catalog JSON object, or loaded StandardCorpus.

    search returns directory records; get returns an exact-version envelope.
    None of these records are engineering evidence or complete clause IDs.
    """

    def __init__(self, catalog, embeddings=None):
        if hasattr(catalog, "catalog"):
            catalog = catalog.catalog
        elif isinstance(catalog, dict):
            catalog = catalog.get("standards", [])
        if not isinstance(catalog, list):
            raise ValueError("catalog须为目录列表或含standards列表的对象。")
        self.records = deepcopy(catalog)
        self.embeddings = embeddings
        self._embedding_lock = RLock()
        self._embedding_condition = Condition(self._embedding_lock)
        self._embedding_generation = 0
        self._metadata_initializing = False
        self._metadata_index = None
        self._metadata_embedding_attempted = False
        self._async_embedding_lock = None
        self._candidate_async_override = ContextVar(
            f"catalog_async_hits_{id(self)}", default=None)
        self._candidate_diagnostics = ContextVar(
            f"catalog_candidate_diagnostics_{id(self)}", default=None)
        self._identities, self._exact_identities, self._tokens = [], [], []
        self._identifier_refs = []
        self._metadata_texts = []
        ids = set()
        for record in self.records:
            if not isinstance(record, dict) or not isinstance(record.get("std_id"), str) or not record["std_id"]:
                raise ValueError("目录条目须有唯一非空std_id。")
            if record["std_id"] in ids:
                raise ValueError("目录std_id重复。")
            ids.add(record["std_id"])
            code = record.get("code", record.get("std_no"))
            if not isinstance(code, str) or not code.strip() or not isinstance(record.get("title"), str):
                raise ValueError("目录条目须有规范号和中文题名。")
            names = [code, record["std_id"], *_strings(record.get("aliases"))]
            identities = {_key(name) for name in names if name.strip()}
            exact_identities = set(identities)
            for name in names:
                for parsed in parse_standard_references(name):
                    normalized = _key(parsed["normalized"])
                    identities.add(normalized)
                    exact_identities.add(normalized)
            identities.update(re.sub(r"-\d{4}$", "", alias) for alias in tuple(identities))
            self._identities.append(identities)
            self._exact_identities.append(exact_identities)
            self._identifier_refs.append({
                "code": parse_standard_references(code),
                "aliases": [item for alias in _strings(record.get("aliases"))
                            for item in parse_standard_references(alias)],
            })
            text = " ".join([*names, record["title"], *_strings(record.get("tags")),
                             *_strings(record.get("search_tags")), *_strings(record.get("retrieval_tags")),
                             *_strings(record.get("domain_tags")), *_strings(record.get("function_tags")),
                             *_strings(record.get("trigger_tags")), *_strings(record.get("scope_summary")),
                             *_strings(record.get("history_search_texts"))])
            self._tokens.append(set(tokenize(text)))
            metadata = " ".join([code, record["title"], *_strings(record.get("aliases")),
                                 *_strings(record.get("tags")),
                                 *_strings(record.get("search_tags")),
                                 *_strings(record.get("retrieval_tags")),
                                 *_strings(record.get("domain_tags")),
                                 *_strings(record.get("function_tags")),
                                 *_strings(record.get("trigger_tags")),
                                 *_strings(record.get("scope_summary")),
                                 *_strings(record.get("history_search_texts"))])
            self._metadata_texts.append(metadata)

    def _new_candidate_state(self):
        state = {"warnings": [], "semantic_used": False, "diagnostic": None}
        self._candidate_diagnostics.set(state)
        return state

    def _candidate_state(self):
        state = self._candidate_diagnostics.get()
        if state is None:
            state = self._new_candidate_state()
        return state

    @property
    def last_candidate_diagnostic(self):
        return deepcopy(self._candidate_state()["diagnostic"])

    @last_candidate_diagnostic.setter
    def last_candidate_diagnostic(self, value):
        state = self._candidate_state()
        state["diagnostic"] = deepcopy(value)
        if value is not None:
            state["warnings"].append(str(value.get("message") or value.get("stage") or "embedding异常"))

    @property
    def last_diagnostics(self):
        """Diagnostics for the latest candidates() call in this request context."""
        state = self._candidate_state()
        result = {"warnings": list(dict.fromkeys(state["warnings"])),
                  "semantic_used": bool(state["semantic_used"])}
        if state["diagnostic"] is not None:
            result["diagnostic"] = deepcopy(state["diagnostic"])
        return result

    def set_embeddings(self, embeddings=None):
        """Replace the optional metadata provider and drop only its lazy index."""
        with self._embedding_lock:
            self.embeddings = embeddings
            self._embedding_generation += 1
            self._metadata_index = None
            self._metadata_embedding_attempted = False
        return self

    def prepare_embeddings(self):
        """Build the optional metadata-only vector index without changing search semantics."""
        return self._metadata_index_for_candidates()

    @staticmethod
    def _visible(record, score=None):
        metadata_only = record.get("entry_kind") == "metadata_only" or record.get("library_class") == "C"
        readable = False if metadata_only or record.get("entry_kind") == "source_only" else bool(record.get("text_available", record.get("fulltext_available", record.get("record_count", 0))))
        original = False if metadata_only else bool(record.get("original_available", record.get("source_files")))
        page_reading = original and bool(record.get("page_reading_available", any(
            Path(item.get("path", "")).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
            for item in record.get("source_files", []))))
        grade = "C" if metadata_only else record.get("library_class") or ("A" if readable else "B" if record.get("source_files") else "C")
        fields = ("std_id", "title", "aliases", "tags", "search_tags", "retrieval_tags", "domain_tags", "function_tags", "trigger_tags", "scope_summary", "metadata_claims", "metadata_sources", "source_adapters",
                  "metadata_verification", "verification_status", "verification", "review_state", "verified_at",
                  "status_claim", "status_checked_at", "implementation_date", "publication_date", "source_files")
        value = {field: deepcopy(record[field]) for field in fields if field in record}
        value.update(code=record.get("code", record.get("std_no")),
                     std_no=record.get("std_no", record.get("code")),
                     catalog_id="CATALOG/" + record["std_id"], discovery_only=True,
                     library_class=grade, fulltext_available=readable, text_available=readable,
                     original_available=original, page_reading_available=page_reading,
                     metadata_sources=deepcopy(record.get("metadata_sources", [])),
                     metadata_verification=deepcopy(record.get("metadata_verification", record.get("verification_status", "unknown"))),
                     usage_note="目录仅用于发现、库存及来源定位，不是条款正文、现行性或工程合规证据。")
        if not value["metadata_sources"] and original:
            # Older A catalogs and source-only B entries already have concrete
            # registered originals.  Those paths support inventory facts only.
            value["metadata_sources"] = [
                {**{key: deepcopy(source[key]) for key in ("source_id", "path", "url") if source.get(key)},
                 "kind": "registered_original_inventory", "verification": "inventory_only_not_legal_status"}
                for source in record.get("source_files", [])
                if isinstance(source, dict) and (source.get("path") or source.get("url"))]
        if score is not None:
            value["score"] = round(score, 6)
        return value

    def search(self, query, limit=10, library_class=None):
        if not isinstance(query, str) or not query.strip():
            return []
        if type(limit) is not int or limit < 1:
            raise ValueError("limit须为正整数。")
        if library_class not in {None, "A", "B", "C"}:
            raise ValueError("library_class只能是A/B/C或null。")
        terms, key = set(tokenize(query)), _key(query)
        hits = []
        for record, identities, words in zip(self.records, self._identities, self._tokens):
            visible = self._visible(record)
            if library_class is not None and visible["library_class"] != library_class:
                continue
            overlap = terms & words
            exact = key in identities
            if not overlap and not exact:
                continue
            score = len(overlap) / math.sqrt(max(1, len(words)))
            score += 100.0 if exact else 0.0
            if query.strip() == record["title"].strip():
                score += 80.0
            elif query.strip() in record["title"]:
                score += 8.0
            hits.append(self._visible(record, score))
        return sorted(hits, key=lambda item: (-item["score"], item["std_id"]))[:limit]

    @staticmethod
    def _candidate_note_value(value):
        return value if value not in (None, "") else "未提供"

    def _identifier_match(self, requested, record_index):
        """Return a scored, explicitly qualified code relationship."""
        refs = self._identifier_refs[record_index]
        code_refs = refs["code"]
        alias_refs = refs["aliases"]
        requested_key = _key(requested["normalized"])
        for candidate in code_refs:
            if requested_key == _key(candidate["normalized"]):
                return {"score": 1100.0, "method": "exact_code", "candidate_only": False,
                        "notes": ["目录 code 与请求规范化编号一致；候选结果不替代现行性或适用性核对。"]}
        for candidate in alias_refs:
            if requested_key == _key(candidate["normalized"]):
                return {"score": 1080.0, "method": "registered_alias", "candidate_only": False,
                        "notes": ["匹配目录正式 aliases；别名身份仍由状态层核对，不据此扩展同版关系。"]}

        # A code relationship must retain the prefix, number, part and year
        # differences.  A shared number alone is a weak directory candidate.
        best = None
        for candidate in code_refs:
            if requested.get("number") != candidate.get("number"):
                continue
            prefix_same = requested.get("prefix") == candidate.get("prefix")
            part_same = requested.get("part") == candidate.get("part")
            notes = []
            if not prefix_same:
                notes.append(f"前缀不同：请求 {requested['prefix']}，目录 {candidate['prefix']}。")
            if not part_same:
                notes.append("分册不同：请求 {}，目录 {}。".format(
                    self._candidate_note_value(requested.get("part")),
                    self._candidate_note_value(candidate.get("part"))))
            req_kind, req_year = requested.get("year_kind"), requested.get("year_text")
            cand_kind, cand_year = candidate.get("year_kind"), candidate.get("year_text")
            if req_kind == "missing":
                if cand_year is not None:
                    notes.append(f"请求未提供年份，目录登记年份为 {cand_year}。")
            elif req_kind == "two_digit":
                notes.append(f"输入保留两位年份 {req_year}，未补世纪；目录年份为 {self._candidate_note_value(cand_year)}，需核对。")
            elif req_kind == "malformed":
                notes.append(f"输入年份 {req_year} 格式异常，未按年份同版处理。")
            elif req_year != cand_year:
                notes.append(f"年份不同：请求 {req_year}，目录 {self._candidate_note_value(cand_year)}。")
            if not notes and prefix_same and part_same and req_year == cand_year:
                # This path is normally covered by normalized exact matching;
                # retain a conservative fallback for unusual spacing forms.
                notes.append("编号字段相同，但未达到目录精确身份匹配。")

            if not prefix_same:
                method = "prefix_variant"
            elif not part_same:
                method = "part_variant"
            elif req_kind == "two_digit":
                method = "two_digit_year_candidate"
            elif req_kind == "malformed":
                method = "malformed_year_candidate"
            elif req_kind == "missing":
                method = "family_prefix"
            elif req_year != cand_year:
                method = "year_variant"
            else:
                method = "identifier_variant"
            score = 820.0 + (100.0 if prefix_same else 20.0) + (40.0 if part_same else 0.0)
            score += 20.0 if req_kind == cand_kind else 0.0
            if (req_kind == "two_digit" and isinstance(req_year, str)
                    and isinstance(cand_year, str) and cand_year.endswith(req_year)):
                score += 10.0
                notes.append("目录年份末两位相同，仅作候选排序参考，未补世纪。")
            row = {"score": score, "method": method, "candidate_only": True,
                   "notes": notes or ["编号相近但未确认同版。"]}
            if best is None or row["score"] > best["score"]:
                best = row
        return best

    def _metadata_index_for_candidates(self):
        while True:
            with self._embedding_condition:
                while self._metadata_initializing:
                    self._embedding_condition.wait()
                if self.embeddings is None:
                    return None
                if self._metadata_embedding_attempted:
                    return self._metadata_index
                embeddings = self.embeddings
                generation = self._embedding_generation
                self._metadata_initializing = True

            # LocalIndex asks for prepared metadata vectors here. Build it
            # outside the shared lock so configuration and query requests can
            # continue while initialization performs cache or provider work.
            try:
                chunks = [Chunk(record["std_id"], record.get("code", record.get("std_no", "")),
                                f"catalog:{record['std_id']}", text, {}, "catalog", "catalog")
                          for record, text in zip(self.records, self._metadata_texts)]
                index = LocalIndex(chunks, embeddings)
            except BaseException as exc:
                with self._embedding_condition:
                    self._metadata_initializing = False
                    if generation == self._embedding_generation and isinstance(exc, Exception):
                        self._metadata_index = None
                        self._metadata_embedding_attempted = False
                        self.last_candidate_diagnostic = {
                            "stage": "metadata_embedding_initialization",
                            "error_type": type(exc).__name__,
                            "message": f"目录元数据向量初始化失败（{type(exc).__name__}）。",
                        }
                    self._embedding_condition.notify_all()
                if not isinstance(exc, Exception):
                    raise
                return None

            with self._embedding_condition:
                self._metadata_initializing = False
                if (generation == self._embedding_generation
                        and embeddings is self.embeddings):
                    self._metadata_index = index
                    self._metadata_embedding_attempted = True
                    self._embedding_condition.notify_all()
                    return index
                self._embedding_condition.notify_all()

    def _metadata_embedding_hits(self, query, limit):
        override = self._candidate_async_override.get()
        if override is not None:
            hits, semantic_used, diagnostic = override
            self._candidate_state()["semantic_used"] = bool(semantic_used)
            if diagnostic is not None:
                self.last_candidate_diagnostic = diagnostic
            return hits
        if self.embeddings is None:
            return {}
        index = self._metadata_index_for_candidates()
        if index is None:
            return {}
        self._candidate_state()["semantic_used"] = True
        try:
            # Query embedding may wait on a remote provider. Do not hold the
            # metadata-index lock across that call; the index is immutable.
            hits = index.search(query, top_k=min(len(self.records), max(20, limit * 4)))
        except Exception as exc:
            self.last_candidate_diagnostic = {
                "stage": "metadata_embedding_query",
                "error_type": type(exc).__name__,
                "message": f"目录元数据向量查询失败（{type(exc).__name__}）。",
            }
            return {}
        return {hit.chunk.doc_id: float(hit.score) for hit in hits}

    async def _ametadata_index_for_candidates(self):
        if self._async_embedding_lock is None:
            with self._embedding_lock:
                if self._async_embedding_lock is None:
                    self._async_embedding_lock = asyncio.Lock()
        async with self._async_embedding_lock:
            while True:
                with self._embedding_lock:
                    if self.embeddings is None:
                        return None
                    if self._metadata_embedding_attempted:
                        return self._metadata_index
                    embeddings = self.embeddings
                    generation = self._embedding_generation
                chunks = await run_local(lambda: [
                    Chunk(record["std_id"], record.get("code", record.get("std_no", "")),
                          f"catalog:{record['std_id']}", text, {}, "catalog", "catalog")
                    for record, text in zip(self.records, self._metadata_texts)])
                try:
                    index = await LocalIndex.acreate(chunks, embeddings)
                except Exception as exc:
                    with self._embedding_lock:
                        if generation == self._embedding_generation and embeddings is self.embeddings:
                            self._metadata_index = None
                            self._metadata_embedding_attempted = False
                            self.last_candidate_diagnostic = {
                                "stage": "metadata_embedding_initialization",
                                "error_type": type(exc).__name__,
                                "message": f"目录元数据向量初始化失败（{type(exc).__name__}）。",
                            }
                            return None
                    continue
                with self._embedding_lock:
                    if generation == self._embedding_generation and embeddings is self.embeddings:
                        self._metadata_index = index
                        self._metadata_embedding_attempted = True
                        return index

    async def _ametadata_embedding_hits(self, query, limit):
        if self.embeddings is None:
            return {}, False, None
        index = await self._ametadata_index_for_candidates()
        if index is None:
            return {}, False, self.last_candidate_diagnostic
        try:
            hits = await index.asearch(query, top_k=min(len(self.records), max(20, limit * 4)))
        except Exception as exc:
            diagnostic = {
                "stage": "metadata_embedding_query",
                "error_type": type(exc).__name__,
                "message": f"目录元数据向量查询失败（{type(exc).__name__}）。",
            }
            self.last_candidate_diagnostic = diagnostic
            return {}, True, diagnostic
        return {hit.chunk.doc_id: float(hit.score) for hit in hits}, True, None

    async def asearch(self, query, limit=10, library_class=None):
        """Async lexical catalog search; it does not need a provider call."""
        return await run_local(self.search, query, limit, library_class)

    async def acandidates(self, query, limit=5, library_class=None):
        """Async candidate lookup using async metadata embedding providers."""
        if not isinstance(query, str) or not query.strip():
            self._new_candidate_state()
            return []
        if type(limit) is not int or limit < 1:
            raise ValueError("limit须为正整数。")
        if library_class not in {None, "A", "B", "C"}:
            raise ValueError("library_class只能是A/B/C或null。")
        query = query.strip()
        self._new_candidate_state()

        def semantic_is_needed():
            requested = parse_standard_references(query)
            exact_found = False
            identifier_candidate_found = False
            for requested_item in requested:
                for index in range(len(self.records)):
                    relation = self._identifier_match(requested_item, index)
                    if relation is not None:
                        identifier_candidate_found = True
                        exact_found = exact_found or not relation["candidate_only"]
            return (self.embeddings is not None and not exact_found
                    and (not requested or not identifier_candidate_found))

        need_semantic = await run_local(semantic_is_needed)
        override = ({}, False, None)
        if need_semantic:
            override = await self._ametadata_embedding_hits(query, limit)
        token = self._candidate_async_override.set(override)
        try:
            rows, state = await run_local(
                lambda: (self.candidates(query, limit, library_class),
                         deepcopy(self._candidate_state())))
        finally:
            self._candidate_async_override.reset(token)
        self._candidate_diagnostics.set(state)
        return rows

    def candidates(self, query, limit=5, library_class=None):
        """Return public directory candidates without promoting identity.

        Number/prefix/year relationships are ranked before metadata lexical or
        embedding matches.  Only public directory fields feed the optional
        metadata vector index; no clause or body text is added.
        """
        state = self._new_candidate_state()
        if not isinstance(query, str) or not query.strip():
            return []
        if type(limit) is not int or limit < 1:
            raise ValueError("limit须为正整数。")
        if library_class not in {None, "A", "B", "C"}:
            raise ValueError("library_class只能是A/B/C或null。")
        query = query.strip()
        requested = parse_standard_references(query)
        requested_identifier = deepcopy(requested[0]) if requested else {"raw": query}
        identifier_only = bool(requested) and len(requested) == 1 and _key(query) == _key(requested[0]["raw"])
        candidates = {}
        method_rank = {
            "exact_code": 5, "registered_alias": 5, "family_prefix": 4,
            "two_digit_year_candidate": 4, "malformed_year_candidate": 4,
            "year_variant": 4, "part_variant": 4, "prefix_variant": 4,
            "identifier_variant": 4, "metadata_embedding": 2,
            "metadata_lexical": 1,
        }

        def add(record_index, score, method, candidate_only, notes, requested_value):
            record = self.records[record_index]
            visible = self._visible(record)
            if library_class is not None and visible["library_class"] != library_class:
                return
            std_id = record["std_id"]
            row = {"score": float(score), "method": method,
                   "candidate_only": bool(candidate_only), "notes": notes,
                   "requested_identifier": requested_value,
                   "record_index": record_index}
            previous = candidates.get(std_id)
            if previous is None:
                candidates[std_id] = row
                return
            old_rank = method_rank.get(previous["method"], 0)
            new_rank = method_rank.get(method, 0)
            if new_rank > old_rank or (new_rank == old_rank and score > previous["score"]):
                candidates[std_id] = row

        exact_found = False
        identifier_candidate_found = False
        if requested:
            for requested_item in requested:
                for index in range(len(self.records)):
                    relation = self._identifier_match(requested_item, index)
                    if relation is None:
                        continue
                    identifier_candidate_found = True
                    add(index, relation["score"], relation["method"], relation["candidate_only"],
                        relation["notes"], deepcopy(requested_item))
                    exact_found = exact_found or not relation["candidate_only"]

        # An identifier-only query must not turn a shared year/prefix token
        # into unrelated catalog matches.  Topic text may still use lexical
        # and semantic metadata recall when it accompanies an identifier.
        if not identifier_only:
            terms, key = set(tokenize(query)), _key(query)
            for index, (record, identities, words) in enumerate(zip(self.records, self._identities, self._tokens)):
                visible = self._visible(record)
                if library_class is not None and visible["library_class"] != library_class:
                    continue
                overlap = terms & words
                exact = key in identities
                if not overlap and not exact:
                    continue
                score = len(overlap) / math.sqrt(max(1, len(words))) + (100.0 if exact else 0.0)
                if query == record["title"].strip():
                    score += 80.0
                elif query in record["title"]:
                    score += 8.0
                add(index, score, "metadata_lexical", True,
                    ["按目录公开字段词法匹配；未确认编号、年份或分册同版。"],
                    requested_identifier)

        if (self.embeddings is not None and not exact_found
                and (not requested or not identifier_candidate_found)):
            for std_id, score in self._metadata_embedding_hits(query, limit).items():
                index = next((pos for pos, row in enumerate(self.records) if row["std_id"] == std_id), None)
                if index is None:
                    continue
                add(index, score, "metadata_embedding", True,
                    ["按目录公开元数据向量召回；编号、前缀、分册、年份均需核对，候选不代表同版。"],
                    requested_identifier)

        rows = []
        for value in candidates.values():
            row = self._visible(self.records[value["record_index"]], value["score"])
            row.update(candidate_only=value["candidate_only"], match_method=value["method"],
                       match_notes=value["notes"], requested_identifier=value["requested_identifier"])
            if self.last_candidate_diagnostic is not None:
                row["diagnostic"] = deepcopy(self.last_candidate_diagnostic)
            rows.append((value, row))
        rows.sort(key=lambda item: (-method_rank.get(item[0]["method"], 0),
                                    -item[0]["score"], item[1]["std_id"]))
        return [row for _, row in rows[:limit]]

    def get(self, std_no):
        if not isinstance(std_no, str) or not std_no.strip():
            raise ValueError("std_no须为非空字符串。")
        parsed = parse_standard_references(std_no)
        requested = (_key(parsed[0]["normalized"])
                     if len(parsed) == 1 and _key(parsed[0]["raw"]) == _key(std_no)
                     else _key(std_no))
        matches = []
        year = re.search(r"-\d{4}$", requested)
        for record, identities, exact_identities in zip(self.records, self._identities, self._exact_identities):
            actual_value = record.get("code", record.get("std_no", ""))
            actual_refs = parse_standard_references(actual_value)
            actual = (_key(actual_refs[0]["normalized"]) if actual_refs else _key(actual_value))
            if requested in identities and (not year or actual == requested or requested in exact_identities):
                matches.append(self._visible(record))
        matches.sort(key=lambda item: item["std_id"])
        return {"ok": True, "status": "ok" if len(matches) == 1 else "ambiguous" if matches else "not_found",
                "catalog_found": bool(matches), "discovery_only": True, "matches": matches}

    def source_missing(self, std_no, clause_no=None):
        """Missing-text envelope distinguishes collected originals from C gaps."""
        result = self.get(std_no)
        matches = result["matches"]
        status, reason, action = result["status"], "standard_not_found", None
        message = "目录未唯一命中；不证明标准不存在。"
        if status == "ambiguous":
            reason, action = "ambiguous_standard", "choose_standard_version"
        if len(matches) == 1:
            entry = matches[0]
            if entry["text_available"]:
                status, reason, action = "clause_not_found", "clause_not_found", "check_clause_locator"
                message = "已有文字，但本次未取得所需条文；不证明原标准没有该条。"
            elif entry["original_available"]:
                status, reason = "source_missing", "text_not_prepared"
                action = "read_original_pages" if entry["page_reading_available"] else "prepare_original_adapter"
                message = "已取得原件，尚未准备检索文字；可按已登记能力读取原页或准备格式适配。"
            else:
                status, reason, action = "source_missing", "fulltext_not_collected", "acquire_original"
                message = "已登记该版本，本地未取得原件或正文；只能提供目录和获取来源。"
        return {**result, "status": status, "reason": reason, "next_action": action,
                "std_no": std_no, "clause_no": clause_no,
                "found": False, "text_available": matches[0]["text_available"] if len(matches) == 1 else None,
                "original_available": matches[0]["original_available"] if len(matches) == 1 else None,
                "page_reading_available": matches[0]["page_reading_available"] if len(matches) == 1 else None,
                "catalog_info": matches[0] if len(matches) == 1 else None,
                "evidence": [], "raw_ids": [], "context_ids": [], "complete_context_ids": [],
                "context_complete": False, "message": message}
