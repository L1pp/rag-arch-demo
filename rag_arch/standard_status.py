"""Local, evidence-qualified status lookup; no network/model calls at runtime.

Publication, whole-standard registry status and repeal of former mandatory
clauses are separate facts. Absence from the repeal register establishes none
of those facts. Results describe the saved verification date, not live law.
"""
from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit


from .config import ROOT as DEFAULT_PROJECT_ROOT
from .standard_lifecycle import LifecycleRegistry, collect_status_observations
STATUS_SOURCES_PATH = "data/processed/regulations_v0/status_sources.json"


def _key(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).upper().translate(str.maketrans({
        "－": "-", "—": "-", "（": "(", "）": ")", "〔": "[", "〕": "]", "．": ".",
    }))


def _edition_identity(value):
    """Read explicit edition syntax without inventing a century or revision."""
    match = re.fullmatch(
        r"(?P<stem>.+?)(?P<separator>[-:])(?P<year>\d{4}|\d{2})"
        r"(?P<revision>\((?:\d{4}年?)?修订[本版]?\)|(?:\d{4}年?)?修订[本版]?)?", _key(value))
    if not match:
        return None
    parts = match.groupdict()
    revision_year = re.search(r"\d{4}", parts["revision"] or "")
    return {**parts, "revision": bool(parts["revision"]),
            "revision_year": revision_year.group() if revision_year else None}


def _explicit_canonical_edition(value):
    parts = _edition_identity(value)
    return bool(parts and (len(parts["year"]) == 4 or
                           parts["stem"].startswith("CECS") and parts["separator"] == ":"))


_QUALIFIED_EDITION_SUFFIX = re.compile(
    r"\((?:\d{2}|\d{4})年?(?:修正(?:本|版)?|修订(?:本|版)?|年版|版)\)$")


def _unqualified_edition_code(value):
    """Return the literal base of a registered code with a dated qualifier."""
    key = _key(value)
    return _QUALIFIED_EDITION_SUFFIX.sub("", key) if _QUALIFIED_EDITION_SUFFIX.search(key) else None


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(value: str) -> list[str]:
    return [v.strip() for v in value.split(";") if v.strip()]



def _catalog_text_available(entry):
    if entry.get("entry_kind") in {"metadata_only", "source_only"} or entry.get("library_class") in {"B", "C"}:
        return False
    for key in ("fulltext_available", "text_available"):
        if key in entry:
            return entry[key] is True
    return bool(entry.get("record_count", 0))


_WHOLE_SUCCESSOR_TYPES = {"replaces", "merges", "splits", "abolishes_with_successor",
                          "whole_replacement", "merge_replacement", "split_replacement"}
_VERIFICATION_FIELDS = ("verification_status", "verification_state", "verification",
                        "review_state", "state", "status")
_CONFIRMED_STATES = {"confirmed", "verified", "verified_primary_source", "verified_official_source",
                     "documented", "inherited_documented", "official_documented", "source_verified",
                     "verified_for_declared_relation", "verified_document_relation",
                     "accepted_user_review_report"}


def _relation_confirmed(edge):
    """Confirm the declared fact, separately from permission to infer successors.

    An explicit fact flag permits confirmed non-successor facts to retain a
    false chain flag. Legacy false flags and uncertainty markers remain vetoes.
    """
    explicit = edge.get("relation_confirmed")
    if explicit is not None and explicit is not True:
        return False
    values = [edge[key] for key in _VERIFICATION_FIELDS if edge.get(key) is not None]
    negative = ("unverified", "unconfirmed", "not_verified", "not_confirmed", "unknown", "unresolved",
                "candidate", "pending", "conflict", "lead", "待核", "未核", "未确认", "冲突")
    if any(value is False or str(value).strip().lower() in {"false", "0", "no", "否"}
           or any(token in str(value).lower() for token in negative) for value in values):
        return False
    flag = edge.get("use_in_confirmed_replacement_chain")
    if explicit is True:
        return True
    if flag is not None and flag is not True:
        return False
    if flag is True or any(str(value).lower() in _CONFIRMED_STATES for value in values):
        return True
    evidence = edge.get("evidence", [])
    evidence = evidence if isinstance(evidence, list) else []
    return any(str(item[key]).lower() in _CONFIRMED_STATES
               for item in evidence if isinstance(item, dict)
               for key in (*_VERIFICATION_FIELDS, "evidence_level") if item.get(key) is not None)


def _confirmed_replacement_edge(edge):
    """Only confirmed whole-standard relations authorize successor traversal."""
    if edge.get("relation") not in _WHOLE_SUCCESSOR_TYPES or not _relation_confirmed(edge):
        return False
    return all(edge.get(key) is None or edge[key] is True for key in
               ("use_in_confirmed_replacement_chain", "whole_standard_successor_allowed"))


def _relation_view(edge):
    return {**edge, "relation_confirmed": _relation_confirmed(edge),
            "whole_standard_successor_allowed": _confirmed_replacement_edge(edge)}


class StandardStatusRegistry:
    """Load once when enriching a corpus or wiring the three local tools."""

    def __init__(self, project_root: str | Path = DEFAULT_PROJECT_ROOT,
                 clauses: list[dict] | None = None, *, catalog=None, relations=None, corpus_dir=None,
                 as_of_date=None, corpus=None):
        self.root = Path(project_root)
        self.standards = _rows(self.root / "data/meta/standards.csv")
        self.sources = {r["source_id"]: r for r in _rows(self.root / "data/meta/sources.csv")}
        self._load_catalog(catalog, relations, corpus_dir)
        verified = self.root / "data/meta/abolished_clauses_verified.csv"
        self.relations_path = verified if verified.exists() else self.root / "data/meta/abolished_clauses.csv"
        self.relations = _rows(self.relations_path)
        relation_keys = [(_key(r["std_no"]), _key(r["clause_no"])) for r in self.relations]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("Duplicate standard/clause repeal relations")
        source_path = self.root / STATUS_SOURCES_PATH
        self.verification = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
        self.evidence = list(self.verification.get("sources", []))
        for observation in collect_status_observations(self.catalog):
            if observation not in self.evidence:
                self.evidence.append(observation)
            source_id = observation.get("source_id")
            if source_id:
                self.sources.setdefault(source_id, observation)
        self.lifecycle = LifecycleRegistry(self.catalog, self.evidence,
            [edge for edge in self._version_edges if _confirmed_replacement_edge(edge)], as_of_date)
        status_records = getattr(corpus, "status_records", None) if corpus is not None else None
        if not callable(status_records):
            status_records = getattr(clauses, "status_records", None)
        if callable(status_records):
            clauses = status_records()
        elif clauses is None and corpus is not None:
            clauses = getattr(corpus, "clauses", [])
        if clauses is None:
            clauses = []
            # Reviewed local stages are permitted as existence evidence. They
            # never establish legal status and there is no recursive disk scan.
            if corpus_dir is not None:
                directory = Path(corpus_dir)
                if not directory.is_absolute():
                    directory = self.root / directory
                clauses.extend(_read_jsonl(directory / "clauses.jsonl"))
            for relative in (() if corpus_dir is not None else (
                "data/corpus/clauses.jsonl", "data/processed/standards_v0/clauses.jsonl",
                "data/processed/regulations_v0/clauses.jsonl", "data/processed/jgj162_v0/clauses.jsonl",
                "data/processed/jgj130_v0/clauses.jsonl", "data/processed/concrete_scoped_v0/clauses.jsonl",
            )):
                clauses.extend(_read_jsonl(self.root / relative))
        self.clause_keys = set()
        self._clause_mandatory_metadata = {}
        for clause in clauses:
            document_part = clause.get("document_part", "body")
            if (document_part == "body"
                    or clause.get("import_origin") == "manifest_v1" and document_part != "commentary"):
                clause_key = (_key(clause["std_no"]), _key(clause["clause_no"]))
                self.clause_keys.add(clause_key)
            # Commentary and mere clause-number matches are not publication proof.
            if document_part != "body":
                continue
            clause_key = (_key(clause["std_no"]), _key(clause["clause_no"]))
            # Ordinary dict-backed corpora retain only fields used by status
            # reasoning. Lazy record mappings are already lightweight views;
            # preserving them lets original_mandatory_evidence decode only
            # when a matching status query needs it.
            if isinstance(clause, dict):
                clause = {field: clause[field] for field in (
                    "std_no", "clause_no", "document_part", "import_origin",
                    "original_mandatory", "original_mandatory_evidence",
                ) if field in clause}
            self._clause_mandatory_metadata.setdefault(clause_key, []).append(clause)

    def _original_nonmandatory_sources(self, std_key, clause):
        """Accept an explicit False only with a reviewed complete edition list.

        Raw corpus flags, wording such as 'must', and repeal-register misses do
        not establish either publication identity or nonmandatory status.
        """
        rows = self._clause_mandatory_metadata.get((std_key, clause), [])
        if any(row.get("original_mandatory") is True for row in rows):
            return []
        proofs = []
        for row in rows:
            if row.get("original_mandatory") is not False:
                continue
            saved = row.get("original_mandatory_evidence")
            for proof in saved if isinstance(saved, list) else [saved]:
                if not isinstance(proof, dict):
                    continue
                code = proof.get("std_no")
                edition = _edition_identity(code) if isinstance(code, str) else None
                if (proof.get("evidence_type") != "complete_original_mandatory_clause_list"
                        or proof.get("scope") != "entire_original_publication_body"
                        or proof.get("complete_list") is not True
                        or proof.get("verification_status") != "verified_official_source"
                        or proof.get("conflicts")
                        or not edition or len(edition["year"]) != 4 or _key(code) != std_key
                        or not isinstance(proof.get("source_url"), str)
                        or proof.get("source_type") not in {
                            "official_government_publication", "official_government_reprint"}):
                    continue
                try:
                    published = date.fromisoformat(proof["publication_date"])
                    checked = date.fromisoformat(proof["checked_at"])
                    url = urlsplit(proof["source_url"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (not published <= checked <= self.lifecycle.as_of
                        or url.scheme != "https" or not (url.hostname or "").endswith(".gov.cn")
                        or url.username or url.password
                        or not isinstance(proof.get("source_id"), str) or not proof["source_id"].strip()
                        or not isinstance(proof.get("notice_number"), str) or not proof["notice_number"].strip()):
                    continue
                numbers = proof.get("mandatory_clause_nos")
                if (not isinstance(numbers, list) or not numbers
                        or any(not isinstance(number, str) or not re.fullmatch(
                            r"\d+(?:\.\d+)+(?:\(\d+\))?", _key(number)) for number in numbers)):
                    continue
                keys = {_key(number) for number in numbers}
                if len(keys) != len(numbers):
                    continue
                proofs.append((keys, proof))
        if not proofs or any(keys != proofs[0][0] for keys, _ in proofs):
            return []
        # A listed child/parent clause prevents an expanded negative conclusion.
        if any(number == clause or clause.startswith((number + "(", number + "."))
               or number.startswith((clause + "(", clause + "."))
               for number in proofs[0][0]):
            return []
        return [{**proof, "evidence_level": "official_publication_notice",
                 "supports_original_nonmandatory": True} for _, proof in proofs]


    def _load_catalog(self, catalog, relations, corpus_dir):
        directory = Path(corpus_dir) if corpus_dir is not None else self.root / "data/corpus"
        if not directory.is_absolute():
            directory = self.root / directory
        def metadata(value, name, key):
            if value is None:
                path = directory / name
                value = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else []
            return value.get(key, []) if isinstance(value, dict) else value
        self.catalog = list(metadata(catalog, "catalog.json", "standards") or [])
        self.version_relations = list(metadata(relations, "relations.json", "relations") or [])
        self._catalog_by_code = {_key(r.get("std_no", r.get("code", ""))): r for r in self.catalog}
        self._catalog_by_id = {r["std_id"]: r for r in self.catalog}
        self._version_nodes = {}
        self._version_edges = []
        self._lifecycle_facts = []
        self._identity_relations = []
        self._historical_ids = {}
        existing = {_key(r["std_no"]) for r in self.standards}
        for entry in self.catalog:
            code = entry.get("std_no", entry.get("code"))
            for source in entry.get("source_files", []):
                sid = source.get("source_id")
                if sid and sid not in self.sources:
                    self.sources[sid] = {"source_id": sid, "source_url": source.get("url", ""),
                                         "source_path": source.get("path"), "title": entry.get("title", code)}
            if _key(code) not in existing:
                self.standards.append({"doc_id": entry["std_id"], "std_no": code, "title": entry["title"],
                    "status": ("local_text_registered" if _catalog_text_available(entry) else
                               "local_original_registered" if entry.get("source_files") else "catalog_metadata_registered"),
                    "in_corpus": "true" if _catalog_text_available(entry) else "false",
                    "source_id": ";".join(s["source_id"] for s in entry.get("source_files", []) if s.get("source_id")),
                    "metadata_origin": "catalog"})
                existing.add(_key(code))
        for relation in self.version_relations:
            if relation.get("level") in {"standard_lifecycle", "standard_metadata", "regulatory_lifecycle"}:
                self._lifecycle_facts.append(_relation_view(relation))
                code = relation.get("subject_code")
                if code:
                    self._version_nodes[_key(code)] = code
                    if relation.get("subject_std_id"):
                        self._historical_ids[_key(relation["subject_std_id"])] = code
                continue
            if relation.get("level") == "standard_identity":
                self._identity_relations.append(_relation_view(relation))
                code = relation.get("canonical_code")
                # An unversioned alias is never a separate historical version.
                if code and _explicit_canonical_edition(code):
                    self._version_nodes[_key(code)] = code
                    if relation.get("canonical_std_id"):
                        self._historical_ids[_key(relation["canonical_std_id"])] = code
                continue
            if relation.get("level") != "standard_version" and not relation.get("old_code"):
                continue
            old = relation.get("source_code", relation.get("old_code"))
            new = relation.get("target_code", relation.get("new_code"))
            if not old or not new:
                continue
            edge = _relation_view(dict(relation, old_code=old, new_code=new))
            self._version_edges.append(edge)
            self._version_nodes[_key(old)] = old
            self._version_nodes[_key(new)] = new

    def _historical_standard(self, query):
        code = self._version_nodes.get(_key(query)) or self._historical_ids.get(_key(query))
        if code is None:
            return None
        return {"std_no": code, "title": code, "doc_id": _key(code),
                "status": "historical_relation_registered", "in_corpus": "false",
                "metadata_origin": "version_relations"}

    def _two_digit_edition_matches(self, query):
        """Find exact registered editions for a two-digit year spelling.

        The year suffix identifies candidates, not a century. A caller may
        resolve the spelling only when exactly one same-edition candidate is
        present in the catalog or registered version nodes.
        """
        requested = _edition_identity(query)
        if not requested or len(requested["year"]) != 2:
            return []

        registered = {}
        for standard in self.standards:
            code = standard.get("std_no", "")
            if code:
                registered.setdefault(_key(code), code)
        for code in self._version_nodes.values():
            if code:
                registered.setdefault(_key(code), code)
        for entry in self._catalog_by_code.values():
            code = entry.get("std_no", entry.get("code", ""))
            if code:
                registered.setdefault(_key(code), code)

        matches = {}
        requested_stem = _key(requested["stem"])
        for code in registered.values():
            candidate = _edition_identity(code)
            if not candidate or len(candidate["year"]) != 4:
                continue
            if (_key(candidate["stem"]) != requested_stem
                    or candidate["separator"] != requested["separator"]
                    or candidate["year"][-2:] != requested["year"][-2:]
                    or candidate["revision"] != requested["revision"]):
                continue
            if requested["revision_year"] and candidate["revision_year"] != requested["revision_year"]:
                continue
            matches.setdefault(_key(code), code)
        return sorted(matches.values(), key=_key)

    def _qualified_edition_candidates(self, query):
        """List exact-base matches whose registered code states a dated edition.

        This is a suggestion path only: even a single match must be queried by
        its full registered code before status or clauses are returned.
        """
        query_key = _key(query)
        if _QUALIFIED_EDITION_SUFFIX.search(query_key):
            return []

        candidates = {}

        def add(code, title=None, std_id=None, rank=0):
            base = _unqualified_edition_code(code)
            if not base or base != query_key:
                return
            key = _key(code)
            current = candidates.get(key)
            if current is None or rank >= current["_rank"]:
                candidates[key] = {"std_no": code, "title": title or code,
                                   "std_id": std_id or key, "_rank": rank}

        for code in self._version_nodes.values():
            add(code, code, _key(code), rank=0)
        for standard in self.standards:
            add(standard.get("std_no", ""), standard.get("title"),
                standard.get("doc_id"), rank=1)
        for entry in self._catalog_by_code.values():
            add(entry.get("std_no", entry.get("code", "")), entry.get("title"),
                entry.get("std_id"), rank=2)
        return [{key: value for key, value in row.items() if key != "_rank"}
                for _, row in sorted(candidates.items())]

    def resolve_standard_alias(self, query):
        """Resolve explicit, unambiguous same-edition identities; never successors."""
        key = _key(query)
        rows = [row for row in self._identity_relations
                if row.get("relation") == "alias_of" and _key(row.get("alias_code", "")) == key]
        result = {"query": query, "status": "not_registered", "canonical_code": None,
                  "canonical_std_id": None, "relations": rows}
        if not rows:
            direct = self._standard(query) or self._historical_standard(query)
            if direct:
                canonical = direct["std_no"]
                entry = self._catalog_by_code.get(_key(canonical))
                canonical_id = entry["std_id"] if entry else direct.get("doc_id")
                exact = key in {_key(canonical), _key(canonical_id or "")}
                result.update(status="exact_match" if exact else "unversioned_unique_match",
                              canonical_code=canonical, canonical_std_id=canonical_id,
                              match_method="standard_code_or_id" if exact else "unique_registered_edition")
                return result
            candidates = self._two_digit_edition_matches(query)
            if len(candidates) == 1:
                canonical = candidates[0]
                entry = self._catalog_by_code.get(_key(canonical))
                standard = self._standard(canonical) or self._historical_standard(canonical)
                result.update(status="unique_registered_edition", canonical_code=canonical,
                              canonical_std_id=entry["std_id"] if entry else (standard or {}).get("doc_id"),
                              match_method="unique_registered_two_digit_year")
            elif candidates:
                result.update(status="ambiguous_alias", match_method="two_digit_year_candidates",
                              candidate_codes=candidates)
            else:
                suggestions = self._qualified_edition_candidates(query)
                if suggestions:
                    result.update(status="ambiguous_alias", match_method="qualified_edition_candidates",
                                  candidate_codes=[row["std_no"] for row in suggestions],
                                  candidate_standards=suggestions)
            return result
        targets = {_key(row.get("canonical_code", "")) for row in rows}
        target_ids = {_key(row["canonical_std_id"]) for row in rows if row.get("canonical_std_id")}
        direct = self._standard(query)
        bad = any(not _explicit_canonical_edition(row.get("canonical_code", "")) for row in rows)
        for row in rows:
            target = self._catalog_by_id.get(row.get("canonical_std_id"))
            if target and _key(target.get("std_no", target.get("code", ""))) != _key(row.get("canonical_code", "")):
                bad = True
            query_edition = _edition_identity(query)
            target_edition = _edition_identity(row.get("canonical_code", ""))
            if query_edition and target_edition:
                left, right = query_edition["year"], target_edition["year"]
                # A recorded alias can connect printed two-digit and explicit
                # four-digit years; this comparison never generates a century.
                if (left != right if len(left) == len(right) else left[-2:] != right[-2:]):
                    bad = True
                if query_edition["revision"] != target_edition["revision"]:
                    bad = True
                if (query_edition["revision_year"] and target_edition["revision_year"]
                        and query_edition["revision_year"] != target_edition["revision_year"]):
                    bad = True
                left_prefix = re.match(r"(GB/T|GBJ|GB)(?=\d)", query_edition["stem"])
                right_prefix = re.match(r"(GB/T|GBJ|GB)(?=\d)", target_edition["stem"])
                if left_prefix and right_prefix and left_prefix.group() != right_prefix.group():
                    bad = True
            elif target_edition and target_edition["revision"]:
                # A bare unversioned name must not silently select a revision.
                bad = True
        if len(targets) != 1 or len(target_ids) > 1 or bad or (direct and _key(direct["std_no"]) not in targets):
            result["status"] = "ambiguous_alias"
        elif not all(_relation_confirmed(row) for row in rows):
            result["status"] = "unconfirmed_alias"
        else:
            result.update(status="confirmed_alias", canonical_code=rows[0]["canonical_code"],
                          canonical_std_id=rows[0].get("canonical_std_id"))
        return result

    def _add_lifecycle_details(self, result, standard):
        keys = {_key(result["std_no"]), _key((standard or {}).get("doc_id", ""))} - {""}
        facts = [row for row in self._lifecycle_facts if
                 keys.intersection({_key(row.get("subject_code", "")), _key(row.get("subject_std_id", ""))})]
        result["lifecycle_facts"] = facts
        result["standard_status_conflicts"] = []
        abolished = [row for row in facts if row.get("relation") == "abolished" and _relation_confirmed(row)]
        if not abolished or result.get("identity_resolution", {}).get("status") in {
                "ambiguous_alias", "unconfirmed_alias"}:
            return
        # Conflicting saved registry observations are exposed, not silently
        # overridden by an older or undated historical report.
        registry = [source for source in result.get("sources", []) if
                    source.get("evidence_level") == "current_registry" and source.get("supports_current_status")]
        conflicting = [source for source in registry if source.get("observed_status") not in
                       {None, "unknown", "abolished", "withdrawn", "repealed"}]
        dates = {row.get("effective_date") for row in abolished if row.get("effective_date")}
        result["abolished_effective_date"] = next(iter(dates)) if len(dates) == 1 else None
        result["abolished_date_known"] = len(dates) == 1
        if conflicting or len(dates) > 1:
            result["registry_observed_status"] = registry[-1].get("observed_status") if registry else None
            result["standard_status_conflicts"] = [{"kind": "lifecycle_registry_or_date_conflict",
                                                     "registry_observations": registry, "lifecycle_facts": abolished}]
            result.update(standard_status="unknown", standard_status_evidence_level="conflicting_saved_evidence")
            if result.get("current_mandatory_status") == "active":
                result["current_mandatory_status"] = "unknown"
            result["notes"].append("已登记废止事实与现行目录或废止日期存在冲突，整本状态保留未知；未覆盖任一原始依据。")
        else:
            levels = [row.get("evidence_level") or row.get("verification_state") or "confirmed_lifecycle_fact"
                      for row in abolished]
            if not any(source.get("observed_status") in {"abolished", "withdrawn", "repealed"} for source in registry):
                result.update(standard_status="abolished", standard_status_evidence_level=levels[-1])
                result["status_checked_at"] = max((row.get("checked_at") or row.get("verified_at") or ""
                                                  for row in abolished), default="") or None
            result["notes"].append("已确认标准废止事实；未由此指定后继标准、建立整本替代边或自动判定全部条款状态。")
        if not result["abolished_date_known"]:
            result["notes"].append("废止生效日期尚未明确，不用报告接收日期代替。")

    def _add_version_details(self, result, standard):
        query = _key(result["std_no"])
        entry = self._catalog_by_code.get(query)
        in_catalog = entry is not None or (standard is not None and standard.get("metadata_origin") != "version_relations")
        in_loaded = (_catalog_text_available(entry) if entry is not None else
                     standard is not None and str(standard.get("in_corpus", "")).lower() in {"1", "true", "yes"})
        adjacent = [r for r in self._version_edges if query in {_key(r["old_code"]), _key(r["new_code"])}]
        reached, queue, chain = {query}, [query], []
        while queue:
            current = queue.pop(0)
            for edge in self._version_edges:
                if (_key(edge["old_code"]) != current or edge["relation"] not in _WHOLE_SUCCESSOR_TYPES
                        or not _confirmed_replacement_edge(edge)):
                    continue
                if edge not in chain:
                    chain.append(edge)
                following = _key(edge["new_code"])
                if following not in reached:
                    reached.add(following)
                    queue.append(following)
        successors = list(dict.fromkeys(e["new_code"] for e in chain
                          if _key(e["old_code"]) == query and e["relation"] not in {"splits", "split_replacement"}))
        sid = entry["std_id"] if entry else standard.get("doc_id") if standard else None
        revisions = [r for r in self.version_relations if r.get("source_part_id") and
                     sid in {r.get("source_std_id"), r.get("target_std_id")}]
        result.update(in_corpus=in_loaded, in_catalog=in_catalog, text_available=in_loaded,
            fulltext_available=in_loaded,
            original_available=bool(entry.get("original_available", entry.get("source_files"))) if entry else False,
            page_reading_available=bool(entry.get("page_reading_available", any(Path(item.get("path", "")).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"} for item in entry.get("source_files", [])))) if entry else False,
            library_class=(entry.get("library_class") or ("A" if in_loaded else "B" if entry.get("source_files") else "C")) if entry else None,
            version_relation_candidates=[edge for edge in adjacent if not _relation_confirmed(edge)],
            confirmed_version_relations=[edge for edge in adjacent if _relation_confirmed(edge)],
            version_relations_non_successor=[edge for edge in adjacent if _relation_confirmed(edge)
                                            and not _confirmed_replacement_edge(edge)],
            version_relations=adjacent,
            version_chain={"query": result["std_no"], "edges": chain,
                           "nodes": [self._version_nodes.get(key, result["std_no"]) for key in sorted(reached)],
                           "complete_history_verified": False},
            whole_replacement_successors=successors, revision_relations=revisions)
        if entry:
            result["catalog_notes"] = list(entry.get("notes", []))
            result["metadata_sources"] = entry.get("metadata_sources", [])
            result["metadata_verification"] = entry.get("metadata_verification", entry.get("verification_status", "unknown"))
            if not in_loaded:
                result["notes"].append("已登记目录但本地无可检索正文；元数据不是条款或合规证据。")
                if result["original_available"]:
                    result["registered_status"] = "local_original_registered"
                    if result.get("clause_no"):
                        result["clause_lookup_status"] = "text_not_prepared"
                if entry.get("entry_kind") == "metadata_only" or entry.get("library_class") == "C":
                    result["registered_status"] = "catalog_metadata_registered"
                    if result.get("clause_no"):
                        result["clause_lookup_status"] = "source_missing"
        if result["version_relation_candidates"]:
            result["notes"].append("事实未确认的关系仅供线索展示，不参与自动后继或整本替代推断。")
        if result["version_relations_non_successor"]:
            result["notes"].append("部分关系事实已确认，但其关系类型或范围不支持整本替代推断。")
        if successors:
            result["notes"].append("已登记整本替代/合并或分册承接关系，具体范围见版本边；未因此认定链末版本当前强制属性。")
        if any(e["relation"] in {"amends", "partially_replaces", "repeals_mandatory_clauses"} for e in adjacent):
            result["notes"].append("局修、部分内容承接和旧强条废止不是整本废止，不能按这些边推定整本失效。")
        if revisions:
            result["notes"].append("正文与修订/勘误分别保存；查阅相关条文时应一并读取对应修订。")
        self._add_lifecycle_details(result, standard)
        result["identity_relations"] = [row for row in self._identity_relations if query in
            {_key(row.get("alias_code", "")), _key(row.get("canonical_code", ""))}]
        result["history_coverage"] = {
            "registered_version_relation_count": len(adjacent),
            "confirmed_version_relation_count": len(result["confirmed_version_relations"]),
            "candidate_version_relation_count": len(result["version_relation_candidates"]),
            "lifecycle_fact_count": len(result["lifecycle_facts"]),
            "identity_relation_count": len(result["identity_relations"]),
            "complete_history_verified": False,
        }
        if entry:
            if isinstance(entry.get("history_coverage"), dict):
                result["history_coverage"].update(entry["history_coverage"])
                result["history_coverage"]["complete_history_verified"] = False
            if "history_review" in entry:
                result["history_review"] = entry["history_review"]
        urls = []
        for edge in adjacent + chain + result["lifecycle_facts"] + result["identity_relations"]:
            for evidence in edge.get("evidence", []):
                if isinstance(evidence, dict):
                    urls.extend(evidence.get("urls", []))
                    urls.extend([evidence["source_url"]] if evidence.get("source_url") else [])
        result["source_urls"] = list(dict.fromkeys(result.get("source_urls", []) + urls))
        self._add_current_version_details(result)
        return result

    def _add_current_version_details(self, result):
        """Keep saved current status, latest publication and future implementation distinct."""
        identity = result.get("identity_resolution", {}).get("status")
        if identity in {"ambiguous_alias", "unconfirmed_alias"}:
            return
        lifecycle = self.lifecycle.describe(result["std_no"])
        legacy_status = result.get("standard_status", "unknown")
        # A separately verified abolition fact remains valid even when this
        # particular edition has no registry observation in the new adapter.
        override = (lifecycle["standard_status"] != "unknown" or lifecycle["status_sources"]
                    or lifecycle["status_conflicts"] or lifecycle["requires_status_recheck"])
        for name, value in lifecycle.items():
            if name in {"standard_status", "standard_status_evidence_level", "status_checked_at"} and not override:
                continue
            result[name] = value
        if result.get("standard_status_conflicts"):
            result["standard_status"] = "unknown"
            result["status_conflicts"] = list(result.get("status_conflicts", [])) + result["standard_status_conflicts"]
        elif legacy_status == "abolished" and result["standard_status"] == "current":
            result["standard_status"] = "unknown"
            result["status_conflicts"] = list(result.get("status_conflicts", [])) + [{"kind": "registered_abolition_vs_current_observation"}]
        future = {_key(code) for code in lifecycle["known_future_successors"]}
        result["whole_replacement_successors"] = [code for code in result.get("whole_replacement_successors", []) if _key(code) not in future]
        if result["standard_status"] == "abolished" and len(result["whole_replacement_successors"]) == 1:
            result["superseded_by"] = result["whole_replacement_successors"][0]
        if lifecycle["upcoming_versions"]:
            result["notes"].append("已发布但尚未实施的新版本单列；不能用它覆盖当前有效版或立即判旧版失效。")
        if lifecycle["standard_status"] != "unknown":
            result["notes"] = [note for note in result["notes"] if not note.startswith("尚无本轮直接现行登记证据")]
            result["notes"].append("状态来自已保存的登记观察或明确生效的整本替代事实；请同时查看核验日期和来源。")
        result["source_urls"] = list(dict.fromkeys(result.get("source_urls", []) + [
            source["source_url"] for item in lifecycle["version_statuses"]
            for source in item["status_sources"] if source.get("source_url")]))
        selected = next((item for item in lifecycle["version_statuses"] if _key(item["std_no"]) == _key(result["std_no"])), None)
        if selected and selected.get("effective_date"):
            result["current_effective_date"] = selected["effective_date"]
        if result.get("current_mandatory_status") == "active" and result["standard_status"] != "current":
            result["current_mandatory_status"] = "unknown"


    def _standard(self, query: str) -> dict | None:
        query_key = _key(query)
        matches = []
        for standard in self.standards:
            full = _key(standard["std_no"])
            aliases = {full, _key(standard.get("doc_id", "")), re.sub(r"-\d{4}$", "", full)}
            if query_key and query_key in aliases:
                matches.append(standard)
        return matches[0] if len(matches) == 1 else None

    def check(self, std_no: str, clause_no: str | None = None) -> dict:
        identity = self.resolve_standard_alias(std_no)
        canonical = identity["canonical_code"] if identity["status"] in {
            "confirmed_alias", "unique_registered_edition"} else std_no
        standard = (self._standard(canonical) or self._historical_standard(canonical)) if identity["status"] not in {
            "ambiguous_alias", "unconfirmed_alias"} else None
        clause = _key(clause_no) if clause_no is not None else None
        result = {
            "found": standard is not None, "query_std_no": std_no, "identity_resolution": identity,
            "std_no": standard["std_no"] if standard else std_no,
            "clause_no": clause_no, "registered_status": "unknown",
            "standard_status": "unknown", "standard_status_evidence_level": "unknown",
            "current_effective_date": None, "superseded_by": None,
            "original_mandatory": None, "current_mandatory_status": "unknown",
            "full_text_mandatory_at_publication": None,
            "clause_found": None, "clause_lookup_status": "not_requested" if clause is None else "unknown",
            "matched_relations": [], "related_subclause_relations": [],
            "registered_relation_count": 0, "source_ids": [], "source_urls": [], "sources": [],
            "verification_date": self.verification.get("checked_at"),
            "status_checked_at": None, "relation_verified_at": None,
            "lookup_mode": "offline_saved_evidence", "compliance_conclusion": None,
            "notes": ["依据本次收录资料及所列核验日期查询；不判断历史编制时是否违规，不自动给出合规结论。"],
        }
        if identity["status"] in {"ambiguous_alias", "unconfirmed_alias"}:
            if identity.get("match_method") == "qualified_edition_candidates":
                result["notes"].append("登记中找到带年份修正/修订限定的候选；请使用返回的完整标准号重新查询，本次未自动选择版本或判断现行性。")
            else:
                result["notes"].append("别名存在冲突或尚未确认，未自动选定标准版本。")
        elif identity["status"] == "confirmed_alias":
            result["notes"].append("按已确认的同版身份别名定位；未跳转到后继或最新版本。")
        elif identity["status"] == "unique_registered_edition":
            result["notes"].append("按登记版本中唯一匹配的两位年份定位；未推断世纪，也未跳转到后继版本。")
        if standard is None:
            result["notes"].append("未在本项目登记表中唯一命中；不等于该标准不存在、已废止或现行。")
            return self._add_version_details(result, standard)
        std_key = _key(standard["std_no"])
        evidence = [e for e in self.evidence if std_key in {_key(n) for n in e.get("std_nos", [])}]
        whole = [e for e in evidence if e.get("evidence_level") == "current_registry" and e.get("supports_current_status")]
        mandatory = [e for e in evidence if e.get("all_body_clauses_mandatory_at_publication") is True
                     and e.get("evidence_level") == "official_publication_notice"]
        fallback_level = "registration_only"
        if any(e.get("evidence_level") == "official_publication_notice" for e in evidence):
            fallback_level = "publication_verified"
        for level in ("official_text_verified", "revised_text_verified"):
            if any(e.get("evidence_level") == level for e in evidence):
                fallback_level = level
        status = standard.get("status", "unknown")
        relation_rows = [r for r in self.relations if _key(r["std_no"]) == std_key]
        result.update(
            title=standard["title"], doc_id=standard.get("doc_id"), registered_status=status,
            registered_status_checked_at=standard.get("status_checked_at") or None,
            current_effective_date=standard.get("current_effective_date") or None,
            superseded_by=standard.get("superseded_by") or None,
            registered_relation_count=len(relation_rows),
            full_text_mandatory_at_publication=True if mandatory else None,
            standard_status=whole[-1]["observed_status"] if whole else "unknown",
            standard_status_evidence_level="current_registry" if whole else fallback_level,
            status_checked_at=whole[-1]["checked_at"] if whole else None,
            sources=list(evidence),
        )
        if not whole:
            result["notes"].append("尚无本轮直接现行登记证据；发布/修订正文、近期引用或旧目录均不单独证明整本现行性。")
        else:
            result["notes"].append("整本状态为所列官方现行文件库或标准目录在核验日的登记结果；不能据此恢复已废止原强条或推定工程适用性。")
        if clause is not None:
            exists = (std_key, clause) in self.clause_keys
            exact = [r for r in relation_rows if _key(r["clause_no"]) == clause]
            parents = [r for r in relation_rows if clause.startswith(_key(r["clause_no"]) + "(")]
            children = [r for r in relation_rows if _key(r["clause_no"]).startswith(clause + "(")]
            applicable = exact or parents
            result.update(clause_found=True if exists or exact else None,
                          clause_lookup_status="in_local_corpus" if exists else "in_repeal_register" if exact else "not_in_loaded_corpus",
                          matched_relations=[self._relation(r, "exact" if r in exact else "parent_clause") for r in applicable],
                          related_subclause_relations=[self._relation(r, "subclause_only") for r in children])
            if applicable and (exact or exists):
                result.update(original_mandatory=True, current_mandatory_status="abolished",
                              relation_verified_at=max((r.get("verified_at", "") for r in applicable), default="") or None)
                result["notes"].append("登记的是所列条/款的原强制性条文废止关系；不等于整本标准或全部条文文本失效，不证明与新条文一一替代。")
            elif applicable:
                result["notes"].append("命中父条的废止关系，但本项目未确认所查询款号存在；不自动将猜测款号标成已废止条文。")
            elif children:
                result["notes"].append("仅命中该条下的指定款废止记录；不得扩大成整条废止，其余款强制属性未知。")
            elif exists and mandatory:
                result["original_mandatory"] = True
                result["current_mandatory_status"] = "active" if result["standard_status"] == "current" else "unknown"
                result["notes"].append("原公告确认规范正文全部条文强制；现行强制状态另需本轮整本现行登记，且仍须核对工程适用条件。")
            else:
                nonmandatory = self._original_nonmandatory_sources(std_key, clause) if exists else []
                if nonmandatory:
                    result["original_mandatory"] = False
                    result["sources"].extend(nonmandatory)
                    result["notes"].append("已核验原公告完整原强条清单，确认此版本正文条款不属原强条；仍须执行适用正文要求，不据此判断当前强制属性或工程适用性。")
                else:
                    result["notes"].append("未命中已登记废止关系，不证明该条为现行强条，也不推定为普通条文；未收录条号不据此判断不存在。")
        else:
            result["matched_relations"] = [self._relation(r, "standard_relation_list") for r in relation_rows]
            result["notes"].append("原强条逐条废止表与标准级版本关系分别列示；不代表全部条文状态均已核实。")
        source_ids = _ids(standard.get("source_id", ""))
        source_ids += [e["source_id"] for e in result["sources"] if e.get("source_id")]
        selected_relations = result["matched_relations"] + result["related_subclause_relations"]
        source_ids += [r["source_id"] for r in selected_relations if r.get("source_id")]
        result["source_ids"] = list(dict.fromkeys(source_ids))
        existing_sources = {s["source_id"] for s in result["sources"] if s.get("source_id")}
        result["sources"] += [{**self.sources[s], "evidence_level": "registered_source"}
                              for s in result["source_ids"] if s in self.sources and s not in existing_sources]
        urls = [s.get("source_url", "") for s in result["sources"]]
        urls += [r.get("source_url", "") for r in selected_relations]
        result["source_urls"] = list(dict.fromkeys(url for url in urls if url))
        return self._add_version_details(result, standard)

    @staticmethod
    def _relation(row: dict, match_scope: str) -> dict:
        return {**row, "match_scope": match_scope, "relation_type": "former_mandatory_clause_abolished",
                "whole_standard_abolished": False, "clause_text_invalidated": None}


def check_standard_status(std_no: str, clause_no: str | None = None,
                          project_root: str | Path = DEFAULT_PROJECT_ROOT) -> dict:
    """Return a JSON-serialisable ordinary dict from local registered evidence."""
    return StandardStatusRegistry(project_root).check(std_no, clause_no)
