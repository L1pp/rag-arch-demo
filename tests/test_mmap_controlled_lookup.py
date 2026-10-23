"""Synthetic regressions for controlled lookup over lazy mmap-style mappings."""
from collections.abc import Sequence
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rag_arch.binary_corpus import BinaryCorpusStore, build_core_corpus_mmap
from rag_arch.evidence_index import EvidenceIndex, _ChunkMapping
from rag_arch.models import Chunk
from rag_arch.standard_tools import StandardTools


class _LazyChunks(Sequence):
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.reads = []

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        self.reads.append(index)
        return self._chunks[index]


class _Status:
    def check(self, std_no, clause_no=None):
        return {"found": True, "std_no": std_no, "clause_no": clause_no}


class _Index:
    def __init__(self, chunks, catalog=None, use_dict=False):
        self.corpus = SimpleNamespace(catalog=list(catalog or []))
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks = _LazyChunks(chunks)
        self._chunk_metadata = tuple(chunk.locator for chunk in chunks)
        if use_dict:
            self._chunk_metadata = None
            self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        else:
            self.by_id = _ChunkMapping(self.chunks, self._chunk_ids)

    def lookup_metadata(self, row):
        return self._chunk_metadata[row]

    def get_clause(self, std_no, clause_no):
        for row, evidence_id in enumerate(self._chunk_ids):
            meta = self.chunks._chunks[row].locator
            if (meta.get("std_no") == std_no
                    and meta.get("clause_no") == clause_no
                    and meta.get("exact_clause_lookup") is not False):
                return self.by_id[evidence_id]
        return None

    def expand(self, raw_hits):
        return raw_hits, []


def _chunk(std_no, evidence_id, *, kind="clause", clause_no=None, table_id=None,
           table_no=None, printed_table_no_verified=False, record_kind=None,
           formula_no=None, exact_clause_lookup=True):
    text = "synthetic evidence"
    return Chunk(
        "standard:" + std_no,
        std_no,
        evidence_id,
        text,
        {
            "std_no": std_no,
            "evidence_kind": kind,
            "clause_no": clause_no,
            "table_id": table_id,
            "table_no": table_no,
            "printed_table_no_verified": printed_table_no_verified,
            "record_kind": record_kind,
            "formula_no": formula_no,
            "exact_clause_lookup": exact_clause_lookup,
            "content_complete": True,
            "content_status": "text_available",
            "stored_text_chars": len(text),
            "document_part": "body",
            "source_spans": [],
        },
        "standard",
        "synthetic",
    )


class MmapControlledLookupTests(unittest.TestCase):
    def make_tools(self, chunks, catalog=None, use_dict=False):
        index = _Index(chunks, catalog=catalog, use_dict=use_dict)
        return index, StandardTools(index, _Status(), default_mode="bm25")

    def test_full_evidence_id_reads_back_from_lazy_mapping(self):
        evidence_id = "GBTEST-2025/4.1.1"
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", evidence_id, clause_no="4.1.1"),
            *[_chunk("GBOTHER-2025", f"GBOTHER-2025/{number}", clause_no=number)
              for number in (f"9.{i}" for i in range(40))],
        ])

        result = tools.get_clause("GBTEST-2025", evidence_id)

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [evidence_id])
        self.assertEqual(index.chunks.reads, [0])

    def test_table_id_reads_back_from_lazy_mapping(self):
        table_id = "GBTEST-2025/table/9.2"
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", table_id, kind="table", table_id=table_id),
        ])

        result = tools.get_clause("GBTEST-2025", table_id)

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [table_id])
        self.assertEqual(result["evidence"][0]["kind"], "table")

    def test_verified_table_number_alias_resolves(self):
        table_id = "GBTEST-2025/table/object-proof-17"
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", table_id, kind="table", table_id=table_id,
                   table_no="9.2", printed_table_no_verified=True),
        ])

        result = tools.get_clause("GBTEST-2025", "table/9.2")

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [table_id])

    def test_formula_number_alias_resolves(self):
        evidence_id = "GBTEST-2025/4.1.1-1"
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", evidence_id, clause_no="4.1.1-1",
                   record_kind="formula", formula_no="4.1.1-1"),
        ])

        result = tools.get_clause("GBTEST-2025", "formula/4.1.1-1")

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [evidence_id])

    def test_clause_number_alias_resolves_without_decoding_unrelated_rows(self):
        evidence_id = "GBTEST-2025/4.1.1"
        chunks = [
            *[_chunk("GBOTHER-2025", f"GBOTHER-2025/{i}", clause_no=str(i))
              for i in range(40)],
            _chunk("GBTEST-2025", evidence_id, clause_no="4.1.1"),
        ]
        index, tools = self.make_tools(chunks)

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [evidence_id])
        self.assertLessEqual(len(index.chunks.reads), 2)

    def test_no_match_does_not_decode_lazy_chunks(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
        ])

        result = tools.get_clause("GBTEST-2025", "9.9")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(index.chunks.reads, [])

    def test_cross_version_alias_is_rejected_as_ambiguous(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2020", "GBTEST-2020/4.1.1", clause_no="4.1.1"),
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
        ])

        result = tools.get_clause("GBTEST", "4.1.1")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lookup_mode"], "ambiguous_standard")
        self.assertEqual(len(result["standard_candidates"]), 2)

    def test_same_edition_duplicate_alias_is_rejected(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
            _chunk("GBTEST-2025", "GBTEST-2025/part-2/4.1.1", clause_no="4.1.1"),
        ])

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lookup_mode"], "ambiguous_alias")
        self.assertEqual(len(result["candidate_ids"]), 2)

    def test_explicit_other_version_never_falls_back(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
        ])

        result = tools.get_clause("GBTEST-2020", "4.1.1")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(index.chunks.reads, [])

    def test_non_exact_clause_candidate_does_not_gain_bare_alias(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/candidate/4.1.1",
                   clause_no="4.1.1", exact_clause_lookup=False),
        ])

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(index.chunks.reads, [])

    def test_dict_mapping_reads_full_evidence_id(self):
        evidence_id = "GBTEST-2025/4.1.1"
        index, tools = self.make_tools(
            [_chunk("GBTEST-2025", evidence_id, clause_no="4.1.1")],
            use_dict=True,
        )

        result = tools.get_clause("GBTEST-2025", evidence_id)

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [evidence_id])

    def test_dict_mapping_resolves_clause_number_alias(self):
        evidence_id = "GBTEST-2025/4.1.1"
        index, tools = self.make_tools(
            [_chunk("GBTEST-2025", evidence_id, clause_no="4.1.1")],
            use_dict=True,
        )

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertTrue(result["found"], result)
        self.assertEqual(result["raw_ids"], [evidence_id])

    def test_dict_mapping_rejects_same_edition_alias_ambiguity(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
            _chunk("GBTEST-2025", "GBTEST-2025/part-2/4.1.1", clause_no="4.1.1"),
        ], use_dict=True)

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lookup_mode"], "ambiguous_alias")
        self.assertEqual(len(result["candidate_ids"]), 2)

    def test_dict_mapping_rejects_cross_version_alias_ambiguity(self):
        index, tools = self.make_tools([
            _chunk("GBTEST-2020", "GBTEST-2020/4.1.1", clause_no="4.1.1"),
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
        ], use_dict=True)

        result = tools.get_clause("GBTEST", "4.1.1")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lookup_mode"], "ambiguous_standard")
        self.assertEqual(len(result["standard_candidates"]), 2)

    def test_dict_mapping_attachment_parent_returns_member_candidates(self):
        evidence_id = "GBTEST-2025/附件1/1"
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", evidence_id, clause_no="附件1/1"),
        ], use_dict=True)

        result = tools.get_clause("GBTEST-2025", "附件1")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lookup_mode"], "attachment_members")
        self.assertEqual(result["candidate_ids"], [evidence_id])

    def test_real_mmap_old_projection_resolves_formula_and_verified_table_aliases(self):
        with tempfile.TemporaryDirectory(prefix="controlled-lookup-mmap-") as temporary:
            root = Path(temporary)
            source = root / "source"
            package = root / "core_corpus"
            source.mkdir()
            clauses = [{
                "clause_id": "GBTEST-2025/4.1.1-1",
                "std_id": "GBTEST-2025",
                "std_no": "GBTEST-2025",
                "clause_no": "4.1.1-1",
                "record_kind": "formula",
                "formula_no": "4.1.1-1",
                "text": "合成公式内容。",
                "document_part": "body",
                "content_status": "text_available",
            }]
            tables = [{
                "table_id": "GBTEST-2025/table/object-proof-17",
                "std_id": "GBTEST-2025",
                "std_no": "GBTEST-2025",
                "table_no": "9.2",
                "printed_table_no_verified": True,
                "caption": "合成已核验表号",
                "markdown": "| 项目 | 值 |\n|---|---|\n| 合成 | 1 |",
                "flat_text": "项目 值 合成 1",
                "content_status": "text_available",
                "document_part": "body",
            }]
            tables.append({
                "table_id": "GBTEST-2025/table/object-unverified-93",
                "std_id": "GBTEST-2025",
                "std_no": "GBTEST-2025",
                "table_no": "9.3",
                "printed_table_no_verified": False,
                "caption": "synthetic unverified table number",
                "markdown": "unverified table body",
                "flat_text": "unverified table body",
                "content_status": "text_available",
                "document_part": "body",
            })
            for filename, records in (("clauses.jsonl", clauses), ("tables.jsonl", tables)):
                (source / filename).write_text(
                    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                    encoding="utf-8",
                )

            build_core_corpus_mmap(source, package)
            store = BinaryCorpusStore.open(package, source_dir=source)
            try:
                index = EvidenceIndex(store.corpus)
                try:
                    tools = StandardTools(index, _Status(), default_mode="bm25")
                    before = store.cache_stats["chunk_decode_count"]
                    formula_projection = index.lookup_metadata(0)
                    table_projection = index.lookup_metadata(1)
                    self.assertNotIn("formula_no", formula_projection)
                    self.assertNotIn("printed_table_no_verified", table_projection)

                    formula = tools.get_clause("GBTEST-2025", "formula/4.1.1-1")
                    table = tools.get_clause("GBTEST-2025", "table/9.2")
                    unverified = tools.get_clause("GBTEST-2025", "table/9.3")

                    self.assertTrue(formula["found"], formula)
                    self.assertEqual(formula["raw_ids"], ["GBTEST-2025/4.1.1-1"])
                    self.assertTrue(table["found"], table)
                    self.assertEqual(table["raw_ids"], ["GBTEST-2025/table/object-proof-17"])
                    self.assertEqual(unverified["status"], "not_found")
                    self.assertLessEqual(store.cache_stats["chunk_decode_count"] - before, 3)
                finally:
                    index.local.close_bm25()
            finally:
                store.close()

    def test_catalog_only_entry_remains_discovery_not_evidence(self):
        catalog = [{
            "std_id": "GBTEST-2025",
            "std_no": "GBTEST-2025",
            "code": "GBTEST-2025",
            "title": "合成目录条目",
            "entry_kind": "metadata_only",
            "library_class": "C",
        }]
        index, tools = self.make_tools([
            _chunk("GBTEST-2025", "GBTEST-2025/4.1.1", clause_no="4.1.1"),
        ], catalog=catalog)

        result = tools.get_clause("GBTEST-2025", "4.1.1")

        self.assertEqual(result["lookup_mode"], "catalog_only")
        self.assertFalse(result["found"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(index.chunks.reads, [])


if __name__ == "__main__":
    unittest.main()
