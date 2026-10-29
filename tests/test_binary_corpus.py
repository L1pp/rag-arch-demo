from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from rag_arch.binary_corpus import (
    BinaryCorpusStore,
    BinaryCorpusStoreError,
    build_core_corpus_mmap,
)
import rag_arch.binary_corpus as binary_corpus
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.standards import load_corpus


def _clause(identifier, std_id, std_no, clause_no, text, **extra):
    return {
        "clause_id": identifier,
        "std_id": std_id,
        "std_no": std_no,
        "clause_no": clause_no,
        "text": text,
        "heading_path": ["总则", clause_no],
        "document_part": "body",
        "content_status": "text_available",
        "numeric_text_usable": True,
        "exact_clause_lookup": True,
        "original_mandatory": True,
        **extra,
    }


def _table(identifier, std_id, std_no, table_no, **extra):
    return {
        "table_id": identifier,
        "std_id": std_id,
        "std_no": std_no,
        "table_no": table_no,
        "caption": f"表{table_no} 配置表",
        "markdown": "| 项目 | 数值 |\n|---|---|\n| 钢筋 | 18mm |",
        "flat_text": "钢筋 18mm",
        "notes": ["单位见注"],
        "document_part": "body",
        "content_status": "text_available",
        **extra,
    }


def _fixture_records():
    clauses = [
        _clause(
            "GB-1/1.1", "GB-1", "GB 1-2020", "1.1", "普通条文 钢筋直径18mm",
            source_spans=[
                {"source_path": "data/raw/a.pdf", "source_id": "pdf-a",
                 "logical_page": 1, "document_page_start": 5},
                {"source_path": "data/raw/shared.pdf", "source_id": "pdf-shared",
                 "logical_page": 2},
                {"source_path": "data/raw/a.pdf", "source_id": "duplicate-a"},
                {"source_path": "https://example.invalid/a.pdf", "source_id": "web"},
            ],
            image_refs=[{"image_path": "data/images/a.png", "source_id": "img-a",
                         "logical_page": 3}],
            asset_refs=[{"asset_path": "data/assets/a.bin", "source_id": "asset-a"}],
            source_page_images=[{"file": "data/pages/a-5.png", "source_id": "page-a",
                                 "document_page_start": 5}]),
        _clause(
            "GB-1/formula/F1", "GB-1", "GB 1-2020", "4.1",
            "公式内容 M = 1.8 × N", record_kind="formula", formula_no="F1",
            formula_latex="M=1.8N", formula_text="弯矩等于1.8倍", variables=["M", "N"],
            units={"M": "kN·m"}),
        _clause(
            "GB-1/commentary/1.1", "GB-1", "GB 1-2020", "1.1",
            "条文说明，不进入检索", document_part="commentary"),
        _clause(
            "GB-1/appendix/A.1", "GB-1", "GB 1-2020", "A.1", "附录条文正文",
            document_part="normative_appendix", exact_clause_lookup=False,
            original_mandatory=False,
            original_mandatory_evidence={"reason": "synthetic proof", "evidence": [1, 2, 3]}),
        _clause(
            "GB-2/1.1", "GB-2", "GB 1-2021", "1.1", "另一版标准同号条文",
            import_origin={"source": "synthetic", "revision": 2},
            source_spans=[{"source_path": "data/raw/std2.pdf", "source_id": "std2"}]),
    ]
    tables = [
        _table("GB-1/table/T1", "GB-1", "GB 1-2020", "1.1",
               source_spans=[{"source_path": "data/raw/shared.pdf", "source_id": "table-shared"},
                             {"source_path": "data/raw/table.pdf", "source_id": "table-pdf"}]),
    ]
    return clauses, tables


def _write_fixture(directory: Path, clauses=None, tables=None):
    directory.mkdir(parents=True, exist_ok=True)
    default_clauses, default_tables = _fixture_records()
    clauses = default_clauses if clauses is None else clauses
    tables = default_tables if tables is None else tables
    for name, rows in (("clauses.jsonl", clauses), ("tables.jsonl", tables)):
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps({
        "document_count": 2, "documents": [
            {"std_no": "GB 1-2020", "title": "标准一"},
            {"std_no": "GB 1-2021", "title": "标准二"},
        ]}, ensure_ascii=False), encoding="utf-8")
    (directory / "catalog.json").write_text(json.dumps({"standards": [
        {"std_no": "GB 1-2020", "title": "标准一"}],
    }, ensure_ascii=False), encoding="utf-8")
    (directory / "relations.json").write_text(json.dumps({"relations": [
        {"relation_id": "rel-1", "relation": "amends", "source_part_id": "part-a",
         "target_std_id": "GB-1"}],
    }, ensure_ascii=False), encoding="utf-8")
    return clauses, tables


class BinaryCorpusStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "corpus"
        self.output = self.root / "core_corpus"
        self.clauses, self.tables = _write_fixture(self.source)
        self.version = build_core_corpus_mmap(self.source, self.output)

    def tearDown(self):
        self.temp.cleanup()

    def open(self, *, source_dir=None, cache_bytes=8 * 1024 * 1024):
        return BinaryCorpusStore.open(self.output, source_dir=source_dir,
                                      cache_bytes=cache_bytes)

    def test_invalid_summary_documents_fails_closed_in_source_and_store(self):
        original_read = binary_corpus._read_json_object
        summary_path = self.source / "summary.json"
        for invalid in (None, {}, ["not an object"]):
            with self.subTest(documents=invalid):
                summary_path.write_text(json.dumps({"documents": invalid}), encoding="utf-8")
                with self.assertRaisesRegex(BinaryCorpusStoreError, "summary.documents"):
                    binary_corpus._load_small_source_metadata(self.source)

                def read_with_invalid_documents(path):
                    result = original_read(path)
                    if path.name == "corpus_meta.json":
                        result["summary"]["documents"] = invalid
                    return result

                with patch.object(binary_corpus, "_read_json_object",
                                  side_effect=read_with_invalid_documents):
                    with self.assertRaisesRegex(BinaryCorpusStoreError, "summary.documents"):
                        self.open()

    def test_records_getters_and_lazy_chunks_match_legacy_exactly(self):
        store = self.open(source_dir=self.source)
        try:
            self.assertEqual(0, store.cache_stats["record_decode_count"])
            self.assertEqual(len(self.clauses), len(store.corpus.clauses))
            self.assertEqual(len(self.tables), len(store.corpus.tables))
            for row, expected in enumerate(self.clauses):
                self.assertEqual(expected, store.corpus.clauses[row])
            for row, expected in enumerate(self.tables):
                self.assertEqual(expected, store.corpus.tables[row])

            self.assertEqual("GB-1", store.corpus.resolve_standard_id("GB 1-2020"))
            self.assertIsNone(store.corpus.resolve_standard_id("GB 1"))
            self.assertEqual(self.clauses[0], store.corpus.get_clause("GB 1-2020", "1.1"))
            self.assertIsNone(store.corpus.get_clause("GB 1-2020", "A.1"))
            self.assertEqual(self.clauses[3],
                             store.corpus.get_clause("GB 1-2020", "appendix/A.1"))
            self.assertEqual(self.tables[0], store.corpus.get_table("GB-1/table/T1"))

            expected = EvidenceIndex(load_corpus(self.source))
            actual_chunks = store.chunk_sequence("standards:test-collection")
            self.assertTrue(actual_chunks.is_lazy_chunk_sequence)
            self.assertEqual([chunk.chunk_id for chunk in expected.chunks], actual_chunks.row_ids)
            self.assertEqual(len(actual_chunks), len(expected.chunks))
            self.assertEqual(len({chunk.text for chunk in expected.chunks}),
                             actual_chunks.unique_text_count)
            self.assertEqual(5, len(actual_chunks))  # commentary source row is retained but not chunked
            for row, expected_chunk in enumerate(expected.chunks):
                actual_chunk = actual_chunks[row]
                self.assertEqual(expected_chunk.chunk_id, actual_chunk.chunk_id)
                self.assertEqual(expected_chunk.text, actual_chunk.text)
                self.assertEqual(expected_chunk.locator, actual_chunk.locator)
                self.assertEqual(expected_chunk.source_kind, actual_chunk.source_kind)
                self.assertEqual(actual_chunk.collection_id, "standards:test-collection")
                self.assertEqual((actual_chunk.chunk_id, actual_chunk.text),
                                 actual_chunks.id_text_rows[row])
            self.assertEqual(len(actual_chunks), store.cache_stats["chunk_decode_count"])
        finally:
            store.close()

    def test_lazy_metadata_and_id_text_do_not_decode_full_records_and_are_isolated(self):
        record_size = len(json.dumps(self.clauses[0], ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8"))
        self.assertGreater(record_size, 512)
        store = self.open(cache_bytes=512)
        try:
            chunks = store.chunk_sequence("collection-a")
            self.assertEqual(0, store.cache_stats["record_decode_count"])
            record_metadata = store.corpus.clauses._metadata_rows
            self.assertIs(next(iter(record_metadata[0])), next(iter(record_metadata[1])))
            self.assertIs(record_metadata[0]["std_no"], record_metadata[1]["std_no"])
            self.assertIs(store.corpus.clauses.metadata.filter_metadata[0]["std_no"],
                          record_metadata[0]["std_no"])
            self.assertEqual("GB-1/1.1", chunks.filter_metadata[0]["chunk_id"])
            with self.assertRaises(TypeError):
                chunks.filter_metadata[0]["std_no"] = "modified"
            with self.assertRaises(TypeError):
                store.corpus.clauses.metadata.filter_metadata[4]["import_origin"]["source"] = "modified"
            metadata = chunks.metadata[0]
            metadata["std_no"] = "modified"
            self.assertEqual("GB 1-2020", chunks.metadata[0]["std_no"])
            row_id, text = chunks.id_text_rows[0]
            self.assertEqual("GB-1/1.1", row_id)
            self.assertIn("普通条文", text)
            self.assertEqual(0, store.cache_stats["record_decode_count"])
            self.assertEqual(1, store.cache_stats["id_text_decode_count"])

            record = store.corpus.clauses[0]
            original_text = record["text"]
            record["text"] = "polluted"
            reread = store.corpus.clauses[0]
            self.assertEqual(original_text, reread["text"])
            self.assertEqual(0, store.cache_stats["cache_bytes"])
            self.assertEqual(0, store.cache_stats["cache_hits"])

            chunk = chunks[0]
            chunk.locator["polluted"] = True
            next_chunk = chunks[0]
            self.assertNotIn("polluted", next_chunk.locator)
            self.assertEqual(text, next_chunk.text)
            for row in range(len(store.corpus.clauses)):
                store.corpus.clauses[row]
        finally:
            store.close()

    def test_record_cache_concurrent_same_key_misses_keep_exact_byte_accounting(self):
        serialized_sizes = [len(json.dumps(row, ensure_ascii=False,
                                           separators=(",", ":")).encode("utf-8"))
                            for row in [*self.clauses, *self.tables]]
        cache_limit = max(serialized_sizes) * 2
        self.assertGreater(sum(serialized_sizes), cache_limit)
        store = self.open(cache_bytes=cache_limit)
        original_slice = store._blob_slice_unleased
        barrier = threading.Barrier(2)

        def simultaneous_slice(blob, offsets, row):
            value = original_slice(blob, offsets, row)
            barrier.wait(timeout=3)
            return value

        try:
            with patch.object(store, "_blob_slice_unleased", side_effect=simultaneous_slice):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    records = list(executor.map(lambda _: store.corpus.clauses[0], range(2)))
            self.assertEqual(records[0], records[1])
            with store._condition:
                cached_values = tuple(store._record_cache.values())
            stats = store.cache_stats
            self.assertEqual(sum(map(len, cached_values)), stats["cache_bytes"])
            self.assertEqual(serialized_sizes[0], stats["cache_bytes"])
            self.assertLessEqual(stats["cache_bytes"], cache_limit)

            store.corpus.clauses[0]
            self.assertGreater(store.cache_stats["cache_hits"], 0)
            for row in range(len(store.corpus.clauses)):
                store.corpus.clauses[row]
            for row in range(len(store.corpus.tables)):
                store.corpus.tables[row]
            with store._condition:
                cached_values = tuple(store._record_cache.values())
            stats = store.cache_stats
            self.assertEqual(sum(map(len, cached_values)), stats["cache_bytes"])
            self.assertLessEqual(stats["cache_bytes"], cache_limit)
            self.assertGreater(stats["cache_evictions"], 0)
        finally:
            store.close()

    def test_status_records_load_proof_on_demand_and_source_projection_is_small_ordered(self):
        store = self.open()
        try:
            status_rows = list(store.corpus.status_records())
            self.assertEqual(len(self.clauses), len(status_rows))
            appendix = next(row for row in status_rows if row["clause_no"] == "A.1")
            self.assertFalse(appendix["original_mandatory"])
            self.assertEqual({"reason": "synthetic proof", "evidence": [1, 2, 3]},
                             appendix["original_mandatory_evidence"])
            self.assertEqual(0, store.cache_stats["record_decode_count"])

            all_sources = list(store.corpus.iter_source_records())
            filtered = list(store.corpus.iter_source_records(selected={"GB 1-2021"}))
            self.assertEqual([row["std_no"] for row in all_sources],
                             ["GB 1-2020", "GB 1-2021", "GB 1-2020"])
            self.assertEqual(["GB 1-2021"], [row["std_no"] for row in filtered])
            first_paths = [location.get("source_path", location.get("image_path",
                              location.get("asset_path", location.get("file"))))
                           for source in all_sources
                           for field in ("source_spans", "image_refs", "asset_refs", "source_page_images")
                           for location in source.get(field, [])]
            self.assertEqual([
                "data/raw/a.pdf", "data/raw/shared.pdf", "data/images/a.png",
                "data/assets/a.bin", "data/pages/a-5.png", "data/raw/std2.pdf",
                "data/raw/table.pdf",
            ], first_paths)
            self.assertEqual(4, store.cache_stats["source_projection_reads"])
        finally:
            store.close()

    def test_legacy_manifest_digest_and_source_stats_are_ignored(self):
        copied = self.root / "copied-corpus"
        shutil.copytree(self.source, copied)
        manifest_path = self.version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("source_stat_before", manifest)
        self.assertNotIn("source_stat_after", manifest)
        for entry in manifest["files"].values():
            self.assertNotIn("sha256", entry)
            entry["sha256"] = "legacy-unchecked"
        manifest["source_stat_before"] = "legacy-unchecked"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        store = self.open(source_dir=copied)
        try:
            self.assertEqual(0, store.cache_stats["source_validation_records"])
        finally:
            store.close()

    def test_close_waits_for_active_read_and_returned_objects_survive(self):
        store = self.open()
        sequence = store.chunk_sequence("collection-close")
        entered = threading.Event()
        release = threading.Event()
        getter_done = threading.Event()
        closer_started = threading.Event()
        closer_done = threading.Event()
        original = store._decode_record_unleased
        returned = {}

        def blocking_decode(kind, row):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release corpus decode")
            return original(kind, row)

        def getter():
            try:
                returned["chunk"] = sequence[0]
            finally:
                getter_done.set()

        def closer():
            closer_started.set()
            store.close()
            closer_done.set()

        try:
            with patch.object(store, "_decode_record_unleased", side_effect=blocking_decode):
                read_thread = threading.Thread(target=getter)
                read_thread.start()
                self.assertTrue(entered.wait(2))
                close_thread = threading.Thread(target=closer)
                close_thread.start()
                self.assertTrue(closer_started.wait(2))
                self.assertFalse(closer_done.wait(0.05))
                release.set()
                read_thread.join(2)
                close_thread.join(2)
            self.assertTrue(getter_done.is_set())
            self.assertTrue(closer_done.is_set())
            chunk = returned["chunk"]
            self.assertIn("普通条文", chunk.text)
            with self.assertRaises(BinaryCorpusStoreError):
                sequence[0]
            store.close()
        finally:
            release.set()
            store.close()

    def test_publish_is_immutable_and_pointer_failure_keeps_previous_current(self):
        first = self.version
        first_id = (self.output / "CURRENT").read_text(encoding="ascii")
        original_replace = Path.replace

        def fail_current_replace(path, target):
            if path.name.startswith(".CURRENT-"):
                raise OSError("simulated pointer replacement failure")
            return original_replace(path, target)

        with patch.object(Path, "replace", new=fail_current_replace):
            with self.assertRaises(BinaryCorpusStoreError):
                build_core_corpus_mmap(self.source, self.output)
        self.assertEqual(first_id, (self.output / "CURRENT").read_text(encoding="ascii"))
        self.assertTrue(first.is_dir())

    def test_corrupt_offsets_fail_and_mmap_open_is_readonly(self):
        store = self.open()
        try:
            self.assertTrue(store._maps)
            self.assertTrue(all(not array.flags.writeable for array in store._maps))
        finally:
            store.close()

        offsets_path = self.version / "clauses.record_offsets.npy"
        offsets = np.load(offsets_path, mmap_mode="r+", allow_pickle=False)
        offsets[1] = offsets[0]
        offsets.flush()
        offsets._mmap.close()
        with self.assertRaises(BinaryCorpusStoreError):
            self.open()
