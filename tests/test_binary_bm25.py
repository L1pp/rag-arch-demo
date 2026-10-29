from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rag_arch.binary_bm25 import (
    BinaryBM25Store,
    BinaryBM25StoreError,
    build_core_bm25,
)
import rag_arch.binary_bm25 as binary_bm25
from rag_arch.retrieval import LocalIndex


def _chunk(chunk_id, text):
    return SimpleNamespace(chunk_id=chunk_id, text=text)


class BinaryBM25StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "core_bm25"
        (self.root / "clauses.jsonl").write_text("synthetic clauses\n", encoding="utf-8")
        (self.root / "tables.jsonl").write_text("synthetic tables\n", encoding="utf-8")
        self.chunks = [
            _chunk("clause-a", "钢筋混凝土 1.8MPa >= 18mm"),
            _chunk("clause-b", "钢筋混凝土 18MPa <= 1.8mm"),
            _chunk("table-c", "钢筋配置表 18mm >= 1.8"),
            _chunk("clause-d", "没有检索词的正文"),
            _chunk("blank", " \n "),
        ]

    def tearDown(self):
        self.temp.cleanup()

    def build(self, chunks=None):
        return build_core_bm25(self.directory, chunks=self.chunks if chunks is None else chunks,
                               source_paths=(self.root / "clauses.jsonl",))

    def open(self, chunks=None):
        return BinaryBM25Store.open(self.directory,
                                    chunks=self.chunks if chunks is None else chunks)

    def test_scores_match_localindex_exactly_including_numeric_operators_and_ties(self):
        self.build()
        store = self.open()
        try:
            baseline = LocalIndex(self.chunks)
            queries = [
                "钢筋混凝土",
                "18mm >= 1.8MPa",
                "1.8 <= 18mm",
                "钢筋钢筋 ≥ 18mm",
                "没有出现过的词",
                "",
                ">= <= !=",
            ]
            for query in queries:
                with self.subTest(query=query):
                    self.assertEqual(baseline._bm25_scores(query), store.scores(query))
            with patch("rag_arch.binary_bm25._SCORE_BLOCK", 2):
                for query in queries:
                    self.assertEqual(baseline._bm25_scores(query), store.scores(query))
            self.assertEqual([0.0] * 4, store.scores("量子泡沫噪声"))
            self.assertIsInstance(store.scores("钢筋"), list)
            self.assertTrue(all(type(score) is float for score in store.scores("钢筋")))
            self.assertIsInstance(store.doc_lengths, np.memmap)
            self.assertFalse(store.doc_lengths.flags.writeable)
            self.assertIsInstance(store.postings, np.memmap)
            self.assertFalse(store.postings.flags.writeable)
            self.assertIsInstance(store.offsets, np.memmap)
            self.assertIsInstance(store.doc_freqs, np.memmap)
            self.assertEqual(4, store.document_count)
            self.assertEqual(baseline._average_length, store.average_length)
            self.assertEqual(4, store.manifest["document_count"])
            self.assertEqual([0.0] * 4, store.scores("量子泡沫噪声"))
        finally:
            store.close()

    def test_empty_query_empty_corpus_and_non_tokenizable_text(self):
        self.build(chunks=[])
        empty = self.open(chunks=[])
        try:
            self.assertEqual(0, empty.document_count)
            self.assertEqual([], empty.scores("anything"))
            self.assertEqual([], empty.scores(""))
        finally:
            empty.close()

        self.build(chunks=[_chunk("symbol-only", "😀❗")])
        no_tokens = self.open(chunks=[_chunk("symbol-only", "😀❗")])
        try:
            self.assertEqual(1, no_tokens.document_count)
            self.assertEqual(0.0, no_tokens.average_length)
            self.assertEqual([0.0], no_tokens.scores("钢筋"))
        finally:
            no_tokens.close()

    def test_store_is_readonly_and_multiple_versions_keep_old_mmaps_valid(self):
        first = self.build()
        old_store = self.open()
        try:
            old_scores = old_store.scores("钢筋混凝土")
            second = self.build(chunks=[*self.chunks, _chunk("new", "钢筋新条款")])
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertEqual(old_scores, old_store.scores("钢筋混凝土"))
            new_chunks = [*self.chunks, _chunk("new", "钢筋新条款")]
            new_store = self.open(new_chunks)
            try:
                self.assertEqual(LocalIndex(new_chunks)._bm25_scores("钢筋"),
                                 new_store.scores("钢筋"))
                self.assertEqual([new.chunk_id for new in new_chunks if new.text.strip()],
                                 new_store.manifest["row_ids"])
            finally:
                new_store.close()
        finally:
            old_store.close()

    def test_metadata_id_text_order_and_tokenizer_mismatches_fail_closed(self):
        self.build()
        variants = [
            [self.chunks[1], self.chunks[0], *self.chunks[2:]],
            [_chunk("clause-a", "changed text"), *self.chunks[1:]],
            [_chunk("changed-id", self.chunks[0].text), *self.chunks[1:]],
        ]
        for chunks in variants:
            with self.subTest(chunks=chunks[0]):
                with self.assertRaises(BinaryBM25StoreError):
                    self.open(chunks)

        manifest_path = self._current_manifest()
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, value in (("rules_id", "different-tokenizer"),
                             ("unicode_version", "different-unicode")):
            with self.subTest(tokenizer_field=field):
                manifest = json.loads(json.dumps(original))
                manifest["tokenizer"][field] = value
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(BinaryBM25StoreError):
                    self.open()

    def test_build_omits_source_stats_and_loads_legacy_fields(self):
        version = self.build()
        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("source_stat_before", manifest)
        self.assertNotIn("source_stat_after", manifest)
        self.assertNotIn("array_sha256", manifest)
        self.assertNotIn("vocabulary_sha256", manifest)
        self.assertNotIn(str(self.root), json.dumps(manifest))
        manifest["source_stat_before"] = "legacy-unchecked"
        manifest["array_sha256"] = "legacy-unchecked"
        manifest["vocabulary_sha256"] = "legacy-unchecked"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        store = self.open()
        store.close()

    def test_missing_current_corrupt_arrays_and_extra_metadata_row_fail(self):
        version = self.build()
        current = self.directory / "CURRENT"
        current.unlink()
        with self.assertRaises(BinaryBM25StoreError):
            self.open()
        current.write_text(version.name + "\n", encoding="ascii")

        postings_path = version / "postings.npy"
        postings = np.load(postings_path, mmap_mode="r+", allow_pickle=False)
        postings[0]["tf"] += 1
        postings.flush()
        postings._mmap.close()
        with self.assertRaises(BinaryBM25StoreError):
            self.open()

        # A fresh package allows checking exact metadata row-count validation.
        version = self.build()
        metadata = version / "metadata.jsonl"
        with metadata.open("a", encoding="utf-8") as handle:
            handle.write('{"chunk_id":"extra","text":"extra"}\n')
        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata_bytes"] = metadata.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(BinaryBM25StoreError):
            self.open()

    def test_npy_trailing_bytes_fail_even_if_manifest_length_is_updated(self):
        version = self.build()
        path = version / "offsets.npy"
        with path.open("ab") as handle:
            handle.write(b"unexpected-tail")
        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["array_bytes"]["offsets.npy"] = path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(BinaryBM25StoreError, "数据区长度"):
            self.open()
        path.replace(version / "offsets-corrupt.npy")

    def test_threaded_read_only_scores_match_and_close_waits_for_active_score(self):
        self.build()
        store = self.open()
        baseline = LocalIndex(self.chunks)
        queries = ["钢筋混凝土", "18mm >= 1.8", "完全未知", "<= 18MPa"] * 12
        expected = [baseline._bm25_scores(query) for query in queries]
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(store.scores, queries))
        self.assertEqual(expected, actual)

        entered = threading.Event()
        release = threading.Event()
        score_done = threading.Event()
        close_started = threading.Event()
        close_done = threading.Event()
        original_log1p = __import__("math").log1p

        def blocking_log1p(value):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release the active score")
            return original_log1p(value)

        def score_worker():
            try:
                store.scores("钢筋")
            finally:
                score_done.set()

        def close_worker():
            close_started.set()
            store.close()
            close_done.set()

        try:
            with patch("rag_arch.binary_bm25.math.log1p", side_effect=blocking_log1p):
                scorer = threading.Thread(target=score_worker)
                scorer.start()
                self.assertTrue(entered.wait(2))
                closer = threading.Thread(target=close_worker)
                closer.start()
                self.assertTrue(close_started.wait(2))
                self.assertFalse(close_done.wait(0.05))
                release.set()
                scorer.join(2)
                closer.join(2)
            self.assertTrue(score_done.is_set())
            self.assertTrue(close_done.is_set())
            store.close()
            with self.assertRaises(BinaryBM25StoreError):
                store.scores("钢筋")
        finally:
            release.set()
            store.close()

    def test_close_retries_after_an_external_mmap_view_is_released(self):
        self.build()
        store = self.open()
        exported = memoryview(store.postings._mmap)
        try:
            with self.assertRaises(BinaryBM25StoreError):
                store.close()
            self.assertEqual("closing", store._state)
        finally:
            exported.release()
        store.close()
        store.close()
        self.assertEqual("closed", store._state)

    def _current_manifest(self):
        version_id = (self.directory / "CURRENT").read_text(encoding="ascii").strip()
        return self.directory / "versions" / version_id / "manifest.json"


if __name__ == "__main__":
    unittest.main()
