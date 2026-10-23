"""Synthetic integration coverage for lazy binary corpus consumers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_arch.binary_corpus import (
    BinaryCorpusStore,
    BinaryCorpusStoreError,
    build_core_corpus_mmap,
)
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.models import SearchHit
from rag_arch.standards import load_corpus
from rag_arch.web.app import create_app
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.task_service import TaskService
from tests.helpers.mmap_fixtures import (
    _build_synthetic_packages,
    _chunk_payload,
    _compare_records_and_chunks,
    _load_worker_corpus,
    _make_synthetic_records,
    _write_synthetic_corpus,
)


class MmapCorpusIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag-corpus-mmap-test-")
        self.root = Path(self.temp.name)
        self.corpus_dir = self.root / "source"
        self.addCleanup(self.temp.cleanup)

    def build_package(self, *, filler_count=128, with_bm25=False):
        if with_bm25:
            return _build_synthetic_packages(
                self.corpus_dir, self.root / "packages",
                filler_count=filler_count)
        _write_synthetic_corpus(self.corpus_dir, filler_count=filler_count)
        core_dir = self.root / "packages" / "core_corpus"
        build_core_corpus_mmap(self.corpus_dir, core_dir)
        return core_dir, None

    def test_source_records_chunks_lookup_and_revision_match_json(self):
        core_dir, bm25_dir = self.build_package(with_bm25=True)
        store = BinaryCorpusStore.open(
            core_dir, source_dir=self.corpus_dir, cache_bytes=8 * 1024)
        self.addCleanup(store.close)
        compare = _compare_records_and_chunks(self.corpus_dir, store)
        clauses, tables, _summary, _catalog, _relations = _make_synthetic_records()
        self.assertEqual(compare["source_record_rows_compared"], len(clauses) + len(tables))
        self.assertEqual(compare["chunk_rows_compared"], len(store.corpus.chunk_sequence("verify")))
        self.assertTrue(compare["source_record_equal"])
        self.assertTrue(compare["chunk_text_locator_equal"])

        legacy_corpus = load_corpus(self.corpus_dir)
        mapped_corpus = store.corpus
        standard = "GB 50000-2020"
        self.assertEqual(
            legacy_corpus.get_clause(standard, "4.2.1"),
            mapped_corpus.get_clause(standard, "4.2.1"),
        )
        self.assertIsNone(legacy_corpus.get_clause("GB 50000", "4.2.1"))
        self.assertIsNone(mapped_corpus.get_clause("GB 50000", "4.2.1"))
        self.assertIsNone(legacy_corpus.get_clause(standard, "9.9"))
        self.assertIsNone(mapped_corpus.get_clause(standard, "9.9"))
        self.assertEqual(
            legacy_corpus.get_clause(standard, "appendix-a/9.9"),
            mapped_corpus.get_clause(standard, "appendix-a/9.9"),
        )
        self.assertEqual(
            legacy_corpus.get_clause(standard, "appendix-b/9.9"),
            mapped_corpus.get_clause(standard, "appendix-b/9.9"),
        )
        self.assertIsNone(legacy_corpus.get_clause(standard, "9.9"))
        self.assertIsNone(mapped_corpus.get_clause(standard, "9.9"))
        self.assertEqual(
            legacy_corpus.get_clause(standard, "4.2.2"),
            mapped_corpus.get_clause(standard, "4.2.2"),
        )

        legacy = EvidenceIndex(legacy_corpus, bm25_directory=bm25_dir)
        mapped = EvidenceIndex(mapped_corpus, bm25_directory=bm25_dir)
        self.addCleanup(legacy.local.close_bm25)
        self.addCleanup(mapped.local.close_bm25)
        table_id = "GB50000-2020/table/4.2.1-1"
        self.assertEqual(
            _chunk_payload(legacy.get_table(table_id)),
            _chunk_payload(mapped.get_table(table_id)),
        )
        self.assertIsNone(legacy.get_clause(standard, "4.2.2"))
        self.assertIsNone(mapped.get_clause(standard, "4.2.2"))
        old_formula = legacy.get_clause(standard, "4.2.1-1")
        new_formula = mapped.get_clause(standard, "4.2.1-1")
        self.assertIsNotNone(old_formula)
        self.assertEqual(_chunk_payload(old_formula), _chunk_payload(new_formula))

        for clause_no in ("4.2.1", "5.1"):
            old_chunk = legacy.get_clause(standard, clause_no)
            new_chunk = mapped.get_clause(standard, clause_no)
            self.assertIsNotNone(old_chunk)
            self.assertEqual(_chunk_payload(old_chunk), _chunk_payload(new_chunk))

        old_owner = legacy.get_clause(standard, "5.1")
        new_owner = mapped.get_clause(standard, "5.1")
        old_context, old_missing = legacy.expand([SearchHit(old_owner, 1.0)])
        new_context, new_missing = mapped.expand([SearchHit(new_owner, 1.0)])
        self.assertEqual([_chunk_payload(hit.chunk) for hit in old_context],
                         [_chunk_payload(hit.chunk) for hit in new_context])
        self.assertEqual(old_missing, new_missing)
        self.assertIn("GB50000-2020/amendment/2024/formula-5.1",
                      [hit.chunk.chunk_id for hit in new_context])

        old_clause_context, _ = legacy.expand([
            SearchHit(legacy.get_clause(standard, "4.2.1"), 1.0)])
        new_clause_context, _ = mapped.expand([
            SearchHit(mapped.get_clause(standard, "4.2.1"), 1.0)])
        self.assertEqual(
            [_chunk_payload(hit.chunk) for hit in old_clause_context],
            [_chunk_payload(hit.chunk) for hit in new_clause_context],
        )
        self.assertIn("GB50000-2020/formula/4.2.1-1",
                      [hit.chunk.chunk_id for hit in new_clause_context])

    def test_lazy_reads_metadata_filter_cache_and_task_copy_isolation(self):
        core_dir, bm25_dir = self.build_package(with_bm25=True, filler_count=128)
        _worker_corpus, worker_store, _load_seconds = _load_worker_corpus(
            "mmap", self.corpus_dir, core_dir, 257)
        self.addCleanup(worker_store.close)
        self.assertEqual(worker_store.cache_stats["cache_limit_bytes"], 257)

        store = BinaryCorpusStore.open(
            core_dir, source_dir=self.corpus_dir, cache_bytes=2048)
        self.addCleanup(store.close)
        initial = store.cache_stats
        self.assertEqual(initial["record_decode_count"], 0)
        self.assertEqual(initial["chunk_decode_count"], 0)
        self.assertEqual(initial["id_text_decode_count"], 0)

        first = store.corpus.clauses[0]
        pristine_text = first["text"]
        pristine_source = first["source_spans"][0]["source_path"]
        first["text"] = "task-a mutation"
        first["source_spans"][0]["source_path"] = "task-a/private.pdf"
        second = store.corpus.clauses[0]
        self.assertEqual(second["text"], pristine_text)
        self.assertEqual(second["source_spans"][0]["source_path"], pristine_source)
        self.assertGreaterEqual(store.cache_stats["cache_hits"], 1)

        metadata_copy = store.corpus.clauses.metadata[0]
        metadata_copy["std_no"] = "task-a mutation"
        self.assertNotEqual(store.corpus.clauses.metadata[0]["std_no"], "task-a mutation")

        lazy = store.chunk_sequence("workspace-a")
        row_ids = lazy.row_ids
        original_id = row_ids[0]
        row_ids[0] = "task-a mutation"
        self.assertEqual(lazy.row_ids[0], original_id)

        legacy = EvidenceIndex(load_corpus(self.corpus_dir), bm25_directory=bm25_dir)
        mapped_a = EvidenceIndex(store.corpus, bm25_directory=bm25_dir)
        mapped_b = EvidenceIndex(store.corpus, bm25_directory=bm25_dir)
        self.addCleanup(legacy.local.close_bm25)
        self.addCleanup(mapped_a.local.close_bm25)
        self.addCleanup(mapped_b.local.close_bm25)
        self.assertTrue(getattr(mapped_a.chunks, "is_lazy_chunk_sequence", False))
        stats_bound = store.cache_stats
        # Reads are decoded on demand even when the underlying JSON bytes hit
        # the bounded cache; cache hits avoid disk I/O, not JSON object copies.
        self.assertEqual(stats_bound["record_decode_count"], 2)
        self.assertEqual(stats_bound["chunk_decode_count"], 0)
        self.assertGreater(stats_bound["id_text_decode_count"], 0)
        before_filter = stats_bound["chunk_decode_count"]
        mapped_a._allowed_indices({"std_no": "GB 50000-2020", "kind": "clause"})
        self.assertEqual(store.cache_stats["chunk_decode_count"], before_filter)

        hit_a = mapped_a.get_clause("GB 50000-2020", "4.2.1")
        hit_b = mapped_b.get_clause("GB 50000-2020", "4.2.1")
        pristine_chunk = _chunk_payload(hit_b)
        hit_a.locator["source_spans"][0]["source_path"] = "task-a/changed.pdf"
        hit_a.text = "task-a changed text"
        self.assertEqual(_chunk_payload(hit_b), pristine_chunk)
        self.assertEqual(
            _chunk_payload(mapped_b.get_clause("GB 50000-2020", "4.2.1")),
            pristine_chunk,
        )

        total_chunks = len(mapped_a.chunks)
        before_top8 = store.cache_stats["chunk_decode_count"]
        top8_first = mapped_a.retrieve("共同过滤词", mode="bm25", top_k=8)
        top8_second = mapped_a.retrieve("共同过滤词", mode="bm25", top_k=8)
        self.assertEqual(
            [_chunk_payload(hit.chunk) for hit in top8_first.raw_hits],
            [_chunk_payload(hit.chunk) for hit in top8_second.raw_hits],
        )
        top8_chunk_decodes = store.cache_stats["chunk_decode_count"] - before_top8
        self.assertLess(top8_chunk_decodes, total_chunks)

        # Read each record in turn to force LRU eviction while keeping the
        # configured byte bound; no list of decoded records is retained here.
        for record in store.corpus.clauses:
            self.assertIsInstance(record["clause_id"], str)
        bounded = store.cache_stats
        self.assertLessEqual(bounded["cache_bytes"], 2048)
        self.assertGreater(bounded["cache_evictions"], 0)

    def test_owner_close_waits_for_reader_and_returned_objects_survive(self):
        core_dir, _bm25_dir = self.build_package(filler_count=32)
        store = BinaryCorpusStore.open(
            core_dir, source_dir=self.corpus_dir, cache_bytes=1024)
        sequence = store.corpus.clauses
        borrowed_record = sequence[0]
        borrowed_chunk = store.chunk_sequence("borrower")[0]
        original_text = borrowed_record["text"]
        original_chunk_text = borrowed_chunk.text
        entered = threading.Event()
        release = threading.Event()
        original_decode = store._decode_record_unleased

        def paused_decode(kind, row):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("synthetic read lease was not released")
            return original_decode(kind, row)

        executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(executor.shutdown, wait=True, cancel_futures=True)
        with patch.object(store, "_decode_record_unleased", side_effect=paused_decode):
            reader = executor.submit(lambda: sequence[1])
            self.assertTrue(entered.wait(3))
            closer = executor.submit(store.close)
            deadline = time.monotonic() + 3
            while store.cache_stats["state"] != "closing" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(store.cache_stats["state"], "closing")
            self.assertEqual(store.cache_stats["active_readers"], 1)
            self.assertFalse(closer.done())
            release.set()
            returned_record = reader.result(timeout=3)
            closer.result(timeout=3)

        self.assertEqual(store.cache_stats["state"], "closed")
        self.assertEqual(borrowed_record["text"], original_text)
        self.assertEqual(borrowed_chunk.text, original_chunk_text)
        self.assertIsInstance(returned_record["clause_id"], str)
        with self.assertRaises(BinaryCorpusStoreError):
            _ = sequence[0]

    def test_resource_loader_auto_json_mmap_hit_and_stale_or_bad_503(self):
        def app_for(root, resources):
            service = TaskService(archive_root=None)
            return create_app(
                service, resources=resources, root=root,
                auth_enabled=False, serve_frontend=False)

        # A current package is selected by auto and served through the existing
        # read-only catalog route.
        good_root = self.root / "good"
        source = good_root / "data" / "corpus"
        package = good_root / "data" / "runtime" / "core_corpus"
        _write_synthetic_corpus(source, filler_count=32)
        build_core_corpus_mmap(source, package)
        mmap_resources = StandardsResources(good_root)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_CORPUS_BACKEND": "auto"}):
            with TestClient(app_for(good_root, mmap_resources)) as client:
                response = client.get("/api/standards/catalog")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(mmap_resources._core_corpus_source, "mmap")
                self.assertIsNotNone(mmap_resources._corpus_store)

        # Missing CURRENT is an explicit JSON fallback in auto mode and emits
        # the operator warning. The source corpus is small and synthetic.
        json_root = self.root / "json-fallback"
        _write_synthetic_corpus(json_root / "data" / "corpus", filler_count=32)
        json_resources = StandardsResources(json_root)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_CORPUS_BACKEND": "auto"}):
            with TestClient(app_for(json_root, json_resources)) as client:
                with self.assertLogs(
                        "rag_arch.web.standards_resources", level="WARNING") as logs:
                    response = client.get("/api/standards/catalog")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json_resources._core_corpus_source, "json")
                self.assertTrue(any("JSON语料" in line for line in logs.output))

        # A package built from an older source snapshot must fail closed as 503.
        stale_root = self.root / "stale"
        stale_source = stale_root / "data" / "corpus"
        stale_package = stale_root / "data" / "runtime" / "core_corpus"
        _write_synthetic_corpus(stale_source, filler_count=32)
        build_core_corpus_mmap(stale_source, stale_package)
        clauses_path = stale_source / "clauses.jsonl"
        clauses_path.write_text(
            clauses_path.read_text(encoding="utf-8").replace(
                "主体结构共同过滤词", "过期版本主体结构文本"),
            encoding="utf-8",
        )
        stale_resources = StandardsResources(stale_root)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_CORPUS_BACKEND": "mmap"}):
            with TestClient(app_for(stale_root, stale_resources)) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertEqual(client.get("/api/standards/catalog").status_code, 503)

        # A malformed CURRENT target is also unavailable, never an implicit
        # fallback when mmap was selected explicitly.
        bad_root = self.root / "bad-package"
        bad_source = bad_root / "data" / "corpus"
        bad_package = bad_root / "data" / "runtime" / "core_corpus"
        _write_synthetic_corpus(bad_source, filler_count=32)
        build_core_corpus_mmap(bad_source, bad_package)
        (bad_package / "CURRENT").write_text("v1-" + "0" * 32 + "\n", encoding="ascii")
        bad_resources = StandardsResources(bad_root)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_CORPUS_BACKEND": "mmap"}):
            with TestClient(app_for(bad_root, bad_resources)) as client:
                self.assertEqual(client.get("/api/standards/catalog").status_code, 503)


if __name__ == "__main__":
    unittest.main()
