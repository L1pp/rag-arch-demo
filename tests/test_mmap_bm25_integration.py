"""Small-corpus integration tests for immutable read-only BM25 mappings."""
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_arch.binary_bm25 import BinaryBM25StoreError, build_core_bm25
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.models import Chunk
from rag_arch.retrieval import LocalIndex
from rag_arch.standards import StandardCorpus
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.standards_resources import PublicBM25UnavailableError
from rag_arch.web.standards_routes import create_standards_router


def corpus_fixture():
    clauses = [
        {"clause_id": "GB99999-2026/4.1.1", "std_id": "GB99999-2026",
         "std_no": "GB 99999-2026", "clause_no": "4.1.1",
         "text": "模板支撑条件应符合表格，数字值为1.8，边界不得超过18。",
         "table_ids": ["GB99999-2026/table/4.1.1"], "document_part": "body"},
        {"clause_id": "GB99999-2026/4.1.2", "std_id": "GB99999-2026",
         "std_no": "GB 99999-2026", "clause_no": "4.1.2",
         "text": "模板支撑条件应检查，数字值为1.8，边界不得超过18。",
         "table_ids": [], "document_part": "body"},
    ]
    tables = [{
        "table_id": "GB99999-2026/table/4.1.1", "std_id": "GB99999-2026",
        "std_no": "GB 99999-2026", "caption": "表4.1.1 模板支撑参数",
        "markdown": "| 条件 | 数值 |\n|---|---|\n| 模板支撑 | 1.8 |",
        "notes": ["最大值18。"], "content_status": "text_available",
        "owner_clause_id": "GB99999-2026/4.1.1", "document_part": "body",
    }]
    return StandardCorpus(clauses, tables)


class QueryOnlyEmbeddings:
    dimensions = 2

    def __init__(self):
        self.sync_calls = []
        self.async_calls = []
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        raise AssertionError("prepared matrix integration must not embed documents")

    async def aembed_documents(self, texts):
        self.document_calls += 1
        raise AssertionError("prepared matrix integration must not embed documents")

    def embed_query(self, text):
        self.sync_calls.append(text)
        return [1.0, 0.0]

    async def aembed_query(self, text):
        self.async_calls.append(text)
        return [1.0, 0.0]


class FailingDocumentEmbeddings:
    dimensions = 2

    async def aembed_documents(self, texts):
        raise RuntimeError("synthetic provider failure")


class BlockingDocumentEmbeddings:
    dimensions = 2

    def __init__(self, started):
        self.started = started
        self.never = asyncio.Event()

    async def aembed_documents(self, texts):
        self.started.set()
        await self.never.wait()


class MmapBM25IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = corpus_fixture()
        self.memory_index = EvidenceIndex(self.corpus)
        self.chunks = self.memory_index.chunks
        self.directory = self.root / "core_bm25"
        build_core_bm25(self.directory, chunks=self.chunks)

    def tearDown(self):
        self.memory_index.local.close_bm25()
        self.temp.cleanup()

    def _vectors(self):
        matrix = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64),
                         (len(self.chunks), 1))
        matrix.setflags(write=False)
        return matrix

    def test_sync_and_async_indices_match_exact_memory_scores_for_lexical_probes(self):
        memory = LocalIndex(self.chunks)
        mapped = LocalIndex(self.chunks, bm25_directory=self.directory)
        self.addCleanup(mapped.close_bm25)
        self.assertIsNone(mapped._counts)
        self.assertIsNone(mapped._lengths)
        self.assertIsNone(mapped._document_frequency)
        self.assertFalse(mapped._bm25_store.offsets.flags.writeable)
        self.assertFalse(mapped._bm25_store.postings.flags.writeable)
        self.assertEqual(mapped._average_length, memory._average_length)

        probes = ["模板支撑", "1.8", "18", "1.8≤18", "1.8 >= 18",
                  "≤ ≥ ≠", "", "definitely_oov_token", "相同词频"]
        for query in probes:
            with self.subTest(query=query):
                self.assertEqual(mapped._bm25_scores(query), memory._bm25_scores(query))
                self.assertEqual(mapped.search(query, top_k=30), memory.search(query, top_k=30))

        async def compare_async():
            async_mapped = await LocalIndex.acreate(
                self.chunks, bm25_directory=self.directory)
            try:
                async_memory = await LocalIndex.acreate(self.chunks)
                for query in probes:
                    self.assertEqual(async_mapped._bm25_scores(query),
                                     async_memory._bm25_scores(query))
                    actual = await async_mapped.asearch(query, top_k=30)
                    expected = await async_memory.asearch(query, top_k=30)
                    self.assertEqual([(h.chunk.chunk_id, h.score) for h in actual],
                                     [(h.chunk.chunk_id, h.score) for h in expected])
                async_memory.close_bm25()
            finally:
                async_mapped.close_bm25()

        asyncio.run(compare_async())
        mapped.close_bm25()

    def test_mmap_rrf_ties_filters_and_table_expansion_match_memory(self):
        vectors = self._vectors()
        memory_embeddings = QueryOnlyEmbeddings()
        self.memory_index.local.bind_prepared_vectors(vectors, memory_embeddings)
        self.memory_index.embeddings = memory_embeddings

        mapped_embeddings = QueryOnlyEmbeddings()
        mapped = EvidenceIndex(
            self.corpus, mapped_embeddings, prepared_vectors=vectors,
            bm25_directory=self.directory)
        self.addCleanup(mapped.local.close_bm25)
        try:
            cases = [
                ("模板支撑", "bm25", None),
                ("模板支撑", "rrf", None),
                ("模板支撑", "rrf", {"kind": "table"}),
                ("1.8", "rrf", {"std_no": "GB 99999-2026"}),
                ("an_oov_query", "rrf", {"document_part": "body"}),
            ]
            for query, mode, filters in cases:
                with self.subTest(query=query, mode=mode, filters=filters):
                    expected = self.memory_index.retrieve(
                        query, mode=mode, top_k=30 if mode == "bm25" else 8,
                        filters=filters)
                    actual = mapped.retrieve(
                        query, mode=mode, top_k=30 if mode == "bm25" else 8,
                        filters=filters)
                    self.assertEqual(actual.raw_ids, expected.raw_ids)
                    self.assertEqual([hit.score for hit in actual.raw_hits],
                                     [hit.score for hit in expected.raw_hits])
                    self.assertEqual(actual.context_ids, expected.context_ids)
                    self.assertEqual(actual.missing_evidence, expected.missing_evidence)

            table = mapped.retrieve("模板支撑", mode="rrf", top_k=8,
                                    filters={"kind": "table"})
            self.assertEqual(table.raw_ids, ["GB99999-2026/table/4.1.1"])
            self.assertIn("GB99999-2026/4.1.1", table.context_ids)
            tied = mapped.retrieve("an_oov_query", mode="rrf", top_k=8)
            self.assertEqual(tied.raw_ids,
                             sorted(chunk.chunk_id for chunk in mapped.chunks))
            self.assertEqual(mapped_embeddings.document_calls, 0)
        finally:
            mapped.local.close_bm25()

        async def check_async():
            embeddings = QueryOnlyEmbeddings()
            index = await EvidenceIndex.acreate(
                self.corpus, embeddings, prepared_vectors=vectors,
                bm25_directory=self.directory)
            try:
                expected = self.memory_index.retrieve(
                    "模板支撑", mode="rrf", top_k=8, filters={"kind": "table"})
                result = await index.aretrieve(
                    "模板支撑", mode="rrf", top_k=8, filters={"kind": "table"})
                self.assertEqual(result.raw_ids, expected.raw_ids)
                self.assertEqual(result.context_ids, expected.context_ids)
                self.assertEqual(embeddings.document_calls, 0)
                self.assertEqual(embeddings.async_calls, ["模板支撑"])
            finally:
                index.local.close_bm25()

        asyncio.run(check_async())

    def test_concurrent_threads_share_one_readonly_store_and_match_reference(self):
        reference = LocalIndex(self.chunks)
        mapped = LocalIndex(self.chunks, bm25_directory=self.directory)
        self.addCleanup(mapped.close_bm25)
        queries = ["模板", "支撑", "1.8", "18", "1.8≤18", "missing"] * 8
        expected = {query: reference._bm25_scores(query) for query in set(queries)}
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(mapped._bm25_scores, queries))
            self.assertEqual(results, [expected[query] for query in queries])
        finally:
            mapped.close_bm25()

    def test_task_wrappers_borrow_the_resource_mapping_until_owner_closes(self):
        resource_root = self.root / "resource_owner"
        resource_directory = resource_root / "data" / "runtime" / "core_bm25"
        build_core_bm25(resource_directory, chunks=self.chunks)
        resources = StandardsResources(resource_root, corpus=self.corpus)
        self.addCleanup(resources.close)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "mmap"}, clear=False):
            base = resources._reference_index(self.corpus)
        resources._bundle = SimpleNamespace(index=base)
        first, second = resources.plan_index(), resources.plan_index()
        store = base.local._bm25_store
        self.assertIsNotNone(store)
        self.assertIs(first.local._bm25_store, store)
        self.assertIs(second.local._bm25_store, store)
        self.assertIsNone(first.local._counts)
        self.assertIsNone(second.local._counts)
        self.assertEqual(first.local._bm25_scores("模板支撑"),
                         second.local._bm25_scores("模板支撑"))
        first.local.close_bm25()
        self.assertEqual(base.local._bm25_scores("模板支撑"),
                         second.local._bm25_scores("模板支撑"))

        resources.close()
        for label, wrapper in (("owner", base), ("first_borrower", first),
                               ("second_borrower", second)):
            with self.subTest(wrapper=label):
                with self.assertRaises(BinaryBM25StoreError):
                    wrapper.local._bm25_scores("模板支撑")
        resources.close()  # idempotent owner cleanup

    def test_missing_corrupt_package_policy_and_explicit_memory_rollback(self):
        absent = self.root / "absent"
        with self.assertRaises(BinaryBM25StoreError):
            LocalIndex(self.chunks, bm25_directory=absent)
        with self.assertRaises(BinaryBM25StoreError):
            from rag_arch.binary_bm25 import BinaryBM25Store
            BinaryBM25Store.open(self.directory, chunks=list(reversed(self.chunks)))

        auto_root = self.root / "auto"
        resources = StandardsResources(auto_root, corpus=self.corpus)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "auto"}, clear=False):
            with self.assertLogs("rag_arch.web.standards_resources", level="WARNING") as logs:
                legacy = resources._reference_index(self.corpus)
        self.assertIsNone(legacy.local._bm25_store)
        self.assertIsNotNone(legacy.local._counts)
        self.assertIn("尚未预建", " ".join(logs.output))
        resources.close()

        corrupt_root = self.root / "corrupt"
        corrupt_package = corrupt_root / "data" / "runtime" / "core_bm25"
        corrupt_package.mkdir(parents=True)
        (corrupt_package / "CURRENT").write_text("v1-invalid", encoding="ascii")
        broken_resources = StandardsResources(corrupt_root, corpus=self.corpus)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "auto"}, clear=False):
            with self.assertRaisesRegex(PublicBM25UnavailableError, "加载失败"):
                broken_resources._reference_index(self.corpus)
            self.assertIsNone(broken_resources._prepared_index)
            broken_resources._page_services[None] = object()
            app = FastAPI()
            app.state.standards = broken_resources
            app.include_router(create_standards_router())
            with TestClient(app) as client:
                response = client.get("/api/standards/search", params={"query": "模板支撑"})
            self.assertEqual(response.status_code, 503)
        broken_resources.close()

        missing_mmap = StandardsResources(self.root / "missing_mmap", corpus=self.corpus)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "mmap"}, clear=False):
            with self.assertRaises(PublicBM25UnavailableError):
                missing_mmap._reference_index(self.corpus)
        missing_mmap.close()

        stale_root = self.root / "stale"
        build_core_bm25(stale_root / "data" / "runtime" / "core_bm25",
                        chunks=self.chunks)
        stale_corpus = corpus_fixture()
        stale_corpus.clauses[0]["text"] += " stale source revision"
        stale = StandardsResources(stale_root, corpus=stale_corpus)
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "auto"}, clear=False):
            with self.assertRaises(PublicBM25UnavailableError):
                stale._reference_index(stale_corpus)
        self.assertIsNone(stale._prepared_index, "stale package must not silently rebuild in memory")
        stale.close()

        rollback_root = self.root / "rollback"
        rollback_resources = StandardsResources(rollback_root, corpus=self.corpus)
        rollback_directory = rollback_root / "data" / "runtime" / "core_bm25"
        rollback_directory.mkdir(parents=True)
        (rollback_directory / "CURRENT").write_text("v1-invalid", encoding="ascii")
        with patch.dict(os.environ, {"RAG_ARCH_CORE_BM25_BACKEND": "memory"}, clear=False):
            rollback = rollback_resources._reference_index(self.corpus)
        self.assertIsNone(rollback.local._bm25_store)
        self.assertIsNotNone(rollback.local._counts)
        rollback_resources.close()

    def test_async_embedding_failure_and_cancellation_release_open_mmaps(self):
        from rag_arch.binary_bm25 import BinaryBM25Store

        original_open = BinaryBM25Store.open.__func__
        opened = []

        def record_open(cls, directory, *, chunks):
            store = original_open(cls, directory, chunks=chunks)
            opened.append(store)
            return store

        async def fail_local():
            return await LocalIndex.acreate(
                self.chunks, FailingDocumentEmbeddings(), bm25_directory=self.directory)

        async def fail_evidence():
            return await EvidenceIndex.acreate(
                self.corpus, FailingDocumentEmbeddings(), bm25_directory=self.directory)

        with patch.object(BinaryBM25Store, "open", new=classmethod(record_open)):
            for create in (fail_local, fail_evidence):
                with self.subTest(operation=create.__name__):
                    before = len(opened)
                    with self.assertRaisesRegex(RuntimeError, "文档向量调用失败"):
                        asyncio.run(create())
                    self.assertEqual(len(opened), before + 1)
                    self.assertEqual(opened[-1]._state, "closed")

            async def cancel_create(create_index):
                started = asyncio.Event()
                provider = BlockingDocumentEmbeddings(started)
                task = asyncio.create_task(create_index(provider))
                await asyncio.wait_for(started.wait(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            async def cancel_local(provider):
                return await LocalIndex.acreate(
                    self.chunks, provider, bm25_directory=self.directory)

            async def cancel_evidence(provider):
                return await EvidenceIndex.acreate(
                    self.corpus, provider, bm25_directory=self.directory)

            for create in (cancel_local, cancel_evidence):
                with self.subTest(operation=create.__name__):
                    before = len(opened)
                    asyncio.run(cancel_create(create))
                    self.assertEqual(len(opened), before + 1)
                    self.assertEqual(opened[-1]._state, "closed")

    def test_sync_prepared_vector_validation_failure_releases_open_mmap(self):
        from rag_arch.binary_bm25 import BinaryBM25Store

        original_open = BinaryBM25Store.open.__func__
        opened = []

        def record_open(cls, directory, *, chunks):
            store = original_open(cls, directory, chunks=chunks)
            opened.append(store)
            return store

        invalid = np.ones((len(self.chunks), 2), dtype=np.float64)
        with patch.object(BinaryBM25Store, "open", new=classmethod(record_open)):
            with self.assertRaisesRegex(ValueError, "只读"):
                LocalIndex(self.chunks, prepared_vectors=invalid,
                           bm25_directory=self.directory)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]._state, "closed")


if __name__ == "__main__":
    unittest.main()
