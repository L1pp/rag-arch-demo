"""Source identity, cross-page links and the new read-only reference routes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pymupdf
from fastapi.testclient import TestClient

from rag_arch.standards import StandardCorpus
from rag_arch.config import Settings
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.web.app import create_app
from rag_arch.web.catalog_tools import catalog_request_scope
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.task_service import TaskService


class StandardsResourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, count in (("selected.pdf", 3), ("other-copy.pdf", 5)):
            with pymupdf.open() as doc:
                for number in range(count):
                    doc.new_page().insert_text((40, 40), f"{name} physical page {number + 1}")
                doc.save(self.root / name)
        self.catalog = [
            {"std_id": "GB99999-2099", "std_no": "GB 99999-2099", "title": "测试施工规范",
             "library_class": "A", "text_available": True, "source_files": [
                 {"path": "other-copy.pdf"}, {"path": "selected.pdf"}]},
            {"std_id": "GB99999-2098", "std_no": "GB 99999-2098", "title": "测试施工规范旧版",
             "library_class": "C", "entry_kind": "metadata_only", "source_files": []},
        ]
        self.record = {"clause_id": "GB99999-2099/4.2.4", "std_id": "GB99999-2099",
                       "std_no": "GB 99999-2099", "clause_no": "4.2.4", "text": "跨页条文",
                       "source_spans": [{"source_path": "selected.pdf", "page_start": 2, "page_end": 3}]}
        unmapped = {**self.record, "clause_id": "GB99999-2099/5.1", "clause_no": "5.1", "source_spans": []}
        self.corpus = StandardCorpus([self.record, unmapped], [], catalog=self.catalog)
        self.resources = StandardsResources(self.root, corpus=self.corpus)
        self.service = TaskService(archive_root=None)
        self.client = TestClient(create_app(self.service, resources=self.resources, root=self.root, auth_enabled=False))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_catalog_chinese_names_and_metadata_only_no_original(self):
        response = self.client.get("/api/standards/catalog")
        self.assertEqual(response.status_code, 200)
        current, old = response.json()["entries"]
        self.assertEqual(current["title"], "测试施工规范")
        self.assertIn("/viewer/pages?", current["page_url"])
        self.assertIsNone(old["page_url"])
        self.assertIn("2098", old["history_url"])
        self.assertIsNone(self.resources._bundle, "Catalog browsing must not build a retrieval index")

    def test_cross_page_view_retains_exact_source_not_other_copy(self):
        citation = self.resources.citation({**self.record, "evidence_id": self.record["clause_id"]})
        query = parse_qs(urlsplit(citation["page_url"]).query)
        response = self.client.get("/api/standards/pages", params={key: values[0] for key, values in query.items()})
        self.assertEqual(response.status_code, 200)
        view = response.json()
        self.assertEqual(view["filename"], "selected.pdf")
        self.assertEqual([item["page"] for item in view["pages"]], [2, 3])
        self.assertEqual(view["page_count"], 3)
        self.assertEqual([s["filename"] for s in view["sources"]], ["selected.pdf"])
        for page in view["pages"]:
            rendered = self.client.get(page["image_url"])
            self.assertEqual(rendered.status_code, 200)
            self.assertEqual(rendered.headers["content-type"], "image/png")
            self.assertTrue(rendered.content.startswith(b"\x89PNG"))

    def test_paging_is_explicit_and_missing_mapping_not_guessed(self):
        view = self.resources.page_view("GB99999-2099", evidence=self.record["clause_id"], page=1)
        self.assertEqual([p["page"] for p in view["pages"]], [1])
        missing = self.client.get("/api/standards/pages", params={"std": "GB99999-2099", "evidence": "GB99999-2099/5.1"})
        self.assertEqual(missing.status_code, 422)
        citation = self.resources.citation({"std_no": "GB 99999-2099", "evidence_id": "GB99999-2099/5.1"})
        self.assertIsNone(citation["page_url"])

    def test_old_version_missing_original_ambiguous_code_and_arbitrary_source(self):
        self.assertEqual(self.client.get("/api/standards/pages", params={"std": "GB99999-2098"}).status_code, 422)
        self.assertEqual(self.client.get("/api/standards/pages", params={"std": "GB99999"}).status_code, 422)
        self.assertEqual(self.client.get("/api/standards/pages", params={"std": "UNKNOWN"}).status_code, 404)
        self.assertEqual(self.client.get("/api/standards/page-image", params={
            "std": "GB99999-2099", "source_ref": "../../.env", "page": 1}).status_code, 404)
        self.assertEqual(self.client.get("/api/standards/pages", params={
            "std": "GB99999-2099", "page": 600}).status_code, 422)
        with self.assertRaises(KeyError):
            self.resources._evidence(self.resources.resolve("GB99999-2098"), self.record["clause_id"])

    def test_shared_index_but_independent_selected_source_services(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            bundles = list(executor.map(lambda _: self.resources.get(), range(3)))
        self.assertTrue(all(bundle is bundles[0] for bundle in bundles))
        self.assertIsNot(self.resources.page_service("GB 99999-2099"), bundles[0].pages)

    def test_plan_indices_clone_run_state_and_share_only_read_only_vectors(self):
        class Embeddings:
            dimensions = 2
            before_request = None
            persist_batches = False

            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

            def clone(self):
                return Embeddings()

        base = EvidenceIndex(self.corpus, embeddings=Embeddings())
        resources = StandardsResources(self.root, corpus=self.corpus)
        resources._prepared_index = base

        first, second = resources.plan_index(), resources.plan_index()

        self.assertIs(first.corpus, base.corpus)
        self.assertIs(first.chunks, base.chunks)
        self.assertIsNot(first, second)
        self.assertIsNot(first.local, second.local)
        self.assertIsNot(first.embeddings, second.embeddings)
        self.assertIs(first.local._vectors, base.local._vectors)
        self.assertIs(first.local.embeddings, first.embeddings)
        self.assertIs(second.local.embeddings, second.embeddings)
        self.assertIsNone(first.reranker)
        self.assertIsNone(second.reranker)

        callback = lambda: None
        first.embeddings.before_request = callback
        first.reranker = object()
        self.assertIs(second.embeddings.before_request, None)
        self.assertIsNone(second.reranker)
        self.assertIsNone(base.reranker)

    def test_viewing_does_not_create_tasks_or_events(self):
        before = self.service.snapshot()
        self.client.get("/api/standards/catalog")
        view = self.client.get("/api/standards/pages", params={"std": "GB99999-2099"}).json()
        self.client.get(view["pages"][0]["image_url"])
        self.assertEqual(before, self.service.snapshot())

    def test_renderer_exception_is_contained_with_readable_message(self):
        before = self.service.snapshot()
        with patch.object(self.resources, "page_image", side_effect=RuntimeError("worker stopped")):
            with self.assertLogs("rag_arch.web.standards_routes", level="ERROR"):
                response = self.client.get("/api/standards/page-image", params={
                    "std": "GB99999-2099", "source_ref": "standard:1", "page": 1})
        self.assertEqual(response.status_code, 503)
        self.assertIn("当前页面暂不支持", response.json()["detail"])
        self.assertNotIn("worker stopped", response.text)
        self.assertEqual(before, self.service.snapshot())

    def test_more_than_two_evidence_pages_remain_navigable(self):
        self.record["source_spans"][0]["page_start"] = 1
        view = self.resources.page_view("GB99999-2099", evidence=self.record["clause_id"])
        self.assertEqual(view["sources"][0]["pages"], [1, 2, 3])
        self.assertEqual([p["page"] for p in view["pages"]], [1, 2])
        last = self.resources.page_view("GB99999-2099", evidence=self.record["clause_id"], page=3)
        self.assertEqual(last["pages"][0]["page"], 3)

    def test_prepared_vectors_load_without_document_requests_and_query_uses_provider(self):
        texts = [row["text"] for row in self.corpus.clauses]
        texts = [chunk.text for chunk in EvidenceIndex(self.corpus).chunks]
        cache = self.root / "data/runtime/core_embeddings.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"schema_version": 1,
            "identity": {"base_url": "https://example.invalid/v1",
                          "model": "prepared-model", "dimensions": 2},
            "vector_dimensions": 2,
            "documents": {text: [1.0, 0.0] for text in texts}, "queries": {}},
            ensure_ascii=False), encoding="utf-8")

        class Provider:
            dimensions = 2
            chunk_size = 20
            def __init__(self):
                self.document_calls = 0
                self.query_calls = 0
            def embed_documents(self, values):
                self.document_calls += 1
                raise AssertionError("prepared Web load must not request document vectors")
            def embed_query(self, value):
                self.query_calls += 1
                return [0.0, 1.0]

        provider = Provider()
        settings = Settings(embedding_model="prepared-model",
                            embedding_base_url="https://example.invalid/v1",
                            embedding_api_key="synthetic", embedding_dimensions=2)
        with patch("rag_arch.providers.create_embeddings", return_value=provider):
            self.resources.configure_embeddings(settings, backend="json")
        bundle = self.resources.get()
        self.assertIsNotNone(bundle.index.embeddings)
        bundle.index.embeddings.embed_query("新问题")
        self.assertEqual(provider.document_calls, 0)
        self.assertEqual(provider.query_calls, 1)

    def test_catalog_candidates_use_prepared_metadata_vectors_without_document_requests(self):
        self.resources._inventory()
        texts = list(self.resources._catalog._metadata_texts)
        cache = self.root / "data/runtime/catalog_embeddings.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"schema_version": 1,
            "identity": {"base_url": "https://example.invalid/v1",
                          "model": "prepared-model", "dimensions": 2},
            "vector_dimensions": 2,
            "documents": {text: [1.0, 0.0] for text in texts}, "queries": {}},
            ensure_ascii=False), encoding="utf-8")

        class Provider:
            dimensions = 2
            chunk_size = 20
            def __init__(self):
                self.document_calls = 0
                self.query_calls = 0
            def embed_documents(self, values):
                self.document_calls += 1
                raise AssertionError("catalog metadata vectors must be prebuilt")
            def embed_query(self, value):
                self.query_calls += 1
                return [0.0, 1.0]

        provider = Provider()
        settings = Settings(embedding_model="prepared-model",
                            embedding_base_url="https://example.invalid/v1",
                            embedding_api_key="synthetic", embedding_dimensions=2)
        with patch("rag_arch.providers.create_embeddings", return_value=provider):
            self.resources.configure_catalog(settings)
            self.resources.catalog_candidates("施工规范")
        self.assertEqual(provider.document_calls, 0)
        self.assertEqual(provider.query_calls, 1)

    def test_prepared_catalog_query_runs_request_guard(self):
        self.resources._inventory()
        texts = list(self.resources._catalog._metadata_texts)
        cache = self.root / "data/runtime/catalog_embeddings.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"schema_version": 1,
            "identity": {"base_url": "https://example.invalid/v1",
                          "model": "prepared-model", "dimensions": 2},
            "vector_dimensions": 2,
            "documents": {text: [1.0, 0.0] for text in texts}, "queries": {}},
            ensure_ascii=False), encoding="utf-8")

        class Provider:
            dimensions = 2
            chunk_size = 20
            def embed_documents(self, values):
                raise AssertionError("metadata documents must already be prepared")
            def embed_query(self, value):
                return [0.0, 1.0]

        settings = Settings(embedding_model="prepared-model",
                            embedding_base_url="https://example.invalid/v1",
                            embedding_api_key="synthetic", embedding_dimensions=2)
        seen = []
        with patch("rag_arch.providers.create_embeddings", return_value=Provider()):
            self.resources.configure_catalog(settings)
            with catalog_request_scope(lambda: seen.append(True)):
                self.resources.catalog_candidates("施工规范")
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
