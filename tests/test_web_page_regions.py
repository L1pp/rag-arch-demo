"""Protected region routes, source links and public result propagation."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from rag_arch.standards import StandardCorpus
from rag_arch.multimodal import image_source_record
from rag_arch.pdf_backend import ProcessPdfBackend
from rag_arch.web.app import create_app
from rag_arch.web.page_region_query import page_source_url, parse_region_query
from rag_arch.web.plan_runner import _citation_rows, _public_page_sources
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.task_service import TaskService
from tests import test_web_document_api as document_api
from tests import test_web_standards_resources as standards_api


REGION = [0.1, 0.2, 0.6, 0.45]
REGION_QUERY = "0.1,0.2,0.6,0.45"


class DocumentRegionHTTPTests(unittest.TestCase):
    setUp = document_api.DocumentAPITests.setUp
    tearDown = document_api.DocumentAPITests.tearDown
    upload = document_api.DocumentAPITests.upload

    def test_uploaded_plan_model_page_links_resolve_through_real_document_store(self):
        asset = self.upload("uploaded-plan.pdf", self.pdf).json()["asset_id"]
        store = self.client.app.state.documents
        loaded = store.get_plan(attachment_ids=[asset])
        plan_pages = store.pages_for_plan(loaded.document, loaded.asset)
        resources = SimpleNamespace(citation=lambda row: self.fail("uploaded plan is not a standard"))
        base = f"/viewer/pages?asset={asset}&workspace_id={self.ws}"
        for region in (None, REGION):
            with self.subTest(region=region):
                image = plan_pages.read_page("plan:current", 1, **({"region": region} if region else {}))
                source = image_source_record(image, sent=True)
                identifier = source.get("image_id") or source["evidence_id"]
                evidence = {"evidence_id": identifier, "kind": "page_image", "source_kind": "plan", "page_source": source}
                output = SimpleNamespace(evidence=[evidence], evidence_ids=[identifier], findings=[], runtime={"page_sources": [source]})
                citation = _citation_rows(output, resources, base, "uploaded-plan.pdf")[0]
                public_source = _public_page_sources(output, resources, base)[0]
                self.assertEqual(citation["page_url"], public_source["page_url"])
                query = urlsplit(citation["page_url"]).query
                view = self.client.get("/api/documents/pages?" + query)
                self.assertEqual(view.status_code, 200, view.text)
                self.assertEqual(view.json()["source_ref"], f"upload:{asset}")
                image_response = self.client.get(view.json()["pages"][0]["image_url"])
                self.assertEqual(image_response.status_code, 200, image_response.text[:100])
                self.assertTrue(image_response.content.startswith(b"\x89PNG"))
                self.assertEqual(public_source["source_ref"], "plan:current", "model provenance must not be rewritten")
                self.assertEqual(source["source_ref"], "plan:current")
                self.assertEqual(public_source.get("region"), region)
        unregistered_alias = self.client.get("/api/documents/pages", params={
            "asset": asset, "workspace_id": self.ws, "source_ref": "plan:current", "page": 1})
        self.assertEqual(unregistered_alias.status_code, 404, "the API must still reject task-local source aliases")

    def test_region_preview_returns_protected_high_resolution_crop_without_task(self):
        asset = self.upload("region.pdf", self.pdf).json()["asset_id"]
        params = {"asset": asset, "workspace_id": self.ws, "region": REGION_QUERY}
        view = self.client.get("/api/documents/pages", params=params)
        self.assertEqual(view.status_code, 200, view.text)
        self.assertEqual(view.json()["pages"][0]["region"], REGION)
        image = self.client.get(view.json()["pages"][0]["image_url"])
        self.assertEqual(image.status_code, 200, image.text[:100])
        self.assertEqual(image.headers["Cache-Control"], "private, no-store")
        self.assertEqual(image.headers["Vary"], "Cookie")
        with Image.open(BytesIO(image.content)) as crop:
            self.assertGreater(crop.width, 1000)
            self.assertAlmostEqual(crop.width / crop.height, (595 * .5) / (842 * .25), delta=.02)
            self.assertLessEqual(crop.width * crop.height, 4_000_000)
        self.assertEqual(self.requests, [])

    def test_malformed_regions_and_unregistered_sources_do_not_fall_back_to_full_page(self):
        asset = self.upload("region.pdf", self.pdf).json()["asset_id"]
        params = {"asset": asset, "workspace_id": self.ws}
        view = self.client.get("/api/documents/pages", params=params).json()
        source = view["source_ref"]
        for invalid in ("", "0,0,1", "0,0,1,1,1", "0,0,NaN,1", "0,0,Infinity,1", "0,0,True,1", "0,0,2,1", "0.5,0,0.4,1"):
            for endpoint in ("/api/documents/pages", "/api/documents/page-image"):
                with self.subTest(region=invalid, endpoint=endpoint):
                    response = self.client.get(endpoint, params={**params, "source_ref": source, "page": 1, "region": invalid})
                    self.assertEqual(response.status_code, 422, response.text)
        for endpoint in ("/api/documents/pages", "/api/documents/page-image"):
            response = self.client.get(endpoint, params={**params, "source_ref": "../../.env", "page": 1, "region": REGION_QUERY})
            self.assertEqual(response.status_code, 404, response.text)

    def test_supervised_tiny_region_is_422_and_the_worker_can_read_afterward(self):
        backend = ProcessPdfBackend(call_timeout=10)
        store = self.client.app.state.documents
        previous_backend = store.pdf_backend
        store.pdf_backend = backend
        try:
            asset = self.upload("tiny-region.pdf", self.pdf).json()["asset_id"]
            params = {"asset": asset, "workspace_id": self.ws}
            view = self.client.get("/api/documents/pages", params=params)
            self.assertEqual(view.status_code, 200, view.text)
            source_ref = view.json()["source_ref"]
            tiny = self.client.get("/api/documents/page-image", params={**params,
                "source_ref": source_ref, "page": 1, "region": "0,0,1e-300,0.5"})
            self.assertEqual(tiny.status_code, 422, tiny.text)
            self.assertIn("区域过小", tiny.json()["detail"])
            self.assertNotIn(str(self.root), tiny.text)
            normal = self.client.get(view.json()["pages"][0]["image_url"])
            self.assertEqual(normal.status_code, 200, normal.text[:100])
            self.assertTrue(normal.content.startswith(b"\x89PNG"))
            self.assertEqual(len(backend.worker_pids), 1)
        finally:
            store.pdf_backend = previous_backend
            backend.close()


class StandardRegionHTTPTests(unittest.TestCase):
    setUp = standards_api.StandardsResourcesTests.setUp
    tearDown = standards_api.StandardsResourcesTests.tearDown

    def test_standard_region_citation_keeps_source_and_region(self):
        service = self.resources.page_service("GB 99999-2099")
        source = next(row for row in service.sources() if row["filename"] == "selected.pdf")
        record = {"evidence_id": "PAGE:ignored:2:REGION:0.1,0.2,0.6,0.45", "kind": "page_image",
                  "std_no": "GB 99999-2099", "text": "image", "page_source": {**source, "page": 2, "region": REGION}}
        link = self.resources.citation(record)["page_url"]
        query = {key: values[0] for key, values in parse_qs(urlsplit(link).query).items()}
        self.assertEqual(query["page"], "2")
        self.assertEqual(query["source_ref"], source["source_ref"])
        self.assertEqual(query["region"], REGION_QUERY)
        view = self.client.get("/api/standards/pages", params=query)
        self.assertEqual(view.status_code, 200, view.text)
        image = self.client.get(view.json()["pages"][0]["image_url"])
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.content.startswith(b"\x89PNG"))
        self.assertIn("no-store", image.headers["Cache-Control"])

    def test_region_requires_page_and_valid_shape(self):
        self.assertEqual(self.client.get("/api/standards/pages", params={"std": "GB99999-2099", "region": REGION_QUERY}).status_code, 422)
        self.assertEqual(self.client.get("/api/standards/pages", params={"std": "GB99999-2099", "page": 2, "region": "0,0,0,1"}).status_code, 422)

    def test_unknown_standard_source_is_404_for_directory_and_image(self):
        for endpoint in ("/api/standards/pages", "/api/standards/page-image"):
            response = self.client.get(endpoint, params={"std": "GB99999-2099", "source_ref": "other-source", "page": 1, "region": REGION_QUERY})
            self.assertEqual(response.status_code, 404, response.text)


class RegionPublicResultTests(unittest.TestCase):
    def test_region_citations_and_consulted_images_use_explicit_page_metadata(self):
        source = {"image_id": "PAGE:plan:current:7:REGION:0.1,0.2,0.6,0.45", "source_ref": "plan:current",
                  "page": 7, "region": REGION, "filename": "plan.docx", "sent_to_model": True, "source_kind": "plan"}
        source["evidence_id"] = source["image_id"]
        record = {"evidence_id": source["image_id"], "kind": "page_image", "page_source": source, "source_kind": "plan"}
        output = SimpleNamespace(evidence=[record], evidence_ids=[source["image_id"]], findings=[],
                                 runtime={"page_sources": [source]})
        resources = SimpleNamespace(citation=lambda row: self.fail("plan source must not use standards mapping"))
        url = "/viewer/pages?asset=private&workspace_id=owner"
        rows = _citation_rows(output, resources, url, "plan.docx")
        public = _public_page_sources(output, resources, url)
        self.assertEqual(rows[0]["page_url"], public[0]["page_url"])
        self.assertEqual(parse_qs(urlsplit(rows[0]["page_url"]).query)["page"], ["7"])
        self.assertTrue(public[0]["cited"])
        self.assertEqual(public[0]["region"], REGION)
        self.assertFalse(source.get("cited", False), "mapping must not mutate stored source records")

    def test_task_http_contract_preserves_the_visual_ledger(self):
        ledger = {"images": [{"image_id": "PAGE:plan:current:1", "state": "response_received"}],
                  "attempts": [{"request_index": 1, "response_received": True}], "initial_selection": {"selected_pages": [1]}}
        with TemporaryDirectory(prefix="rag-region-contract-") as directory:
            root = Path(directory)
            service = TaskService(runner=lambda request, progress, cancel: {"text": "done", "visual_review": ledger},
                                  archive_root=None, execution_modes={"chat": "model", "standards": "model", "plan": "model"})
            resources = StandardsResources(root, corpus=StandardCorpus([], []))
            with TestClient(create_app(service, root=root, resources=resources, auth_enabled=False)) as client:
                response = client.post("/api/tasks", json={"workspace_id": service.workspace_id, "assistant_id": "chat",
                    "request_id": "region-contract", "prompt": "test"})
                self.assertEqual(response.status_code, 202, response.text)
                task_id = response.json()["task_id"]
                for _ in range(100):
                    task = client.get(f"/api/tasks/{task_id}", params={"workspace_id": service.workspace_id}).json()
                    if task["status"] == "completed":
                        break
                    time.sleep(.005)
                self.assertEqual(task["result"]["visual_review"], ledger)

    def test_region_query_keeps_coordinates_without_parsing_image_id(self):
        self.assertEqual(parse_region_query(REGION_QUERY), tuple(REGION))
        self.assertIn("region=", page_source_url("/viewer/pages?asset=x", {"page": 3, "region": REGION}))


if __name__ == "__main__":
    unittest.main()
