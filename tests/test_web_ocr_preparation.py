from concurrent.futures import CancelledError
import asyncio
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import threading
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
import httpx
import pymupdf

from rag_arch.cloud_ocr import OCRPageResult, OCRSettings, VLOCRClient
from rag_arch.models import LoadedDocument, TextUnit
from rag_arch.web.documents import DocumentStore, _document_cache_weight, _chat_cache_weight
from rag_arch.web.ocr_preparation import (OCRDocumentUnreadable, ocr_preparation_scope,
                                          supplement_document)
from rag_arch.web.task_worker import _is_external_work_start, _safe_error


def png():
    data = BytesIO()
    Image.new("RGB", (100, 100), "white").save(data, format="PNG")
    return data.getvalue()


def candidate(page, needs=False, reason="native_text", blank=False):
    return {"page": page, "needs_ocr": needs, "reason": reason, "blank": blank}


class Pages:
    def __init__(self, rows):
        self.rows, self.read = rows, []
    def ocr_candidates(self, source):
        return self.rows
    def read_page(self, source, page, **kwargs):
        self.read.append(page)
        return {"image_bytes": png()}


class Client:
    def __init__(self, status="completed", cached=False):
        self.calls, self.status, self.cached = [], status, cached
    def peek_cached_page(self, *, page, cache_dir):
        if self.cached:
            return OCRPageResult(self.status, page,
                                 text="荷载 1.5 kN/m" if self.status == "completed" else "",
                                 from_cache=True)
        return None

    def recognize_page(self, image, *, page, cache_dir, before_request):
        if not self.cached:
            before_request()
        self.calls.append((page, cache_dir))
        return OCRPageResult(self.status, page, text="荷载 1.5 kN/m" if self.status == "completed" else "",
                             from_cache=self.cached)


class OCRPreparationTests(unittest.TestCase):
    def loaded(self, kind="pdf"):
        return LoadedDocument("doc", "test."+kind, kind,
                              [TextUnit("原生正文", {"page": 1})], [], [2, 3], 3)

    def test_no_scope_means_no_network(self):
        client = Client()
        with self.assertRaises(RuntimeError):
            supplement_document(self.loaded(), Pages([]), "source", client, Path("unused"))
        self.assertEqual(client.calls, [])

    def test_only_candidates_and_native_units_unchanged(self):
        loaded = self.loaded()
        original = loaded.units[0]
        pages = Pages([candidate(1), candidate(2, True, "image_dominant_sparse_body"), candidate(3, blank=True)])
        client, guard = Client(), Mock()
        with ocr_preparation_scope(guard):
            result = supplement_document(loaded, pages, "source", client, Path("cache"))
        self.assertIs(result.units[0], original)
        self.assertEqual([p for p,_ in client.calls], [2])
        self.assertEqual(result.unread_pages, [])
        self.assertEqual(result.units[-1].locator["page"], 2)
        self.assertIs(result.units[-1].locator["recognition_verified"], False)
        self.assertTrue(any("核对原页" in note or "回看原页" in note for note in result.warnings))
        guard.assert_called_once_with(2)

    def test_ocr_does_not_relabel_word_paragraph_as_page(self):
        loaded = self.loaded("docx")
        loaded.units[0].locator = {"paragraph": 1}
        with ocr_preparation_scope(Mock()):
            result = supplement_document(loaded, Pages([candidate(1, True)]), "source", Client(), Path("cache"))
        self.assertEqual(result.units[0].locator, {"paragraph": 1})
        self.assertEqual(result.units[-1].locator["page_numbering"], "system_rendered_page")
        self.assertEqual(result.ocr_pages[0]["page_numbering"], "system_rendered_page")

    def test_cache_weights_include_expanded_ocr_text_not_only_upload_bytes(self):
        from types import SimpleNamespace
        loaded = LoadedDocument("doc", "tiny.pdf", "pdf", [TextUnit("字" * 10000, {"page": 1})], [], [])
        self.assertGreaterEqual(_document_cache_weight(loaded, 10), 40000)
        prepared = SimpleNamespace(collection=SimpleNamespace(documents=[loaded], chunks=[SimpleNamespace(text="字" * 10000)]))
        self.assertGreaterEqual(_chat_cache_weight(prepared, 10), 80000)

    def test_document_text_budget_does_not_silently_truncate_or_keep_spending(self):
        loaded, client = self.loaded(), Client()
        with patch("rag_arch.web.ocr_preparation.MAX_DOCUMENT_OCR_CHARACTERS", 16), ocr_preparation_scope(Mock()):
            result = supplement_document(loaded, Pages([candidate(1, True), candidate(2, True), candidate(3, True)]), "source", client, Path("cache"))
        self.assertEqual(len(result.units), 2)  # One native unit plus one complete OCR unit.
        self.assertLessEqual(sum(len(u.text) for u in result.units if u.locator.get("text_source") == "cloud_ocr"), 16)
        self.assertTrue(any(row.get("reason") == "document_text_budget" for row in result.ocr_pages))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(result.ocr_pages[1]["request_completed"])
        self.assertEqual(result.ocr_pages[2]["status"], "not_attempted")
        self.assertTrue(any("拆分" in note for note in result.warnings))

    def test_unknown_page_preserved_and_later_requests_not_sent(self):
        client = Client("unknown")
        loaded = self.loaded()
        with ocr_preparation_scope(Mock()):
            result = supplement_document(loaded, Pages([candidate(1, True), candidate(2, True)]), "source", client, Path("cache"))
        self.assertEqual([p for p,_ in client.calls], [1])
        self.assertEqual([row["status"] for row in result.ocr_pages], ["unknown", "not_attempted"])
        self.assertIn(1, result.unread_pages)
        self.assertEqual(len(result.units), 1)

    def test_task_budget_shared_and_cached_success_free(self):
        pages, guard = Pages([candidate(1, True), candidate(2, True)]), Mock()
        client = Client()
        with ocr_preparation_scope(guard, max_pages=1):
            first = supplement_document(self.loaded(), pages, "source", client, Path("cache"))
            cached = Client(cached=True)
            second = supplement_document(self.loaded(), pages, "source", cached, Path("cache"))
        self.assertEqual([r["status"] for r in first.ocr_pages], ["completed", "not_attempted"])
        self.assertEqual([r["status"] for r in second.ocr_pages], ["completed", "completed"])

    def test_cancel_guard_propagates_and_no_request(self):
        client = Client()
        with ocr_preparation_scope(Mock(side_effect=CancelledError())):
            with self.assertRaises(CancelledError):
                supplement_document(self.loaded(), Pages([candidate(1, True)]), "source", client, Path("cache"))
        self.assertEqual(client.calls, [])

    def test_old_no_ocr_warning_not_misreported_after_success(self):
        loaded = self.loaded()
        loaded.warnings = ["没有提取到可用于问答的文字；扫描件自动 OCR 尚未提供。", "以下物理页未完整读取：2。问答不能覆盖这些页面的图像正文。"]
        with ocr_preparation_scope(Mock()):
            result = supplement_document(loaded, Pages([candidate(1), candidate(2, True)]), "source", Client(), Path("cache"))
        self.assertFalse(any("尚未提供" in note for note in result.warnings))

    def test_api_store_default_is_no_ocr_even_when_enabled(self):
        with TemporaryDirectory() as temporary, patch.dict("os.environ", {"RAG_ARCH_OCR_ENABLED": "true"}):
            store = DocumentStore(Path(temporary), "workspace")
            self.addCleanup(store.close)
            self.assertIsNone(store.ocr_client)
            asset = store.add("text.pdf", b"%PDF-fake")
            with patch("rag_arch.web.documents.load_document", return_value=self.loaded()), patch.object(store, "pages_for_asset") as service:
                store.load(asset["asset_id"])
            service.assert_not_called()

    def test_worker_error_mentions_ocr_without_provider_secrets(self):
        exc = OCRDocumentUnreadable([{"page": 2, "status": "unknown"}])
        result = json.loads(_safe_error(exc, "plan"))
        self.assertEqual(result["code"], "ocr_reading_unavailable")
        self.assertIn("第 2 页", result["message"])
        self.assertTrue(_is_external_work_start("ocr", "正在识别"))

    def test_real_pdf_preview_zero_requests_and_reopened_store_reuses_page_cache(self):
        scanned = Image.new("RGB", (900, 1200), "white")
        draw = ImageDraw.Draw(scanned)
        for y in range(50, 1150, 50): draw.text((50, y), "Load 1.5 kN/m; span 600 mm", fill="black")
        image = BytesIO(); scanned.save(image, format="PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((50, 80), "Normal native text with units 1.5 kN/m.\n" * 20)
            page = pdf.new_page(); page.insert_image(page.rect, stream=image.getvalue())
            pdf.new_page()
            binary = pdf.tobytes()
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"output": {"choices": [{"finish_reason": "stop", "message": {"content": [{"text": "Load 1.5 kN/m; span 600 mm"}]}}]}, "usage": {"input_tokens": 100, "output_tokens": 20}})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = VLOCRClient(root, OCRSettings(True, "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "fake-test-key"), transport=httpx.MockTransport(handler))
            api = DocumentStore(root, "workspace")
            self.addCleanup(api.close)
            asset = api.add("mixed.pdf", binary)
            api.pages_for_asset(asset["asset_id"]).read_page("upload:"+asset["asset_id"], 2)
            self.assertEqual(requests, [])
            worker = DocumentStore(root, "workspace", ocr_client=client)
            worker._assets.update(api._assets)  # Stand-in for durable owner-scoped metadata reload.
            self.addCleanup(worker.close)
            with ocr_preparation_scope(Mock()): loaded = worker.load(asset["asset_id"])
            self.assertEqual(len(requests), 1)
            self.assertEqual([r["page"] for r in loaded.ocr_pages], [2])
            worker2 = DocumentStore(root, "workspace", ocr_client=client)
            worker2._assets.update(api._assets)
            self.addCleanup(worker2.close)
            with ocr_preparation_scope(Mock()): loaded2 = worker2.load(asset["asset_id"])
            self.assertEqual(len(requests), 1)
            self.assertTrue(loaded2.ocr_pages[0]["from_cache"])

    def test_async_cancellation_waits_for_the_ocr_thread_to_drain(self):
        from rag_arch.async_support import run_local
        from rag_arch.request_control import request_control_scope
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        client = Client()
        original = client.recognize_page
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(3)
            return original(*args, **kwargs)
        client.recognize_page = blocked
        def guard():
            if cancelled.is_set(): raise CancelledError()
        async def scenario():
            with request_control_scope(guard), ocr_preparation_scope(Mock()):
                task = asyncio.create_task(run_local(supplement_document, self.loaded(), Pages([candidate(1, True), candidate(2, True)]), "source", client, Path("cache")))
                try:
                    for _ in range(100):
                        if entered.is_set(): break
                        await asyncio.sleep(0.01)
                    self.assertTrue(entered.is_set())
                    cancelled.set(); task.cancel()
                    await asyncio.sleep(0.03)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError): await task
        asyncio.run(scenario())
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
