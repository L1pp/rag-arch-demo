"""OCR failure and per-task limits retain readable source-page chat paths."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
import httpx
import pymupdf

from rag_arch.cloud_ocr import OCRPageResult, OCRSettings, VLOCRClient
from rag_arch.config import Settings
from rag_arch.models import Answer
from rag_arch.multimodal import image_source_record
from rag_arch.web.chat_runner import arun_chat, run_chat
from rag_arch.web.documents import DocumentStore
from rag_arch.web.ocr_preparation import OCRDocumentUnreadable, ocr_preparation_scope
from rag_arch.web.plan_runner import _require_plan_text, arun_plan, run_plan
from tests.test_web_async_runners import AsyncOnlyModel


def scan_pdf(page_count=1):
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    for y in range(50, 1150, 50):
        draw.text((50, y), "Fictional visual page, check 6.5m.", fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    with pymupdf.open() as pdf:
        for _ in range(page_count):
            page = pdf.new_page()
            page.insert_image(page.rect, stream=stream.getvalue())
        return pdf.tobytes(deflate=True)


class FakeOCR:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def recognize_page(self, image, *, page, cache_dir, before_request):
        before_request()
        self.calls.append(page)
        return OCRPageResult(self.status, page,
            text="合成 OCR 文字" if self.status == "completed" else "",
            error_code="synthetic_failure" if self.status != "completed" else None)


def ocr_client(root, calls, status="completed"):
    def handler(request):
        calls.append(request)
        if status == "unknown":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if status == "failed":
            return httpx.Response(503, json={"code": "SyntheticUnavailable", "message": "synthetic failure"})
        return httpx.Response(200, json={"output": {"choices": [{"finish_reason": "stop",
            "message": {"content": [{"text": "合成 OCR 文字"}]}}]}})
    settings = OCRSettings(True, "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                           "synthetic-ocr-key", 15)
    return VLOCRClient(root, settings, transport=httpx.MockTransport(handler))


def answer_from_page(question, **kwargs):
    pages = kwargs["page_service"]
    source = pages.sources()[0]
    image = pages.read_page(source["source_ref"], 1)
    assert image["image_bytes"].startswith(b"\x89PNG")
    return Answer("合成页图已读取", [], [], True, citation_check={},
                  page_sources=[image_source_record(image, sent=True)])


async def async_answer_from_page(question, **kwargs):
    return answer_from_page(question, **kwargs)


class OCRPageFallbackTests(unittest.TestCase):
    def setUp(self):
        self.pdf = scan_pdf()
        # Keep both PDFs and their prepared collection below the LRU capacity;
        # eviction must not hide whether failed OCR was incorrectly cached.
        self.assertLess(len(self.pdf), 200_000)
        self.settings = Settings(chat_model="synthetic", chat_base_url="https://example.invalid/v1",
                                 chat_api_key="synthetic-key", chat_max_tokens=256,
                                 chat_context_budget_tokens=4096)

    def test_failed_and_unknown_ocr_still_allow_page_chat_sync_and_async(self):
        for status in ("failed", "unknown"):
            for asynchronous in (False, True):
                with self.subTest(status=status, asynchronous=asynchronous), TemporaryDirectory() as temp:
                    root = Path(temp)
                    ocr = FakeOCR(status)
                    store = DocumentStore(root, "workspace", ocr_client=ocr)
                    self.addCleanup(store.close)
                    asset = store.add("fictional-scan.pdf", self.pdf)
                    request = {"task_id": "chat-scan", "prompt": "合成问题",
                               "attachment_ids": [asset["asset_id"]]}
                    model = AsyncOnlyModel()
                    with ocr_preparation_scope(Mock()), \
                         patch("rag_arch.runtime_audit.AUDIT_ROOT", root / "audits"), \
                         patch("rag_arch.web.chat_runner.create_web_search_provider", return_value=None), \
                         patch("rag_arch.web.chat_runner.answer_question", side_effect=answer_from_page), \
                         patch("rag_arch.qa.aanswer_question", side_effect=async_answer_from_page):
                        if asynchronous:
                            result = asyncio.run(arun_chat(request, lambda *_: None, Event(),
                                settings=self.settings, documents=store, model_factory=lambda _: model))
                        else:
                            result = run_chat(request, lambda *_: None, Event(),
                                settings=self.settings, documents=store, model_factory=lambda _: model)
                    self.assertEqual(ocr.calls, [1])
                    self.assertEqual(result["file_mode"], "pages_only")
                    self.assertEqual(result["files"][0]["ocr_pages"][0]["status"], status)
                    self.assertFalse(result["files"][0]["ocr_pages"][0]["recognition_verified"])
                    self.assertEqual(result["page_sources"][0]["page"], 1)
                    self.assertIn("OCR", " ".join(result["warnings"]))
                    if asynchronous:
                        self.assertTrue(model.client.closed)
                    store.close()

    def test_exhausted_page_budget_keeps_both_attachments_available(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ocr = FakeOCR("completed")
            store = DocumentStore(root, "workspace", ocr_client=ocr)
            self.addCleanup(store.close)
            first = store.add("fictional-a.pdf", self.pdf)
            second = store.add("fictional-b.pdf", self.pdf)
            request = {"task_id": "chat-two-scans", "prompt": "合成问题",
                       "attachment_ids": [first["asset_id"], second["asset_id"]]}
            with ocr_preparation_scope(Mock(), max_pages=1), \
                 patch("rag_arch.runtime_audit.AUDIT_ROOT", root / "audits"), \
                 patch("rag_arch.web.chat_runner.create_web_search_provider", return_value=None), \
                 patch("rag_arch.web.chat_runner.answer_question", side_effect=answer_from_page):
                result = run_chat(request, lambda *_: None, Event(), settings=self.settings,
                                  documents=store, model_factory=lambda _: AsyncOnlyModel())
            self.assertEqual(ocr.calls, [1])
            self.assertEqual({row["filename"] for row in result["files"]},
                             {"fictional-a.pdf", "fictional-b.pdf"})
            statuses = {row["ocr_pages"][0]["status"] for row in result["files"]}
            self.assertEqual(statuses, {"completed", "not_attempted"})
            self.assertIn("未尝试", " ".join(result["warnings"]))
            store.close()

    def test_partial_document_budget_deferral_is_not_cached_between_tasks(self):
        pdf = scan_pdf(page_count=2)
        self.assertLess(len(pdf), 200_000)
        for workflow in ("loaded", "chat", "async_chat", "plan"):
            with self.subTest(workflow=workflow), TemporaryDirectory() as temp:
                root = Path(temp)
                calls = []
                store = DocumentStore(root, "workspace", ocr_client=ocr_client(root, calls))
                self.addCleanup(store.close)
                asset = store.add("fictional-two-page.pdf", pdf)
                ids = [asset["asset_id"]]
                def prepare():
                    if workflow == "loaded":
                        return store.load(ids[0])
                    if workflow == "chat":
                        return store.prepare_chat(ids)
                    if workflow == "async_chat":
                        return asyncio.run(store.aprepare_chat(ids))
                    return store.get_plan(attachment_ids=ids)
                def selected(value):
                    if workflow == "loaded":
                        return value
                    if workflow in {"chat", "async_chat"}:
                        return value.collection.documents[0]
                    return value.document
                with ocr_preparation_scope(Mock(), max_pages=1):
                    initial = selected(prepare())
                    self.assertEqual(len(calls), 1)
                    self.assertEqual([row["status"] for row in initial.ocr_pages],
                                     ["completed", "not_attempted"])
                    self.assertEqual(initial.ocr_pages[1]["reason"], "task_page_budget")
                    self.assertEqual(len(initial.units), 1)
                    repeated = selected(prepare())
                    self.assertEqual(len(calls), 1, "same task must not bypass its page budget")
                    self.assertEqual(repeated.ocr_pages[1]["reason"], "task_page_budget")
                with ocr_preparation_scope(Mock(), max_pages=1):
                    completed_value = prepare()
                completed = selected(completed_value)
                self.assertEqual(len(calls), 2, "previously unattempted page is requested once")
                self.assertEqual([row["status"] for row in completed.ocr_pages],
                                 ["completed", "completed"])
                self.assertTrue(completed.ocr_pages[0]["from_cache"])
                self.assertFalse(completed.ocr_pages[1]["from_cache"])
                self.assertEqual(len(completed.units), 2)
                with ocr_preparation_scope(Mock(), max_pages=1):
                    cached_value = prepare()
                self.assertIs(cached_value, completed_value)
                self.assertEqual(len(calls), 2)
                store.close()

    def test_other_partial_ocr_outcomes_keep_existing_cache_and_no_retries(self):
        pdf = scan_pdf(page_count=2)
        text = "合成 OCR 文字"
        for second_status in ("failed", "unknown", "document_text_budget"):
            with self.subTest(second_status=second_status), TemporaryDirectory() as temp:
                root = Path(temp)
                calls = []
                def handler(request):
                    calls.append(request)
                    if len(calls) == 2 and second_status == "unknown":
                        raise httpx.ReadTimeout("synthetic timeout", request=request)
                    if len(calls) == 2 and second_status == "failed":
                        return httpx.Response(503, json={"code": "SyntheticUnavailable"})
                    return httpx.Response(200, json={"output": {"choices": [{"finish_reason": "stop",
                        "message": {"content": [{"text": text}]}}]}})
                settings = OCRSettings(True, "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                                       "synthetic-ocr-key", 15)
                client = VLOCRClient(root, settings, transport=httpx.MockTransport(handler))
                store = DocumentStore(root, "workspace", ocr_client=client)
                self.addCleanup(store.close)
                asset = store.add("fictional-partial.pdf", pdf)
                budget = len(text) + 1 if second_status == "document_text_budget" else 1_000_000
                with patch("rag_arch.web.ocr_preparation.MAX_DOCUMENT_OCR_CHARACTERS", budget),                      ocr_preparation_scope(Mock(), max_pages=2):
                    initial = store.load(asset["asset_id"])
                self.assertEqual(len(initial.units), 1)
                self.assertEqual(initial.ocr_pages[0]["status"], "completed")
                if second_status == "document_text_budget":
                    self.assertEqual(initial.ocr_pages[1]["reason"], "document_text_budget")
                    self.assertEqual(initial.ocr_pages[1]["status"], "failed")
                else:
                    self.assertEqual(initial.ocr_pages[1]["status"], second_status)
                self.assertEqual(len(calls), 2)
                with ocr_preparation_scope(Mock(), max_pages=2):
                    later = store.load(asset["asset_id"])
                self.assertIs(later, initial)
                self.assertEqual(len(calls), 2)
                store.close()

    def test_unattempted_page_is_first_requested_in_a_later_chat_task(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), TemporaryDirectory() as temp:
                root = Path(temp)
                calls = []
                store = DocumentStore(root, "workspace", ocr_client=ocr_client(root, calls))
                self.addCleanup(store.close)
                first = store.add("fictional-a.pdf", self.pdf)
                second = store.add("fictional-b.pdf", self.pdf)
                ids = [first["asset_id"], second["asset_id"]]
                def prepare():
                    if asynchronous:
                        return asyncio.run(store.aprepare_chat(ids))
                    return store.prepare_chat(ids)
                with ocr_preparation_scope(Mock(), max_pages=1):
                    initial = prepare()
                    self.assertEqual(len(calls), 1)
                    statuses = [d.ocr_pages[0]["status"] for d in initial.collection.documents]
                    self.assertEqual(set(statuses), {"completed", "not_attempted"})
                    # Re-entering in the same task cannot spend another request.
                    repeated = prepare()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual({d.ocr_pages[0]["status"] for d in repeated.collection.documents},
                                     {"completed", "not_attempted"})
                with ocr_preparation_scope(Mock(), max_pages=1):
                    later = prepare()
                self.assertEqual(len(calls), 2)
                self.assertEqual([d.ocr_pages[0]["status"] for d in later.collection.documents],
                                 ["completed", "completed"])
                self.assertEqual(len(later.collection.chunks), 2)
                store.close()

    def test_failed_and_unknown_markers_prevent_new_requests_in_later_task(self):
        for status in ("failed", "unknown"):
            for asynchronous in (False, True):
                with self.subTest(status=status, asynchronous=asynchronous), TemporaryDirectory() as temp:
                    root = Path(temp)
                    calls = []
                    store = DocumentStore(root, "workspace", ocr_client=ocr_client(root, calls, status))
                    self.addCleanup(store.close)
                    asset = store.add("fictional-marker.pdf", self.pdf)
                    def prepare():
                        if asynchronous:
                            return asyncio.run(store.aprepare_chat([asset["asset_id"]]))
                        return store.prepare_chat([asset["asset_id"]])
                    with ocr_preparation_scope(Mock()):
                        initial = prepare()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(initial.collection.documents[0].ocr_pages[0]["status"], status)
                    with ocr_preparation_scope(Mock()):
                        later = prepare()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(later.collection.documents[0].ocr_pages[0]["status"], status)
                    self.assertTrue(later.collection.documents[0].ocr_pages[0]["from_cache"])
                    store.close()

    def test_concurrent_no_text_loads_share_one_failed_ocr_request(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls, entered, release, waiting = [], Event(), Event(), Event()
            def handler(request):
                calls.append(request)
                entered.set()
                if not release.wait(10):
                    raise AssertionError("test did not release OCR request")
                return httpx.Response(503, json={"code": "SyntheticUnavailable"})
            settings = OCRSettings(True,
                "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "synthetic-ocr-key", 15)
            client = VLOCRClient(root, settings, transport=httpx.MockTransport(handler))
            store = DocumentStore(root, "workspace", ocr_client=client)
            self.addCleanup(store.close)
            asset = store.add("fictional-concurrent.pdf", self.pdf)
            def load():
                with ocr_preparation_scope(Mock()):
                    return store.load(asset["asset_id"])
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(load)
                try:
                    self.assertTrue(entered.wait(10), "first OCR call did not start")
                    with store._lock:
                        inflight = store._inflight[("loaded", asset["asset_id"])]
                    original_result = inflight.result
                    def watched_result(*args, **kwargs):
                        waiting.set()
                        return original_result(*args, **kwargs)
                    with patch.object(inflight, "result", side_effect=watched_result):
                        second = executor.submit(load)
                        self.assertTrue(waiting.wait(10), "second load did not join the in-flight build")
                        self.assertFalse(second.done())
                        release.set()
                        first_document = first.result(timeout=15)
                        second_document = second.result(timeout=15)
                finally:
                    release.set()
            self.assertIs(first_document, second_document)
            self.assertEqual(first_document.ocr_pages[0]["status"], "failed")
            self.assertEqual(len(calls), 1)
            with ocr_preparation_scope(Mock()):
                later = store.load(asset["asset_id"])
            self.assertTrue(later.ocr_pages[0]["from_cache"])
            self.assertEqual(len(calls), 1)
            store.close()

    def test_cancelled_shared_async_build_preserves_no_text_cache_policy(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls, entered, release = [], Event(), Event()
            store = DocumentStore(root, "workspace", ocr_client=ocr_client(root, calls, "failed"))
            self.addCleanup(store.close)
            asset = store.add("fictional-cancel.pdf", self.pdf)
            ids = [asset["asset_id"]]
            with ocr_preparation_scope(Mock()):
                initial = store.load(asset["asset_id"])
            self.assertEqual(initial.ocr_pages[0]["status"], "failed")
            self.assertEqual(len(calls), 1)
            from rag_arch.web import documents as document_module
            original_build = document_module.build_collection
            builds = []
            def controlled_build(*args, **kwargs):
                builds.append(1)
                if len(builds) == 1:
                    entered.set()
                    if not release.wait(15):
                        raise AssertionError("test did not release shared build")
                return original_build(*args, **kwargs)
            token = ("chat", (tuple(ids), False))
            async def scenario():
                with ocr_preparation_scope(Mock()):
                    owner = asyncio.create_task(store.aprepare_chat(ids))
                    self.assertTrue(await asyncio.to_thread(entered.wait, 15))
                    peer = asyncio.create_task(store.aprepare_chat(ids))
                    try:
                        async def wait_for_peer():
                            while True:
                                with store._lock:
                                    waiters = store._async_inflight_waiters.get(token, 0)
                                    builder = store._async_inflight_tasks.get(token)
                                if waiters == 2 and builder is not None:
                                    return builder
                                await asyncio.sleep(0.01)
                        builder = await asyncio.wait_for(wait_for_peer(), 15)
                        owner.cancel()
                        async def wait_for_cancellation():
                            while not builder.cancelling():
                                await asyncio.sleep(0.01)
                        await asyncio.wait_for(wait_for_cancellation(), 15)
                    finally:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                    return await asyncio.wait_for(peer, 15)
            with patch("rag_arch.web.documents.build_collection", side_effect=controlled_build):
                result = asyncio.run(scenario())
            self.assertEqual(len(builds), 2)
            self.assertEqual(result.collection.documents[0].ocr_pages[0]["status"], "failed")
            self.assertTrue(result.collection.documents[0].ocr_pages[0]["from_cache"])
            self.assertIsNone(store._cache_get(*token))
            self.assertEqual(len(calls), 1)
            with ocr_preparation_scope(Mock()):
                later = store.prepare_chat(ids)
            self.assertTrue(later.collection.documents[0].ocr_pages[0]["from_cache"])
            self.assertEqual(len(calls), 1)
            store.close()

    def test_unattempted_page_is_first_requested_in_a_later_plan_task(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            store = DocumentStore(root, "workspace", ocr_client=ocr_client(root, calls))
            self.addCleanup(store.close)
            first = store.add("fictional-first.pdf", self.pdf)
            second = store.add("fictional-plan.pdf", self.pdf)
            with ocr_preparation_scope(Mock(), max_pages=1):
                store.load(first["asset_id"])
                initial = store.get_plan(attachment_ids=[second["asset_id"]])
                self.assertEqual(initial.document.ocr_pages[0]["status"], "not_attempted")
                with self.assertRaises(OCRDocumentUnreadable):
                    _require_plan_text(initial.document, "review")
            self.assertEqual(len(calls), 1)
            with ocr_preparation_scope(Mock(), max_pages=1):
                later = store.get_plan(attachment_ids=[second["asset_id"]])
            self.assertEqual(len(calls), 2)
            self.assertTrue(later.document.units)
            self.assertEqual(later.document.ocr_pages[0]["status"], "completed")
            _require_plan_text(later.document, "review")
            store.close()

    def test_full_plan_review_keeps_safe_ocr_error_before_model(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), TemporaryDirectory() as temp:
                root = Path(temp)
                ocr = FakeOCR("failed")
                store = DocumentStore(root, "workspace", ocr_client=ocr)
                self.addCleanup(store.close)
                asset = store.add("fictional-plan.pdf", self.pdf)
                model_factory = Mock(side_effect=AssertionError("model must not be called"))
                resources = SimpleNamespace(plan_index=lambda: None, aplan_index=lambda: None,
                                            aprepare=lambda: None)
                request = {"task_id": "plan-scan", "plan_action": "review",
                           "attachment_ids": [asset["asset_id"]]}
                with ocr_preparation_scope(Mock()), self.assertRaises(OCRDocumentUnreadable) as failure:
                    if asynchronous:
                        asyncio.run(arun_plan(request, lambda *_: None, Event(), resources=resources,
                            documents=store, settings=self.settings, model_factory=model_factory))
                    else:
                        run_plan(request, lambda *_: None, Event(), resources=resources,
                            documents=store, settings=self.settings, model_factory=model_factory)
                error = failure.exception.as_error()
                self.assertEqual(error["code"], "ocr_reading_unavailable")
                self.assertIn("第 1 页 OCR 失败", error["message"])
                self.assertNotIn("synthetic_failure", json.dumps(error, ensure_ascii=False))
                model_factory.assert_not_called()
                store.close()


if __name__ == "__main__":
    unittest.main()
