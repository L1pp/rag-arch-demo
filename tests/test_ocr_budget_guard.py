"""OCR resource bounds use self-authored pages and simulated transport only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import Mock, patch

import httpx

from rag_arch.cloud_ocr import OCRPageResult, OCRSettings, VLOCRClient
from rag_arch.models import LoadedDocument, TextUnit
from rag_arch.web.ocr_preparation import ocr_preparation_scope, supplement_document
from tests.test_web_ocr_preparation import Pages, candidate, png
from tests.test_web_task_worker import FakeRepository, _claim, _runtime


class CountingClient:
    def __init__(self, *, status="completed", text="示例文字", cached=None):
        self.status, self.text = status, text
        self.cached = dict(cached or {})
        self.requests = []

    def peek_cached_page(self, *, page, cache_dir):
        return self.cached.get(page)

    def recognize_page(self, image, *, page, cache_dir, before_request):
        before_request()
        self.requests.append(page)
        return OCRPageResult(self.status, page, text=self.text if self.status == "completed" else "")


class OCRBudgetGuardTests(unittest.TestCase):
    def loaded(self):
        return LoadedDocument("fictional", "fictional.pdf", "pdf",
                              [TextUnit("自编正文", {"page": 1})], [], [2, 3], 3)

    def test_task_page_cap_stops_later_rendering_and_requests(self):
        pages = Pages([candidate(page, True) for page in range(1, 5)])
        client = CountingClient()
        with ocr_preparation_scope(Mock(), max_pages=1):
            result = supplement_document(self.loaded(), pages, "source", client, Path("synthetic-cache"))
        self.assertEqual(pages.read, [1])
        self.assertEqual(client.requests, [1])
        self.assertEqual([row["status"] for row in result.ocr_pages],
                         ["completed", "not_attempted", "not_attempted", "not_attempted"])
        self.assertTrue(all(row.get("reason") == "task_page_budget" for row in result.ocr_pages[1:]))

    def test_blocked_or_text_exhausted_never_renders_following_pages(self):
        for status, maximum in (("unknown", 100), ("completed", 2)):
            with self.subTest(status=status):
                pages = Pages([candidate(page, True) for page in range(1, 4)])
                client = CountingClient(status=status, text="长于预算的自编识别文字")
                with patch("rag_arch.web.ocr_preparation.MAX_DOCUMENT_OCR_CHARACTERS", maximum), \
                     ocr_preparation_scope(Mock(), max_pages=5):
                    result = supplement_document(self.loaded(), pages, "source", client, Path("synthetic-cache"))
                self.assertEqual(pages.read, [1])
                self.assertEqual(client.requests, [1])
                self.assertEqual([r["status"] for r in result.ocr_pages[1:]], ["not_attempted"] * 2)
                if status == "completed":
                    self.assertTrue(result.ocr_pages[0]["request_completed"])
                    self.assertEqual(result.ocr_pages[0]["reason"], "document_text_budget")

    def test_cached_success_after_unconfirmed_request_is_reused_without_render(self):
        cached = OCRPageResult("completed", 2, text="自编缓存文字", from_cache=True)
        pages = Pages([candidate(page, True) for page in range(1, 4)])
        client = CountingClient(status="unknown", cached={2: cached})
        with ocr_preparation_scope(Mock(), max_pages=5):
            result = supplement_document(self.loaded(), pages, "source", client, Path("synthetic-cache"))
        self.assertEqual(pages.read, [1])
        self.assertEqual(client.requests, [1])
        self.assertEqual([r["status"] for r in result.ocr_pages],
                         ["unknown", "completed", "not_attempted"])
        self.assertTrue(result.ocr_pages[1]["from_cache"])

    def test_real_cache_peek_skips_render_and_keeps_budget_for_uncached_page(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"output": {"choices": [{"finish_reason": "stop",
                "message": {"content": [{"text": "自编识别文字"}]}}]}})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "data/runtime/web_uploads/7/workspace/asset/ocr"
            client = VLOCRClient(root, OCRSettings(True,
                "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "synthetic-key"),
                transport=httpx.MockTransport(handler))
            client.recognize_page(png(), page=1, cache_dir=cache)
            pages = Pages([candidate(1, True), candidate(2, True)])
            guard = Mock()
            with ocr_preparation_scope(guard, max_pages=1):
                result = supplement_document(self.loaded(), pages, "source", client, cache)
            self.assertEqual(pages.read, [2])
            self.assertEqual(len(requests), 2)
            self.assertTrue(result.ocr_pages[0]["from_cache"])
            self.assertFalse(result.ocr_pages[1]["from_cache"])
            guard.assert_called_once_with(2)

    def test_cached_failed_and_unknown_results_remain_no_retry(self):
        for failure in ("failed", "unknown"):
            with self.subTest(failure=failure), TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = root / "data/runtime/web_uploads/7/workspace/asset/ocr"
                requests = []
                def handler(request):
                    requests.append(request)
                    if failure == "unknown":
                        raise httpx.ReadTimeout("synthetic timeout", request=request)
                    return httpx.Response(403, json={"code": "AccessDenied"})
                client = VLOCRClient(root, OCRSettings(True,
                    "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "synthetic-key"),
                    transport=httpx.MockTransport(handler))
                client.recognize_page(png(), page=1, cache_dir=cache)
                pages = Pages([candidate(1, True), candidate(2, True)])
                with ocr_preparation_scope(Mock(), max_pages=1):
                    result = supplement_document(self.loaded(), pages, "source", client, cache)
                self.assertEqual(pages.read, [])
                self.assertEqual(len(requests), 1)
                self.assertEqual([r["status"] for r in result.ocr_pages], [failure, "not_attempted"])
                self.assertTrue(result.ocr_pages[0]["from_cache"])

    def test_unreadable_cache_fails_closed_without_render_or_request(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "data/runtime/web_uploads/7/workspace/asset/ocr"
            cache.mkdir(parents=True)
            (cache / "page-000001.json").write_text("not-json", encoding="utf-8")
            requests = []
            def handler(request):
                requests.append(request)
                return httpx.Response(200, json={})
            client = VLOCRClient(root, OCRSettings(True,
                "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "synthetic-key"),
                transport=httpx.MockTransport(handler))
            pages = Pages([candidate(1, True), candidate(2, True)])
            with ocr_preparation_scope(Mock(), max_pages=1):
                result = supplement_document(self.loaded(), pages, "source", client, cache)
            self.assertEqual(pages.read, [])
            self.assertEqual(requests, [])
            self.assertEqual([r["status"] for r in result.ocr_pages], ["unknown", "not_attempted"])
            self.assertEqual(result.ocr_pages[0]["error_code"], "cache_unreadable")

    def test_invalid_implicit_limit_waits_for_actual_candidate_and_never_renders(self):
        with patch.dict(os.environ, {"RAG_ARCH_OCR_MAX_PAGES_PER_TASK": "invalid"}):
            with ocr_preparation_scope(Mock()):
                pass
            pages = Pages([candidate(1, True)])
            client = CountingClient()
            with ocr_preparation_scope(Mock()), self.assertRaisesRegex(ValueError, "每任务页数"):
                supplement_document(self.loaded(), pages, "source", client, Path("synthetic-cache"))
        self.assertEqual(pages.read, [])
        self.assertEqual(client.requests, [])
        with self.assertRaisesRegex(ValueError, "每任务页数"), ocr_preparation_scope(Mock(), max_pages="invalid"):
            pass


class OCRDisabledConfigurationTests(unittest.TestCase):
    def test_disabled_timeout_ignored_but_enabled_timeout_remains_strict(self):
        with TemporaryDirectory() as temporary:
            disabled = OCRSettings.from_root(temporary, environ={"RAG_ARCH_OCR_ENABLED": "false",
                                                                  "RAG_ARCH_OCR_TIMEOUT_SECONDS": "invalid"})
            self.assertFalse(disabled.enabled)
            self.assertEqual(disabled.timeout_seconds, 120)
            with self.assertRaisesRegex(ValueError, "超时"):
                OCRSettings.from_root(temporary, environ={"RAG_ARCH_OCR_ENABLED": "true",
                                                         "RAG_ARCH_OCR_TIMEOUT_SECONDS": "invalid"})

    def test_simulation_ignores_disabled_bad_optional_values(self):
        repository = FakeRepository([_claim("simulation", simulation=True)])
        with TemporaryDirectory() as temporary, \
             patch.dict(os.environ, {"RAG_ARCH_OCR_ENABLED": "false",
                                     "RAG_ARCH_OCR_TIMEOUT_SECONDS": "invalid",
                                     "RAG_ARCH_OCR_MAX_PAGES_PER_TASK": "invalid"}), \
             patch("rag_arch.web.demo.run_demo", return_value={"text": "自编离线结果"}) as demo:
            _runtime(repository, simulation=True, root=temporary).run(max_tasks=1)
        self.assertEqual(repository.finishes[0][2], "completed")
        demo.assert_called_once()

    def test_disabled_bad_timeout_does_not_break_ordinary_store(self):
        resources = SimpleNamespace(pdf_backend=None, close=lambda: None)
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "RAG_ARCH_OCR_ENABLED": "false", "RAG_ARCH_OCR_TIMEOUT_SECONDS": "invalid"}):
            runtime = _runtime(FakeRepository(), root=temporary, database=object(), resources=resources)
            store, lease = runtime._acquire_document_store(_claim("ordinary")["request"])
            self.assertIsNone(store.ocr_client)
            lease.release()
            runtime._close_tenant_document_stores()

    def test_disabled_bad_optional_values_do_not_break_regular_claim(self):
        resources = SimpleNamespace(pdf_backend=None, close=lambda: None)
        repository = FakeRepository([_claim("ordinary")])
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "RAG_ARCH_OCR_ENABLED": "false", "RAG_ARCH_OCR_TIMEOUT_SECONDS": "invalid",
                "RAG_ARCH_OCR_MAX_PAGES_PER_TASK": "invalid"}), \
             patch("rag_arch.web.chat_runner.run_chat", return_value={"text": "自编结果"}) as chat:
            _runtime(repository, root=temporary, database=object(), settings=object(),
                     resources=resources).run(max_tasks=1)
        self.assertEqual(repository.finishes[0][2], "completed")
        self.assertIsNone(chat.call_args.kwargs["documents"].ocr_client)

    def test_enabled_store_created_without_attachment_remains_ocr_capable_when_reused(self):
        resources = SimpleNamespace(pdf_backend=None, close=lambda: None)
        values = {"RAG_ARCH_OCR_ENABLED": "true", "RAG_ARCH_OCR_TIMEOUT_SECONDS": "15",
                  "RAG_OCR_BASE_URL": "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                  "RAG_OCR_API_KEY": "synthetic-key"}
        with TemporaryDirectory() as temporary, patch.dict(os.environ, values):
            runtime = _runtime(FakeRepository(), root=temporary, database=object(), resources=resources)
            first, lease = runtime._acquire_document_store(_claim("first")["request"])
            lease.release()
            later, lease = runtime._acquire_document_store(_claim("later", attachment_ids=["fictional"])["request"])
            self.assertIs(first, later)
            self.assertIsInstance(later.ocr_client, VLOCRClient)
            lease.release()
            runtime._close_tenant_document_stores()


if __name__ == "__main__":
    unittest.main()
