from concurrent.futures import CancelledError, ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from rag_arch.cloud_ocr import MAX_OCR_TEXT_CHARS, MODEL, OCRSettings, VLOCRClient
from rag_arch.request_control import request_control_scope


def page_png(width=80, height=100):
    target = BytesIO()
    Image.new("RGB", (width, height), "white").save(target, format="PNG")
    return target.getvalue()


class CloudOcrTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "data/runtime/web_uploads/7/workspace/asset/ocr"
        self.image = page_png()
        self.key = "synthetic-local-ocr-key"
        self.settings = OCRSettings(True, "https://ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", self.key, 15)
        self.calls = []

    def client(self, handler):
        return VLOCRClient(self.root, settings=self.settings, transport=httpx.MockTransport(handler))

    @staticmethod
    def data(text="\u6807\u9ad8 1.5m", finish="stop"):
        return {"request_id": "request-123", "output": {"choices": [{
            "finish_reason": finish, "message": {"content": [{"text": text}]}}]},
                "usage": {"input_tokens": 1000, "output_tokens": 500,
                          "input_tokens_details": {"image_tokens": 900, "text_tokens": 100}}}

    def good_handler(self, request):
        self.calls.append(request)
        return httpx.Response(200, json=self.data())

    def test_disabled_by_default_and_never_uses_tokenplan_credential(self):
        (self.root / ".env").write_text(
            "RAG_CHAT_BASE_URL=https://ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n"
            "RAG_CHAT_API_KEY=normal-test-key\nRAG_TOKENPLAN_API_KEY=token-plan-only\n"
            "RAG_TOKENPLAN_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n",
            encoding="utf-8")
        disabled = OCRSettings.from_root(self.root, environ={})
        self.assertFalse(disabled.enabled)
        settings = OCRSettings.from_root(self.root, environ={"RAG_ARCH_OCR_ENABLED": "true"})
        self.assertEqual(settings.api_key, "normal-test-key")
        self.assertNotIn("normal-test-key", repr(settings))
        self.assertNotIn("token-plan-only", repr(settings))
        (self.root / ".env").write_text(
            "RAG_CHAT_BASE_URL=https://ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n"
            "RAG_TOKENPLAN_API_KEY=token-plan-only\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "密钥"):
            OCRSettings.from_root(self.root, environ={"RAG_ARCH_OCR_ENABLED": "true"})

    def test_endpoint_rejects_tokenplan_foreign_host_and_credentials_in_url(self):
        for base in ("https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                     "http://ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                     "https://ws-test123.cn-beijing.maas.aliyuncs.com.attacker.invalid/compatible-mode/v1",
                     "https://synthetic-user:synthetic-password@ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                     "https://ws-test123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1?token=secret"):
            with self.subTest(base=base):
                client = VLOCRClient(self.root, OCRSettings(True, base, self.key, 10), httpx.MockTransport(self.good_handler))
                with self.assertRaises(ValueError):
                    client.recognize_page(self.image, page=1, cache_dir=self.cache)
        self.assertEqual(self.calls, [])

    def test_success_cache_redacts_provider_secrets_and_is_reused(self):
        signed = "https://docmind-api-cn-hangzhou.oss-cn-hangzhou.aliyuncs.com/a.png?Signature=hidden&security-token=private"
        text = "0.104mm \u2264 1.5mm " + signed + " data:image/png;base64,QUJDRA== " + self.key

        def handler(request):
            self.calls.append(request)
            body = json.loads(request.content)
            self.assertEqual(request.url.path, "/api/v1/services/aigc/multimodal-generation/generation")
            self.assertEqual(request.headers["authorization"], "Bearer " + self.key)
            self.assertEqual(request.headers["accept-encoding"], "identity")
            self.assertEqual(body["model"], MODEL)
            self.assertEqual(body["parameters"]["ocr_options"]["task"], "document_parsing")
            self.assertEqual(body["parameters"]["max_tokens"], 8192)
            self.assertTrue(body["input"]["messages"][0]["content"][0]["image"].startswith("data:image/png;base64,"))
            return httpx.Response(200, json=self.data(text))

        client = self.client(handler)
        first = client.recognize_page(self.image, page=1, cache_dir=self.cache)
        second = client.recognize_page(self.image, page=1, cache_dir=self.cache)
        self.assertEqual((first.status, second.status), ("completed", "completed"))
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertFalse(first.recognition_verified)
        self.assertEqual(first.usage, {"input_tokens": 1000, "output_tokens": 500})
        self.assertEqual(first.cost_cny, "0.00055")
        self.assertEqual(len(self.calls), 1)
        self.assertIn("0.104mm", first.text)
        for secret in ("hidden", "private", self.key, "data:image/png;base64"):
            self.assertNotIn(secret, first.text)
            self.assertNotIn(secret, (self.cache / "page-000001.json").read_text(encoding="utf-8"))
        self.assertEqual(second.text, first.text)
        self.assertEqual(second.to_dict()["recognition_verified"], False)
        json.dumps(second.to_dict(), ensure_ascii=False)

    def test_generic_key_query_parameter_is_redacted_but_ordinary_url_is_preserved(self):
        text = ("https://example.org/page?key=private-value&section=1 "
                "https://example.org/figure?section=1&amp;%4B%45%59=private-other "
                "https://example.org/reference?section=3")
        client = self.client(lambda request: httpx.Response(200, json=self.data(text)))
        result = client.recognize_page(self.image, page=17, cache_dir=self.cache)
        self.assertEqual(result.status, "completed")
        self.assertNotIn("private-value", result.text)
        self.assertNotIn("private-other", result.text)
        self.assertIn("https://example.org/reference?section=3", result.text)
        stored = (self.cache / "page-000017.json").read_text(encoding="utf-8")
        self.assertNotIn("private-value", stored)
        self.assertNotIn("private-other", stored)

    def test_oversized_ocr_text_fails_without_discarding_usage_or_retrying(self):
        def handler(request):
            self.calls.append(request)
            return httpx.Response(200, json=self.data("长" * (MAX_OCR_TEXT_CHARS + 1)))
        client = self.client(handler)
        first = client.recognize_page(self.image, page=18, cache_dir=self.cache)
        self.assertEqual(first.status, "failed")
        self.assertEqual(first.error_code, "ocr_text_too_large")
        self.assertEqual(first.text, "")
        self.assertEqual(first.usage, {"input_tokens": 1000, "output_tokens": 500})
        self.assertEqual(first.cost_cny, "0.00055")
        marker = self.cache / "page-000018.json"
        self.assertLess(marker.stat().st_size, 2000)
        self.assertNotIn("长长", marker.read_text(encoding="utf-8"))
        again = client.recognize_page(self.image, page=18, cache_dir=self.cache)
        self.assertEqual(again.status, "failed")
        self.assertTrue(again.from_cache)
        self.assertEqual(again.usage, first.usage)
        self.assertEqual(len(self.calls), 1)

    def test_text_at_limit_is_accepted(self):
        client = self.client(lambda request: httpx.Response(200, json=self.data("文" * MAX_OCR_TEXT_CHARS)))
        result = client.recognize_page(self.image, page=19, cache_dir=self.cache)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.text), MAX_OCR_TEXT_CHARS)

    def test_old_oversized_cache_is_compacted_without_a_request(self):
        self.cache.mkdir(parents=True)
        marker = self.cache / "page-000020.json"
        old = {"schema": "rag-arch-vl-ocr-page-v1", "status": "completed", "page": 20,
               "model": MODEL, "text": "旧" * (MAX_OCR_TEXT_CHARS + 1),
               "request_id": "old-request", "usage": {"input_tokens": 1000, "output_tokens": 500},
               "cost_cny": "0.00055", "warnings": ["old warning"],
               "recognition_verified": False, "error_code": None}
        marker.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        os.chmod(marker, 0o600)
        client = self.client(self.good_handler)
        result = client.recognize_page(self.image, page=20, cache_dir=self.cache)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "ocr_text_too_large")
        self.assertEqual(result.text, "")
        self.assertEqual(result.usage, old["usage"])
        self.assertEqual(result.cost_cny, old["cost_cny"])
        self.assertEqual(len(self.calls), 0)
        saved = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(saved["text"], "")
        self.assertEqual(saved["status"], "failed")
        self.assertLess(marker.stat().st_size, 2000)

    def test_old_cache_query_keys_are_cleaned_without_a_request(self):
        self.cache.mkdir(parents=True)
        marker = self.cache / "page-000021.json"
        old = {"schema": "rag-arch-vl-ocr-page-v1", "status": "completed", "page": 21,
               "model": MODEL, "text": "原文 https://example.org/image?key=old-private",
               "request_id": "old-request", "usage": {"input_tokens": 1000, "output_tokens": 500},
               "cost_cny": "0.00055", "warnings": ["旧页需核对原文"],
               "recognition_verified": False, "error_code": None}
        marker.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        os.chmod(marker, 0o600)
        result = self.client(self.good_handler).recognize_page(self.image, page=21, cache_dir=self.cache)
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.from_cache)
        self.assertNotIn("old-private", result.text)
        self.assertEqual(result.usage, old["usage"])
        self.assertEqual(result.cost_cny, old["cost_cny"])
        self.assertNotIn("old-private", marker.read_text(encoding="utf-8"))
        self.assertEqual(len(self.calls), 0)

    def test_timeout_is_unknown_and_never_retried(self):
        def timeout(request):
            self.calls.append(request)
            raise httpx.ReadTimeout("should never expose " + self.key, request=request)
        first = self.client(timeout).recognize_page(self.image, page=2, cache_dir=self.cache)
        second = self.client(self.good_handler).recognize_page(self.image, page=2, cache_dir=self.cache)
        self.assertEqual((first.status, second.status), ("unknown", "unknown"))
        self.assertIsNone(first.cost_cny)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(self.key, (self.cache / "page-000002.json").read_text(encoding="utf-8"))

    def test_wall_deadline_preserves_unknown_and_never_resends(self):
        client = self.client(self.good_handler)
        with patch("rag_arch.cloud_ocr.monotonic", side_effect=(0.0, 16.0)):
            first = client.recognize_page(self.image, page=11, cache_dir=self.cache)
        self.assertEqual(first.status, "unknown")
        self.assertEqual(len(self.calls), 1)
        again = client.recognize_page(self.image, page=11, cache_dir=self.cache)
        self.assertEqual(again.status, "unknown")
        self.assertTrue(again.from_cache)
        self.assertEqual(len(self.calls), 1)

    def test_malformed_cache_fails_closed(self):
        self.cache.mkdir(parents=True)
        (self.cache / "page-000012.json").write_text("{incomplete", encoding="utf-8")
        client = self.client(self.good_handler)
        result = client.recognize_page(self.image, page=12, cache_dir=self.cache)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.error_code, "cache_unreadable")
        self.assertEqual(len(self.calls), 0)

    def test_success_without_usage_does_not_invent_cost(self):
        def handler(request):
            data = self.data()
            del data["usage"]
            return httpx.Response(200, json=data)
        result = self.client(handler).recognize_page(self.image, page=13, cache_dir=self.cache)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage, {})
        self.assertIsNone(result.cost_cny)

    def test_unexpected_compression_or_oversized_body_stays_unknown(self):
        def compressed(request):
            self.calls.append(request)
            return httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"not decoded")
        first = self.client(compressed).recognize_page(self.image, page=15, cache_dir=self.cache)
        self.assertEqual(first.status, "unknown")
        self.assertEqual(len(self.calls), 1)
        again = self.client(self.good_handler).recognize_page(self.image, page=15, cache_dir=self.cache)
        self.assertEqual(again.status, "unknown")
        self.assertEqual(len(self.calls), 1)
        def oversized(request):
            self.calls.append(request)
            return httpx.Response(200, content=b"x" * (4 * 1024 * 1024 + 1))
        value = self.client(oversized).recognize_page(self.image, page=16, cache_dir=self.cache)
        self.assertEqual(value.status, "unknown")
        self.assertEqual(len(self.calls), 2)

    def test_failed_and_truncated_responses_do_not_become_complete_or_repeat(self):
        def handler(request):
            self.calls.append(request)
            return httpx.Response(403, json={"code": "AllocationQuota.FreeTierOnly", "message": self.key})
        client = self.client(handler)
        first = client.recognize_page(self.image, page=3, cache_dir=self.cache)
        second = client.recognize_page(self.image, page=3, cache_dir=self.cache)
        self.assertEqual((first.status, second.status), ("failed", "failed"))
        self.assertEqual(first.error_code, "AllocationQuota.FreeTierOnly")
        self.assertNotIn(self.key, (self.cache / "page-000003.json").read_text(encoding="utf-8"))
        self.assertEqual(len(self.calls), 1)
        truncated = self.client(lambda request: httpx.Response(200, json=self.data("partial", "length")))
        value = truncated.recognize_page(self.image, page=4, cache_dir=self.cache)
        self.assertEqual(value.status, "failed")
        self.assertEqual(value.text, "")
        self.assertEqual(value.error_code, "truncated_or_empty")
        self.assertEqual(value.usage, {"input_tokens": 1000, "output_tokens": 500})

    def test_invalid_page_image_and_cache_paths_cannot_send(self):
        client = self.client(self.good_handler)
        for page in (True, 0, -1, 1.2, 1_000_001):
            with self.subTest(page=page), self.assertRaises(ValueError):
                client.recognize_page(self.image, page=page, cache_dir=self.cache)
        with self.assertRaises(ValueError):
            client.recognize_page(b"not a PNG", page=1, cache_dir=self.cache)
        with self.assertRaises(ValueError):
            client.recognize_page(page_png(2001, 2001), page=1, cache_dir=self.cache)
        with self.assertRaises(ValueError):
            client.recognize_page(self.image, page=1, cache_dir=self.root / "outside")
        self.assertEqual(len(self.calls), 0)

    def test_symlinked_cache_path_rejected(self):
        parent = self.root / "data/runtime/web_uploads/7/workspace/asset"
        parent.mkdir(parents=True)
        outside = self.root / "other"
        outside.mkdir()
        try:
            (parent / "ocr").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest("creating directory symlinks unavailable: " + type(exc).__name__)
        client = self.client(self.good_handler)
        with self.assertRaises(ValueError):
            client.recognize_page(self.image, page=1, cache_dir=parent / "ocr")
        self.assertEqual(len(self.calls), 0)

    def test_guard_before_request_prevents_network_and_unneeded_marker(self):
        client = self.client(self.good_handler)
        with self.assertRaises(CancelledError):
            client.recognize_page(self.image, page=5, cache_dir=self.cache,
                                  before_request=lambda: (_ for _ in ()).throw(CancelledError()))
        self.assertFalse((self.cache / "page-000005.json").exists())
        self.assertEqual(len(self.calls), 0)

    def test_cancel_after_marker_before_send_does_not_poison_future_request(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 2:
                raise CancelledError()
        client = self.client(self.good_handler)
        with request_control_scope(guard), self.assertRaises(CancelledError):
            client.recognize_page(self.image, page=6, cache_dir=self.cache)
        self.assertFalse((self.cache / "page-000006.json").exists())
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(client.recognize_page(self.image, page=6, cache_dir=self.cache).status, "completed")
        self.assertEqual(len(self.calls), 1)

    def test_cancel_propagates_when_unstarted_marker_cannot_be_removed(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 2:
                raise CancelledError()
        client = self.client(self.good_handler)
        with request_control_scope(guard), patch("rag_arch.cloud_ocr._clear_unstarted", side_effect=OSError()), self.assertRaises(CancelledError):
            client.recognize_page(self.image, page=14, cache_dir=self.cache)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(client.recognize_page(self.image, page=14, cache_dir=self.cache).status, "unknown")
        self.assertEqual(len(self.calls), 0)

    def test_cancel_during_response_is_propagated_and_recorded_unknown(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 3:
                raise CancelledError()
        client = self.client(self.good_handler)
        with request_control_scope(guard), self.assertRaises(CancelledError):
            client.recognize_page(self.image, page=7, cache_dir=self.cache)
        record = json.loads((self.cache / "page-000007.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "unknown")
        again = client.recognize_page(self.image, page=7, cache_dir=self.cache)
        self.assertEqual(again.status, "unknown")
        self.assertEqual(len(self.calls), 1)

    def test_control_valueerror_during_response_is_not_swallowed(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 3:
                raise ValueError("control failure")
        with request_control_scope(guard), self.assertRaisesRegex(ValueError, "control failure"):
            self.client(self.good_handler).recognize_page(self.image, page=8, cache_dir=self.cache)
        record = json.loads((self.cache / "page-000008.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "unknown")

    def test_cancel_after_success_keeps_completed_cache(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 4:
                raise CancelledError()
        client = self.client(self.good_handler)
        with request_control_scope(guard), self.assertRaises(CancelledError):
            client.recognize_page(self.image, page=9, cache_dir=self.cache)
        record = json.loads((self.cache / "page-000009.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "completed")
        again = client.recognize_page(self.image, page=9, cache_dir=self.cache)
        self.assertTrue(again.from_cache)
        self.assertEqual(again.status, "completed")
        self.assertEqual(len(self.calls), 1)

    def test_concurrent_same_page_only_starts_one_request(self):
        started, release = threading.Event(), threading.Event()
        def slow_handler(request):
            self.calls.append(request)
            started.set()
            if not release.wait(3):
                raise RuntimeError("fixture timeout")
            return httpx.Response(200, json=self.data())
        client = self.client(slow_handler)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(client.recognize_page, self.image, page=10, cache_dir=self.cache)
            self.assertTrue(started.wait(2))
            second = pool.submit(client.recognize_page, self.image, page=10, cache_dir=self.cache).result(timeout=2)
            self.assertEqual(second.status, "unknown")
            release.set()
            first = first_future.result(timeout=3)
        self.assertEqual(first.status, "completed")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(client.recognize_page(self.image, page=10, cache_dir=self.cache).from_cache)

    def test_cache_is_asset_local(self):
        client = self.client(self.good_handler)
        other = self.root / "data/runtime/web_uploads/8/workspace/another-asset/ocr"
        first = client.recognize_page(self.image, page=1, cache_dir=self.cache)
        second = client.recognize_page(self.image, page=1, cache_dir=other)
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "completed")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue((self.cache / "page-000001.json").exists())
        self.assertTrue((other / "page-000001.json").exists())


if __name__ == "__main__":
    unittest.main()
