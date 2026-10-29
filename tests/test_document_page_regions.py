"""Local and supervised renderers keep region identity and bounded caches."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_arch.document_pages import DocumentPageService
from rag_arch.pdf_backend import ProcessPdfBackend, _safe_worker_error
from tests.test_page_regions import coloured_pdf
from tests.test_document_pages import make_docx


class DocumentPageRegionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def service(self, **kwargs):
        service = DocumentPageService(self.root, **kwargs)
        service.register("plan", "plan.pdf", data=coloured_pdf(90, crop=True))
        return service

    def test_whole_and_different_regions_keep_separate_cache_records(self):
        service = self.service()
        whole = service.read_page("plan", 1)
        left = service.read_page("plan", 1, region=[0, 0, .5, .5])
        right = service.read_page("plan", 1, region=[.5, 0, 1, .5])
        self.assertEqual(len(service._page_cache), 3)
        self.assertEqual(len({row["evidence_id"] for row in (whole, left, right)}), 3)
        self.assertNotIn("LEFT", left["text"])
        self.assertIn("LEFT", right["text"])
        left["region"][0] = .2
        self.assertEqual(service.read_page("plan", 1, region=[0, 0, .5, .5])["region"][0], 0)
        with self.assertRaisesRegex(ValueError, "未登记"):
            service.read_page("other", 1, region=[0, 0, .5, .5])

    def test_encoded_image_cache_respects_byte_limit_and_reregistration_clears_weight(self):
        service = self.service()
        first = service.read_page("plan", 1, region=[0, 0, .5, .5])
        second = service.read_page("plan", 1, region=[.5, 0, 1, .5])
        budget = max(service._cached_page_weight(first), service._cached_page_weight(second)) + 64
        with patch("rag_arch.document_pages.MAX_PAGE_CACHE_BYTES", budget):
            service.read_page("plan", 1, region=[0, .5, .5, 1])
            self.assertLessEqual(service._page_cache_bytes, budget)
            self.assertEqual(service._page_cache_bytes,
                             sum(service._cached_page_weight(value) for value in service._page_cache.values()))
        service.register("plan", "replacement.pdf", data=coloured_pdf())
        self.assertEqual(service._page_cache_bytes, 0)
        self.assertEqual(len(service._page_cache), 0)

    def test_supervised_worker_accepts_regions_and_reuses_one_process(self):
        backend = ProcessPdfBackend(call_timeout=10)
        self.addCleanup(backend.close)
        service = self.service(pdf_backend=backend)
        detail = service.read_page("plan", 1, region=[.5, 0, 1, .5])
        self.assertIn("LEFT 7 kN", detail["text"])
        self.assertEqual(detail["rotation"], 90)
        self.assertEqual(detail["page_width"], 240)
        self.assertEqual(detail["page_height"], 300)
        self.assertEqual(detail["text_scope"], "region")
        self.assertEqual(len(backend.worker_pids), 1)
        self.assertEqual(detail, service.read_page("plan", 1, region=[.5, 0, 1, .5]))
        service.read_page("plan", 1)
        self.assertEqual(len(backend.worker_pids), 1)

    def test_supervised_tiny_region_keeps_domain_error_and_worker_remains_usable(self):
        backend = ProcessPdfBackend(call_timeout=10)
        self.addCleanup(backend.close)
        service = self.service(pdf_backend=backend)
        with self.assertRaisesRegex(ValueError, "区域过小，请选择更大的页面区域"):
            service.read_page("plan", 1, region=[0, 0, 1e-300, .5])
        self.assertEqual(len(service._page_cache), 0)
        self.assertEqual(len(backend.worker_pids), 1)
        self.assertIn("LEFT", service.read_page("plan", 1)["text"])
        self.assertEqual(len(backend.worker_pids), 1)

    def test_only_known_fixed_region_errors_are_preserved(self):
        for message in ("区域过小，请选择更大的页面区域。",
                        "页面图像过大或区域过小，无法在图像限制内读取。"):
            self.assertEqual(_safe_worker_error(ValueError(message)),
                             ("builtins", "ValueError", message))
        for error in (ValueError("private C:/not-for-output/file.pdf sensitive-value"),
                      RuntimeError("区域过小，请选择更大的页面区域。")):
            self.assertEqual(_safe_worker_error(error),
                             ("builtins", "RuntimeError", "PDF worker 处理失败。"))

    def test_substituted_backend_page_source_or_region_never_gets_requested_evidence_id(self):
        original = self.service().read_page("plan", 1)
        substitutions = [({**original, "page": 2}, "页码"),
                         ({**original, "page": True}, "页码"),
                         ({**original, "source_ref": "other"}, "来源"),
                         (original, "区域"),
                         ({**original, "region": [0, 0, .6, .5]}, "区域")]
        for result, problem in substitutions:
            with self.subTest(problem=problem, result_page=result.get("page")):
                class SubstitutedBackend:
                    def validate(self, source):
                        return 1
                    def read_page(self, *args, **kwargs):
                        return dict(result)
                service = self.service(pdf_backend=SubstitutedBackend())
                with self.assertRaisesRegex(ValueError, f"返回{problem}与请求不一致"):
                    service.read_page("plan", 1, region=[0, 0, .5, .5])
                self.assertEqual(len(service._page_cache), 0)
                self.assertEqual(service._page_cache_bytes, 0)

    def test_legacy_backend_missing_region_remains_compatible_for_whole_pages(self):
        original = self.service().read_page("plan", 1)
        original.pop("region")
        class LegacyBackend:
            def validate(self, source):
                return 1
            def read_page(self, *args):
                return dict(original)
        service = self.service(pdf_backend=LegacyBackend())
        result = service.read_page("plan", 1)
        self.assertEqual(result["evidence_id"], "PAGE:plan:1")
        self.assertIn("LEFT", result["text"])
        self.assertEqual(result, service.read_page("plan", 1))

    def test_docx_regions_share_the_original_system_rendered_pdf(self):
        renders = []
        def renderer(source, output):
            renders.append(source)
            output.write_bytes(coloured_pdf())
        service = DocumentPageService(self.root, docx_renderer=renderer)
        service.register("word", "plan.docx", data=make_docx())
        detail = service.read_page("word", 1, region=[0, 0, .5, .5])
        whole = service.read_page("word", 1)
        self.assertEqual(len(renders), 1)
        self.assertEqual(detail["page_numbering"], "system_rendered_page")
        self.assertEqual(whole["page_numbering"], "system_rendered_page")
        self.assertIn("LEFT 7 kN", detail["text"])
        self.assertNotIn("RIGHT", detail["text"])


if __name__ == "__main__":
    unittest.main()
