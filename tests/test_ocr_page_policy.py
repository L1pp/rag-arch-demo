"""OCR page selection is local, conservative and resource-supervised."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from docx import Document
from PIL import Image, ImageDraw
import pymupdf

from rag_arch.document_pages import DocumentPageService
from rag_arch.ocr_page_policy import build_ocr_page_candidates, ocr_page_candidate
from rag_arch.pdf_backend import ProcessPdfBackend


def _image() -> bytes:
    """A deterministic simulated scan, not an OCR request."""
    image = Image.new("RGB", (300, 400), "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, 370, 20):
        draw.rectangle((25, y, 275, y + 3), fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _synthetic_pages() -> bytes:
    scan = _image()
    blank_image = BytesIO()
    Image.new("RGB", (300, 400), "white").save(blank_image, format="PNG")
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=400)
        for y in range(70, 250, 15):
            page.insert_text((30, y), "ordinary native text and numerical data 120 mm")

        page = pdf.new_page(width=300, height=400)
        page.insert_text((90, 180), "Short title")

        pdf.new_page(width=300, height=400)  # genuinely empty page

        page = pdf.new_page(width=300, height=400)
        page.insert_image(pymupdf.Rect(20, 45, 280, 355), stream=scan)

        page = pdf.new_page(width=300, height=400)
        page.insert_text((30, 22), "Header and page 5")
        page.insert_image(pymupdf.Rect(20, 45, 280, 355), stream=scan)

        page = pdf.new_page(width=300, height=400)
        for y in range(70, 125, 12):
            page.insert_text((30, y), "body text outside the embedded scan")
        # The source image retains its aspect ratio when inserted.  This area
        # remains larger than 20% of the page after that fit.
        page.insert_image(pymupdf.Rect(25, 140, 275, 335), stream=scan)

        page = pdf.new_page(width=300, height=400)
        page.insert_text((30, 180), "Body with a small logo")
        page.insert_image(pymupdf.Rect(20, 15, 55, 40), stream=scan)

        page = pdf.new_page(width=300, height=400)
        page.insert_image(pymupdf.Rect(20, 45, 280, 355), stream=scan)
        for y in range(80, 260, 15):
            page.insert_text((30, y), "existing readable text layer on scan")

        page = pdf.new_page(width=300, height=400)
        for y in range(80, 260, 30):
            page.draw_rect(pymupdf.Rect(40, y, 260, y + 20), color=(0, 0, 0))

        page = pdf.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=blank_image.getvalue())

        page = pdf.new_page(width=300, height=400)
        page.draw_line((35, 200), (265, 200), color=(0, 0, 0), width=1)
        return pdf.tobytes()


def _docx() -> bytes:
    stream = BytesIO()
    document = Document()
    document.add_paragraph("text before the simulated image page")
    document.save(stream)
    return stream.getvalue()


class OcrPagePolicyTests(unittest.TestCase):
    def setUp(self):
        self.pdf = _synthetic_pages()

    def test_local_page_policy_covers_text_scans_mixed_and_blank(self):
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            rows = build_ocr_page_candidates(document)

        self.assertEqual([row["page"] for row in rows], list(range(1, 12)))
        self.assertEqual([row["needs_ocr"] for row in rows],
                         [False, False, False, True, True, True, False, False, True, False, False])
        self.assertEqual(rows[1]["reason"], "native_text")  # short title
        self.assertEqual(rows[2]["reason"], "blank")
        self.assertTrue(rows[2]["blank"])
        self.assertEqual(rows[3]["reason"], "image_dominant_sparse_body")
        self.assertGreater(rows[4]["native_chars"], 0)  # header alone
        self.assertEqual(rows[4]["body_chars"], 0)
        self.assertEqual(rows[4]["reason"], "image_dominant_sparse_body")
        self.assertEqual(rows[5]["reason"], "image_region_sparse_text")
        self.assertLess(rows[6]["image_coverage"], 0.02)  # small logo
        self.assertGreater(rows[7]["body_chars"], 80)  # scan with text layer
        self.assertEqual(rows[8]["reason"], "vector_content_without_text")
        self.assertTrue(rows[9]["blank"])
        self.assertEqual(rows[9]["reason"], "visually_blank_image")
        self.assertFalse(rows[10]["blank"])
        self.assertEqual(rows[10]["reason"], "minor_drawing_without_text")

    def test_bad_text_layer_and_analysis_failure_are_not_silently_complete(self):
        page = Mock()
        page.rect = (0, 0, 300, 400)
        page.get_text.side_effect = lambda kind, **_: (
            "bad\ufffd\ufffd\ufffd\ufffd" if kind == "text"
            else [(20, 100, 100, 110, "bad\ufffd\ufffd\ufffd\ufffd", 0, 0, 0)])
        page.get_image_info.return_value = []
        row = ocr_page_candidate(page, 1)
        self.assertTrue(row["needs_ocr"])
        self.assertEqual(row["reason"], "unreadable_text_layer")
        page.get_image_info.assert_called_once_with(hashes=False)

        page.get_text.side_effect = RuntimeError("unreadable page")
        failed = ocr_page_candidate(page, 2)
        self.assertFalse(failed["needs_ocr"])
        self.assertFalse(failed["blank"])
        self.assertEqual(failed["reason"], "page_analysis_failed")
        for number in (0, -1, True, "1"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                ocr_page_candidate(page, number)

    def test_watermark_in_body_does_not_make_scanned_page_look_readable(self):
        page = Mock()
        page.rect = (0, 0, 300, 400)
        page.get_text.side_effect = lambda kind, **_: (
            "仅供内部参考\nPage 5\n" if kind == "text"
            else [(20, 10, 70, 20, "Page", 0, 0, 0),
                  (72, 10, 80, 20, "5", 0, 0, 1),
                  (70, 190, 230, 210, "仅供内部参考", 1, 0, 0)])
        page.get_image_info.return_value = [{"bbox": (0, 0, 300, 400)}]
        row = ocr_page_candidate(page, 5)
        self.assertGreater(row["native_chars"], 0)
        self.assertEqual(row["body_chars"], 0)
        self.assertTrue(row["needs_ocr"])
        self.assertEqual(row["reason"], "image_dominant_sparse_body")

    def test_legacy_image_info_does_not_retry_with_unknown_hash_defaults(self):
        page = Mock()
        page.rect = (0, 0, 300, 400)
        page.rotation = 0
        page.get_text.side_effect = lambda kind, **_: (
            "native text" if kind == "text"
            else [(20, 100, 90, 110, "native", 0, 0, 0)])
        page.get_image_info.side_effect = TypeError("hashes option not supported")
        row = ocr_page_candidate(page, 1)
        page.get_image_info.assert_called_once_with(hashes=False)
        self.assertEqual(row["reason"], "page_analysis_failed")
        self.assertFalse(row["needs_ocr"])
        self.assertFalse(row["blank"])

    def test_rotated_pdf_uses_matching_unrotated_geometry(self):
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            for index in (3, 5, 7):
                page = document[index]
                unrotated = ocr_page_candidate(page, index + 1)
                for rotation in (90, 180, 270):
                    with self.subTest(page=index + 1, rotation=rotation):
                        page.set_rotation(rotation)
                        rotated = ocr_page_candidate(page, index + 1)
                        self.assertEqual(rotated["needs_ocr"], unrotated["needs_ocr"])
                        self.assertEqual(rotated["reason"], unrotated["reason"])
                        self.assertEqual(rotated["native_chars"], unrotated["native_chars"])
                        self.assertEqual(rotated["body_chars"], unrotated["body_chars"])
                        self.assertAlmostEqual(rotated["image_coverage"], unrotated["image_coverage"], places=5)

    def test_overlaid_images_do_not_inflate_page_coverage(self):
        page = Mock()
        page.rect = (0, 0, 300, 400)
        page.rotation = 0
        page.get_text.side_effect = lambda kind, **_: (
            "native readable text" if kind == "text"
            else [(10, 10, 100, 20, "native readable text", 0, 0, 0)])
        page.get_image_info.return_value = [
            {"bbox": (0, 100, 300, 200)}, {"bbox": (0, 100, 300, 200)}]
        row = ocr_page_candidate(page, 1)
        self.assertAlmostEqual(row["image_coverage"], 0.25)
        self.assertEqual(row["reason"], "image_region_sparse_text")

    def test_candidates_use_the_existing_supervised_pdf_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = ProcessPdfBackend(temp_root=temporary, call_timeout=20)
            self.addCleanup(backend.close)
            rows = backend.ocr_candidates(self.pdf)
            self.assertEqual(len(rows), 11)
            self.assertEqual([r["page"] for r in rows if r["needs_ocr"]], [4, 5, 6, 9])
            self.assertEqual(len(backend.worker_pids), 1)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            backend.close()

    def test_word_uses_cached_rendered_pdf_for_detection_and_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered = []

            def renderer(source, output):
                rendered.append((source, output))
                output.write_bytes(self.pdf)

            service = DocumentPageService(root, docx_renderer=renderer)
            service.register("word", "plan.docx", data=_docx())
            rows = service.ocr_candidates("word")
            self.assertEqual([r["page"] for r in rows if r["needs_ocr"]], [4, 5, 6, 9])
            page = service.read_page("word", 4)
            self.assertTrue(page["image_bytes"].startswith(b"\x89PNG"))
            self.assertEqual(page["page_numbering"], "system_rendered_page")
            self.assertEqual(len(rendered), 1)
            self.assertEqual(len(service.ocr_candidates("word")), 11)
            self.assertEqual(len(rendered), 1)

    def test_legacy_doc_uses_normalized_word_and_cached_pdf_for_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized, rendered = [], []

            class Office:
                def ensure_docx(self, source, target):
                    normalized.append((source, target))
                    Path(target).write_bytes(_docx())

                def render_pdf(self, source, target):
                    rendered.append((source, target))
                    Path(target).write_bytes(self_pdf)

            self_pdf = self.pdf
            office = Office()
            service = DocumentPageService(root, office_converter=office,
                                          docx_renderer=office.render_pdf)
            legacy = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1synthetic-doc"
            service.register("legacy", "plan.doc", data=legacy)
            rows = service.ocr_candidates("legacy")
            self.assertEqual([r["page"] for r in rows if r["needs_ocr"]], [4, 5, 6, 9])
            self.assertEqual(service.read_page("legacy", 4)["page"], 4)
            self.assertEqual(len(normalized), 1)
            self.assertEqual(len(rendered), 1)
            self.assertEqual(rendered[0][0].suffix, ".docx")

    def test_invalid_worker_rows_are_rejected_before_ocr_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = Mock()
            backend.validate.return_value = 11
            backend.ocr_candidates.return_value = [{"page": 1, "needs_ocr": True}]
            service = DocumentPageService(Path(temporary), pdf_backend=backend)
            service.register("bad", "example.pdf", data=self.pdf)
            with self.assertRaisesRegex(ValueError, "候选页检测结果无效"):
                service.ocr_candidates("bad")


if __name__ == "__main__":
    unittest.main()
