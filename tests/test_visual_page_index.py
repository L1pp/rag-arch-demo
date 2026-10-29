from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf

from rag_arch.document_pages import DocumentPageService


def _bitmap(width=180, height=220, value=120):
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    image.clear_with(value)
    return image.tobytes("png")


def _visual_pdf():
    with pymupdf.open() as pdf:
        plain = pdf.new_page(width=300, height=400)
        plain.insert_text((20, 30), "A page of ordinary text")

        bitmap = pdf.new_page(width=300, height=400)
        bitmap.insert_image(pymupdf.Rect(20, 80, 280, 300), stream=_bitmap())
        bitmap.insert_text((20, 40), "Bitmap caption")

        vector = pdf.new_page(width=300, height=400)
        for x in (40, 100, 160, 220):
            vector.draw_line((x, 90), (x, 300), color=(0, 0, 0))
        for y in (90, 140, 190, 240, 300):
            vector.draw_line((40, y), (220, y), color=(0, 0, 0))
        vector.insert_text((40, 50), "Table 1")

        frame = pdf.new_page(width=300, height=400)
        frame.draw_rect(frame.rect, color=(0, 0, 0))
        frame.insert_image(pymupdf.Rect(10, 10, 30, 30), stream=_bitmap(20, 20, 200))
        frame.insert_text((40, 40), "Header logo and page frame")

        scan = pdf.new_page(width=300, height=400)
        scan.insert_image(scan.rect, stream=_bitmap(300, 400, 180))
        return pdf.tobytes()


def _docx_bytes():
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as package:
        package.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body/></w:document>",
        )
    return buffer.getvalue()


class VisualPageIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def test_metadata_candidates_are_conservative_and_short(self):
        service = DocumentPageService(self.root)
        service.register("pdf", "visual.pdf", data=_visual_pdf())

        with patch("pymupdf.Page.get_pixmap", side_effect=AssertionError("rendered")):
            listing = service.list_pages("pdf", visual_only=True)

        by_page = {row["page"]: row for row in listing["pages"]}
        self.assertEqual(list(by_page), [2, 3, 5])
        self.assertIn("image", by_page[2]["visual_signals"])
        self.assertIn("vector_or_table", by_page[3]["visual_signals"])
        # This fixture has an empty grid with only an outside caption.  It is
        # useful navigation, but must not become an automatic first-page image.
        self.assertFalse(by_page[3]["auto_attach_eligible"])
        self.assertTrue(all(not row["auto_attach_eligible"] for row in listing["pages"]))
        self.assertNotIn(4, by_page)
        self.assertIn("scanned_page", by_page[5]["visual_signals"])
        self.assertEqual(listing["visual_candidate_count"], 3)
        self.assertTrue(listing["visual_index_complete"])
        self.assertTrue(all(len(row["text_preview"]) <= 160 for row in listing["pages"]))
        self.assertTrue(all(set(row) >= {
            "page", "has_text", "text_preview", "visual_candidate", "visual_signals"
        } for row in listing["pages"]))

    def test_physical_pagination_query_and_metadata_cache(self):
        service = DocumentPageService(self.root)
        service.register("pdf", "visual.pdf", data=_visual_pdf())
        with patch("rag_arch.document_pages.build_visual_page_index",
                   wraps=__import__("rag_arch.visual_page_index", fromlist=["build_visual_page_index"])
                   .build_visual_page_index) as scan:
            first = service.list_pages("pdf", limit=1, visual_only=True)
            second = service.list_pages("pdf", start=first["next_start"],
                                        limit=1, visual_only=True)
            filtered = service.list_pages("pdf", query="table 1", visual_only=True)
        self.assertEqual(first["pages"][0]["page"], 2)
        self.assertEqual(first["next_start"], 3)
        self.assertEqual(second["pages"][0]["page"], 3)
        self.assertEqual(second["next_start"], 4)
        self.assertEqual([row["page"] for row in filtered["pages"]], [3])
        self.assertEqual(scan.call_count, 1)

    def test_docx_conversion_is_reused_for_visual_index_and_read(self):
        calls = []

        def renderer(source, output):
            calls.append((source, output))
            output.write_bytes(_visual_pdf())

        service = DocumentPageService(self.root, docx_renderer=renderer)
        service.register("word", "plan.docx", data=_docx_bytes())
        first = service.list_pages("word", visual_only=True)
        second = service.list_pages("word", visual_only=True)
        page = service.read_page("word", first["pages"][0]["page"])
        self.assertTrue(first["pages"])
        self.assertEqual(second["visual_candidate_count"], first["visual_candidate_count"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(page["image_bytes"].startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
