from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pymupdf

from rag_arch.collection import build_collection
from rag_arch.loaders import load_document
from rag_arch.pdf_backend import ProcessPdfBackend
from rag_arch.plan_documents import load_uploaded_document


def image_only_pdf() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(180)
        page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        return document.tobytes()


def mixed_pdf() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(180)
        page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        page = document.new_page(width=300, height=400)
        page.insert_text((20, 30), "钢梁 PRIVATE_UPLOAD_123")
        return document.tobytes()


def table_docx() -> bytes:
    body = ("<w:p><w:r><w:t>前言</w:t></w:r></w:p>"
            "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>项目</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>值</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
            "<w:p><w:r><w:t>结论</w:t></w:r></w:p>")
    document = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body}</w:body></w:document>")
    buffer = BytesIO()
    with ZipFile(buffer, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/document.xml", document)
    return buffer.getvalue()


class WebUploadParsingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.backend = ProcessPdfBackend()
        self.addCleanup(self.backend.close)

    def test_image_only_pdf_worker_matches_original_loader(self):
        data = image_only_pdf()
        original = load_document("scan.pdf", data)
        isolated = load_document("scan.pdf", data, pdf_backend=self.backend)
        self.assertEqual(isolated.source_kind, original.source_kind)
        self.assertEqual(isolated.page_count, original.page_count)
        self.assertEqual(isolated.units, original.units)
        self.assertEqual(isolated.unread_pages, original.unread_pages)
        self.assertEqual(isolated.warnings, original.warnings)
        self.assertEqual(len(self.backend.worker_pids), 1)
        self.assertNotIn(next(iter(self.backend.worker_pids)), {os.getpid()})

    def test_docx_table_keeps_original_loader_path_without_worker(self):
        data = table_docx()
        original = load_document("table.docx", data)
        isolated = load_document("table.docx", data, pdf_backend=self.backend)
        self.assertEqual(isolated.source_kind, original.source_kind)
        self.assertEqual(isolated.units, original.units)
        self.assertEqual(isolated.warnings, original.warnings)
        self.assertEqual(isolated.page_count, original.page_count)
        self.assertEqual(self.backend.worker_pids, ())

    def test_collection_and_plan_reuse_loaded_document_without_reparsing(self):
        data = mixed_pdf()
        loaded = load_document("plan.pdf", data)
        files = [("plan.pdf", data)]
        with patch("rag_arch.collection.load_document", side_effect=AssertionError("reparsed")):
            collection = build_collection(files, documents=[loaded])
        self.assertIs(collection.documents[0], loaded)

        with patch("rag_arch.plan_documents.load_document", side_effect=AssertionError("reparsed")):
            plan = load_uploaded_document("plan.pdf", data, loaded_document=loaded)
        self.assertEqual(plan.page_count, loaded.page_count)
        self.assertEqual(plan.unread_pages, loaded.unread_pages)
        self.assertEqual(plan.warnings, loaded.warnings)
        self.assertEqual(plan.units[0]["locator"], loaded.units[0].locator)


if __name__ == "__main__":
    unittest.main()
