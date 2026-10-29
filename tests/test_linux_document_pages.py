from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from rag_arch.document_pages import DocumentPageService
from tests.test_document_pages import make_docx, make_pdf


DOC_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"synthetic-doc"


class FakeOffice:
    def __init__(self):
        self.normalizations = []
        self.renders = []

    def ensure_docx(self, source, target):
        self.normalizations.append((Path(source), Path(target)))
        Path(target).write_bytes(make_docx())

    def render_pdf(self, source, target):
        self.renders.append((Path(source), Path(target)))
        Path(target).write_bytes(make_pdf("Original page one", "Original page two"))

    def ensure_pdf(self, source, target):
        if not Path(target).exists():
            self.render_pdf(source, target)


class LinuxDocumentPageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.office = FakeOffice()
        self.service = DocumentPageService(
            self.root, office_converter=self.office,
            docx_renderer=self.office.render_pdf)

    def test_legacy_doc_normalizes_once_and_uses_same_pdf_for_text_and_image(self):
        self.service.register("legacy", "plan.doc", data=DOC_BYTES)
        self.assertEqual(self.office.normalizations, [])
        listing = self.service.list_pages("legacy")
        image = self.service.read_page("legacy", 2)
        self.assertEqual(listing["page_count"], 2)
        self.assertEqual(image["text"], "Original page two")
        self.assertEqual(self.service.describe("legacy")["filename"], "plan.doc")
        self.assertEqual(len(self.office.normalizations), 1)
        self.assertEqual(len(self.office.renders), 1)
        self.assertEqual(self.office.renders[0][0].suffix, ".docx")
        self.assertEqual(image["image_bytes"][:8], b"\x89PNG\r\n\x1a\n")
        self.service.register("legacy", "plan.doc", data=DOC_BYTES)
        self.service.read_page("legacy", 1)
        self.assertEqual(len(self.office.renders), 1)

    def test_legacy_upload_reuses_the_normalized_document_used_for_text_extraction(self):
        original = self.root / "source.doc"
        normalized = self.root / "normalized.docx"
        original.write_bytes(DOC_BYTES)
        normalized.write_bytes(make_docx())
        expected = normalized.read_bytes()
        self.service.register("legacy", "source.doc", path=original,
                              normalized_docx_path=normalized)
        self.service.read_page("legacy", 1)
        self.assertEqual(self.office.normalizations, [])
        rendered_input = self.office.renders[0][0]
        self.assertEqual(rendered_input.read_bytes(), expected)
        self.assertNotEqual(rendered_input, normalized)
        self.assertEqual(original.read_bytes(), DOC_BYTES)
        normalized.write_bytes(expected + b"changed")
        with self.assertRaisesRegex(ValueError, "派生文字文档已更新"):
            self.service.read_page("legacy", 1)

    def test_derived_active_content_is_rejected_before_rendering(self):
        original = self.root / "source.doc"
        normalized = self.root / "normalized.docx"
        original.write_bytes(DOC_BYTES)
        normalized.write_bytes(make_docx({"word/vbaProject.bin": b"unsafe"}))
        self.service.register("legacy", "source.doc", path=original,
                              normalized_docx_path=normalized)
        with self.assertRaisesRegex(ValueError, "宏"):
            self.service.read_page("legacy", 1)
        self.assertEqual(self.office.renders, [])

    def test_default_renderer_preserves_windows_word_and_uses_office_elsewhere(self):
        service = DocumentPageService(self.root, office_converter=self.office)
        source, target = self.root / "source.docx", self.root / "out.pdf"
        with patch.object(service, "_render_with_word") as word:
            service._render_office(source, target)
            if os.name == "nt":
                word.assert_called_once_with(source, target)
                self.assertEqual(self.office.renders, [])
            else:
                word.assert_not_called()
                self.assertEqual(len(self.office.renders), 1)

    def test_linux_upload_services_reuse_one_asset_pdf(self):
        asset = self.root / "data/runtime/web_uploads/1/workspace/asset/source.docx"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(make_docx())
        with patch.object(DocumentPageService, "_uses_libreoffice", return_value=True):
            first = DocumentPageService(self.root, office_converter=self.office)
            second = DocumentPageService(self.root, office_converter=self.office)
            first.register("preview", "upload.docx", path=asset)
            second.register("plan", "upload.docx", path=asset)
            self.assertIn("Original page one", first.read_page("preview", 1)["text"])
            self.assertIn("Original page two", second.read_page("plan", 2)["text"])
        self.assertEqual(len(self.office.renders), 1)
        self.assertEqual(self.office.renders[0][0], asset)
        self.assertEqual(self.office.renders[0][1], asset.with_name("pages.pdf"))


if __name__ == "__main__":
    unittest.main()
