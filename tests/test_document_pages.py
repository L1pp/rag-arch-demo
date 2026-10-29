from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
import pymupdf

from rag_arch.document_pages import (
    DocumentPageService, MAX_IMAGE_EDGE, MAX_IMAGE_PIXELS,
)


def make_pdf(*texts, size=(300, 400)):
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page(width=size[0], height=size[1])
            if text:
                page.insert_text((20, 30), text)
            else:
                image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
                image.clear_with(200)
                page.insert_image(page.rect, stream=image.tobytes("png"))
        return pdf.tobytes()


def make_docx(extra=None):
    buffer = BytesIO()
    document = Document()
    document.add_paragraph("Original unpaginated source")
    document.save(buffer)
    if not extra:
        return buffer.getvalue()
    output = BytesIO()
    with ZipFile(buffer) as old, ZipFile(output, "w", ZIP_DEFLATED) as new:
        for name in old.namelist():
            new.writestr(name, old.read(name))
        for name, data in extra.items():
            new.writestr(name, data)
    return output.getvalue()


class DocumentPageServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.calls = []

        def renderer(source, output):
            self.calls.append((source, output))
            output.write_bytes(make_pdf("Rendered page one", "Rendered page two"))

        self.renderer = renderer
        self.service = DocumentPageService(self.root, docx_renderer=renderer)

    def test_path_catalog_never_reads_copies_or_renders_library(self):
        paths = []
        for n in range(3):
            path = self.root / f"standard-{n}.pdf"
            path.write_bytes(make_pdf(f"Standard {n}"))
            paths.append(path)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("eager read")), \
                patch("rag_arch.document_pages.shutil.copyfile",
                      side_effect=AssertionError("eager copy")), \
                patch("rag_arch.document_pages.pymupdf.open",
                      side_effect=AssertionError("eager parsing")):
            for n, path in enumerate(paths):
                self.service.register(f"S{n}", path.name, path=path, label="规范")
            catalog = self.service.sources()
            self.assertEqual(len(catalog), 3)
            self.assertIsNone(catalog[0]["page_count"])
            self.assertEqual(self.service.describe()["sources"], catalog)
        self.assertEqual(list(self.service.cache_dir.iterdir()), [])
        result = self.service.read_page("S1", 1)
        self.assertIn("Standard 1", result["text"])
        self.assertEqual(list(self.service.cache_dir.iterdir()), [])

    def test_catalog_sanitizes_filename_and_sources_are_isolated(self):
        self.service.register("upload", r"C:\fakepath\example.docx", data=make_docx())
        source = self.service.describe("upload")
        self.assertEqual(source["filename"], "example.docx")
        self.assertEqual(source["label"], "example.docx")
        self.assertIsNone(source["page_count"])
        self.assertEqual(self.calls, [])
        self.assertNotIn(str(self.root), str(self.service.sources()))
        other = DocumentPageService(self.root)
        for ref in ("upload", str(self.root / "source.pdf"), "../source.pdf"):
            with self.assertRaisesRegex(ValueError, "未登记"):
                other.read_page(ref, 1)

    def test_pdf_search_pagination_and_image_match_the_same_page(self):
        self.service.register("pdf", "sample.pdf",
                              data=make_pdf("First plain page", "Second needle page", "Third page"))
        result = self.service.list_pages("pdf", limit=1)
        self.assertEqual(result["page_count"], 3)
        self.assertEqual(result["next_start"], 2)
        found = self.service.list_pages("pdf", query="NEEDLE")
        self.assertEqual([x["page"] for x in found["pages"]], [2])
        page = self.service.read_page("pdf", 2)
        self.assertIn("Second needle page", page["text"])
        self.assertEqual(page["mime_type"], "image/png")
        self.assertTrue(page["image_bytes"].startswith(b"\x89PNG"))
        self.assertGreater(page["width"], 0)
        self.assertNotIn("input_path", page)

    def test_scanned_pdf_and_registered_images_remain_readable_as_images(self):
        self.service.register("scan", "scan.pdf", data=make_pdf(""))
        listing = self.service.list_pages("scan")
        self.assertFalse(listing["pages"][0]["has_text"])
        self.assertEqual(self.service.read_page("scan", 1)["text"], "")
        self.assertEqual(self.service.list_pages("scan", query="absent")["pages"], [])
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 30, 40), False)
        pixmap.clear_with(150)
        for extension, encoding in [("png", "png"), ("jpg", "jpeg")]:
            with self.subTest(extension=extension):
                self.service.register(extension, "图." + extension, data=pixmap.tobytes(encoding))
                self.assertEqual(self.service.describe(extension)["page_count"], 1)
                image = self.service.read_page(extension, 1)
                self.assertEqual(image["page_count"], 1)
                self.assertEqual(image["text"], "")

    def test_docx_uses_one_rendered_pdf_for_text_and_image_and_reuses_it(self):
        data = make_docx()
        self.service.register("word", "plan.docx", data=data)
        listing = self.service.list_pages("word", query="two")
        self.assertEqual(listing["pages"][0]["page"], 2)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0][0].is_relative_to(self.service.cache_root))
        page = self.service.read_page("word", 2)
        self.assertEqual(page["text"], "Rendered page two")
        self.assertEqual(page["page_count"], 2)
        self.service.register("word", "plan.docx", data=data)
        self.service.read_page("word", 1)
        self.assertEqual(len(self.calls), 1)

    def test_docx_path_is_staged_only_when_read(self):
        path = self.root / "plan.docx"
        path.write_bytes(make_docx())
        self.service.register("word", path.name, path=path)
        self.assertEqual(list(self.service.cache_dir.iterdir()), [])
        self.service.read_page("word", 1)
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0][0], path)

    def test_changed_path_requires_reregistration_and_duplicate_names_do_not_mix(self):
        path = self.root / "example.pdf"
        path.write_bytes(make_pdf("Old page"))
        self.service.register("path", "same.pdf", path=path)
        self.service.register("bytes", "same.pdf", data=make_pdf("Other source"))
        path.write_bytes(make_pdf("Changed first page", "Changed second page"))
        with self.assertRaisesRegex(ValueError, "已更新"):
            self.service.read_page("path", 1)
        self.service.register("path", "same.pdf", path=path)
        self.assertEqual(self.service.read_page("path", 2)["page_count"], 2)
        self.assertEqual(self.service.read_page("bytes", 1)["text"], "Other source")

    def test_invalid_page_and_render_limits(self):
        self.service.register("a", "a.pdf", data=make_pdf("Large", size=(10000, 10000)))
        for page in [0, -1, True, 1.2, 2, "1"]:
            with self.subTest(page=page), self.assertRaises(ValueError):
                self.service.read_page("a", page)
        for kwargs in [{"start": 0}, {"limit": 51}, {"query": "x" * 201}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.service.list_pages("a", **kwargs)
        for scale in [0, True, float("inf"), float("nan"), 5]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.service.read_page("a", 1, scale)
        result = self.service.read_page("a", 1, 4)
        self.assertLessEqual(max(result["width"], result["height"]), MAX_IMAGE_EDGE)
        self.assertLessEqual(result["width"] * result["height"], MAX_IMAGE_PIXELS)

    def test_bad_format_encryption_and_cache_outside_project_fail(self):
        with self.assertRaises(ValueError):
            self.service.register("bad", "bad.doc", data=b"x")
        with self.assertRaises(ValueError):
            self.service.register("bad", "bad.pdf")
        with self.assertRaises(ValueError):
            DocumentPageService(self.root, self.root / "elsewhere")
        self.service.register("bad", "bad.pdf", data=b"not a PDF")
        with self.assertRaises(ValueError):
            self.service.read_page("bad", 1)
        with pymupdf.open() as pdf:
            pdf.new_page()
            encrypted = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                    owner_pw="owner", user_pw="password")
        self.service.register("locked", "locked.pdf", data=encrypted)
        with self.assertRaisesRegex(ValueError, "加密"):
            self.service.list_pages("locked")
        self.service.register("bad-word", "bad.docx", data=b"bad zip")
        with self.assertRaisesRegex(ValueError, "DOCX"):
            self.service.read_page("bad-word", 1)

    def test_docx_active_content_is_rejected_but_internal_fields_and_links_allowed(self):
        fields = ('<w:root xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                  '<w:instrText> TOC PAGE REF section </w:instrText></w:root>')
        rel_start = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        hyperlink = (rel_start + '<Relationship Id="r1" Target="https://example.invalid" '
                     'TargetMode="External" Type="test/hyperlink"/></Relationships>')
        self.service.register("safe", "safe.docx", data=make_docx({
            "word/test.xml": fields, "word/_rels/test.xml.rels": hyperlink}))
        self.service.read_page("safe", 1)
        self.assertEqual(len(self.calls), 1)
        dangerous = [
            {"word/vbaProject.bin": b"not executed"},
            {"word/_rels/test.xml.rels": hyperlink.replace("test/hyperlink", "test/attachedTemplate")},
            {"word/test.xml": fields.replace("TOC PAGE REF section", "INCLUDETEXT external")},
        ]
        for index, extra in enumerate(dangerous):
            ref = f"unsafe-{index}"
            self.service.register(ref, "unsafe.docx", data=make_docx(extra))
            with self.subTest(extra=list(extra)), self.assertRaises(ValueError):
                self.service.read_page(ref, 1)
        self.assertEqual(len(self.calls), 1)

    def test_word_script_opens_cache_readonly_before_acquiring_shutdown_authority(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/render_docx_pages.ps1").read_text(
            encoding="utf-8")
        self.assertNotIn("$documents.Add()", script)
        owner_write = script.index("[IO.File]::Move($ownerTemp, $OwnerFile)")
        readonly_open = script.index("$documents.Open($InputPath, $false, $true, $false,")
        window = script.index("GetWindowThreadProcessId([IntPtr]$word.Hwnd")
        ownership_guard = script.index("$before -contains [int]$wordProcessId")
        owned = script.index("$owned = $true")
        self.assertLess(script.index("$word.AutomationSecurity = 3"), readonly_open)
        self.assertLess(window, ownership_guard)
        self.assertLess(ownership_guard, owned)
        self.assertLess(owned, owner_write)
        self.assertLess(owner_write, readonly_open)
        self.assertIn("$process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$owner.start_ticks",
                      script)
        self.assertIn("$process.Kill()", script)
        self.assertLess(script.index("$ownedProcessHandle = $process.Handle"),
                        script.index("$process.Kill()"))
        self.assertIn("$process.WaitForExit(10000)", script)
        self.assertIn("$process.Dispose()", script)
        self.assertIn("$remaining.StartTime.ToUniversalTime().Ticks.ToString()", script)
        self.assertIn("$wordStartTicks = $process.StartTime.ToUniversalTime().Ticks.ToString()", script)
        self.assertNotIn("process_start_token", script)
        self.assertIn("[Diagnostics.Process]::GetProcessesByName('WINWORD')", script)
        self.assertIn("[Diagnostics.Process]::GetProcessById($ProcessId)", script)
        self.assertNotIn("Get-Process -Id", script)
        self.assertIn("The owned Word identity file is missing", script)
        self.assertNotIn("Stop-Process -Id", script)
        self.assertIn("DOCX_PAGE_WORD_CLEANUP_VERIFIED", script)
        self.assertIn("if ($owned) {", script)


if __name__ == "__main__":
    unittest.main()
