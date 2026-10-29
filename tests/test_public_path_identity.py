"""Source paths follow the actual filesystem's case semantics."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pymupdf
from rag_arch.page_sources import PageSources, _role_path, add_standard_sources


def write_pdf(path, text):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((20, 30), text)
        pdf.save(path)


class PublicPathIdentityTests(unittest.TestCase):
    def test_lookup_does_not_invent_a_case_variant_on_case_sensitive_filesystems(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            original, alternate = root / "Demo.pdf", root / "demo.pdf"
            write_pdf(original, "AUTHORED DEMO")
            pages = PageSources(root)
            pages.add("standard:1", original.name, path=original, source_kind="standard")
            self.assertEqual(pages.source_ref_for_path(str(original)), "standard:1")
            self.assertEqual(pages.source_ref_for_path(str(alternate)),
                             "standard:1" if alternate.exists() else None)

    def test_case_distinct_originals_keep_separate_registration_and_role_keys(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            upper, lower = root / "A.pdf", root / "a.pdf"
            write_pdf(upper, "UPPER ORIGINAL")
            if lower.exists():
                self.skipTest("case-sensitive filesystem required")
            write_pdf(lower, "lower original")
            corpus = SimpleNamespace(catalog=[], tables=[], clauses=[{
                "std_no": "SYNTHETIC-1001", "source_spans": [
                    {"source_path": str(upper), "page_start": 1, "page_end": 1},
                    {"source_path": str(lower), "page_start": 1, "page_end": 1},
                ]}])
            pages = PageSources(root)
            add_standard_sources(pages, corpus)
            self.assertEqual(len(pages.sources()), 2)
            upper_ref = pages.source_ref_for_path(str(upper))
            lower_ref = pages.source_ref_for_path(str(lower))
            self.assertNotEqual(upper_ref, lower_ref)
            self.assertNotEqual(_role_path(root, str(upper)), _role_path(root, str(lower)))
            self.assertIn("UPPER ORIGINAL", pages.read_page(upper_ref, 1)["text"])
            self.assertIn("lower original", pages.read_page(lower_ref, 1)["text"])
