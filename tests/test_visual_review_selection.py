"""Real PDF geometry checks for conservative first-request image selection."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import pymupdf

from rag_arch.visual_page_index import build_visual_page_index, paginate_visual_page_index
from rag_arch.visual_review_selection import select_initial_visual_pages


def add_numeric_table(pdf, *, values=("12mm", "24mm", "36mm")):
    page = pdf.new_page(width=300, height=400)
    page.insert_text((30, 50), "Table 1 - dimensions")
    for x in (30, 150, 270):
        page.draw_line((x, 100), (x, 220))
    for y in (100, 140, 180, 220):
        page.draw_line((30, y), (270, y))
    for i, value in enumerate(values):
        page.insert_text((40, 125 + i * 40), f"Dimension {i + 1}")
        page.insert_text((160, 125 + i * 40), value)
    return page


def bitmap():
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400), False)
    image.clear_with(220)
    return image.tobytes("png")


def ordinary_text_bitmap():
    with pymupdf.open() as source:
        page = source.new_page(width=300, height=400)
        page.insert_text((20, 40), "Table 1 is mentioned in ordinary text.")
        for index in range(8):
            page.insert_text((20, 80 + index * 25), f"Paragraph {index}: dimension 24mm, load 3kN.")
        return page.get_pixmap().tobytes("png")


class ConservativeSelectionTests(unittest.TestCase):
    def test_text_scan_logo_and_empty_grid_do_not_auto_attach(self):
        with pymupdf.open() as pdf:
            text = pdf.new_page(width=300, height=400)
            text.insert_text((20, 40), "Table 1 is referenced in this ordinary text.")
            text.insert_text((20, 70), "12mm 24mm 36mm 48mm 60mm")

            frame = pdf.new_page(width=300, height=400)
            frame.draw_rect(frame.rect)
            frame.insert_image(pymupdf.Rect(10, 10, 30, 30), stream=bitmap())
            frame.draw_line((30, 60), (270, 60))
            frame.insert_text((30, 80), "Table 2 in ordinary prose")

            scan = pdf.new_page(width=300, height=400)
            scan.insert_image(scan.rect, stream=ordinary_text_bitmap())

            empty = pdf.new_page(width=300, height=400)
            for x in (30, 150, 270):
                empty.draw_line((x, 100), (x, 220))
            for y in (100, 140, 180, 220):
                empty.draw_line((30, y), (270, y))
            empty.insert_text((30, 50), "Table 3 - blank form")

            with patch("pymupdf.Page.get_pixmap", side_effect=AssertionError("must not render")):
                index = build_visual_page_index(pdf)
            self.assertTrue(index["pages"])
            self.assertEqual(select_initial_visual_pages(index["pages"]), [])
            self.assertTrue(all(row["auto_attach_eligible"] is False for row in index["pages"]))

    def test_populated_numeric_grid_qualifies_without_rendering(self):
        with pymupdf.open() as pdf:
            add_numeric_table(pdf)
            with patch("pymupdf.Page.get_pixmap", side_effect=AssertionError("must not render")):
                index = build_visual_page_index(pdf)
            chosen = select_initial_visual_pages(index["pages"])
            self.assertEqual([row["page"] for row in chosen], [1])
            self.assertIn("table_grid", chosen[0]["visual_signals"])
            self.assertEqual(chosen[0]["auto_attach_reasons"], ["populated_numeric_table_grid"])
            self.assertEqual(chosen[0]["auto_attach_table_region"], [0.1, 0.25, 0.9, 0.55])
            self.assertIn("不代表识别已核验", chosen[0]["selection_reason"])

    def test_plain_numeric_text_inside_only_one_column_does_not_qualify(self):
        with pymupdf.open() as pdf:
            add_numeric_table(pdf, values=("", "", ""))
            index = build_visual_page_index(pdf)
            self.assertEqual(select_initial_visual_pages(index["pages"]), [])

    def test_rotated_table_stays_in_navigation_without_automatic_coordinate_inference(self):
        with pymupdf.open() as pdf:
            page = add_numeric_table(pdf)
            page.set_rotation(90)
            index = build_visual_page_index(pdf)
            self.assertTrue(index["pages"])
            self.assertEqual(select_initial_visual_pages(index["pages"]), [])

    def test_later_high_confidence_candidate_is_available_after_directory_pagination(self):
        with pymupdf.open() as pdf:
            for i in range(35):
                page = pdf.new_page(width=300, height=400)
                page.insert_text((20, 40), f"Table {i + 1} is mentioned only in prose.")
            add_numeric_table(pdf)
            index = build_visual_page_index(pdf)
            first = paginate_visual_page_index(index, 1, 30)
            self.assertEqual(select_initial_visual_pages(first["pages"]), [])
            self.assertIsNotNone(first["next_start"])
            all_candidates = list(first["pages"])
            next_page = first["next_start"]
            while next_page is not None:
                batch = paginate_visual_page_index(index, next_page, 30)
                all_candidates.extend(batch["pages"])
                next_page = batch["next_start"]
            self.assertEqual([row["page"] for row in select_initial_visual_pages(all_candidates)], [36])

    def test_hashes_false_is_never_retried_without_explicit_setting(self):
        with pymupdf.open() as pdf:
            add_numeric_table(pdf)
            with patch("pymupdf.Page.get_image_info", side_effect=TypeError("unsupported")) as getter:
                index = build_visual_page_index(pdf)
            getter.assert_called_once_with(hashes=False)
            self.assertEqual([row["page"] for row in select_initial_visual_pages(index["pages"])], [1])

    def test_default_limit_ranking_and_absolute_cap(self):
        with pymupdf.open() as pdf:
            add_numeric_table(pdf)
            source = build_visual_page_index(pdf)["pages"][0]
        rows = []
        for page, priority in ((1, 3), (50, 10), (80, 10)):
            row = deepcopy(source)
            row.update(page=page, auto_attach_priority=priority)
            rows.append(row)
        self.assertEqual([row["page"] for row in select_initial_visual_pages(rows)], [50])
        self.assertEqual([row["page"] for row in select_initial_visual_pages(rows, max_pages=2)], [50, 80])
        self.assertEqual(select_initial_visual_pages(rows, max_pages=0), [])
        for invalid in (True, -1, 3, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_initial_visual_pages(rows, max_pages=invalid)

    def test_old_or_uncertain_metadata_never_becomes_eligible(self):
        rows = [{"page": 1, "visual_signals": ["image", "title_cue"]},
                {"page": 2, "auto_attach_eligible": True, "visual_signals": ["table_grid"]},
                {"page": True, "auto_attach_eligible": True, "visual_signals": ["table_grid"],
                 "auto_attach_reasons": ["populated_numeric_table_grid"]}]
        self.assertEqual(select_initial_visual_pages(rows), [])


if __name__ == "__main__":
    unittest.main()
