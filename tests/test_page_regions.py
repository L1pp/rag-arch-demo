"""Region coordinates refer to displayed pixels, not unrotated PDF axes."""
from io import BytesIO
import unittest

import pymupdf
from PIL import Image

from rag_arch.page_regions import (normalize_region, page_image_id, render_page_image,
                                   validate_page_result_identity)


def coloured_pdf(rotation=0, crop=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=300)
        for rect, color in [((0, 0, 200, 150), (1, 0, 0)),
                            ((200, 0, 400, 150), (0, 1, 0)),
                            ((0, 150, 200, 300), (0, 0, 1)),
                            ((200, 150, 400, 300), (1, 1, 0))]:
            page.draw_rect(rect, color=color, fill=color)
        page.insert_text((70, 70), "LEFT 7 kN", color=(0, 0, 0))
        page.insert_text((250, 200), "RIGHT 9 mm", color=(0, 0, 0))
        if crop:
            page.set_cropbox(pymupdf.Rect(50, 30, 350, 270))
        page.set_rotation(rotation)
        return pdf.tobytes()


class PageRegionTests(unittest.TestCase):
    def test_result_identity_requires_exact_source_page_and_explicit_requested_crop(self):
        whole = {"source_ref": "plan:current", "page": 1}
        self.assertIs(validate_page_result_identity(whole, "plan:current", 1), whole)
        detail = {**whole, "region": [0, 0, .5, .5]}
        self.assertIs(validate_page_result_identity(detail, "plan:current", 1,
                                                   (0.0, 0.0, .5, .5)), detail)
        invalid = [({}, None), ({**whole, "source_ref": "other"}, None),
                   ({**whole, "page": 2}, None), ({**whole, "page": True}, None),
                   (whole, [0, 0, .5, .5]), ({**whole, "region": None}, [0, 0, .5, .5]),
                   (detail, None), ({**whole, "region": [0, 0, .6, .5]}, [0, 0, .5, .5]),
                   ({**whole, "region": [False, 0, .5, .5]}, [0, 0, .5, .5])]
        for result, requested in invalid:
            with self.subTest(result=result, region=requested), self.assertRaises(ValueError):
                validate_page_result_identity(result, "plan:current", 1, requested)

    def test_validation_and_explicit_coordinate_identity(self):
        self.assertIsNone(normalize_region(None))
        self.assertEqual(normalize_region([-0.0, 0, .5, 1]), (0.0, 0.0, .5, 1.0))
        self.assertEqual(page_image_id("plan:current", 7), "PAGE:plan:current:7")
        self.assertEqual(page_image_id("plan:current", 7, [0, 0, .5, .5]),
                         "PAGE:plan:current:7:REGION:0.0,0.0,0.5,0.5")
        for region in ([False, 0, 1, 1], [0, 0, float("nan"), 1], [0, 0, float("inf"), 1],
                       [0, 0, 10 ** 500, 1], [0, 0, 0, 1], [0, .5, 1, .5],
                       [-.1, 0, .5, 1], [0, 0, 1.1, 1], [0, 0, 1], "0,0,1,1", {}):
            with self.subTest(region=repr(region)), self.assertRaises(ValueError):
                normalize_region(region)

    def test_display_crop_stays_on_the_same_quadrant_after_rotation_and_cropbox(self):
        expected = {0: (255, 0, 0), 90: (0, 0, 255), 180: (255, 255, 0), 270: (0, 255, 0)}
        for rotation in expected:
            for cropbox in (False, True):
                with self.subTest(rotation=rotation, cropbox=cropbox):
                    with pymupdf.open(stream=coloured_pdf(rotation, cropbox), filetype="pdf") as pdf:
                        result = render_page_image(pdf[0], region=[0, 0, .5, .5])
                    with Image.open(BytesIO(result["image_bytes"])) as image:
                        pixel = image.getpixel((image.width // 2, image.height // 3))
                    self.assertEqual(pixel, expected[rotation])
                    self.assertEqual(result["rotation"], rotation)
                    self.assertEqual(result["text_scope"], "region")

    def test_rotated_region_text_excludes_other_quadrants(self):
        for cropped in (False, True):
            with self.subTest(cropbox=cropped):
                with pymupdf.open(stream=coloured_pdf(90, cropped), filetype="pdf") as pdf:
                    result = render_page_image(pdf[0], region=[.5, 0, 1, .5])
                self.assertIn("LEFT 7 kN", result["text"])
                self.assertNotIn("RIGHT", result["text"])

    def test_local_render_has_more_detail_than_the_same_region_of_overview(self):
        with pymupdf.open(stream=coloured_pdf(), filetype="pdf") as pdf:
            whole = render_page_image(pdf[0])
            detail = render_page_image(pdf[0], region=[0, 0, .5, .5])
        self.assertGreater(detail["rendered_scale"], whole["rendered_scale"])
        self.assertGreater(detail["width"], whole["width"] * .5)
        self.assertLessEqual(max(detail["width"], detail["height"]), 2400)
        self.assertLessEqual(detail["width"] * detail["height"], 4_000_000)
        self.assertLessEqual(len(detail["image_bytes"]), 8 * 1024 * 1024)
        self.assertEqual(whole["text_scope"], "page")
        self.assertIsNone(whole["region"])

    def test_extreme_sizes_and_subpixel_regions_fail_or_remain_bounded(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=10000, height=10000)
            result = render_page_image(page, scale=4, region=[0, 0, .8, .8])
            self.assertLessEqual(result["width"] * result["height"], 4_000_000)
            self.assertLessEqual(max(result["width"], result["height"]), 2400)
            with self.assertRaisesRegex(ValueError, "区域过小"):
                render_page_image(page, region=[0, 0, 1e-300, .5])


if __name__ == "__main__":
    unittest.main()
