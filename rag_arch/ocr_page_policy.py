"""Select pages whose existing text layer may not cover visible content.

This is a conservative geometry/text heuristic, not an OCR or image classifier.
It intentionally does not assert that non-candidates have complete text: a small
image, a difficult text layer, or a drawing can still contain missing content.
The caller is responsible for exposing this limitation to the user.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .visual_page_index import _vector_signal


# Ignore small logos and ordinary illustrations.  A substantial image may be
# a scan or an image table, but can also be a diagram or photograph.
LARGE_IMAGE_FRACTION = 0.20
IMAGE_DOMINANT_FRACTION = 0.50
SPARSE_TEXT_CHARS = 80
BODY_TOP_FRACTION = 0.10
BODY_BOTTOM_FRACTION = 0.90


def _rect_values(value: Any) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        return None
    return values


def _intersection(rect, bounds):
    x0, y0 = max(rect[0], bounds[0]), max(rect[1], bounds[1])
    x1, y1 = min(rect[2], bounds[2]), min(rect[3], bounds[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _area(rect) -> float:
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def _union_area(rectangles) -> float:
    """Return geometric union, so duplicated/overlaid images are not summed."""
    xs = sorted({r[0] for r in rectangles} | {r[2] for r in rectangles})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        low = high = None
        height = 0.0
        for bottom, top in intervals:
            if low is None:
                low, high = bottom, top
            elif bottom <= high:
                high = max(high, top)
            else:
                height += high - low
                low, high = bottom, top
        if low is not None:
            height += high - low
        total += (right - left) * height
    return total


def _compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _words(page, strip_watermark) -> list[tuple[tuple[float, float, float, float], str]]:
    words = []
    for item in page.get_text("words"):
        if len(item) < 5 or not isinstance(item[4], str):
            continue
        rect = _rect_values(item[:4])
        text, _ = strip_watermark(item[4])
        if rect is not None and text.strip():
            words.append((rect, text))
    return words


def _inside(word, rect) -> bool:
    bounds, _ = word
    x = (bounds[0] + bounds[2]) / 2
    y = (bounds[1] + bounds[3]) / 2
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _visually_white(page, bounds) -> bool:
    """Only detect an essentially pure-white page at a bounded thumbnail size.

    Gray or textured backgrounds are intentionally left as possible OCR pages.
    A thumbnail cannot guarantee that all extremely small marks are visible.
    """
    import pymupdf
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    scale = min(1.0, 128.0 / max(width, height))
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                             colorspace=pymupdf.csGRAY, alpha=False)
    return bool(pixmap.samples) and all(value >= 254 for value in pixmap.samples)


def ocr_page_candidate(page: Any, number: int) -> dict:
    """Return one page's local, non-billing OCR-selection metadata.

    ``blank`` means no extractable text or substantial visual signal was
    found, or a no-text image page appeared pure white at thumbnail scale; it
    is a heuristic rather than proof no faint marks exist. No recognized text
    or user content is returned.
    """
    if type(number) is not int or number < 1:
        raise ValueError("物理页码必须是正整数。")
    result = {"page": number, "needs_ocr": False, "reason": "native_text",
              "native_chars": 0, "body_chars": 0, "image_coverage": 0.0,
              "blank": False}
    try:
        # PyMuPDF returns word/image/drawing coordinates in unrotated space,
        # whereas page.rect reflects rotation. Put the bounds in the same
        # coordinate space before comparing geometry.
        rotation = getattr(page, "rotation", 0)
        page_bounds = page.rect
        if type(rotation) is int and rotation % 360 in (90, 180, 270):
            page_bounds = page_bounds * page.derotation_matrix
        bounds = _rect_values(page_bounds)
        if bounds is None:
            raise ValueError("页面尺寸无效。")
        page_area = _area(bounds)
        # Match the upload loader's watermarked-page handling, without adding a
        # second definition that could treat a watermark as useful body text.
        from .loaders import _without_watermark
        raw_text = page.get_text("text", sort=True)
        cleaned_text, _ = _without_watermark(raw_text)
        count = _compact_length(cleaned_text)
        result["native_chars"] = count
        words = _words(page, _without_watermark)
        body_top = bounds[1] + (bounds[3] - bounds[1]) * BODY_TOP_FRACTION
        body_bottom = bounds[1] + (bounds[3] - bounds[1]) * BODY_BOTTOM_FRACTION
        body_rect = (bounds[0], body_top, bounds[2], body_bottom)
        result["body_chars"] = sum(_compact_length(text) for word, text in words
                                   if _inside((word, text), body_rect))

        # Only geometric placement is needed. Explicitly disable image hashes.
        # If the installed API cannot explicitly disable hashing, the outer
        # handler marks this page uncertain instead of guessing its defaults.
        image_info = page.get_image_info(hashes=False)
        if len(image_info or ()) > 2048:
            raise ValueError("页面图片区域过多，无法可靠检测。")
        rectangles = []
        for item in image_info or ():
            if not isinstance(item, dict):
                continue
            rect = _rect_values(item.get("bbox"))
            if rect is not None:
                clipped = _intersection(rect, bounds)
                if clipped is not None:
                    rectangles.append(clipped)
        coverage = min(1.0, _union_area(rectangles) / page_area)
        result["image_coverage"] = round(coverage, 6)

        if not count and rectangles and _visually_white(page, bounds):
            result.update(blank=True, reason="visually_blank_image")
            return result

        bad_chars = cleaned_text.count("\ufffd") + sum(
            1 for char in cleaned_text if ord(char) < 32 and char not in "\t\r\n\f")
        if bad_chars > max(1, count * 0.10):
            result.update(needs_ocr=True, reason="unreadable_text_layer")
            return result

        large_images = [rect for rect in rectangles if _area(rect) / page_area >= LARGE_IMAGE_FRACTION]
        if coverage >= IMAGE_DOMINANT_FRACTION and result["body_chars"] < SPARSE_TEXT_CHARS:
            result.update(needs_ocr=True, reason="image_dominant_sparse_body")
            return result
        for image_rect in large_images:
            image_text = sum(_compact_length(text) for word, text in words
                             if _inside((word, text), image_rect))
            if image_text < SPARSE_TEXT_CHARS:
                result.update(needs_ocr=True, reason="image_region_sparse_text")
                return result

        if not count:
            substantial_drawing = _vector_signal(page, bounds)
            if substantial_drawing:
                result.update(needs_ocr=True, reason="vector_content_without_text")
            elif rectangles:
                # A small logo by itself is not a reason to submit a page.
                result["reason"] = "small_image_without_text"
            elif not _visually_white(page, bounds):
                # A sparse decorative line or mark is not a substantial
                # diagram, but the page is not objectively blank either.
                result["reason"] = "minor_drawing_without_text"
            else:
                result.update(blank=True, reason="blank")
        return result
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        # Explicitly expose detection failure. The caller must retain its
        # unread-page warning; absence of a candidate is not coverage proof.
        result.update(needs_ocr=False, reason="page_analysis_failed", blank=False)
        return result


def build_ocr_page_candidates(document: Any) -> list[dict]:
    """Examine every page, preserving physical page order and zero-based API."""
    return [ocr_page_candidate(document[index], index + 1) for index in range(len(document))]
