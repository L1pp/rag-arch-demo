"""Conservative, metadata-only candidates for pages worth viewing.

The index deliberately avoids rasterising pages, OCR, image hashes, and table
reconstruction.  It is a navigation hint for a later ``read_page`` call, not
an assertion that a page contains a particular kind of diagram or table.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


TEXT_PREVIEW_LIMIT = 160

_TITLE_CUE_RE = re.compile(
    r"(?:^|[\s（(\[])(?:图|表)\s*\d{1,4}(?:\s|$|[：:、.。)])"
    r"|(?:^|[\s\[(])(?:figure|fig\.?|table|chart|diagram|schematic)\s*\d{1,4}",
    re.IGNORECASE,
)


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _preview(text: str, query: str = "") -> str:
    """Return a short preview, keeping a query match near the beginning."""
    compact = _compact_text(text)
    if not compact:
        return ""
    if query:
        position = compact.casefold().find(query.casefold())
        if position >= 0:
            start = max(0, min(position - 40, len(compact) - TEXT_PREVIEW_LIMIT))
            return compact[start:start + TEXT_PREVIEW_LIMIT]
    return compact[:TEXT_PREVIEW_LIMIT]


def _page_text(page: Any) -> str:
    try:
        value = page.get_text("text")
    except TypeError:  # Small fakes in callers may expose get_text() only.
        value = page.get_text()
    return str(value or "").strip()


def _rect_values(rect: Any) -> tuple[float, float, float, float] | None:
    """Read a PyMuPDF Rect or bbox tuple without depending on its repr."""
    if rect is None:
        return None
    try:
        # PyMuPDF Point objects expose x/y rather than a four-value iterator.
        point_x, point_y = float(rect.x), float(rect.y)
        if not hasattr(rect, "x1") and not hasattr(rect, "y1"):
            return point_x, point_y, point_x, point_y
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        values = tuple(float(value) for value in rect)
    except (TypeError, ValueError):
        try:
            values = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        except (AttributeError, TypeError, ValueError):
            return None
    if len(values) != 4:
        return None
    x0, y0, x1, y1 = values
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _rect_area(rect: tuple[float, float, float, float] | None) -> float:
    if rect is None:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _union_rect(rects: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    values = list(rects)
    if not values:
        return None
    return (
        min(rect[0] for rect in values),
        min(rect[1] for rect in values),
        max(rect[2] for rect in values),
        max(rect[3] for rect in values),
    )


def _near_page_edge(rect: tuple[float, float, float, float], page_rect: tuple[float, float, float, float]) -> bool:
    px0, py0, px1, py1 = page_rect
    width, height = px1 - px0, py1 - py0
    edge_x = max(2.0, width * 0.025)
    edge_y = max(2.0, height * 0.025)
    return (
        rect[0] <= px0 + edge_x
        or rect[1] <= py0 + edge_y
        or rect[2] >= px1 - edge_x
        or rect[3] >= py1 - edge_y
    )


def _is_page_frame(rect: tuple[float, float, float, float], page_rect: tuple[float, float, float, float]) -> bool:
    page_area = _rect_area(page_rect)
    area = _rect_area(rect)
    if not page_area or area / page_area < 0.80:
        return False
    px0, py0, px1, py1 = page_rect
    width, height = px1 - px0, py1 - py0
    return (
        rect[0] <= px0 + max(4.0, width * 0.06)
        and rect[1] <= py0 + max(4.0, height * 0.06)
        and rect[2] >= px1 - max(4.0, width * 0.06)
        and rect[3] >= py1 - max(4.0, height * 0.06)
    )


def _image_signals(page: Any, page_rect: tuple[float, float, float, float]) -> tuple[bool, bool]:
    """Return (meaningful image, full-page scan) using image placement metadata."""
    try:
        # Explicitly keep hashes off: the index only needs placement geometry.
        infos = page.get_image_info(hashes=False)
    except TypeError:
        infos = page.get_image_info()
    except (AttributeError, RuntimeError, ValueError):
        infos = []
    page_area = _rect_area(page_rect)
    if not page_area:
        return False, False
    meaningful = False
    largest_ratio = 0.0
    total_ratio = 0.0
    for info in infos or ():
        bbox = _rect_values(info.get("bbox") if isinstance(info, dict) else None)
        area = _rect_area(bbox)
        ratio = area / page_area
        largest_ratio = max(largest_ratio, ratio)
        total_ratio += ratio
        # A tiny header logo is intentionally ignored.  The area threshold is
        # conservative; a later page read remains the authority.
        if ratio >= 0.035:
            meaningful = True
    scanned = largest_ratio >= 0.55 and total_ratio >= 0.55
    return meaningful, scanned


def _vector_signal(page: Any, page_rect: tuple[float, float, float, float]) -> bool:
    """Recognise substantial vector structure while ignoring page frames/logos."""
    try:
        drawings = page.get_drawings()
    except (AttributeError, RuntimeError, ValueError):
        drawings = []
    page_area = _rect_area(page_rect)
    if not page_area:
        return False

    useful_rects: list[tuple[float, float, float, float]] = []
    line_count = 0
    shape_count = 0
    curve_count = 0
    for drawing in drawings or ():
        if not isinstance(drawing, dict):
            continue
        rect = _rect_values(drawing.get("rect"))
        if rect is None or _is_page_frame(rect, page_rect):
            continue
        items = drawing.get("items") or ()
        local_useful = False
        for item in items:
            if not item:
                continue
            kind = item[0]
            if kind == "l":
                endpoints = [_rect_values(point) for point in item[1:3]]
                line_rect = _union_rect([point for point in endpoints if point is not None])
                page_width = page_rect[2] - page_rect[0]
                page_height = page_rect[3] - page_rect[1]
                line_width = line_rect[2] - line_rect[0] if line_rect is not None else 0.0
                line_height = line_rect[3] - line_rect[1] if line_rect is not None else 0.0
                edge_x = max(2.0, page_width * 0.025)
                edge_y = max(2.0, page_height * 0.025)
                is_page_edge_line = bool(
                    line_rect is not None
                    and (
                        (line_width >= 0.80 * page_width and line_height <= edge_y
                         and (line_rect[1] <= page_rect[1] + edge_y
                              or line_rect[3] >= page_rect[3] - edge_y))
                        or (line_height >= 0.80 * page_height and line_width <= edge_x
                            and (line_rect[0] <= page_rect[0] + edge_x
                                 or line_rect[2] >= page_rect[2] - edge_x))
                    )
                )
                if line_rect is not None and not (
                    _near_page_edge(line_rect, page_rect) and is_page_edge_line
                ):
                    line_count += 1
                    local_useful = True
            elif kind in {"re", "qu"}:
                shape_count += 1
                local_useful = True
            elif kind in {"c", "bs"}:
                curve_count += 1
                local_useful = True
        if local_useful:
            useful_rects.append(rect)

    region = _union_rect(useful_rects)
    region_ratio = _rect_area(region) / page_area if region is not None else 0.0
    if region_ratio < 0.06:
        return False
    # Tables tend to have several rules or cells; charts/diagrams tend to have
    # multiple paths.  One interior box or one decorative rule is insufficient.
    return (
        (line_count >= 3 and region_ratio >= 0.10)
        or (shape_count >= 2 and line_count >= 1 and region_ratio >= 0.08)
        or (shape_count >= 3 and region_ratio >= 0.08)
        or (curve_count >= 2 and region_ratio >= 0.08)
        or (line_count + shape_count + curve_count >= 6 and region_ratio >= 0.08)
    )


def _title_signal(text: str) -> bool:
    return bool(_TITLE_CUE_RE.search(_compact_text(text)))


def _inspect_page(page: Any, number: int, text: str) -> dict | None:
    page_rect = _rect_values(getattr(page, "rect", None))
    if page_rect is None:
        return None
    has_text = bool(text)
    meaningful_image, scanned = _image_signals(page, page_rect)
    vector = _vector_signal(page, page_rect)
    title = _title_signal(text)
    signals: list[str] = []
    if meaningful_image:
        signals.append("image")
    if vector:
        signals.append("vector_or_table")
    if scanned and not has_text:
        signals.append("scanned_page")
    if title:
        signals.append("title_cue")
    if not signals:
        return None
    return {
        "page": number,
        "has_text": has_text,
        "text_preview": _preview(text),
        "visual_candidate": True,
        "visual_signals": signals,
    }


def build_visual_page_index(document: Any) -> dict:
    """Scan one already-open PDF and return a complete candidate index."""
    page_count = len(document)
    pages: list[dict] = []
    text_by_page: dict[int, str] = {}
    for index in range(page_count):
        page = document[index]
        text = _page_text(page)
        text_by_page[index + 1] = text
        candidate = _inspect_page(page, index + 1, text)
        if candidate is not None:
            pages.append(candidate)
    return {
        "page_count": page_count,
        "pages": pages,
        "visual_candidate_count": len(pages),
        "visual_index_complete": True,
        # Kept internal to the page service so query filtering can inspect the
        # complete text without returning a long body to the model.
        "_text_by_page": text_by_page,
    }


def paginate_visual_page_index(index: dict, start: int, limit: int, query: str = "") -> dict:
    """Filter and page a complete index using physical PDF page numbers."""
    page_count = int(index.get("page_count", 0))
    if start > page_count:
        raise ValueError("起始页码超出来源页面范围。")
    text_by_page = index.get("_text_by_page", {})
    folded_query = query.casefold() if query else ""
    matches: list[dict] = []
    for candidate in index.get("pages", ()):
        page = int(candidate["page"])
        if page < start:
            continue
        text = str(text_by_page.get(page, ""))
        if not text and query:
            # A lightweight injected scanner may only return the short preview.
            text = str(candidate.get("text_preview", ""))
        if folded_query and folded_query not in text.casefold():
            continue
        row = {key: value for key, value in candidate.items() if key != "text_preview"}
        row["text_preview"] = (_preview(text, query) if text
                                else str(candidate.get("text_preview", ""))[:TEXT_PREVIEW_LIMIT])
        matches.append(row)
    rows = matches[:limit]
    next_start = rows[-1]["page"] + 1 if len(matches) > limit else None
    return {
        "page_count": page_count,
        "pages": rows,
        "visual_candidate_count": int(index.get("visual_candidate_count", len(index.get("pages", ())))),
        "visual_index_complete": bool(index.get("visual_index_complete", True)),
        "next_start": next_start,
    }
