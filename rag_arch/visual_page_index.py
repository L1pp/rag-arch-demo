"""Conservative, metadata-only candidates for pages worth viewing.

The index deliberately avoids rasterising pages, OCR, image hashes, and table
reconstruction.  It is a navigation hint for a later ``read_page`` call, not
an assertion that a page contains a particular kind of diagram or table.
"""
from __future__ import annotations

from bisect import bisect_right
import math
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
    except (TypeError, AttributeError, RuntimeError, ValueError):
        # Older/injected APIs may not support an explicit hashes=False.  Do not
        # fall back to an unspecified call; geometry hints are optional.
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


def _vector_signal(page: Any, page_rect: tuple[float, float, float, float], drawings=None) -> bool:
    """Recognise substantial vector structure while ignoring page frames/logos."""
    if drawings is None:
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


def _merged_rules(rules):
    """Merge adjacent collinear grid segments, retaining a small bounded set."""
    ordered = sorted(rules)
    merged = []
    for axis, start, end in ordered:
        if merged and abs(merged[-1][0] - axis) <= 0.75 and start <= merged[-1][2] + 1.0:
            previous = merged[-1]
            merged[-1] = (previous[0], min(previous[1], start), max(previous[2], end))
        else:
            merged.append((axis, start, end))
    return merged


def _numeric_grid(page: Any, page_rect, drawings) -> dict | None:
    """Identify only a clear, populated numeric table grid for auto-attachment.

    Broad diagram/photo/scan candidates remain useful navigation, but do not
    qualify here.  Missing geometry or word positions yields no automatic page.
    This intentionally loses some real tables rather than attaching text pages.
    """
    if getattr(page, "rotation", 0):
        # Do not infer a table with mixed coordinate systems.  Rotated pages
        # remain in the broad directory and can be read explicitly.
        return None
    px0, py0, px1, py1 = page_rect
    page_width, page_height = px1 - px0, py1 - py0
    horizontal, vertical = [], []
    item_count = 0
    for drawing in drawings or ():
        if not isinstance(drawing, dict):
            continue
        rect = _rect_values(drawing.get("rect"))
        if rect is None or _is_page_frame(rect, page_rect):
            continue
        for item in drawing.get("items") or ():
            item_count += 1
            if item_count > 4000:
                return None
            if not item:
                continue
            lines = []
            if item[0] == "l" and len(item) >= 3:
                first, last = _rect_values(item[1]), _rect_values(item[2])
                if first is not None and last is not None:
                    lines = [(first[0], first[1], last[0], last[1])]
            elif item[0] == "re" and len(item) >= 2:
                box = _rect_values(item[1])
                if box is not None and not _is_page_frame(box, page_rect):
                    x0, y0, x1, y1 = box
                    lines = [(x0, y0, x1, y0), (x0, y1, x1, y1),
                             (x0, y0, x0, y1), (x1, y0, x1, y1)]
            for x0, y0, x1, y1 in lines:
                if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                    continue
                if abs(y1 - y0) <= 0.5 and abs(x1 - x0) >= page_width * 0.05:
                    horizontal.append(((y0 + y1) / 2, min(x0, x1), max(x0, x1)))
                elif abs(x1 - x0) <= 0.5 and abs(y1 - y0) >= page_height * 0.02:
                    vertical.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))
    horizontal, vertical = _merged_rules(horizontal), _merged_rules(vertical)
    if len(horizontal) > 128 or len(vertical) > 128:
        return None

    def crosses(h, v):
        return h[1] - 1 <= v[0] <= h[2] + 1 and v[1] - 1 <= h[0] <= v[2] + 1

    # Remove disconnected decoration.  A qualifying grid needs at least three
    # columns of vertical rules and four horizontal rules (six cells).
    for _ in range(4):
        next_h = [h for h in horizontal if sum(crosses(h, v) for v in vertical) >= 3]
        next_v = [v for v in vertical if sum(crosses(h, v) for h in next_h) >= 4]
        if next_h == horizontal and next_v == vertical:
            break
        horizontal, vertical = next_h, next_v
    if len(horizontal) < 4 or len(vertical) < 3:
        return None
    intersections = sum(crosses(h, v) for h in horizontal for v in vertical)
    if intersections < len(horizontal) * len(vertical) * 0.90:
        return None
    xs = sorted({round(v[0], 2) for v in vertical})
    ys = sorted({round(h[0], 2) for h in horizontal})
    if len(xs) < 3 or len(ys) < 4 or min(b - a for a, b in zip(xs, xs[1:])) < 8:
        return None
    if min(b - a for a, b in zip(ys, ys[1:])) < 8:
        return None
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    if ((x1 - x0) < page_width * 0.25 or (y1 - y0) < page_height * 0.08
            or (x1 - x0) * (y1 - y0) < page_width * page_height * 0.05):
        return None
    try:
        words = page.get_text("words")
    except (TypeError, AttributeError, RuntimeError, ValueError):
        return None
    populated, rows, columns, numeric_count, unit_count = set(), set(), set(), 0, 0
    for word in words if isinstance(words, (list, tuple)) else ():
        if not isinstance(word, (list, tuple)) or len(word) < 5:
            continue
        try:
            left, top, right, bottom = (float(value) for value in word[:4])
        except (TypeError, ValueError):
            continue
        cx, cy = (left + right) / 2, (top + bottom) / 2
        value = str(word[4]).strip()
        if not value or not (x0 + 1 < cx < x1 - 1 and y0 + 1 < cy < y1 - 1):
            continue
        col, row = bisect_right(xs, cx) - 1, bisect_right(ys, cy) - 1
        if not (0 <= col < len(xs) - 1 and 0 <= row < len(ys) - 1):
            continue
        populated.add((row, col)); rows.add(row); columns.add(col)
        numeric_count += bool(re.search(r"\d", value))
        unit_count += bool(re.search(r"(?:mm|cm|kN|MPa|kPa|kg|m[²³23]?|N·m|毫米|千牛|兆帕)", value))
    if len(populated) < 5 or len(rows) < 3 or len(columns) < 2 or numeric_count < 2:
        return None
    return {"auto_attach_eligible": True,
            "auto_attach_reasons": ["populated_numeric_table_grid"],
            "auto_attach_priority": min(numeric_count, 10) + min(unit_count, 5) * 2,
            "auto_attach_table_region": [(x0 - px0) / page_width, (y0 - py0) / page_height,
                                         (x1 - px0) / page_width, (y1 - py0) / page_height]}


def _inspect_page(page: Any, number: int, text: str) -> dict | None:
    page_rect = _rect_values(getattr(page, "rect", None))
    if page_rect is None:
        return None
    has_text = bool(text)
    meaningful_image, scanned = _image_signals(page, page_rect)
    try:
        drawings = page.get_drawings()
    except (AttributeError, RuntimeError, ValueError):
        drawings = []
    vector = _vector_signal(page, page_rect, drawings)
    numeric_grid = _numeric_grid(page, page_rect, drawings)
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
    if numeric_grid is not None:
        signals.append("table_grid")
    if not signals:
        return None
    return {
        "page": number,
        "has_text": has_text,
        "text_preview": _preview(text),
        "visual_candidate": True,
        "visual_signals": signals,
        "auto_attach_eligible": False,
        "auto_attach_reasons": [],
        "auto_attach_priority": 0,
        **(numeric_grid or {}),
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
