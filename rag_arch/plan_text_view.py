"""Lightweight plan text for the model; unit mapping stays in the program.

Anchors mark natural paragraph groups, pages or tables. They locate quotes and
are not a batching mechanism. Original units, locators and character ranges are
unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata


PLAN_TEXT_VIEW_VERSION = "anchored-continuous-text-v1"
ANCHOR_GROUP_CHARS = 2000
_SOURCE_REF = re.compile(r"^\[?S(\d{1,6})\]?$", re.I)


def source_ref_label(index: int) -> str:
    if type(index) is not int or index < 1:
        raise ValueError("锚点序号须为正整数。")
    return f"S{index:03d}"


def normalize_source_ref(value) -> str | None:
    """Accept S001, [S001] or a unit_id that is itself an anchor."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    match = _SOURCE_REF.fullmatch(text)
    if match is None:
        return None
    return source_ref_label(int(match.group(1)))


def _boundary_key(locator):
    if not isinstance(locator, dict):
        return ("para",)
    table = locator.get("table")
    if table is not None:
        page = locator.get("page", locator.get("page_start"))
        return ("table", table, page)
    page = locator.get("page", locator.get("page_start"))
    if page is not None:
        return ("page", page)
    return ("para",)


def group_plan_units(units, *, target_chars: int = ANCHOR_GROUP_CHARS) -> list[list[dict]]:
    """Group complete source units without splitting a table or a natural unit."""
    if type(target_chars) is not int or target_chars < 1:
        raise ValueError("target_chars须为正整数。")
    groups, current, chars, bound = [], [], 0, None
    for unit in units:
        if not isinstance(unit, dict):
            continue
        next_bound = _boundary_key(unit.get("locator"))
        same_table = bound is not None and bound[0] == "table" and next_bound == bound
        leave_table = bound is not None and bound[0] == "table" and not same_table
        enter_table = bool(current) and next_bound[0] == "table" and next_bound != bound
        page_break = bool(current) and bound is not None and bound[0] == "page" and next_bound[0] == "page" and next_bound != bound
        full_enough = bool(current) and chars >= target_chars
        if current and not same_table and (leave_table or enter_table or page_break or full_enough):
            groups.append(current)
            current, chars, bound = [], 0, None
        current.append(unit)
        chars += len(unit.get("text") or "")
        if bound is None or next_bound[0] == "table":
            bound = next_bound
        elif bound[0] != "table" and next_bound[0] == "page":
            bound = next_bound
    if current:
        groups.append(current)
    return groups


def document_anchor_index(units) -> dict:
    """Map every unit_id to a stable document-level source_ref."""
    groups = group_plan_units(units)
    by_unit, by_ref = {}, {}
    for index, group in enumerate(groups, 1):
        ref = source_ref_label(index)
        ids = []
        for unit in group:
            uid = unit.get("unit_id")
            if isinstance(uid, str) and uid and uid not in by_unit:
                by_unit[uid] = ref
                ids.append(uid)
        by_ref[ref] = ids
    return {"by_unit": by_unit, "by_ref": by_ref, "groups": groups}


def join_unit_texts(units) -> tuple[str, list[dict]]:
    """Concatenate unit bodies in order; record rendered-to-unit character spans."""
    pieces, spans, pos = [], [], 0
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get("unit_id"), str):
            continue
        text = unit.get("text") if isinstance(unit.get("text"), str) else ""
        if pieces and not pieces[-1].endswith("\n") and text and not text.startswith("\n"):
            pieces.append("\n")
            pos += 1
        start = pos
        pieces.append(text)
        pos += len(text)
        origin = unit.get("text_start", 0)
        origin = origin if type(origin) is int and origin >= 0 else 0
        spans.append({"unit_id": unit["unit_id"], "rendered_start": start, "rendered_end": pos,
                      "text_start": origin, "text_end": origin + len(text)})
    return "".join(pieces), spans


def stamp_source_refs(units, *, anchors=None) -> list[dict]:
    """Copy units and fill source_ref from a document-level index when missing."""
    index = anchors if isinstance(anchors, dict) else document_anchor_index(units)
    by_unit = index.get("by_unit") if isinstance(index, dict) else {}
    stamped = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        row = deepcopy(unit)
        if not isinstance(row.get("source_ref"), str) or not row["source_ref"].strip():
            ref = by_unit.get(row.get("unit_id"))
            if isinstance(ref, str) and ref:
                row["source_ref"] = ref
        stamped.append(row)
    return stamped


def coverage_units(units) -> list[dict]:
    """Keep only fields the local coverage ledger needs."""
    rows = []
    for unit in units or []:
        if not isinstance(unit, dict) or not isinstance(unit.get("unit_id"), str):
            continue
        text = unit.get("text") if isinstance(unit.get("text"), str) else ""
        start = unit.get("text_start", 0)
        start = start if type(start) is int and start >= 0 else 0
        end = unit.get("text_end", start + len(text))
        end = end if type(end) is int and end >= start else start + len(text)
        row = {"unit_id": unit["unit_id"], "text": text, "text_start": start, "text_end": end}
        for key in ("complete_unit", "unit_text_chars", "ordinal", "locator", "source_ref"):
            if key in unit:
                row[key] = deepcopy(unit[key])
        rows.append(row)
    return rows


def render_anchored_text(units, *, anchors=None) -> str:
    """Render continuous text with short anchors; empty groups are omitted."""
    if not units:
        return ""
    index = anchors if isinstance(anchors, dict) else document_anchor_index(units)
    by_unit = index.get("by_unit") if isinstance(index, dict) else {}
    blocks, current_ref, current = [], None, []

    def flush():
        nonlocal current, current_ref
        body, _ = join_unit_texts(current)
        if body:
            blocks.append(f"[{current_ref}]\n{body}" if current_ref else body)
        current = []

    for unit in units:
        if not isinstance(unit, dict):
            continue
        ref = unit.get("source_ref")
        if not isinstance(ref, str) or not ref.strip():
            ref = by_unit.get(unit.get("unit_id"))
        if current and ref != current_ref:
            flush()
        current_ref = ref
        current.append(unit)
    if current:
        flush()
    text = "\n".join(blocks)
    return text if not text or text.endswith("\n") else text + "\n"


def rendered_span_to_units(spans, start: int, end: int) -> list[dict]:
    """Map a rendered [start, end) onto original unit character ranges."""
    if type(start) is not int or type(end) is not int or end <= start:
        return []
    covered = []
    for span in spans:
        left = max(start, span["rendered_start"])
        right = min(end, span["rendered_end"])
        if right <= left:
            continue
        shift = span["text_start"] - span["rendered_start"]
        covered.append({"unit_id": span["unit_id"], "start": left + shift, "end": right + shift})
    return covered


def quote_matches_in_units(units, quote: str, provided_ranges=None) -> list[dict]:
    """Exact then whitespace-normalized matches; never choose among duplicates."""
    if not isinstance(quote, str) or not quote.strip():
        return []
    matches = []
    seen = set()

    def add(unit_id, span, *, cross=None):
        key = (unit_id, span["start"], span["end"], tuple(
            (row["unit_id"], row["start"], row["end"]) for row in (cross or ())))
        if key in seen:
            return
        seen.add(key)
        item = {"unit_id": unit_id, "start": span["start"], "end": span["end"],
                "quote": span["quote"], "whitespace_normalized": span["whitespace_normalized"]}
        if cross and len(cross) > 1:
            item["span_units"] = list(cross)
        matches.append(item)

    for unit in units:
        uid = unit.get("unit_id")
        body = unit.get("text")
        origin = unit.get("text_start", 0)
        origin = origin if type(origin) is int and origin >= 0 else 0
        if not isinstance(uid, str) or not isinstance(body, str):
            continue
        for span in _normalized_quote_spans(body, quote):
            absolute = {"start": origin + span["start"], "end": origin + span["end"],
                        "quote": span["quote"], "whitespace_normalized": span["whitespace_normalized"]}
            if provided_ranges is not None and not _contains(provided_ranges, uid, absolute):
                continue
            add(uid, absolute)

    joined, rendered = join_unit_texts(units)
    for span in _normalized_quote_spans(joined, quote):
        cross = rendered_span_to_units(rendered, span["start"], span["end"])
        if provided_ranges is not None and any(
                not _contains(provided_ranges, row["unit_id"], row) for row in cross):
            continue
        if len(cross) <= 1:
            continue
        first = cross[0]
        add(first["unit_id"], {"start": first["start"], "end": first["end"],
                               "quote": span["quote"], "whitespace_normalized": span["whitespace_normalized"]},
            cross=cross)
    return matches


def _contains(ranges, unit_id, span):
    return any(left <= span["start"] and span["end"] <= right
               for left, right in (ranges or {}).get(unit_id, []))


def _normalized_quote_spans(body: str, quote: str) -> list[dict]:
    if not isinstance(body, str) or not isinstance(quote, str) or not quote.strip():
        return []
    start = body.find(quote)
    if start >= 0:
        spans = [{"start": start, "end": start + len(quote), "quote": quote, "whitespace_normalized": False}]
        cursor = start + 1
        while True:
            found = body.find(quote, cursor)
            if found < 0:
                break
            spans.append({"start": found, "end": found + len(quote), "quote": quote,
                          "whitespace_normalized": False})
            cursor = found + 1
        if len(spans) > 1:
            return spans
        return spans
    chars, positions = [], []
    for pos, char in enumerate(body):
        for normalized in unicodedata.normalize("NFKC", char):
            if not normalized.isspace():
                chars.append(normalized)
                positions.append(pos)
    needle = "".join(
        normalized
        for char in quote
        for normalized in unicodedata.normalize("NFKC", char)
        if not normalized.isspace()
    )
    if not needle:
        return []
    haystack = "".join(chars)
    spans, cursor = [], 0
    while True:
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        end_index = start + len(needle) - 1
        left, right = positions[start], positions[end_index] + 1
        actual = body[left:right]
        spans.append({"start": left, "end": right, "quote": actual,
                      "whitespace_normalized": actual != quote})
        cursor = start + 1
    return spans
