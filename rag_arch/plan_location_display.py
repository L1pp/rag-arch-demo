"""Conservative display of model-written plan page labels.

The caller retains the model's original answer/JSON. This pure helper returns
only derived display text and a small change log; it never changes a unit,
locator, quotation, file, or acquisition record. Unit IDs are identifiers, not
page numbers. Unbound prose is deliberately outside this helper's scope.
"""
from __future__ import annotations

from collections.abc import Mapping
import re


DISPLAY_VERSION = "plan-location-display-v2"
_PROSE_FIELDS = {"answer", "title", "description", "conditions", "recommendation", "limitations"}
_UNIT = r"[A-Za-z0-9_-]+/u[0-9]+"
_FOLLOW_UNIT = r"(?:[A-Za-z0-9_-]+/)?u[0-9]+"
_UNIT_LIST = _UNIT + r"(?:\s*[、,，]\s*" + _FOLLOW_UNIT + r")*"
_PAGE = (
    r"(?P<page_kind>PDF物理|PDF|物理|印刷|页脚|正文)?\s*第"
    r"(?P<pages>[0-9]+(?:\s*[、,，]\s*[0-9]+)*)\s*页"
)
_TAG = r"(?:(?:unit(?:_ids?|s)?|单元(?:ID)?)\s*[:：]?\s*)?"
_PAGE_FIRST = re.compile(
    _PAGE + r"\s*(?P<open>[（(])\s*" + _TAG
    + r"(?P<units>" + _UNIT_LIST + r")\s*(?P<close>[）)])", re.I
)
_UNIT_FIRST = re.compile(
    r"(?P<units>" + _UNIT_LIST + r")\s*(?P<open>[（(])\s*"
    + _PAGE + r"\s*(?P<close>[）)])", re.I
)
_PAGE_MENTION = re.compile(
    r"(?:PDF物理|PDF|物理|印刷|页脚|正文)?\s*第[0-9]+(?:\s*[、,，]\s*[0-9]+)*\s*页",
    re.I,
)
_NON_ASSERTIVE = re.compile(
    r"并非|并不|不是|不在|不能|不应|不要|不正确|不准确|勿|错误|误写|误标|误称|误记|写错|错页|"
    r"更正|纠正|修正|改写|改为|改成|原称|旧报告|原报告|候选报告|旧答复|"
    r"上一轮|上轮|此前|曾经|转述|是否|假设|如果|"
    r"(?:模型|原文|有人|他|她|报告|复核|答复)(?:曾|中)?(?:写|称|说|记|标|提到)"
)
_FOREIGN_DOCUMENT = re.compile(
    r"(?:(?:规范|标准)(?:原文|文件|PDF)?|(?:GB|JGJ)[0-9 ./_-]+)\s*$", re.I
)


def _positive_integer(value):
    return type(value) is int and value > 0


def _physical_bounds(locator):
    if not isinstance(locator, Mapping):
        return None
    page_kind = locator.get("page_kind", locator.get("page_type"))
    if page_kind not in (None, "physical", "pdf_physical"):
        return None
    page = locator.get("page")
    start, end = locator.get("page_start"), locator.get("page_end")
    if page is not None:
        if not _positive_integer(page):
            return None
        # True == 1 and 4.0 == 4, so duplicate fields need the same strict check.
        if start is not None and not _positive_integer(start):
            return None
        if end is not None and not _positive_integer(end):
            return None
        if start is not None or end is not None:
            if start != page or end not in (None, page):
                return None
        return page, page
    if not _positive_integer(start):
        return None
    end = start if end is None else end
    if not _positive_integer(end) or end < start:
        return None
    return start, end


def format_physical_locator(locator) -> str | None:
    """Format actual physical-page metadata; return None for DOCX/unknown data."""
    bounds = _physical_bounds(locator)
    if bounds is None:
        return None
    first, last = bounds
    number = str(first) if first == last else f"{first}–{last}"
    return f"物理第{number}页"


def _protected_ranges(text):
    """Protect source quotations and Markdown code/quote blocks, conservatively."""
    ranges = []
    tick = chr(96)
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = re.match(r"(" + re.escape(tick) + r"{3,}|~{3,})", stripped)
        if fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1]:
                ranges.append((fence[2], offset + len(line)))
                fence = None
        elif marker:
            fence = marker.group(1)[0], len(marker.group(1)), offset
        elif stripped.startswith(">"):
            ranges.append((offset, offset + len(line)))
        offset += len(line)
    if fence is not None:
        ranges.append((fence[2], len(text)))

    # Do not inspect quoted original wording to decide whether its page is wrong.
    quotes = {"“": "”", "‘": "’", '"': '"', "'": "'", "「": "」", "『": "』", "《": "》"}
    opened = None
    position = 0
    while position < len(text):
        containing = next(((start, end) for start, end in ranges if start <= position < end), None)
        if containing:
            position = containing[1]
            continue
        char = text[position]
        if opened is not None:
            if char == opened[1] and (position == 0 or text[position - 1] != "\\"):
                ranges.append((opened[0], position + 1))
                opened = None
        elif char == tick:
            count = 1
            while position + count < len(text) and text[position + count] == tick:
                count += 1
            closing = text.find(tick * count, position + count)
            end = len(text) if closing < 0 else closing + count
            ranges.append((position, end))
            position = end
            continue
        elif char in quotes:
            opened = position, quotes[char]
        position += 1
    if opened is not None:
        ranges.append((opened[0], len(text)))
    return ranges


def _context_reason(text, start, end):
    left = max((text.rfind(separator, 0, start) for separator in ("\n", "。", "！", "？", ";", "；")), default=-1) + 1
    following = [position for separator in ("\n", "。", "！", "？", ";", "；")
                 if (position := text.find(separator, end)) >= 0]
    right = min(following) if following else len(text)
    prefix = text[max(left, start - 60):start]
    suffix = text[end:min(right, end + 45)]
    if _NON_ASSERTIVE.search(prefix + suffix):
        return "negated_or_reported_wording"
    if _FOREIGN_DOCUMENT.search(prefix):
        return "different_document_context"
    return None


def normalize_plan_location_display(text: str, units: Mapping, *, field: str = "answer") -> dict:
    """Render explicitly bound page/unit labels using authoritative locators.

    Supported labels are page-first or unit-first, including equal-length page
    and unit lists from one recorded source_path. Missing page-numbering type
    produces an ambiguity, not a claim that a printed page is wrong. Explicit
    printed-page labels, source quotations, negation/reporting, unknown units,
    mixed sources, and unequal lists are retained without substitution.

    The units argument is the existing unit_id to unit/reference mapping.
    Each value has a locator and, for list normalization, a nonempty source_path.
    This helper does not assert that a unit was read or supports a finding.
    """
    if not isinstance(text, str) or not isinstance(units, Mapping):
        raise TypeError("text须为字符串，units须为实际unit_id到原文定位记录的映射。")
    result = {"display_text": text, "changes": [], "diagnostics": [], "skipped": [],
              "display_version": DISPLAY_VERSION}
    if field not in _PROSE_FIELDS:
        result["skipped"].append({"reason": "protected_field", "field": field})
        return result
    protected = _protected_ranges(text)
    matches = sorted(
        [match for pattern in (_PAGE_FIRST, _UNIT_FIRST) for match in pattern.finditer(text)],
        key=lambda match: (match.start(), match.end()),
    )
    replacements = []
    previous_end = -1
    for match in matches:
        start, end = match.span()
        if start < previous_end:
            result["skipped"].append({"start": start, "end": end, "reason": "overlapping_binding"})
            continue
        previous_end = end
        if any(a < end and start < b for a, b in protected):
            result["skipped"].append({"start": start, "end": end, "reason": "quoted_or_code_source"})
            continue
        context_reason = _context_reason(text, start, end)
        if context_reason:
            result["skipped"].append({"start": start, "end": end, "reason": context_reason})
            continue
        if (match["open"], match["close"]) not in {("(", ")"), ("（", "）")}:
            result["skipped"].append({"start": start, "end": end, "reason": "unbalanced_label"})
            continue
        kind = (match["page_kind"] or "").upper()
        if kind in {"印刷", "页脚", "正文"}:
            result["skipped"].append({"start": start, "end": end, "reason": "explicit_printed_page_preserved"})
            continue
        tokens = re.findall(_FOLLOW_UNIT, match["units"])
        prefix = tokens[0].split("/", 1)[0]
        ids = [token if "/" in token else prefix + "/" + token for token in tokens]
        claimed = [int(number) for number in re.findall(r"[0-9]+", match["pages"])]
        if len(ids) != len(claimed) or len(ids) != len(set(ids)):
            result["skipped"].append({"start": start, "end": end, "reason": "unequal_or_duplicate_binding",
                                      "unit_ids": ids, "claimed_pages": claimed})
            continue
        authorities = []
        for uid in ids:
            entry = units.get(uid)
            if not isinstance(entry, Mapping) or entry.get("unit_id", uid) != uid:
                break
            bounds = _physical_bounds(entry.get("locator"))
            if bounds is None:
                break
            authorities.append({"unit_id": uid, "page_start": bounds[0], "page_end": bounds[1],
                                "source_path": entry.get("source_path"),
                                "label": uid + " · " + format_physical_locator(entry["locator"])})
        if len(authorities) != len(ids):
            result["skipped"].append({"start": start, "end": end, "reason": "unknown_unit_or_physical_locator",
                                      "unit_ids": ids})
            continue
        sources = [entry["source_path"] for entry in authorities]
        if len(ids) > 1 and (any(not isinstance(source, str) or not source for source in sources)
                             or len(set(sources)) != 1):
            result["skipped"].append({"start": start, "end": end, "reason": "multiple_or_unknown_sources",
                                      "unit_ids": ids})
            continue
        canonical = "；".join(entry["label"] for entry in authorities)
        outside = [uid for uid, page, entry in zip(ids, claimed, authorities)
                   if not entry["page_start"] <= page <= entry["page_end"]]
        original = text[start:end]
        label = _PAGE_MENTION.search(original).group(0).strip()
        annotation = {"start": start, "end": end, "original": original, "unit_ids": ids,
                      "claimed_pages": claimed, "authoritative_locations": authorities}
        if not kind:
            annotation["code"] = "ambiguous_page_numbering"
            result["diagnostics"].append(dict(annotation))
            if outside:
                canonical += f"（模型原标“{label}”未注明页码口径）"
        elif outside:
            annotation["code"] = "physical_page_conflict"
            annotation["conflicting_unit_ids"] = outside
            result["diagnostics"].append(dict(annotation))
            canonical += f"（模型原标“{label}”与物理定位不符）"
        else:
            annotation["code"] = "physical_location_confirmed"
        annotation["replacement"] = canonical
        result["changes"].append(annotation)
        replacements.append((start, end, canonical))

    for start, end, replacement in reversed(replacements):
        result["display_text"] = result["display_text"][:start] + replacement + result["display_text"][end:]
    return result


def _report_value(report, key, default=None):
    return report.get(key, default) if isinstance(report, Mapping) else getattr(report, key, default)


def source_location_index(report, document):
    """Index only model-mentioned units using actual document metadata."""
    fields = [_report_value(report, "answer", ""), *_report_value(report, "limitations", [])]
    for finding in _report_value(report, "findings", []):
        fields.extend(finding.get(key, "") for key in ("title", "description", "recommendation"))
        fields.extend(finding.get("conditions", []))
        fields.extend(ref.get("unit_id", "") for ref in finding.get("plan_evidence", []))
    mentioned = set()
    for value in fields:
        if not isinstance(value, str):
            continue
        for token in re.findall(_FOLLOW_UNIT, value):
            uid = token if "/" in token else document.document_id + "/" + token
            if uid in document.by_id:
                mentioned.add(uid)
    return {uid: {"unit_id": uid, "locator": dict(document.by_id[uid].get("locator", {})),
                  "source_path": document.by_id[uid].get("source_path")}
            for uid in sorted(mentioned)}


def build_report_display(report, *, units=None):
    """Derive UI/Markdown prose without overwriting model outputs or quotes."""
    if units is None:
        units = dict((_report_value(report, "runtime", {}) or {}).get("source_location_index", {}))
        for finding in _report_value(report, "findings", []):
            for ref in finding.get("plan_evidence", []):
                uid = ref.get("unit_id")
                if uid and uid not in units:
                    units[uid] = ref
    changes, diagnostics = [], []
    def render(value, field, path):
        value = value if isinstance(value, str) else ""
        result = normalize_plan_location_display(value, units, field=field)
        changes.extend({"field": path, **item} for item in result["changes"])
        diagnostics.extend({"field": path, **item} for item in result["diagnostics"])
        return result["display_text"]
    output = {"answer": render(_report_value(report, "answer", ""), "answer", "answer"),
              "findings": [], "limitations": [], "display_version": DISPLAY_VERSION}
    for index, finding in enumerate(_report_value(report, "findings", [])):
        copy = dict(finding)
        for key in ("title", "description", "recommendation"):
            copy[key] = render(finding.get(key), key, f"findings/{index}/{key}")
        copy["conditions"] = [render(value, "conditions", f"findings/{index}/conditions/{number}")
                              for number, value in enumerate(finding.get("conditions", []))]
        output["findings"].append(copy)
    output["limitations"] = [render(value, "limitations", f"limitations/{index}")
                             for index, value in enumerate(_report_value(report, "limitations", []))]
    output["changes"], output["diagnostics"] = changes, diagnostics
    return output
