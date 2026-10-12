"""Conservative parsing of standard identifiers found in user text.

This module only parses the identifier shape.  It does not infer a century,
resolve a withdrawn/current version, or declare two similar identifiers to be
the same standard.
"""
from __future__ import annotations

import re
import unicodedata


# Keep longer/slashed prefixes before their shorter forms.  TSG may carry a
# letter family (for example ``TSG D``), while its numeric part remains the
# identifier number.
_PREFIX = r"(?:"
_PREFIX += r"GB\s*/\s*T|GBT|GBZ|GB"
_PREFIX += r"|JGJ\s*/\s*T|JGJT|JGJ"
_PREFIX += r"|SY\s*/\s*T|SYT|SY"
_PREFIX += r"|NB\s*/\s*T|NBT|NB"
_PREFIX += r"|SH\s*/\s*T|SHT|SH"
_PREFIX += r"|HG\s*/\s*T|HGT|HG"
_PREFIX += r"|CJJ\s*/\s*T|CJJT|CJJ"
_PREFIX += r"|CECS\s*/\s*T|CECST|T\s*/\s*CECS|CECS"
_PREFIX += r"|DL\s*/\s*T|DLT|DL"
_PREFIX += r"|TSG(?:\s*[A-Z]{1,3})?"
_PREFIX += r")"

_DASH = r"[\-‐‑‒–—−－]"
_BASE_NUMBER = r"\d{2,7}"
_PART = r"\d+(?:\s*[.．]\s*\d+)*"

# A suffix is intentionally broad enough to retain malformed input such as
# ``-201`` or ``-20A4``.  The caller can decide whether it is acceptable.
_STANDARD = re.compile(
    rf"(?<![A-Za-z0-9])(?P<prefix>{_PREFIX})\s*(?P<number>{_BASE_NUMBER})"
    rf"(?:\s*[.．]\s*(?P<part>{_PART}))?"
    rf"(?:\s*(?:[:：]|{_DASH})\s*(?P<year>[0-9A-Za-z]{{0,8}}))?"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _compact_number(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).replace("．", ".")


def _prefix(value: str) -> str:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).upper()
    compact = compact.replace("／", "/")
    if compact == "GBT":
        return "GB/T"
    if compact == "JGJT":
        return "JGJ/T"
    if compact == "SYT":
        return "SY/T"
    if compact == "NBT":
        return "NB/T"
    if compact == "SHT":
        return "SH/T"
    if compact == "HGT":
        return "HG/T"
    if compact == "CJJT":
        return "CJJ/T"
    if compact == "CECST":
        return "CECS/T"
    if compact == "T/CECS":
        return compact
    if compact.startswith("TSG") and len(compact) > 3:
        return "TSG " + compact[3:]
    return compact


def _year_kind(value: str | None) -> str:
    if value is None:
        return "missing"
    if value.isdigit() and len(value) == 4:
        return "four_digit"
    if value.isdigit() and len(value) == 2:
        return "two_digit"
    return "malformed"


def parse_standard_references(query: str) -> list[dict]:
    """Parse supported standard identifiers from ``query``.

    The returned ``raw`` and ``year_text`` preserve the user's spelling.  A
    dotted suffix is exposed as ``part``; colon and dash suffixes are treated
    as year separators.  The ``normalized`` form only standardizes case,
    spaces, dots, and separators; it never converts a two-digit year to a
    four-digit year.
    """
    if not isinstance(query, str) or not query:
        return []
    results = []
    for match in _STANDARD.finditer(query):
        raw = match.group(0)
        prefix = _prefix(match.group("prefix"))
        number = _compact_number(match.group("number"))
        part = _compact_number(match.group("part"))
        year_text = match.group("year")
        year_kind = _year_kind(year_text)
        normalized = prefix + (number or "")
        if part is not None:
            normalized += "." + part
        if year_text is not None:
            normalized += "-" + year_text.upper()
        results.append({
            "raw": raw,
            "normalized": normalized,
            "prefix": prefix,
            "number": number,
            "part": part,
            "year_text": year_text,
            "year_kind": year_kind,
        })
    return results


__all__ = ["parse_standard_references"]
