"""Strict parsing for a model response that is one JSON object.

The parser performs only syntax-level normalization.  It accepts surrounding
whitespace and one complete Markdown code fence, then delegates JSON/schema
validation to the caller.  It deliberately does not search for an object in
prose, repair a truncated response, or infer/modify any fields.
"""
from __future__ import annotations

import json
import re
from typing import Any


# This identifier is written into new Agent/experiment provenance so that a
# later run can be distinguished from historical results produced by the old
# literal-LF-only fence handling.
JSON_OBJECT_PARSER_VARIANT = "strict-single-json-object-v1"

_CODE_FENCE = re.compile(
    r"\A```[ \t]*(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


def parse_single_json_object(text: str) -> dict[str, Any]:
    """Parse exactly one JSON object from *text*.

    Accepted input is a JSON object surrounded by whitespace, or the same
    object enclosed by a complete ``````, `````json`` or `````JSON`` Markdown
    fence.  Any prose, a second JSON value, an incomplete fence, or a
    non-object JSON value is rejected.  Object field/schema checks remain the
    responsibility of the caller.
    """
    if not isinstance(text, str):
        raise TypeError("JSON response must be text")

    candidate = text.strip()
    fenced = _CODE_FENCE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group("body").strip()

    # json.loads consumes a complete JSON value (apart from JSON whitespace),
    # so explanatory text and adjacent/multiple objects fail closed here.
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("JSON response must be an object")
    return payload

