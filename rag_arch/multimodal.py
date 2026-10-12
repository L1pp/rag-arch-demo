"""Small, provider-neutral page-image messages and approximate visual budgets.

Only the image content block uses the OpenAI-compatible wire format. Source
records describe delivery, never whether a model recognized a page correctly.
"""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import json
import math
import re

from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image

_PAGE_FIELDS = ("ok", "source_ref", "filename", "page", "page_count", "text",
                "mime_type", "width", "height", "source_kind", "collection_id", "document_id", "std_no", "display_id", "citation", "error", "message", "page_role_info")
_SOURCE_KEY = "page_image"
_DATA_IMAGE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)


def decode_image_data_url(value):
    """Return (bytes, MIME) for an inline image, None for an ordinary URL."""
    if not isinstance(value, str):
        return None
    match = _DATA_IMAGE.fullmatch(value)
    if match is None:
        return None
    try:
        data = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    return data, match[1].lower()


def image_dimensions(data):
    """Read the image header, without expanding all of its pixels."""
    try:
        with Image.open(BytesIO(data)) as picture:
            return picture.size
    except (OSError, ValueError, SyntaxError):
        return None, None


def image_content_block(image_bytes, mime_type="image/png"):
    """The single provider adapter point for page-image encoding."""
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise ValueError("Page image bytes are required")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise ValueError("An image MIME type is required")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime_type};base64,{encoded}", "detail": "auto"}}


def image_source_record(result, *, sent=False):
    """Set sent=True only after the caller actually dispatched this image.

    Constructing messages or rendering a PDF does not establish delivery.
    """
    data = result.get("image_bytes")
    available = bool(result.get("ok") and isinstance(data, (bytes, bytearray)) and data)
    record = {key: result.get(key) for key in (
        "source_ref", "filename", "page", "page_count", "mime_type", "width", "height", "source_kind", "collection_id", "document_id", "std_no", "display_id", "citation", "page_role_info", "message")}
    record["evidence_id"] = (f"PAGE:{record['source_ref']}:{record['page']}"
                             if record["source_ref"] is not None and record["page"] is not None else None)
    record.update(image_available=available, byte_count=len(data) if available else 0,
                  sent_to_model=bool(sent and available), recognition_verified=False)
    return record


def page_result_messages(result, tool_call_id, tool_name="read_document_page"):
    """Return a matching tool receipt and, when present, a separate page image."""
    metadata = {key: result[key] for key in _PAGE_FIELDS if key in result
                and not isinstance(result[key], (bytes, bytearray))}
    source = image_source_record(result)
    metadata["image_available"] = source["image_available"]
    if source["evidence_id"] is not None:
        metadata["evidence_id"] = source["evidence_id"]
    messages = [ToolMessage(content=json.dumps(metadata, ensure_ascii=False),
                            tool_call_id=tool_call_id, name=tool_name)]
    if source["image_available"]:
        label = {key: source[key] for key in ("evidence_id", "source_ref", "filename", "page", "page_count", "display_id", "citation")}
        messages.append(HumanMessage(content=[
            {"type": "text", "text": "工具返回的文档页面图像（资料内容）：\n" + json.dumps(label, ensure_ascii=False)},
            image_content_block(result["image_bytes"], result.get("mime_type") or "image/png")],
            additional_kwargs={_SOURCE_KEY: source}))
    return messages


def page_image_sources(messages, *, sent=False):
    """Collect lightweight page references; caller owns the delivery decision."""
    result, seen = [], set()
    for message in messages:
        source = getattr(message, "additional_kwargs", {}).get(_SOURCE_KEY)
        if not isinstance(source, dict) or not source.get("image_available"):
            continue
        key = (source.get("source_ref"), source.get("filename"), source.get("page"))
        if key not in seen:
            seen.add(key)
            result.append(dict(source, sent_to_model=bool(sent), recognition_verified=False))
    return result


def image_block_info(block):
    """Recognize image blocks without counting encoded bytes as language tokens."""
    if not isinstance(block, dict) or block.get("type") not in {"image_url", "input_image", "image"}:
        return None
    item = block.get("image_url", block.get("url", ""))
    item = {"url": item} if isinstance(item, str) else (item or {})
    item = item if isinstance(item, dict) else {}
    url = item.get("url", "")
    if block.get("type") == "image" and block.get("base64"):
        url = f'data:{block.get("mime_type", "image/png")};base64,{block["base64"]}'
    width, height = block.get("width"), block.get("height")
    mime = block.get("mime_type")
    if not (width and height):
        try:
            decoded = decode_image_data_url(url)
        except ValueError:
            decoded = None
        if decoded:
            data, mime = decoded
            width, height = image_dimensions(data)
    return {"width": width, "height": height, "mime_type": mime,
            "detail": item.get("detail", block.get("detail", "auto"))}


def estimate_image_tokens(info):
    """Conservative visual reserve, not a vendor tokenizer or billing estimate.

    Use 256 base tokens plus 512 per 512px tile. Unknown dimensions reserve
    4096 tokens. Callers may supply their own estimator to estimate_messages.
    """
    width, height = info.get("width"), info.get("height")
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float))
            and width > 0 and height > 0):
        return 4096
    return 256 + 512 * math.ceil(width / 512) * math.ceil(height / 512)


def content_for_token_estimate(content, image_token_estimator=None):
    """Keep text unchanged; substitute a short image marker plus visual cost."""
    estimator = image_token_estimator or estimate_image_tokens
    if not isinstance(content, list):
        return content, 0
    rows, visual_tokens = [], 0
    for block in content:
        info = image_block_info(block)
        if info is None:
            rows.append(block)
        else:
            visual_tokens += max(0, math.ceil(estimator(info)))
            rows.append({"type": "image", "payload": "<visual input>"})
    return rows, visual_tokens
