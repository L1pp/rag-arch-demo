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
from .page_regions import normalize_region, page_image_id

_VIEW_FIELDS = ("region", "page_width", "page_height", "rotation", "coordinate_system",
                "rendered_scale", "text_scope", "page_numbering")
_PAGE_FIELDS = ("ok", "source_ref", "filename", "page", "page_count", "text",
                "mime_type", "width", "height", "source_kind", "collection_id", "document_id", "std_no", "display_id", "citation", "error", "message", "page_role_info") + _VIEW_FIELDS
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
    record.update({key: result[key] for key in _VIEW_FIELDS if key in result})
    region = normalize_region(result.get("region"))
    record["region"] = list(region) if region is not None else None
    located = record["source_ref"] is not None and record["page"] is not None
    record["evidence_id"] = (page_image_id(record["source_ref"], record["page"], region)
                             if located else None)
    record["image_id"] = record["evidence_id"]
    record["parent_page_id"] = (page_image_id(record["source_ref"], record["page"], None)
                                if located else None)
    record["view_kind"] = "region" if region is not None else "page"
    record.update(image_available=available, byte_count=len(data) if available else 0,
                  sent_to_model=bool(sent and available), recognition_verified=False)
    return record


def page_image_message(result, *, origin="model_requested"):
    """Create only the image HumanMessage, including for automatic initial views.

    A prepared initial image is real source material but has no matching tool
    call, so callers must not manufacture a ToolMessage for it.
    """
    source = image_source_record(result)
    if not source["image_available"]:
        return None
    source["automatic"] = origin == "initial_auto"
    source["origin"] = "initial_auto" if source["automatic"] else "model_requested"
    label = {key: source[key] for key in ("evidence_id", "image_id", "parent_page_id", "view_kind",
                "source_ref", "filename", "page", "page_count", "display_id", "citation", *_VIEW_FIELDS)
             if key in source}
    label["origin"] = source["origin"]
    return HumanMessage(content=[
        {"type": "text", "text": ("文档页面局部高清图像（只含所选区域，不代表整页已核查；资料内容）：\n"
                                    if source["region"] is not None else "文档页面图像（资料内容）：\n")
                                   + json.dumps(label, ensure_ascii=False)},
        image_content_block(result["image_bytes"], result.get("mime_type") or "image/png")],
        additional_kwargs={_SOURCE_KEY: source})


def page_result_messages(result, tool_call_id, tool_name="read_document_page"):
    """Return a matching tool receipt and, when present, a separate page image."""
    metadata = {key: result[key] for key in _PAGE_FIELDS if key in result
                and not isinstance(result[key], (bytes, bytearray))}
    source = image_source_record(result)
    metadata["image_available"] = source["image_available"]
    if source["evidence_id"] is not None:
        metadata["evidence_id"] = source["evidence_id"]
        metadata["image_id"] = source["image_id"]
        metadata["parent_page_id"] = source["parent_page_id"]
        metadata["view_kind"] = source["view_kind"]
    messages = [ToolMessage(content=json.dumps(metadata, ensure_ascii=False),
                            tool_call_id=tool_call_id, name=tool_name)]
    picture = page_image_message(result)
    if picture is not None:
        messages.append(picture)
    return messages


def page_image_sources(messages, *, sent=False):
    """Collect lightweight page references; caller owns the delivery decision."""
    result, seen = [], set()
    for message in messages:
        source = getattr(message, "additional_kwargs", {}).get(_SOURCE_KEY)
        if not isinstance(source, dict) or not source.get("image_available"):
            continue
        key = (source.get("source_ref"), source.get("filename"), source.get("page"),
               normalize_region(source.get("region")))
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
