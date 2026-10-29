"""Small, deterministic image budgets; none of these numbers imply coverage."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from .multimodal import page_image_sources
from .plan_review_context import estimate_messages

INITIAL_IMAGE_LIMIT = 1
INITIAL_IMAGE_BYTES = 1024 * 1024
INITIAL_IMAGE_TOKENS = 6000
RETAINED_IMAGE_LIMIT = 12
RETAINED_IMAGE_BYTES = 12 * 1024 * 1024
REQUEST_IMAGE_LIMIT = 8
REQUEST_IMAGE_BYTES = 4 * 1024 * 1024
REQUEST_IMAGE_TOKENS = 24000
PAGE_READ_LIMIT = 24
DIRECTORY_ROW_LIMIT = 2000


def image_pointer(source):
    fields = ("image_id", "evidence_id", "source_ref", "filename", "page", "region",
              "parent_page_id", "page_numbering", "coordinate_system")
    return HumanMessage(content="原页图本次未随请求重发；以下仅是回读定位，不是图像或识别证据：\n" +
        json.dumps({key: source[key] for key in fields if key in source}, ensure_ascii=False))


def bounded_image_messages(messages):
    """Keep recent images under per-request limits and replace others by pointers.

    Whole tool receipts stay intact. Older pixels may leave this request, but
    their page/region can still be requested and earlier delivery stays logged.
    """
    keep, skipped = set(), []
    byte_count = token_count = 0
    seen = set()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        sources = page_image_sources([message])
        if not sources:
            continue
        source = sources[0]
        identity = source.get("image_id") or source.get("evidence_id")
        size = source.get("byte_count", 0)
        size = size if type(size) is int and size >= 0 else REQUEST_IMAGE_BYTES + 1
        tokens = estimate_messages([message])
        fits = (identity not in seen and len(keep) < REQUEST_IMAGE_LIMIT
                and byte_count + size <= REQUEST_IMAGE_BYTES
                and token_count + tokens <= REQUEST_IMAGE_TOKENS)
        if fits:
            keep.add(index)
            seen.add(identity)
            byte_count += size
            token_count += tokens
        else:
            skipped.append(identity)
    result = []
    for index, message in enumerate(messages):
        source = page_image_sources([message])
        result.append(image_pointer(source[0]) if source and index not in keep else message)
    return result, {"image_count": len(keep), "image_bytes": byte_count,
                    "estimated_image_message_tokens": token_count,
                    "omitted_image_ids": list(reversed(skipped))}
