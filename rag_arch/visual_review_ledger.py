"""Bounded visual-source bookkeeping, never a claim of correct recognition.

Acquisition, attempted requests, returned model calls and exact citations are
separate observations. The ledger retains no messages, image bytes or encoding.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .multimodal import page_image_sources
from .page_regions import normalize_region


_SOURCE_FIELDS = {
    "image_id", "evidence_id", "parent_page_id", "source_ref", "filename", "page", "page_count",
    "width", "height", "byte_count", "mime_type", "source_kind", "collection_id", "document_id",
    "std_no", "display_id", "citation", "region", "page_width", "page_height", "rotation",
    "coordinate_system", "rendered_scale", "text_scope", "page_numbering", "view_kind", "origin",
}
_GEOMETRY_FIELDS = {
    "width", "height", "byte_count", "mime_type", "rendered_scale", "page_width", "page_height",
    "rotation", "coordinate_system", "page_numbering", "text_scope", "region",
}
_OMIT_KEYS = {"text", "prompt", "answer", "image_bytes", "image_url", "url", "base64",
              "authorization", "cookie", "password", "api_key", "access_key", "secret", "token",
              "text_preview", "preview", "source_path", "path"}


def _safe(value, *, depth=0):
    """Keep small JSON metadata, omitting payloads even in injected fixtures."""
    if depth > 5 or isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if value is None or isinstance(value, bool) or type(value) is int:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if "data:" in value.lower() or "base64" in value.lower():
            return None
        return value[:512]
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth=depth + 1) for item in value[:1000]
                if not isinstance(item, (bytes, bytearray, memoryview))]
    if isinstance(value, dict):
        return {key: _safe(item, depth=depth + 1) for key, item in list(value.items())[:1000]
                if isinstance(key, str) and key.lower() not in _OMIT_KEYS
                and not isinstance(item, (bytes, bytearray, memoryview))}
    return None


class VisualReviewLedger:
    def __init__(self):
        self.initial_selection = {}
        self._images = {}
        self._attempts = []
        self._read_failures = []

    def acquire(self, source, automatic=False):
        if not isinstance(source, dict):
            raise ValueError("页图来源须为对象。")
        identifier = source.get("image_id") or source.get("evidence_id")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 512:
            raise ValueError("页图来源缺少有效图像编号。")
        metadata = {key: _safe(value) for key, value in source.items() if key in _SOURCE_FIELDS}
        previous = self._images.get(identifier, {})
        available = bool(source.get("image_available", source.get("acquired", False)))
        image = {**previous, **metadata, "image_id": identifier,
                 "evidence_id": metadata.get("evidence_id") or identifier,
                 "automatic": bool(automatic or source.get("automatic") or previous.get("automatic")),
                 "acquired": bool(available or previous.get("acquired")),
                 "request_attempted": bool(previous.get("request_attempted")),
                 "response_received": bool(previous.get("response_received")),
                 "sent_to_model": bool(previous.get("response_received")),
                 "recognition_verified": False}
        if available:
            image["latest_acquired_view"] = deepcopy(metadata)
        if previous.get("response_received"):
            # A newly rendered sharper image has not necessarily been sent or
            # returned successfully. Keep public delivery geometry tied to the
            # last successful request, rather than mixing it with that render.
            for key in _GEOMETRY_FIELDS:
                if key in previous:
                    image[key] = previous[key]
                else:
                    image.pop(key, None)
        self._images[identifier] = image
        return identifier

    def attempt(self, messages, request_index):
        """Record an invocation attempt, not proof that a server received it."""
        sources = page_image_sources(messages, sent=False)
        if not sources:
            return None
        if type(request_index) is not int or request_index < 0:
            raise ValueError("模型请求序号须为非负整数。")
        image_ids = []
        for source in sources:
            identifier = self.acquire(source, automatic=source.get("automatic", False))
            self._images[identifier]["request_attempted"] = True
            image_ids.append(identifier)
        attempt_id = len(self._attempts) + 1
        self._attempts.append({"attempt_id": attempt_id, "request_index": request_index,
                               "image_ids": image_ids, "response_received": False,
                               "images": [{key: _safe(source[key]) for key in (
                                   "image_id", "evidence_id", "source_ref", "page", "region",
                                   "width", "height", "byte_count", "mime_type", "rendered_scale", "page_width",
                                   "page_height", "rotation", "coordinate_system", "page_numbering", "text_scope")
                                   if key in source} for source in sources],
                               "status": "unconfirmed"})
        return attempt_id

    def responded(self, attempt_id):
        """Called only after a successful AIMessage from this invocation."""
        if attempt_id is None:
            return
        if type(attempt_id) is not int or not 1 <= attempt_id <= len(self._attempts):
            raise ValueError("未知页图请求尝试编号。")
        attempt = self._attempts[attempt_id - 1]
        if attempt["response_received"]:
            return
        attempt["response_received"] = True
        attempt["status"] = "response_received"
        views = {view.get("image_id") or view.get("evidence_id"): view for view in attempt["images"]}
        for identifier in attempt["image_ids"]:
            image = self._images[identifier]
            view = views[identifier]
            for key in _GEOMETRY_FIELDS:
                if key in view:
                    image[key] = deepcopy(view[key])
                else:
                    image.pop(key, None)
            image["delivered_view"] = dict(deepcopy(view), attempt_id=attempt_id,
                                            request_index=attempt["request_index"])
            image["response_received"] = True
            image["sent_to_model"] = True

    def failed_read(self, source_ref, page, region, error_type):
        try:
            region = normalize_region(region)
        except ValueError:
            region = None
        error_type = error_type if isinstance(error_type, str) and error_type.isidentifier() else "ReadError"
        self._read_failures.append({"source_ref": _safe(source_ref),
                                    "page": page if type(page) is int and page > 0 else None,
                                    "region": list(region) if region is not None else None,
                                    "error_type": error_type[:80]})

    def snapshot(self, cited_ids=()):
        cited = {cited_ids} if isinstance(cited_ids, str) else set(cited_ids or ())
        images = [dict(image, cited=identifier in cited, recognition_verified=False)
                  for identifier, image in self._images.items()]
        return deepcopy({"images": images, "attempts": self._attempts,
                         "read_failures": self._read_failures,
                         "initial_selection": _safe(self.initial_selection),
                         "meaning": "请求尝试、成功返回和引用仅记录来源使用，均不证明识别正确或全部图表已核查。"})
