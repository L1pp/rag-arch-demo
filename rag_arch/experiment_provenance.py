"""Small, non-secret retrieval snapshots for future experiment records.

Only named public model/size attributes are inspected. ``present`` means an
object was injected, not that its remote service or credentials were verified.
Unknown attributes stay null; no settings, endpoints or provider objects leave
this module. A cell's unused components are omitted from configuration identity
by replacing their details with ``not_used`` and null fields.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

from .embedding_cache import CachedEmbeddings


SCHEMA_VERSION = "retrieval-config-v1"
_MISSING = object()
_AVAILABILITY = {"present", "unavailable", "unknown"}


def _attribute(value, name):
    try:
        return getattr(value, name, _MISSING)
    except Exception:
        # Property failures do not turn an unavailable metadata field into a
        # failed experiment or disclose an exception containing credentials.
        return _MISSING


def _model(value):
    # Accept literal model identifiers only, never stringify an arbitrary
    # object or accept URL/auth/query syntax in this metadata field.
    return value if isinstance(value, str) and re.fullmatch(r"[\w./-]{1,200}", value) else None


def _positive(value, *, integer=False):
    if type(value) is int and value > 0:
        return value
    if not integer and type(value) is float and math.isfinite(value) and value > 0:
        return value
    return None


def _availability(provider):
    return "unknown" if provider is _MISSING else "unavailable" if provider is None else "present"


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def sanitize_retrieval_config(snapshot) -> dict:
    """Rebuild even caller-supplied snapshots using only the allowed fields."""
    source = _mapping(snapshot)
    embedding, reranker = _mapping(source.get("embedding")), _mapping(source.get("reranker"))

    def provider(value, *, embedding=False):
        availability = value.get("availability")
        if not isinstance(availability, str) or availability not in _AVAILABILITY:
            availability = "unknown"
        result = {"availability": availability, "model": _model(value.get("model"))}
        if not embedding and value.get("provider") in ("local", "bailian"):
            result["provider"] = value["provider"]
            if value["provider"] == "local":
                result.update({"revision": _model(value.get("revision")),
                               "device": value.get("device") if isinstance(value.get("device"), str)
                                   and re.fullmatch(r"cpu|cuda(?::[0-9]+)?", value["device"]) else None,
                               "dtype": value.get("dtype") if isinstance(value.get("dtype"), str)
                                   and value["dtype"] in {"float32", "float16", "bfloat16"} else None,
                               "max_length": _positive(value.get("max_length"), integer=True),
                               "batch_size": _positive(value.get("batch_size"), integer=True)})
        if embedding:
            result["dimensions"] = _positive(value.get("dimensions"), integer=True)
        if availability == "unavailable":
            result.update({key: None for key in result if key != "availability"})
        return result

    def parameter(name, *, integer=False):
        raw = _mapping(source.get(name))
        value = _positive(raw.get("value"), integer=integer) if raw.get("status") == "known" else None
        return {"status": "known" if value is not None else "unknown", "value": value}

    return {"schema_version": SCHEMA_VERSION,
            "embedding": provider(embedding, embedding=True), "reranker": provider(reranker),
            "candidate_pool": parameter("candidate_pool", integer=True),
            "rrf_constant": parameter("rrf_constant")}


def describe_retrieval(index) -> dict:
    """Read whitelisted index/provider attributes, without network or defaults."""
    embeddings, reranker = _attribute(index, "embeddings"), _attribute(index, "reranker")
    if isinstance(embeddings, CachedEmbeddings):
        identity = _mapping(_attribute(embeddings, "identity"))
        model = identity.get("model")
        # A learned actual dimension can be present when configured dimensions
        # were null. Never substitute a stale configured dimension for it.
        dimensions = _attribute(embeddings, "dimensions")
    else:
        model = _attribute(embeddings, "model")
        dimensions = _attribute(embeddings, "dimensions")
    snapshot = {"embedding": {"availability": _availability(embeddings), "model": model, "dimensions": dimensions},
                "reranker": {"availability": _availability(reranker), "model": _attribute(reranker, "model"),
                             **{name: _attribute(reranker, name) for name in
                                ("provider", "revision", "device", "dtype", "max_length", "batch_size")}},
                "candidate_pool": {"status": "known", "value": _attribute(index, "candidate_pool")},
                "rrf_constant": {"status": "known", "value": _attribute(index, "rrf_constant")}}
    return sanitize_retrieval_config(snapshot)


def retrieval_for_cell(snapshot, mode: str, *, use_reranker=False) -> dict:
    """Describe configured participation, even if execution fails or is skipped.

    Hybrid Agent rows describe their configured Hybrid search tool. Whether a
    search actually happened and its actual mode remain in the tool traces.
    """
    if mode not in {"bm25", "dense", "rrf", "hybrid"} or type(use_reranker) is not bool:
        raise ValueError("检索配置须使用明确模式及布尔重排开关。")
    result = sanitize_retrieval_config(snapshot)
    result["mode"] = mode
    if mode == "bm25":
        result["embedding"] = {"availability": "not_used", "model": None, "dimensions": None}
    if not use_reranker and mode != "hybrid":
        result["reranker"] = {"availability": "not_used", "model": None}
    if mode in {"bm25", "dense"}:
        result["rrf_constant"] = {"status": "not_used", "value": None}
    return result
