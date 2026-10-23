"""Prepare reusable vectors using the project's explicitly configured provider."""
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from time import monotonic, sleep
import json
import re
import sys
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_arch.config import load_settings
from rag_arch.providers import create_embeddings
from rag_arch.embedding_cache import CachedEmbeddings
from rag_arch.catalog_index import CatalogIndex
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.standards import load_corpus


# Public model metadata uses 100,000 tokens per 6 seconds. Keep a
# conservative byte budget below that, with no concurrency or load testing.
THROUGHPUT_RETRY_DELAY = 60.0
MAX_THROUGHPUT_RETRIES = 2
INPUT_BUDGET_BYTES = 80_000
INPUT_WINDOW_SECONDS = 6.0


def _sanitize_error_text(value, known_secret=None) -> str:
    text = str(value or "")
    if isinstance(known_secret, str) and known_secret:
        text = text.replace(known_secret, "<redacted>")
    text = re.sub(r"(?i)\bsk(?:-sp)?-[A-Za-z0-9_-]+\b", "<redacted>", text)
    text = re.sub(r"(?i)(?:api[-_ ]?key|authorization|access[-_ ]?key[-_ ]?secret)\s*[:=]\s*\S+",
                  "<redacted>", text)
    return text[:240]


def _provider_error_metadata(error, known_secret=None) -> dict:
    payload = None
    for name in ("body", "response"):
        candidate = getattr(error, name, None)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
        candidate_json = getattr(candidate, "json", None)
        if callable(candidate_json):
            try:
                parsed = candidate_json()
            except Exception:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
                break
    nested = payload.get("error", payload) if isinstance(payload, Mapping) else {}
    code = (getattr(error, "code", None) or getattr(error, "error_code", None)
            or (nested.get("code") if isinstance(nested, Mapping) else None))
    request_id = (getattr(error, "request_id", None) or getattr(error, "requestId", None)
                  or (nested.get("request_id") if isinstance(nested, Mapping) else None)
                  or (nested.get("requestId") if isinstance(nested, Mapping) else None))
    status = (getattr(error, "status_code", None) or getattr(error, "status", None)
              or (nested.get("status_code") if isinstance(nested, Mapping) else None)
              or (nested.get("status") if isinstance(nested, Mapping) else None))
    message = nested.get("message") if isinstance(nested, Mapping) else None
    if not message:
        message = getattr(error, "message", None)
    if not message:
        message = error
    if status is not None and not isinstance(status, (str, int, float, bool)):
        status = _sanitize_error_text(status)
    return {"code": str(code) if code is not None else None,
            "message": _sanitize_error_text(message, known_secret),
            "request_id": _sanitize_error_text(request_id, known_secret) if request_id is not None else None,
            "status": status}


def _retryable_throughput(metadata: dict) -> bool:
    text = " ".join(str(metadata.get(key) or "") for key in ("code", "message")).casefold()
    if any(marker in text for marker in (
            "allocationquota.freetieronly", "freetieronly", "free tier", "free-tier",
            "free_tier", "free quota", "免费", "arrearage", "arrears", "欠费",
            "insufficient balance", "insufficient_balance")):
        return False
    return any(marker in text for marker in (
        "insufficient_quota", "rate_limit", "ratelimit", "too_many_requests",
        "throttl", "tpm", "tps"))


class _PrebuildProvider:
    """Add bounded retries around sequential provider document calls only."""

    def __init__(self, provider, known_secret=None, *, clock=None, sleeper=None):
        self.provider = provider
        self.dimensions = provider.dimensions
        self.chunk_size = min(provider.chunk_size, 10)
        self.provider_requests = 0
        self.provider_retries = 0
        self.provider_errors = []
        self.known_secret = known_secret
        self._clock = clock or monotonic
        self._sleeper = sleeper
        self._input_window = deque()
        self._input_window_bytes = 0
        self.estimated_input_bytes = 0
        self.budget_waits = 0

    def _sleep(self, seconds):
        (self._sleeper or sleep)(seconds)

    def _reserve_input(self, input_bytes: int):
        if input_bytes > INPUT_BUDGET_BYTES:
            raise ValueError(
                f"单批输入估算为 {input_bytes} UTF-8 字节，超过预建工作预算 {INPUT_BUDGET_BYTES}；"
                "不会发送或拆分该批次。")
        while True:
            now = self._clock()
            cutoff = now - INPUT_WINDOW_SECONDS
            while self._input_window and self._input_window[0][0] <= cutoff:
                _, expired = self._input_window.popleft()
                self._input_window_bytes -= expired
            if self._input_window_bytes + input_bytes <= INPUT_BUDGET_BYTES:
                self._input_window.append((now, input_bytes))
                self._input_window_bytes += input_bytes
                self.estimated_input_bytes += input_bytes
                return
            wait_seconds = max(0.0, INPUT_WINDOW_SECONDS - (now - self._input_window[0][0]))
            self.budget_waits += 1
            print(json.dumps({
                "phase": "provider_budget_wait",
                "wait_seconds": round(wait_seconds, 3),
                "estimated_input_bytes": input_bytes,
                "window_input_bytes": self._input_window_bytes,
                "budget_input_bytes": INPUT_BUDGET_BYTES,
                "window_seconds": INPUT_WINDOW_SECONDS,
            }, ensure_ascii=False), flush=True)
            self._sleep(wait_seconds)

    def embed_documents(self, texts):
        retry_count = 0
        input_bytes = sum(len(text.encode("utf-8")) for text in texts)
        while True:
            self._reserve_input(input_bytes)
            self.provider_requests += 1
            try:
                return self.provider.embed_documents(texts)
            except Exception as error:
                metadata = _provider_error_metadata(error, self.known_secret)
                retryable = _retryable_throughput(metadata)
                metadata["retry_count"] = retry_count
                metadata["retryable"] = retryable
                metadata["estimated_input_bytes"] = input_bytes
                will_retry = retryable and retry_count < MAX_THROUGHPUT_RETRIES
                wait_seconds = THROUGHPUT_RETRY_DELAY if will_retry else 0
                metadata["wait_seconds"] = wait_seconds
                self.provider_errors.append(metadata)
                print(json.dumps({"phase": "provider_retry", **metadata},
                                 ensure_ascii=False), flush=True)
                if not will_retry:
                    raise
                retry_count += 1
                self.provider_retries += 1
                self._sleep(THROUGHPUT_RETRY_DELAY)

    def embed_query(self, text):
        return self.provider.embed_query(text)


def main():
    started = monotonic()
    settings = load_settings()
    provider = create_embeddings(settings)
    if provider is None:
        raise ValueError('项目尚未配置Embedding。')
    request_provider = _PrebuildProvider(provider, known_secret=getattr(settings, "embedding_api_key", None))
    identity = {'base_url': settings.embedding_base_url.rstrip('/'),
                'model': settings.embedding_model, 'dimensions': provider.dimensions}
    cache = CachedEmbeddings(request_provider, ROOT / 'data/runtime/core_embeddings.json', identity,
        batch_size=request_provider.chunk_size, persist_batches=False)
    summary = {'started_at': datetime.now(timezone.utc).isoformat(), 'model': settings.embedding_model,
               'dimensions': provider.dimensions, 'cache_load_status': cache.load_status,
               'chat_requests': 0, 'complete': False}
    core_result = {"complete": False, "document_texts": 0}
    catalog_result = {"complete": False, "metadata_texts": 0}
    corpus = None
    try:
        corpus = load_corpus(ROOT / 'data/corpus')
        index = EvidenceIndex(corpus)
        texts = list(dict.fromkeys(chunk.text for chunk in index.chunks if chunk.text.strip()))
        core_result.update(evidence_records=len(index.chunks),
                           document_text_characters=sum(len(c.text) for c in index.chunks),
                           document_texts=len(texts))

        def core_progress():
            stats = cache.stats
            done = min(len(texts), stats['document_cache_hits'] + stats['document_texts_submitted'])
            print(json.dumps({'phase': 'core', 'completed_records': done,
                              'total_records': len(texts),
                              'seconds': round(monotonic() - started, 1)}, ensure_ascii=False), flush=True)

        cache.before_request = core_progress
        vectors = cache.embed_documents(texts)
        core_result["complete"] = len(vectors) == len(texts)
        print(json.dumps({'phase': 'core', 'completed_records': len(vectors),
                          'total_records': len(texts),
                          'seconds': round(monotonic() - started, 1),
                          'complete': core_result["complete"]},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        # Do not print a provider response body or credentials.
        error_code = getattr(exc, 'code', None) or getattr(exc, 'error_code', None)
        core_result.update(error_type=type(exc).__name__,
                           status_code=getattr(exc, 'status_code', None))
        if isinstance(error_code, (str, int)):
            core_result["error_code"] = error_code
        summary.update(error_type=type(exc).__name__, status_code=getattr(exc, 'status_code', None),
                       **({"error_code": error_code} if isinstance(error_code, (str, int)) else {}))

    core_result["provider_requests"] = request_provider.provider_requests
    core_result["provider_retries"] = request_provider.provider_retries
    core_provider_requests = request_provider.provider_requests
    core_provider_retries = request_provider.provider_retries

    catalog_cache = None
    try:
        records = getattr(corpus, "catalog", []) if corpus is not None else []
        catalog_cache = CachedEmbeddings(
            request_provider, ROOT / 'data/runtime/catalog_embeddings.json', identity,
            batch_size=request_provider.chunk_size, persist_batches=False)
        catalog = CatalogIndex(records, embeddings=catalog_cache)
        catalog_result["metadata_texts"] = len(catalog.records)

        def catalog_progress():
            stats = catalog_cache.stats
            done = min(catalog_result["metadata_texts"],
                       stats['document_cache_hits'] + stats['document_texts_submitted'])
            print(json.dumps({'phase': 'catalog', 'completed_records': done,
                              'total_records': catalog_result["metadata_texts"],
                              'seconds': round(monotonic() - started, 1)}, ensure_ascii=False), flush=True)

        catalog_cache.before_request = catalog_progress
        metadata_index = catalog.prepare_embeddings()
        catalog_result["complete"] = metadata_index is not None
        print(json.dumps({'phase': 'catalog',
                          'completed_records': catalog_result["metadata_texts"] if catalog_result["complete"] else 0,
                          'total_records': catalog_result["metadata_texts"],
                          'seconds': round(monotonic() - started, 1),
                          'complete': catalog_result["complete"]},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        # Do not print a provider response body or credentials.
        error_code = getattr(exc, 'code', None) or getattr(exc, 'error_code', None)
        catalog_result.update(error_type=type(exc).__name__,
                              status_code=getattr(exc, 'status_code', None))
        if isinstance(error_code, (str, int)):
            catalog_result["error_code"] = error_code
    finally:
        catalog_result["provider_requests"] = request_provider.provider_requests - core_provider_requests
        catalog_result["provider_retries"] = request_provider.provider_retries - core_provider_retries
        summary.update({key: value for key, value in core_result.items() if key != "complete"})
        summary.update(core=core_result, catalog=catalog_result,
                       complete=bool(core_result.get("complete") and catalog_result.get("complete")),
                       seconds=round(monotonic() - started, 2),
                       cache_stats=cache.stats,
                       catalog_cache_stats=catalog_cache.stats if catalog_cache is not None else None,
                       provider_requests=request_provider.provider_requests,
                       provider_retries=request_provider.provider_retries,
                       provider_estimated_input_bytes=request_provider.estimated_input_bytes,
                       provider_budget_waits=request_provider.budget_waits,
                       provider_input_accounting="utf8_bytes_conservative_prebuild_only",
                       provider_errors=request_provider.provider_errors)
        path = ROOT / 'data/runtime/core_index_build.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
