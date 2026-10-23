"""Small JSON embedding cache with exact text keys and per-batch persistence.

Use one owner per cache path. Documents and queries have separate namespaces;
no provider credentials, pickle, hashes, database or automatic retries are used.
"""
from collections.abc import Mapping, Sequence
import asyncio
from concurrent.futures import CancelledError as ConcurrentCancelledError
import inspect
import json
import math
from numbers import Real
from pathlib import Path
import threading
from urllib.parse import urlsplit
from uuid import uuid4


def _dimension(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("向量维度须为正整数或 None。")
    return value


def _identity(value: Mapping) -> dict:
    if not isinstance(value, Mapping) or set(value) != {"base_url", "model", "dimensions"}:
        raise ValueError("缓存身份只能且必须包含 base_url、model、dimensions，不接受 API Key。")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in ("base_url", "model")):
        raise ValueError("缓存身份的接口地址和模型不能为空。")
    try:
        url = urlsplit(value["base_url"])
        valid = (url.scheme in {"http", "https"} and url.hostname and not url.username
                 and not url.password and not url.query and not url.fragment)
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("缓存接口地址须为不含凭据、查询参数或片段的 HTTP(S) 地址。")
    return {"base_url": value["base_url"], "model": value["model"],
            "dimensions": _dimension(value["dimensions"])}


def _vectors(raw, count: int, dimensions: int | None):
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != count:
        raise ValueError("向量条数与提交文本数不符。")
    validated = []
    for vector in raw:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            raise ValueError("向量须为非空数值序列。")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions:
            raise ValueError("向量维度与缓存身份或已有向量不符。")
        values = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("向量须仅含有限数值。")
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise ValueError("向量含无法表示的数值。") from None
            if not math.isfinite(number):
                raise ValueError("向量含非有限数值。")
            values.append(number)
        if not any(values):
            raise ValueError("向量不能为零向量。")
        validated.append(values)
    return validated, dimensions


async def aclose_embedding_provider(provider):
    """Close an owned provider transport without requiring a document cache."""
    provider_aclose = getattr(provider, "aclose", None)
    if callable(provider_aclose):
        result = provider_aclose()
        if inspect.isawaitable(result):
            await result
        return
    provider_async = getattr(provider, "async_client", None)
    for candidate in (provider_async, getattr(provider_async, "_client", None)):
        if candidate is None:
            continue
        close = getattr(candidate, "aclose", None) or getattr(candidate, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return
    close = getattr(provider, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


class CachedEmbeddings:
    """Wrap embed_documents/embed_query without changing provider/model choice.

    An identity mismatch invalidates all old entries in memory. The old file is
    replaced only after a successful new batch, and load_status exposes this
    event. Exact original texts are saved locally, so keep this cache private.
    """

    def __init__(self, provider, cache_path: str | Path, identity: Mapping,
                 batch_size: int = 20, *, before_request=None,
                 persist_batches: bool = True, allow_document_requests: bool = True,
                 persist_queries: bool = False):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("缓存批量须为正整数。")
        if before_request is not None and not callable(before_request):
            raise TypeError("before_request须为可调用对象或 None。")
        if type(persist_batches) is not bool:
            raise TypeError("persist_batches须为布尔值。")
        if type(allow_document_requests) is not bool:
            raise TypeError("allow_document_requests须为布尔值。")
        if type(persist_queries) is not bool:
            raise TypeError("persist_queries须为布尔值。")
        self.provider = provider
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.before_request = before_request
        self.persist_batches = persist_batches
        self.allow_document_requests = allow_document_requests
        self.persist_queries = persist_queries
        self._identity = _identity(identity)
        self._dimensions = self._identity["dimensions"]
        self._documents: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}
        self._dirty = False
        self._state_lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._cache_version = 0
        self._async_inflight: dict[tuple[int, str, str], asyncio.Future] = {}
        self._closed = False
        self._close_complete = False
        self.load_status = "absent"
        self._stats = {"document_requests": 0, "document_texts_submitted": 0,
                       "document_cache_hits": 0, "query_requests": 0,
                       "query_cache_hits": 0, "cache_writes": 0}
        self._load()

    @property
    def identity(self) -> dict:
        return dict(self._identity)

    @property
    def dimensions(self) -> int | None:
        with self._state_lock:
            return self._dimensions

    @property
    def stats(self) -> dict:
        """Per-instance wrapper calls/hits, not vendor billing or HTTP retries."""
        with self._state_lock:
            return dict(self._stats)

    def _before_request(self) -> None:
        if self.before_request is not None:
            self.before_request()

    def _load(self):
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError("向量缓存不是有效的 UTF-8 JSON；未使用或覆盖该文件。") from None
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1):
            raise ValueError("向量缓存格式版本无效；未使用或覆盖该文件。")
        stored_identity = _identity(data.get("identity"))
        if stored_identity != self._identity:
            self.load_status = "identity_mismatch"
            return
        dimensions = _dimension(data.get("vector_dimensions"))
        if self._dimensions is not None and dimensions != self._dimensions:
            raise ValueError("缓存的实际维度与声明身份不符。")
        groups = {}
        for name in ("documents", "queries"):
            records = data.get(name)
            if not isinstance(records, dict) or any(not isinstance(text, str) for text in records):
                raise ValueError("向量缓存的文本映射格式无效。")
            if records and dimensions is None:
                raise ValueError("非空向量缓存缺少实际维度。")
            vectors, _ = _vectors(list(records.values()), len(records), dimensions)
            groups[name] = dict(zip(records, vectors))
        self._dimensions = dimensions
        self._documents, self._queries = groups["documents"], groups["queries"]
        self.load_status = "loaded"

    def _save(self):
        with self._save_lock:
            with self._state_lock:
                if not self._dirty:
                    return
                version = self._cache_version
                payload = {"schema_version": 1, "identity": dict(self._identity),
                           "vector_dimensions": self._dimensions,
                           "documents": {key: list(vector) for key, vector in self._documents.items()},
                           "queries": {key: list(vector) for key, vector in self._queries.items()}}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_name(f".{self.cache_path.name}.{uuid4().hex}.tmp")
            try:
                # Same-directory UUID file avoids tempfile's 0700 sandbox directory.
                with temporary.open("x", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                temporary.replace(self.cache_path)
            finally:
                # A failed replacement leaves the previous cache untouched. Valid
                # new vectors remain in memory; the next call retries disk only.
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._state_lock:
                if self._cache_version == version:
                    self._dirty = False
                self._stats["cache_writes"] += 1

    async def _save_async(self):
        # File serialization/replacement is local blocking work. run_local is
        # bounded and, on cancellation, waits until the physical write ends.
        with self._state_lock:
            if not self._dirty:
                return
        from .async_support import run_local
        await run_local(self._save)

    def _commit_vectors(self, group: dict[str, list[float]], texts, raw, *, dirty=True):
        """Validate and merge a completed provider response under the cache lock."""
        with self._state_lock:
            vectors, dimensions = _vectors(raw, len(texts), self._dimensions)
            self._dimensions = dimensions
            # A concurrent sync/async request may have committed the same text
            # first. Keep its vector and merge every other result atomically.
            for text, vector in zip(texts, vectors):
                group.setdefault(text, vector)
            if dirty:
                self._dirty = True
                self._cache_version += 1
            return [list(group[text]) for text in texts]

    @staticmethod
    def _consume_future(future):
        if not future.cancelled():
            try:
                future.exception()
            except BaseException:
                pass

    def _claim_async(self, loop, namespace: str, text: str):
        key = (id(loop), namespace, text)
        with self._state_lock:
            future = self._async_inflight.get(key)
            if future is not None:
                return key, future, False
            future = loop.create_future()
            future.add_done_callback(self._consume_future)
            self._async_inflight[key] = future
            return key, future, True

    def _finish_async(self, key, future, *, result=None, error=None):
        with self._state_lock:
            if self._async_inflight.get(key) is future:
                self._async_inflight.pop(key, None)
        if future.done():
            return
        if isinstance(error, (asyncio.CancelledError, ConcurrentCancelledError)):
            future.cancel()
        elif error is not None:
            future.set_exception(error)
        else:
            future.set_result(list(result))

    async def _await_async_vector(self, namespace, text, provider_call, *, request_stat,
                                  submitted_texts=0):
        loop = asyncio.get_running_loop()
        group = self._queries if namespace == "queries" else self._documents
        while True:
            with self._state_lock:
                cached = group.get(text)
                if cached is not None:
                    self._stats["query_cache_hits" if namespace == "queries" else "document_cache_hits"] += 1
                    cached = list(cached)
            if cached is not None:
                self._before_request()
                return cached
            key, future, owner = self._claim_async(loop, namespace, text)
            if not owner:
                self._before_request()
                try:
                    return list(await asyncio.shield(future))
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    # The producer belonged to another request and was
                    # cancelled. Retry under this caller's own control scope.
                    continue
            try:
                if self.persist_batches:
                    await self._save_async()
                self._before_request()
                with self._state_lock:
                    self._stats[request_stat] += 1
                    if submitted_texts:
                        self._stats["document_texts_submitted"] += submitted_texts
                raw = await provider_call()
                committed = self._commit_vectors(
                    group, [text], [raw], dirty=namespace != "queries" or self.persist_queries)[0]
                if self.persist_batches:
                    await self._save_async()
                self._finish_async(key, future, result=committed)
                return list(committed)
            except BaseException as exc:
                self._finish_async(key, future, error=exc)
                raise

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed through the provider's native async API with per-text single-flight."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("文档向量输入须为文本序列。")
        items = list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("文档向量输入须全部为文本。")
        if self._closed:
            raise RuntimeError("向量缓存底层客户端已关闭。")
        loop = asyncio.get_running_loop()
        with self._state_lock:
            self._stats["document_cache_hits"] += sum(text in self._documents for text in items)
            missing = list(dict.fromkeys(text for text in items if text not in self._documents))
            if missing and not self.allow_document_requests:
                raise ValueError("文档向量未预建；当前缓存仅允许使用已有文档向量，请先运行预建任务。")
        try:
            if not missing:
                if items:
                    self._before_request()
            else:
                start = 0
                while start < len(missing):
                    candidate = missing[start:start + self.batch_size]
                    owned = []
                    claimed = []
                    waiting = []
                    with self._state_lock:
                        for text in candidate:
                            if text in self._documents:
                                continue
                            key, future, owner = self._claim_async(loop, "documents", text)
                            if owner:
                                owned.append(text)
                                claimed.append((key, future, text))
                            else:
                                waiting.append((future, text))
                    if owned:
                        try:
                            if self.persist_batches:
                                await self._save_async()
                            self._before_request()
                            with self._state_lock:
                                self._stats["document_requests"] += 1
                                self._stats["document_texts_submitted"] += len(owned)
                            async_embed = getattr(self.provider, "aembed_documents", None)
                            if not callable(async_embed):
                                raise TypeError("向量提供者须提供原生异步 aembed_documents 方法。")
                            raw = await async_embed(owned)
                            committed = self._commit_vectors(self._documents, owned, raw)
                            if self.persist_batches:
                                await self._save_async()
                            for (key, future, text), vector in zip(claimed, committed):
                                self._finish_async(key, future,
                                                   result=vector)
                        except BaseException as exc:
                            for key, future, _ in claimed:
                                self._finish_async(key, future, error=exc)
                            raise
                    for future, text in waiting:
                        try:
                            await asyncio.shield(future)
                        except asyncio.CancelledError:
                            current = asyncio.current_task()
                            if current is not None and current.cancelling():
                                raise
                            async def embed_one(value=text):
                                async_embed = getattr(self.provider, "aembed_documents", None)
                                if not callable(async_embed):
                                    raise TypeError("向量提供者须提供原生异步 aembed_documents 方法。")
                                vectors = await async_embed([value])
                                if not isinstance(vectors, Sequence) or len(vectors) != 1:
                                    raise ValueError("向量条数与提交文本数不符。")
                                return vectors[0]
                            await self._await_async_vector(
                                "documents", text, embed_one,
                                request_stat="document_requests", submitted_texts=1)
                    start += len(candidate)
            with self._state_lock:
                return [list(self._documents[text]) for text in items]
        finally:
            if not self.persist_batches:
                await self._save_async()

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query with the provider's native asynchronous query method."""
        if not isinstance(text, str):
            raise TypeError("问题向量输入须为文本。")
        if self._closed:
            raise RuntimeError("向量缓存底层客户端已关闭。")
        try:
            async def call_provider():
                async_embed = getattr(self.provider, "aembed_query", None)
                if not callable(async_embed):
                    raise TypeError("向量提供者须提供原生异步 aembed_query 方法。")
                return await async_embed(text)

            return await self._await_async_vector(
                "queries", text, call_provider, request_stat="query_requests")
        finally:
            if not self.persist_batches:
                await self._save_async()

    async def aclose(self):
        """Close the shared provider's async HTTP client at resource shutdown."""
        if self._close_complete:
            return
        # Fence new requests even if the underlying close fails; a later
        # shutdown retry remains possible until close completes successfully.
        self._closed = True
        await aclose_embedding_provider(self.provider)
        self._close_complete = True

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("文档向量输入须为文本序列。")
        items = list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("文档向量输入须全部为文本。")
        with self._state_lock:
            self._stats["document_cache_hits"] += sum(text in self._documents for text in items)
            missing = list(dict.fromkeys(text for text in items if text not in self._documents))
        if missing and not self.allow_document_requests:
            raise ValueError("文档向量未预建；当前缓存仅允许使用已有文档向量，请先运行预建任务。")
        try:
            if self.persist_batches:
                # Persist any valid results left in memory by an earlier disk
                # failure before handing additional texts to the provider.
                self._save()
            if not missing:
                if items:
                    self._before_request()
            else:
                for offset in range(0, len(missing), self.batch_size):
                    batch = missing[offset:offset + self.batch_size]
                    self._before_request()
                    with self._state_lock:
                        self._stats["document_requests"] += 1
                        self._stats["document_texts_submitted"] += len(batch)
                    raw = self.provider.embed_documents(batch)
                    self._commit_vectors(self._documents, batch, raw)
                    if self.persist_batches:
                        self._save()
            # Callers receive fresh lists so they cannot corrupt cached vectors.
            with self._state_lock:
                return [list(self._documents[text]) for text in items]
        finally:
            if not self.persist_batches:
                self._save()

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("问题向量输入须为文本。")
        try:
            if self.persist_batches:
                self._save()
            with self._state_lock:
                cached = self._queries.get(text)
                if cached is not None:
                    self._stats["query_cache_hits"] += 1
                    cached = list(cached)
            if cached is not None:
                self._before_request()
                return cached
            self._before_request()
            with self._state_lock:
                self._stats["query_requests"] += 1
            raw = self.provider.embed_query(text)
            result = self._commit_vectors(self._queries, [text], [raw],
                                          dirty=self.persist_queries)[0]
            if self.persist_batches:
                self._save()
            return result
        finally:
            if not self.persist_batches:
                self._save()
