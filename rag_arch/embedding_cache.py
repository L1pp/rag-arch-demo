"""Small JSON embedding cache with exact text keys and per-batch persistence.

Use one owner per cache path. Documents and queries have separate namespaces;
no provider credentials, pickle, hashes, database or automatic retries are used.
"""
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
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
        return self._dimensions

    @property
    def stats(self) -> dict:
        """Per-instance wrapper calls/hits, not vendor billing or HTTP retries."""
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
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_name(f".{self.cache_path.name}.{uuid4().hex}.tmp")
        payload = {"schema_version": 1, "identity": self._identity,
                   "vector_dimensions": self._dimensions,
                   "documents": self._documents, "queries": self._queries}
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
        self._dirty = False
        self._stats["cache_writes"] += 1

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("文档向量输入须为文本序列。")
        items = list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("文档向量输入须全部为文本。")
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
                    self._stats["document_requests"] += 1
                    self._stats["document_texts_submitted"] += len(batch)
                    raw = self.provider.embed_documents(batch)
                    vectors, dimensions = _vectors(raw, len(batch), self._dimensions)
                    self._dimensions = dimensions
                    self._documents.update(zip(batch, vectors))
                    self._dirty = True
                    if self.persist_batches:
                        self._save()
            # Callers receive fresh lists so they cannot corrupt cached vectors.
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
            if text in self._queries:
                self._stats["query_cache_hits"] += 1
                self._before_request()
                return list(self._queries[text])
            self._before_request()
            self._stats["query_requests"] += 1
            raw = self.provider.embed_query(text)
            vectors, dimensions = _vectors([raw], 1, self._dimensions)
            self._dimensions = dimensions
            self._queries[text] = vectors[0]
            if self.persist_queries:
                self._dirty = True
            if self.persist_batches:
                self._save()
            return list(self._queries[text])
        finally:
            if not self.persist_batches:
                self._save()
