"""One read-only corpus per backend; original pages are loaded only on demand."""
from dataclasses import dataclass
import asyncio
import logging
import os
from copy import copy, deepcopy
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from urllib.parse import urlencode

from ..catalog_index import CatalogIndex
from ..async_support import run_local
from ..evidence_index import EvidenceIndex
from ..page_sources import _locations, standard_page_sources
from ..standard_status import StandardStatusRegistry
from ..standards import load_corpus

ROOT = Path(__file__).resolve().parents[2]
_LOG = logging.getLogger(__name__)


class PublicBM25UnavailableError(OSError):
    """Invalid/missing public package; existing read routes return HTTP 503."""


class PublicCorpusUnavailableError(OSError):
    """Invalid/missing public corpus package; read routes return HTTP 503."""


@dataclass(frozen=True)
class StandardsBundle:
    index: EvidenceIndex
    registry: StandardStatusRegistry
    catalog: CatalogIndex
    pages: object


class _PreparedEmbeddings:
    """Read prepared document vectors; send only query texts to the provider."""

    def __init__(self, cache):
        self.cache = cache
        self.provider = cache.provider
        self.before_request = cache.before_request
        self.persist_batches = False
        self.dimensions = cache.dimensions

    def _before(self):
        if self.before_request is not None:
            self.before_request()

    def embed_documents(self, texts):
        self._before()
        return self.cache.embed_documents(texts)

    async def aembed_documents(self, texts):
        self._before()
        return await self.cache.aembed_documents(texts)

    def embed_query(self, text):
        self._before()
        return self.provider.embed_query(text)

    async def aembed_query(self, text):
        self._before()
        return await self.provider.aembed_query(text)

    def clone(self):
        return _PreparedEmbeddings(self.cache)

    async def aclose(self):
        await self.cache.aclose()


class _QueryOnlyEmbeddings:
    """Own a query transport; document vectors are already in the mmap."""

    persist_batches = False

    def __init__(self, provider, *, dimensions=None, owns_provider=True):
        self.provider = provider
        self.dimensions = dimensions if dimensions is not None else getattr(provider, "dimensions", None)
        self.before_request = None
        self._owns_provider = owns_provider
        self._closed = False
        self._close_complete = False

    def _before(self):
        if self._closed:
            raise RuntimeError("向量查询客户端已关闭。")
        if self.before_request is not None:
            self.before_request()

    def embed_documents(self, texts):
        raise RuntimeError("已预建的正文向量必须直接绑定矩阵，不能重新请求文档向量。")

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    def embed_query(self, text):
        self._before()
        return self.provider.embed_query(text)

    async def aembed_query(self, text):
        self._before()
        return await self.provider.aembed_query(text)

    def clone(self):
        return _QueryOnlyEmbeddings(self.provider, dimensions=self.dimensions, owns_provider=False)

    async def aclose(self):
        if not self._owns_provider or self._close_complete:
            return
        from ..embedding_cache import aclose_embedding_provider
        self._closed = True
        await aclose_embedding_provider(self.provider)
        self._close_complete = True


class StandardsResources:
    """No reload while tasks run: restarting the backend adopts a new corpus.

    The lock only coordinates initial resource construction. PDF work is delegated
    to one process, not performed while holding this resource lock.
    """

    def __init__(self, root=ROOT, *, corpus=None, pdf_backend=None,
                 recover_pdf_after_confirmed_abort: bool = False):
        if type(recover_pdf_after_confirmed_abort) is not bool:
            raise ValueError("recover_pdf_after_confirmed_abort须为布尔值。")
        self.root = Path(root)
        self._lock = RLock()
        self._corpus = corpus
        self._corpus_store = None
        self._core_corpus_source = "provided" if corpus is not None else None
        self._catalog = None
        self._bundle = None
        self._page_services = {}
        self._pdf_backend = pdf_backend
        self._owns_pdf_backend = pdf_backend is None
        self._recover_pdf_after_confirmed_abort = recover_pdf_after_confirmed_abort
        self._catalog_settings = None
        self._catalog_warning = None
        self._prepared_index = None
        self._binary_store = None
        self._bm25_store = None
        self._core_bm25_source = None
        self._core_vector_source = None
        self._embedding_warning = None
        self._async_prepare_lock = None
        self._async_close_lock = None
        self._async_resources = []
        self._async_resource_ids = set()
        self._closed = False
        self._close_complete = False

    def _remember_async_resource(self, resource):
        identity = id(resource)
        with self._lock:
            if identity not in self._async_resource_ids:
                self._async_resource_ids.add(identity)
                self._async_resources.append(resource)
        return resource

    def configure_catalog(self, settings):
        """Store provider configuration and load prepared metadata vectors if present."""
        with self._lock:
            if self._closed:
                raise RuntimeError("标准资源已关闭。")
            self._catalog_settings = settings
            self._catalog_warning = None
            if self._catalog is not None:
                self._catalog.set_embeddings(self._catalog_embeddings())
                self._prepare_catalog_embeddings()

    def _prepare_catalog_embeddings(self):
        if self._catalog is None or self._catalog.embeddings is None:
            return
        if self._catalog.prepare_embeddings() is None:
            # Keep missing/invalid prebuilds on the lexical path.  Clearing the
            # provider prevents every candidate query from retrying document
            # metadata initialization and cannot issue a document request.
            self._catalog.set_embeddings(None)
            self._catalog_warning = (
                "目录元数据向量未预建，本次使用字面候选；请先完成目录向量预建。")

    def configure_embeddings(self, settings, *, backend=None):
        """Prepare immutable public vectors once, before publishing the bundle.

        The default binary backend never falls back to loading the large JSON.
        ``backend='json'`` is an explicit rollback path for existing prebuilds.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("标准资源已关闭。")
            if (self._bundle is not None or self._binary_store is not None
                    or (self._prepared_index is not None
                        and self._prepared_index.local._vectors is not None)):
                raise RuntimeError("标准向量资源不能在使用期间重配；请重启worker采用新版本。")
            self._embedding_warning = None
            self._core_vector_source = None
            if settings is None or not settings.embedding_ready:
                return
            selected = backend or os.environ.get("RAG_ARCH_CORE_VECTOR_BACKEND", "binary")
            if selected not in {"binary", "json"}:
                raise ValueError("RAG_ARCH_CORE_VECTOR_BACKEND须为binary或json。")
            corpus, _ = self._inventory()
            # Lexical-package errors must remain visible; treating them as an
            # embedding failure could hide a corrupt/stale public index.
            index = self._reference_index(corpus)
            store = None
            try:
                from ..providers import create_embeddings
                provider = create_embeddings(settings)
                identity = {"base_url": settings.embedding_base_url.rstrip("/"),
                            "model": settings.embedding_model,
                            "dimensions": getattr(provider, "dimensions", settings.embedding_dimensions)}
                if selected == "binary":
                    from ..binary_vectors import BinaryVectorStore
                    embeddings = self._remember_async_resource(_QueryOnlyEmbeddings(provider))
                    store = BinaryVectorStore.open(
                        self.root / "data/runtime/core_vectors", identity=identity, chunks=index.chunks)
                    embeddings.dimensions = store.vectors.shape[1]
                    index.local.bind_prepared_vectors(store.vectors, embeddings=embeddings)
                    index.embeddings = embeddings
                    self._binary_store = store
                else:
                    from ..embedding_cache import CachedEmbeddings
                    cache = CachedEmbeddings(
                        provider, self.root / "data/runtime/core_embeddings.json", identity,
                        batch_size=settings.embedding_batch_size,
                        persist_batches=False, allow_document_requests=False)
                    embeddings = self._remember_async_resource(_PreparedEmbeddings(cache))
                    # Reuse the prepared lexical index on explicit JSON
                    # rollback too; never build a second Counter/mmap index.
                    matrix = index.local._document_vectors(
                        embeddings.embed_documents([chunk.text for chunk in index.local.chunks]))
                    matrix.flags.writeable = False
                    index.local.bind_prepared_vectors(matrix, embeddings)
                    index.embeddings = embeddings
                self._core_vector_source = selected
            except Exception as exc:
                if store is not None:
                    store.close()
                self._embedding_warning = (
                    f"规范文档向量未加载（{type(exc).__name__}）；本轮使用BM25。"
                    "请运行 python -m scripts.convert_core_vectors 转换已有向量，或显式设置"
                    " RAG_ARCH_CORE_VECTOR_BACKEND=json 回滚旧缓存；不会自动请求文档向量。")
                _LOG.warning(self._embedding_warning)

    def _reference_index(self, corpus):
        """Prepare one public lexical index; task wrappers borrow its mapping.

        Auto keeps first-run compatibility when CURRENT is absent. An existing
        package is always validated strictly: corrupt/stale packages never
        trigger an unexpected full-memory rebuild. Production may require mmap.
        """
        if self._prepared_index is not None:
            return self._prepared_index
        selected = os.environ.get("RAG_ARCH_CORE_BM25_BACKEND", "auto")
        if selected not in {"auto", "mmap", "memory"}:
            raise ValueError("RAG_ARCH_CORE_BM25_BACKEND须为auto、mmap或memory。")
        directory = self.root / "data/runtime/core_bm25"
        if selected == "auto" and not os.path.lexists(directory / "CURRENT"):
            _LOG.warning("公共BM25包尚未预建，本进程使用内存索引；请运行 "
                         "python -m scripts.build_core_bm25。正式部署可设置 "
                         "RAG_ARCH_CORE_BM25_BACKEND=mmap 强制使用只读包。")
            selected = "memory"
        try:
            index = EvidenceIndex(corpus, bm25_directory=None if selected == "memory" else directory)
        except Exception as exc:
            raise PublicBM25UnavailableError(
                "公共BM25索引加载失败；请运行 python -m scripts.build_core_bm25 "
                "重新预建并重启，或显式设置 RAG_ARCH_CORE_BM25_BACKEND=memory 回滚。") from exc
        self._bm25_store = index.local._bm25_store
        self._core_bm25_source = "mmap" if self._bm25_store is not None else "memory"
        self._prepared_index = index
        return index

    def _catalog_embeddings(self):
        settings = self._catalog_settings
        if settings is None or not settings.embedding_ready:
            return None
        try:
            from ..providers import create_embeddings
            from ..embedding_cache import CachedEmbeddings
            from .catalog_tools import check_catalog_request
            provider = create_embeddings(settings)
            cache = CachedEmbeddings(provider, self.root / "data/runtime/catalog_embeddings.json",
                {"base_url": settings.embedding_base_url.rstrip("/"), "model": settings.embedding_model,
                 "dimensions": getattr(provider, "dimensions", settings.embedding_dimensions)},
                batch_size=settings.embedding_batch_size, before_request=check_catalog_request,
                persist_batches=False, allow_document_requests=False)
            return self._remember_async_resource(_PreparedEmbeddings(cache))
        except Exception as exc:
            self._catalog_warning = f"目录向量客户端未就绪（{type(exc).__name__}），本次可使用字面候选。"
            return None

    def _inventory(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("标准资源已关闭。")
            if self._corpus is None:
                self._load_public_corpus()
            if self._catalog is None:
                self._catalog = CatalogIndex(self._corpus.catalog, embeddings=self._catalog_embeddings())
                self._prepare_catalog_embeddings()
            return self._corpus, self._catalog

    def _load_public_corpus(self):
        selected = os.environ.get("RAG_ARCH_CORE_CORPUS_BACKEND", "auto")
        if selected not in {"auto", "mmap", "json"}:
            raise ValueError("RAG_ARCH_CORE_CORPUS_BACKEND须为auto、mmap或json。")
        directory = self.root / "data/runtime/core_corpus"
        if selected == "auto" and not os.path.lexists(directory / "CURRENT"):
            _LOG.warning("公共语料包尚未预建，本进程使用JSON语料；请运行 "
                         "python -m scripts.build_core_corpus_mmap。正式部署可设置 "
                         "RAG_ARCH_CORE_CORPUS_BACKEND=mmap 强制使用只读包。")
            selected = "json"
        if selected == "json":
            self._corpus = load_corpus(self.root / "data/corpus")
            self._core_corpus_source = "json"
            return
        try:
            from ..binary_corpus import BinaryCorpusStore
            store = BinaryCorpusStore.open(directory, source_dir=self.root / "data/corpus")
        except Exception as exc:
            raise PublicCorpusUnavailableError(
                "公共语料包加载失败；请运行 python -m scripts.build_core_corpus_mmap "
                "重新预建并重启，或显式设置 RAG_ARCH_CORE_CORPUS_BACKEND=json 回滚。") from exc
        self._corpus_store = store
        self._corpus = store.corpus
        self._core_corpus_source = "mmap"

    def _pdf(self):
        if self._closed:
            raise RuntimeError("标准资源已关闭。")
        if self._pdf_backend is None:
            from ..pdf_backend import ProcessPdfBackend
            self._pdf_backend = ProcessPdfBackend(
                reopen_after_confirmed_abort=self._recover_pdf_after_confirmed_abort)
        return self._pdf_backend

    def page_service(self, std_no=None):
        with self._lock:
            corpus, _ = self._inventory()
            if std_no not in self._page_services:
                index = SimpleNamespace(corpus=corpus, collection_id="web:standards")
                self._page_services[std_no] = standard_page_sources(
                    index, selected={std_no} if std_no else None, root=self.root,
                    pdf_backend=self._pdf())
            return self._page_services[std_no]

    def get(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("标准资源已关闭。")
            if self._bundle is None:
                corpus, catalog = self._inventory()
                self._bundle = StandardsBundle(
                    self._reference_index(corpus),
                    StandardStatusRegistry(self.root, corpus.clauses,
                                           catalog=corpus.catalog, relations=corpus.relations),
                    catalog, self.page_service())
            return self._bundle

    async def aprepare(self) -> StandardsBundle:
        """Prepare the shared standard bundle without blocking the worker loop."""
        if self._async_prepare_lock is None:
            with self._lock:
                if self._async_prepare_lock is None:
                    self._async_prepare_lock = asyncio.Lock()
        async with self._async_prepare_lock:
            if self._closed:
                raise RuntimeError("标准资源已关闭。")
            return await run_local(self.get)

    @property
    def pdf_backend(self):
        with self._lock:
            return self._pdf()

    def plan_index(self):
        """Create a task-local plan index over the shared read-only corpus data.

        BM25 statistics, corpus records and prepared document vectors are
        immutable after startup and can be shared.  The index/local wrappers,
        embedding callback and lazy reranker belong to this invocation only.
        """
        base = self.get().index
        index = copy(base)
        index.local = copy(base.local)
        embeddings = getattr(base, "embeddings", None)
        clone = getattr(embeddings, "clone", None)
        if embeddings is None:
            index.embeddings = None
            index.local.embeddings = None
            index.local._vectors = None
        elif callable(clone):
            index.embeddings = clone()
            index.local.embeddings = index.embeddings
        else:
            # A non-cloneable provider wrapper may carry request-local hooks or
            # mutable diagnostics. Do not expose it to concurrent plan runs.
            index.embeddings = None
            index.local.embeddings = None
            index.local._vectors = None
        # PlanStandardTools creates a reranker lazily for this private index.
        index.reranker = None
        return index

    async def aplan_index(self):
        await self.aprepare()
        return await run_local(self.plan_index)

    def resolve(self, standard):
        _, catalog = self._inventory()
        key = standard.removeprefix("CATALOG/")
        exact = [CatalogIndex._visible(row) for row in catalog.records if row["std_id"] == key]
        matches = exact or catalog.get(standard)["matches"]
        if not matches:
            raise KeyError("本地目录未登记这项标准。")
        if len(matches) != 1:
            raise ValueError("存在多个版本，请选择明确的标准号和年份。")
        return matches[0]

    def catalog_entry(self, row):
        entry = self.resolve(row.get("catalog_id") or row.get("std_id") or row.get("std_no", ""))
        from ..catalog_runtime import CANDIDATE_FIELDS
        return {key: entry[key] for key in (
            "catalog_id", "std_no", "title", "library_class", "text_available", "original_available")} | {
            "page_url": "/viewer/pages?" + urlencode({"std": entry["catalog_id"]})
                        if entry["page_reading_available"] else None,
            "history_url": "/viewer/history/?" + urlencode({"std": entry["std_no"]}),
        } | {key: deepcopy(row[key]) for key in CANDIDATE_FIELDS if key in row}

    def catalog_candidates(self, query):
        _, catalog = self._inventory()
        rows = catalog.candidates(query, limit=5)
        diagnostics = catalog.last_diagnostics
        warnings = list(diagnostics.get("warnings", []))
        if self._catalog_warning:
            warnings.append(self._catalog_warning)
        return {"query": query, "matches": [self.catalog_entry(row) for row in rows],
                "semantic_used": diagnostics.get("semantic_used", False), "warnings": warnings}

    def catalog_entries(self):
        _, catalog = self._inventory()
        return [self.catalog_entry({"std_id": row["std_id"]}) for row in catalog.records]

    def _evidence(self, entry, evidence_id):
        corpus, _ = self._inventory()
        # Both StandardCorpus backends index IDs; a citation must not scan and
        # decode the entire corpus for every returned hit.
        row = corpus._clauses.get(evidence_id) or corpus._tables.get(evidence_id)
        if row and row["std_no"] == entry["std_no"]:
            return row
        raise KeyError("未找到属于所选标准的这条证据；不会改用其他版本。")

    @staticmethod
    def _mapped_pages(record, service):
        """Pages belong to a named file; never transfer numbers to another copy."""
        mapped = {}
        for location in _locations(record):
            for key in ("source_path", "asset_path", "image_path", "file"):
                path = location.get(key)
                ref = service.source_ref_for_path(path)
                if not ref:
                    continue
                if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    start = end = 1
                else:
                    start = location.get("page", location.get("page_start"))
                    end = location.get("page_end", start)
                if type(start) is int and type(end) is int and 0 < start <= end:
                    target = mapped.setdefault(ref, {})
                    for page in range(start, end + 1):
                        # Only display book numbering explicitly recorded for this location.
                        logical = location.get("logical_page", location.get("document_page_start"))
                        target.setdefault(page, logical if page == start else None)
                break
        return mapped

    def citation(self, record):
        entry = self.resolve(record.get("std_no", ""))
        evidence_id = record.get("evidence_id", record.get("clause_id", record.get("table_id", "")))
        url = None
        try:
            original = self._evidence(entry, evidence_id)
            if self._mapped_pages(original, self.page_service(entry["std_no"])):
                url = "/viewer/pages?" + urlencode({"std": entry["catalog_id"], "evidence": evidence_id})
        except KeyError:
            # A model-read image is an actual per-run page, not a corpus clause.
            # Resolve its saved source path in the per-standard page registry.
            if record.get("kind") == "page_image":
                mapped = self._mapped_pages(record, self.page_service(entry["std_no"]))
                if mapped:
                    ref = next(iter(mapped))
                    url = "/viewer/pages?" + urlencode({"std": entry["catalog_id"],
                                                        "source_ref": ref, "page": min(mapped[ref])})
        return {"evidence_id": evidence_id, "std_no": entry["std_no"], "title": entry["title"],
                "clause_no": record.get("clause_no"), "text": record.get("text", ""),
                "page_url": url,
                "history_url": "/viewer/history/?" + urlencode({"std": entry["std_no"]})}

    def page_view(self, std, *, evidence=None, source_ref=None, page=None):
        entry = self.resolve(std)
        if not entry["original_available"]:
            raise ValueError("这项标准仅有目录登记，本地没有可显示的原件。")
        service = self.page_service(entry["std_no"])
        sources = service.sources()
        mapping = self._mapped_pages(self._evidence(entry, evidence), service) if evidence else {}
        if evidence and not mapping:
            raise ValueError("这条证据没有可用的原件页码映射，不能猜测对应页。")
        # Citation navigation lists only copies with an explicit mapping. Catalog
        # navigation can choose any registered original, beginning at physical page 1.
        choices = [row for row in sources if not evidence or row["source_ref"] in mapping]
        if not choices:
            raise ValueError("登记原件当前不可用，或没有支持分页的来源。")
        selected = next((row for row in choices if row["source_ref"] == source_ref), None) if source_ref else choices[0]
        if selected is None:
            raise KeyError("所选来源不属于这项标准或这条证据。")
        ref = selected["source_ref"]
        mapped = mapping.get(ref, {})
        requested = [page] if page is not None else (sorted(mapped)[:2] or [1])
        if any(type(p) is not int or p < 1 for p in requested):
            raise ValueError("页码必须是从 1 开始的整数。")
        # list_pages(limit=1) also validates/decrypt-checks in the PDF worker;
        # no page image is rendered until the browser requests image_url.
        description = service.list_pages(ref, start=1, limit=1)
        if page is not None and page > description["page_count"]:
            raise ValueError("登记页码超出此原件范围，请核对来源定位。")
        notes = ["页码为当前来源文件的物理页，不一定等于书内印刷页码。"]
        if any(p > description["page_count"] for p in mapped):
            notes.append("部分登记页码超出原件范围，对应页面暂不支持；其余页仍可查看。")
        if evidence and len(choices) > 1:
            notes.append(f"本条证据涉及 {len(choices)} 个来源文件，请切换来源查看其余登记页。")
        return {"std_no": entry["std_no"], "title": entry["title"], "source_ref": ref,
                "filename": selected["filename"], "page_count": description["page_count"],
                "sources": [{"source_ref": row["source_ref"], "filename": row["filename"],
                             "pages": sorted(mapping.get(row["source_ref"], {}))} for row in choices],
                "pages": [{"page": p, "document_page": mapped.get(p),
                           "image_url": "/api/standards/page-image?" + urlencode(
                               {"std": entry["catalog_id"], "source_ref": ref, "page": p})}
                          for p in requested],
                "warning": " ".join(notes)}

    def page_image(self, std, source_ref, page):
        entry = self.resolve(std)
        if not entry["original_available"]:
            raise ValueError("本地没有这项标准的原件。")
        return self.page_service(entry["std_no"]).read_page(source_ref, page)["image_bytes"]

    def close(self):
        with self._lock:
            self._closed = True
        # Runtime callers drain all borrowers before closing shared resources.
        # Task-local index clones never own or close this mapping.
        owned = ([self._pdf_backend] if self._owns_pdf_backend and self._pdf_backend is not None else [])
        owned += [self._binary_store, self._bm25_store, self._corpus_store]
        failures = []
        for resource in owned:
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise RuntimeError("关闭公共标准资源时有后端未能正常退出，可释放借用后重试。") from failures[0]

    async def aclose(self):
        if self._async_close_lock is None:
            with self._lock:
                if self._async_close_lock is None:
                    self._async_close_lock = asyncio.Lock()
        async with self._async_close_lock:
            with self._lock:
                if self._close_complete:
                    return
                # Fence new prepares immediately, but retain retryability until
                # every provider and owned local backend has actually closed.
                self._closed = True
                resources = list(self._async_resources)
            failures = []
            for resource in resources:
                close = getattr(resource, "aclose", None)
                if not callable(close):
                    continue
                try:
                    await close()
                except Exception as exc:
                    failures.append(exc)
            try:
                await run_local(self.close)
            except Exception as exc:
                failures.append(exc)
            if failures:
                raise RuntimeError("关闭标准资源时有异步客户端或页面后端未能正常退出。") from failures[0]
            with self._lock:
                self._close_complete = True
