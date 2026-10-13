"""One read-only corpus per backend; original pages are loaded only on demand."""
from dataclasses import dataclass
from copy import copy, deepcopy
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from urllib.parse import urlencode

from ..catalog_index import CatalogIndex
from ..evidence_index import EvidenceIndex
from ..page_sources import _locations, standard_page_sources
from ..standard_status import StandardStatusRegistry
from ..standards import load_corpus

ROOT = Path(__file__).resolve().parents[2]


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

    def embed_query(self, text):
        self._before()
        return self.provider.embed_query(text)

    def clone(self):
        return _PreparedEmbeddings(self.cache)


class StandardsResources:
    """No reload while tasks run: restarting the backend adopts a new corpus.

    The lock only coordinates initial resource construction. PDF work is delegated
    to one process, not performed while holding this resource lock.
    """

    def __init__(self, root=ROOT, *, corpus=None, pdf_backend=None):
        self.root = Path(root)
        self._lock = RLock()
        self._corpus = corpus
        self._catalog = None
        self._bundle = None
        self._page_services = {}
        self._pdf_backend = pdf_backend
        self._owns_pdf_backend = pdf_backend is None
        self._plan_index = None
        self._catalog_settings = None
        self._catalog_warning = None
        self._prepared_index = None
        self._embedding_warning = None

    def configure_catalog(self, settings):
        """Store provider configuration and load prepared metadata vectors if present."""
        with self._lock:
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

    def configure_embeddings(self, settings):
        """Load a complete document-vector cache without issuing document requests."""
        with self._lock:
            self._prepared_index = None
            self._embedding_warning = None
            if settings is None or not settings.embedding_ready:
                return
            try:
                from ..providers import create_embeddings
                from ..embedding_cache import CachedEmbeddings
                corpus, _ = self._inventory()
                provider = create_embeddings(settings)
                cache = CachedEmbeddings(
                    provider, self.root / "data/runtime/core_embeddings.json",
                    {"base_url": settings.embedding_base_url.rstrip("/"),
                     "model": settings.embedding_model,
                     "dimensions": getattr(provider, "dimensions", settings.embedding_dimensions)},
                    batch_size=settings.embedding_batch_size,
                    persist_batches=False, allow_document_requests=False)
                embeddings = _PreparedEmbeddings(cache)
                # EvidenceIndex asks for every corpus text here.  The cache's
                # document-request gate therefore validates readiness from the
                # actual cached texts, without trusting a stale build marker.
                self._prepared_index = EvidenceIndex(corpus, embeddings=embeddings)
            except Exception as exc:
                self._embedding_warning = (
                    f"规范文档向量未加载（{type(exc).__name__}）；本轮保留BM25，"
                    "请先完成规范向量预建。")

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
            return _PreparedEmbeddings(cache)
        except Exception as exc:
            self._catalog_warning = f"目录向量客户端未就绪（{type(exc).__name__}），本次可使用字面候选。"
            return None

    def _inventory(self):
        with self._lock:
            if self._corpus is None:
                self._corpus = load_corpus(self.root / "data/corpus")
            if self._catalog is None:
                self._catalog = CatalogIndex(self._corpus.catalog, embeddings=self._catalog_embeddings())
                self._prepare_catalog_embeddings()
            return self._corpus, self._catalog

    def _pdf(self):
        if self._pdf_backend is None:
            from ..pdf_backend import ProcessPdfBackend
            self._pdf_backend = ProcessPdfBackend()
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
            if self._bundle is None:
                corpus, catalog = self._inventory()
                self._bundle = StandardsBundle(
                    self._prepared_index or EvidenceIndex(corpus),
                    StandardStatusRegistry(self.root, corpus.clauses,
                                           catalog=corpus.catalog, relations=corpus.relations),
                    catalog, self.page_service())
            return self._bundle

    @property
    def pdf_backend(self):
        with self._lock:
            return self._pdf()

    def plan_index(self):
        """Share immutable BM25 data; keep plan vector/reranker state separate.

        TaskService allows one active plan task. Its lazy provider setup must not
        mutate the index that a concurrent standards question is reading.
        """
        with self._lock:
            if self._plan_index is None:
                base = self.get().index
                self._plan_index = copy(base)
                self._plan_index.local = copy(base.local)
                if base.embeddings is None:
                    self._plan_index.embeddings = self._plan_index.reranker = None
                    self._plan_index.local.embeddings = None
                    self._plan_index.local._vectors = None
                else:
                    self._plan_index.embeddings = base.embeddings.clone()
                    self._plan_index.reranker = None
                    self._plan_index.local.embeddings = self._plan_index.embeddings
            return self._plan_index

    def set_plan_index(self, index):
        with self._lock:
            if index.corpus is not self.get().index.corpus:
                raise ValueError("方案索引必须使用本次后端已加载的语料。")
            self._plan_index = index

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
        for row in corpus.clauses + corpus.tables:
            if (row.get("clause_id", row.get("table_id")) == evidence_id
                    and row["std_no"] == entry["std_no"]):
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
        if self._owns_pdf_backend and self._pdf_backend is not None:
            self._pdf_backend.close()
