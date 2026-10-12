"""Shared whole-plan runtime; vector services are initialized only on demand."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT
from .evidence_index import EvidenceIndex, load_evidence_index
from .standard_status import StandardStatusRegistry
from .standard_tools import StandardTools, _service_error_metadata


class PlanStandardTools(StandardTools):
    def __init__(self, settings, *, root: Path = ROOT, index=None, top_k: int = 4,
                 default_mode: str = "bm25",
                 embedding_cache_path: Path | None = None, before_request=None,
                 persist_batches: bool = True, allow_index_build: bool = True,
                 default_hybrid_fallback_to_rrf: bool = False,
                 fixed_search_mode: str | None = None):
        if before_request is not None and not callable(before_request):
            raise TypeError("before_request须为可调用对象或 None。")
        if type(persist_batches) is not bool:
            raise TypeError("persist_batches须为布尔值。")
        if type(allow_index_build) is not bool:
            raise TypeError("allow_index_build须为布尔值。")
        if fixed_search_mode not in {None, "bm25", "dense", "rrf", "hybrid"}:
            raise ValueError("fixed_search_mode须为bm25、dense、rrf、hybrid或None。")
        if fixed_search_mode is not None and default_mode != fixed_search_mode:
            raise ValueError("锁定检索模式时default_mode必须与fixed_search_mode相同。")
        self.settings = settings
        self.root = root
        self.embedding_cache_path = (Path(embedding_cache_path) if embedding_cache_path is not None
                                     else self.root / "data/runtime/core_embeddings.json")
        self.before_request = before_request
        self.persist_batches = persist_batches
        self.allow_index_build = allow_index_build
        self.fixed_search_mode = fixed_search_mode
        self.tool_profile_id = (f"local_{fixed_search_mode}_rag"
                                if fixed_search_mode is not None else "local_rag")
        index = index if index is not None else load_evidence_index(root / "data/corpus")
        corpus = index.corpus
        registry = StandardStatusRegistry(root, getattr(corpus, "clauses", []),
            catalog=getattr(corpus, "catalog", []), relations=getattr(corpus, "relations", []))
        super().__init__(index, registry, default_mode=default_mode, top_k=top_k,
                         default_hybrid_fallback_to_rrf=default_hybrid_fallback_to_rrf)
        self._update_embedding_runtime_options()
        self.setup_events = []
        # These are scoped to this PlanStandardTools instance, which is the
        # lifetime of one plan-review conversation/run.  They are diagnostic
        # service states, not permanent configuration changes.
        self._disabled_services: dict[str, dict] = {}
        self._run_setup_event_start = len(self.setup_events)

    def library_catalog(self) -> list[dict]:
        """Discover A/B/C records without treating names or tags as clauses."""
        entries = super().library_catalog()
        if entries:
            return entries
        # Legacy injected corpora without a directory retain their prior names.
        from .standard_status import _key
        loaded = {_key(row.get("std_no", "")) for row in self.index.corpus.clauses}
        return [{"std_no": row["std_no"], "title": row.get("title", "")}
                for row in self.status_registry.standards if _key(row.get("std_no", "")) in loaded]

    def embedding_cache_metadata(self) -> dict:
        """Return the selected cache path and current wrapper state."""
        cache = getattr(self.index, "embeddings", None)
        actual = Path(getattr(cache, "cache_path", None) or self.embedding_cache_path)
        return {"path": str(actual), "resolved_path": str(actual.resolve()),
                "used": cache is not None,
                "load_status": getattr(cache, "load_status", None),
                "stats": getattr(cache, "stats", None)}

    def begin_run(self) -> None:
        """Start a logical Agent run without rebuilding local or remote-backed objects.

        Provider failures disable dependent modes only for the current run. The
        loaded index, embedding cache and reranker object stay attached, while
        setup_events remains historical diagnostic information.
        """
        self._disabled_services = {}
        self._run_setup_event_start = len(self.setup_events)

    @property
    def current_run_setup_events(self) -> list[dict]:
        """Return setup events recorded since the most recent begin_run call."""
        start = min(max(self._run_setup_event_start, 0), len(self.setup_events))
        return list(self.setup_events[start:])

    @property
    def corpus_id(self) -> str:
        """Identify the loaded local corpus from its own summary metadata only."""
        corpus = getattr(self.index, "corpus", None)
        summary = getattr(corpus, "summary", None)
        generated_at = summary.get("generated_at") if isinstance(summary, dict) else None
        if isinstance(generated_at, str) and generated_at.strip():
            return "core-local@" + generated_at.strip()
        return "unknown"
    @property
    def available_modes(self):
        modes = ["bm25"]
        disabled = getattr(self, "_disabled_services", {})
        vector_ready = (self.index.embeddings is not None or self.allow_index_build)
        if vector_ready and self.settings.embedding_ready and "embedding" not in disabled:
            modes.extend(["dense", "rrf"])
            if self.settings.reranker_ready and "reranker" not in disabled:
                modes.append("hybrid")
        if self.fixed_search_mode is not None:
            return [self.fixed_search_mode] if self.fixed_search_mode in modes else []
        return modes

    def _disabled_service_for_mode(self, mode: str) -> str | None:
        disabled = getattr(self, "_disabled_services", {})
        if not isinstance(mode, str):
            return None
        if mode in {"dense", "rrf"} and "embedding" in disabled:
            return "embedding"
        if mode == "hybrid":
            if "embedding" in disabled:
                return "embedding"
            if "reranker" in disabled:
                return "reranker"
        return None

    def _disable_service(self, service_error: dict | None) -> None:
        if not isinstance(service_error, dict):
            return
        service = service_error.get("service")
        if service not in {"embedding", "reranker"}:
            return
        disabled = getattr(self, "_disabled_services", None)
        if not isinstance(disabled, dict):
            disabled = {}
            self._disabled_services = disabled
        # Keep the first bounded diagnostic for this run; no exception object
        # or provider response body is retained.
        disabled.setdefault(service, dict(service_error))

    def _update_embedding_runtime_options(self) -> None:
        cache = getattr(self.index, "embeddings", None)
        if (cache is not None and hasattr(cache, "before_request")
                and hasattr(cache, "persist_batches")):
            cache.before_request = self.before_request
            cache.persist_batches = self.persist_batches

    def _available_mode_error(self, selected: str) -> dict:
        blocked = self._disabled_service_for_mode(selected)
        extra = {"available_modes": self.available_modes}
        if blocked is not None:
            extra.update({"mode_disabled_for_run": selected, "disabled_service": blocked})
        return extra

    def _enable(self, mode):
        from .providers import create_embeddings, create_reranker
        from .embedding_cache import CachedEmbeddings
        self._update_embedding_runtime_options()
        if mode in {"dense", "rrf", "hybrid"} and self.index.embeddings is None:
            if not self.allow_index_build:
                raise ValueError("本轮未加载已准备的文档向量索引；请先运行规范向量预建任务。")
            if not self.settings.embedding_ready:
                raise ValueError("当前未配置向量服务。")
            provider = create_embeddings(self.settings)
            cache_path = self.embedding_cache_path
            cache = CachedEmbeddings(provider, cache_path,
                {"base_url": self.settings.embedding_base_url.rstrip("/"),
                 "model": self.settings.embedding_model,
                 "dimensions": getattr(provider, "dimensions", self.settings.embedding_dimensions)},
                batch_size=self.settings.embedding_batch_size,
                before_request=self.before_request,
                persist_batches=self.persist_batches)
            self.index = EvidenceIndex(self.index.corpus, embeddings=cache, reranker=self.index.reranker)
            self.setup_events.append({"component": "embedding", "cache_path": str(cache_path),
                                      "cache_path_resolved": str(cache_path.resolve()),
                                      "cache_status": cache.load_status, "cache_stats": dict(cache.stats)})
        if mode == "hybrid" and self.index.reranker is None:
            if not self.settings.reranker_ready:
                raise ValueError("当前未配置重排服务。")
            reranker = create_reranker(self.settings)
            from .local_reranking import LocalReranker
            identity = ({"provider": "bailian", "model": self.settings.reranker_model}
                        if not isinstance(reranker, LocalReranker) else reranker.prepare())
            self.index.reranker = reranker
            self.setup_events.append({"component": "reranker", **identity})

    def search(self, query: str, mode: str | None = None, filters: dict | str | None = None,
               scope: str = "all") -> dict:
        if self.fixed_search_mode is not None and mode not in {None, self.fixed_search_mode}:
            return self._error("search", "mode_locked",
                               f"本轮规范检索固定为{self.fixed_search_mode}；未执行请求的{mode}模式。",
                               requested_mode=mode, effective_mode=self.fixed_search_mode,
                               available_modes=self.available_modes)
        selected = self.fixed_search_mode or (self.default_mode if mode is None else mode)
        filters, decoding_error = self._decode_filter_object(filters)
        invalid_filters = decoding_error or self._validate_filters(filters)
        if invalid_filters:
            return self._error("search", "invalid_filters", invalid_filters)
        if not self._text_arguments(query=query):
            return self._error("search", "invalid_argument", "query须为非空文本。")
        if not isinstance(scope, str) or scope not in {"all", "evidence", "catalog"}:
            return self._error("search", "invalid_scope", "scope须为all、evidence或catalog。")
        from .catalog_runtime import explicit_standard_requests
        requested = explicit_standard_requests(query)
        selected_codes = filters.get("std_nos", [filters["std_no"]] if filters.get("std_no") else []) if filters else []
        selected_codes = [selected_codes] if isinstance(selected_codes, str) else selected_codes
        bound_codes = selected_codes or requested
        no_text = bool(bound_codes and self.catalog_index.records) and all(
            not self.catalog_index.get(name)["matches"] or
            all(not row["text_available"] for row in self.catalog_index.get(name)["matches"])
            for name in bound_codes)
        if scope == "catalog" or no_text:
            return super().search(query, selected, filters, scope=scope)
        # Invalid modes/filters use the base validator without starting services.
        if selected not in self.available_modes:
            return self._error("search", "mode_unavailable",
                               f"本轮可用检索模式为{','.join(self.available_modes)}；所选{selected}未运行。",
                               **self._available_mode_error(selected))
        try:
            self._enable(selected)
        except Exception as exc:
            service_error = _service_error_metadata(exc, mode=selected)
            self._disable_service(service_error)
            event = {"component": selected, "status": "error", "error_type": type(exc).__name__}
            if service_error is not None:
                event["service_error"] = dict(service_error)
            self.setup_events.append(event)
            extra = {"available_modes": self.available_modes}
            if service_error is not None:
                extra["service_error"] = dict(service_error)
                extra.update({"mode_disabled_for_run": selected,
                              "disabled_service": service_error["service"]})
            if (self.default_hybrid_fallback_to_rrf and selected == "hybrid"
                    and isinstance(service_error, dict)
                    and service_error.get("service") == "reranker"
                    and getattr(self.index, "embeddings", None) is not None):
                result = super().search(query, "rrf", filters, scope=scope)
                result.update({"requested_mode": mode,
                               "effective_mode": "hybrid",
                               "default_mode": self.default_mode,
                               "executed_mode": "rrf",
                               "reranker_fallback": True,
                               "fallback_reason": "reranker_setup_error",
                               "service_error": dict(service_error),
                               "available_modes": self.available_modes,
                               "mode_disabled_for_run": "hybrid",
                               "disabled_service": "reranker"})
                return result
            return self._error("search", "retrieval_setup_failed",
                               "所选模式初始化失败；未静默改为其他模式。可根据当前任务明确选用其他可用方法。",
                               **extra)
        result = super().search(query, selected, filters, scope=scope)
        service_error = result.get("service_error") if isinstance(result, dict) else None
        if isinstance(service_error, dict):
            self._disable_service(service_error)
            result["available_modes"] = self.available_modes
            result["mode_disabled_for_run"] = selected
            result["disabled_service"] = service_error.get("service")
        return result
