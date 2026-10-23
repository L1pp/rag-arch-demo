"""Directory discovery assists an unsuccessful identity lookup without replacing it."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from ..async_support import run_local
from ..evidence_index import EvidenceIndex
from ..plan_review_runtime import PlanStandardTools
from ..standard_tools import StandardTools

_request_check = ContextVar("catalog_request_check", default=None)


def check_catalog_request():
    callback = _request_check.get()
    if callback is not None:
        callback()


@contextmanager
def catalog_request_scope(callback):
    token = _request_check.set(callback)
    try:
        yield
    finally:
        _request_check.reset(token)


class _CandidateLookup:
    def discover(self, query, filters=None, limit=None):
        with catalog_request_scope(getattr(self, "catalog_request_check", None)):
            return super().discover(query, filters, limit)

    async def adiscover(self, query, filters=None, limit=None):
        with catalog_request_scope(getattr(self, "catalog_request_check", None)):
            return await super().adiscover(query, filters, limit)

    def check_standard_status(self, std_no: str, clause_no: str | None = None,
                              detail: Literal["summary", "history", "sources"] = "summary",
                              offset: int = 0, limit: int = 10):
        result = super().check_standard_status(std_no, clause_no, detail=detail,
                                               offset=offset, limit=limit)
        if result.get("ok") is not True or result.get("found"):
            return result
        candidates = self.discover(std_no, limit=5)
        if not candidates:
            return result
        return {**result, "status": "catalog_only", "catalog_found": True,
                "catalog_candidates": candidates,
                "candidate_note": "精确状态查询未确认该写法。以下仅为目录候选，请核对编号、年份及分册；候选不证明同版或替代关系。"}

    async def acheck_standard_status(self, std_no: str, clause_no: str | None = None,
                                     detail: Literal["summary", "history", "sources"] = "summary",
                                     offset: int = 0, limit: int = 10):
        result = await run_local(super().check_standard_status, std_no, clause_no,
                                 detail=detail, offset=offset, limit=limit)
        if result.get("ok") is not True or result.get("found"):
            return result
        candidates = await self.adiscover(std_no, limit=5)
        if not candidates:
            return result
        return {**result, "status": "catalog_only", "catalog_found": True,
                "catalog_candidates": candidates,
                "candidate_note": "精确状态查询未确认该写法。以下仅为目录候选，请核对编号、年份及分册；候选不证明同版或替代关系。"}


class CatalogStandardTools(_CandidateLookup, StandardTools):
    pass


class CatalogPlanTools(_CandidateLookup, PlanStandardTools):
    async def _aenable(self, mode):
        from ..providers import create_embeddings, create_reranker
        from ..embedding_cache import CachedEmbeddings
        from ..local_reranking import LocalReranker

        self._update_embedding_runtime_options()
        if mode in {"dense", "rrf", "hybrid"} and self.index.embeddings is None:
            if not self.allow_index_build:
                raise ValueError("本轮未加载已准备的文档向量索引；请先运行规范向量预建任务。")
            if not self.settings.embedding_ready:
                raise ValueError("当前未配置向量服务。")

            def create_cache():
                provider = create_embeddings(self.settings)
                cache_path = self.embedding_cache_path
                cache = CachedEmbeddings(provider, cache_path,
                    {"base_url": self.settings.embedding_base_url.rstrip("/"),
                     "model": self.settings.embedding_model,
                     "dimensions": getattr(provider, "dimensions", self.settings.embedding_dimensions)},
                    batch_size=self.settings.embedding_batch_size,
                    before_request=self.before_request,
                    persist_batches=self.persist_batches)
                return cache_path, cache

            cache_path, cache = await run_local(create_cache)
            self.index = await EvidenceIndex.acreate(
                self.index.corpus, embeddings=cache, reranker=self.index.reranker)
            self.setup_events.append({"component": "embedding", "cache_path": str(cache_path),
                                      "cache_path_resolved": str(cache_path.resolve()),
                                      "cache_status": cache.load_status,
                                      "cache_stats": dict(cache.stats)})
        if mode == "hybrid" and self.index.reranker is None:
            if not self.settings.reranker_ready:
                raise ValueError("当前未配置重排服务。")
            reranker = await run_local(create_reranker, self.settings)
            if isinstance(reranker, LocalReranker):
                identity = await reranker.aprepare()
            else:
                identity = {"provider": "bailian", "model": self.settings.reranker_model}
            self.index.reranker = reranker
            self.setup_events.append({"component": "reranker", **identity})

    async def asearch(self, query: str, mode: str | None = None, filters=None,
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
        from ..catalog_runtime import explicit_standard_requests
        requested = await run_local(explicit_standard_requests, query)
        selected_codes = filters.get("std_nos", [filters["std_no"]]
                                     if filters.get("std_no") else []) if filters else []
        selected_codes = [selected_codes] if isinstance(selected_codes, str) else selected_codes
        bound_codes = selected_codes or requested

        def has_no_text():
            return bool(bound_codes and self.catalog_index.records) and all(
                not self.catalog_index.get(name)["matches"] or
                all(not row["text_available"] for row in self.catalog_index.get(name)["matches"])
                for name in bound_codes)

        no_text = await run_local(has_no_text)
        if scope == "catalog" or no_text:
            return await super().asearch(query, selected, filters, scope=scope)
        if selected not in self.available_modes:
            return self._error("search", "mode_unavailable",
                               f"本轮可用检索模式为{','.join(self.available_modes)}；所选{selected}未运行。",
                               **self._available_mode_error(selected))
        try:
            await self._aenable(selected)
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
                result = await super().asearch(query, "rrf", filters, scope=scope)
                result.update({"requested_mode": mode, "effective_mode": "hybrid",
                               "default_mode": self.default_mode, "executed_mode": "rrf",
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
        result = await super().asearch(query, selected, filters, scope=scope)
        service_error = result.get("service_error") if isinstance(result, dict) else None
        if isinstance(service_error, dict):
            self._disable_service(service_error)
            result["available_modes"] = self.available_modes
            result["mode_disabled_for_run"] = selected
            result["disabled_service"] = service_error.get("service")
        return result
