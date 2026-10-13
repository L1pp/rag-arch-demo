"""Directory discovery assists an unsuccessful identity lookup without replacing it."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

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


class CatalogStandardTools(_CandidateLookup, StandardTools):
    pass


class CatalogPlanTools(_CandidateLookup, PlanStandardTools):
    pass
