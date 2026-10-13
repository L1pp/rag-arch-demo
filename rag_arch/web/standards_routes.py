"""Small read-only HTTP surface for catalog, retrieval and original-page viewing."""
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ..standard_tools import StandardTools


def create_standards_router():
    router = APIRouter(prefix="/api/standards", tags=["standards"])

    def resources(request):
        return request.app.state.standards

    def read(operation, *, page_view=False):
        try:
            return operation()
        except KeyError as exc:
            raise HTTPException(404, ("当前页面暂不支持：" if page_view else "") + str(exc.args[0])) from None
        except ValueError as exc:
            raise HTTPException(422, ("当前页面暂不支持：" if page_view else "") + str(exc)) from None
        except OSError:
            raise HTTPException(503, ("当前页面暂不支持：" if page_view else "") + "本地资料暂不可读，请检查原件是否仍在登记位置。") from None
        except Exception:
            if not page_view:
                raise
            logging.getLogger(__name__).exception("Original-page viewer could not load the requested page")
            raise HTTPException(503, "当前页面暂不支持：分页或渲染未完成，请稍后重试。") from None

    @router.get("/catalog")
    def catalog(request: Request):
        entries = read(lambda: resources(request).catalog_entries())
        return {"entries": entries, "total": len(entries)}

    @router.get("/candidates")
    def candidates(request: Request, query: str = Query(min_length=1, max_length=2000)):
        return read(lambda: resources(request).catalog_candidates(query))

    @router.get("/search")
    def search(request: Request, query: str = Query(min_length=1, max_length=2000),
               std: str | None = None):
        def perform():
            owner = resources(request)
            bundle = owner.get()
            filters = {"std_no": owner.resolve(std)["std_no"]} if std else None
            result = StandardTools(bundle.index, bundle.registry, default_mode="bm25").search(query, filters=filters)
            return {"ok": result.get("ok", True), "citations": [owner.citation(row) for row in result.get("evidence", [])],
                    "catalog_matches": [owner.catalog_entry(row) for row in result.get("catalog_matches", [])]}
        return read(perform)

    @router.get("/clause")
    def clause(request: Request, std: str, number: str):
        def perform():
            owner = resources(request)
            bundle = owner.get()
            result = StandardTools(bundle.index, bundle.registry, default_mode="bm25").get_clause(
                owner.resolve(std)["std_no"], number)
            return {**result, "citations": [owner.citation(row) for row in result.get("evidence", [])]}
        return read(perform)

    @router.get("/status")
    def status(request: Request, std: str):
        # Old versions can be present in relationship records without a catalog
        # entry or PDF; do not force status/history through original availability.
        def perform():
            owner = resources(request)
            result = owner.get().registry.check(std)
            if not result.get("found"):
                candidates = owner.catalog_candidates(std)
                result = {**result, "catalog_candidates": candidates["matches"],
                          "candidate_warnings": candidates["warnings"]}
            return result
        return read(perform)

    @router.get("/pages")
    def pages(request: Request, std: str, evidence: str | None = None,
              source_ref: str | None = None, page: int | None = Query(default=None, ge=1)):
        return read(lambda: resources(request).page_view(std, evidence=evidence, source_ref=source_ref, page=page), page_view=True)

    @router.get("/page-image")
    def page_image(request: Request, std: str, source_ref: str, page: int = Query(ge=1)):
        image = read(lambda: resources(request).page_image(std, source_ref, page), page_view=True)
        return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    return router
