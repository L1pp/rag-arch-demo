"""Upload immutable originals and browse only registered files in this workspace."""
import logging
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from ..config import DEFAULT_PLAN_MAX_STEPS, DEFAULT_PLAN_MAX_TOOL_CALLS
from ..loaders import MAX_UPLOAD_BYTES, validate_upload_sizes
from .tenancy import check_workspace, current_documents
from .page_region_query import parse_region_query, region_query_value


def validate_task_documents(body, store):
    """Metadata only here; parsing and model work belong to the task worker."""
    if len(set(body.attachment_ids)) != len(body.attachment_ids):
        raise ValueError("同一附件不能重复选择。")
    files = [store.get(asset_id) for asset_id in body.attachment_ids]
    validate_upload_sizes(asset.size_bytes for asset in files)
    if body.assistant_id == "standards" and files:
        raise ValueError("规范问答使用已入库资料；上传附件请使用聊天或方案助手。")
    if body.assistant_id != "plan":
        if body.registered_plan_id or body.plan_action != "question":
            raise ValueError("已登记方案及完整核查操作仅供方案助手使用。")
    elif body.registered_plan_id:
        if files:
            raise ValueError("请选择已登记方案或上传方案中的一种。")
        if body.registered_plan_id not in {row["plan_id"] for row in store.registered_plans()}:
            raise ValueError("请选择已有的公开登记方案。")
    elif len(files) != 1:
        raise ValueError("方案助手需要选择一份已登记方案或一个上传文件。")


def create_document_router():
    router = APIRouter(tags=["documents"])

    def asset_view(row, workspace_id):
        return {**row, "page_url": "/viewer/pages?" + urlencode({
            "asset": row["asset_id"], "workspace_id": workspace_id})
            if Path(row["filename"]).suffix.lower() in {".pdf", ".doc", ".docx"} else None}

    def read(operation, *, page=False):
        try:
            return operation()
        except HTTPException:
            raise
        except KeyError as exc:
            raise HTTPException(404, ("当前页面暂不支持：" if page else "") + str(exc.args[0])) from None
        except ValueError as exc:
            raise HTTPException(422, ("当前页面暂不支持：" if page else "") + str(exc)) from None
        except OSError:
            raise HTTPException(503, "当前页面暂不支持：登记文件暂不可读取。" if page else "文件暂不可读取或保存，请稍后重试。") from None
        except Exception:
            if not page:
                raise
            logging.getLogger(__name__).exception("Uploaded or plan page rendering failed")
            raise HTTPException(503, "当前页面暂不支持：分页或渲染未完成，请稍后重试。") from None

    @router.get("/api/uploads")
    def uploads(request: Request, workspace_id: str, response: Response):
        store = current_documents(request)
        check_workspace(request, workspace_id)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Cookie"
        return {"uploads": [asset_view(row, workspace_id) for row in store.list()]}

    @router.post("/api/uploads", status_code=201, openapi_extra={"requestBody": {
        "required": True, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
    async def upload(request: Request, workspace_id: str, response: Response,
                     filename: str = Query(min_length=1, max_length=255)):
        store = current_documents(request)
        check_workspace(request, workspace_id)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Cookie"
        content_length = request.headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "单个文件不能超过 10 MB。")
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "单个文件不能超过 10 MB。")
            content.extend(chunk)
        result = await run_in_threadpool(lambda: read(lambda: store.add(filename, bytes(content))))
        return asset_view(result, workspace_id)

    @router.get("/api/uploads/{asset_id}/download")
    def original(asset_id: str, request: Request, workspace_id: str):
        store = current_documents(request)
        check_workspace(request, workspace_id)
        asset = read(lambda: store.get(asset_id))
        return FileResponse(asset.path, filename=asset.filename,
                            headers={"Cache-Control": "private, no-store", "Vary": "Cookie"})

    @router.get("/api/plans")
    def plans(request: Request):
        store = current_documents(request)
        rows = read(lambda: store.registered_plans())
        return {"plans": [{**row, "page_url": "/viewer/pages?" + urlencode({"plan_id": row["plan_id"]})
                           if Path(row["filename"]).suffix.lower() in {".pdf", ".doc", ".docx"} else None} for row in rows],
                "defaults": {"max_steps": DEFAULT_PLAN_MAX_STEPS, "max_tool_calls": DEFAULT_PLAN_MAX_TOOL_CALLS}}

    def page_service(request, store, asset, plan_id, workspace_id):
        if bool(asset) == bool(plan_id):
            raise ValueError("请选择一个上传文件或一份已登记方案。")
        if asset:
            check_workspace(request, workspace_id)
            source = store.get(asset)
            service = store.pages_for_asset(asset)
            if service is None:
                raise ValueError("此文件不是可分页查看的 PDF、DOC 或 DOCX 原件。")
            return service, source.filename, "上传资料"
        plan = store.get_plan(registered_plan_id=plan_id)
        return store.pages_for_plan(plan.document, plan.asset, corpus=None), plan.document.filename, plan_id

    @router.get("/api/documents/pages")
    def pages(request: Request, response: Response, asset: str | None = None, plan_id: str | None = None,
              workspace_id: str | None = None, source_ref: str | None = None,
              page: int = Query(default=1, ge=1), region: str | None = Query(default=None, max_length=200)):
        store = current_documents(request)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Cookie"

        def perform():
            service, title, code = page_service(request, store, asset, plan_id, workspace_id)
            selected_region = parse_region_query(region)
            sources = service.sources()
            if not sources:
                raise ValueError("此来源暂不支持分页图像；PDF、DOC 和 DOCX 原件可按页查看。")
            selected = next((row for row in sources if row["source_ref"] == source_ref), None) if source_ref else sources[0]
            if selected is None:
                raise KeyError("所选来源未登记。")
            ref = selected["source_ref"]
            info = service.list_pages(ref, start=page, limit=1)
            query = {"source_ref": ref, "page": page}
            if selected_region is not None:
                query["region"] = region_query_value(selected_region)
            query.update({"asset": asset, "workspace_id": workspace_id} if asset else {"plan_id": plan_id})
            return {"std_no": code, "title": title, "filename": selected["filename"], "source_ref": ref,
                    "page_count": info["page_count"],
                    "sources": [{"source_ref": row["source_ref"], "filename": row["filename"], "pages": []} for row in sources],
                    "pages": [{"page": page, "region": selected_region,
                               "image_url": "/api/documents/page-image?" + urlencode(query)}],
                    "warning": "PDF使用原件物理页码；DOC和DOCX使用后台排版后的物理页码，不推测段落对应页。"}
        return read(perform, page=True)

    @router.get("/api/documents/page-image")
    def image(request: Request, source_ref: str, page: int = Query(ge=1), asset: str | None = None,
              plan_id: str | None = None, workspace_id: str | None = None,
              region: str | None = Query(default=None, max_length=200)):
        store = current_documents(request)

        def perform():
            service, _, _ = page_service(request, store, asset, plan_id, workspace_id)
            if not any(row.get("source_ref") == source_ref for row in service.sources()):
                raise KeyError("所选来源未登记。")
            selected_region = parse_region_query(region)
            return service.read_page(source_ref, page,
                **({"region": selected_region} if selected_region is not None else {}))["image_bytes"]
        data = read(perform, page=True)
        return Response(data, media_type="image/png",
                        headers={"Cache-Control": "private, no-store", "Vary": "Cookie"})

    return router
