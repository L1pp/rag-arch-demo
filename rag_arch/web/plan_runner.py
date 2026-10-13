"""真实方案复核/追问的 Web 任务执行器。

方案 Agent 的检索、取证、时间预算和报告校验仍由 PlanReviewAgent 负责；本
模块只注入 Web 任务的文档/规范资源，观察模型与工具调用，并把完整结果
留在本地 audit 中。
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
import functools
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from urllib.parse import urlencode

from langchain_core.tools import StructuredTool

from ..config import (DEFAULT_PLAN_MAX_STEPS, DEFAULT_PLAN_MAX_TOOL_CALLS,
                      DEFAULT_PLAN_CONTEXT_BUDGET)
from ..generation_experiments import safe_error
from ..graph_agent import _usage, _usage_total
from ..plan_review_agent import (DEFAULT_REVIEW_TASK, PLAN_TIME_LIMITS_SECONDS,
                                 PlanReviewAgent, PlanReviewResult)
from ..plan_review_report import report_markdown
from .catalog_tools import CatalogPlanTools as PlanStandardTools
from ..providers import create_web_search_provider
from ..runtime_audit import record_event, record_run
from ..tool_chat import create_tool_chat_model


_TECHNICAL_FAILURE_REASONS = frozenset({"model_error", "context_summary_failed"})


class _PlanCancelled(BaseException):
    """Escape PlanReviewAgent's ordinary Exception handlers on cancellation."""


class _PlanInputError(ValueError):
    """A user/resource input cannot support the requested plan action."""


class _PlanTechnicalFailure(RuntimeError):
    """The underlying Agent stopped because its model service failed."""


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _PlanCancelled("本轮方案任务已取消。")


def _notify(progress, cancel_event, stage, detail):
    _check_cancel(cancel_event)
    if callable(progress):
        try:
            progress(stage, detail)
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _PlanCancelled("本轮方案任务已取消。") from None
            raise
    _check_cancel(cancel_event)


def _wrapped_callable(name, function, progress, cancel_event):
    """Wrap a callable while retaining its signature for StructuredTool schemas."""
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        _notify(progress, cancel_event, "tool", f"正在调用方案工具 {name}")
        record_event("plan_tool_started", tool_name=name, arguments=deepcopy(kwargs))
        try:
            result = function(*args, **kwargs)
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _PlanCancelled("本轮方案任务已取消。") from None
            raise
        _check_cancel(cancel_event)
        record_event("plan_tool_returned", tool_name=name, result=deepcopy(result))
        _notify(progress, cancel_event, "tool", f"方案工具 {name}已返回")
        return result
    return invoke


def _observed_tool(tool, progress, cancel_event):
    name = getattr(tool, "name", None) or "plan_tool"
    description = getattr(tool, "description", None) or "读取本轮方案或规范资料。"
    args_schema = getattr(tool, "args_schema", None)
    response_format = getattr(tool, "response_format", "content")
    def invoke_tool(**kwargs):
        _notify(progress, cancel_event, "tool", f"正在调用方案工具 {name}")
        record_event("plan_tool_started", tool_name=name, arguments=deepcopy(kwargs))
        try:
            value = tool.invoke(deepcopy(kwargs))
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _PlanCancelled("本轮方案任务已取消。") from None
            raise
        _check_cancel(cancel_event)
        record_event("plan_tool_returned", tool_name=name, result=deepcopy(value))
        _notify(progress, cancel_event, "tool", f"方案工具 {name}已返回")
        return value
    options = {"func": invoke_tool, "name": name, "description": description,
               "return_direct": bool(getattr(tool, "return_direct", False)),
               "args_schema": args_schema, "infer_schema": args_schema is None}
    if response_format in {"content", "content_and_artifact"}:
        options["response_format"] = response_format
    return StructuredTool.from_function(**options)


class _ObservedPlanTools:
    def __init__(self, tools, progress, cancel_event):
        self._tools = tools
        self._progress = progress
        self._cancel_event = cancel_event

    def __getattr__(self, name):
        return getattr(self._tools, name)

    def as_langchain_tools(self):
        return [_observed_tool(tool, self._progress, self._cancel_event)
                for tool in self._tools.as_langchain_tools()]


class _ObservedDocument:
    """Proxy PlanDocument methods used both during setup and as Agent tools."""
    _METHODS = {"read", "search", "navigation"}

    def __init__(self, document, progress, cancel_event):
        self._document = document
        self._progress = progress
        self._cancel_event = cancel_event

    def __getattr__(self, name):
        value = getattr(self._document, name)
        if name in self._METHODS and callable(value):
            return _wrapped_callable(name, value, self._progress, self._cancel_event)
        return value


class _ObservedPageService:
    """Compact the Agent-facing source directory; retain full page operations."""
    _METHODS = {"list_pages", "read_page"}

    def __init__(self, service, progress, cancel_event):
        self._service = service
        self._progress = progress
        self._cancel_event = cancel_event

    def __getattr__(self, name):
        value = getattr(self._service, name)
        if name in self._METHODS and callable(value):
            return _wrapped_callable(name, value, self._progress, self._cancel_event)
        return value

    def sources(self):
        rows = self._service.sources()
        return [{key: row.get(key) for key in ("source_ref", "std_no", "filename")}
                for row in rows if isinstance(row, dict)]


class _ObservedWebProvider:
    def __init__(self, provider, progress, cancel_event):
        self._provider = provider
        self._progress = progress
        self._cancel_event = cancel_event
        self.available = getattr(provider, "available", False)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def as_langchain_tool(self):
        original = self._provider.as_langchain_tool()
        return _observed_tool(original, self._progress, self._cancel_event)


class _ObservedPlanModel:
    """Observe model calls and use BaseException cancellation semantics."""
    def __init__(self, model, progress, cancel_event, samples):
        self._model = model
        self._progress = progress
        self._cancel_event = cancel_event
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _wrap(self, model):
        return _ObservedPlanModel(model, self._progress, self._cancel_event, self._samples)

    def bind_tools(self, *args, **kwargs):
        return self._wrap(self._model.bind_tools(*args, **kwargs))

    def plain(self):
        plain = getattr(self._model, "plain", None)
        return self._wrap(plain() if callable(plain) else self._model)

    def invoke(self, *args, **kwargs):
        _notify(self._progress, self._cancel_event, "model",
                f"正在调用方案模型（第{len(self._samples) + 1}次）")
        position = len(self._samples)
        self._samples.append(None)
        response = self._model.invoke(*args, **kwargs)
        usage = _usage(SimpleNamespace(
            usage_metadata=getattr(response, "usage_metadata", None),
            response_metadata=getattr(response, "response_metadata", None) or {}))
        self._samples[position] = usage
        record_event("web_plan_model_returned", call_number=position + 1, token_usage=usage)
        _check_cancel(self._cancel_event)
        return response


def _factory_call(factory, settings, output_delivery, output_schema=None):
    kwargs = {"json_output": output_delivery == "json",
              "response_schema": output_schema, "output_method": output_delivery}
    if factory is create_tool_chat_model:
        kwargs["stream_responses"] = True
    return factory(settings, **kwargs)


def _plan_settings(settings):
    if settings is None:
        raise ValueError("方案 runner 需要配置对象。")
    return replace(settings, chat_max_tokens=settings.plan_max_tokens,
                   chat_thinking_budget=settings.plan_thinking_budget,
                   chat_timeout_seconds=settings.plan_timeout_seconds,
                   chat_context_budget_tokens=settings.plan_context_budget_tokens)


def _load_plan(documents, attachment_ids, registered_plan_id):
    getter = getattr(documents, "get_plan", None)
    if not callable(getter):
        raise ValueError("方案文档资源未提供get_plan。")
    bundle = getter(registered_plan_id=registered_plan_id, attachment_ids=attachment_ids)
    if bundle is None or not hasattr(bundle, "document"):
        raise ValueError("方案文档资源未返回document。")
    return bundle


def _history_rows(rows, document=None):
    selected = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        result = row.get("result")
        if not isinstance(result, dict) or result.get("simulation") is not False:
            continue
        history = result.get("report_history")
        if isinstance(history, dict):
            if document is not None and (
                    history.get("document_id") != getattr(document, "document_id", None)
                    or history.get("source_revision") != getattr(document, "source_revision", None)):
                continue
            selected.append(deepcopy(history))
    return selected


def _selected_evidence_ids(result):
    ids = list(getattr(result, "evidence_ids", []) or [])
    for finding in getattr(result, "findings", []) or []:
        if isinstance(finding, dict):
            ids.extend(finding.get("standard_evidence_ids", []) or [])
    return list(dict.fromkeys(eid for eid in ids if isinstance(eid, str) and eid))


def _citation_rows(result, resources, plan_page_url=None, plan_filename=None):
    by_id = {row.get("evidence_id"): row for row in getattr(result, "evidence", [])
             if isinstance(row, dict)}
    rows = []
    for eid in _selected_evidence_ids(result):
        row = by_id.get(eid)
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == "page_image":
            page_source = deepcopy(row.get("page_source", {}))
            is_plan_page = (row.get("source_kind") == "plan"
                            or str(page_source.get("source_ref", "")).startswith("plan:"))
            if is_plan_page:
                page_url = plan_page_url
                page = page_source.get("page")
                if page_url and type(page) is int and page > 0:
                    page_url += ("&" if "?" in page_url else "?") + urlencode({"page": page})
                rows.append({"evidence_id": eid, "kind": "page_image",
                             "title": plan_filename or page_source.get("filename"),
                             "std_no": None, "clause_no": None,
                             "text": row.get("text", ""), "page_url": page_url,
                             "history_url": None, "page_source": page_source,
                             "filename": plan_filename or page_source.get("filename")})
                continue
            mapped = resources.citation(deepcopy(row))
            if not isinstance(mapped, dict):
                raise ValueError("方案规范引用转换失败。")
            citation = {**mapped, "evidence_id": eid, "kind": kind,
                        "page_source": page_source}
            if page_source.get("filename"):
                citation["filename"] = page_source["filename"]
            rows.append(citation)
        elif kind in {"clause", "table"}:
            mapped = resources.citation(deepcopy(row))
            if not isinstance(mapped, dict):
                raise ValueError("方案规范引用转换失败。")
            rows.append({**mapped, "evidence_id": eid, "kind": kind})
        elif kind == "web":
            source = deepcopy(row.get("web_source", {}))
            rows.append({"evidence_id": eid, "kind": "web", **source})
        elif kind == "status":
            rows.append({key: deepcopy(row[key]) for key in (
                "evidence_id", "kind", "std_no", "clause_no", "text", "source_spans",
                "status_info") if key in row})
        else:
            rows.append({key: deepcopy(row[key]) for key in (
                "evidence_id", "kind", "std_no", "clause_no", "text", "source_spans",
                "complete", "page_source") if key in row})
    return rows


def _document_page_url(document, asset, documents, request):
    if Path(document.filename).suffix.lower() not in {".pdf", ".docx"}:
        return None
    registered = request.get("registered_plan_id")
    if isinstance(registered, str) and registered.strip():
        return "/viewer/pages?" + urlencode({"plan_id": registered.strip()})
    attachment_ids = request.get("attachment_ids") or []
    workspace_id = getattr(documents, "workspace_id", None)
    if attachment_ids and isinstance(workspace_id, str) and workspace_id:
        return "/viewer/pages?" + urlencode({"asset": attachment_ids[0],
                                               "workspace_id": workspace_id})
    return None


def _file_size(asset):
    path = asset.get("path") if isinstance(asset, dict) else None
    try:
        return Path(path).stat().st_size if path else 0
    except (OSError, TypeError, ValueError):
        return 0


def _public_document(document, asset, request, documents, page_url):
    metadata = deepcopy(document.metadata())
    metadata["page_url"] = page_url
    asset_id = (request.get("registered_plan_id") or
                (request.get("attachment_ids") or [None])[0] or document.document_id)
    files = [{"asset_id": asset_id, "filename": document.filename,
              "size_bytes": _file_size(asset), "page_url": page_url}]
    return metadata, files


def _public_result(output, plan_action, settings, audit, request, report_text, context_id,
                   resources, document, asset, documents):
    runtime = deepcopy(getattr(output, "runtime", {}) or {})
    report_format = getattr(output, "report_format", None) or runtime.get("report_format")
    report = {key: deepcopy(getattr(output, key)) for key in (
        "document", "answer", "findings", "status", "source_coverage",
        "reviewed_topics", "limitations", "validation_notes", "stop_reason")}
    if report_format:
        report["report_format"] = report_format
    history = output.as_history_turn()
    page_url = _document_page_url(document, asset, documents, request)
    public_document, files = _public_document(document, asset, request, documents, page_url)
    result = {
        "text": report_text,
        "simulation": False,
        "model": output.model or settings.chat_model,
        "answer": output.answer,
        "report_markdown": report_text if plan_action == "review" else None,
        "report": report if plan_action == "review" else None,
        "report_format": report_format,
        # Keep only the identifiers the agent explicitly returned as cited;
        # consulted evidence remains available through the audit result and
        # source/citation fields below.
        "evidence_ids": deepcopy(getattr(output, "evidence_ids", []) or []),
        "report_history": history,
        "citations": _citation_rows(output, resources, page_url, document.filename),
        "files": files,
        "page_sources": deepcopy(runtime.get("page_sources", [])),
        "web_sources": deepcopy(runtime.get("web_sources", [])),
        "native_tool_usage": {"counts": deepcopy(runtime.get("native_tool_counts", {}))},
        "warnings": deepcopy(output.limitations) + deepcopy(output.validation_notes),
        "status": output.status,
        "stop_reason": output.stop_reason,
        "source_coverage": deepcopy(output.source_coverage),
        "token_usage": deepcopy(output.token_usage),
        "audit_log": audit.info,
        "model_calls": output.model_calls,
        "tool_calls": output.tool_calls,
        "document": public_document,
        "context_id": context_id,
    }
    audit.set_result({"task_id": request.get("task_id"), "result": result,
                      "plan_result": output, "report": output.to_dict(),
                      "report_markdown": report_text})
    return result


def _persist_plan_index(resources, toolset):
    setter = getattr(resources, "set_plan_index", None)
    index = getattr(toolset, "index", None) if toolset is not None else None
    if index is not None and callable(setter):
        setter(index)


def _plan_before_norm_request(progress, cancel_event, started_at):
    """Guard each embedding batch/query without changing Agent wall-time flow."""
    _check_cancel(cancel_event)
    if started_at[0] is not None and monotonic() - started_at[0] >= PLAN_TIME_LIMITS_SECONDS[-1]:
        _notify(progress, cancel_event, "embedding", "已达到40分钟，不再新增规范向量请求")
        raise TimeoutError("已达到40分钟探索阈值，不再新增规范向量请求。")
    _notify(progress, cancel_event, "embedding", "正在准备规范向量")


def run_plan(request, progress, cancel_event, *, resources, documents,
             settings=None, model_factory=None):
    """Run one real plan review/question task with local audit persistence."""
    if not isinstance(request, dict):
        raise ValueError("方案任务请求须为对象。")
    attachment_ids = request.get("attachment_ids", [])
    if not isinstance(attachment_ids, list) or any(not isinstance(v, str) or not v.strip()
                                                   for v in attachment_ids):
        raise ValueError("attachment_ids须为字符串列表。")
    registered_plan_id = request.get("registered_plan_id")
    if registered_plan_id is not None and (not isinstance(registered_plan_id, str) or not registered_plan_id.strip()):
        raise ValueError("registered_plan_id须为非空文本或None。")
    plan_action = request.get("plan_action", "review")
    if plan_action not in {"review", "question"}:
        raise ValueError("plan_action须为review或question。")
    max_steps = request.get("max_steps")
    max_tool_calls = request.get("max_tool_calls")
    if max_steps is not None and (type(max_steps) is not int or not 3 <= max_steps <= 80):
        raise ValueError("max_steps须为3至80的整数或None。")
    if max_tool_calls is not None and (type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 200):
        raise ValueError("max_tool_calls须为1至200的整数或None。")
    question = DEFAULT_REVIEW_TASK if plan_action == "review" else request.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入方案问题。")
    if resources is None or not callable(getattr(resources, "plan_index", None)):
        raise ValueError("方案规范资源未注入plan_index。")
    if documents is None:
        raise ValueError("方案文档资源未注入。")
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("本轮方案任务已取消。")
    settings = _plan_settings(settings)
    _notify(progress, cancel_event, "reading", "正在读取方案文档")
    bundle = _load_plan(documents, attachment_ids, registered_plan_id)
    document, asset = bundle.document, getattr(bundle, "asset", None)
    context_id = getattr(bundle, "context_id", None)
    if document is None:
        raise _PlanInputError("未找到可用方案文档。")
    if plan_action == "review" and not getattr(document, "units", None):
        raise _PlanInputError("当前方案没有可读文字，整份核查已拒绝；这可能是扫描件或尚未完成OCR。")
    try:
        with record_run("web_plan_review", settings, question, task_id=request.get("task_id"),
                        document_id=getattr(document, "document_id", None),
                        source_revision=getattr(document, "source_revision", None),
                        plan_action=plan_action, context_id=context_id) as audit:
            output = None
            toolset = None
            samples = []
            run_started = [None]
            try:
                _notify(progress, cancel_event, "preparing", "正在准备方案和规范工具")
                plan_index = resources.plan_index()
                plan_index.candidate_pool = 30
                plan_index.rrf_constant = 60
                default_mode = ("hybrid" if getattr(plan_index, "embeddings", None) is not None
                                and settings.reranker_default_enabled and settings.reranker_ready else
                                "rrf" if getattr(plan_index, "embeddings", None) is not None else "bm25")
                before_norm_request = lambda: _plan_before_norm_request(
                    progress, cancel_event, run_started)
                toolset = PlanStandardTools(settings, index=plan_index,
                                            default_mode=default_mode, top_k=8,
                                            before_request=before_norm_request,
                                            persist_batches=False,
                                            allow_index_build=False,
                                            default_hybrid_fallback_to_rrf=True,
                                            # Keep the selected mode as the
                                            # initial preference while allowing
                                            # the natural-language agent to
                                            # choose search.mode/scope/filters.
                                            fixed_search_mode=None)
                toolset.catalog_index = resources.get().catalog
                toolset.catalog_request_check = before_norm_request
                toolset.begin_run()
                observed_tools = _ObservedPlanTools(toolset, progress, cancel_event)
                observed_document = _ObservedDocument(document, progress, cancel_event)
                pages = documents.pages_for_plan(document, asset,
                                                 getattr(plan_index, "corpus", None))
                _check_cancel(cancel_event)
                page_service = _ObservedPageService(pages, progress, cancel_event) if pages is not None else None
                # Production plan calls return readable Markdown directly.
                # create_tool_chat_model still binds local/native tools; the
                # text delivery only removes the forced submit_answer schema.
                output_delivery = "text"
                report_format = "markdown"
                model = _factory_call(model_factory or create_tool_chat_model,
                                      settings, output_delivery, None)
                if model is None:
                    raise ValueError("聊天模型未配置。")
                provider = create_web_search_provider(settings)
                if provider is not None:
                    provider = _ObservedWebProvider(provider, progress, cancel_event)
                observed_model = _ObservedPlanModel(model, progress, cancel_event, samples)
                agent = PlanReviewAgent(
                    observed_model, observed_document, observed_tools,
                    mode=plan_action,
                    max_steps=max_steps if max_steps is not None else (DEFAULT_PLAN_MAX_STEPS if plan_action == "review" else 8),
                    max_tool_calls=max_tool_calls if max_tool_calls is not None else (DEFAULT_PLAN_MAX_TOOL_CALLS if plan_action == "review" else 20),
                    progress_callback=lambda event: _agent_progress(event, progress, cancel_event),
                    report_format=report_format,
                    context_budget_tokens=settings.plan_context_budget_tokens,
                    allow_context_compaction=False,
                    context_summary_ratio=settings.plan_context_summary_ratio,
                    time_limits_seconds=PLAN_TIME_LIMITS_SECONDS,
                    page_service=page_service, web_provider=provider)
                run_started[0] = monotonic()
                output = agent.run(question, history=(
                    _history_rows(request.get("history"), document) if plan_action == "question" else []))
                _check_cancel(cancel_event)
                _notify(progress, cancel_event, "assembling", "正在整理方案报告和来源")
                markdown = (report_markdown(output)
                            if plan_action == "review"
                            else (output.answer.strip() if isinstance(output.answer, str)
                                  else report_markdown(output)))
                _persist_plan_index(resources, toolset)
                result = _public_result(output, plan_action, settings, audit, request,
                                        markdown, context_id, resources, document, asset, documents)
                if output.stop_reason in _TECHNICAL_FAILURE_REASONS and not (output.findings or output.raw_final_text):
                    raise _PlanTechnicalFailure("方案 Agent 技术调用失败，未生成可用结论。")
                return result
            except BaseException as exc:
                _persist_plan_index(resources, toolset)
                cancelled = cancel_event is not None and cancel_event.is_set()
                if cancelled or isinstance(exc, (CancelledError, _PlanCancelled)):
                    payload = {"task_id": request.get("task_id"), "status": "cancelled",
                               "simulation": False, "model": settings.chat_model,
                               "token_usage": (deepcopy(output.token_usage)
                                                if output is not None else _usage_total(samples)),
                               "audit_log": audit.info}
                    if output is not None:
                        payload["plan_result"] = output
                    audit.set_result(payload)
                    raise CancelledError("本轮方案任务已取消。") from None
                if isinstance(exc, _PlanInputError):
                    error = {"component": "plan", "code": "input_unreadable", "http_status": None,
                             "message": str(exc), "stop_service": False}
                elif isinstance(exc, _PlanTechnicalFailure):
                    error = {"component": "plan", "code": "execution_failed", "http_status": None,
                             "message": str(exc), "stop_service": False}
                else:
                    error = safe_error(exc if isinstance(exc, Exception) else Exception(type(exc).__name__),
                                       component="plan")
                payload = {"task_id": request.get("task_id"), "status": "failed",
                           "simulation": False, "model": settings.chat_model,
                           "error": error, "audit_log": audit.info}
                if output is not None:
                    payload.update({"plan_result": output, "report": output.to_dict(),
                                    "traces": deepcopy(output.traces),
                                    "token_usage": deepcopy(output.token_usage)})
                audit.set_result(payload)
                raise RuntimeError(json.dumps(error, ensure_ascii=False)) from None
    except _PlanInputError:
        raise


def _agent_progress(event, progress, cancel_event):
    if not isinstance(event, dict):
        _notify(progress, cancel_event, "plan", str(event))
        return
    phase = event.get("phase", "plan")
    if phase == "visual_directory":
        _notify(progress, cancel_event, "preparing", "正在定位疑似图表页；需要时才读取页图")
        return
    if phase == "review_batch":
        _notify(progress, cancel_event, "review_batch",
                f"正文分批审阅：第{min(event['batch'], event['total_batches'])}/{event['total_batches']}批；随后整理全案发现")
        return
    name = event.get("name") or event.get("status") or "方案 Agent"
    step = event.get("step")
    detail = f"第{step}轮：{name}" if step is not None else str(name)
    _notify(progress, cancel_event, str(phase), detail)
