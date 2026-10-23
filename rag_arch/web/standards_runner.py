"""真实规范 Agent 的 Web 任务执行器。

这个模块只负责把 Web 任务服务的输入接到现有的规范 Agent。规范库、状态
登记、目录和原件页服务由 ``StandardsResources`` 注入；本执行器不会自行
重新加载资料。已加载向量可用且生产开关启用时，只构造本轮任务级重排客户端。
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from copy import copy, deepcopy
import asyncio
import json

from langchain_core.tools import StructuredTool

from ..config import load_settings
from ..generation_experiments import safe_error
from ..graph_agent import StandardAgent, _usage_total
from ..output_schemas import STANDARD_ANSWER_SCHEMA, schema_for_model
from ..providers import create_reranker, create_web_search_provider
from ..runtime_audit import record_event, record_run, tenant_audit_metadata
from ..runtime_audit import arecord_event, arecord_run
from .catalog_tools import CatalogPlanTools as StandardTools
from ..tool_chat import create_tool_chat_model
from .chat_runner import _ObservedModel
from ..async_support import TaskResourceCleanupError, run_local


class _CancelledDuringTool(BaseException):
    """Escape StandardAgent's ordinary tool-error handler on cancellation."""


_TECHNICAL_FAILURE_REASONS = frozenset({"model_error", "tool_error"})


class _NavigationPageService:
    """Keep the Agent's source directory small while preserving page services."""

    def __init__(self, service):
        self._service = service

    def __getattr__(self, name):
        return getattr(self._service, name)

    def sources(self):
        rows = self._service.sources()
        return [{key: row.get(key) for key in ("source_ref", "std_no", "filename")}
                for row in rows if isinstance(row, dict)]


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _CancelledDuringTool("本轮规范问答已取消。")


def _notify(progress, cancel_event, stage, detail):
    _check_cancel(cancel_event)
    if callable(progress):
        try:
            progress(stage, detail)
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledDuringTool("本轮规范问答已取消。") from None
            raise
    _check_cancel(cancel_event)


def _observed_tools(standard_tools, progress, cancel_event):
    """Return a StandardTools-shaped object with cancellable LangChain tools.

    StandardAgent owns the tool protocol and error policy. Wrapping the
    already-created tools keeps that behavior intact while allowing the Web
    worker to notice cancellation immediately before and after local lookups.
    """
    class ObservedTools:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def as_langchain_tools(self):
            wrapped = []
            for tool in self._inner.as_langchain_tools():
                wrapped.append(_observed_tool(tool, progress, cancel_event))
            return wrapped

    return ObservedTools(standard_tools)


def _observed_tool(tool, progress, cancel_event):
    name = getattr(tool, "name", None) or "standard_tool"
    description = getattr(tool, "description", None) or "读取本地规范资料。"
    args_schema = getattr(tool, "args_schema", None)
    return_direct = bool(getattr(tool, "return_direct", False))
    response_format = getattr(tool, "response_format", "content")

    def invoke_tool(**kwargs):
        _notify(progress, cancel_event, "tool", f"正在调用规范工具 {name}")
        record_event("standards_tool_started", tool_name=name, arguments=deepcopy(kwargs))
        try:
            value = tool.invoke(deepcopy(kwargs))
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledDuringTool("本轮规范问答已取消。") from None
            raise
        _check_cancel(cancel_event)
        record_event("standards_tool_returned", tool_name=name, result=deepcopy(value))
        _notify(progress, cancel_event, "tool", f"规范工具 {name}已返回")
        return value

    kwargs = {
        "func": invoke_tool,
        "name": name,
        "description": description,
        "return_direct": return_direct,
        "args_schema": args_schema,
        "infer_schema": args_schema is None,
    }
    # ``response_format`` exists in the supported LangChain version, but keep
    # compatibility with older injected test doubles when it is absent.
    if response_format in {"content", "content_and_artifact"}:
        kwargs["response_format"] = response_format
    return StructuredTool.from_function(**kwargs)


def _async_observed_tool(tool, progress, cancel_event, *, require_coroutine=False):
    """Wrap a tool with a real coroutine; local sync work uses the bounded executor."""
    name = getattr(tool, "name", None) or "standard_tool"
    description = getattr(tool, "description", None) or "读取本地规范资料。"
    args_schema = getattr(tool, "args_schema", None)
    return_direct = bool(getattr(tool, "return_direct", False))
    response_format = getattr(tool, "response_format", "content")
    has_coroutine = callable(getattr(tool, "coroutine", None))
    network_named = any(token in name.lower() for token in ("web", "http", "online", "remote"))

    async def invoke_tool(**kwargs):
        await _aprogress(progress, cancel_event, "tool", f"正在调用规范工具 {name}")
        await arecord_event("standards_tool_started", tool_name=name, arguments=deepcopy(kwargs))
        try:
            if has_coroutine:
                value = await tool.ainvoke(deepcopy(kwargs))
            elif require_coroutine or network_named:
                raise RuntimeError(f"工具 {name} 未提供异步网络接口。")
            else:
                value = await run_local(tool.invoke, deepcopy(kwargs))
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledDuringTool("本轮规范问答已取消。") from None
            raise
        _check_cancel(cancel_event)
        await arecord_event("standards_tool_returned", tool_name=name, result=deepcopy(value))
        await _aprogress(progress, cancel_event, "tool", f"规范工具 {name}已返回")
        return value

    options = {"func": None, "coroutine": invoke_tool, "name": name,
               "description": description, "return_direct": return_direct,
               "args_schema": args_schema, "infer_schema": args_schema is None}
    if response_format in {"content", "content_and_artifact"}:
        options["response_format"] = response_format
    return StructuredTool.from_function(**options)


def _async_observed_tools(standard_tools, progress, cancel_event):
    class AsyncObservedTools:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def as_langchain_tools(self):
            return [_async_observed_tool(tool, progress, cancel_event)
                    for tool in self._inner.as_langchain_tools()]

    return AsyncObservedTools(standard_tools)


class _AsyncObservedWebProvider:
    def __init__(self, provider, progress, cancel_event):
        self._provider = provider
        self._progress = progress
        self._cancel_event = cancel_event
        self.available = getattr(provider, "available", False)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def as_langchain_tool(self):
        original = self._provider.as_langchain_tool()
        return _async_observed_tool(original, self._progress, self._cancel_event,
                                    require_coroutine=True)


async def _aprogress(progress, cancel_event, stage, detail):
    _check_cancel(cancel_event)
    if callable(progress):
        try:
            await run_local(progress, stage, detail)
        except CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledDuringTool("本轮规范问答已取消。") from None
            raise
    _check_cancel(cancel_event)


def _factory_call(factory, settings, schema):
    return factory(settings, json_output=True, response_schema=schema,
                   output_method="terminal_tool" if schema is not None else "json")


def _history_for_agent(rows, corpus_id):
    """Translate TaskService result rows into StandardAgent history records."""
    selected = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("assistant_id", "standards") != "standards":
            continue
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            continue
        result = row.get("result")
        if not isinstance(result, dict) or result.get("simulation") is not False:
            continue
        if result.get("corpus_id") != corpus_id:
            continue
        answer = result.get("text")
        # Catalog answers and evidence-gap explanations still resolve follow-up
        # references. The agent labels ALL history as unverified, never evidence.
        selected.append({"corpus_id": corpus_id, "question": row["question"],
                         "answer": answer if isinstance(answer, str) else ""})
    return selected


def _evidence_map(agent_result):
    return {record.get("evidence_id"): record for record in getattr(agent_result, "evidence", [])
            if isinstance(record, dict) and isinstance(record.get("evidence_id"), str)}


def _public_citation(citation, evidence, resources):
    """Map one model-selected citation without promoting unselected evidence."""
    evidence_id = citation.get("evidence_id") if isinstance(citation, dict) else None
    record = evidence.get(evidence_id, {})
    kind = citation.get("kind") if isinstance(citation, dict) else None
    kind = kind or record.get("kind")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("模型返回了无效的引用ID。")
    if kind in {"clause", "table"}:
        if not record:
            raise ValueError("引用对应的正文证据未保存在本轮记录中。")
        # resources.citation owns standard/version resolution and page URLs.
        mapped = resources.citation(deepcopy(record))
        if not isinstance(mapped, dict):
            raise ValueError("正文证据引用转换失败。")
        return mapped
    if kind == "catalog":
        info = deepcopy(record.get("catalog_info", {}))
        if not info:
            info = {"catalog_id": citation.get("catalog_id")}
        entry = resources.catalog_entry(info)
        if not isinstance(entry, dict):
            raise ValueError("目录引用转换失败。")
        return {**entry, "evidence_id": evidence_id, "kind": "catalog",
                "claim_scope": "metadata_only"}
    if kind == "status":
        info = deepcopy(record.get("status_info", {}))
        # Status records are not corpus clause citations. In particular, do
        # not pass a missing std_no through resources.citation().
        return {"evidence_id": evidence_id, "kind": "status",
                "std_no": record.get("std_no") or info.get("std_no"),
                "title": "规范状态登记", "clause_no": record.get("clause_no"),
                "text": record.get("text", ""), "page_url": None,
                "history_url": None, "source_spans": deepcopy(record.get("source_spans", []))}
    if kind == "web":
        source = deepcopy(citation.get("web_source") or record.get("web_source") or {})
        return {"evidence_id": evidence_id, "kind": "web", **source}
    if kind == "page_image":
        if not record:
            raise ValueError("引用对应的页图证据未保存在本轮记录中。")
        mapped = resources.citation(deepcopy(record))
        if not isinstance(mapped, dict):
            raise ValueError("页图证据引用转换失败。")
        page_source = deepcopy(citation.get("page_source") or record.get("page_source") or {})
        result = {**mapped, "evidence_id": evidence_id, "kind": "page_image",
                  "page_source": page_source}
        if isinstance(page_source.get("filename"), str) and page_source["filename"]:
            result["filename"] = page_source["filename"]
        return result
    raise ValueError("模型返回了不支持的引用类型。")


def _public_catalog_matches(agent_result, resources):
    output = []
    seen = set()
    for row in getattr(agent_result, "catalog_matches", []) or []:
        if not isinstance(row, dict):
            continue
        key = row.get("catalog_id") or row.get("std_id") or row.get("std_no")
        if key in seen:
            continue
        seen.add(key)
        entry = resources.catalog_entry(deepcopy(row))
        if not isinstance(entry, dict):
            raise ValueError("目录命中转换失败。")
        output.append(entry)
    return output


def _build_result(agent_result, resources, settings, audit, request):
    evidence = _evidence_map(agent_result)
    citations = [_public_citation(item, evidence, resources)
                 for item in getattr(agent_result, "citations", []) or []]
    catalog_matches = _public_catalog_matches(agent_result, resources)
    warnings = list(getattr(agent_result, "warnings", []) or [])
    status = getattr(agent_result, "status", "insufficient_evidence")
    termination_reason = getattr(agent_result, "termination_reason", None)
    if termination_reason == "citation_rejected":
        validation_warning = "引用未通过核验；当前保留的原文仅是模型输出，不能视为已验证结论。"
    elif status != "answered":
        validation_warning = "资料不足或输出未通过核验；当前回答不能视为有证据支持的结论。"
    else:
        validation_warning = None
    if validation_warning and validation_warning not in warnings:
        warnings.append(validation_warning)
    retrieved = len(evidence)
    if retrieved and not citations:
        warnings.append("本轮取得了检索资料，但模型未明确选择正式引用；这些资料仅用于本轮检索，未列为正式引用。")
    result = {
        "text": agent_result.text,
        "simulation": False,
        "model": settings.chat_model,
        "web_sources": deepcopy(getattr(agent_result, "web_sources", []) or []),
        "warnings": warnings,
        "native_tool_usage": deepcopy(getattr(agent_result, "native_tool_usage", {}) or {}),
        "audit_log": audit.info,
        "token_usage": deepcopy(getattr(agent_result, "token_usage", {}) or {}),
        "citations": citations,
        "catalog_matches": catalog_matches,
        "status": status,
        "termination_reason": termination_reason,
        "answer_scope": getattr(agent_result, "answer_scope", "evidence"),
        "tool_calls": getattr(agent_result, "tool_calls", 0),
        "model_calls": getattr(agent_result, "model_calls", 0),
        "corpus_id": getattr(agent_result, "corpus_id", None),
        "grounded": bool(getattr(agent_result, "grounded", False)),
    }
    audit.set_result({"task_id": request.get("task_id"), "result": result,
                      "agent_result": agent_result, "evidence": deepcopy(getattr(agent_result, "evidence", [])),
                      "traces": deepcopy(getattr(agent_result, "traces", []))})
    return result


def run_standards(request, progress, cancel_event, *, resources, settings=None, model_factory=None):
    """Run one real standards task in an independent local audit context."""
    if not isinstance(request, dict):
        raise ValueError("规范任务请求须为对象。")
    question = request.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入规范问题。")
    if resources is None or not callable(getattr(resources, "get", None)):
        raise ValueError("规范资料资源未注入。")
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("本轮规范问答已取消。")
    try:
        settings = settings if settings is not None else load_settings()
    except Exception as exc:
        raise RuntimeError(json.dumps(safe_error(exc, component="standards"), ensure_ascii=False)) from None
    question = question.strip()
    samples = []
    with record_run("web_standard_agent", settings, question,
                    **tenant_audit_metadata(request)) as audit:
        agent_output = None
        try:
            _notify(progress, cancel_event, "preparing", "正在准备规范资料和 Agent")
            bundle = resources.get()
            index = copy(bundle.index)
            index.local = copy(bundle.index.local)
            embeddings = getattr(index, "embeddings", None)
            clone = getattr(embeddings, "clone", None)
            if embeddings is not None:
                if callable(clone):
                    embeddings = clone()
                    index.embeddings = embeddings
                    index.local.embeddings = embeddings
                else:
                    index.embeddings = None
                    index.local.embeddings = None
                    index.local._vectors = None
            index.candidate_pool = 30
            index.rrf_constant = 60
            if (getattr(index, "embeddings", None) is not None
                    and settings.reranker_default_enabled and settings.reranker_ready):
                index.reranker = create_reranker(settings)
            default_mode = ("hybrid" if getattr(index, "embeddings", None) is not None
                            and getattr(index, "reranker", None) is not None else
                            "rrf" if getattr(index, "embeddings", None) is not None else "bm25")
            corpus_id = getattr(index, "collection_id", None)
            if not isinstance(corpus_id, str) or not corpus_id:
                raise ValueError("规范语料缺少稳定的corpus_id。")
            toolset = StandardTools(settings, index=index,
                                    default_mode=default_mode, top_k=8,
                                    before_request=lambda: _check_cancel(cancel_event),
                                    persist_batches=False, allow_index_build=False,
                                    default_hybrid_fallback_to_rrf=True,
                                    fixed_search_mode=default_mode)
            toolset.begin_run()
            toolset.catalog_index = bundle.catalog
            toolset.catalog_request_check = lambda: _check_cancel(cancel_event)
            observed_tools = _observed_tools(toolset, progress, cancel_event)
            schema = schema_for_model(settings.chat_model, STANDARD_ANSWER_SCHEMA)
            factory = model_factory or create_tool_chat_model
            model = _factory_call(factory, settings, schema)
            if model is None:
                raise ValueError("聊天模型未配置。")
            web_provider = create_web_search_provider(settings)
            observed_model = _ObservedModel(model, progress, cancel_event, samples)
            history = _history_for_agent(request.get("history"), corpus_id)
            page_service = getattr(bundle, "pages", None)
            if page_service is not None:
                page_service = _NavigationPageService(page_service)
            agent = StandardAgent(observed_model, observed_tools, page_service=page_service,
                                  web_provider=web_provider,
                                  input_budget_tokens=(settings.chat_context_budget_tokens - settings.chat_max_tokens))
            output = agent.run(question, corpus_id=corpus_id, history=history)
            agent_output = output
            _check_cancel(cancel_event)
            _notify(progress, cancel_event, "assembling", "正在整理规范证据和引用")
            result = _build_result(output, resources, settings, audit, request)
            if getattr(output, "termination_reason", None) in _TECHNICAL_FAILURE_REASONS:
                raise RuntimeError("规范 Agent 技术调用失败，未生成可用结论。")
            return result
        except BaseException as exc:
            cancelled = (cancel_event is not None and cancel_event.is_set())
            if cancelled or isinstance(exc, (CancelledError, _CancelledDuringTool)):
                payload = {"task_id": request.get("task_id"), "status": "cancelled",
                           "simulation": False, "model": settings.chat_model,
                           "token_usage": _usage_total(samples), "audit_log": audit.info}
                if agent_output is not None:
                    payload.update({"agent_result": agent_output,
                                    "evidence": deepcopy(getattr(agent_output, "evidence", [])),
                                    "traces": deepcopy(getattr(agent_output, "traces", []))})
                audit.set_result(payload)
                raise CancelledError("本轮规范问答已取消。") from None
            error = safe_error(exc if isinstance(exc, Exception) else Exception(type(exc).__name__),
                               component="standards")
            payload = {"task_id": request.get("task_id"), "status": "failed",
                       "simulation": False, "model": settings.chat_model, "error": error,
                       "token_usage": _usage_total(samples), "audit_log": audit.info}
            if agent_output is not None:
                payload.update({"agent_result": agent_output,
                                "evidence": deepcopy(getattr(agent_output, "evidence", [])),
                                "traces": deepcopy(getattr(agent_output, "traces", []))})
            audit.set_result(payload)
            raise RuntimeError(json.dumps(error, ensure_ascii=False)) from None


async def arun_standards(request, progress, cancel_event, *, resources, settings=None,
                         model_factory=None):
    """Run standards QA through native async model, index and provider APIs."""
    from ..providers import async_close_model
    from ..tool_chat import create_async_tool_chat_model

    if not isinstance(request, dict):
        raise ValueError("规范任务请求须为对象。")
    question = request.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入规范问题。")
    if resources is None or not callable(getattr(resources, "aprepare", None)):
        raise ValueError("规范资料资源未提供异步准备接口。")
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("本轮规范问答已取消。")
    try:
        settings = settings if settings is not None else load_settings()
    except Exception as exc:
        raise RuntimeError(json.dumps(safe_error(exc, component="standards"), ensure_ascii=False)) from None

    question = question.strip()
    samples = []
    model = None
    owned_models = []
    web_provider = None
    completed_result = None
    primary_failed = False
    primary_cancelled = False
    async with arecord_run("web_standard_agent", settings, question,
                           **tenant_audit_metadata(request)) as audit:
        agent_output = None
        try:
            await _aprogress(progress, cancel_event, "preparing", "正在准备规范资料和 Agent")
            bundle = await resources.aprepare()
            index = copy(bundle.index)
            index.local = copy(bundle.index.local)
            embeddings = getattr(index, "embeddings", None)
            clone = getattr(embeddings, "clone", None)
            if embeddings is not None:
                if callable(clone):
                    embeddings = clone()
                    index.embeddings = embeddings
                    index.local.embeddings = embeddings
                else:
                    index.embeddings = None
                    index.local.embeddings = None
                    index.local._vectors = None
            index.candidate_pool = 30
            index.rrf_constant = 60
            if (getattr(index, "embeddings", None) is not None
                    and settings.reranker_default_enabled and settings.reranker_ready):
                index.reranker = await run_local(create_reranker, settings)
            default_mode = ("hybrid" if getattr(index, "embeddings", None) is not None
                            and getattr(index, "reranker", None) is not None else
                            "rrf" if getattr(index, "embeddings", None) is not None else "bm25")
            corpus_id = getattr(index, "collection_id", None)
            if not isinstance(corpus_id, str) or not corpus_id:
                raise ValueError("规范语料缺少稳定的corpus_id。")
            toolset = StandardTools(settings, index=index,
                                    default_mode=default_mode, top_k=8,
                                    before_request=lambda: _check_cancel(cancel_event),
                                    persist_batches=False, allow_index_build=False,
                                    default_hybrid_fallback_to_rrf=True,
                                    fixed_search_mode=default_mode)
            toolset.begin_run()
            toolset.catalog_index = bundle.catalog
            toolset.catalog_request_check = lambda: _check_cancel(cancel_event)
            observed_tools = _async_observed_tools(toolset, progress, cancel_event)
            schema = await run_local(schema_for_model, settings.chat_model, STANDARD_ANSWER_SCHEMA)
            factory = model_factory or create_async_tool_chat_model
            def _create_model():
                created = _factory_call(factory, settings, schema)
                if created is not None:
                    owned_models.append(created)
                return created

            model = await run_local(_create_model)
            if model is None:
                raise ValueError("聊天模型未配置。")
            web_provider = create_web_search_provider(settings)
            observed_model = _ObservedModel(model, progress, cancel_event, samples)
            history = await run_local(_history_for_agent, request.get("history"), corpus_id)
            page_service = getattr(bundle, "pages", None)
            if page_service is not None:
                page_service = _NavigationPageService(page_service)
            def _create_agent():
                return StandardAgent(observed_model, observed_tools, page_service=page_service,
                    web_provider=(_AsyncObservedWebProvider(web_provider, progress, cancel_event)
                                  if web_provider is not None else None),
                    input_budget_tokens=(settings.chat_context_budget_tokens - settings.chat_max_tokens))

            agent = await run_local(_create_agent)
            output = await agent.arun(question, corpus_id=corpus_id, history=history)
            agent_output = output
            _check_cancel(cancel_event)
            await _aprogress(progress, cancel_event, "assembling", "正在整理规范证据和引用")
            result = await run_local(_build_result, output, resources, settings, audit, request)
            if getattr(output, "termination_reason", None) in _TECHNICAL_FAILURE_REASONS:
                raise RuntimeError("规范 Agent 技术调用失败，未生成可用结论。")
            completed_result = result
            return result
        except BaseException as exc:
            cancelled = ((cancel_event is not None and cancel_event.is_set())
                         or isinstance(exc, (CancelledError, _CancelledDuringTool, asyncio.CancelledError)))
            if cancelled:
                primary_cancelled = True
                payload = {"task_id": request.get("task_id"), "status": "cancelled",
                           "simulation": False, "model": settings.chat_model,
                           "token_usage": _usage_total(samples), "audit_log": audit.info}
                if agent_output is not None:
                    payload.update({"agent_result": agent_output,
                                    "evidence": deepcopy(getattr(agent_output, "evidence", [])),
                                    "traces": deepcopy(getattr(agent_output, "traces", []))})
                audit.set_result(payload)
                current_task = asyncio.current_task()
                if (isinstance(exc, asyncio.CancelledError) and current_task is not None
                        and current_task.cancelling()):
                    raise
                raise CancelledError("本轮规范问答已取消。") from None
            primary_failed = True
            error = safe_error(exc if isinstance(exc, Exception) else Exception(type(exc).__name__),
                               component="standards")
            payload = {"task_id": request.get("task_id"), "status": "failed",
                       "simulation": False, "model": settings.chat_model, "error": error,
                       "token_usage": _usage_total(samples), "audit_log": audit.info}
            if agent_output is not None:
                payload.update({"agent_result": agent_output,
                                "evidence": deepcopy(getattr(agent_output, "evidence", [])),
                                "traces": deepcopy(getattr(agent_output, "traces", []))})
            audit.set_result(payload)
            raise RuntimeError(json.dumps(error, ensure_ascii=False)) from None
        finally:
            close_failures = []
            seen = set()
            for resource in (*owned_models, web_provider):
                if resource is None or id(resource) in seen:
                    continue
                seen.add(id(resource))
                try:
                    await async_close_model(resource)
                except BaseException as exc:
                    close_failures.append(type(exc).__name__)
            if close_failures:
                mark_cleanup = getattr(cancel_event, "mark_resource_cleanup_failed", None)
                if callable(mark_cleanup):
                    mark_cleanup()
                await arecord_event("web_standards_resource_close_failed", exception_types=close_failures)
                if completed_result is not None:
                    cleanup_error = TaskResourceCleanupError("standards",
                                                            exception_types=close_failures)
                    error = cleanup_error.as_error()
                    audit.set_result({"task_id": request.get("task_id"), "status": "failed",
                        "simulation": False, "model": settings.chat_model, "error": error,
                        "token_usage": _usage_total(samples), "audit_log": audit.info})
                    raise cleanup_error
                if primary_failed and not primary_cancelled:
                    cleanup_error = TaskResourceCleanupError("standards",
                                                            exception_types=close_failures)
                    previous = audit.result if isinstance(audit.result, dict) else {}
                    error = cleanup_error.as_error()
                    if previous.get("error") is not None:
                        error["operation_error"] = previous["error"]
                    audit.set_result({**previous, "status": "failed", "error": error})
                    raise cleanup_error
