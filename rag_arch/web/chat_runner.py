"""Web chat worker; optional attachments come only from the workspace store."""
from concurrent.futures import CancelledError
from copy import copy, deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
from urllib.parse import urlencode

from ..config import load_settings
from ..generation_experiments import safe_error
from ..graph_agent import _usage, _usage_total
from ..models import Answer, Chunk, ConversationTurn, SearchHit
from ..providers import (create_chat_model, create_embeddings, create_reranker,
                         create_web_search_provider)
from ..qa import answer_question
from ..runtime_audit import record_event, record_run


def _cancelled(cancel_event):
    if cancel_event.is_set():
        raise CancelledError("本轮聊天已取消。")


class _CancellableEmbeddings:
    """Check this run's cancellation between short outgoing embedding batches."""
    def __init__(self, provider, cancel_event, batch_size):
        self._provider, self._cancel_event = provider, cancel_event
        limits = [batch_size] + [value for name in ("batch_size", "chunk_size")
                                if type(value := getattr(provider, name, None)) is int and value > 0]
        self.batch_size = min(limits)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            _cancelled(self._cancel_event)
            vectors.extend(self._provider.embed_documents(texts[start:start + self.batch_size]))
            _cancelled(self._cancel_event)
        return vectors

    def embed_query(self, text):
        _cancelled(self._cancel_event)
        vector = self._provider.embed_query(text)
        _cancelled(self._cancel_event)
        return vector


def _saved_hits(rows):
    """Restore only serialized source records, including their collection ids."""
    hits = []
    for row in rows or []:
        try:
            if isinstance(row, dict) and isinstance(row.get("chunk"), dict):
                hits.append(SearchHit(Chunk(**deepcopy(row["chunk"])), row["score"]))
        except (KeyError, TypeError):
            # Older/incomplete history cannot establish a file collection.
            continue
    return hits


def _history_turns(rows):
    """TaskService supplies completed turns; never reuse its infrastructure-demo replies."""
    turns = []
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            continue
        result = row.get("result")
        if (not isinstance(result, dict) or result.get("simulation") is not False
                or not isinstance(result.get("text"), str)
                or row.get("status", "completed") != "completed"
                or row.get("assistant_id", "chat") != "chat"):
            continue
        answer = Answer(text=result["text"], hits=_saved_hits(result.get("hits")),
                        warnings=deepcopy(result.get("warnings") or []), grounded=bool(result.get("grounded", False)),
                        citation_check=deepcopy(result.get("citation_check")),
                        page_sources=deepcopy(result.get("page_sources") or []),
                        web_sources=deepcopy(result.get("web_sources") or []),
                        native_tool_observations=deepcopy(result.get("native_tool_observations") or []),
                        native_tool_usage=deepcopy(result.get("native_tool_usage") or {}))
        turns.append(ConversationTurn(row["question"], answer))
    return turns


def _file_fields(prepared, answer, file_mode):
    """Public file references keep asset identity, never resolve a filename/path."""
    files = []
    for item in prepared.assets:
        public = {key: item[key] for key in ("asset_id", "filename", "size_bytes")}
        public["page_url"] = (
            "/viewer/pages?" + urlencode({"asset": public["asset_id"], "workspace_id": prepared.workspace_id})
            if public["filename"].lower().endswith((".pdf", ".docx")) else None)
        files.append(public)
    by_asset = {row["asset_id"]: row for row in files}
    file_warnings = []
    for document in prepared.collection.documents:
        file_warnings.extend(f"{document.filename}（文字提取）：{warning}" for warning in document.warnings)
        asset = prepared.document_assets.get(document.doc_id)
        if asset and asset["asset_id"] in by_asset:
            by_asset[asset["asset_id"]].update(warnings=list(document.warnings),
                unread_pages=list(document.unread_pages), page_count=document.page_count)
    if file_mode == "pages_only":
        file_warnings.append("附件没有可检索文字；可按需读取原页。本轮未执行OCR，页图读取不代表全文文字已完整提取。")
    citations = []
    for number in (answer.citation_check or {}).get("cited_numbers", []):
        if type(number) is not int or not 1 <= number <= len(answer.hits):
            continue
        chunk = answer.hits[number - 1].chunk
        asset = prepared.document_assets.get(chunk.doc_id)
        if not asset or asset["asset_id"] not in by_asset:
            continue
        file = by_asset[asset["asset_id"]]
        url = file["page_url"]
        # PDF text units have exact physical pages. DOCX paragraphs/tables do
        # not establish rendered pagination, so retain their original locator.
        page = chunk.locator.get("page") if chunk.source_kind == "pdf" else None
        if url and type(page) is int and page > 0:
            url += "&" + urlencode({"page": page})
        citations.append({"evidence_id": f"S{number}", "kind": "file_chunk",
            "asset_id": file["asset_id"], "filename": chunk.filename,
            "text": chunk.text, "locator": deepcopy(chunk.locator), "page_url": url})
    page_sources = deepcopy(answer.page_sources)
    contexts = getattr(prepared.pages, "source_context", {})
    for source in page_sources:
        asset_id = contexts.get(source.get("source_ref"), {}).get("asset_id")
        file = by_asset.get(asset_id)
        if file:
            source["asset_id"] = asset_id
            source["page_url"] = file["page_url"]
            if source["page_url"] and type(source.get("page")) is int and source["page"] > 0:
                source["page_url"] += "&" + urlencode({"page": source["page"]})
    return {"files": files, "file_mode": file_mode,
            "uses_embeddings": bool(prepared.collection.uses_embeddings and prepared.collection.chunks),
            "citations": citations, "page_sources": page_sources,
            "grounded": answer.grounded, "citation_check": deepcopy(answer.citation_check),
            "hits": [asdict(hit) for hit in answer.hits],
            "warnings": list(dict.fromkeys(file_warnings + list(answer.warnings)))}


class _ObservedModel:
    """Keep provider behavior/tools; observe only calls made by the existing QA flow."""

    def __init__(self, model, progress, cancel_event, samples):
        self._model = model
        self._progress = progress
        self._cancel_event = cancel_event
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _wrap(self, model):
        return _ObservedModel(model, self._progress, self._cancel_event, self._samples)

    def bind_tools(self, *args, **kwargs):
        return self._wrap(self._model.bind_tools(*args, **kwargs))

    def plain(self):
        plain = getattr(self._model, "plain", None)
        return self._wrap(plain() if callable(plain) else self._model)

    def invoke(self, *args, **kwargs):
        _cancelled(self._cancel_event)
        self._progress("model", f"正在调用聊天模型（第{len(self._samples) + 1}次）")
        _cancelled(self._cancel_event)
        position = len(self._samples)
        self._samples.append(None)
        response = self._model.invoke(*args, **kwargs)
        # Record returned usage even if cancellation arrived during this request.
        usage = _usage(SimpleNamespace(
            usage_metadata=getattr(response, "usage_metadata", None),
            response_metadata=getattr(response, "response_metadata", None) or {}))
        self._samples[position] = usage
        record_event("web_chat_model_returned", call_number=position + 1, token_usage=usage)
        _cancelled(self._cancel_event)
        return response


def run_chat(request, progress, cancel_event, *, settings=None, model_factory=None, documents=None):
    """Run the configured chat model in this worker's independent audit context.

    Cancellation is cooperative: an in-flight provider call returns before its
    answer is discarded. TaskService turns CancelledError into a cancelled task.
    """
    _cancelled(cancel_event)
    question = request.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入聊天问题。")
    question = question.strip()
    try:
        settings = settings if settings is not None else load_settings()
    except Exception as exc:
        _cancelled(cancel_event)
        raise RuntimeError(json.dumps(safe_error(exc, component="chat"), ensure_ascii=False)) from None
    samples = []
    with record_run("web_chat", settings, question, task_id=request.get("task_id")) as audit:
        try:
            _cancelled(cancel_event)
            progress("preparing", "正在准备聊天上下文")
            _cancelled(cancel_event)
            prepared, file_mode = None, None
            qa_options = {"index": None, "page_service": None,
                "history": _history_turns(request.get("history")),
                "context_budget_tokens": settings.chat_context_budget_tokens,
                "output_budget_tokens": settings.chat_max_tokens}
            attachment_ids = request.get("attachment_ids") or []
            if attachment_ids:
                if documents is None:
                    raise ValueError("上传文件服务未提供，不能读取这些附件。")
                legacy_use_embeddings = request.get("use_embeddings", False)
                if type(legacy_use_embeddings) is not bool:
                    raise ValueError("向量检索选项须为布尔值。")
                progress("files", "正在读取附件全文并检查输入规模")
                _cancelled(cancel_event)

                embedding_proxy = None

                def _make_embeddings():
                    nonlocal embedding_proxy
                    _cancelled(cancel_event)
                    provider = create_embeddings(settings)
                    _cancelled(cancel_event)
                    if provider is not None:
                        embedding_proxy = _CancellableEmbeddings(
                            provider, cancel_event, settings.embedding_batch_size)
                    return embedding_proxy

                prepared = documents.prepare_chat(
                    attachment_ids, embeddings_factory=_make_embeddings)
                _cancelled(cancel_event)
                collection, pages = prepared.collection, prepared.pages
                usable = bool(collection.chunks)
                if not usable and (pages is None or not pages.sources()):
                    raise ValueError("附件没有可用文字或可按页读取的原件，暂不能问答。")
                file_mode = collection.mode if usable else "pages_only"

                run_index = None
                rerank_warnings = []
                if usable and collection.uses_embeddings and collection.index is not None:
                    # Each request gets its own lightweight index view so its cancellable
                    # provider and reranker callback cannot race with another chat.
                    if embedding_proxy is None:
                        provider = create_embeddings(settings)
                        _cancelled(cancel_event)
                        if provider is None:
                            raise ValueError("附件提取正文超出全文直读门槛，向量服务不可用；未改用关键词检索。")
                        embedding_proxy = _CancellableEmbeddings(
                            provider, cancel_event, settings.embedding_batch_size)
                    run_index = copy(collection.index)
                    run_index.embeddings = embedding_proxy
                    run_index.candidate_pool = 30
                    run_index.rrf_constant = 60

                    def _record_rerank_fallback(failure):
                        status = failure.get("status_code")
                        code = failure.get("code") or "RemoteError"
                        record_event("web_chat_file_reranker_fallback",
                            fallback_reason="reranker_service_error",
                            fallback_to="rrf_top_k", candidate_pool=30,
                            rrf_constant=60, top_k=8, error_code=code,
                            http_status=status)
                        status_text = f"HTTP {status}，" if status is not None else ""
                        rerank_warnings.append(
                            f"附件重排服务异常（{status_text}code={code}），本轮按 RRF 30 候选中的前 8 条回答。")

                    run_index.rerank_error_handler = _record_rerank_fallback
                    if (settings.reranker_provider == "bailian"
                            and settings.reranker_model == "qwen3-rerank"
                            and settings.reranker_ready):
                        run_index.reranker = create_reranker(settings)
                    else:
                        run_index.reranker = None
                    record_event("web_chat_file_retrieval_strategy",
                        mode="bm25_dense_rrf", embedding_model=settings.embedding_model,
                        candidate_pool=30, rrf_constant=60, top_k=8,
                        reranker=("qwen3-rerank" if run_index.reranker is not None else None),
                        legacy_use_embeddings_ignored=legacy_use_embeddings)
                    if run_index.reranker is None:
                        reason = ("reranker_not_configured" if not settings.reranker_ready
                                  else "reranker_configuration_not_qwen3")
                        record_event("web_chat_file_reranker_fallback",
                            fallback_reason=reason, fallback_to="rrf_top_k",
                            candidate_pool=30, rrf_constant=60, top_k=8)
                        rerank_warnings.append(
                            "附件重排服务未按百炼 qwen3-rerank 配置，本轮按 RRF 30 候选中的前 8 条回答。")
                elif usable:
                    record_event("web_chat_file_retrieval_strategy",
                        mode="direct" if collection.mode == "direct" else "bm25",
                        text_bytes=collection.text_bytes,
                        text_tokens=collection.text_tokens,
                        legacy_use_embeddings_ignored=legacy_use_embeddings)

                qa_options.update(index=run_index if usable else None,
                    direct_chunks=collection.chunks if usable and collection.mode == "direct" else None,
                    page_service=pages, web_provider=create_web_search_provider(settings), top_k=8)
                record_event("web_chat_files_prepared", attachment_ids=list(attachment_ids),
                    file_mode=file_mode, text_bytes=collection.text_bytes,
                    text_tokens=collection.text_tokens,
                    uses_embeddings=bool(collection.uses_embeddings and usable),
                    legacy_use_embeddings_ignored=legacy_use_embeddings)
            model = (model_factory or create_chat_model)(settings)
            if model is None:
                raise ValueError("聊天模型未配置。")
            observed = _ObservedModel(model, progress, cancel_event, samples)
            answer = answer_question(question, chat_model=observed, **qa_options)
            _cancelled(cancel_event)
            if prepared is not None and rerank_warnings:
                answer.warnings.extend(row for row in dict.fromkeys(rerank_warnings)
                                       if row not in answer.warnings)
            result = {
                "text": answer.text, "simulation": False, "model": settings.chat_model,
                "web_sources": deepcopy(answer.web_sources),
                "warnings": list(answer.warnings),
                "native_tool_usage": deepcopy(answer.native_tool_usage),
                "audit_log": audit.info, "token_usage": _usage_total(samples),
                "model_calls": len(samples),
                "tool_calls": answer.tool_calls,
            }
            if prepared is not None:
                result.update(_file_fields(prepared, answer, file_mode))
            # The audit keeps the full Answer, including native tool observations.
            audit.set_result({**result, "task_id": request.get("task_id"), "answer": answer})
        except Exception as exc:
            if cancel_event.is_set() or isinstance(exc, CancelledError):
                audit.set_result({"status": "cancelled", "simulation": False,
                    "task_id": request.get("task_id"), "model": settings.chat_model,
                    "token_usage": _usage_total(samples), "audit_log": audit.info})
                raise CancelledError("本轮聊天已取消。") from None
            error = safe_error(exc, component="chat")
            audit.set_result({"status": "failed", "simulation": False,
                "task_id": request.get("task_id"), "model": settings.chat_model,
                "error": error, "token_usage": _usage_total(samples), "audit_log": audit.info})
            # Never pass provider str(exc), response bodies or credential-bearing URLs outward.
            raise RuntimeError(json.dumps(error, ensure_ascii=False)) from None
    return result
