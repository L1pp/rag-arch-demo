"""Async Web runner tests use only async fakes; no model or provider HTTP calls."""
from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
import numpy as np

from rag_arch.async_support import run_local
from rag_arch.async_support import TaskResourceCleanupError
from rag_arch.config import Settings
from rag_arch.models import Chunk
from rag_arch.retrieval import LocalIndex
from rag_arch.reranking import RerankError
from rag_arch.runtime_audit import _current
from rag_arch.web.chat_runner import arun_chat
from rag_arch.web.plan_runner import arun_plan
from rag_arch.web.standards_runner import arun_standards


def fake_settings():
    return Settings(chat_model="synthetic-async-model",
                    chat_base_url="https://example.invalid/v1",
                    chat_api_key="synthetic-private-key",
                    chat_max_tokens=256, chat_context_budget_tokens=4096,
                    plan_max_tokens=256, plan_timeout_seconds=30,
                    plan_context_budget_tokens=100000,
                    plan_context_summary_ratio=0)


class FakeAsyncClient:
    def __init__(self, *, fail=False):
        self.closed = False
        self.fail = fail
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1
        self.closed = True
        if self.fail:
            raise RuntimeError("synthetic close failure")


class AsyncOnlyModel:
    """A model with no sync invoke method, so a blocking path fails loudly."""

    native_tool_names = ()
    max_tokens = 256

    def __init__(self, text="合成异步回答", *, on_call=None, started=None, release=None):
        self.text = text
        self.on_call = on_call
        self.started = started
        self.release = release
        self.calls = []
        self.client = FakeAsyncClient()
        self.root_async_client = self.client
        self._owns_root_async_client = True

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.on_call is not None:
            self.on_call()
        return AIMessage(content=self.text,
                         usage_metadata={"input_tokens": 7, "output_tokens": 3,
                                         "total_tokens": 10})

    def bind_tools(self, *args, **kwargs):
        return self

    def plain(self):
        return self


class FakeEmbeddingProvider:
    def __init__(self):
        self.client = FakeAsyncClient()
        self.async_client = SimpleNamespace(_client=self.client)
        self.document_batches = []
        self.query_calls = []
        self.batch_size = 8

    async def aembed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 0.0]


class FakeAsyncIndex:
    def __init__(self):
        self.embeddings = None
        self.local = SimpleNamespace(embeddings=None, _vectors={})
        self.reranker = None
        self.candidate_pool = None
        self.rrf_constant = None
        self.search_calls = []

    async def asearch(self, query, top_k=8):
        self.search_calls.append(query)
        if self.embeddings is not None:
            await self.embeddings.aembed_query(query)
        return []


def _answer(text="合成回答"):
    return SimpleNamespace(text=text, hits=[], warnings=[], citation_check={},
                           page_sources=[], web_sources=[], native_tool_observations=[],
                           native_tool_usage={}, grounded=False, tool_calls=0)


def prepared_chat(index=None):
    index = index or FakeAsyncIndex()
    collection = SimpleNamespace(chunks=[object()], uses_embeddings=True, index=index,
                                 mode="hybrid", documents=[], text_bytes=5, text_tokens=2)
    return SimpleNamespace(collection=collection, pages=None, assets=[],
                           document_assets={}, workspace_id="ws-chat-test")


class FakeDocumentStore:
    """Minimal async DocumentStore shape, optionally held behind a shared builder gate."""

    def __init__(self, *, prepared=None, entered=None, release=None,
                 call_embedding_factory=True):
        self.prepared = prepared or prepared_chat()
        self.entered = entered
        self.release = release
        self.call_embedding_factory = call_embedding_factory
        self.calls = []
        self.embedding_wrapper = None

    async def aprepare_chat(self, attachment_ids, *, embeddings=None, embeddings_factory=None):
        self.calls.append(list(attachment_ids))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        provider = embeddings
        if provider is None and self.call_embedding_factory and embeddings_factory is not None:
            provider = await run_local(embeddings_factory)
        self.embedding_wrapper = provider
        if provider is not None:
            await provider.aembed_documents(["合成附件正文"])
        return self.prepared


class FakeWebProvider:
    available = True

    def __init__(self):
        self.async_client = FakeAsyncClient()
        self._owns_async_client = True
        self.calls = []

    async def asearch(self, query):
        self.calls.append(query)
        return {"sources": [{"url": "https://example.invalid/source", "title": "合成来源"}]}

    def as_langchain_tool(self):
        async def asearch(query: str):
            return await self.asearch(query)

        def sync_search(query: str):
            raise AssertionError("async runner called sync web search")

        return StructuredTool.from_function(func=sync_search, coroutine=asearch,
                                            name="web_search", description="synthetic web search")


def local_tool(name="local_lookup"):
    calls = []

    def invoke(query: str) -> dict:
        calls.append((threading.get_ident(), query))
        return {"query": query, "local": True}

    return StructuredTool.from_function(invoke, name=name, description="synthetic local tool"), calls


class AsyncRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.audit = TemporaryDirectory(prefix="rag-web-async-runner-")
        self.addCleanup(self.audit.cleanup)
        self.audit_patch = patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(self.audit.name))
        self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.settings = fake_settings()
        self.loop_thread = threading.get_ident()
        self.progress_events = []
        self.progress_threads = []

    def progress(self, stage, detail=""):
        self.progress_events.append((stage, detail))
        self.progress_threads.append(threading.get_ident())

    def audit_result_rows(self):
        return [json.loads(path.read_text(encoding="utf-8"))
                for path in Path(self.audit.name).glob("users/*/*/*/result.json")]

    def audit_event_rows(self):
        events = []
        for path in Path(self.audit.name).glob("users/*/*/*/events.jsonl"):
            events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                          if line.strip())
        return events

    async def test_chat_uses_async_only_models_and_keeps_parallel_audits_tenant_scoped(self):
        models = {101: AsyncOnlyModel("用户101回答"), 202: AsyncOnlyModel("用户202回答")}
        model_queue = iter(models.values())

        def create_async_chat_model(_settings):
            return next(model_queue)

        async def fake_qa(question, *, chat_model, **kwargs):
            response = await chat_model.ainvoke([question])
            return _answer(response.content)

        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             patch("rag_arch.providers.create_async_chat_model", new=create_async_chat_model), \
             patch("rag_arch.web.chat_runner.create_chat_model",
                   side_effect=AssertionError("async runner called sync model factory")):
            results = await asyncio.gather(*[
                arun_chat({"task_id": f"chat-{uid}", "assistant_id": "chat",
                           "prompt": "异步问题", "user_id": uid,
                           "workspace_id": f"workspace-{uid}"}, self.progress, asyncio.Event(),
                          settings=self.settings)
                for uid in models
            ])

        self.assertEqual([result["text"] for result in results], ["用户101回答", "用户202回答"])
        self.assertTrue(all(result["token_usage"]["total_tokens"] == 10 for result in results))
        self.assertTrue(all(result["model_calls"] == 1 for result in results))
        self.assertTrue(all(model.client.closed for model in models.values()))
        self.assertIsNone(_current.get())
        rows = self.audit_result_rows()
        self.assertEqual({row["audit_log"]["user_id"] for row in rows}, {101, 202})
        self.assertEqual({row["audit_log"]["workspace_id"] for row in rows},
                         {"workspace-101", "workspace-202"})
        self.assertEqual({result["audit_log"]["user_id"] for result in results}, {101, 202})
        self.assertTrue(all(thread != self.loop_thread for thread in self.progress_threads))

    async def test_chat_embedding_prepare_retrieval_and_cancel_cleanup_are_async(self):
        cancel = asyncio.Event()
        embedding = FakeEmbeddingProvider()
        document_store = FakeDocumentStore(prepared=prepared_chat())
        model = AsyncOnlyModel(on_call=cancel.set)

        async def fake_qa(question, *, chat_model, index, **kwargs):
            await index.asearch(question)
            await chat_model.ainvoke([question])
            return _answer("不会交付")

        request = {"task_id": "chat-cancel", "assistant_id": "chat", "prompt": "附件问题",
                   "attachment_ids": ["asset-1"], "user_id": 303,
                   "workspace_id": "workspace-303"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             patch("rag_arch.web.chat_runner.create_embeddings", return_value=embedding), \
             self.assertRaises(CancelledError):
            await arun_chat(request, self.progress, cancel, settings=self.settings,
                            model_factory=lambda _settings: model, documents=document_store)

        self.assertEqual(embedding.document_batches, [["合成附件正文"]])
        self.assertEqual(embedding.query_calls, ["附件问题"])
        self.assertTrue(embedding.client.closed)
        self.assertTrue(model.client.closed)
        row, = self.audit_result_rows()
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["audit_log"]["user_id"], 303)

    async def test_chat_close_failure_fails_task_but_still_closes_other_owned_clients(self):
        embedding = FakeEmbeddingProvider()
        documents = FakeDocumentStore(prepared=prepared_chat())
        model = AsyncOnlyModel("合成成功回答")
        model.client.fail = True

        async def fake_qa(question, *, chat_model, index, **kwargs):
            await index.asearch(question)
            response = await chat_model.ainvoke([question])
            return _answer(response.content)

        request = {"task_id": "chat-close-failure", "assistant_id": "chat", "prompt": "附件问题",
                   "attachment_ids": ["asset-1"], "user_id": 304,
                   "workspace_id": "workspace-304"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             patch("rag_arch.web.chat_runner.create_embeddings", return_value=embedding), \
             self.assertRaises(TaskResourceCleanupError):
            await arun_chat(request, self.progress, asyncio.Event(), settings=self.settings,
                            model_factory=lambda _settings: model, documents=documents)

        self.assertTrue(model.client.closed)
        self.assertTrue(embedding.client.closed, "later cleanup must still run after model close fails")
        row, = self.audit_result_rows()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"]["code"], "resource_cleanup_failed")

    async def test_chat_primary_error_with_close_failure_preserves_error_and_retires_worker(self):
        def fail_model_call():
            raise RuntimeError("synthetic primary failure")

        model = AsyncOnlyModel(on_call=fail_model_call)
        model.client.fail = True

        async def fake_qa(question, *, chat_model, **kwargs):
            return await chat_model.ainvoke([question])

        request = {"task_id": "chat-primary-close-failure", "assistant_id": "chat",
                   "prompt": "异步失败", "user_id": 306,
                   "workspace_id": "workspace-306"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             self.assertRaises(TaskResourceCleanupError):
            await arun_chat(request, self.progress, asyncio.Event(), settings=self.settings,
                            model_factory=lambda _settings: model)

        self.assertTrue(model.client.closed)
        row, = self.audit_result_rows()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"]["code"], "resource_cleanup_failed")
        self.assertIsNotNone(row["error"]["operation_error"])

    async def test_chat_task_cancellation_writes_cancelled_result_and_closes_model(self):
        started, release = asyncio.Event(), asyncio.Event()
        model = AsyncOnlyModel("不会返回", started=started, release=release)

        async def fake_qa(question, *, chat_model, **kwargs):
            response = await chat_model.ainvoke([question])
            return _answer(response.content)

        request = {"task_id": "chat-direct-cancel", "assistant_id": "chat", "prompt": "等待中的请求",
                   "user_id": 305, "workspace_id": "workspace-305"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa):
            task = asyncio.create_task(arun_chat(request, self.progress, asyncio.Event(),
                settings=self.settings, model_factory=lambda _settings: model))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

        self.assertTrue(model.client.closed)
        row, = self.audit_result_rows()
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["task_id"], "chat-direct-cancel")

    async def test_chat_waits_for_slow_attachment_prepare_even_after_cancel(self):
        entered, release = asyncio.Event(), asyncio.Event()
        documents = FakeDocumentStore(entered=entered, release=release)
        cancel = asyncio.Event()
        model = AsyncOnlyModel()
        request = {"task_id": "chat-slow-prepare", "assistant_id": "chat", "prompt": "附件问题",
                   "attachment_ids": ["asset-1"], "user_id": 404,
                   "workspace_id": "workspace-404"}

        task = asyncio.create_task(arun_chat(request, self.progress, cancel,
                                             settings=self.settings,
                                             model_factory=lambda _settings: model,
                                             documents=documents))
        await asyncio.wait_for(entered.wait(), timeout=2)
        cancel.set()
        await asyncio.sleep(0)
        self.assertFalse(task.done(), "runner finished while the shared attachment builder was still running")
        release.set()
        with self.assertRaises(CancelledError):
            await asyncio.wait_for(task, timeout=2)
        self.assertEqual(model.calls, [])
        self.assertEqual(documents.calls, [["asset-1"]])

    async def test_chat_drains_cancelled_embedding_factory_and_closes_new_client_once(self):
        started, release = threading.Event(), threading.Event()
        cancel = threading.Event()
        embedding = FakeEmbeddingProvider()
        documents = FakeDocumentStore(prepared=prepared_chat(), call_embedding_factory=False)
        model = AsyncOnlyModel()

        def slow_embedding_factory(_settings):
            started.set()
            release.wait(timeout=5)
            return embedding

        async def fake_qa(question, *, chat_model, **kwargs):
            response = await chat_model.ainvoke([question])
            return _answer(response.content)

        request = {"task_id": "chat-cancel-embedding", "assistant_id": "chat",
                   "prompt": "附件向量问题", "attachment_ids": ["asset-1"],
                   "user_id": 406, "workspace_id": "workspace-406"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             patch("rag_arch.web.chat_runner.create_embeddings", side_effect=slow_embedding_factory):
            task = asyncio.create_task(arun_chat(request, self.progress, cancel,
                settings=self.settings, model_factory=lambda _settings: model,
                documents=documents))
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(started.is_set(), "embedding factory did not start")
            cancel.set()
            await asyncio.sleep(0.03)
            self.assertFalse(task.done(), "runner released its slot while embedding client construction was active")
            release.set()
            with self.assertRaises(CancelledError):
                await asyncio.wait_for(task, timeout=2)

        self.assertEqual(embedding.client.close_calls, 1)
        self.assertEqual(model.calls, [])

    async def test_chat_real_local_index_async_rerank_error_keeps_rrf_fallback(self):
        model = AsyncOnlyModel("RRF降级回答")
        embedding = FakeEmbeddingProvider()
        chunk = Chunk(doc_id="doc-1", filename="long.txt", chunk_id="c1",
                      text="建筑材料应符合要求", locator={"unit": 1},
                      source_kind="txt", collection_id="collection-1")
        index = LocalIndex([chunk])
        index._vectors = np.asarray([[1.0, 0.0]], dtype=np.float64)
        documents = FakeDocumentStore(
            prepared=prepared_chat(index), call_embedding_factory=False)
        from dataclasses import replace
        settings = replace(self.settings, reranker_provider="bailian",
                           reranker_model="qwen3-rerank",
                           reranker_base_url="https://rerank.example.invalid",
                           reranker_api_key="synthetic-reranker-key",
                           reranker_default_enabled=True)

        class FailingReranker:
            async def arerank(self, query, documents, top_n=None):
                raise RerankError("TransportError", 503)

        async def fake_qa(question, *, chat_model, index, **kwargs):
            hits = await index.asearch(question, top_k=8)
            response = await chat_model.ainvoke([question])
            answer = _answer(response.content)
            answer.hits = hits
            return answer

        request = {"task_id": "chat-rerank-fallback", "assistant_id": "chat",
                   "prompt": "材料要求", "attachment_ids": ["asset-1"],
                   "user_id": 407, "workspace_id": "workspace-407"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa), \
             patch("rag_arch.web.chat_runner.create_embeddings", return_value=embedding), \
             patch("rag_arch.web.chat_runner.create_reranker", return_value=FailingReranker()):
            result = await arun_chat(request, self.progress, threading.Event(),
                                     settings=settings,
                                     model_factory=lambda _settings: model,
                                     documents=documents)

        self.assertEqual(result["text"], "RRF降级回答")
        self.assertTrue(any("重排服务异常" in warning for warning in result["warnings"]))
        row, = self.audit_result_rows()
        self.assertEqual(row["task_id"], "chat-rerank-fallback")
        self.assertTrue(any(event.get("kind") == "web_chat_file_reranker_fallback"
                            for event in self.audit_event_rows()))

    async def test_chat_drains_cancelled_model_constructor_and_closes_created_client(self):
        started, release = threading.Event(), threading.Event()
        model = AsyncOnlyModel("constructor result")

        def slow_factory(_settings):
            started.set()
            release.wait(timeout=5)
            return model

        async def fake_qa(question, *, chat_model, **kwargs):
            response = await chat_model.ainvoke([question])
            return _answer(response.content)

        request = {"task_id": "chat-cancel-constructor", "assistant_id": "chat",
                   "prompt": "取消构造", "user_id": 405,
                   "workspace_id": "workspace-405"}
        with patch("rag_arch.qa.aanswer_question", side_effect=fake_qa):
            task = asyncio.create_task(arun_chat(request, self.progress, asyncio.Event(),
                settings=self.settings, model_factory=slow_factory))
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(started.is_set(), "model constructor did not start")
            task.cancel()
            await asyncio.sleep(0.03)
            self.assertFalse(task.done(), "cancellation detached the still-running local constructor")
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

        self.assertTrue(model.client.closed, "owned model returned during cancellation must be closed")

    async def test_standards_runner_uses_async_agent_model_and_network_tools(self):
        model = AsyncOnlyModel("规范答复")
        web = FakeWebProvider()
        web_client = web.async_client
        local, local_calls = local_tool("lookup_local")
        bundle = SimpleNamespace(index=SimpleNamespace(local=SimpleNamespace(), embeddings=None,
                                                        collection_id="corpus-async"),
                                 catalog=object(), pages=None)

        class Resources:
            async def aprepare(self):
                return bundle

            def get(self):
                raise AssertionError("async runner used synchronous resources.get")

            def citation(self, row):
                return row

            def catalog_entry(self, row):
                return row

        class Tools:
            def __init__(self, settings, **kwargs):
                pass

            def begin_run(self):
                pass

            def as_langchain_tools(self):
                return [local]

        class Agent:
            def __init__(self, observed_model, tools, *, web_provider=None, **kwargs):
                self.model, self.tools = observed_model, tools
                self.web_provider = web_provider

            async def arun(self, question, *, corpus_id, history=None):
                response = await self.model.ainvoke([question])
                wrapped = self.tools.as_langchain_tools()[0]
                await wrapped.ainvoke({"query": "本地查找"})
                web_tool = self.web_provider.as_langchain_tool()
                await web_tool.ainvoke({"query": "外部搜索"})
                return SimpleNamespace(text=response.content, evidence=[], citations=[],
                    catalog_matches=[], warnings=[], native_tool_usage={},
                    token_usage={"total_tokens": 10}, status="answered",
                    answer_scope="evidence", tool_calls=2, model_calls=1,
                    corpus_id=corpus_id, grounded=True, traces=[])

        factory_kwargs = []

        def create_async_tool_chat_model(_settings, **kwargs):
            factory_kwargs.append(kwargs)
            return model

        request = {"task_id": "standards-async", "assistant_id": "standards",
                   "prompt": "规范问题", "user_id": 505, "workspace_id": "workspace-505"}
        with patch("rag_arch.web.standards_runner.StandardTools", Tools), \
             patch("rag_arch.web.standards_runner.StandardAgent", Agent), \
             patch("rag_arch.web.standards_runner.create_web_search_provider", return_value=web), \
             patch("rag_arch.tool_chat.create_async_tool_chat_model", new=create_async_tool_chat_model), \
             patch("rag_arch.web.standards_runner.create_tool_chat_model",
                   side_effect=AssertionError("async standards runner called sync model factory")):
            result = await arun_standards(request, self.progress, asyncio.Event(),
                                          resources=Resources(), settings=self.settings)

        self.assertEqual(result["text"], "规范答复")
        self.assertEqual(result["token_usage"]["total_tokens"], 10)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(factory_kwargs[0]["json_output"],
                         factory_kwargs[0]["response_schema"] is None)
        self.assertEqual(web.calls, ["外部搜索"])
        self.assertTrue(web_client.closed)
        self.assertNotEqual(local_calls[0][0], self.loop_thread)
        row, = self.audit_result_rows()
        self.assertEqual((row["result"]["audit_log"]["user_id"],
                          row["result"]["audit_log"]["workspace_id"]),
                         (505, "workspace-505"))

    async def test_plan_runner_uses_async_agent_model_tools_and_threadsafe_progress(self):
        model = AsyncOnlyModel("方案追问答复")
        web = FakeWebProvider()
        web_client = web.async_client
        local, local_calls = local_tool("plan_local_lookup")
        class PlanDocument:
            document_id = "PLAN-ASYNC"
            source_revision = "rev-1"
            filename = "plan.txt"
            units = [{"text": "synthetic"}]

            def read(self, start=1, offset=0):
                return {"text": "synthetic plan body", "start": start, "offset": offset}

            def search(self, query):
                return {"query": query, "matches": []}

            def navigation(self, start=1):
                return {"start": start, "units": self.units}

            def metadata(self):
                return {"document_id": self.document_id, "filename": self.filename}

        document = PlanDocument()
        plan_index = FakeAsyncIndex()
        plan_index.corpus = object()
        resource_bundle = SimpleNamespace(catalog=object())

        class Resources:
            async def aprepare(self):
                return resource_bundle

            async def aplan_index(self):
                return plan_index

            def get(self):
                raise AssertionError("async runner used synchronous resources.get")

            def citation(self, row):
                return row

        class Documents:
            workspace_id = "workspace-606"

            def get_plan(self, *, registered_plan_id, attachment_ids):
                self.get_thread = threading.get_ident()
                return SimpleNamespace(document=document, asset=None, context_id="ctx-1")

            def pages_for_plan(self, *args):
                self.pages_thread = threading.get_ident()
                return None

        documents = Documents()

        class Tools:
            def __init__(self, settings, **kwargs):
                self.kwargs = kwargs
                self.index = kwargs["index"]

            def begin_run(self):
                pass

            def as_langchain_tools(self):
                return [local]

        class Output:
            answer = "方案追问答复"
            findings = []
            evidence = []
            evidence_ids = []
            limitations = []
            validation_notes = []
            status = "answered"
            stop_reason = "answered"
            source_coverage = {}
            reviewed_topics = []
            token_usage = {"total_tokens": 10}
            model_calls = 1
            tool_calls = 1
            model = "synthetic-async-model"
            report_format = "markdown"
            document = {"document_id": "PLAN-ASYNC"}
            runtime = {"page_sources": [], "web_sources": [], "native_tool_counts": {},
                       "report_format": "markdown"}
            raw_final_text = "方案追问答复"
            traces = []

            def as_history_turn(self):
                return {"document_id": "PLAN-ASYNC", "source_revision": "rev-1",
                        "question": "方案追问", "answer": self.answer}

            def to_dict(self):
                return {"answer": self.answer, "status": self.status}

        output = Output()

        class Agent:
            def __init__(self, observed_model, observed_document, observed_tools, **kwargs):
                self.model = observed_model
                self.document = observed_document
                self.tools = observed_tools
                self.progress_callback = kwargs["progress_callback"]
                self.web_provider = kwargs["web_provider"]

            async def arun(self, question, *, history=None):
                self.progress_callback({"phase": "review_batch", "batch": 1, "total_batches": 1})
                await run_local(self.document.read, start=1)
                response = await self.model.ainvoke([question])
                await self.tools.as_langchain_tools()[0].ainvoke({"query": "本地方案工具"})
                await self.web_provider.as_langchain_tool().ainvoke({"query": "方案联网"})
                return output

        def make_plan_tools(*args, **kwargs):
            return Tools(*args, **kwargs)

        def make_agent(*args, **kwargs):
            return Agent(*args, **kwargs)

        factory_kwargs = []

        def create_async_tool_chat_model(_settings, **kwargs):
            factory_kwargs.append(kwargs)
            return model

        request = {"task_id": "plan-async", "assistant_id": "plan", "plan_action": "question",
                   "prompt": "方案追问", "attachment_ids": [], "registered_plan_id": "PLAN-ASYNC",
                   "user_id": 606, "workspace_id": "workspace-606"}
        with patch("rag_arch.web.plan_runner.PlanStandardTools", side_effect=make_plan_tools), \
             patch("rag_arch.web.plan_runner.PlanReviewAgent", side_effect=make_agent), \
             patch("rag_arch.web.plan_runner.create_web_search_provider", return_value=web), \
             patch("rag_arch.tool_chat.create_async_tool_chat_model", new=create_async_tool_chat_model), \
             patch("rag_arch.web.plan_runner.create_tool_chat_model",
                   side_effect=AssertionError("async plan runner called sync model factory")), \
             patch("rag_arch.web.plan_runner.safe_error",
                   side_effect=lambda exc, **kwargs: {"component": "plan", "message": str(exc)}):
            result = await arun_plan(request, self.progress, asyncio.Event(),
                                     resources=Resources(), documents=documents,
                                     settings=self.settings)

        self.assertEqual(result["answer"], "方案追问答复")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertTrue(factory_kwargs[0]["stream_responses"])
        self.assertEqual(web.calls, ["方案联网"])
        self.assertTrue(web_client.closed)
        self.assertTrue(model.client.closed)
        self.assertNotEqual(documents.get_thread, self.loop_thread)
        self.assertNotEqual(documents.pages_thread, self.loop_thread)
        self.assertNotEqual(local_calls[0][0], self.loop_thread)
        self.assertIn("review_batch", [stage for stage, _ in self.progress_events])
        self.assertTrue(all(thread != self.loop_thread for thread in self.progress_threads))
        row, = self.audit_result_rows()
        self.assertEqual((row["result"]["audit_log"]["user_id"],
                          row["result"]["audit_log"]["workspace_id"]),
                         (606, "workspace-606"))


if __name__ == "__main__":
    unittest.main()
