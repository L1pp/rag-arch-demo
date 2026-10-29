"""Offline provider adaptation and stream recovery using synthetic HTTP transports."""
import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage

from rag_arch.config import Settings
from rag_arch.providers import create_async_chat_model, create_chat_model
from rag_arch.responses_chat import assemble_streamed_response
from rag_arch.runtime_audit import record_run
from rag_arch.tool_chat import create_async_tool_chat_model, create_tool_chat_model
from rag_arch.web.plan_runner import _plan_settings
from rag_arch.web_search import create_web_search_provider, main_native_tools, select_chat_protocol


def settings():
    return Settings(
        chat_model="deepseek-v4.1-flash-1017",
        chat_base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        chat_api_key="synthetic-secret",
        chat_max_tokens=131072,
        chat_thinking_budget=4096,
        plan_max_tokens=131072,
        plan_thinking_budget=16384,
    )


def response(output):
    return {"id": "mock-1", "object": "response", "model": "deepseek-v4.1-flash-1017",
            "status": "completed", "output": output}


def text_response():
    return response([{"type": "message", "id": "message-1", "status": "completed",
                      "role": "assistant", "content": [{"type": "output_text", "text": "完成"}]}])


class PublicProviderSyncTests(unittest.TestCase):
    def test_exact_model_protocol_and_tool_switches(self):
        s = settings()
        self.assertEqual(select_chat_protocol(s), "responses")
        self.assertEqual(select_chat_protocol(replace(s, chat_api_protocol="responses")), "responses")
        self.assertEqual({t["type"] for t in main_native_tools(s)},
                         {"web_search", "web_extractor", "code_interpreter"})
        self.assertEqual(main_native_tools(replace(s, chat_web_search_enabled=False)),
                         [{"type": "code_interpreter"}])
        self.assertEqual(select_chat_protocol(replace(s, chat_model="deepseek-v4.1-flash-1017-1234")),
                         "chat_completions")
        self.assertEqual(select_chat_protocol(replace(s, chat_base_url="https://example.invalid/v1")),
                         "chat_completions")


    def test_async_chat_and_tool_factories_keep_same_wire_policy(self):
        captured = []
        async def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=text_response())
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                for factory in (create_async_chat_model, create_async_tool_chat_model):
                    model = factory(settings(), http_async_client=client)
                    self.assertEqual(model.reasoning_configuration["configured_thinking_budget"], 4096)
                    self.assertEqual((await model.ainvoke([HumanMessage(content="测试")])).content, "完成")
        asyncio.run(run())
        self.assertEqual(len(captured), 2)
        for body in captured:
            self.assertNotIn("thinking_budget", body)
            self.assertNotIn("reasoning", body)


    def test_independent_search_uses_model_specific_reasoning_in_sync_and_async(self):
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=response([]))

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            result = create_web_search_provider(settings(), client=client).search("公开查询")
            self.assertTrue(result["ok"])
            qwen = replace(settings(), chat_model="qwen3.8-flash")
            self.assertTrue(create_web_search_provider(qwen, client=client).search("公开查询")["ok"])

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                result = await create_web_search_provider(settings(), async_client=client).asearch("公开查询")
                self.assertTrue(result["ok"])
                qwen = replace(settings(), chat_model="qwen3.8-flash")
                result = await create_web_search_provider(qwen, async_client=client).asearch("公开查询")
                self.assertTrue(result["ok"])

        asyncio.run(run())
        self.assertEqual(len(captured), 4)
        for body in (captured[0], captured[2]):
            self.assertNotIn("reasoning", body)
            self.assertNotIn("thinking_budget", body)
        for body in (captured[1], captured[3]):
            self.assertEqual(body["reasoning"], {"effort": "low"})


    def test_stream_disconnect_records_bounded_previews_without_changing_error(self):
        failure = httpx.RemoteProtocolError("private transport detail synthetic-secret")

        def event(kind, **fields):
            return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **fields}, ensure_ascii=False)
                    + "\n\n").encode("utf-8")

        class DisconnectStream(httpx.AsyncByteStream):
            def __init__(self):
                self.closed = False

            async def __aiter__(self):
                yield event("response.created", response={"id": "mock-disconnect"})
                yield event("response.reasoning_summary_text.delta", delta="思考 synthetic-secret")
                yield event("response.output_text.delta", delta="已收到少量输出")
                yield b": keepalive\n\n"
                raise failure

            async def aclose(self):
                self.closed = True

        stream = DisconnectStream()
        def handle(_request):
            return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                configured = _plan_settings(settings())
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(httpx.RemoteProtocolError) as caught:
                        with record_run("deepseek-disconnect-mock", configured, "测试", root=folder) as audit:
                            model = create_tool_chat_model(configured, http_async_client=client, stream_responses=True)
                            await model.ainvoke([HumanMessage(content="检查断流")])
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(caught.exception.request_accepted)
                    raw = audit.events.read_text(encoding="utf-8")
                    self.assertNotIn("synthetic-secret", raw)
                    self.assertNotIn("private transport detail", raw)
                    rows = [json.loads(line) for line in raw.splitlines()]
                    previews = [row for row in rows if row.get("kind") == "model_http_stream"
                                and "delta_preview" in row]
                    self.assertEqual([row["type"] for row in previews], [
                        "response.reasoning_summary_text.delta", "response.output_text.delta"])
                    self.assertEqual(previews[0]["delta_preview"], "思考 [REDACTED]")
                    self.assertEqual(previews[1]["delta_preview"], "已收到少量输出")
                    self.assertTrue(all(len(row["delta_preview"]) <= 240 for row in previews))
                    errors = [row for row in rows if row.get("kind") == "model_http_stream_error"]
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(errors[0]["error_type"], "RemoteProtocolError")
                    self.assertEqual(errors[0]["stage"], "receiving_events")
                    self.assertTrue(errors[0]["request_accepted"])
                    self.assertEqual(errors[0]["heartbeat_count"], 1)
                    self.assertGreaterEqual(errors[0]["elapsed_seconds"], 0)
                    self.assertTrue(errors[0]["at"])
        asyncio.run(run())
        self.assertTrue(stream.closed)


    def test_failed_preview_audit_does_not_mask_disconnect_or_skip_error_record(self):
        from rag_arch.runtime_audit import arecord_event as real_record_event
        failure = httpx.RemoteProtocolError("private transport detail")

        class DisconnectStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"partial"}\n\n'
                raise failure

        def handle(_request):
            return httpx.Response(200, stream=DisconnectStream(), headers={"content-type": "text/event-stream"})

        async def failing_preview(kind, **fields):
            if kind == "model_http_stream" and fields.get("type") == "response.output_text.delta":
                raise OSError("simulated audit failure")
            await real_record_event(kind, **fields)

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                configured = _plan_settings(settings())
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(httpx.RemoteProtocolError) as caught:
                        with record_run("deepseek-audit-failure-mock", configured, "测试", root=folder) as audit:
                            with patch("rag_arch.runtime_audit.arecord_event", side_effect=failing_preview):
                                model = create_tool_chat_model(configured, http_async_client=client, stream_responses=True)
                                await model.ainvoke([HumanMessage(content="检查断流")])
                    self.assertIs(caught.exception, failure)
                    rows = [json.loads(line) for line in audit.events.read_text(encoding="utf-8").splitlines()]
                    errors = [row for row in rows if row.get("kind") == "model_http_stream_error"]
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(errors[0]["error_type"], "RemoteProtocolError")
                    self.assertEqual(errors[0]["stage"], "receiving_events")
        asyncio.run(run())


    def test_failed_timeout_preview_audit_does_not_mask_timeout(self):
        from rag_arch.runtime_audit import arecord_event as real_record_event
        failure = httpx.ReadTimeout("synthetic timeout")

        class TimeoutStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"partial"}\n\n'
                raise failure

        def handle(_request):
            return httpx.Response(200, stream=TimeoutStream(), headers={"content-type": "text/event-stream"})

        async def failing_preview(kind, **fields):
            if kind == "model_http_stream" and fields.get("type") == "response.output_text.delta":
                raise OSError("simulated audit failure")
            await real_record_event(kind, **fields)

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                configured = _plan_settings(settings())
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(httpx.ReadTimeout) as caught:
                        with record_run("deepseek-timeout-audit-mock", configured, "测试", root=folder) as audit:
                            with patch("rag_arch.runtime_audit.arecord_event", side_effect=failing_preview):
                                model = create_tool_chat_model(configured, http_async_client=client, stream_responses=True)
                                await model.ainvoke([HumanMessage(content="检查超时")])
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(caught.exception.request_accepted)
                    rows = [json.loads(line) for line in audit.events.read_text(encoding="utf-8").splitlines()]
                    errors = [row for row in rows if row.get("kind") == "model_http_stream_error"]
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(errors[0]["timeout_kind"], "read_idle")
                    self.assertTrue(errors[0]["request_accepted"])
                    self.assertEqual(errors[0]["audit_error_types"], ["OSError"])
                    self.assertEqual(caught.exception.audit_error_types, ["OSError"])
        asyncio.run(run())


    def test_failed_recovered_timeout_response_audit_preserves_timeout(self):
        from rag_arch.runtime_audit import arecord_event as real_record_event
        failure = httpx.ReadTimeout("synthetic timeout")
        item = {"type": "message", "id": "partial", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": "已收到部分内容"}]}

        class TimeoutStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                event = {"type": "response.output_item.done", "item": item}
                yield ("event: response.output_item.done\ndata: " + json.dumps(event, ensure_ascii=False)
                       + "\n\n").encode("utf-8")
                raise failure

        def handle(_request):
            return httpx.Response(200, stream=TimeoutStream(), headers={"content-type": "text/event-stream"})

        async def failing_response(kind, **fields):
            if kind == "model_http_response":
                raise OSError("simulated recovered response audit failure")
            await real_record_event(kind, **fields)

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                configured = _plan_settings(settings())
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(httpx.ReadTimeout) as caught:
                        with record_run("deepseek-timeout-response-mock", configured, "测试", root=folder):
                            with patch("rag_arch.runtime_audit.arecord_event", side_effect=failing_response):
                                model = create_tool_chat_model(configured, http_async_client=client, stream_responses=True)
                                await model.ainvoke([HumanMessage(content="检查超时")])
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(caught.exception.request_accepted)
                    self.assertEqual(caught.exception.audit_error_types, ["OSError"])
        asyncio.run(run())


    def test_other_models_keep_old_reasoning_and_chat_completions_rejects_budget(self):
        s = settings()
        qwen = create_chat_model(replace(s, chat_model="qwen3.8-flash", chat_thinking_budget=16384))
        self.assertEqual(qwen.reasoning_effort, "medium")
        self.assertEqual(qwen.extra_body["thinking_budget"], 16384)
        qwen_body, _, _, _ = qwen._request_data([HumanMessage(content="测试")])
        self.assertEqual(qwen_body["reasoning"], {"effort": "medium"})
        self.assertNotIn("thinking_budget", qwen_body)
        with self.assertRaisesRegex(ValueError, "extra_body"):
            qwen.model_copy(update={"extra_body": {"not_supported": True}})._request_data(
                [HumanMessage(content="测试")])
        unsupported = replace(s, chat_model="kimi-k3")
        with self.assertRaises(ValueError):
            create_chat_model(unsupported)
        with self.assertRaises(ValueError):
            create_async_chat_model(unsupported)
        chat = replace(s, chat_api_protocol="chat_completions")
        with self.assertRaises(ValueError):
            create_chat_model(chat)
        with self.assertRaises(ValueError):
            create_tool_chat_model(chat)
        self.assertIsNotNone(create_tool_chat_model(replace(chat, chat_thinking_budget=None)))

        async def reject_async_chat_completions():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
                for factory in (create_async_chat_model, create_async_tool_chat_model):
                    with self.assertRaises(ValueError):
                        factory(chat, http_async_client=client)
        asyncio.run(reject_async_chat_completions())


    def test_completed_items_are_recovered_only_when_terminal_output_is_missing(self):
        item = {"type": "message", "id": "item-1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": "合成回答"}]}
        done = ("response.output_item.done", {"type": "response.output_item.done", "item": item})
        terminal = ("response.completed", {"type": "response.completed",
                                          "response": {"status": "completed", "usage": {"total_tokens": 3}}})
        payload, items, error = assemble_streamed_response([done, terminal])
        self.assertEqual(payload["output"], [item])
        self.assertEqual(payload["usage"]["total_tokens"], 3)
        self.assertEqual(items, [item])
        self.assertIsNone(error)
        explicit_empty = ("response.completed", {"type": "response.completed",
                        "response": {"status": "completed", "output": []}})
        payload, _, _ = assemble_streamed_response([done, explicit_empty])
        self.assertEqual(payload["output"], [])

    def test_streamed_completed_item_reaches_caller_when_terminal_omits_output(self):
        item = {"type": "message", "id": "item-1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": "合成回答"}]}
        events = [
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": "mock", "status": "completed"}},
        ]
        body = ''.join('event: '+event["type"]+'\ndata: '+json.dumps(event, ensure_ascii=False)+'\n\n'
                       for event in events).encode('utf-8')
        captured = []
        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                model = create_tool_chat_model(settings(), http_async_client=client, stream_responses=True)
                message = await model.ainvoke([HumanMessage(content="自编问题")])
                self.assertEqual(message.content, "合成回答")
        asyncio.run(run())
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["stream"])
        self.assertNotIn("thinking_budget", captured[0])
        self.assertNotIn("reasoning", captured[0])


if __name__ == "__main__":
    unittest.main()
