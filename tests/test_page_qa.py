"""Page-QA protocol tests with local fake models; no provider requests."""
import base64
import json
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_arch.models import Answer, Chunk, ConversationTurn, SearchHit
from rag_arch.page_qa import invoke_with_pages
from rag_arch.plan_review_context import ContextBudgetError
from rag_arch.qa import answer_question


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII=")


def request(*calls):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": f"call-{i}"}
        for i, (name, args) in enumerate(calls)])


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.bound_schemas = []

    def bind_tools(self, tools):
        self.bound_schemas.append(tools)
        return SimpleNamespace(invoke=lambda messages: self._invoke(messages, True))

    def invoke(self, messages):
        return self._invoke(messages, False)

    def _invoke(self, messages, bound):
        self.calls.append((bound, list(messages)))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakePages:
    collection_id = "collection-A"

    def __init__(self):
        self.reads = []
        self.listings = []
        self.text = ""
        self.failure = None

    def sources(self):
        return [{"source_ref": "doc-A", "filename": "图纸.pdf", "page_count": 2}]

    def list_pages(self, **args):
        self.listings.append(args)
        return {"ok": True, "source_ref": args["source_ref"],
                "pages": [{"page": 1, "has_text": False}], "next_start": None}

    def read_page(self, source_ref, page):
        self.reads.append((source_ref, page))
        if self.failure:
            raise self.failure
        return {"ok": True, "source_ref": source_ref, "filename": "图纸.pdf", "page": page,
                "page_count": 2, "text": self.text, "image_bytes": PNG,
                "mime_type": "image/png", "width": 1, "height": 1}


class PageLoopTests(unittest.TestCase):
    messages = [SystemMessage(content="回答问题"), HumanMessage(content="有什么？")]

    def run_loop(self, model, service=None, **kwargs):
        return invoke_with_pages(model, self.messages, service or FakePages(),
                                 budget_tokens=kwargs.pop("budget_tokens", 20000), **kwargs)

    def test_model_can_answer_without_reading_pages(self):
        service = FakePages()
        model = FakeModel(AIMessage(content="已有文字足够。"))
        result = self.run_loop(model, service)
        self.assertEqual(result.response.content, "已有文字足够。")
        self.assertEqual(result.page_sources, [])
        self.assertEqual(service.reads, [])
        self.assertEqual(len(model.calls), 1)

    def test_multiple_tool_receipts_precede_all_images(self):
        model = FakeModel(request(
            ("read_document_page", {"source_ref": "doc-A", "page": 1}),
            ("read_document_page", {"source_ref": "doc-A", "page": 2})),
            AIMessage(content="两页分别说明。[P1][P2]"))
        result = self.run_loop(model)
        sent = model.calls[1][1]
        position = next(i for i, row in enumerate(sent) if isinstance(row, AIMessage))
        self.assertEqual([type(row) for row in sent[position + 1:]],
                         [ToolMessage, ToolMessage, HumanMessage, HumanMessage])
        self.assertEqual([row.tool_call_id for row in sent[position + 1:position + 3]],
                         ["call-0", "call-1"])
        self.assertEqual([row["display_id"] for row in result.page_sources], ["P1", "P2"])
        self.assertTrue(all(row["sent_to_model"] for row in result.page_sources))
        self.assertTrue(all(row["collection_id"] == "collection-A" for row in result.page_sources))
        self.assertNotIn("image_bytes", str(result.page_sources))
        self.assertNotIn("base64", str(result.page_sources))

    def test_listing_is_navigation_not_a_page_citation(self):
        model = FakeModel(request(("list_document_pages", {"source_ref": "doc-A"})),
                          AIMessage(content="需要明确哪页。"))
        service = FakePages()
        result = self.run_loop(model, service)
        self.assertEqual(service.listings, [{"source_ref": "doc-A", "start": 1,
                                            "limit": 20, "query": ""}])
        self.assertEqual(result.page_sources, [])

    def test_arbitrary_path_and_bad_parameters_never_reach_service(self):
        for name, args in [
            ("read_document_page", {"source_ref": "D:/private.pdf", "page": 1}),
            ("read_document_page", {"source_ref": "doc-A", "page": True}),
            ("read_document_page", {"source_ref": "doc-A", "page": 1, "path": "secret"}),
            ("list_document_pages", {"source_ref": "doc-A", "limit": 500}),
            ("unknown_tool", {"source_ref": "doc-A"}),
        ]:
            with self.subTest(name=name, args=args):
                model = FakeModel(request((name, args)), AIMessage(content="工具参数有误。"))
                service = FakePages()
                result = self.run_loop(model, service)
                self.assertEqual(service.reads, [])
                self.assertEqual(service.listings, [])
                receipt = next(row for row in model.calls[1][1] if isinstance(row, ToolMessage))
                self.assertFalse(json.loads(receipt.content)["ok"])
                self.assertEqual(result.page_sources, [])

    def test_service_failure_is_returned_without_fake_source(self):
        service = FakePages()
        service.failure = OSError("synthetic rendering error")
        model = FakeModel(request(("read_document_page", {"source_ref": "doc-A", "page": 1})),
                          AIMessage(content="本轮无法读取页面。"))
        result = self.run_loop(model, service)
        receipt = next(row for row in model.calls[1][1] if isinstance(row, ToolMessage))
        self.assertIn("synthetic rendering error", receipt.content)
        self.assertEqual(result.page_sources, [])

    def test_tool_limit_completes_entire_batch_then_one_plain_answer(self):
        service = FakePages()
        model = FakeModel(request(
            ("read_document_page", {"source_ref": "doc-A", "page": 1}),
            ("read_document_page", {"source_ref": "doc-A", "page": 2})),
            AIMessage(content="仅看了第一页。[P1]"))
        result = self.run_loop(model, service, max_tool_calls=1)
        self.assertEqual(service.reads, [("doc-A", 1)])
        self.assertEqual([bound for bound, _ in model.calls], [True, False])
        receipts = [row for row in model.calls[1][1] if isinstance(row, ToolMessage)]
        self.assertEqual(len(receipts), 2)
        self.assertFalse(json.loads(receipts[1].content)["ok"])
        self.assertTrue(result.warnings)

    def test_budget_checks_initial_and_tool_enlarged_input(self):
        model = FakeModel(AIMessage(content="must not run"))
        with self.assertRaises(ContextBudgetError):
            self.run_loop(model, budget_tokens=10)
        self.assertEqual(model.calls, [])
        service = FakePages()
        service.text = "中" * 6000
        model = FakeModel(request(("read_document_page", {"source_ref": "doc-A", "page": 1})))
        with self.assertRaises(ContextBudgetError):
            self.run_loop(model, service, budget_tokens=5000)
        self.assertEqual(len(model.calls), 1)

    def test_repeated_page_keeps_the_same_current_turn_id(self):
        call = ("read_document_page", {"source_ref": "doc-A", "page": 1})
        model = FakeModel(request(call), request(call), AIMessage(content="同一页。[P1]"))
        result = self.run_loop(model)
        self.assertEqual(len(result.page_sources), 1)
        self.assertEqual(result.page_sources[0]["display_id"], "P1")


class PageAnswerIntegrationTests(unittest.TestCase):
    def ask(self, model, service, **kwargs):
        return answer_question("这个图说明什么？", None, model, direct_chunks=[],
                               page_service=service, context_budget_tokens=30000,
                               output_budget_tokens=1000, **kwargs)

    def answer_model(self, text="图中信息。[P1]"):
        return FakeModel(request(("read_document_page", {"source_ref": "doc-A", "page": 1})),
                         AIMessage(content=text))

    def test_image_only_answer_has_valid_page_reference_without_hits(self):
        answer = self.ask(self.answer_model(), FakePages())
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.hits, [])
        self.assertEqual(answer.citation_check["cited_page_numbers"], [1])
        self.assertTrue(answer.page_sources[0]["sent_to_model"])

    def test_image_only_history_keeps_locations_but_not_old_images(self):
        service = FakePages()
        first = self.ask(self.answer_model(), service)
        model = self.answer_model()
        history = [ConversationTurn("HISTORY_MARKER [P1]", first)]
        answer = self.ask(model, service, history=history)
        self.assertTrue(answer.grounded)
        initial = model.calls[0][1]
        self.assertIn("HISTORY_MARKER", str(initial))
        self.assertIn("previous_page_locations", str(initial))
        self.assertNotIn("base64", str(initial))
        history_body = next(row.content for row in initial if isinstance(row, HumanMessage)
                            and isinstance(row.content, str) and row.content.startswith("对话历史"))
        old_rows = json.JSONDecoder().raw_decode(history_body.split("：\n", 1)[1])[0]
        self.assertNotIn("[P1]", str(old_rows))
        self.assertEqual(old_rows[0]["previous_page_locations"][0]["page"], 1)

    def test_new_collection_does_not_reuse_old_image_only_turn(self):
        service = FakePages()
        first = self.ask(self.answer_model(), service)
        other = FakePages()
        other.collection_id = "collection-B"
        model = self.answer_model()
        self.ask(model, other, history=[ConversationTurn("OTHER_PRIVATE_MARKER", first)])
        self.assertNotIn("OTHER_PRIVATE_MARKER", str(model.calls[0][1]))

    def test_text_and_page_citations_remain_separate(self):
        chunk = Chunk("d", "图纸.pdf", "c", "文字说明。", {"page": 1}, "pdf", "collection-A")
        model = self.answer_model("文字说明与图中内容。[S1][P1]")
        answer = answer_question("图文说明", None, model, direct_chunks=[chunk],
                                 page_service=FakePages(), context_budget_tokens=30000,
                                 output_budget_tokens=1000)
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citation_check["cited_numbers"], [1])
        self.assertEqual(answer.citation_check["cited_page_numbers"], [1])
        self.assertEqual(answer.hits[0].chunk.text, "文字说明。")
        self.assertEqual(answer.page_sources[0]["display_id"], "P1")

    def test_unread_or_bad_page_citation_is_rejected(self):
        for text in ["没有读过。[P2]", "错误编号。[Pabc]"]:
            with self.subTest(text=text):
                answer = self.ask(self.answer_model(text), FakePages())
                self.assertFalse(answer.grounded)
                self.assertEqual(answer.text, text)
                self.assertFalse(answer.citation_check["passed"])
        answer = self.ask(FakeModel(AIMessage(content="并没有读取。[P1]")), FakePages())
        self.assertFalse(answer.grounded)


class FakeWeb:
    available = True
    status = {"enabled": True}

    def __init__(self, *, performed=True, url="https://example.invalid/official"):
        self.queries = []
        self.performed, self.url = performed, url
        self.failure = None

    def as_langchain_tool(self):
        return {"type": "function", "function": {"name": "search_web",
            "description": "Search public pages", "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

    def search(self, query):
        self.queries.append(query)
        if self.failure:
            raise self.failure
        return {"ok": True, "performed": self.performed, "query": query,
                "summary": "搜索服务给出的外部摘要。", "sources": [
                    {"url": self.url, "title": "公开资料", "snippet": "外部网页说明。"}]}


class WebAnswerTests(unittest.TestCase):
    def ask(self, model, provider, **kwargs):
        return answer_question("请查看公开资料", None, model, web_provider=provider,
                               context_budget_tokens=30000, output_budget_tokens=1000, **kwargs)

    def search_model(self):
        return FakeModel(request(("search_web", {"query": "公开资料"})),
                         AIMessage(content="外部网页说明。[W1]"))

    def test_chat_without_attachments_can_search_and_return_real_sources(self):
        web = FakeWeb()
        answer = self.ask(self.search_model(), web)
        self.assertEqual(web.queries, ["公开资料"])
        self.assertEqual(answer.hits, [])
        self.assertEqual(answer.page_sources, [])
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.web_sources[0]["url"], web.url)
        self.assertEqual(answer.web_sources[0]["display_id"], "W1")
        self.assertFalse(answer.web_sources[0]["verified"])
        self.assertEqual(answer.citation_check["cited_web_numbers"], [1])
        self.assertTrue(any("外部参考" in note for note in answer.warnings))

    def test_optional_search_does_not_require_a_search_on_every_chat_turn(self):
        web = FakeWeb()
        model = FakeModel(AIMessage(content="你好，可以继续聊。"))
        answer = self.ask(model, web)
        self.assertEqual(answer.text, "你好，可以继续聊。")
        self.assertEqual(web.queries, [])
        self.assertFalse(answer.grounded)
        self.assertEqual(answer.web_sources, [])

    def test_disabled_provider_keeps_plain_chat_without_binding_tools(self):
        web = FakeWeb()
        web.available = False
        model = FakeModel(AIMessage(content="普通聊天回答"))
        answer = self.ask(model, web)
        self.assertEqual(answer.text, "普通聊天回答")
        self.assertEqual(model.bound_schemas, [])
        self.assertEqual(web.queries, [])
        bare = FakeModel(AIMessage(content="无工具"))
        result = invoke_with_pages(bare, [HumanMessage(content="继续")], budget_tokens=1000)
        self.assertEqual(result.response.content, "无工具")
        self.assertEqual(bare.bound_schemas, [])

    def test_unperformed_search_does_not_create_a_citable_url(self):
        answer = self.ask(self.search_model(), FakeWeb(performed=False))
        self.assertEqual(answer.web_sources, [])
        self.assertFalse(answer.grounded)
        self.assertEqual(answer.text, "外部网页说明。[W1]")
        self.assertFalse(answer.citation_check["passed"])

    def test_search_failure_is_visible_to_model_without_invented_sources(self):
        web = FakeWeb()
        web.failure = OSError("synthetic web failure")
        model = FakeModel(request(("search_web", {"query": "公开资料"})),
                          AIMessage(content="本轮搜索失败，请稍后重试。"))
        answer = self.ask(model, web)
        receipts = [row for row in model.calls[1][1] if isinstance(row, ToolMessage)]
        self.assertIn("synthetic web failure", receipts[0].content)
        self.assertFalse(json.loads(receipts[0].content)["ok"])
        self.assertEqual(answer.web_sources, [])

    def test_web_and_page_tools_share_limit_and_receipt_order(self):
        web, pages = FakeWeb(), FakePages()
        model = FakeModel(request(
            ("search_web", {"query": "公开资料"}),
            ("read_document_page", {"source_ref": "doc-A", "page": 1})),
            AIMessage(content="网页和原图。[W1][P1]"))
        result = invoke_with_pages(model, [HumanMessage(content="核对图文")], pages,
                                   web_provider=web, budget_tokens=20000)
        sent = model.calls[1][1]
        start = next(i for i, row in enumerate(sent) if isinstance(row, AIMessage))
        self.assertEqual([type(row) for row in sent[start+1:]],
                         [ToolMessage, ToolMessage, HumanMessage])
        self.assertEqual(len(result.web_sources), 1)
        self.assertEqual(len(result.page_sources), 1)
        limited = FakeModel(request(
            ("search_web", {"query": "公开资料"}),
            ("read_document_page", {"source_ref": "doc-A", "page": 1})),
            AIMessage(content="仅网页。[W1]"))
        pages = FakePages()
        result = invoke_with_pages(limited, [HumanMessage(content="核对")], pages,
                                   web_provider=FakeWeb(), budget_tokens=20000, max_tool_calls=1)
        self.assertEqual(pages.reads, [])
        self.assertEqual([bound for bound, _ in limited.calls], [True, False])
        self.assertTrue(result.warnings)

    def test_history_keeps_url_pointer_without_reusing_old_web_citation(self):
        first = self.ask(self.search_model(), FakeWeb())
        current_web = FakeWeb(url="https://example.invalid/new")
        model = self.search_model()
        second = self.ask(model, current_web, history=[ConversationTurn("之前的问题[W1]", first)])
        history_body = next(row.content for row in model.calls[0][1] if isinstance(row, HumanMessage)
                            and isinstance(row.content, str) and row.content.startswith("对话历史"))
        old = json.JSONDecoder().raw_decode(history_body.split("：\n", 1)[1])[0]
        self.assertNotIn("[W1]", str(old))
        self.assertEqual(old[0]["previous_web_locations"][0]["url"], "https://example.invalid/official")
        self.assertEqual([row["url"] for row in second.web_sources], [current_web.url])

    def test_missing_local_hits_can_use_external_references(self):
        chunk = Chunk("d", "资料.txt", "c", "不相关正文", {}, "txt", "collection-A")
        local = SimpleNamespace(chunks=[chunk], search=lambda query, top_k: [])
        answer = answer_question("新规范来源", local, self.search_model(), web_provider=FakeWeb(),
                                 context_budget_tokens=30000, output_budget_tokens=1000)
        self.assertEqual(answer.hits, [])
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.web_sources[0]["collection_id"], "collection-A")


class FakeNativeModel(FakeModel):
    native_tool_names = ("web_search", "web_extractor", "code_interpreter")

    def __init__(self, *responses):
        super().__init__(*responses)
        self.plain_calls = []

    def plain(self):
        def invoke(messages):
            self.plain_calls.append(list(messages))
            return self._invoke(messages, False)
        return SimpleNamespace(invoke=invoke)


def native_reply(text, *, sources=True, calls=None):
    kinds = calls or ["web_search", "web_extractor", "code_interpreter"]
    return AIMessage(content=text, additional_kwargs={
        "native_tool_observations": [{"id": f"native-{i}", "type": kind, "status": "completed"}
                                     for i, kind in enumerate(kinds)],
        "native_tool_usage": {"counts": {kind: 1 for kind in kinds}, "provider_usage": {"input_tokens": 20}},
        "native_web_sources": ([{"url": "https://example.invalid/native", "title": "原生来源",
                                  "source_origin": "provider_metadata"}] if sources else [])})


class NativeAnswerTests(unittest.TestCase):
    def test_native_plain_chat_records_real_sources_without_duplicate_search(self):
        fallback = FakeWeb()
        response = native_reply("原生工具结论，不强造W编号。")
        model = FakeNativeModel(response)
        answer = answer_question("当前信息", None, model, web_provider=fallback,
                                 context_budget_tokens=30000, output_budget_tokens=1000)
        self.assertEqual(model.bound_schemas, [])
        self.assertEqual(fallback.queries, [])
        self.assertEqual(answer.text, response.content)
        self.assertEqual(answer.web_sources[0]["url"], "https://example.invalid/native")
        self.assertEqual(answer.citation_check["cited_web_numbers"], [])
        self.assertTrue(answer.citation_check["native_reference_metadata"])
        self.assertEqual(answer.native_tool_usage["counts"]["code_interpreter"], 1)
        self.assertEqual(len(answer.native_tool_observations), 3)
        self.assertFalse(any(isinstance(row, ToolMessage) for row in model.calls[0][1]))

    def test_native_code_only_still_keeps_observations_without_web_sources(self):
        model = FakeNativeModel(native_reply("计算结果为2。", sources=False, calls=["code_interpreter"]))
        answer = answer_question("计算", None, model, context_budget_tokens=30000, output_budget_tokens=1000)
        self.assertEqual(answer.text, "计算结果为2。")
        self.assertEqual(answer.web_sources, [])
        self.assertEqual(answer.native_tool_usage["counts"], {"code_interpreter": 1})

    def test_internal_query_rewrite_disables_native_tools(self):
        chunk = Chunk("d", "资料.txt", "c", "钢筋应检查。", {}, "txt", "collection-A")
        local = SimpleNamespace(chunks=[chunk], search=lambda query, top_k: [SearchHit(chunk, 1)])
        history = [ConversationTurn("钢筋？", Answer("应检查。[S1]", [SearchHit(chunk, 1)], [], True))]
        model = FakeNativeModel(AIMessage(content="钢筋应如何检查？"),
                                native_reply("应检查。[S1]", sources=False, calls=["code_interpreter"]))
        answer = answer_question("它呢？", local, model, history=history,
                                 context_budget_tokens=30000, output_budget_tokens=1000)
        self.assertTrue(answer.grounded)
        self.assertEqual(len(model.plain_calls), 1)
        self.assertIn("只输出问题", model.plain_calls[0][0].content)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(answer.native_tool_usage["counts"]["code_interpreter"], 1)

    def test_tool_ceiling_disables_native_tools_for_last_summary(self):
        model = FakeNativeModel(request(("read_document_page", {"source_ref": "doc-A", "page": 1})),
                                AIMessage(content="所读页面。[P1]"))
        result = invoke_with_pages(model, [HumanMessage(content="读图")], FakePages(),
                                   budget_tokens=20000, max_tool_calls=1)
        self.assertEqual(len(model.plain_calls), 1)
        self.assertEqual(result.response.content, "所读页面。[P1]")


if __name__ == "__main__":
    unittest.main()
