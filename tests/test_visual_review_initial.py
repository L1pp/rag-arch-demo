"""The first request contains actual, conservative and budgeted images."""
from concurrent.futures import CancelledError
from io import BytesIO
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from rag_arch.multimodal import page_image_sources, page_image_message
from rag_arch.plan_review_context import estimate_messages
from rag_arch.page_regions import page_image_id
from rag_arch.plan_review_agent import PlanReviewAgent, plan_system_prompt
from rag_arch.tool_chat import create_tool_chat_model
from rag_arch.request_control import request_control_scope
from rag_arch.web.task_worker import TaskControlUnavailable, StaleWorkerAttempt
from rag_arch.visual_review_policy import bounded_image_messages, REQUEST_IMAGE_BYTES
from tests.helpers.visual_fixtures import FakeChat, FakeTools, action, call
from tests.helpers.visual_fixtures import Pages
from tests.helpers.visual_fixtures import document, final
from tests.helpers.visual_fixtures import response_settings, payload, text_output


class InitialPages(Pages):
    def __init__(self, *, strong=True, later=False, failure=None, oversized=False):
        self.strong, self.later = strong, later
        self.failure, self.oversized = failure, oversized
        self.reads, self.lists = [], []

    def list_pages(self, source_ref, start=1, limit=20, query="", visual_only=False):
        self.lists.append(start)
        row = {"page": 37 if self.later else 2, "text_preview": "Dimensions 24mm 36mm",
               "visual_signals": ["table_grid"] if self.strong else ["title_cue"],
               "auto_attach_eligible": self.strong,
               "auto_attach_reasons": ["populated_numeric_table_grid"] if self.strong else [],
               "auto_attach_priority": 2, "auto_attach_table_region": [0.2, 0.2, 0.8, 0.8]}
        return {"ok": True, "page_count": 60, "visual_candidate_count": 2 if self.later else 1,
                "pages": [{"page": 1, "visual_signals": ["title_cue"]}] if self.later and start == 1 else [row],
                "next_start": 37 if self.later and start == 1 else None}

    def read_page(self, source_ref, page, region=None):
        self.reads.append((page, region))
        if self.failure:
            raise self.failure
        result = {**super().read_page(source_ref, 1), "page": page, "page_count": 60,
                  "region": region, "source_kind": "plan", "page_numbering": "physical_page"}
        if self.oversized:
            result["image_bytes"] = b"x" * (1024 * 1024 + 1)
        return result


class InitialVisualTests(unittest.TestCase):
    def test_initial_picture_reaches_first_production_responses_http_input(self):
        seen = []
        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=payload([text_output("关键数字仍须结合部位核对。")]))
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            model = create_tool_chat_model(response_settings(), http_client=client, output_method="text")
            result = PlanReviewAgent(model, document(), FakeTools(), page_service=InitialPages(),
                                     report_format="markdown").run()
        blocks = [block for item in seen[0]["input"] if isinstance(item.get("content"), list)
                  for block in item["content"]]
        images = [block for block in blocks if block.get("type") == "input_image"]
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["image_url"].startswith("data:image/png;base64,"))
        tool = next(tool for tool in seen[0]["tools"] if tool.get("name") == "read_document_page")
        self.assertEqual(set(tool["parameters"]["required"]), {"source_ref", "page"})
        self.assertIn("region", tool["parameters"]["properties"])
        self.assertTrue(result.runtime["visual_review"]["attempts"][0]["response_received"])

    def test_async_first_request_uses_the_same_auto_image_and_ledger(self):
        class AsyncChat(FakeChat):
            async def ainvoke(self, messages, **kwargs):
                return self.invoke(messages, **kwargs)
        model = AsyncChat([final([])])
        agent = PlanReviewAgent(model, document(), FakeTools(), page_service=InitialPages())
        result = asyncio.run(agent.arun())
        self.assertEqual(len(page_image_sources(model.requests[0])), 1)
        self.assertTrue(result.runtime["visual_review"]["images"][0]["response_received"])

    def test_real_image_is_in_first_request_without_invented_tool_receipt(self):
        pages = InitialPages()
        model = FakeChat([final([])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=pages).run()
        first = model.requests[0]
        self.assertIn("data:image/png;base64,", str([m.content for m in first]))
        self.assertFalse(any(isinstance(m, ToolMessage) for m in first))
        sources = page_image_sources(first)
        self.assertEqual(len(sources), 1)
        self.assertTrue(sources[0]["automatic"])
        ledger = result.runtime["visual_review"]
        self.assertEqual(ledger["initial_selection"]["selected_pages"], [2])
        self.assertEqual(ledger["attempts"][0]["request_index"], 1)
        self.assertTrue(ledger["images"][0]["response_received"])
        self.assertFalse(ledger["images"][0]["recognition_verified"])

    def test_uncertain_title_or_scan_does_not_auto_read(self):
        pages = InitialPages(strong=False)
        model = FakeChat([final([])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=pages).run()
        self.assertEqual(pages.reads, [])
        self.assertFalse(page_image_sources(model.requests[0]))
        self.assertEqual(result.runtime["visual_review"]["images"], [])

    def test_selection_includes_later_directory_pages(self):
        pages = InitialPages(later=True)
        result = PlanReviewAgent(FakeChat([final([])]), document(), FakeTools(), page_service=pages).run()
        self.assertEqual(pages.lists, [1, 37])
        self.assertEqual(pages.reads, [(37, None)])
        self.assertEqual(result.runtime["visual_review"]["initial_selection"]["selected_pages"], [37])

    def test_question_does_not_auto_read_whole_plan(self):
        pages = InitialPages()
        model = FakeChat([final([])])
        PlanReviewAgent(model, document(), FakeTools(), page_service=pages, mode="question").run("仅说明正文")
        self.assertEqual(pages.reads, [])
        self.assertEqual(pages.lists, [])
        self.assertFalse(page_image_sources(model.requests[0]))

    def test_oversized_initial_view_is_acquired_but_not_sent(self):
        pages = InitialPages(oversized=True)
        model = FakeChat([final([])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=pages).run()
        visual = result.runtime["visual_review"]
        self.assertTrue(visual["images"][0]["acquired"])
        self.assertFalse(visual["images"][0]["request_attempted"])
        self.assertFalse(page_image_sources(model.requests[0]))
        self.assertEqual(visual["initial_selection"]["skipped"][0]["reason"], "initial_image_budget")

    def test_initial_read_failure_does_not_abort_text_review(self):
        result = PlanReviewAgent(FakeChat([final([])]), document(), FakeTools(),
                                 page_service=InitialPages(failure=ValueError("synthetic"))).run()
        self.assertEqual(result.stop_reason, "answered")
        self.assertEqual(result.runtime["visual_review"]["read_failures"][0]["error_type"], "ValueError")

    def test_cancel_before_initial_read_propagates(self):
        with self.assertRaises(CancelledError):
            PlanReviewAgent(FakeChat([final([])]), document(), FakeTools(),
                            page_service=InitialPages(failure=CancelledError())).run()

    def test_control_plane_failures_do_not_become_visual_fallbacks(self):
        for error in (TaskControlUnavailable, StaleWorkerAttempt):
            with self.subTest(error=error.__name__, phase="directory"):
                model = FakeChat([])
                def guard():
                    raise error("synthetic control failure")
                with request_control_scope(guard), self.assertRaises(error):
                    PlanReviewAgent(model, document(), FakeTools(), page_service=InitialPages()).run()
                self.assertEqual(model.requests, [])
            with self.subTest(error=error.__name__, phase="initial_read"):
                model = FakeChat([])
                with self.assertRaises(error):
                    PlanReviewAgent(model, document(), FakeTools(),
                                    page_service=InitialPages(failure=error("synthetic"))).run()
                self.assertEqual(model.requests, [])
            with self.subTest(error=error.__name__, phase="tool_read"):
                model = FakeChat([action(call("read_document_page", "image", {
                    "source_ref": "plan:current", "page": 2}))])
                with self.assertRaises(error):
                    PlanReviewAgent(model, document(), FakeTools(),
                        page_service=InitialPages(strong=False, failure=error("synthetic"))).run()
                self.assertEqual(len(model.requests), 1)

    def test_table_region_hint_is_transferred_without_counting_as_a_crop(self):
        result = PlanReviewAgent(FakeChat([final([])]), document(), FakeTools(), page_service=InitialPages()).run()
        self.assertEqual(result.runtime["visual_page_index"]["pages"][0]["auto_attach_table_region"],
                         [0.2, 0.2, 0.8, 0.8])
        self.assertIsNone(result.runtime["page_sources"][0]["region"])

    def test_initial_wrong_source_and_tool_ignored_region_are_not_accepted(self):
        class WrongSource(InitialPages):
            def read_page(self, source_ref, page, region=None):
                return {**super().read_page(source_ref, page, region), "source_ref": "plan:other"}
        model = FakeChat([final([])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=WrongSource()).run()
        self.assertFalse(page_image_sources(model.requests[0]))
        self.assertEqual(result.runtime["page_sources"], [])
        class IgnoreRegion(InitialPages):
            def read_page(self, source_ref, page, region=None):
                return super().read_page(source_ref, page, None)
        region = [0.2, 0.2, 0.8, 0.8]
        model = FakeChat([action(call("read_document_page", "crop", {
            "source_ref": "plan:current", "page": 2, "region": region})), final([])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=IgnoreRegion()).run()
        self.assertEqual(len(result.runtime["page_sources"]), 1)
        self.assertIsNone(result.runtime["page_sources"][0]["region"])
        self.assertTrue(result.runtime["visual_review"]["read_failures"])

    def test_failed_model_call_keeps_attempt_unconfirmed(self):
        result = PlanReviewAgent(FakeChat([RuntimeError("synthetic")]), document(), FakeTools(),
                                 page_service=InitialPages()).run()
        self.assertEqual(result.runtime["page_sources"], [])
        visual = result.runtime["visual_review"]
        self.assertEqual(visual["attempts"][0]["status"], "unconfirmed")
        self.assertFalse(visual["images"][0]["sent_to_model"])

    def test_initial_page_and_requested_region_are_distinct_and_citable(self):
        region = [0.2, 0.2, 0.8, 0.8]
        eid = page_image_id("plan:current", 2, region)
        model = FakeChat([action(call("read_document_page", "crop", {
            "source_ref": "plan:current", "page": 2, "region": region})), final([], evidence_ids=[eid])])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=InitialPages()).run()
        self.assertEqual(len(page_image_sources(model.requests[0])), 1)
        self.assertEqual(len(page_image_sources(model.requests[1])), 2)
        sources = result.runtime["page_sources"]
        self.assertEqual(len(sources), 2)
        self.assertEqual({s["evidence_id"] for s in sources}, {"PAGE:plan:current:2", eid})
        self.assertTrue(next(s for s in sources if s["evidence_id"] == eid)["cited"])

    def test_prompt_profiles_remove_shared_text_only_limit(self):
        for output in ("structured", "markdown"):
            for local in (False, True):
                for pages in (False, True):
                    prompt = plan_system_prompt(report_format=output, local_standard_tools_enabled=local,
                        page_service_available=pages, web_provider_available=True, native_tool_names=("web_search",))
                    self.assertNotIn("仍只覆盖可读文字", prompt)
                    self.assertIn("非文字内容尽力而为", prompt)
                    if pages:
                        self.assertIn("可选region=[左,上,右,下]", prompt)

    def test_request_budget_keeps_recent_region_and_replaces_old_pixels_with_pointer(self):
        pages = InitialPages()
        model = FakeChat([final([])])
        PlanReviewAgent(model, document(), FakeTools(), page_service=pages).run()
        image = next(m for m in model.requests[0] if page_image_sources([m]))
        image.additional_kwargs["page_image"]["byte_count"] = REQUEST_IMAGE_BYTES
        newer = image.model_copy(deep=True)
        newer.additional_kwargs["page_image"]["evidence_id"] = "PAGE:plan:current:3"
        newer.additional_kwargs["page_image"]["image_id"] = "PAGE:plan:current:3"
        newer.additional_kwargs["page_image"]["page"] = 3
        bounded, info = bounded_image_messages([image, newer])
        self.assertEqual(len(page_image_sources(bounded)), 1)
        self.assertIn("原页图本次未随请求重发", bounded[0].content)
        self.assertEqual(info["image_bytes"], REQUEST_IMAGE_BYTES)

    def test_structured_evidence_fit_applies_image_limits_before_rejecting_context(self):
        stream = BytesIO()
        Image.new("RGB", (1600, 2400), "white").save(stream, format="PNG")
        images = [page_image_message({"ok": True, "source_ref": "plan:current", "page": page,
            "filename": "x.pdf", "source_kind": "plan", "image_bytes": stream.getvalue(),
            "width": 1600, "height": 2400, "mime_type": "image/png"}) for page in (1, 2, 3)]
        agent = PlanReviewAgent(FakeChat([final([])]), document(), FakeTools(),
                                context_budget_tokens=25000, allow_context_compaction=True)
        state, _, _ = agent._prepare_run("check")
        compose = lambda payload: images + [HumanMessage(content=json.dumps(payload))]
        messages, _ = agent._fit_evidence_messages(state, compose,
            phase="candidate_drafting", initial_max_chars=32000)
        self.assertEqual(len(page_image_sources(messages)), 2)
        self.assertLessEqual(estimate_messages(messages, agent.schema_tokens), 25000)
        self.assertIn("本次未随请求重发", messages[0].content)


if __name__ == "__main__":
    unittest.main()
