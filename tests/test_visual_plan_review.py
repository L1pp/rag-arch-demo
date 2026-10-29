"""Candidate navigation must never stand in for actual page-image delivery."""
import unittest

from langchain_core.messages import AIMessage

from rag_arch.plan_review_agent import DEFAULT_REVIEW_TASK, PlanReviewAgent, plan_system_prompt
from rag_arch.tool_chat import create_tool_chat_model
from tests.helpers.visual_fixtures import FakeChat, FakeTools, action, call
from tests.helpers.visual_fixtures import Pages
from tests.helpers.visual_fixtures import response_settings as responses_settings
from tests.helpers.visual_fixtures import document


class CandidatePages(Pages):
    def __init__(self, fail=False):
        self.lists = []
        self.reads = []
        self.fail = fail

    def sources(self):
        return [{"source_ref": "plan:current", "filename": "drawing.pdf", "page_count": 4}]

    def list_pages(self, source_ref, start=1, limit=20, query="", visual_only=False):
        self.lists.append((source_ref, start, limit, query, visual_only))
        if self.fail:
            raise ValueError("synthetic conversion failure")
        return {"ok": True, "source_ref": source_ref, "page_count": 4,
                "visual_candidate_count": 2, "next_start": None,
                "pages": [{"page": p, "text_preview": "图示线索", "visual_signals": ["vector_or_table"]}
                          for p in (2, 4)]}

    def read_page(self, source_ref, page, **kwargs):
        self.reads.append(page)
        return {**super().read_page(source_ref=source_ref, page=page, **kwargs),
                "page": page, "page_count": 4, "source_kind": "plan"}


class VisualPlanReviewTests(unittest.TestCase):
    def test_review_scope_applies_to_every_tool_profile_and_output_format(self):
        for local in (False, True):
            for output in ("markdown", "structured"):
                for available in (False, True):
                    with self.subTest(local=local, output=output, pages=available):
                        prompt = plan_system_prompt(local_standard_tools_enabled=local,
                            report_format=output, page_service_available=available, mode="review")
                        self.assertIn("非文字内容，本轮应主动关注", prompt)
                        self.assertIn("非文字内容尽力而为", prompt)
                        self.assertIn("优先核对影响结论的关键数字、尺寸、单位", prompt)
                        self.assertIn("不要求穷尽全部图表、复杂构造或公式复算", prompt)
                        self.assertIn("表头及行列/合并单元格对应", prompt)
                        self.assertIn("未读取、读取失败、辨认不清", prompt)
                        self.assertNotIn("文字辅助核查", prompt)
                        self.assertNotIn("不能称已完成图纸审查", prompt)
                        if available:
                            self.assertIn("next_start非空表示目录还有后续", prompt)
                            self.assertIn("不要求逐页穷尽全部候选图表", prompt)
                            self.assertIn("即使不在候选目录中", prompt)
                        else:
                            self.assertIn("当前未提供原页读图工具", prompt)
                            self.assertNotIn("用read_document_page读取", prompt)

    def test_question_profiles_keep_visual_reading_within_the_current_question(self):
        for local in (False, True):
            for output in ("markdown", "structured"):
                for available in (False, True):
                    with self.subTest(local=local, output=output, pages=available):
                        prompt = plan_system_prompt(local_standard_tools_enabled=local,
                            report_format=output, page_service_available=available, mode="question")
                        self.assertNotIn("非文字内容，本轮应主动关注", prompt)
                        self.assertNotIn("next_start非空表示目录还有后续", prompt)
                        if available:
                            self.assertIn("无须为局部追问重新检查全案", prompt)

    def test_directory_alone_never_renders_or_counts_as_image_delivery(self):
        pages = CandidatePages()
        model = FakeChat([AIMessage(content="当前文字可支持以下判断。")])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=pages,
                                 report_format="markdown").run()
        self.assertEqual(pages.reads, [])
        self.assertEqual(pages.lists, [("plan:current", 1, 30, "", True)])
        self.assertEqual(result.source_coverage["visual_candidate_page_count"], 2)
        self.assertEqual(result.runtime["page_sources"], [])
        self.assertEqual(result.source_coverage["non_text_review"], "not_performed")
        first = "\n".join(str(m.content) for m in model.requests[0])
        self.assertIn("visual_pages", first)
        self.assertNotIn("data:image", first)
        self.assertNotIn("完整方案核查仍只覆盖可读文字", first)
        self.assertIn(DEFAULT_REVIEW_TASK, first)
        self.assertNotIn("不要求重算复杂公式或全面审图", first)
        guidance = result.traces[0]["execution_progress"]["next_step_guidance"]
        self.assertIn("本轮同时核查文字与图表等非文字内容", guidance)
        self.assertTrue(any("未做图像复核" in item for item in result.limitations))
        self.assertEqual(result.runtime["prompt_version"], "whole-plan-natural-text-and-visual-v4")

    def test_model_selects_one_page_and_can_read_outside_candidate_directory(self):
        for selected_page in (2, 3):
            with self.subTest(page=selected_page):
                pages = CandidatePages()
                model = FakeChat([action(call("read_document_page", "image", {
                    "source_ref": "plan:current", "page": selected_page})),
                    AIMessage(content=f"已参考方案第{selected_page}页图示。")])
                result = PlanReviewAgent(model, document(), FakeTools(), page_service=pages,
                                         report_format="markdown").run()
                self.assertEqual(result.stop_reason, "answered")
                self.assertEqual(pages.reads, [selected_page])
                self.assertFalse(any(isinstance(m.content, list) for m in model.requests[0]))
                self.assertTrue(any(isinstance(m.content, list) for m in model.requests[1]))
                self.assertEqual([p["page"] for p in result.runtime["page_sources"]], [selected_page])
                self.assertEqual(result.source_coverage["non_text_review"],
                                 "on_demand_images_provided_not_full_review")
                self.assertTrue(any("页图投递记录不证明" in item for item in result.limitations))

    def test_unavailable_visual_index_does_not_abort_text_review(self):
        model = FakeChat([AIMessage(content="完成可读文字的审阅。")])
        result = PlanReviewAgent(model, document(), FakeTools(), page_service=CandidatePages(fail=True),
                                 report_format="markdown").run()
        self.assertEqual(result.stop_reason, "answered")
        self.assertFalse(result.source_coverage["visual_page_index_available"])
        self.assertTrue(any("目录暂不可用" in item for item in result.limitations))

    def test_bailian_responses_exposes_its_input_window_reserve(self):
        model = create_tool_chat_model(responses_settings(), output_method="text")
        agent = PlanReviewAgent(model, document(), FakeTools(), context_budget_tokens=1000000,
                                 report_format="markdown")
        self.assertEqual(agent.runtime["provider_input_context_ratio"], 0.8)
        self.assertEqual(agent.input_budget_tokens, 800000)


if __name__ == "__main__":
    unittest.main()
