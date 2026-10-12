from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo.runtime import (
    DEFAULT_PLAN, _chat, build_runtime, make_plan_document, plan_document_text,
    run_plan_review,
)
from rag_arch.config import DEFAULT_PLAN_CONTEXT_BUDGET, Settings
from rag_arch.tool_chat import create_tool_chat_model


class DemoRuntimeRegressionTests(unittest.TestCase):
    def test_editor_text_matches_cli_and_reparses_all_lines_without_old_unit_mapping(self):
        plan_data = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
        editor_text = plan_document_text(plan_data)
        cli_document = make_plan_document(plan_path=DEFAULT_PLAN)

        self.assertEqual(editor_text, "\n".join(unit["text"].rstrip("\n") for unit in cli_document.units))
        self.assertEqual(len(cli_document.units), 2)

        edited_text = "第一段编辑后内容。\n第二段完整保留，定位到第二行。"
        edited_document = make_plan_document(plan_data=plan_data, text=edited_text)

        self.assertEqual([unit["text"].rstrip("\n") for unit in edited_document.units],
                         ["第一段编辑后内容。", "第二段完整保留，定位到第二行。"])
        self.assertEqual([unit["locator"]["line_start"] for unit in edited_document.units], [1, 2])
        self.assertEqual([unit["unit_id"] for unit in edited_document.units],
                         ["DEMO-PLAN-001/u0001", "DEMO-PLAN-001/u0002"])
        self.assertTrue(all("demo_input_locator" not in unit["locator"]
                            for unit in edited_document.units))

    def test_live_workflows_use_separate_chat_budgets_and_plan_context_budget(self):
        settings = Settings(
            chat_model="demo-compatible-chat",
            chat_base_url="https://example.invalid/v1",
            chat_api_key="placeholder",
            chat_api_protocol="chat_completions",
            timeout_seconds=7,
            chat_timeout_seconds=7,
            chat_max_tokens=1234,
            chat_thinking_budget=256,
            chat_context_budget_tokens=32000,
            plan_timeout_seconds=19,
            plan_max_tokens=4321,
            plan_thinking_budget=512,
            plan_context_budget_tokens=16000,
        )
        runtime = build_runtime("offline")
        runtime.mode = "live"
        runtime.settings = settings

        observed = {}

        class CapturePlanReview:
            def __init__(self, model, document, tools, **kwargs):
                observed["model"] = model
                observed["context_budget_tokens"] = kwargs["context_budget_tokens"]

            def run(self, _request):
                return SimpleNamespace(runtime={})

        def capture_client(**kwargs):
            return kwargs

        with patch("rag_arch.tool_chat.create_tool_chat_model", wraps=create_tool_chat_model) as factory, \
             patch("rag_arch.tool_chat.WebSearchChat", side_effect=capture_client), \
             patch("rag_arch.tool_chat.attach_chat_audit", side_effect=lambda client: client), \
             patch("demo.runtime.PlanReviewAgent", CapturePlanReview):
            plan_client = _chat(runtime, "plan")
            chat_client = _chat(runtime, "qa")
            run_plan_review(runtime, make_plan_document())

        self.assertEqual(factory.call_args_list[0].args[0].chat_context_budget_tokens, 16000)
        self.assertEqual(factory.call_args_list[1].args[0].chat_context_budget_tokens, 32000)
        self.assertEqual(plan_client["max_tokens"], 4321)
        self.assertEqual(plan_client["timeout"], 19)
        self.assertEqual(plan_client["extra_body"]["thinking_budget"], 512)
        self.assertEqual(chat_client["max_tokens"], 1234)
        self.assertEqual(chat_client["timeout"], 7)
        self.assertEqual(chat_client["extra_body"]["thinking_budget"], 256)
        self.assertIsNot(plan_client, chat_client)
        self.assertEqual(observed["model"], plan_client)
        self.assertEqual(observed["context_budget_tokens"], 16000)
        self.assertEqual(runtime.settings.chat_max_tokens, 1234)
        self.assertEqual(runtime.settings.chat_timeout_seconds, 7)

    def test_offline_plan_review_uses_default_context_without_live_settings(self):
        runtime = build_runtime("offline")
        self.assertIsNone(runtime.settings)

        observed = {}

        class CapturePlanReview:
            def __init__(self, *_args, **kwargs):
                observed["context_budget_tokens"] = kwargs["context_budget_tokens"]

            def run(self, _request):
                return SimpleNamespace(runtime={})

        with patch("demo.runtime.PlanReviewAgent", CapturePlanReview):
            run_plan_review(runtime, make_plan_document())

        self.assertEqual(observed["context_budget_tokens"], DEFAULT_PLAN_CONTEXT_BUDGET)


if __name__ == "__main__":
    unittest.main()
