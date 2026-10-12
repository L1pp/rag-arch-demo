from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from demo.runtime import DEFAULT_PLAN, ScriptedChat, _read_json, build_runtime, plan_document_text


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class PublicDemoUITests(unittest.TestCase):
    def test_plan_editor_loads_all_sample_units(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        expected = plan_document_text(_read_json(DEFAULT_PLAN))

        self.assertEqual(len(app.text_area), 1)
        self.assertEqual(app.text_area[0].value, expected)
        self.assertIn("支撑间距记录为 1.50 m", app.text_area[0].value)
        self.assertIn("检查日期栏暂未填写", app.text_area[0].value)

    def test_rejected_citation_is_prominently_marked_as_unverified(self):
        original_invoke = ScriptedChat.invoke

        def invalid_citation(self, messages, **kwargs):
            if self.workflow == "qa" and any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content=json.dumps({
                    "status": "answered",
                    "answer": "这个答案带有不存在的来源。 [MISSING-SOURCE]",
                    "evidence_ids": ["MISSING-SOURCE"],
                }, ensure_ascii=False))
            return original_invoke(self, messages, **kwargs)

        app = AppTest.from_file(str(APP_PATH)).run()
        with patch.object(ScriptedChat, "invoke", invalid_citation):
            app.button(key="run_qa").click().run()

        payload = app.session_state["qa_payload"]
        self.assertEqual(payload["termination_reason"], "citation_rejected")
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["citations"], [])
        self.assertTrue(any("引用未通过核验" in item.value for item in app.error))
        self.assertTrue(any("未通过引用核验的模型原文" in item.value for item in app.caption))
        self.assertTrue(any("MISSING-SOURCE" in item.value for item in app.text))

    def test_invalid_model_format_is_not_mislabeled_as_missing_evidence(self):
        original_invoke = ScriptedChat.invoke

        def invalid_format(self, messages, **kwargs):
            if self.workflow == "qa" and any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content="这是一条格式不合规的模型输出。")
            return original_invoke(self, messages, **kwargs)

        app = AppTest.from_file(str(APP_PATH)).run()
        with patch.object(ScriptedChat, "invoke", invalid_format):
            app.button(key="run_qa").click().run()

        payload = app.session_state["qa_payload"]
        self.assertEqual(payload["termination_reason"], "invalid_result")
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertTrue(any("输出格式或引用核验未完成" in item.value for item in app.warning))
        self.assertFalse(any("资料不足" in item.value for item in app.warning))
        self.assertTrue(any("未通过核验的模型原文" in item.value for item in app.caption))

    def test_valid_offline_answer_keeps_citation_without_warning(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        app.button(key="run_qa").click().run()

        payload = app.session_state["qa_payload"]
        self.assertEqual(payload["termination_reason"], "answered")
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["citations"][0]["evidence_id"], "DEMO-STRUCT-2026/4.1.1")
        self.assertFalse(app.error)
        self.assertFalse(app.warning)
        self.assertTrue(any("程序记录的引用" in item.value for item in app.markdown))
        self.assertTrue(any("DEMO-STRUCT-2026/4.1.1" in item.value for item in app.code))
        self.assertTrue(any(item.label == "下载问答 JSON" for item in app.get("download_button")))

    def test_insufficient_evidence_gets_a_clear_explanation(self):
        empty_data = {
            "standard": {"std_id": "DEMO-EMPTY", "std_no": "DEMO-EMPTY", "title": "空白虚构演示集"},
            "clauses": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            standards_path = Path(directory) / "empty.json"
            standards_path.write_text(json.dumps(empty_data, ensure_ascii=False), encoding="utf-8")
            app = AppTest.from_file(str(APP_PATH))
            with patch(
                "demo.runtime.build_runtime",
                side_effect=lambda mode: build_runtime(mode, standards_path=standards_path),
            ):
                app.run()
                app.button(key="run_qa").click().run()

        payload = app.session_state["qa_payload"]
        self.assertEqual(payload["termination_reason"], "insufficient_evidence")
        self.assertEqual(payload["citations"], [])
        self.assertTrue(any("资料不足" in item.value for item in app.warning))
        self.assertTrue(any("尚未构成经资料支持的结论" in item.value for item in app.caption))
        self.assertTrue(any("没有取得可引用的完整证据" in item.value for item in app.text))


if __name__ == "__main__":
    unittest.main()
