from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo.runtime import (
    DEFAULT_PLAN, PUBLIC_MATERIAL_NOTICE, build_runtime, make_plan_document,
    render_plan_report, result_to_dict, run_offline_demo,
    run_plan_review, run_standard_question,
)


class PublicDemoRuntimeTests(unittest.TestCase):
    def test_offline_agents_retrieve_and_validate_real_source_references(self):
        runtime = build_runtime()
        document = make_plan_document()
        self.assertEqual(document.source_kind, "txt")
        self.assertEqual(document.units[0]["unit_id"], "DEMO-PLAN-001/u1")
        self.assertEqual(document.units[0]["locator"]["line_start"], 1)
        question = run_standard_question(runtime, "支撑间距的上限是多少？")
        self.assertTrue(question.grounded)
        self.assertEqual(len(question.citations), 1)
        self.assertEqual(question.citations[0]["evidence_id"], "DEMO-STRUCT-2026/4.1.1")
        self.assertGreaterEqual(question.tool_calls, 1)
        self.assertIsNotNone(runtime.index.embeddings)
        self.assertIsNotNone(runtime.index.reranker)

        review = run_plan_review(runtime, make_plan_document())
        row = result_to_dict(review)
        self.assertGreaterEqual(review.tool_calls, 2)
        self.assertIn("DEMO-STRUCT-2026/4.1.1", row["evidence_ids"])
        self.assertTrue(row["findings"])
        finding = row["findings"][0]
        self.assertEqual(finding["standard_evidence_ids"], ["DEMO-STRUCT-2026/4.1.1"])
        self.assertEqual(finding["citation_check"]["status"], "locations_verified")
        report = render_plan_report(review)
        self.assertIn("ScriptedChat", report)
        self.assertIn("DEMO-STRUCT-2026/4.1.1", report)
        self.assertIn(PUBLIC_MATERIAL_NOTICE, report)
        self.assertIn("模型调用0次", report)

    def test_editing_sample_plan_changes_the_rendered_report(self):
        runtime = build_runtime()
        original = run_plan_review(runtime, make_plan_document())
        plan_data = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
        changed_text = plan_data["units"][0]["text"].replace("1.50 m", "1.65 m")
        changed = run_plan_review(runtime, make_plan_document(text=changed_text))
        original_report = render_plan_report(original)
        changed_report = render_plan_report(changed)
        self.assertNotEqual(original_report, changed_report)
        self.assertIn("1.65 m", changed_report)

    def test_missing_standard_evidence_does_not_create_citation(self):
        empty_data = {
            "standard": {"std_id": "DEMO-EMPTY", "std_no": "DEMO-EMPTY", "title": "空白虚构演示集"},
            "clauses": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text(json.dumps(empty_data, ensure_ascii=False), encoding="utf-8")
            runtime = build_runtime(standards_path=path)
            qa = run_standard_question(runtime, "查询旋翼校准的虚构安全参数。")
            self.assertFalse(qa.grounded)
            self.assertEqual(qa.citations, [])
            self.assertEqual(qa.citations, [])

            review = run_plan_review(runtime, make_plan_document())
            row = result_to_dict(review)
            self.assertEqual(row["evidence_ids"], [])
            self.assertTrue(row["findings"])
            self.assertEqual(row["findings"][0]["standard_evidence_ids"], [])
            self.assertNotIn("DEMO-STRUCT-2026/4.1.1", render_plan_report(review))

    def test_default_offline_path_never_constructs_remote_providers(self):
        with patch("rag_arch.providers.create_embeddings", side_effect=AssertionError("offline called cloud embedding")) as embeddings, \
             patch("rag_arch.providers.create_reranker", side_effect=AssertionError("offline called cloud reranker")) as reranker, \
             patch("rag_arch.tool_chat.create_tool_chat_model", side_effect=AssertionError("offline called cloud chat")) as chat:
            runtime = build_runtime()
            answer = run_standard_question(runtime, "支撑间距的上限是多少？")
            review = run_plan_review(runtime)
        self.assertTrue(answer.grounded)
        self.assertTrue(result_to_dict(review)["findings"])
        embeddings.assert_not_called()
        reranker.assert_not_called()
        chat.assert_not_called()

    def test_cli_output_bundle_contains_report_and_machine_readable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run_offline_demo(directory)
            output = Path(directory)
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "standard_answer.json").is_file())
            self.assertTrue((output / "plan_review.json").is_file())
            self.assertIn("离线脚本演示", (output / "report.md").read_text(encoding="utf-8"))
            result = json.loads((output / "plan_review.json").read_text(encoding="utf-8"))
            self.assertEqual(result["execution_mode"], "offline")
            self.assertEqual(result["network_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()