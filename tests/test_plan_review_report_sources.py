"""Export synthetic source provenance without certifying a model's engineering judgment."""
from copy import deepcopy
import unittest

from rag_arch.plan_review_report import report_markdown


def report_row():
    plan_image = {"evidence_id": "PAGE:plan:current:3", "source_ref": "plan:current",
                  "filename": "plan.docx", "page": 3, "page_count": 50,
                  "sent_to_model": True, "recognition_verified": False}
    standard_image = {"evidence_id": "PAGE:standard:1:8", "source_ref": "standard:1",
                      "filename": "standard.pdf", "page": 8, "sent_to_model": True}
    web = {"evidence_id": "WEB:1", "citation": "[W1]", "display_id": "W1",
           "title": "官方发布公告", "url": "https://example.test/notice",
           "snippet": "搜索服务摘要", "verified": False}
    return {"document": {"filename": "plan.docx"}, "verification_date": "2026-09-13",
        "model": "synthetic", "answer": "原模型回答保持不变。",
        "source_coverage": {"provided_units": 1, "total_units": 4, "partially_provided_units": 0},
        "findings": [{"finding_id": "F01", "title": "图表项目", "verdict": "不符合",
            "description": "原模型工程判断。", "citation_check": {"issues": []},
            "plan_evidence": [{"unit_id": plan_image["evidence_id"], "quote": "图上标注内容",
                "locator": {"page": 3}, "evidence_kind": "page_image",
                "page_source": plan_image, "recognition_verified": False}],
            "standard_evidence_ids": [standard_image["evidence_id"], "WEB:1"],
            "conditions": [], "recommendation": "原模型建议。"}],
        "evidence": [
            {"evidence_id": standard_image["evidence_id"], "kind": "page_image",
             "page_source": standard_image, "text": "已取得规范原页图。", "source_spans": []},
            {"evidence_id": "WEB:1", "kind": "web", "text": "搜索服务摘要",
             "source_spans": [{"source_url": web["url"]}]}],
        "runtime": {"page_sources": [plan_image, standard_image], "web_sources": [web]},
        "limitations": [], "validation_notes": [], "model_calls": 2, "tool_calls": 3,
        "status": "answered"}


class PlanReviewReportSourcesTests(unittest.TestCase):
    def test_export_preserves_result_and_distinguishes_image_and_web_claims(self):
        row = report_row()
        original = deepcopy(row)
        output = report_markdown(row)
        self.assertEqual(row, original)
        self.assertIn("模型结论：**不符合**", output)
        self.assertIn("原模型工程判断。", output)
        self.assertIn("完整提供1/4个可读单元", output)
        self.assertIn("图像识别引文（模型识别，未逐字核验）", output)
        self.assertIn("系统排版第3页 / 共50页", output)
        self.assertIn("规范页图依据：PAGE:standard:1:8", output)
        self.assertIn("外部网页参考：WEB:1", output)
        self.assertNotIn("规范依据：WEB:1", output)
        self.assertIn("## 本轮页图来源", output)
        self.assertIn("PAGE:plan:current:3", output)
        self.assertIn("[W1] [官方发布公告](<https://example.test/notice>)", output)
        self.assertIn("未逐字核验网页全文或确认规范现行状态", output)

    def test_runtime_sources_export_even_without_findings_and_do_not_dump_payloads(self):
        row = report_row()
        row["findings"] = []
        row["runtime"]["page_sources"][0].update(
            sent_to_model=False, image_bytes=b"DO_NOT_EXPORT_PIXELS")
        row["runtime"]["web_sources"][0]["internal_debug"] = "DO_NOT_EXPORT_DEBUG"
        output = report_markdown(row)
        self.assertIn("## 本轮页图来源", output)
        self.assertIn("未确认发送给模型", output)
        self.assertIn("## 本轮外部网页参考", output)
        self.assertNotIn("DO_NOT_EXPORT", output)

    def test_legacy_text_only_export_keeps_text_citation_style(self):
        row = report_row()
        row.pop("runtime")
        row["evidence"] = []
        row["findings"][0]["standard_evidence_ids"] = []
        row["findings"][0]["plan_evidence"] = [{
            "unit_id": "DEMO-PLAN/u1", "quote": "已核对的原文", "locator": {"paragraph": 1}}]
        output = report_markdown(row)
        self.assertIn("方案位置：DEMO-PLAN/u1", output)
        self.assertIn("> 已核对的原文", output)
        self.assertNotIn("图像识别引文", output)
        self.assertNotIn("## 本轮页图来源", output)
        self.assertNotIn("## 本轮外部网页参考", output)

    def test_compact_export_keeps_all_findings_and_full_result(self):
        row = report_row()
        row.pop("runtime")
        refs = [{"unit_id": f"DEMO-PLAN/u{i}", "quote": f"引文{i}：" + "正文" * 200,
                 "locator": {"paragraph": i}} for i in range(1, 6)]
        finding = row["findings"][0]
        finding["plan_evidence"] = [refs[0], deepcopy(refs[0]), *refs[1:]]
        finding["standard_evidence_ids"] = ["STD/1", "STD/1", "STD/2", "STD/3", "STD/4", "MISSING/5"]
        finding["citation_check"]["issues"] = ["第5处引文仍待核对"]
        row["evidence"] = [{"evidence_id": f"STD/{i}", "kind": "clause",
                            "text": f"规范正文{i}：" + "条文" * 200, "source_spans": []}
                           for i in range(1, 5)]
        row["findings"] = [dict(deepcopy(finding), finding_id=f"F{i:02d}", title=f"独立问题{i}")
                           for i in range(1, 9)]
        original = deepcopy(row)
        output = report_markdown(row)
        self.assertEqual(row, original)
        self.assertIn("独立问题8", output)  # Presentation is not a seven-finding gate.
        self.assertEqual(output.count("规范正文1："), 1)
        self.assertEqual(output.count("引文1："), 8)  # Same reference is not repeated within a finding.
        self.assertNotIn("引文4：", output)
        self.assertIn("DEMO-PLAN/u4 · 段落4", output)
        self.assertIn("STD/4；MISSING/5（本轮未取得）", output)
        self.assertIn("第5处引文仍待核对", output)
        self.assertIn("…（节选，完整内容见JSON）", output)
        self.assertIn("原模型工程判断。", output)
        self.assertIn("原模型建议。", output)

    def test_compact_related_locations_keep_corrections_and_image_uncertainty(self):
        row = report_row()
        refs = [{"unit_id": f"DEMO-PLAN/u{i}", "quote": f"引文{i}", "locator": {"paragraph": i}}
                for i in range(1, 4)]
        corrected = dict(refs[0], resolution_basis="unique_quote_in_provided_units", model_unit_id="wrong-id")
        image_ref = row["findings"][0]["plan_evidence"][0]
        row["findings"][0]["plan_evidence"] = [*refs, corrected, image_ref]
        output = report_markdown(row)
        extra_locations = output.split("其他方案位置：", 1)[1].split("完整引文见JSON。", 1)[0]
        self.assertIn("原模型位置 wrong-id 未直接命中", extra_locations)
        self.assertIn("图像识别，未逐字核验", extra_locations)
        self.assertIn("系统排版第3页", extra_locations)

    def test_unstructured_readable_response_is_not_shortened(self):
        row = report_row()
        row["findings"] = []
        row["stop_reason"] = "invalid_result"
        row["answer"] = "模型原始可读回复" * 100
        row["raw_final_text"] = row["answer"]
        output = report_markdown(row)
        self.assertIn(row["answer"], output)
        self.assertIn("保留原始可读文本", output)


if __name__ == "__main__":
    unittest.main()
