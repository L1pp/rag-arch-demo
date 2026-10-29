"""Observed visual delivery is not recognition or complete review coverage."""
from copy import deepcopy
import json
import unittest

from langchain_core.messages import HumanMessage, ToolMessage

from rag_arch.multimodal import image_source_record, page_image_message
from rag_arch.plan_context_summary import natural_summary_material, summary_request
from rag_arch.plan_review_report import report_markdown
from rag_arch.visual_review_ledger import VisualReviewLedger
from tests.test_multimodal import page_result
from tests.test_plan_review_report_sources import report_row


REGION = [0.1, 0.2, 0.6, 0.7]


def result(region=None):
    return {**page_result(), "source_ref": "plan:current", "page": 3, "filename": "plan.docx",
            "region": region, "page_width": 595.0, "page_height": 842.0, "rotation": 0,
            "coordinate_system": "display_page_normalized_top_left", "rendered_scale": 3.0,
            "text_scope": "region" if region is not None else "page", "page_numbering": "system_rendered_page"}


class VisualReviewLedgerTests(unittest.TestCase):
    def test_initial_picture_is_a_human_message_without_a_fake_tool_receipt(self):
        message = page_image_message(result(), origin="initial_auto")
        self.assertIsInstance(message, HumanMessage)
        self.assertFalse(isinstance(message, ToolMessage))
        self.assertTrue(message.additional_kwargs["page_image"]["automatic"])
        self.assertFalse(message.additional_kwargs["page_image"]["sent_to_model"])
        self.assertIn('"origin": "initial_auto"', message.content[0]["text"])
        self.assertIsNone(page_image_message({"ok": False, "error": "synthetic"}))

    def test_acquired_attempted_responded_and_cited_are_separate_observations(self):
        ledger = VisualReviewLedger()
        automatic = page_image_message(result(), origin="initial_auto")
        crop = page_image_message(result(REGION))
        full_id = ledger.acquire(automatic.additional_kwargs["page_image"], automatic=True)
        crop_id = ledger.acquire(crop.additional_kwargs["page_image"])
        first = ledger.snapshot()
        self.assertTrue(first["images"][0]["acquired"])
        self.assertFalse(first["images"][0]["request_attempted"])
        attempt = ledger.attempt([automatic], 1)
        pending = ledger.snapshot(cited_ids=[full_id])
        self.assertTrue(pending["images"][0]["request_attempted"])
        self.assertFalse(pending["images"][0]["response_received"])
        self.assertFalse(pending["images"][0]["sent_to_model"])
        self.assertTrue(pending["images"][0]["cited"])
        self.assertFalse(pending["images"][1]["cited"])
        ledger.responded(attempt)
        success = ledger.snapshot(cited_ids=[crop_id])
        self.assertTrue(success["images"][0]["response_received"])
        self.assertTrue(success["images"][0]["sent_to_model"])
        self.assertFalse(success["images"][0]["cited"])
        self.assertFalse(success["images"][1]["request_attempted"])
        self.assertFalse(success["images"][1]["response_received"])
        self.assertTrue(all(not image["recognition_verified"] for image in success["images"]))
        self.assertEqual(success["attempts"][0]["image_ids"], [full_id])

    def test_failed_attempts_remain_unconfirmed_even_after_other_images_succeed(self):
        ledger = VisualReviewLedger()
        first = ledger.attempt([page_image_message(result(), origin="initial_auto")], 1)
        second = ledger.attempt([page_image_message(result(REGION))], 2)
        ledger.responded(second)
        snapshot = ledger.snapshot()
        self.assertEqual([attempt["status"] for attempt in snapshot["attempts"]],
                         ["unconfirmed", "response_received"])
        self.assertFalse(snapshot["images"][0]["sent_to_model"])
        self.assertTrue(snapshot["images"][1]["sent_to_model"])
        self.assertEqual(first, 1)
        self.assertEqual(second, 2)

    def test_same_region_keeps_one_logical_view_and_attempt_geometry_does_not_change(self):
        ledger = VisualReviewLedger()
        original = result(REGION)
        first = page_image_message(original, origin="initial_auto")
        ledger.responded(ledger.attempt([first], 1))
        changed = {**original, "rendered_scale": 5.0, "width": 100}
        ledger.attempt([page_image_message(changed)], 2)
        snapshot = ledger.snapshot()
        self.assertEqual(len(snapshot["images"]), 1)
        self.assertTrue(snapshot["images"][0]["automatic"])
        self.assertEqual(snapshot["attempts"][0]["images"][0]["width"], original["width"])
        self.assertEqual(snapshot["attempts"][0]["images"][0]["rendered_scale"], 3.0)
        self.assertEqual(snapshot["attempts"][1]["images"][0]["width"], 100)
        self.assertEqual(snapshot["images"][0]["width"], original["width"])
        self.assertEqual(snapshot["images"][0]["rendered_scale"], 3.0)
        self.assertEqual(snapshot["images"][0]["latest_acquired_view"]["width"], 100)

    def test_new_sharper_acquisition_or_failed_attempt_keeps_last_successful_geometry(self):
        ledger = VisualReviewLedger()
        low = result(REGION)
        first = ledger.attempt([page_image_message(low)], 1)
        ledger.responded(first)
        high = {**low, "width": 1200, "height": 900, "rendered_scale": 6.0}
        ledger.acquire(page_image_message(high).additional_kwargs["page_image"])
        acquired = ledger.snapshot()["images"][0]
        self.assertTrue(acquired["response_received"])
        self.assertEqual(acquired["width"], low["width"])
        self.assertEqual(acquired["rendered_scale"], low["rendered_scale"])
        self.assertEqual(acquired["latest_acquired_view"]["width"], 1200)
        second = ledger.attempt([page_image_message(high)], 2)
        unconfirmed = ledger.snapshot()
        self.assertFalse(unconfirmed["attempts"][1]["response_received"])
        self.assertEqual(unconfirmed["images"][0]["width"], low["width"])
        self.assertEqual(unconfirmed["images"][0]["delivered_view"]["attempt_id"], first)
        self.assertEqual(unconfirmed["images"][0]["latest_acquired_view"]["rendered_scale"], 6.0)
        ledger.responded(second)
        responded = ledger.snapshot()["images"][0]
        self.assertEqual(responded["width"], 1200)
        self.assertEqual(responded["height"], 900)
        self.assertEqual(responded["rendered_scale"], 6.0)
        self.assertEqual(responded["delivered_view"]["attempt_id"], second)
        # Retrying the callback for an older response cannot downgrade it.
        ledger.responded(first)
        self.assertEqual(ledger.snapshot()["images"][0]["rendered_scale"], 6.0)
        self.assertFalse(responded["recognition_verified"])

    def test_no_payload_model_body_or_credentials_are_retained(self):
        ledger = VisualReviewLedger()
        ledger.initial_selection = {"candidate_count": 9, "selected_pages": [3], "prompt": "PRIVATE_BODY",
                                    "api_key": "PRIVATE_KEY", "image_bytes": b"PRIVATE_IMAGE",
                                    "url": "https://private.invalid/key=PRIVATE_KEY",
                                    "text_preview": "PRIVATE_PREVIEW"}
        source = {**image_source_record(result()), "prompt": "PRIVATE_BODY", "api_key": "PRIVATE_KEY",
                  "image_bytes": b"PRIVATE_IMAGE", "source_path": "PRIVATE_PATH"}
        ledger.acquire(source)
        ledger.attempt([HumanMessage(content="PRIVATE_BODY"), page_image_message(result())], 1)
        ledger.failed_read("plan:current", 4, REGION, "ValueError: PRIVATE_ERROR")
        snapshot = ledger.snapshot()
        serialized = json.dumps(snapshot)
        self.assertNotIn("PRIVATE", serialized)
        self.assertNotIn("base64", serialized)
        self.assertNotIn("image_bytes", serialized)
        self.assertEqual(snapshot["initial_selection"], {"candidate_count": 9, "selected_pages": [3]})
        self.assertEqual(snapshot["read_failures"][0]["error_type"], "ReadError")
        snapshot["images"][0]["filename"] = "modified"
        self.assertNotEqual(ledger.snapshot()["images"][0]["filename"], "modified")

    def test_no_images_means_no_attempt_or_implicit_delivery(self):
        ledger = VisualReviewLedger()
        self.assertIsNone(ledger.attempt([HumanMessage(content="文字")], 0))
        ledger.responded(None)
        self.assertEqual(ledger.snapshot()["attempts"], [])
        self.assertEqual(ledger.snapshot()["images"], [])

    def test_summary_keeps_exact_crop_pointer_without_image_encoding(self):
        picture = page_image_message(result(REGION))
        material = natural_summary_material([picture], question="继续核查关键数字")
        self.assertIn('"region":[0.1,0.2,0.6,0.7]', material)
        self.assertIn('"page_numbering":"system_rendered_page"', material)
        self.assertIn('"coordinate_system":"display_page_normalized_top_left"', material)
        self.assertIn('"rendered_scale":3.0', material)
        self.assertIn("使用保留的region重新读取", material)
        self.assertNotIn("data:image", material)
        legacy = summary_request([picture], question="旧格式")[-1].content
        self.assertIn("region", legacy)
        self.assertNotIn("data:image", legacy)


class VisualReviewReportTests(unittest.TestCase):
    def test_report_shows_auto_regions_pending_attempts_and_failures_without_certifying_recognition(self):
        ledger = VisualReviewLedger()
        automatic = page_image_message(result(), origin="initial_auto")
        crop = page_image_message(result(REGION))
        ledger.attempt([automatic], 1)
        ledger.responded(ledger.attempt([crop], 2))
        ledger.failed_read("plan:current", 4, REGION, "TimeoutError")
        row = report_row()
        crop_id = crop.additional_kwargs["page_image"]["evidence_id"]
        row["runtime"]["visual_review"] = ledger.snapshot(cited_ids=[crop_id])
        row["runtime"]["visual_page_index"] = {"ok": True, "visual_candidate_count": 12}
        original = deepcopy(row)
        output = report_markdown(row)
        self.assertEqual(row, original)
        self.assertIn("图表候选目录：12页", output)
        self.assertIn("首批自动附图：已取得1个页图视图；其中0个随成功返回", output)
        self.assertIn("1次带图请求未确认成功返回", output)
        self.assertIn("已尝试请求，投递未确认", output)
        self.assertIn("已提供给模型（请求成功返回）", output)
        self.assertIn("局部区域[10.00%, 20.00%, 60.00%, 70.00%]", output)
        self.assertIn("系统排版第3页", output)
        self.assertIn("模型明确引用（不代表识别已验证）", output)
        self.assertIn("页图读取未完成", output)
        self.assertIn("TimeoutError", output)
        self.assertIn("图像识别引文未逐字核验", output)


if __name__ == "__main__":
    unittest.main()
