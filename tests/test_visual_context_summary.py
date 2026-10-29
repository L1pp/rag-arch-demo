"""Offline checks for stripping page images from rolling-summary inputs."""
import base64
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_arch.multimodal import image_content_block
from rag_arch.plan_context_summary import natural_summary_material, summary_request
from rag_arch.plan_review_context import estimate_messages


class VisualContextSummaryTests(unittest.TestCase):
    def page_message(self):
        image = image_content_block(b"synthetic-page-image", "image/png")
        source = {
            "evidence_id": "PAGE:file:demo:4", "source_ref": "file:demo", "filename": "demo.pdf",
            "page": 4, "page_count": 8, "display_id": "P4", "citation": "[P4]",
        }
        message = HumanMessage(content=[
            {"type": "text", "text": "工具返回的文档页面图像（资料内容）：[P4]"},
            image,
        ], additional_kwargs={"page_image": source})
        return message, image, source

    def test_natural_material_keeps_sources_and_conclusions_without_image_encoding(self):
        page, image, source = self.page_message()
        material = natural_summary_material([
            page,
            AIMessage(content="已经根据页面图像确认梁底存在待回查标记。"),
            ToolMessage(content="工具仍可按页回读", tool_call_id="read-4"),
        ], question="继续核对梁底条件")

        encoded = image["image_url"]["url"]
        payload = base64.b64encode(b"synthetic-page-image").decode("ascii")
        self.assertIsInstance(material, str)
        self.assertNotIn("data:image", material)
        self.assertNotIn(encoded, material)
        self.assertNotIn(payload, material)
        self.assertNotIn("image_bytes", material)
        self.assertIn(source["source_ref"], material)
        self.assertIn(str(source["page"]), material)
        self.assertIn(source["evidence_id"], material)
        self.assertIn("[P4]", material)
        self.assertIn("已经根据页面图像确认", material)
        self.assertIn("工具调用ID：read-4", material)
        self.assertIn("重读原页", material)

    def test_legacy_json_envelope_stays_textual_and_image_free(self):
        page, image, source = self.page_message()
        request = summary_request([page], question="旧任务")
        payload = json.loads(request[-1].content)
        encoded = image["image_url"]["url"]

        self.assertEqual(payload["task"], "旧任务")
        self.assertEqual(payload["older_messages"][0]["role"], "human")
        self.assertIsInstance(payload["older_messages"][0]["content"], str)
        serialized = request[-1].content
        self.assertNotIn("data:image", serialized)
        self.assertNotIn(encoded, serialized)
        self.assertIn(source["source_ref"], serialized)
        self.assertIn(str(source["page"]), serialized)
        self.assertIn(source["evidence_id"], serialized)

    def test_plain_text_summary_keeps_existing_shape_and_visual_estimator_remains_available(self):
        plain = summary_request([HumanMessage(content="旧对话")], question="旧任务")
        self.assertEqual(json.loads(plain[-1].content)["older_messages"][0]["content"], "旧对话")

        page, image, _ = self.page_message()
        direct_visual_cost = estimate_messages([page])
        text_cost = estimate_messages([HumanMessage(content="工具返回的文档页面图像（资料内容）：[P4]")])
        self.assertGreater(direct_visual_cost, text_cost)
        compacted = summary_request([page], question="压缩", natural_notes=True)
        self.assertIsInstance(compacted[-1].content, str)
        self.assertNotIn("data:image", compacted[-1].content)


if __name__ == "__main__":
    unittest.main()
