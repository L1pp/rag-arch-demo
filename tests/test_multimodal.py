import base64
from copy import deepcopy
from io import BytesIO
import json
import unittest

from PIL import Image
from langchain_core.messages import HumanMessage, ToolMessage
from rag_arch.multimodal import (decode_image_data_url, image_block_info,
    image_content_block, image_source_record, page_image_sources, page_result_messages)


def png_bytes(width=32, height=48):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def page_result():
    return {"ok": True, "source_ref": "file:example", "filename": "示例.pdf",
            "page": 2, "page_count": 7, "text": "本页文字", "image_bytes": png_bytes(),
            "mime_type": "image/png", "width": 32, "height": 48,
            "source_kind": "standard", "display_id": "P1", "citation": "[P1]"}


class MultimodalTests(unittest.TestCase):
    def test_page_receipt_and_image_have_provenance_without_binary_in_receipt(self):
        result = page_result(); original = deepcopy(result)
        messages = page_result_messages(result, "read-17")
        receipt, image = messages
        self.assertIsInstance(receipt, ToolMessage)
        self.assertEqual(receipt.tool_call_id, "read-17")
        self.assertEqual(receipt.name, "read_document_page")
        body = json.loads(receipt.content)
        self.assertEqual(body["text"], result["text"])
        self.assertEqual(body["evidence_id"], "PAGE:file:example:2")
        self.assertEqual(body["source_kind"], "standard")
        self.assertEqual(body["citation"], "[P1]")
        self.assertNotIn("image_bytes", body)
        self.assertNotIn("base64", receipt.content)
        self.assertIsInstance(image, HumanMessage)
        source = image.additional_kwargs["page_image"]
        self.assertEqual(source["evidence_id"], body["evidence_id"])
        self.assertEqual(source["display_id"], "P1")
        self.assertNotIn("image_bytes", source)
        encoded = image.content[1]["image_url"]["url"]
        self.assertEqual(decode_image_data_url(encoded), (result["image_bytes"], "image/png"))
        self.assertIn("file:example", image.content[0]["text"])
        self.assertEqual(result, original)

    def test_rendering_and_message_preparation_do_not_mark_delivery(self):
        result = page_result(); messages = page_result_messages(result, "read")
        prepared = page_image_sources(messages)
        self.assertFalse(prepared[0]["sent_to_model"])
        sent = page_image_sources(messages + messages, sent=True)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0]["sent_to_model"])
        self.assertFalse(sent[0]["recognition_verified"])
        self.assertFalse(page_image_sources(messages)[0]["sent_to_model"])
        self.assertNotIn("data:", json.dumps(sent))

    def test_failure_or_text_only_page_produces_only_tool_receipt(self):
        for result in ({"ok": False, "error": "page_out_of_range"},
                       {"ok": True, "text": "正文", "page": 1}):
            messages = page_result_messages(result, "read", "read_page")
            self.assertEqual(len(messages), 1)
            self.assertFalse(image_source_record(result, sent=True)["sent_to_model"])

    def test_dimensions_come_from_image_header_without_fetching_a_url(self):
        info = image_block_info(image_content_block(png_bytes(600, 900)))
        self.assertEqual((info["width"], info["height"]), (600, 900))
        remote = image_block_info({"type": "image_url", "image_url": {"url": "https://example.invalid/p.png"}})
        self.assertIsNone(remote["width"])
        standard = image_block_info({"type": "image", "base64": base64.b64encode(png_bytes()).decode(), "mime_type": "image/png"})
        self.assertEqual(standard["width"], 32)


if __name__ == "__main__":
    unittest.main()
