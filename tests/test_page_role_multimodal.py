import json
import unittest

from rag_arch.multimodal import image_source_record, page_result_messages


class PageRoleMetadataTests(unittest.TestCase):
    def test_page_roles_are_forwarded_to_model_receipt(self):
        role = {"kind": "informative_appendix", "title": "附录A", "use_as_normative_requirement": False}
        result = {"ok": True, "source_ref": "standard:1", "page": 24,
                  "source_kind": "standard", "text": "保留原页文字", "page_role_info": role,
                  "message": "本页是资料性附录，不能直接当作规范性要求。"}
        payload = json.loads(page_result_messages(result, "read-1")[0].content)
        self.assertEqual(payload["page_role_info"], role)
        self.assertEqual(payload["text"], result["text"])
        self.assertEqual(payload["message"], result["message"])

    def test_lightweight_image_reference_retains_role_without_claiming_delivery(self):
        role = {"kind": "commentary", "use_as_normative_requirement": False}
        result = {"ok": True, "source_ref": "standard:2", "page": 28,
                  "source_kind": "standard", "image_bytes": b"test-placeholder", "page_role_info": role,
                  "message": "条文说明仅用于解释。"}
        source = image_source_record(result)
        self.assertEqual(source["page_role_info"], role)
        self.assertEqual(source["message"], result["message"])
        self.assertFalse(source["sent_to_model"])
        self.assertFalse(source["recognition_verified"])

    def test_upload_without_page_roles_does_not_acquire_standard_role(self):
        result = {"ok": True, "source_ref": "upload:1", "page": 1,
                  "source_kind": "upload", "text": "上传资料"}
        payload = json.loads(page_result_messages(result, "read-2")[0].content)
        self.assertNotIn("page_role_info", payload)
        self.assertEqual(payload["source_kind"], "upload")


if __name__ == "__main__":
    unittest.main()
