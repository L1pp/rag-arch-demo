"""Model schemas, crop provenance and ordinary-QA dispatch; no network calls."""
import asyncio
from copy import deepcopy
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from rag_arch.multimodal import image_source_record, page_image_sources, page_result_messages
from rag_arch.page_qa import PAGE_TOOLS, ainvoke_with_pages, invoke_with_pages
from rag_arch.page_regions import normalize_region, page_image_id
from rag_arch.page_sources import page_evidence_record, page_tools
from tests.test_multimodal import page_result
from tests.test_page_qa import FakeModel, PNG, request
from tests.helpers.visual_fixtures import AsyncOnlyModel


REGION_A = [0.1, 0.2, 0.6, 0.7]
REGION_B = [0.5, 0.5, 1.0, 1.0]


class RegionPages:
    collection_id = "region-test-collection"

    def __init__(self, mismatch=False):
        self.reads = []
        self.mismatch = mismatch

    def sources(self):
        return [{"source_ref": "doc-A", "filename": "方案.docx", "page_count": 3}]

    def list_pages(self, **args):
        return {"ok": True, "pages": [], "source_ref": args["source_ref"]}

    def read_page(self, source_ref, page, **kwargs):
        self.reads.append({"source_ref": source_ref, "page": page, **kwargs})
        region = normalize_region(kwargs.get("region"))
        return {"ok": True, "source_ref": source_ref, "page": page, "filename": "方案.docx",
                "page_count": 3, "image_bytes": PNG, "mime_type": "image/png", "text": "150 mm",
                "width": 1, "height": 1, "region": None if self.mismatch else list(region) if region else None,
                "page_width": 595.0, "page_height": 842.0, "rotation": 0,
                "coordinate_system": "display_page_normalized_top_left", "rendered_scale": 4.0,
                "text_scope": "region" if region else "page", "page_numbering": "system_rendered_page"}


class RegionToolSchemaTests(unittest.TestCase):
    def test_plan_tool_keeps_legacy_required_fields_and_forwards_optional_region(self):
        pages = RegionPages()
        reader = next(tool for tool in page_tools(pages) if tool.name == "read_document_page")
        schema = reader.args_schema.model_json_schema()
        self.assertEqual(schema["required"], ["source_ref", "page"])
        self.assertEqual(set(schema["properties"]), {"source_ref", "page", "region"})
        self.assertFalse(schema["additionalProperties"])
        reader.invoke({"source_ref": "doc-A", "page": 1})
        reader.invoke({"source_ref": "doc-A", "page": 1, "region": REGION_A})
        self.assertEqual(pages.reads[0], {"source_ref": "doc-A", "page": 1})
        self.assertEqual(pages.reads[1]["region"], tuple(REGION_A))
        reader.invoke({"source_ref": "doc-A", "page": 1, "region": None})
        self.assertNotIn("region", pages.reads[-1])

    def test_invalid_region_or_extra_fields_fail_before_calling_the_service(self):
        pages = RegionPages()
        reader = next(tool for tool in page_tools(pages) if tool.name == "read_document_page")
        for args in [
            {"source_ref": "doc-A", "page": True},
            {"source_ref": "doc-A", "page": 1, "scale": 4},
            *({"source_ref": "doc-A", "page": 1, "region": region} for region in (
                [], [0, 0, 1], [False, 0, 1, 1], [0, 0, float("nan"), 1],
                [0, 0, float("inf"), 1], [-0.1, 0, 1, 1], [0, 0, 2, 1],
                [0.5, 0, 0.1, 1], [0, 1, 1, 1], ["0", 0, 1, 1], "0,0,1,1")),
        ]:
            with self.subTest(args=args), self.assertRaises(ValidationError):
                reader.invoke(args)
        self.assertEqual(pages.reads, [])

    def test_ordinary_qa_declares_the_same_optional_parameter_without_scale(self):
        schema = next(tool["function"]["parameters"] for tool in PAGE_TOOLS
                      if tool["function"]["name"] == "read_document_page")
        self.assertEqual(schema["required"], ["source_ref", "page"])
        self.assertEqual(set(schema["properties"]), {"source_ref", "page", "region"})
        self.assertFalse(schema["additionalProperties"])


class RegionProvenanceTests(unittest.TestCase):
    def result(self, region):
        return {**page_result(), "region": region, "page_width": 595.0, "page_height": 842.0,
                "rotation": 90, "coordinate_system": "display_page_normalized_top_left",
                "rendered_scale": 4.0, "text_scope": "region" if region else "page",
                "page_numbering": "system_rendered_page", "source_path": "NEVER_MODEL_PATH"}

    def test_same_page_distinct_regions_keep_distinct_identity_and_geometry(self):
        messages = []
        for number, region in enumerate((None, REGION_A, REGION_B, REGION_A)):
            packet = page_result_messages(self.result(region), f"read-{number}")
            metadata = json.loads(packet[0].content)
            image = packet[1]
            source = image.additional_kwargs["page_image"]
            self.assertEqual(metadata["parent_page_id"], "PAGE:file:example:2")
            self.assertEqual(source["region"], region)
            self.assertEqual(source["rotation"], 90)
            self.assertEqual(source["page_numbering"], "system_rendered_page")
            self.assertEqual(source["rendered_scale"], 4.0)
            self.assertEqual(source["page_width"], 595.0)
            self.assertIn('"coordinate_system"', image.content[0]["text"])
            self.assertNotIn("NEVER_MODEL_PATH", json.dumps(metadata))
            self.assertNotIn("NEVER_MODEL_PATH", image.content[0]["text"])
            self.assertFalse(source["recognition_verified"])
            self.assertFalse(source["sent_to_model"])
            if region:
                self.assertIn("只含所选区域", image.content[0]["text"])
            messages.extend(packet)
        records = page_image_sources(messages, sent=True)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["evidence_id"], "PAGE:file:example:2")
        self.assertEqual(records[1]["evidence_id"], page_image_id("file:example", 2, REGION_A))
        self.assertEqual(records[2]["evidence_id"], page_image_id("file:example", 2, REGION_B))
        self.assertTrue(all(record["sent_to_model"] for record in records))
        self.assertTrue(all(not record["recognition_verified"] for record in records))
        self.assertNotIn("image_bytes", json.dumps(records))

    def test_crop_evidence_keeps_region_and_does_not_claim_whole_page_coverage(self):
        result = self.result(REGION_A)
        result["page_role_info"] = {"role_labels": ["资料性附录"], "usage_note": "只作说明参考。"}
        evidence = page_evidence_record(result)
        self.assertEqual(evidence["source_spans"][0]["region"], REGION_A)
        self.assertEqual(evidence["page_source"]["region"], REGION_A)
        self.assertIn("局部高清图", evidence["text"])
        self.assertIn("只作说明参考", evidence["text"])
        self.assertTrue(any("不能作为整页" in note for note in evidence["notes"]))


class RegionQADispatchTests(unittest.TestCase):
    messages = [HumanMessage(content="核对表中150 mm")]

    def calls(self):
        return request(*(("read_document_page", {"source_ref": "doc-A", "page": 1,
                         **({"region": region} if region is not None else {})})
                         for region in (None, REGION_A, REGION_B, REGION_A)))

    def check_sources(self, result):
        self.assertEqual(len(result.page_sources), 3)
        self.assertEqual([source["display_id"] for source in result.page_sources], ["P1", "P2", "P3"])
        self.assertEqual([source["region"] for source in result.page_sources], [None, REGION_A, REGION_B])
        self.assertTrue(all(source["sent_to_model"] for source in result.page_sources))
        self.assertTrue(all(not source["recognition_verified"] for source in result.page_sources))

    def test_sync_crops_have_independent_citation_ids_and_receipts_precede_images(self):
        pages = RegionPages()
        model = FakeModel(self.calls(), AIMessage(content="表中150 mm。[P2]"))
        result = invoke_with_pages(model, self.messages, pages, budget_tokens=30000)
        self.check_sources(result)
        self.assertNotIn("region", pages.reads[0])
        sequence = model.calls[1][1]
        first_receipt = next(i for i, item in enumerate(sequence) if isinstance(item, ToolMessage))
        self.assertTrue(all(isinstance(item, ToolMessage) for item in sequence[first_receipt:first_receipt + 4]))
        self.assertTrue(all(isinstance(item, HumanMessage) for item in sequence[first_receipt + 4:]))

    def test_async_crops_have_the_same_region_identity(self):
        pages = RegionPages()
        model = AsyncOnlyModel(self.calls(), AIMessage(content="表中150 mm。[P2]"))
        result = asyncio.run(ainvoke_with_pages(model, self.messages, pages, budget_tokens=30000))
        self.check_sources(result)

    def test_mismatched_returned_region_cannot_become_a_citation(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                call = request(("read_document_page", {"source_ref": "doc-A", "page": 1, "region": REGION_A}))
                model = (AsyncOnlyModel if asynchronous else FakeModel)(call, AIMessage(content="区域读取失败。"))
                if asynchronous:
                    result = asyncio.run(ainvoke_with_pages(model, self.messages, RegionPages(True), budget_tokens=30000))
                    sent = model.calls[1]
                else:
                    result = invoke_with_pages(model, self.messages, RegionPages(True), budget_tokens=30000)
                    sent = model.calls[1][1]
                self.assertEqual(result.page_sources, [])
                receipt = next(item for item in sent if isinstance(item, ToolMessage))
                self.assertFalse(json.loads(receipt.content)["ok"])
                self.assertIn("区域与请求不一致", receipt.content)

    def test_invalid_regions_and_unknown_source_never_reach_service(self):
        for args in ({"source_ref": "other-user", "page": 1, "region": REGION_A},
                     {"source_ref": "doc-A", "page": 1, "region": [0, 0, True, 1]},
                     {"source_ref": "doc-A", "page": 1, "region": [0, 0, 0, 1]}):
            with self.subTest(args=args):
                pages = RegionPages()
                model = FakeModel(request(("read_document_page", args)), AIMessage(content="参数不合法。"))
                result = invoke_with_pages(model, self.messages, pages, budget_tokens=30000)
                self.assertEqual(pages.reads, [])
                self.assertEqual(result.page_sources, [])


if __name__ == "__main__":
    unittest.main()
