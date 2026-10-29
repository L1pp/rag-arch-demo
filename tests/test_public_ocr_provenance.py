"""Fictional OCR text keeps its unverified status through QA and plan citations."""
import asyncio
from copy import deepcopy
import json
import unittest

from langchain_core.messages import AIMessage

from rag_arch.model_input_view import model_plan_payload
from rag_arch.models import Chunk, LoadedDocument, TextUnit
from rag_arch.plan_documents import PlanDocument, load_uploaded_document
from rag_arch.plan_review_agent import validate_findings, validate_plan_quotes
from rag_arch.qa import aanswer_question, answer_question


def fictional_document():
    locator = {"page": 1, "text_source": "cloud_ocr", "recognition_verified": False,
               "ocr_model": "synthetic-ocr", "ocr_status": "completed",
               "partial_page": True, "page_numbering": "system_rendered_page"}
    return LoadedDocument("fixture-doc", "fictional.docx", "docx",
        [TextUnit("原生示例文字。", {"page": 1}), TextUnit("虚构跨度6.5m。", locator)], [], [2], 3,
        [{"page": 1, "status": "completed", "model": "synthetic-ocr", "recognition_verified": False},
         {"page": 2, "status": "failed", "recognition_verified": False}])


class QAStub:
    max_tokens = 256
    native_tool_names = ()

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content="虚构跨度6.5m [S1]")

    async def ainvoke(self, messages):
        return self.invoke(messages)


class OCRProvenanceTests(unittest.TestCase):
    def test_plan_quote_stays_unverified_after_successful_text_location(self):
        loaded = fictional_document()
        plan = load_uploaded_document("fictional.docx", b"fictional-bytes", document_id="DEMO",
                                      loaded_document=loaded)
        self.assertEqual(plan.metadata()["ocr_pages"], loaded.ocr_pages)
        self.assertIsNot(plan.ocr_pages, loaded.ocr_pages)
        unit = plan.read(max_chars=2000)["units"][1]
        projected = model_plan_payload(plan.read(max_chars=2000))
        source = projected["ocr_sources"][0]
        self.assertEqual(source["source_ref"], unit["source_ref"])
        self.assertFalse(source["recognition_verified"])
        self.assertIn("回看原页", projected["ocr_note"])
        uid = unit["unit_id"]
        span = plan.quote_span(uid, "跨度6.5m")
        self.assertFalse(span["recognition_verified"])
        citations, issues = validate_plan_quotes([{"unit_id": uid, "quote": "跨度6.5m"}], plan,
                                                 {item["unit_id"] for item in plan.units})
        self.assertTrue(citations[0]["location_verified"])
        self.assertFalse(citations[0]["recognition_verified"])
        self.assertIn("OCR", "".join(issues))
        findings, _ = validate_findings([{"title": "核对虚构跨度", "category": "document_consistency",
            "verdict": "不符合", "plan_evidence": [{"unit_id": uid, "quote": "跨度6.5m"}]}],
            plan, {item["unit_id"] for item in plan.units}, {}, set())
        self.assertEqual(findings[0]["citation_check"]["status"], "needs_review")
        self.assertFalse(findings[0]["plan_evidence"][0]["recognition_verified"])

    def test_legacy_document_without_ocr_metadata_remains_compatible(self):
        document = PlanDocument("fictional", "fictional.txt",
                                [{"unit_id": "fictional/u1", "text": "自编原文", "locator": {"paragraph": 1}}])
        self.assertNotIn("ocr_pages", document.metadata())
        self.assertEqual(document.quote_span("fictional/u1", "原文"),
                         {"start": 2, "end": 4, "quote": "原文", "whitespace_normalized": False})
        restored = PlanDocument.from_bundle({"document_id": "fictional", "filename": "fictional.txt", "units": document.units})
        self.assertEqual(restored.ocr_pages, [])

    def test_sync_and_async_qa_keep_ocr_location_and_docx_page_kind(self):
        loaded = fictional_document()
        locator = deepcopy(loaded.units[1].locator)
        locator["custom_original_locator"] = "自编定位"
        chunk = Chunk(loaded.doc_id, loaded.filename, "ocr", loaded.units[1].text,
                      locator, "docx", "fictional-collection")
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                model = QAStub()
                options = {"index": None, "chat_model": model, "direct_chunks": [chunk],
                           "context_budget_tokens": 8192, "output_budget_tokens": 256}
                if asynchronous:
                    asyncio.run(aanswer_question("虚构跨度是多少？", **options))
                else:
                    answer_question("虚构跨度是多少？", **options)
                source = next(message.content for message in model.calls[0]
                              if "资料正文开始" in str(message.content))
                line = next(line for line in source.splitlines() if line.startswith("[S1] "))
                metadata = json.loads(line[len("[S1] "):])
                self.assertEqual(metadata["source_kind"], "docx")
                self.assertIn("页 1", metadata["位置"])
                details = json.loads(metadata["位置"].partition("；")[2])
                self.assertEqual(details["text_source"], "cloud_ocr")
                self.assertFalse(details["recognition_verified"])
                self.assertEqual(details["page_numbering"], "system_rendered_page")
                self.assertEqual(details["custom_original_locator"], "自编定位")
                self.assertEqual(chunk.locator, locator)


if __name__ == "__main__":
    unittest.main()
