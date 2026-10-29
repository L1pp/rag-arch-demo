"""Web OCR provenance and failures use only generated in-memory samples."""
import asyncio
from concurrent.futures import CancelledError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch

from rag_arch.config import Settings
from rag_arch.models import Answer, Chunk, LoadedDocument, SearchHit, TextUnit
from rag_arch.plan_documents import PlanDocument
from rag_arch.web.chat_runner import _file_fields, _ocr_system_prompt, arun_chat, run_chat
from rag_arch.web.ocr_preparation import OCRDocumentUnreadable
from rag_arch.web.plan_runner import _public_result, arun_plan, run_plan
from tests.test_web_task_worker import FakeRepository, _claim, _runtime


def document(filename="example.pdf", kind="pdf"):
    locator = {"page": 1, "text_source": "cloud_ocr", "recognition_verified": False,
               "ocr_model": "synthetic-ocr", "ocr_status": "completed", "partial_page": True}
    if kind in {"doc", "docx"}:
        locator["page_numbering"] = "system_rendered_page"
    return LoadedDocument("doc-id", filename, kind,
        [TextUnit("合成原生文字。", {"page": 1}), TextUnit("合成跨度 6.5m。", locator)],
        [], [2], 3, [{"page": 1, "status": "completed", "model": "synthetic-ocr", "recognition_verified": False},
                     {"page": 2, "status": "failed", "recognition_verified": False}])


class WebOCRProvenanceTests(unittest.TestCase):
    def test_word_ocr_page_and_native_paragraph_remain_distinct(self):
        doc = document("example.docx", "docx")
        ocr = Chunk("doc-id", doc.filename, "ocr", "合成跨度 6.5m", doc.units[1].locator, "docx", "c")
        native = Chunk("doc-id", doc.filename, "native", "合成原生段落", {"paragraph": 3, "page": 2}, "docx", "c")
        prepared = SimpleNamespace(collection=SimpleNamespace(documents=[doc], chunks=[ocr, native], uses_embeddings=False),
            assets=[{"asset_id": "asset", "filename": doc.filename, "size_bytes": 9}], workspace_id="workspace",
            document_assets={"doc-id": {"asset_id": "asset"}}, pages=SimpleNamespace(source_context={}))
        answer = Answer("答复", [SearchHit(ocr, 1.0), SearchHit(native, 0.9)], [], True,
                        citation_check={"cited_numbers": [1, 2]})
        result = _file_fields(prepared, answer, "direct")
        self.assertEqual(parse_qs(urlsplit(result["citations"][0]["page_url"]).query)["page"], ["1"])
        self.assertNotIn("page", parse_qs(urlsplit(result["citations"][1]["page_url"]).query))
        self.assertFalse(result["citations"][0]["recognition_verified"])
        self.assertIn("未经原页", "".join(result["warnings"]))
        self.assertIn("系统排版", "".join(result["warnings"]))
        self.assertEqual(result["files"][0]["ocr_pages"], doc.ocr_pages)
        self.assertIn("未识别或未读取", _ocr_system_prompt(prepared))
        doc.ocr_pages = []
        self.assertIsNotNone(_ocr_system_prompt(prepared))
        doc.units = [doc.units[0]]
        self.assertIsNone(_ocr_system_prompt(prepared))

    def test_plan_warning_and_unreadable_state_survive_public_boundary(self):
        doc = PlanDocument("example", "example.pdf", [{"unit_id": "example/u1", "text": "6.5m", "locator": {"page": 1}}],
                           source_kind="pdf", ocr_pages=[{"page": 1, "status": "completed"}])
        output = SimpleNamespace(runtime={}, report_format=None, document={}, answer="答复", findings=[],
            status="answered", source_coverage={}, reviewed_topics=[], limitations=[], validation_notes=[],
            stop_reason="answered", evidence_ids=[], evidence=[], model="model", token_usage={}, model_calls=1,
            tool_calls=0, as_history_turn=lambda: {}, to_dict=lambda: {})
        result = _public_result(output, "review", SimpleNamespace(chat_model="model"),
            SimpleNamespace(info={}, set_result=lambda _: None), {"attachment_ids": ["asset"]},
            "报告", "context", SimpleNamespace(), doc, {}, SimpleNamespace(workspace_id="w"))
        self.assertIn("未经原页", "".join(result["warnings"]))
        empty = PlanDocument("example", "example.pdf", [], source_kind="pdf",
                             ocr_pages=[{"page": 2, "status": "unknown"}])
        store = SimpleNamespace(get_plan=lambda **kwargs: SimpleNamespace(document=empty, asset={}, context_id="c"))
        with self.assertRaises(OCRDocumentUnreadable) as failure:
            run_plan({"plan_action": "review", "attachment_ids": ["asset"]}, lambda *_: None, Event(),
                     resources=SimpleNamespace(plan_index=lambda: None), documents=store, settings=Settings())
        self.assertIn("第 2 页 OCR 状态未确认", failure.exception.as_error()["message"])

    def test_chat_sends_ocr_guidance_and_keeps_citation_warning(self):
        doc = document()
        chunk = Chunk(doc.doc_id, doc.filename, "ocr", doc.units[1].text, doc.units[1].locator, "pdf", "collection")
        prepared = SimpleNamespace(collection=SimpleNamespace(documents=[doc], chunks=[chunk], mode="direct",
            uses_embeddings=False, text_bytes=20, text_tokens=8),
            assets=[{"asset_id": "asset", "filename": doc.filename, "size_bytes": 9}],
            workspace_id="workspace", document_assets={doc.doc_id: {"asset_id": "asset"}},
            pages=SimpleNamespace(source_context={}))
        documents = SimpleNamespace(prepare_chat=lambda *args, **kwargs: prepared)
        captured = {}
        def answer(question, **kwargs):
            captured.update(kwargs)
            return Answer("需要核对 [S1]", [SearchHit(chunk, 1.0)], [], True, citation_check={"cited_numbers": [1]})
        with TemporaryDirectory() as temporary, \
             patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(temporary)), \
             patch("rag_arch.web.chat_runner.create_web_search_provider", return_value=None), \
             patch("rag_arch.web.chat_runner.answer_question", side_effect=answer):
            result = run_chat({"task_id": "ocr-chat", "prompt": "合成问题", "attachment_ids": ["asset"]},
                              lambda *_: None, Event(), settings=Settings(chat_model="fake"),
                              model_factory=lambda _: SimpleNamespace(), documents=documents)
        self.assertIn("OCR文字可能错写", captured["system_prompt"])
        self.assertFalse(result["citations"][0]["recognition_verified"])
        self.assertIn("未经原页", "".join(result["warnings"]))

    def test_worker_does_not_expose_provider_error_for_empty_ocr_plan(self):
        settings = Settings(chat_model="fake", chat_base_url="https://example.invalid/v1", chat_api_key="synthetic-key")
        resources = SimpleNamespace(plan_index=lambda: None, aprepare=lambda: None, aplan_index=lambda: None)
        sensitive = "synthetic-provider-private-detail"
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), TemporaryDirectory() as temporary:
                doc = PlanDocument("example", "example.pdf", [], source_kind="pdf", unread_pages=[2],
                    ocr_pages=[{"page": 2, "status": "unknown", "reason": sensitive}])
                store = SimpleNamespace(get_plan=lambda **kwargs: SimpleNamespace(document=doc, asset={}, context_id="c"))
                repository = FakeRepository([_claim("plan-empty", assistant_id="plan",
                                                    attachment_ids=["asset"], plan_action="review")])
                def runner(payload, progress, cancelled):
                    call = arun_plan if asynchronous else run_plan
                    return call(payload, progress, cancelled, settings=settings, documents=store, resources=resources)
                with patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(temporary) / "audits"):
                    _runtime(repository, runner=runner, async_execution=asynchronous, root=temporary).run(max_tasks=1)
                error = json.loads(repository.finishes[0][4])
                self.assertEqual(error["code"], "ocr_reading_unavailable")
                self.assertIn("第 2 页 OCR 状态未确认", error["message"])
                self.assertNotIn(sensitive, json.dumps(error, ensure_ascii=False))

    def test_cancellation_takes_priority_and_closes_async_resources(self):
        class Client:
            closed = False
            async def aclose(self):
                self.closed = True
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), TemporaryDirectory() as temporary:
                cancelled = Event()
                client = Client()
                provider = SimpleNamespace(batch_size=8, async_client=SimpleNamespace(_client=client))
                class CancelDuringRead:
                    def prepare_chat(self, *args, **kwargs):
                        cancelled.set()
                        raise OCRDocumentUnreadable([{"page": 2, "status": "failed"}])
                    async def aprepare_chat(self, *args, **kwargs):
                        kwargs["embeddings_factory"]()
                        cancelled.set()
                        raise OCRDocumentUnreadable([{"page": 2, "status": "failed"}])
                request = {"task_id": "cancel-ocr", "prompt": "合成问题", "attachment_ids": ["asset"]}
                with patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(temporary)), \
                     patch("rag_arch.web.chat_runner.create_embeddings", return_value=provider):
                    with self.assertRaises(CancelledError):
                        if asynchronous:
                            asyncio.run(arun_chat(request, lambda *_: None, cancelled,
                                                 settings=Settings(chat_model="fake"), documents=CancelDuringRead()))
                        else:
                            run_chat(request, lambda *_: None, cancelled,
                                     settings=Settings(chat_model="fake"), documents=CancelDuringRead())
                audit = json.loads(next(Path(temporary).rglob("result.json")).read_text(encoding="utf-8"))
                self.assertEqual(audit["status"], "cancelled")
                self.assertNotIn("error", audit)
                if asynchronous:
                    self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
