"""Synthetic checks for tenant-scoped Web audit files and debug downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage

from rag_arch.config import Settings
from rag_arch.runtime_audit import (
    debug_result_json, record_event, record_run, tenant_audit_metadata,
)
from rag_arch.web.chat_runner import run_chat
from rag_arch.web.standards_runner import run_standards
from rag_arch.tool_chat import create_tool_chat_model
from tests.helpers.tenant_audit_fixtures import (
    FakeChat, final, corpus_fixture, FakeStatus, FakeModel, reply,
)


class TenantAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="rag-web-tenant-audit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit_patch = patch("rag_arch.runtime_audit.AUDIT_ROOT", self.root)
        self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.settings = Settings(chat_model="synthetic-model",
            chat_base_url="https://example.invalid/v1", chat_api_key="synthetic-private-key",
            chat_max_tokens=256, chat_context_budget_tokens=100000)

    def _events(self, result):
        info = result["audit_log"]
        directory = self.root / "users" / str(info["user_id"]) / info["workspace_id"] / info["run_id"]
        return [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_chat_runners_isolate_concurrent_model_audits_by_server_identity(self):
        barrier = Barrier(2)

        def worker(identity):
            user_id, workspace_id, marker = identity
            def mark_inside_model():
                barrier.wait(timeout=10)
                record_event("synthetic_tenant_marker", marker=marker)
            request = {"user_id": user_id, "workspace_id": workspace_id,
                       "task_id": f"chat-{marker}", "assistant_id": "chat", "prompt": marker}
            model = FakeModel(reply(marker), action=mark_inside_model)
            result = run_chat(request, lambda *_: None, Event(), settings=self.settings,
                              model_factory=lambda _: model)
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            a, b = list(executor.map(worker, [(31, "workspace-a", "USER_A_ONLY"),
                                               (72, "workspace-b", "USER_B_ONLY")]))

        self.assertNotEqual(a["audit_log"]["run_id"], b["audit_log"]["run_id"])
        for result, user_id, workspace_id, marker in (
                (a, 31, "workspace-a", "USER_A_ONLY"),
                (b, 72, "workspace-b", "USER_B_ONLY")):
            info = result["audit_log"]
            self.assertEqual((info["user_id"], info["workspace_id"], info["task_id"]),
                             (user_id, workspace_id, f"chat-{marker}"))
            self.assertNotIn("directory", info)
            rows = self._events(result)
            started = next(row for row in rows if row["kind"] == "run_started")
            self.assertEqual((started["user_id"], started["workspace_id"], started["task_id"]),
                             (user_id, workspace_id, f"chat-{marker}"))
            markers = [row["marker"] for row in rows if row["kind"] == "synthetic_tenant_marker"]
            self.assertEqual(markers, [marker])
            serialized = debug_result_json(result, user_id=user_id, workspace_id=workspace_id)
            self.assertNotIn(str(self.root), serialized)

        with self.assertRaises(ValueError):
            debug_result_json(a, user_id=72, workspace_id="workspace-b")
        with self.assertRaises(ValueError):
            debug_result_json(a, user_id=31, workspace_id="workspace-b")
        with self.assertRaises(ValueError):
            debug_result_json(a)
        missing_run = dict(a["audit_log"], run_id="20261019_000000_abcdef123456")
        with self.assertRaises(FileNotFoundError):
            debug_result_json({"audit_log": missing_run}, user_id=31,
                              workspace_id="workspace-a", root=self.root)

    def test_standards_runner_uses_the_same_server_tenant_metadata(self):
        from rag_arch.catalog_index import CatalogIndex
        from rag_arch.evidence_index import EvidenceIndex

        corpus = corpus_fixture()
        resources = type("Resources", (), {
            "get": lambda _self: type("Bundle", (), {
                "index": EvidenceIndex(corpus), "registry": FakeStatus(),
                "catalog": CatalogIndex(corpus.catalog), "pages": None,
            })(),
            "citation": lambda _self, record: {
                "evidence_id": record["evidence_id"], "std_no": record["std_no"],
                "title": "合成规范", "clause_no": record.get("clause_no"),
                "text": record.get("text", ""), "page_url": None, "history_url": None,
            },
            "catalog_entry": lambda _self, record: {
                "catalog_id": record.get("catalog_id", "SYNTH"), "std_no": record.get("std_no"),
                "title": record.get("title", "合成规范"), "page_url": None, "history_url": None,
            },
        })()
        request = {"user_id": 88, "workspace_id": "workspace-standards",
                   "task_id": "standards-synthetic", "assistant_id": "standards",
                   "prompt": "合成规范问题", "attempt": 2}
        fake_model = FakeChat([final(ids=[], status="insufficient_evidence", answer="资料不足。")])
        with patch("rag_arch.web.standards_runner.create_web_search_provider", return_value=None):
            result = run_standards(request, lambda *_: None, Event(), resources=resources,
                settings=self.settings,
                model_factory=lambda *_args, **_kwargs: fake_model)

        info = result["audit_log"]
        self.assertEqual((info["user_id"], info["workspace_id"], info["task_id"]),
                         (88, "workspace-standards", "standards-synthetic"))
        self.assertEqual(info["attempt"], 2)
        started = next(row for row in self._events(result) if row["kind"] == "run_started")
        self.assertEqual(started["scope"], "web_standard_agent")
        self.assertEqual(started["user_id"], 88)
        self.assertEqual(started["workspace_id"], "workspace-standards")
        self.assertEqual(started["attempt"], 2)

    def test_model_http_requests_and_responses_stay_in_their_tenant_audit(self):
        barrier = Barrier(2)

        def handle(request):
            body = json.loads(request.content)
            marker = body["messages"][-1]["content"]
            barrier.wait(timeout=10)
            return httpx.Response(200, json={
                "id": f"response-{marker}", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": marker}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            def worker(identity):
                user_id, workspace_id, marker = identity
                request = {"user_id": user_id, "workspace_id": workspace_id,
                           "task_id": f"http-{marker}"}
                with record_run("web_http_synthetic", self.settings, marker,
                                **tenant_audit_metadata(request)) as audit:
                    model = create_tool_chat_model(self.settings, http_client=client)
                    answer = model.invoke([HumanMessage(content=marker)])
                    audit.set_result({"text": answer.content, "audit_log": audit.info})
                rows = self._events({"audit_log": audit.info})
                return marker, audit.info, rows

            with ThreadPoolExecutor(max_workers=2) as executor:
                runs = list(executor.map(worker, [(14, "workspace-http-a", "HTTP_A_ONLY"),
                                                  (28, "workspace-http-b", "HTTP_B_ONLY")]))

        for marker, info, rows in runs:
            requests = [row for row in rows if row["kind"] == "model_http_request"]
            responses = [row for row in rows if row["kind"] == "model_http_response"]
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(responses), 1)
            self.assertEqual(requests[0]["body"]["messages"][-1]["content"], marker)
            self.assertEqual(responses[0]["body"]["choices"][0]["message"]["content"], marker)
            serialized = json.dumps(rows, ensure_ascii=False)
            other_marker = "HTTP_B_ONLY" if marker == "HTTP_A_ONLY" else "HTTP_A_ONLY"
            self.assertNotIn(other_marker, serialized)
            self.assertNotIn("directory", info)

    def test_tenant_metadata_fails_closed_and_legacy_cli_download_stays_compatible(self):
        with self.assertRaises(ValueError):
            tenant_audit_metadata({"user_id": 9, "task_id": "task"})
        with self.assertRaises(ValueError):
            tenant_audit_metadata({"user_id": "../9", "workspace_id": "ws", "task_id": "task"})
        with self.assertRaises(ValueError):
            tenant_audit_metadata({"user_id": 9, "workspace_id": "ws", "task_id": "task",
                                   "attempt": "../2"})

        with record_run("legacy-cli", self.settings, "本地问题", root=self.root,
                        task_id="legacy-task") as audit:
            record_event("legacy_marker", value="local-only")
            audit.set_result({"audit_log": audit.info, "answer": "合成答案"})
        self.assertTrue(Path(audit.info["directory"]).is_relative_to(self.root))
        exported = json.loads(debug_result_json({"audit_log": audit.info, "answer": "合成答案"}))
        self.assertIn("legacy_marker", [event["kind"] for event in exported["call_audit"]["events"]])
        with self.assertRaises(ValueError):
            debug_result_json({"audit_log": audit.info}, user_id=9, workspace_id="ws")


if __name__ == "__main__":
    unittest.main()
