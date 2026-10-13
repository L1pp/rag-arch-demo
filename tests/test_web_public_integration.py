from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_arch.config import Settings
from rag_arch.web.plan_runner import _plan_settings
from rag_arch.web.standards_runner import _build_result
from rag_arch.web.task_service import TaskService


class _Closable:
    def close(self):
        pass


class _Documents(_Closable):
    pdf_backend = None


class _Resources(_Closable):
    def citation(self, record):
        return record

    def catalog_entry(self, record):
        return record


class _Audit:
    info = {"run_id": "test-only"}

    def set_result(self, result):
        self.result = result


_IMPORT_AUTH_DIRECTORY = None


def _load_create_app(directory: Path):
    global _IMPORT_AUTH_DIRECTORY
    module_name = "rag_arch.web.app"
    if module_name in __import__("sys").modules:
        return importlib.import_module(module_name).create_app
    # Keep the module-level AuthStore database temporary and available through process exit.
    _IMPORT_AUTH_DIRECTORY = tempfile.TemporaryDirectory(prefix="rag-arch-web-import-auth-")
    auth_path = Path(_IMPORT_AUTH_DIRECTORY.name) / "import-only.sqlite3"
    with patch.dict(os.environ, {"RAG_ARCH_AUTH_DB": str(auth_path)}):
        return importlib.import_module(module_name).create_app


def _client(directory: Path, runner):
    create_app = _load_create_app(directory)
    service = TaskService(runner=runner, archive_root=None,
                          execution_modes={"chat": "model"}, model_name="injected-test")
    app = create_app(service=service, resources=_Resources(), documents=_Documents(),
                     root=directory, frontend_dir=directory / "frontend", auth_enabled=False)
    return TestClient(app)


def _wait_for_terminal_task(client: TestClient, task_id: str, workspace_id: str):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/tasks/{task_id}", params={"workspace_id": workspace_id})
        if response.status_code != 200:
            raise AssertionError(response.text)
        task = response.json()
        if task["status"] in {"completed", "failed", "cancelled"}:
            return task
        time.sleep(0.01)
    raise AssertionError("The injected task did not reach a terminal state.")


class WebPublicIntegrationTests(unittest.TestCase):
    def test_plan_settings_use_plan_context_budget(self):
        settings = Settings(chat_max_tokens=1234, chat_context_budget_tokens=64000,
                            plan_max_tokens=4321, plan_timeout_seconds=19,
                            plan_thinking_budget=17, plan_context_budget_tokens=16000)
        configured = _plan_settings(settings)
        self.assertEqual(configured.chat_max_tokens, 4321)
        self.assertEqual(configured.chat_timeout_seconds, 19)
        self.assertEqual(configured.chat_thinking_budget, 17)
        self.assertEqual(configured.chat_context_budget_tokens, 16000)

    def test_workspace_and_task_api_preserve_qa_validation_result(self):
        resources = _Resources()
        audit = _Audit()
        agent_result = SimpleNamespace(
            text="模型给出的原文",
            status="insufficient_evidence",
            termination_reason="citation_rejected",
            answer_scope="evidence",
            evidence=[],
            citations=[],
            catalog_matches=[],
            warnings=[],
            web_sources=[],
            native_tool_usage={},
            token_usage={},
            tool_calls=1,
            model_calls=1,
            corpus_id="DEMO-ONLY",
            grounded=False,
        )

        def runner(request, progress, cancel_event):
            return _build_result(agent_result, resources, Settings(chat_model="test-only"),
                                 audit, request)

        with tempfile.TemporaryDirectory(prefix="rag-arch-web-test-") as temporary:
            directory = Path(temporary)
            with _client(directory, runner) as client:
                workspace_response = client.get("/api/workspace")
                self.assertEqual(workspace_response.status_code, 200)
                workspace_id = workspace_response.json()["workspace_id"]
                submitted = client.post("/api/tasks", json={
                    "workspace_id": workspace_id,
                    "assistant_id": "chat",
                    "request_id": "citation-state-test",
                    "prompt": "测试引用状态透传",
                })
                self.assertEqual(submitted.status_code, 202, submitted.text)
                accepted = submitted.json()
                task = _wait_for_terminal_task(client, accepted["task_id"], workspace_id)
                self.assertEqual(task["status"], "completed")
                self.assertEqual(task["result"]["status"], "insufficient_evidence")
                self.assertEqual(task["result"]["termination_reason"], "citation_rejected")
                self.assertTrue(any("引用未通过核验" in warning
                                    for warning in task["result"]["warnings"]))
                exported = client.get(f"/api/tasks/{accepted['task_id']}/download", params={
                    "workspace_id": workspace_id, "format": "md"})
                self.assertEqual(exported.status_code, 200)
                self.assertIn("引用未通过核验", exported.text)
                self.assertIn("模型原文", exported.text)
                self.assertIn("任务状态：completed", exported.text)


    def test_task_failure_status_is_returned_by_workspace_api(self):
        def runner(request, progress, cancel_event):
            raise RuntimeError("injected test failure")

        with tempfile.TemporaryDirectory(prefix="rag-arch-web-failure-") as temporary:
            directory = Path(temporary)
            with _client(directory, runner) as client:
                workspace_id = client.get("/api/workspace").json()["workspace_id"]
                submitted = client.post("/api/tasks", json={
                    "workspace_id": workspace_id,
                    "assistant_id": "chat",
                    "request_id": "failure-state-test",
                    "prompt": "测试失败状态",
                })
                self.assertEqual(submitted.status_code, 202, submitted.text)
                task = _wait_for_terminal_task(
                    client, submitted.json()["task_id"], workspace_id)
                self.assertEqual(task["status"], "failed")
                self.assertIn("injected test failure", task["error"])


if __name__ == "__main__":
    unittest.main()