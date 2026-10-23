"""Step-two HTTP isolation against a disposable real PostgreSQL schema."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_arch.config import Settings
from rag_arch.runtime_audit import record_run, tenant_audit_metadata
from rag_arch.web.app import create_app, event_stream
from rag_arch.web.auth import AuthStore, PostgresAuthStore
from rag_arch.web.db import Database
from rag_arch.web.task_service import TaskCapacityError
from rag_arch.web.tenancy import TenantRegistry, TenantCapacityError


class TenantHTTPTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")
        if not self.url:
            self.skipTest("set RAG_ARCH_TEST_DATABASE_URL to a dedicated test instance")
        self.temp = TemporaryDirectory(prefix="rag-tenant-http-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = Database(self.url, schema="tenant_http_" + uuid4().hex)
        self.addCleanup(self._drop_schema)
        self.database.migrate()
        self.auth = PostgresAuthStore(self.database, password_iterations=1500)
        self.auth.create_user("Alice", "alice-password-1")
        self.auth.create_user("Bob", "bob-password-1")
        self.requests = []
        self.request_lock = Lock()
        self.release = Event()
        self.addCleanup(self.release.set)

        def runner(request, progress, cancel):
            with self.request_lock:
                self.requests.append(request)
            progress("test", "controlled test runner")
            if request["prompt"].startswith("hold"):
                while not self.release.wait(0.01):
                    if cancel.is_set():
                        return {"text": "cancelled work"}
            text = f"user={request['user_id']};prompt={request['prompt']}"
            if request["prompt"] == "audit":
                with record_run("tenant_http_test", Settings(), request["prompt"],
                                root=self.root / "data/runtime/conversations",
                                **tenant_audit_metadata(request)) as audit:
                    audit.set_result({"text": text, "owner": request["user_id"]})
                return {"text": text, "audit_log": audit.info}
            return {"text": text}

        self.app = create_app(root=self.root, database=self.database, auth_store=self.auth,
                              auth_cookie_secure=False, task_runner=runner,
                              task_backend="memory",
                              execution_modes={key: "model" for key in ("chat", "standards", "plan")})
        self.alice = TestClient(self.app)
        self.alice.__enter__()
        self.addCleanup(self.alice.__exit__, None, None, None)
        self.bob = TestClient(self.app)
        self.addCleanup(self.bob.close)
        for client, name, password in ((self.alice, "Alice", "alice-password-1"),
                                       (self.bob, "Bob", "bob-password-1")):
            response = client.post("/login", data={"username": name, "password": password},
                                   headers={"Origin": "http://testserver"}, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
        self.a = self.alice.get("/api/workspace").json()
        self.b = self.bob.get("/api/workspace").json()

    def _drop_schema(self):
        import psycopg
        from psycopg import sql
        with psycopg.connect(self.url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(self.database.schema)))

    def upload(self, client, workspace, name="same.txt", text="private text"):
        response = client.post("/api/uploads", params={"workspace_id": workspace["workspace_id"],
                               "filename": name}, content=text.encode(),
                               headers={"Origin": "http://testserver"})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def submit(self, client, workspace, assistant="chat", prompt="hello", request_id=None, **extra):
        return client.post("/api/tasks", json={"workspace_id": workspace["workspace_id"],
            "assistant_id": assistant, "request_id": request_id or uuid4().hex,
            "prompt": prompt, **extra}, headers={"Origin": "http://testserver"})

    def finished(self, client, workspace, task_id):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            response = client.get(f"/api/tasks/{task_id}", params={"workspace_id": workspace["workspace_id"]})
            self.assertEqual(response.status_code, 200)
            task = response.json()
            if task["status"] in {"completed", "failed", "cancelled"}:
                return task
            time.sleep(0.01)
        self.fail("controlled task did not terminate")

    def test_private_uploads_list_download_and_page_routes_reject_other_user(self):
        self.assertNotEqual(self.a["workspace_id"], self.b["workspace_id"])
        self.assertNotEqual(self.a["owner_user_id"], self.b["owner_user_id"])
        a = self.upload(self.alice, self.a, text="ALICE_PRIVATE")
        b = self.upload(self.bob, self.b, text="BOB_PRIVATE")
        a_list = self.alice.get("/api/uploads", params={"workspace_id": self.a["workspace_id"]}).json()
        self.assertEqual([row["asset_id"] for row in a_list["uploads"]], [a["asset_id"]])
        self.assertEqual(self.alice.get(f"/api/uploads/{a['asset_id']}/download",
            params={"workspace_id": self.a["workspace_id"]}).text, "ALICE_PRIVATE")
        for workspace in (self.a, self.b):
            self.assertEqual(self.bob.get(f"/api/uploads/{a['asset_id']}/download",
                params={"workspace_id": workspace["workspace_id"]}).status_code, 404)
            for endpoint in ("pages", "page-image"):
                self.assertEqual(self.bob.get(f"/api/documents/{endpoint}", params={
                    "asset": a["asset_id"], "workspace_id": workspace["workspace_id"],
                    "source_ref": "forged", "page": 1}).status_code, 404)
        wrong = self.submit(self.bob, self.b, attachment_ids=[a["asset_id"]])
        self.assertEqual(wrong.status_code, 404)
        own = self.submit(self.bob, self.b, attachment_ids=[b["asset_id"]])
        self.assertEqual(own.status_code, 202)

    def test_task_history_cancel_reset_download_and_events_are_user_scoped(self):
        first = self.submit(self.alice, self.a, prompt="ALICE_HISTORY").json()
        self.assertEqual(self.finished(self.alice, self.a, first["task_id"])["status"], "completed")
        other = self.submit(self.bob, self.b, prompt="BOB_HISTORY").json()
        self.finished(self.bob, self.b, other["task_id"])
        follow = self.submit(self.alice, self.a, prompt="ALICE_FOLLOW").json()
        self.finished(self.alice, self.a, follow["task_id"])
        request = next(row for row in self.requests if row["prompt"] == "ALICE_FOLLOW")
        self.assertEqual([turn["question"] for turn in request["history"]], ["ALICE_HISTORY"])
        for endpoint in (f"/api/tasks/{first['task_id']}", f"/api/tasks/{first['task_id']}/download"):
            self.assertEqual(self.bob.get(endpoint, params={"workspace_id": self.b["workspace_id"],
                                                         "format": "json"}).status_code, 404)
        self.assertEqual(self.bob.post(f"/api/tasks/{first['task_id']}/cancel", json={
            "workspace_id": self.b["workspace_id"]}, headers={"Origin": "http://testserver"}).status_code, 404)
        self.assertEqual(self.bob.post("/api/assistants/chat/reset", json={
            "workspace_id": self.a["workspace_id"]}, headers={"Origin": "http://testserver"}).status_code, 404)
        self.assertEqual(self.bob.get("/api/events", params={
            "workspace_id": self.a["workspace_id"]}).status_code, 404)
        a_context = self.app.state.tenants.get(self.a["owner_user_id"])
        b_context = self.app.state.tenants.get(self.b["owner_user_id"])
        serialized = json.dumps(b_context.tasks.events.read_after(0)[0])
        self.assertNotIn("ALICE_HISTORY", serialized)
        self.assertNotIn(first["task_id"], serialized)

        async def first_event():
            stream = event_stream(a_context.tasks, 0)
            try:
                return await stream.__anext__()
            finally:
                await stream.aclose()
        item = asyncio.run(first_event())
        self.assertEqual(item.data["task_id"], first["task_id"])
        self.assertNotIn("BOB_HISTORY", self.alice.get("/api/workspace").text)

    def test_two_users_have_three_slots_each_and_share_one_bounded_executor(self):
        assets = {"Alice": self.upload(self.alice, self.a), "Bob": self.upload(self.bob, self.b)}
        created = []
        for name, client, workspace in (("Alice", self.alice, self.a), ("Bob", self.bob, self.b)):
            for assistant in ("chat", "standards", "plan"):
                fields = {"attachment_ids": [assets[name]["asset_id"]]} if assistant == "plan" else {}
                response = self.submit(client, workspace, assistant, "hold-" + name,
                                       request_id="same-id", **fields)
                self.assertEqual(response.status_code, 202, response.text)
                created.append((client, workspace, response.json()))
        deadline = time.monotonic() + 4
        while len(self.requests) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.requests), 6)
        for client, workspace, task in created:
            fields = {"attachment_ids": task["attachment_ids"]} if task["assistant_id"] == "plan" else {}
            self.assertEqual(self.submit(client, workspace, task["assistant_id"], "new", **fields).status_code, 409)
        first_context = self.app.state.tenants.get(self.a["owner_user_id"])
        second_context = self.app.state.tenants.get(self.b["owner_user_id"])
        self.assertIs(first_context.tasks._executor, second_context.tasks._executor)
        self.assertIsNot(first_context.tasks.events, second_context.tasks.events)
        self.assertIsNot(first_context.documents, second_context.documents)
        self.release.set()
        for client, workspace, task in created:
            self.assertEqual(self.finished(client, workspace, task["task_id"])["status"], "completed")

    def test_debug_download_requires_task_and_audit_owner(self):
        task = self.submit(self.alice, self.a, prompt="audit").json()
        finished = self.finished(self.alice, self.a, task["task_id"])
        self.assertEqual(finished["status"], "completed", finished.get("error"))
        audit = finished["result"]["audit_log"]
        self.assertNotIn("directory", audit)
        response = self.alice.get(f"/api/tasks/{task['task_id']}/download", params={
            "workspace_id": self.a["workspace_id"], "format": "debug"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["owner"], self.a["owner_user_id"])
        self.assertEqual(self.bob.get(f"/api/tasks/{task['task_id']}/download", params={
            "workspace_id": self.b["workspace_id"], "format": "debug"}).status_code, 404)
        plain = self.submit(self.alice, self.a, prompt="without audit").json()
        self.finished(self.alice, self.a, plain["task_id"])
        self.assertEqual(self.alice.get(f"/api/tasks/{plain['task_id']}/download", params={
            "workspace_id": self.a["workspace_id"], "format": "debug"}).status_code, 404)

    def test_shared_executor_and_registry_have_explicit_capacity(self):
        release = Event()
        lock = Lock()
        counter = {"running": 0, "peak": 0, "entered": 0}

        def controlled(request, progress, cancel):
            with lock:
                counter["running"] += 1
                counter["entered"] += 1
                counter["peak"] = max(counter["peak"], counter["running"])
            try:
                while not release.wait(0.01):
                    if cancel.is_set():
                        break
                return {"text": "controlled"}
            finally:
                with lock:
                    counter["running"] -= 1

        registry = TenantRegistry(self.database, root=self.root, max_workers=2,
                                  max_users=2, max_tasks_per_user=2,
                                  runner_factory=lambda _: controlled)
        try:
            alice = registry.get(self.a["owner_user_id"])
            bob = registry.get(self.b["owner_user_id"])
            tasks = [(context, context.tasks.submit(assistant, assistant, "hold"))
                     for context in (alice, bob) for assistant in ("chat", "standards")]
            deadline = time.monotonic() + 3
            while counter["entered"] < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(counter["entered"], 2)
            self.assertEqual(sum(task["status"] == "queued" for _, task in tasks[2:]), 2)
            release.set()
            deadline = time.monotonic() + 3
            while any(context.tasks.get_task(task["task_id"])["status"] not in
                      {"completed", "cancelled", "failed"} for context, task in tasks):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertLessEqual(counter["peak"], 2)
            self.assertEqual(counter["entered"], 4)
            with self.assertRaises(TaskCapacityError):
                alice.tasks.submit("chat", "overflow", "new")
            # An already accepted request stays idempotent even at capacity.
            self.assertEqual(alice.tasks.submit("chat", "chat", "hold")["task_id"], tasks[0][1]["task_id"])
        finally:
            release.set()
            registry.close()
        bounded = TenantRegistry(self.database, root=self.root, max_users=1)
        try:
            bounded.get(self.a["owner_user_id"])
            with self.assertRaises(TenantCapacityError):
                bounded.get(self.b["owner_user_id"])
        finally:
            bounded.close()


class TenantStartupTests(unittest.TestCase):
    def test_authenticated_runtime_without_database_fails_before_loading_models(self):
        with TemporaryDirectory(prefix="rag-tenant-config-") as directory:
            auth = AuthStore(Path(directory) / "auth.sqlite3", password_iterations=1500)
            app = create_app(root=directory, auth_store=auth, task_runner=lambda *_: {"text": "unused"})
            with self.assertRaisesRegex(RuntimeError, "RAG_ARCH_DATABASE_URL"):
                with TestClient(app):
                    pass


if __name__ == "__main__":
    unittest.main()
