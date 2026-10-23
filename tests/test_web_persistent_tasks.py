from __future__ import annotations

import asyncio
import os
import unittest
from uuid import uuid4

from rag_arch.web.db import Database
from rag_arch.web.task_repository import IdempotencyConflict, TaskRepository

from rag_arch.web.persistent_tasks import (
    DatabaseEventReader, DatabaseTaskService, PersistentTenantRegistry,
)


class FakeDatabase:
    def __init__(self):
        self.workspaces = {}
        self.ensure_calls = []

    def ensure_workspace(self, user_id):
        self.ensure_calls.append(user_id)
        return self.workspaces.setdefault(user_id, f"workspace-{user_id}")


class FakeDocuments:
    def __init__(self, root, workspace_id, pdf_backend, *, user_id, database):
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.closed = False

    def close(self):
        self.closed = True


class FakeRepository:
    def __init__(self):
        self.tasks = {}
        self.events = {}
        self.calls = []
        self.cancel_calls = []
        self.snapshot_calls = []

    def submit(self, user_id, assistant_id, request_id, inputs):
        self.calls.append(("submit", user_id, assistant_id, request_id, inputs))
        key = (user_id, assistant_id, request_id)
        existing = next((row for row in self.tasks.values()
                         if (row["user_id"], row["assistant_id"], row["request_id"]) == key), None)
        if existing is not None:
            if existing["inputs"] != inputs:
                raise ValueError("idempotency conflict")
            return existing["snapshot"]
        task_id = f"task-{len(self.tasks) + 1}"
        snapshot = {
            "task_id": task_id, "assistant_id": assistant_id,
            "request_id": request_id, "prompt": inputs["prompt"],
            "status": "queued", "stage": "queued", "detail": "queued",
            "created_at": "now", "started_at": None, "finished_at": None,
            "elapsed_seconds": 0.0, "duration_seconds": inputs["duration_seconds"],
            "updated_seq": 1, "result": None, "error": None,
            "simulation": inputs["simulation"],
            **{name: inputs[name] for name in (
                "attachment_ids", "registered_plan_id", "plan_action", "use_embeddings",
                "max_steps", "max_tool_calls")},
        }
        self.tasks[task_id] = {"user_id": user_id, "assistant_id": assistant_id,
                               "request_id": request_id, "inputs": inputs,
                               "snapshot": snapshot}
        return snapshot

    def get_task(self, user_id, task_id):
        self.calls.append(("get_task", user_id, task_id))
        row = self.tasks.get(task_id)
        if row is None or row["user_id"] != user_id:
            raise KeyError(task_id)
        return row["snapshot"]

    def snapshot(self, user_id, execution_modes, model_name, *, limit):
        self.snapshot_calls.append((user_id, execution_modes, model_name, limit))
        assistants = []
        for assistant_id in ("chat", "standards", "plan"):
            tasks = [row["snapshot"] for row in self.tasks.values()
                     if row["user_id"] == user_id and row["assistant_id"] == assistant_id]
            assistants.append({
                "assistant_id": assistant_id, "title": assistant_id,
                "active_task_id": next((task["task_id"] for task in tasks
                                         if task["status"] in {"queued", "running", "cancelling", "recovering"}), None),
                "tasks": tasks[-limit:], "execution_mode": execution_modes[assistant_id],
            })
        return {"workspace_id": f"workspace-{user_id}", "owner_user_id": user_id,
                "cursor": self.event_cursor(user_id), "mode": "infrastructure_demo",
                "model_name": model_name, "assistants": assistants}

    def cancel(self, user_id, task_id):
        self.cancel_calls.append((user_id, task_id))
        return self.get_task(user_id, task_id)

    def reset(self, user_id, assistant_id):
        self.calls.append(("reset", user_id, assistant_id))
        return 1

    def event_cursor(self, user_id):
        return max((row["seq"] for row in self.events.get(user_id, [])), default=0)

    def events_after(self, user_id, cursor, *, limit):
        self.calls.append(("events_after", user_id, cursor, limit))
        events = [event for event in self.events.get(user_id, []) if event["seq"] > cursor]
        return events[:limit], None


class DatabaseTaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRepository()

    def service(self, user_id=7, *, workspace_id=None, modes=None):
        return DatabaseTaskService(
            self.repository, user_id=user_id, workspace_id=workspace_id or f"workspace-{user_id}",
            execution_modes=modes, model_name="test-model")

    def test_snapshot_is_bounded_and_instance_id_is_stable(self):
        first = self.service()
        second = self.service()
        self.assertEqual(first.server_instance_id, second.server_instance_id)
        self.assertEqual(first.server_instance_id, "persisted-workspace-7")
        snapshot = first.snapshot()
        self.assertEqual(snapshot["server_instance_id"], second.server_instance_id)
        self.assertEqual(snapshot["owner_user_id"], 7)
        self.assertEqual(self.repository.snapshot_calls[0][3], 50)

    def test_submit_keeps_old_shape_and_server_selected_execution_mode(self):
        service = self.service(modes={"chat": "model"})
        task = service.submit(
            "chat", " request-1 ", " hello ", duration_seconds=90,
            fail_at_end=True, attachment_ids=["asset-a"], use_embeddings=True,
            max_steps=8, max_tool_calls=12)
        self.assertEqual(task["prompt"], "hello")
        self.assertFalse(task["simulation"])
        self.assertIsNone(task["duration_seconds"])
        self.assertEqual(task["attachment_ids"], ["asset-a"])
        self.assertEqual(service.execution_mode("chat"), "model")
        submitted = self.repository.calls[0]
        self.assertEqual(submitted[1:4], (7, "chat", "request-1"))
        self.assertFalse(submitted[4]["simulation"])
        self.assertFalse(submitted[4]["fail_at_end"])

    def test_simulation_controls_are_validated_and_forwarded(self):
        service = self.service()
        with self.assertRaises(ValueError):
            service.submit("chat", "bad", "prompt", duration_seconds=-1)
        task = service.submit("chat", "ok", "prompt", duration_seconds=1.5)
        self.assertTrue(task["simulation"])
        self.assertEqual(task["duration_seconds"], 1.5)

    def test_task_reads_and_cancellation_are_always_owner_scoped(self):
        owner = self.service(user_id=7)
        other = self.service(user_id=8)
        task = owner.submit("chat", "r1", "private")
        self.assertEqual(owner.get_task(task["task_id"])["task_id"], task["task_id"])
        with self.assertRaises(KeyError):
            other.get_task(task["task_id"])
        with self.assertRaises(KeyError):
            other.cancel(task["task_id"])
        self.assertEqual(self.repository.cancel_calls, [(8, task["task_id"])])

    def test_database_events_are_read_by_owner_and_facade_close_does_not_cancel(self):
        self.repository.events[7] = [
            {"seq": 1, "type": "task.updated", "data": {"task_id": "a"}},
        ]
        service = self.service(user_id=7)
        self.assertEqual(service.events.cursor, 1)
        self.assertEqual(service.events.read_after(0)[0], self.repository.events[7])
        self.assertEqual(self.repository.calls[-1], ("events_after", 7, 0, 200))
        service.close()
        self.assertEqual(self.repository.cancel_calls, [])

    def test_event_subscriptions_use_bounded_event_loop_timers(self):
        async def run():
            reader = DatabaseEventReader(self.repository, 7, poll_interval=0.01)
            signal = asyncio.Event()
            unsubscribe = reader.subscribe(asyncio.get_running_loop(), signal.set)
            try:
                await asyncio.wait_for(signal.wait(), timeout=0.2)
            finally:
                unsubscribe()
                reader.close()
            self.assertEqual(len(reader._subscriptions), 0)

        asyncio.run(run())


class PersistentTenantRegistryTests(unittest.TestCase):
    def test_lru_cache_does_not_reject_users_or_cancel_persisted_tasks(self):
        database = FakeDatabase()
        repository = FakeRepository()
        registry = PersistentTenantRegistry(
            database, root=".", repository=repository, max_cached_users=2,
            document_factory=FakeDocuments)
        first = registry.get(1)
        self.assertFalse(hasattr(first.tasks, "_executor"))
        self.assertFalse(hasattr(first.tasks, "_runner"))
        self.assertFalse(hasattr(registry, "_executor"))
        registry.get(2)
        third = registry.get(3)
        self.assertEqual(len(registry._users), 2)
        self.assertNotIn(1, registry._users)
        self.assertEqual(third.user_id, 3)
        # Recreating the evicted facade reuses the same stable workspace and
        # its DB-backed task history rather than maintaining a process cap.
        recreated = registry.get(1)
        self.assertEqual(recreated.workspace_id, first.workspace_id)
        self.assertEqual(len(database.ensure_calls), 4)
        registry.close()
        self.assertEqual(repository.cancel_calls, [])

    def test_registry_close_releases_only_local_document_cache(self):
        database = FakeDatabase()
        repository = FakeRepository()
        registry = PersistentTenantRegistry(
            database, root=".", repository=repository, document_factory=FakeDocuments)
        context = registry.get(1)
        registry.close()
        self.assertTrue(context.documents.closed)
        self.assertEqual(repository.cancel_calls, [])
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            registry.get(1)


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "set RAG_ARCH_TEST_DATABASE_URL to a dedicated PostgreSQL test instance")
class PostgreSQLPersistentTaskFacadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:
            raise unittest.SkipTest("psycopg 3 is required for PostgreSQL integration tests") from exc
        cls.psycopg = psycopg
        cls.sql = sql

    def setUp(self):
        self.schema = f"persist-task-{uuid4().hex}"
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("CREATE SCHEMA {}").format(
                self.sql.Identifier(self.schema)))
        self.addCleanup(self._drop_schema)
        self.database = Database(TEST_DATABASE_URL, schema=self.schema)
        self.database.migrate()
        with self.database.transaction() as connection:
            self.owner = self._insert_user(connection, "owner")
            self.other = self._insert_user(connection, "other")
        self.repository = TaskRepository(self.database)
        self.registry = PersistentTenantRegistry(
            self.database, root=".", repository=self.repository,
            document_factory=FakeDocuments)
        self.addCleanup(self.registry.close)

    @staticmethod
    def _insert_user(connection, username):
        return connection.execute(
            """
            INSERT INTO users(
                username, username_key, password_salt, password_hash,
                password_iterations, enabled, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, TRUE, 1, 1) RETURNING id
            """,
            (username, username, b"salt", b"digest", 1000),
        ).fetchone()["id"]

    def _drop_schema(self):
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                self.sql.Identifier(self.schema)))

    def test_api_restart_reads_tasks_events_and_stable_workspace_identity(self):
        first = self.registry.get(self.owner)
        task = first.tasks.submit("chat", "request-1", "persist me")
        original_workspace_id = first.workspace_id
        original_server_id = first.tasks.server_instance_id
        first.tasks.close()

        second_registry = PersistentTenantRegistry(
            Database(TEST_DATABASE_URL, schema=self.schema), root=".",
            repository=TaskRepository(Database(TEST_DATABASE_URL, schema=self.schema)),
            document_factory=FakeDocuments)
        self.addCleanup(second_registry.close)
        reopened = second_registry.get(self.owner)

        self.assertEqual(reopened.workspace_id, original_workspace_id)
        self.assertEqual(reopened.tasks.server_instance_id, original_server_id)
        self.assertEqual(reopened.tasks.get_task(task["task_id"])["status"], "queued")
        snapshot = reopened.tasks.snapshot()
        self.assertEqual(snapshot["assistants"][0]["active_task_id"], task["task_id"])
        events, reason = reopened.tasks.events.read_after(0)
        self.assertIsNone(reason)
        self.assertEqual(events[0]["type"], "task.updated")
        self.assertEqual(events[0]["data"]["task_id"], task["task_id"])

    def test_owner_filter_idempotency_and_close_preserve_queued_task(self):
        owner = self.registry.get(self.owner).tasks
        other = self.registry.get(self.other).tasks
        task = owner.submit("standards", "same-key", "same payload")
        duplicate = owner.submit("standards", "same-key", "same payload")
        self.assertEqual(duplicate["task_id"], task["task_id"])
        with self.assertRaises(IdempotencyConflict):
            owner.submit("standards", "same-key", "different payload")
        with self.assertRaises(KeyError):
            other.get_task(task["task_id"])
        with self.assertRaises(KeyError):
            other.cancel(task["task_id"])

        owner.close()
        persisted = self.repository.get_task(self.owner, task["task_id"])
        self.assertEqual(persisted["status"], "queued")


if __name__ == "__main__":
    unittest.main()
