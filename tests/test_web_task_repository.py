"""PostgreSQL integration tests for durable task scheduling and worker fencing.

Only RAG_ARCH_TEST_DATABASE_URL is used.  Every test gets a random schema and
bounded lock/statement timeouts; no test reads production database settings.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import socket
import threading
import time
import unittest
from uuid import uuid4

from rag_arch.web.db import Database
from rag_arch.web.task_repository import (
    IdempotencyConflict,
    StaleAttempt,
    TaskRepository,
    WorkerIdentityConflict,
)
from rag_arch.web.task_service import BusyError


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL instance")
class PostgreSQLTaskRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise unittest.SkipTest("psycopg 3 is required for task repository integration tests") from exc
        cls.psycopg = psycopg
        cls.sql = sql

    def setUp(self):
        self.schema = "task_repo_" + uuid4().hex
        self.database = Database(TEST_DATABASE_URL, schema=self.schema)
        self.addCleanup(self._drop_schema)
        self.assertEqual(self.database.migrate(), [
            "001_multiuser.sql", "002_task_workers.sql",
            "003_shared_login_limits.sql", "004_process_resources.sql",
        ])

        base_connect = self.database.connect

        def bounded_connect():
            connection = base_connect()
            connection.execute("SET lock_timeout = '4s'")
            connection.execute("SET statement_timeout = '12s'")
            connection.execute("SET idle_in_transaction_session_timeout = '12s'")
            connection.commit()
            return connection

        self.database.connect = bounded_connect
        self.repo = TaskRepository(self.database)
        with self.database.transaction() as connection:
            self.user_a = self._insert_user(connection, "repo-a-" + uuid4().hex[:8])
            self.user_b = self._insert_user(connection, "repo-b-" + uuid4().hex[:8])

    def _drop_schema(self):
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(
                self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    self.sql.Identifier(self.schema)
                )
            )

    @staticmethod
    def _insert_user(connection, username):
        row = connection.execute(
            "INSERT INTO users(username, username_key, password_salt, password_hash, "
            "password_iterations, enabled, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, 1000, TRUE, %s, %s) RETURNING id",
            (username, username.lower(), b"salt", b"digest", time.time(), time.time()),
        ).fetchone()
        return int(row["id"])

    @staticmethod
    def _inputs(prompt="hello", *, simulation=True, attachment_ids=None, plan_id=None):
        return {
            "prompt": prompt,
            "duration_seconds": 0.0 if simulation else None,
            "fail_at_end": False,
            "simulation": simulation,
            "attachment_ids": list(attachment_ids or []),
            "registered_plan_id": plan_id,
            "plan_action": "question",
            "use_embeddings": False,
            "max_steps": None,
            "max_tool_calls": None,
        }

    def _worker(self, worker_instance=None, token=None):
        instance = worker_instance or uuid4().hex
        start = token or ("start-" + uuid4().hex)
        self.repo.register_worker(instance, socket.gethostname(), os.getpid(), start)
        return instance, start

    def _wait_for_database_lock_wait(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.database.transaction() as connection:
                row = connection.execute(
                    "SELECT pid FROM pg_stat_activity WHERE datname = current_database() "
                    "AND usename = current_user AND wait_event_type = 'Lock' "
                    "AND pid <> pg_backend_pid() LIMIT 1"
                ).fetchone()
            if row is not None:
                return
            time.sleep(0.02)
        self.fail("test operation did not reach the expected PostgreSQL row-lock wait")

    def _submit(self, user_id, assistant="chat", request_id=None, prompt="hello", **kwargs):
        return self.repo.submit(
            user_id, assistant, request_id or uuid4().hex,
            self._inputs(prompt, **kwargs),
        )

    def test_migration_adds_worker_registry_without_mutating_initial_schema(self):
        with self.database.transaction() as connection:
            tables = connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
            ).fetchall()
            columns = connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'tasks'"
            ).fetchall()
        self.assertIn("worker_instances", {row["table_name"] for row in tables})
        self.assertTrue({"inputs", "status", "generation", "worker_instance", "attempt_token",
                         "lease_until", "workspace_id"}.issubset({row["column_name"] for row in columns}))

    def test_first_snapshot_initializes_one_stable_workspace_and_three_slots(self):
        first = self.repo.snapshot(self.user_a)
        second = self.repo.snapshot(self.user_a)
        self.assertEqual(first["workspace_id"], second["workspace_id"])
        self.assertEqual(first["cursor"], 0)
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT assistant_id FROM assistant_slots WHERE user_id = %s ORDER BY assistant_id",
                (self.user_a,),
            ).fetchall()
            workspace_count = connection.execute(
                "SELECT count(*) AS count FROM workspaces WHERE user_id = %s",
                (self.user_a,),
            ).fetchone()["count"]
        self.assertEqual([row["assistant_id"] for row in rows], ["chat", "plan", "standards"])
        self.assertEqual(workspace_count, 1)

    def test_submit_is_idempotent_by_structured_inputs_and_one_slot_per_owner(self):
        request_id = "stable-request"
        first = self._submit(self.user_a, request_id=request_id, prompt="question")
        duplicate = self._submit(self.user_a, request_id=request_id, prompt="question")
        self.assertEqual(duplicate["task_id"], first["task_id"])
        self.assertEqual(duplicate["updated_seq"], first["updated_seq"])
        with self.assertRaises(IdempotencyConflict):
            self._submit(self.user_a, request_id=request_id, prompt="different")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="another-request")

        other_user = self._submit(self.user_b, request_id=request_id)
        self.assertNotEqual(other_user["task_id"], first["task_id"])
        self.assertIsNone(self.repo.get_task(self.user_b, first["task_id"]))
        self.assertEqual(self.repo.get_task(self.user_a, first["task_id"])["task_id"], first["task_id"])

    def test_concurrent_same_slot_submission_only_commits_one_task(self):
        barrier = threading.Barrier(2)

        def submit(request_id):
            barrier.wait(timeout=3)
            try:
                return self._submit(self.user_a, request_id=request_id)
            except BusyError:
                return "busy"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(submit, ("parallel-a", "parallel-b")))
        self.assertEqual(sum(value == "busy" for value in outcomes), 1)
        self.assertEqual(sum(isinstance(value, dict) for value in outcomes), 1)
        with self.database.transaction() as connection:
            count = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE user_id = %s AND assistant_id = 'chat' "
                "AND status IN ('queued', 'running', 'cancelling', 'recovering')",
                (self.user_a,),
            ).fetchone()["count"]
        self.assertEqual(count, 1)

    def test_claim_enforces_global_limit_and_round_robins_users(self):
        self.repo.set_global_limit(1)
        first = self._submit(self.user_a, assistant="chat", request_id="a-chat")
        second = self._submit(self.user_b, assistant="chat", request_id="b-chat")
        third = self._submit(self.user_a, assistant="standards", request_id="a-standards")
        worker, _ = self._worker()

        claim_a = self.repo.claim_next(worker)
        self.assertEqual(claim_a["task_id"], first["task_id"])
        self.assertEqual(claim_a["request"]["user_id"], self.user_a)
        self.assertEqual(claim_a["request"]["attempt"], claim_a["attempt_token"])
        self.assertIsNone(self.repo.claim_next(worker), "active task must consume the global slot")
        self.assertTrue(self.repo.finish(worker, first["task_id"], claim_a["attempt_token"],
                                         "completed", result={"text": "a"}))

        claim_b = self.repo.claim_next(worker)
        self.assertEqual(claim_b["task_id"], second["task_id"], "next user should win before A's second task")
        self.assertTrue(self.repo.finish(worker, second["task_id"], claim_b["attempt_token"],
                                         "completed", result={"text": "b"}))
        claim_a2 = self.repo.claim_next(worker)
        self.assertEqual(claim_a2["task_id"], third["task_id"])

    def test_cancel_queued_releases_slot_and_running_cancel_waits_for_finish(self):
        queued = self._submit(self.user_a, request_id="queued-cancel")
        cancelled = self.repo.cancel(self.user_a, queued["task_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        replacement = self._submit(self.user_a, request_id="after-queued-cancel")

        worker, _ = self._worker()
        claim = self.repo.claim_next(worker)
        self.assertEqual(claim["task_id"], replacement["task_id"])
        cancelling = self.repo.cancel(self.user_a, replacement["task_id"])
        self.assertEqual(cancelling["status"], "cancelling")
        self.assertTrue(self.repo.execution_control(worker, replacement["task_id"],
                                                    claim["attempt_token"])["cancel_requested"])
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="during-cancel")
        before_retry = self.repo.event_cursor(self.user_a)
        self.assertTrue(self.repo.finish(worker, replacement["task_id"], claim["attempt_token"],
                                         "completed", result={"text": "discarded"}))
        self.assertEqual(self.repo.get_task(self.user_a, replacement["task_id"])["status"], "cancelled")
        after_first_finish = self.repo.event_cursor(self.user_a)
        self.assertTrue(self.repo.finish(worker, replacement["task_id"], claim["attempt_token"],
                                         "completed", result={"text": "discarded"}))
        self.assertEqual(self.repo.event_cursor(self.user_a), after_first_finish)
        self.assertGreater(after_first_finish, before_retry)
        self._submit(self.user_a, request_id="after-running-cancel")

    def test_attempt_token_fences_progress_and_finish(self):
        task = self._submit(self.user_a, request_id="fenced")
        worker, _ = self._worker()
        claim = self.repo.claim_next(worker)
        self.assertEqual(claim["task_id"], task["task_id"])
        with self.assertRaises(StaleAttempt):
            self.repo.progress(worker, task["task_id"], "wrong-token", "work")
        self.assertFalse(self.repo.finish(worker, task["task_id"], "wrong-token", "completed",
                                          result={"text": "stale"}))
        updated = self.repo.progress(worker, task["task_id"], claim["attempt_token"], "retrieving", "step")
        self.assertEqual(updated["stage"], "retrieving")
        events, reason = self.repo.events_after(self.user_a, task["updated_seq"])
        self.assertIsNone(reason)
        self.assertEqual(events[-1]["type"], "task.updated")
        self.assertEqual(events[-1]["data"]["updated_seq"], events[-1]["seq"])

    def test_stale_lease_keeps_slot_until_exact_process_exit_is_confirmed(self):
        self.repo.set_global_limit(1)
        task = self._submit(self.user_a, request_id="recover-me")
        worker, start_token = self._worker()
        claim = self.repo.claim_next(worker)
        self.assertEqual(claim["task_id"], task["task_id"])
        stale = self.repo.mark_stale_workers(now=time.time() + 120)
        self.assertEqual(stale, [worker])
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "recovering")
        self.assertIsNone(self.repo.claim_next(worker))
        with self.assertRaises(WorkerIdentityConflict):
            self.repo.recover_confirmed_exit(worker, "reused-pid-token")

        self.repo.recover_confirmed_exit(worker, start_token)
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "failed")
        self._submit(self.user_a, request_id="after-confirmed-exit")
        next_worker, _ = self._worker()
        next_claim = self.repo.claim_next(next_worker)
        self.assertIsNotNone(next_claim)
        self.assertNotEqual(next_claim["attempt_token"], claim["attempt_token"])

    def test_model_task_keeps_user_and_global_slots_after_parent_exit(self):
        self.repo.set_global_limit(1)
        model_task = self._submit(self.user_a, assistant="plan", request_id="model-unknown",
                                  prompt="review", simulation=False)
        worker, start_token = self._worker()
        claim = self.repo.claim_next(worker)
        self.assertEqual(claim["task_id"], model_task["task_id"])
        self.repo.mark_stale_workers(now=time.time() + 120)

        # Parent worker exit does not prove an external model request or any
        # document child resource stopped; recovery must keep both capacities.
        finalized = self.repo.recover_confirmed_exit(worker, start_token)
        self.assertEqual(finalized, [])
        current = self.repo.get_task(self.user_a, model_task["task_id"])
        self.assertEqual(current["status"], "recovering")
        self.assertIn("未核验", current["detail"])
        with self.assertRaises(BusyError):
            self._submit(self.user_a, assistant="plan", request_id="model-after-parent-exit")

        queued_elsewhere = self._submit(self.user_b, assistant="chat", request_id="global-capacity-held")
        next_worker, _ = self._worker()
        self.assertIsNone(self.repo.claim_next(next_worker))
        self.assertEqual(self.repo.get_task(self.user_b, queued_elsewhere["task_id"])["status"], "queued")

    def test_worker_registration_heartbeat_and_graceful_stop(self):
        task = self._submit(self.user_a, request_id="worker-stop")
        worker, start_token = self._worker()
        with self.assertRaises(WorkerIdentityConflict):
            self.repo.register_worker(worker, socket.gethostname(), os.getpid(), "different-process")
        claim = self.repo.claim_next(worker)
        self.assertEqual(claim["task_id"], task["task_id"])
        self.assertTrue(self.repo.heartbeat_worker(worker, lease_seconds=20))
        self.assertGreater(self.repo.get_worker(worker)["heartbeat_at"], 0)
        with self.assertRaises(BusyError):
            self.repo.stop_worker(worker)
        self.assertTrue(self.repo.finish(worker, task["task_id"], claim["attempt_token"],
                                         "failed", error="synthetic"))
        self.assertTrue(self.repo.stop_worker(worker))
        self.assertFalse(self.repo.heartbeat_worker(worker))
        self.assertEqual(self.repo.get_worker(worker)["state"], "stopped")

    def test_registration_heartbeat_and_claim_times_start_after_lock_wait(self):
        # A scheduler-row wait must not consume the task's entire lease or
        # create a started_at timestamp from before the task was claimable.
        register_id = uuid4().hex
        lock_connection = self.database.connect()
        self.addCleanup(lock_connection.close)
        lock_connection.execute("BEGIN")
        lock_connection.execute(
            "SELECT singleton_key FROM scheduler_state WHERE singleton_key = 'global' FOR UPDATE"
        ).fetchone()
        started = threading.Event()

        def register():
            started.set()
            return self.repo.register_worker(
                register_id, socket.gethostname(), os.getpid(), "registered-after-wait"
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(register)
            self.assertTrue(started.wait(timeout=2))
            self._wait_for_database_lock_wait()
            released_at = time.time()
            lock_connection.commit()
            future.result(timeout=5)
        lock_connection.close()
        self.assertGreaterEqual(self.repo.get_worker(register_id)["heartbeat_at"], released_at)

        task = self._submit(self.user_a, request_id="lease-after-wait")
        claim = self.repo.claim_next(register_id)
        self.assertEqual(claim["task_id"], task["task_id"])

        worker_lock = self.database.connect()
        self.addCleanup(worker_lock.close)
        worker_lock.execute("BEGIN")
        worker_lock.execute(
            "SELECT worker_instance FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
            (register_id,),
        ).fetchone()
        started = threading.Event()

        def heartbeat():
            started.set()
            return self.repo.heartbeat_worker(register_id, lease_seconds=30)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(heartbeat)
            self.assertTrue(started.wait(timeout=2))
            self._wait_for_database_lock_wait()
            heartbeat_released_at = time.time()
            worker_lock.commit()
            self.assertTrue(future.result(timeout=5))
        worker_lock.close()
        self.assertGreaterEqual(self.repo.get_worker(register_id)["heartbeat_at"], heartbeat_released_at)
        with self.database.transaction() as connection:
            task_lease = connection.execute(
                "SELECT lease_until FROM tasks WHERE task_id = %s", (task["task_id"],)
            ).fetchone()["lease_until"]
        self.assertGreaterEqual(task_lease, heartbeat_released_at + 29.0)

        self.assertTrue(self.repo.finish(register_id, task["task_id"], claim["attempt_token"],
                                         "completed", result={"text": "first"}))
        queued = self._submit(self.user_a, request_id="claim-after-wait")
        scheduler_lock = self.database.connect()
        self.addCleanup(scheduler_lock.close)
        scheduler_lock.execute("BEGIN")
        scheduler_lock.execute(
            "SELECT singleton_key FROM scheduler_state WHERE singleton_key = 'global' FOR UPDATE"
        ).fetchone()
        started = threading.Event()

        def claim():
            started.set()
            return self.repo.claim_next(register_id)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(claim)
            self.assertTrue(started.wait(timeout=2))
            self._wait_for_database_lock_wait()
            claim_released_at = time.time()
            scheduler_lock.commit()
            locked_claim = future.result(timeout=5)
        scheduler_lock.close()
        self.assertEqual(locked_claim["task_id"], queued["task_id"])
        started_at = datetime.fromisoformat(
            locked_claim["event"]["data"]["started_at"]
        ).timestamp()
        self.assertGreaterEqual(started_at, claim_released_at)
        self.assertGreaterEqual(locked_claim["lease_until"], claim_released_at + 29.0)

    def test_history_is_full_scoped_to_generation_and_matching_documents(self):
        worker, _ = self._worker()
        previous = self._submit(self.user_a, request_id="model-one", prompt="first",
                                simulation=False, plan_id="plan-1")
        claim = self.repo.claim_next(worker)
        full_result = {"text": "full", "hits": [{"source": "secret-diagnostic"}],
                       "report_history": [{"version": 1}]}
        self.assertTrue(self.repo.finish(worker, previous["task_id"], claim["attempt_token"],
                                         "completed", result=full_result))
        next_task = self._submit(self.user_a, request_id="model-two", prompt="second",
                                 simulation=False, plan_id="plan-1")
        next_claim = self.repo.claim_next(worker)
        self.assertEqual(next_claim["request"]["history"], [
            {"question": "first", "result": full_result},
        ])
        public = self.repo.get_task(self.user_a, previous["task_id"])
        self.assertEqual(public["result"], {"text": "full"})
        self.repo.finish(worker, next_task["task_id"], next_claim["attempt_token"],
                         "completed", result={"text": "next"})

        generation = self.repo.reset(self.user_a, "chat")
        self.assertEqual(generation, 1)
        after_reset = self._submit(self.user_a, request_id="model-after-reset", simulation=False,
                                   plan_id="plan-1")
        after_reset_claim = self.repo.claim_next(worker)
        self.assertEqual(after_reset_claim["task_id"], after_reset["task_id"])
        self.assertEqual(after_reset_claim["request"]["history"], [])

    def test_persisted_event_cursor_and_snapshot_share_a_consistent_boundary(self):
        task = self._submit(self.user_a, request_id="event-cursor")
        snapshot = self.repo.snapshot(self.user_a, execution_modes={"chat": "simulation"}, limit=50)
        self.assertGreaterEqual(snapshot["cursor"], task["updated_seq"])
        current = next(item for item in snapshot["assistants"] if item["assistant_id"] == "chat")
        self.assertEqual(current["tasks"][0]["task_id"], task["task_id"])
        self.assertEqual(snapshot["workspace_id"], self.database.ensure_workspace(self.user_a))
        events, reason = self.repo.events_after(self.user_a, 0)
        self.assertIsNone(reason)
        self.assertEqual(events[0]["type"], "task.updated")
        self.assertEqual(events[0]["data"]["updated_seq"], events[0]["seq"])
        self.assertEqual(self.repo.events_after(self.user_a, snapshot["cursor"] + 1), ([], "cursor_ahead"))
        self.assertEqual(self.repo.events_after(self.user_a, -1), ([], "invalid_cursor"))

    def test_get_task_and_events_page_do_not_mix_pre_and_post_commit_views(self):
        task = self._submit(self.user_a, request_id="mvcc-read")
        worker, _ = self._worker()
        claim = self.repo.claim_next(worker)
        expected_running_seq = claim["event"]["seq"]

        entered = threading.Event()
        release = threading.Event()
        original_snapshot = self.repo._row_snapshot

        def pause_task_snapshot(connection, row):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release get_task snapshot")
            return original_snapshot(connection, row)

        self.repo._row_snapshot = pause_task_snapshot
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.repo.get_task, self.user_a, task["task_id"])
                self.assertTrue(entered.wait(timeout=3))
                self.assertTrue(self.repo.finish(worker, task["task_id"], claim["attempt_token"],
                                                 "completed", result={"text": "done"}))
                terminal_seq = self.repo.event_cursor(self.user_a)
                release.set()
                observed = future.result(timeout=5)
        finally:
            release.set()
            del self.repo._row_snapshot

        self.assertEqual(observed["status"], "running")
        self.assertEqual(observed["updated_seq"], expected_running_seq)
        self.assertGreater(terminal_seq, observed["updated_seq"])

        # Pause after events_after has read its cursor but before it reads the
        # page.  A later committed event must not leak into the earlier view.
        cursor_before = self.repo.event_cursor(self.user_a)
        entered = threading.Event()
        release = threading.Event()
        original_load = self.repo._load_event_rows

        def pause_event_page(connection, user_id, workspace_id, cursor, limit):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release events_after snapshot")
            return original_load(connection, user_id, workspace_id, cursor, limit)

        self.repo._load_event_rows = pause_event_page
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.repo.events_after, self.user_a, 0)
                self.assertTrue(entered.wait(timeout=3))
                self._submit(self.user_a, assistant="standards", request_id="later-event")
                later_cursor = self.repo.event_cursor(self.user_a)
                release.set()
                events, reason = future.result(timeout=5)
        finally:
            release.set()
            del self.repo._load_event_rows
        self.assertIsNone(reason)
        self.assertGreater(later_cursor, cursor_before)
        self.assertEqual(max(event["seq"] for event in events), cursor_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
