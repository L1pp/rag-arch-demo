"""PostgreSQL integration tests for durable worker process resources.

Only the dedicated RAG_ARCH_TEST_DATABASE_URL is used.  Every test owns a
random schema and exercises repository state; it never starts an OS child or
calls an AI provider.
"""
from __future__ import annotations

import os
import socket
import time
import unittest
from uuid import uuid4

from rag_arch.web.db import Database
from rag_arch.web.task_repository import (
    ResourceIdentityConflict,
    TaskRepository,
    WorkerIdentityConflict,
)
from rag_arch.web.task_service import BusyError


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL instance")
class PostgreSQLProcessResourceRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise unittest.SkipTest("psycopg 3 is required for resource repository integration tests") from exc
        cls.psycopg = psycopg
        cls.sql = sql

    def setUp(self):
        self.schema = "resource_repo_" + uuid4().hex
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
            self.user_a = self._insert_user(connection, "resource-a-" + uuid4().hex[:8])
            self.user_b = self._insert_user(connection, "resource-b-" + uuid4().hex[:8])

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
    def _inputs(prompt="resource test", *, simulation=True):
        return {
            "prompt": prompt,
            "duration_seconds": 0.0 if simulation else None,
            "fail_at_end": False,
            "simulation": simulation,
            "attachment_ids": [],
            "registered_plan_id": None,
            "plan_action": "question",
            "use_embeddings": False,
            "max_steps": None,
            "max_tool_calls": None,
        }

    def _submit(self, user_id, request_id=None, *, assistant="chat", simulation=True):
        return self.repo.submit(
            user_id, assistant, request_id or uuid4().hex,
            self._inputs(simulation=simulation),
        )

    def _worker(self, worker_instance=None):
        instance = worker_instance or uuid4().hex
        token = "start-" + uuid4().hex
        host_id = socket.gethostname()
        self.repo.register_worker(instance, host_id, os.getpid(), token)
        return {"worker_instance": instance, "host_id": host_id,
                "pid": os.getpid(), "process_start_token": token}

    def _claim(self, task, worker):
        claim = self.repo.claim_next(worker["worker_instance"])
        self.assertIsNotNone(claim)
        self.assertEqual(claim["task_id"], task["task_id"])
        return claim

    def _attach(self, worker, resource):
        return self.repo.attach_resource(
            worker["worker_instance"], resource["resource_id"],
            host_id=worker["host_id"], pid=os.getpid(),
            process_start_token="resource-start-" + uuid4().hex,
            executable="C:\\synthetic\\reader.exe",
        )

    @staticmethod
    def _proof(worker, resources, *, ended=True):
        return {
            "operator": "test-admin",
            "reason": "test-only synthetic proof",
            "external_work_confirmed_ended": ended,
            "worker": {
                "worker_instance": worker["worker_instance"],
                "host_id": worker["host_id"], "pid": worker["pid"],
                "process_start_token": worker["process_start_token"],
            },
            "resources": [
                {"resource_id": resource["resource_id"], "host_id": resource["host_id"],
                 "pid": resource["pid"], "process_start_token": resource["process_start_token"]}
                for resource in resources
            ],
        }

    def _expire_worker(self, worker):
        self.assertEqual(
            self.repo.mark_stale_workers(now=time.time() + 120),
            [worker["worker_instance"]],
        )

    def test_task_scope_resource_blocks_finish_and_preserves_both_slots(self):
        task = self._submit(self.user_a)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(
            worker["worker_instance"], "office_document", "task", task["task_id"],
            owner_path="C:\\synthetic\\document.docx",
        )
        attached = self._attach(worker, resource)
        self.assertFalse(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"],
            "completed", result={"text": "must not publish"},
        ))
        self.assertTrue(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "running")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="blocked-by-resource")

        self.assertFalse(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"],
            pid=attached["pid"] + 1,
            process_start_token=attached["process_start_token"],
        ))
        self.assertTrue(self.repo.mark_resource_cleanup_pending(
            worker["worker_instance"], resource["resource_id"],
        ))
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"],
            pid=attached["pid"], process_start_token=attached["process_start_token"],
        ))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"],
            "completed", result={"text": "finished after exit"},
        ))
        self._submit(self.user_a, request_id="slot-released")

    def test_starting_resource_can_be_confirmed_never_created(self):
        task = self._submit(self.user_a)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(
            worker["worker_instance"], "word", "task", task["task_id"],
        )
        self.assertTrue(self.repo.resource_exited(worker["worker_instance"], resource["resource_id"]))
        self.assertTrue(self.repo.resource_exited(worker["worker_instance"], resource["resource_id"]))
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], "failed",
            error="synthetic launch failure",
        ))

    def test_worker_scope_idle_resource_does_not_block_task_finish(self):
        task = self._submit(self.user_a)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        attached = self._attach(worker, resource)
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"],
            "completed", result={"text": "ok"},
        ))
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "completed")
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertEqual(self.repo.resources_for_worker(worker["worker_instance"])[0]["state"], "active")
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))

    def test_worker_cannot_stop_while_child_resource_is_unconfirmed(self):
        worker = self._worker()
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        attached = self._attach(worker, resource)
        with self.assertRaises(BusyError):
            self.repo.stop_worker(worker["worker_instance"])
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))
        self.assertTrue(self.repo.stop_worker(worker["worker_instance"]))
        self.assertEqual(self.repo.get_worker(worker["worker_instance"])["state"], "stopped")

    def test_shared_worker_resource_tracks_each_concurrent_task_use(self):
        self.repo.set_global_limit(2)
        task_a = self._submit(self.user_a, request_id="shared-a")
        task_b = self._submit(self.user_b, request_id="shared-b")
        worker = self._worker()
        claim_a = self._claim(task_a, worker)
        claim_b = self.repo.claim_next(worker["worker_instance"])
        self.assertEqual(claim_b["task_id"], task_b["task_id"])
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        self._attach(worker, resource)
        self.assertTrue(self.repo.use_resource(worker["worker_instance"], resource["resource_id"], task_a["task_id"]))
        self.assertTrue(self.repo.use_resource(worker["worker_instance"], resource["resource_id"], task_b["task_id"]))
        self.assertEqual(len(self.repo.resources_for_worker(worker["worker_instance"])[0]["uses"]), 2)
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task_a["task_id"]))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task_b["task_id"]))
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task_a["task_id"], claim_a["attempt_token"], "completed",
        ))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task_a["task_id"]))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task_b["task_id"]))
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task_b["task_id"], claim_b["attempt_token"], "completed",
        ))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task_b["task_id"]))

    def test_unknown_worker_resource_use_blocks_finish_until_exit_is_proven(self):
        task = self._submit(self.user_a, request_id="unknown-shared-pdf")
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        attached = self._attach(worker, resource)
        self.assertTrue(self.repo.use_resource(
            worker["worker_instance"], resource["resource_id"], task["task_id"],
        ))
        self.assertTrue(self.repo.resource_unknown(
            worker["worker_instance"], resource["resource_id"], "pdf_cleanup_unverified",
        ))
        self.assertTrue(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertFalse(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], "completed",
            result={"text": "must wait"},
        ))
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "running")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="unknown-pdf-retains-user-slot")
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertTrue(self.repo.finish(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], "completed",
            result={"text": "done after exit"},
        ))

    def test_task_scope_resource_cannot_be_used_by_another_task(self):
        task_a = self._submit(self.user_a, request_id="owner-a")
        task_b = self._submit(self.user_b, request_id="owner-b")
        worker = self._worker()
        self._claim(task_a, worker)
        self.repo.set_global_limit(2)
        self.assertEqual(self.repo.claim_next(worker["worker_instance"])["task_id"], task_b["task_id"])
        resource = self.repo.begin_resource(
            worker["worker_instance"], "office", "task", task_a["task_id"],
        )
        self._attach(worker, resource)
        with self.assertRaises(ResourceIdentityConflict):
            self.repo.use_resource(worker["worker_instance"], resource["resource_id"], task_b["task_id"])

    def test_keep_recovering_is_idempotent_and_keeps_global_capacity(self):
        self.repo.set_global_limit(1)
        task = self._submit(self.user_a, request_id="recovering-a")
        waiting = self._submit(self.user_b, request_id="recovering-b")
        worker = self._worker()
        claim = self._claim(task, worker)
        self.assertTrue(self.repo.keep_recovering(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], "resource_unknown",
        ))
        cursor = self.repo.event_cursor(self.user_a)
        self.assertTrue(self.repo.keep_recovering(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], "resource_unknown",
        ))
        self.assertEqual(self.repo.event_cursor(self.user_a), cursor)
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "recovering")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="same-user-slot-retained")
        another_worker = self._worker()
        self.assertIsNone(self.repo.claim_next(another_worker["worker_instance"]))
        self.assertEqual(self.repo.get_task(self.user_b, waiting["task_id"])["status"], "queued")

    def test_simulation_recovery_waits_for_task_scope_process_exit(self):
        task = self._submit(self.user_a, request_id="recover-resource")
        worker = self._worker()
        self._claim(task, worker)
        resource = self.repo.begin_resource(
            worker["worker_instance"], "word", "task", task["task_id"],
        )
        attached = self._attach(worker, resource)
        self._expire_worker(worker)
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [])
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "recovering")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="held-until-resource-exit")
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [task["task_id"]])
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "failed")

    def test_simulation_recovery_ignores_idle_worker_scope_resource(self):
        task = self._submit(self.user_a, request_id="recover-idle-pdf")
        worker = self._worker()
        self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        self._attach(worker, resource)
        self._expire_worker(worker)
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [task["task_id"]])
        self.assertEqual(self.repo.resources_for_worker(worker["worker_instance"])[0]["state"], "active")

    def test_simulation_recovery_waits_for_unreleased_shared_resource_use(self):
        task = self._submit(self.user_a, request_id="recover-shared-pdf")
        worker = self._worker()
        self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        attached = self._attach(worker, resource)
        self.assertTrue(self.repo.use_resource(worker["worker_instance"], resource["resource_id"], task["task_id"]))
        self._expire_worker(worker)
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [])
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [task["task_id"]])

    def test_model_recovery_requires_admin_proof_for_worker_and_every_resource(self):
        task = self._submit(self.user_a, request_id="manual-resolution", simulation=False)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "word", "task", task["task_id"])
        attached = self._attach(worker, resource)
        self._expire_worker(worker)
        self.assertEqual(self.repo.recover_confirmed_exit(
            worker["worker_instance"], worker["process_start_token"],
        ), [])
        self.assertEqual(self.repo.get_worker(worker["worker_instance"])["state"], "exited")
        self.assertFalse(self.repo.resolve_confirmed_task(
            worker["worker_instance"], task["task_id"],
            claim["attempt_token"],
            self._proof(worker, [attached]),
        ), "resource remains active until process identity is OS-confirmed exited")
        self.assertTrue(self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        ))
        proof = self._proof(worker, [attached])
        self.assertTrue(self.repo.resolve_confirmed_task(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], proof,
        ))
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "failed")
        self.assertTrue(self.repo.resolve_confirmed_task(
            worker["worker_instance"], task["task_id"], claim["attempt_token"], proof,
        ), "same proof is idempotent")
        with self.database.transaction() as connection:
            audit = connection.execute(
                "SELECT terminal_status, proof FROM task_resolution_audit WHERE task_id = %s",
                (task["task_id"],),
            ).fetchone()
            count = connection.execute(
                "SELECT count(*) AS count FROM task_resolution_audit WHERE task_id = %s",
                (task["task_id"],),
            ).fetchone()["count"]
        self.assertEqual(audit["terminal_status"], "failed")
        self.assertEqual(audit["proof"], proof)
        self.assertEqual(count, 1)
        with self.assertRaises(Exception):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE task_resolution_audit SET terminal_status = 'cancelled' WHERE task_id = %s",
                    (task["task_id"],),
                )

    def test_admin_resolution_rejects_unproven_or_mismatched_process_identity(self):
        task = self._submit(self.user_a, request_id="proof-validation", simulation=False)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource = self.repo.begin_resource(worker["worker_instance"], "office", "task", task["task_id"])
        attached = self._attach(worker, resource)
        self._expire_worker(worker)
        self.repo.recover_confirmed_exit(worker["worker_instance"], worker["process_start_token"])
        self.repo.resource_exited(
            worker["worker_instance"], resource["resource_id"], pid=attached["pid"],
            process_start_token=attached["process_start_token"],
        )
        proof = self._proof(worker, [attached], ended=False)
        with self.assertRaises(ValueError):
            self.repo.resolve_confirmed_task(worker["worker_instance"], task["task_id"], claim["attempt_token"], proof)
        proof = self._proof(worker, [attached])
        proof["worker"]["pid"] += 1
        with self.assertRaises(WorkerIdentityConflict):
            self.repo.resolve_confirmed_task(worker["worker_instance"], task["task_id"], claim["attempt_token"], proof)
        proof = self._proof(worker, [attached])
        proof["resources"][0]["process_start_token"] = "reused-pid"
        with self.assertRaises(ResourceIdentityConflict):
            self.repo.resolve_confirmed_task(worker["worker_instance"], task["task_id"], claim["attempt_token"], proof)
        self.assertEqual(self.repo.get_task(self.user_a, task["task_id"])["status"], "recovering")
        with self.assertRaises(BusyError):
            self._submit(self.user_a, request_id="still-held")

    def test_admin_resolution_requires_full_worker_resource_set(self):
        task = self._submit(self.user_a, request_id="all-resources-proof", simulation=False)
        worker = self._worker()
        claim = self._claim(task, worker)
        resource_a = self.repo.begin_resource(worker["worker_instance"], "word", "task", task["task_id"])
        attached_a = self._attach(worker, resource_a)
        resource_b = self.repo.begin_resource(worker["worker_instance"], "pdf_reader", "worker")
        attached_b = self._attach(worker, resource_b)
        self._expire_worker(worker)
        self.repo.recover_confirmed_exit(worker["worker_instance"], worker["process_start_token"])
        self.repo.resource_exited(worker["worker_instance"], resource_a["resource_id"],
                                  pid=attached_a["pid"], process_start_token=attached_a["process_start_token"])
        with self.assertRaises(ResourceIdentityConflict):
            self.repo.resolve_confirmed_task(worker["worker_instance"], task["task_id"], claim["attempt_token"],
                                             self._proof(worker, [attached_a]))
        self.repo.resource_exited(worker["worker_instance"], resource_b["resource_id"],
                                  pid=attached_b["pid"], process_start_token=attached_b["process_start_token"])
        self.assertTrue(self.repo.resolve_confirmed_task(
            worker["worker_instance"], task["task_id"], claim["attempt_token"],
            self._proof(worker, [attached_a, attached_b]),
        ))

    def test_resource_identity_is_tenant_worker_scoped(self):
        task_a = self._submit(self.user_a, request_id="resource-a")
        task_b = self._submit(self.user_b, request_id="resource-b")
        self.repo.set_global_limit(2)
        worker_a = self._worker()
        worker_b = self._worker()
        self._claim(task_a, worker_a)
        self._claim(task_b, worker_b)
        resource = self.repo.begin_resource(worker_a["worker_instance"], "pdf_reader", "worker")
        with self.assertRaises(KeyError):
            self.repo.resources_for_worker("missing-worker")
        with self.assertRaises(KeyError):
            self.repo.resource_unknown(worker_b["worker_instance"], resource["resource_id"], "not-owner")
        self.assertTrue(self.repo.resource_unknown(worker_a["worker_instance"], resource["resource_id"], "probe-failed"))
        self.assertTrue(self.repo.resource_unknown(worker_a["worker_instance"], resource["resource_id"], "probe-failed"))
        self.assertEqual(self.repo.resources_for_worker(worker_a["worker_instance"])[0]["state"], "unknown")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
