"""Local-exit isolation: real PostgreSQL transactions, injected OS probes, no APIs."""
from __future__ import annotations

import copy
import json
import os
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from tests import test_web_resource_recovery as recovery_fixture
from tests.test_web_durable_processes import _validate_test_url
from rag_arch.web.resource_recovery import maintain_recovery, recover_resources, resolve_task
from rag_arch.web.task_repository import ResourceIdentityConflict, WorkerIdentityConflict
from rag_arch.web.task_worker import ProcessProbeState


@unittest.skipUnless(os.environ.get("RAG_ARCH_TEST_DATABASE_URL"), "dedicated PostgreSQL required")
class PostgreSQLQueueRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Use the same strict disposable local database target as the public
        # durable-process suite before its fixture creates or drops any schema.
        _validate_test_url(recovery_fixture.TEST_DATABASE_URL)
        # Reuse the established disposable-schema and owned-resource fixture,
        # without inheriting or collecting its separate test cases.
        self.fixture = recovery_fixture.PostgreSQLResourceRecoveryTests()
        self.fixture.setUpClass()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.repo = self.fixture.repo
        self.database = self.fixture.database

    def _begin(self, *, cancel=False):
        self.task, self.claim, self.handle, _, self.pid, self.token, _ = self.fixture._begin(simulation=False)
        if cancel:
            self.repo.cancel(self.fixture.user_id, self.task["task_id"])
        self.probe = self.fixture._probe(self.pid, self.token)
        return self.task

    def _maintenance(self, **kwargs):
        return maintain_recovery(
            self.repo, root=self.fixture.root, probe=kwargs.pop("probe", self.probe),
            cooldown_seconds=kwargs.pop("cooldown_seconds", 0), cleanup_owned=False, **kwargs,
        )

    def _audit(self):
        with self.database.transaction() as connection:
            return connection.execute(
                "SELECT proof FROM task_resolution_audit WHERE task_id = %s",
                (self.task["task_id"],),
            ).fetchall()

    def test_interrupted_model_is_failed_without_retry_and_next_task_can_start(self):
        self._begin()
        self.repo.set_global_limit(1)
        successor = self.fixture._task(simulation=False, assistant="plan")
        new_worker = self.fixture._new_worker()
        self.assertIsNone(self.repo.claim_next(new_worker["worker_instance"]))
        reports = self._maintenance()
        current = self.repo.get_task(self.fixture.user_id, self.task["task_id"])
        self.assertEqual(current["status"], "failed")
        self.assertIsNone(current.get("result"))
        self.assertIn("远端请求结果未确认", current["error"])
        self.assertIn("可能已产生 API 费用", current["error"])
        self.assertFalse(current["error"].startswith("{"))
        self.assertIn(self.task["task_id"], next(
            row["isolated_tasks"] for row in reports if row.get("kind") == "task_isolation"))
        # The new worker is deliberately not included in the old worker's EXITED
        # probe; it stays active and can execute the queued successor once.
        started = self.repo.claim_next(new_worker["worker_instance"])
        self.assertEqual(started["task_id"], successor["task_id"])
        self.assertNotEqual(started["task_id"], self.task["task_id"])
        audit = self._audit()[0]["proof"]
        self.assertEqual(audit["resolution_kind"], "local_exit_isolation")
        self.assertEqual(audit["external_work_state"], "unknown")
        self.assertIs(audit["automatic_retry"], False)
        self.assertNotIn("external_work_confirmed_ended", audit)
        self.assertEqual(len(self._audit()), 1)

    def test_cancelled_interrupted_attempt_keeps_cancelled_status(self):
        self._begin(cancel=True)
        self._maintenance()
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "cancelled")

    def test_same_assistant_can_accept_new_request_after_isolation(self):
        self._begin()
        self._maintenance()
        new_task = self.fixture._task(simulation=False)
        self.assertNotEqual(new_task["task_id"], self.task["task_id"])

    def test_repeated_maintenance_is_idempotent_and_preserves_first_exit_time(self):
        self._begin()
        self._maintenance()
        first_exit = self.repo.get_worker(self.fixture.worker["worker_instance"])["exited_at"]
        cursor = self.repo.event_cursor(self.fixture.user_id)
        self._maintenance()
        self.assertEqual(len(self._audit()), 1)
        self.assertEqual(self.repo.event_cursor(self.fixture.user_id), cursor)
        self.assertEqual(self.repo.get_worker(self.fixture.worker["worker_instance"])["exited_at"], first_exit)

    def test_cooldown_is_bounded_and_not_extended_by_maintenance(self):
        self._begin()
        first = self._maintenance(cooldown_seconds=120)
        first_exit = self.repo.get_worker(self.fixture.worker["worker_instance"])["exited_at"]
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "recovering")
        self.assertIn("cooldown", next(row["blocked"] for row in first if row.get("kind") == "task_isolation"))
        with patch("rag_arch.web.resource_recovery.time.time", return_value=first_exit + 121):
            self._maintenance(cooldown_seconds=120)
        self.assertEqual(self.repo.get_worker(self.fixture.worker["worker_instance"])["exited_at"], first_exit)
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "failed")

    def test_live_parent_is_never_released_by_expired_lease(self):
        self._begin()
        self.repo.mark_stale_workers(now=time.time() + 120)
        probe = recovery_fixture._Probe({
            (self.fixture.worker["host_id"], self.fixture.worker["pid"],
             self.fixture.worker["process_start_token"]): ProcessProbeState.RUNNING,
        })
        self._maintenance(probe=probe)
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "recovering")
        self.assertEqual(self._audit(), [])

    def test_unknown_parent_is_not_released(self):
        self._begin()
        self._maintenance(probe=recovery_fixture._Probe())
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "running")
        self.assertEqual(self._audit(), [])

    def test_unknown_child_stays_recovering_and_unaudited(self):
        self._begin()
        self._maintenance(probe=self.fixture._probe(self.pid, self.token, ProcessProbeState.UNKNOWN))
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "recovering")
        self.assertEqual(self._audit(), [])

    def test_missing_owner_descriptor_never_guesses_child_exit(self):
        self._begin()
        resources = self.repo.resources_for_worker(self.fixture.worker["worker_instance"])
        from pathlib import Path
        Path(resources[0]["owner_descriptor_path"]).unlink()
        self._maintenance()
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "recovering")
        self.assertEqual(self._audit(), [])

    def test_old_attempt_cannot_write_result_after_isolation(self):
        self._begin()
        self._maintenance()
        self.assertFalse(self.repo.finish(
            self.fixture.worker["worker_instance"], self.task["task_id"],
            self.claim["attempt_token"], "completed", result={"answer": "stale"}))
        self.assertEqual(self.repo.get_task(self.fixture.user_id, self.task["task_id"])["status"], "failed")

    def test_repo_rechecks_identity_resource_set_attempt_and_immutable_audit(self):
        self._begin()
        self._maintenance()
        proof = self._audit()[0]["proof"]
        args = (self.fixture.worker["worker_instance"], self.task["task_id"], self.claim["attempt_token"])
        self.assertTrue(self.repo.isolate_confirmed_local_exit(*args, proof, cooldown_seconds=0))
        self.assertFalse(self.repo.isolate_confirmed_local_exit(args[0], args[1], uuid4().hex,
                                                              proof, cooldown_seconds=0))
        changed = copy.deepcopy(proof)
        changed["reason"] = "different proof cannot mutate the append-only record"
        self.assertFalse(self.repo.isolate_confirmed_local_exit(*args, changed, cooldown_seconds=0))
        self.assertEqual(len(self._audit()), 1)

    def test_repository_rejects_wrong_parent_or_missing_resources_before_release(self):
        self._begin()
        recover_resources(self.repo, root=self.fixture.root, probe=self.probe)
        worker = self.fixture.worker
        resource = self.repo.resources_for_worker(worker["worker_instance"])[0]
        proof = {
            "operator": "test", "reason": "local resources exited",
            "resolution_kind": "local_exit_isolation", "external_work_state": "unknown",
            "automatic_retry": False, "worker": dict(worker),
            "resources": [{key: resource[key] for key in (
                "resource_id", "host_id", "pid", "process_start_token")}],
        }
        args = (worker["worker_instance"], self.task["task_id"], self.claim["attempt_token"])
        bad = copy.deepcopy(proof)
        bad["worker"]["pid"] += 1
        with self.assertRaises(WorkerIdentityConflict):
            self.repo.isolate_confirmed_local_exit(*args, bad, cooldown_seconds=0)
        bad = copy.deepcopy(proof)
        bad["resources"] = []
        with self.assertRaises(ResourceIdentityConflict):
            self.repo.isolate_confirmed_local_exit(*args, bad, cooldown_seconds=0)
        self.assertEqual(self._audit(), [])

    def test_existing_admin_resolve_still_requires_external_completion_proof(self):
        self._begin()
        recover_resources(self.repo, root=self.fixture.root, probe=self.probe)
        resource = self.repo.resources_for_worker(self.fixture.worker["worker_instance"])[0]
        proof = {"operator": "admin", "reason": "unknown remote request",
                 "external_work_confirmed_ended": False, "worker": self.fixture.worker,
                 "resources": [{key: resource[key] for key in (
                     "resource_id", "host_id", "pid", "process_start_token")}],}
        with self.assertRaises(ValueError):
            resolve_task(self.repo, self.task["task_id"], proof,
                         root=self.fixture.root, probe=self.probe)
        self.assertEqual(self._audit(), [])


class MaintenanceValidationTests(unittest.TestCase):
    def test_invalid_cooldown_does_not_touch_repository(self):
        for value in (-1, True, float("nan"), float("inf"), "120"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                maintain_recovery(None, cooldown_seconds=value)


if __name__ == "__main__":
    unittest.main()
