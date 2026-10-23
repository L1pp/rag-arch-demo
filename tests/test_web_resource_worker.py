"""Worker behavior when an owned child process cannot be confirmed stopped."""
import json
import tempfile
import threading
import unittest
from uuid import uuid4

from rag_arch.resource_supervision import ResourceCleanupUncertain, begin_resource
from rag_arch.web.task_worker import ProcessIdentity, WorkerRuntime


class RecoveringRepository:
    def __init__(self, claims=()):
        self._claims = list(claims)
        self._lock = threading.Lock()
        self.finishes = []
        self.recovering = []
        self.unknown = []
        self.stopped = []
        self.second_claimed = threading.Event()
        self.recovery_saved = threading.Event()

    def register_worker(self, worker_instance, host_id, pid, process_start_token):
        return None

    def heartbeat_worker(self, worker_instance, *, lease_seconds):
        return None

    def claim_next(self, worker_instance):
        with self._lock:
            if not self._claims:
                return None
            claim = self._claims.pop(0)
            if not self._claims:
                self.second_claimed.set()
            return claim

    def execution_control(self, worker_instance, task_id, attempt_token):
        return {"cancel_requested": False, "status": "running"}

    def progress(self, *args, **kwargs):
        return True

    def begin_resource(self, worker_instance, kind, scope, *, task_id=None, owner_path=None,
                       resource_id=None, owner_descriptor_path=None):
        return {"resource_id": resource_id or uuid4().hex,
                "owner_descriptor_path": owner_descriptor_path,
                "owner_nonce": uuid4().hex}

    def resource_unknown(self, worker_instance, resource_id, error_code):
        self.unknown.append((worker_instance, resource_id, error_code))
        return True

    def keep_recovering(self, worker_instance, task_id, attempt_token, error):
        self.recovering.append((worker_instance, task_id, attempt_token, error))
        self.recovery_saved.set()
        return True

    def finish(self, worker_instance, task_id, attempt_token, status, *, result=None, error=None):
        self.finishes.append((task_id, status, result, error))
        return True

    def stop_worker(self, worker_instance):
        self.stopped.append(worker_instance)


def _claim():
    return _claim_for("task-uncertain")


def _claim_for(task_id):
    return {"task_id": task_id, "attempt_token": f"attempt-{task_id}",
            "request": {"task_id": task_id, "assistant_id": "chat", "simulation": False,
                        "prompt": "synthetic safety test"}}


class WebResourceWorkerTests(unittest.TestCase):
    def test_wrapped_cleanup_uncertainty_stays_recovering_and_never_finishes(self):
        repository = RecoveringRepository()

        def runner(request, progress, cancel_event):
            handle = begin_resource("pdf", scope="task")
            try:
                handle.unknown("synthetic_exit_unknown")
            except ResourceCleanupUncertain as exc:
                # Adapters may wrap the low-level exception. The supervisor's
                # recorded uncertainty must still prevent terminal completion.
                raise ValueError("adapter cleanup wrapper") from exc

        with tempfile.TemporaryDirectory(prefix="rag-worker-resource-") as root:
            runtime = WorkerRuntime(
                repository,
                identity=ProcessIdentity("host-test", 1234, "100.125000"),
                root=root,
                poll_interval=0.01,
                heartbeat_interval=0.05,
                lease_seconds=3,
                runner=runner,
                async_execution=False,
            )
            runtime._database_available.set()
            intent = runtime._execute_claim(_claim(), threading.Event())
            self.assertEqual(intent.status, "recovering")
            self.assertIsNone(intent.result)
            self.assertEqual(json.loads(intent.error)["code"], "resource_cleanup_uncertain")
            self.assertTrue(runtime._retire_requested.is_set())

            runtime._pending_finishes[intent.task_id] = intent
            self.assertTrue(runtime._drain_finishes())

        self.assertEqual(repository.finishes, [])
        self.assertEqual(len(repository.recovering), 1)
        self.assertEqual(repository.recovering[0][1:3], (
            "task-uncertain", "attempt-task-uncertain"))
        self.assertEqual(repository.unknown[0][2], "synthetic_exit_unknown")

    def test_worker_rejects_nonfinite_and_nonpositive_intervals(self):
        invalid = (float("nan"), float("inf"), float("-inf"), 0, -0.1, True)
        for parameter in ("poll_interval", "heartbeat_interval"):
            for value in invalid:
                options = {parameter: value}
                if parameter == "heartbeat_interval":
                    options["poll_interval"] = 0.01
                else:
                    options["heartbeat_interval"] = 0.05
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        WorkerRuntime(
                            RecoveringRepository(),
                            identity=ProcessIdentity("host-test", 1234, "100.125000"),
                            lease_seconds=3,
                            **options,
                        )

    def test_run_retires_after_other_claimed_task_finishes_without_cancelling_it(self):
        first, other = "task-uncertain", "task-other"
        repository = RecoveringRepository([_claim_for(first), _claim_for(other)])
        other_started = threading.Event()
        release_other = threading.Event()
        other_cancelled = []

        def runner(request, progress, cancel_event):
            if request["task_id"] == first:
                handle = begin_resource("word", scope="task")
                if not repository.second_claimed.wait(2):
                    raise AssertionError("worker did not claim the already queued peer task")
                try:
                    handle.unknown("synthetic_word_exit_unknown")
                except ResourceCleanupUncertain as exc:
                    raise ValueError("renderer wrapper") from exc
            other_started.set()
            if not release_other.wait(2):
                raise TimeoutError("test did not release the peer task")
            other_cancelled.append(cancel_event.is_set())
            return {"text": "peer finished"}

        with tempfile.TemporaryDirectory(prefix="rag-worker-resource-run-") as root:
            runtime = WorkerRuntime(
                repository,
                identity=ProcessIdentity("host-test", 1234, "100.125000"),
                root=root,
                concurrency=2,
                poll_interval=0.01,
                heartbeat_interval=0.05,
                lease_seconds=3,
                runner=runner,
                async_execution=False,
            )
            errors = []

            def run_worker():
                try:
                    runtime.run(max_tasks=2)
                except BaseException as exc:  # surface child-thread failures in the test
                    errors.append(exc)

            thread = threading.Thread(target=run_worker, name="test-resource-worker")
            thread.start()
            try:
                self.assertTrue(other_started.wait(2))
                self.assertTrue(repository.recovery_saved.wait(2))
                self.assertTrue(thread.is_alive(), "worker must drain its peer before exiting")
            finally:
                release_other.set()
                thread.join(4)

        self.assertFalse(thread.is_alive(), "worker must exit once other active work has drained")
        self.assertEqual(errors, [])
        self.assertEqual(repository.recovering[0][1:3], (first, f"attempt-{first}"))
        self.assertEqual(repository.finishes, [
            (other, "completed", {"text": "peer finished", "simulation": False}, None),
        ])
        self.assertEqual(other_cancelled, [False])
        self.assertEqual(repository.stopped, [], "held recovery must not publish the worker as stopped")

    def test_direct_cleanup_uncertainty_without_handle_keeps_recovery_slot(self):
        repository = RecoveringRepository([_claim()])

        def runner(request, progress, cancel_event):
            raise ResourceCleanupUncertain("direct_cleanup_uncertain")

        with tempfile.TemporaryDirectory(prefix="rag-worker-direct-uncertain-") as root:
            runtime = WorkerRuntime(
                repository,
                identity=ProcessIdentity("host-test", 1234, "100.125000"),
                root=root,
                poll_interval=0.01,
                heartbeat_interval=0.05,
                lease_seconds=3,
                runner=runner,
                async_execution=False,
            )
            runtime.run(max_tasks=1)

        self.assertEqual(repository.finishes, [])
        self.assertEqual(repository.recovering[0][1:3], (
            "task-uncertain", "attempt-task-uncertain"))
        self.assertEqual(repository.stopped, [])


if __name__ == "__main__":
    unittest.main()
