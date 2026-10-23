import asyncio
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rag_arch.async_support import run_local
from rag_arch.async_support import TaskResourceCleanupError
from rag_arch.web.task_worker import WorkerRuntime
from tests.test_web_task_worker import FakeRepository, IDENTITY, _claim


def runtime(repository, **options):
    return WorkerRuntime(repository, identity=IDENTITY, poll_interval=0.005,
                         heartbeat_interval=0.02, lease_seconds=3, **options)


class AsyncWorkerTests(unittest.TestCase):
    def test_production_dispatch_calls_real_async_chat_and_closes_each_owned_model(self):
        from tests.test_web_async_runners import AsyncOnlyModel, fake_settings
        models = [AsyncOnlyModel("Alice answer"), AsyncOnlyModel("Bob answer")]
        repository = FakeRepository([_claim("alice", user_id=7), _claim("bob", user_id=8)])
        closed_stores = []

        def store_factory(*args, **kwargs):
            return SimpleNamespace(close=lambda: closed_stores.append(kwargs["user_id"]))

        with TemporaryDirectory(prefix="rag-async-worker-dispatch-") as directory:
            with patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(directory) / "audits"), \
                    patch("rag_arch.providers.create_async_chat_model", side_effect=models):
                worker = runtime(repository, concurrency=2, settings=fake_settings(),
                                 database=object(), root=directory,
                                 resources=SimpleNamespace(pdf_backend=None),
                                 document_store_factory=store_factory)
                worker.run(max_tasks=2)
        self.assertTrue(worker.async_execution)
        self.assertTrue(all(row[2] == "completed" for row in repository.finishes), repository.finishes)
        self.assertEqual({row[3]["text"] for row in repository.finishes}, {"Alice answer", "Bob answer"})
        self.assertTrue(all(row[3]["model_calls"] == 1 for row in repository.finishes))
        self.assertTrue(all(row[3]["token_usage"]["total_tokens"] == 10 for row in repository.finishes))
        self.assertTrue(all(len(model.calls) == 1 and model.client.closed for model in models))
        self.assertEqual(set(closed_stores), {7, 8})

    def test_async_tasks_overlap_on_one_loop_and_preserve_request_control(self):
        from rag_arch.request_control import check_request_control
        entered = []
        gate = None

        async def runner(request, progress, cancel):
            nonlocal gate
            if gate is None:
                gate = asyncio.Event()
            check_request_control()
            entered.append((request["user_id"], id(asyncio.get_running_loop())))
            if len(entered) == 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), 2)
            check_request_control()
            return {"text": str(request["user_id"])}

        repository = FakeRepository([_claim("alice", user_id=7), _claim("bob", user_id=8)])
        worker = runtime(repository, concurrency=2, runner=runner)
        worker.run(max_tasks=2)
        self.assertEqual(len(entered), 2)
        self.assertEqual(entered[0][1], entered[1][1])
        self.assertEqual({row[3]["text"] for row in repository.finishes}, {"7", "8"})
        self.assertTrue(all(row[2] == "completed" for row in repository.finishes))
        self.assertFalse(worker._async_runtime._thread.is_alive())

    def test_cancel_does_not_finish_or_stop_before_physical_local_work_and_cleanup(self):
        local_entered, release_local, cleanup = threading.Event(), threading.Event(), threading.Event()

        def physical_work():
            local_entered.set()
            release_local.wait(3)

        async def runner(request, progress, cancel):
            try:
                await run_local(physical_work)
                return {"text": "not published"}
            finally:
                cleanup.set()

        class CheckedRepository(FakeRepository):
            def finish(self, *args, **kwargs):
                self_test.assertTrue(cleanup.is_set())
                return super().finish(*args, **kwargs)

        self_test = self
        repository = CheckedRepository([_claim("cancelled")])
        worker = runtime(repository, concurrency=1, runner=runner)
        thread = threading.Thread(target=lambda: worker.run(max_tasks=1))
        thread.start()
        try:
            self.assertTrue(local_entered.wait(2))
            repository.cancel.add("cancelled")
            time.sleep(0.12)
            self.assertEqual(repository.finishes, [])
            self.assertEqual(repository.stopped, [])
            self.assertFalse(cleanup.is_set())
        finally:
            release_local.set()
            thread.join(4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(repository.finishes[0][2], "cancelled")
        self.assertIsNone(repository.finishes[0][3])
        self.assertTrue(cleanup.is_set())

    def test_task_deadline_cancels_native_await_then_records_failure(self):
        cleanup = threading.Event()

        async def runner(request, progress, cancel):
            try:
                await asyncio.sleep(30)
            finally:
                cleanup.set()

        repository = FakeRepository([_claim("timeout")])
        worker = runtime(repository, concurrency=1, runner=runner, task_timeout_seconds=0.05)
        worker.run(max_tasks=1)
        self.assertTrue(cleanup.is_set())
        self.assertEqual(repository.finishes[0][2], "failed")
        self.assertIsInstance(json.loads(repository.finishes[0][4]), dict)
        self.assertFalse(worker._async_runtime._thread.is_alive())

    def test_database_heartbeat_outage_stops_inflight_async_runner_without_replay(self):
        entered, cleanup = threading.Event(), threading.Event()

        async def runner(request, progress, cancel):
            entered.set()
            try:
                await asyncio.sleep(30)
            finally:
                cleanup.set()

        repository = FakeRepository([_claim("db-outage")])
        worker = runtime(repository, concurrency=1, runner=runner)
        thread = threading.Thread(target=lambda: worker.run(max_tasks=1))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            repository.fail_heartbeats = True
            thread.join(3)
        finally:
            if thread.is_alive():
                repository.fail_heartbeats = False
                worker.request_stop()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(cleanup.is_set())
        self.assertEqual(repository.claimed, ["db-outage"])
        self.assertEqual(repository.finishes[0][2], "failed")

    def test_async_owned_resources_close_on_their_loop_before_worker_stopped(self):
        events = []

        class Resources:
            async def aclose(self):
                await asyncio.sleep(0)
                events.append(("aclose", id(asyncio.get_running_loop())))

            def close(self):
                events.append(("close", None))

        async def runner(request, progress, cancel):
            events.append(("runner", id(asyncio.get_running_loop())))
            return {"text": "ok"}

        class Repository(FakeRepository):
            def stop_worker(self, instance):
                events.append(("stopped", None))
                return super().stop_worker(instance)

        repository = Repository([_claim("resources")])
        worker = runtime(repository, concurrency=1, runner=runner,
                         resource_factory=lambda root: Resources())
        worker._get_resources()
        worker.run(max_tasks=1)
        self.assertEqual([name for name, _ in events], ["runner", "aclose", "close", "stopped"])
        self.assertEqual(events[0][1], events[1][1])

    def test_cleanup_failure_retires_worker_without_cancelling_another_user(self):
        other_entered = threading.Event()

        async def runner(request, progress, cancel):
            if request["task_id"] == "close-failed":
                while not other_entered.is_set():
                    await asyncio.sleep(0.001)
                raise TaskResourceCleanupError("chat", exception_types=["OSError"])
            other_entered.set()
            await asyncio.sleep(0.08)
            self.assertFalse(cancel.is_set())
            return {"text": "other user finished"}

        repository = FakeRepository([_claim("close-failed"), _claim("other", user_id=8),
                                     _claim("must-stay-queued", user_id=9)])
        worker = runtime(repository, concurrency=2, runner=runner)
        worker.run(max_tasks=3)
        self.assertEqual(repository.claimed, ["close-failed", "other"])
        results = {row[0]: row for row in repository.finishes}
        self.assertEqual(results["close-failed"][2], "failed")
        self.assertEqual(json.loads(results["close-failed"][4])["code"], "resource_cleanup_failed")
        self.assertEqual(results["other"][2], "completed")
        self.assertTrue(repository.stopped)

    def test_cancelled_real_dispatch_marks_cleanup_failure_and_leaves_next_task_queued(self):
        from tests.test_web_async_runners import AsyncOnlyModel, fake_settings
        entered = threading.Event()

        class BrokenClient:
            async def aclose(self):
                raise OSError("synthetic idle socket close failure")

        class SlowModel(AsyncOnlyModel):
            async def ainvoke(self, messages, **kwargs):
                entered.set()
                await asyncio.sleep(30)

        model = SlowModel()
        model.root_async_client = BrokenClient()
        repository = FakeRepository([_claim("cancel-close"), _claim("leave-queued", user_id=8)])
        with TemporaryDirectory(prefix="rag-cancel-close-") as directory:
            with patch("rag_arch.runtime_audit.AUDIT_ROOT", Path(directory) / "audits"), \
                    patch("rag_arch.providers.create_async_chat_model", return_value=model):
                worker = runtime(repository, concurrency=1, settings=fake_settings(), database=object(),
                                 root=directory, resources=SimpleNamespace(pdf_backend=None),
                                 document_store_factory=lambda *a, **k: SimpleNamespace(close=lambda: None))
                thread = threading.Thread(target=lambda: worker.run(max_tasks=2))
                thread.start()
                try:
                    self.assertTrue(entered.wait(3))
                    repository.cancel.add("cancel-close")
                    thread.join(5)
                finally:
                    if thread.is_alive():
                        worker.request_stop()
                        thread.join(5)
                self.assertFalse(thread.is_alive())
                audits = list((Path(directory) / "audits").rglob("result.json"))
                self.assertTrue(audits)
                self.assertEqual(json.loads(audits[0].read_text(encoding="utf-8"))["status"], "cancelled")
        self.assertEqual(repository.claimed, ["cancel-close"])
        self.assertEqual(repository.finishes[0][2], "cancelled")
        self.assertTrue(worker._retire_requested.is_set())
