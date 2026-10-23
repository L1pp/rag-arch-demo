from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from uuid import uuid4
from unittest.mock import patch

from rag_arch.web.task_worker import (
    ProcessIdentity,
    ProcessProbeState,
    TaskControlUnavailable,
    WorkerRuntime,
    current_process_identity,
    process_start_token,
    probe_process_identity,
    recover_local_workers,
)


def _claim(task_id="task-1", *, assistant_id="chat", user_id=7,
           workspace_id="workspace-7", **inputs):
    request = {
        "task_id": task_id,
        "request_id": f"request-{task_id}",
        "assistant_id": assistant_id,
        "user_id": user_id,
        "workspace_id": workspace_id,
        "generation": 0,
        "attempt": 1,
        "simulation": False,
        "prompt": "synthetic worker test",
        **inputs,
    }
    return {"task_id": task_id, "attempt_token": f"token-{task_id}",
            "request": request}


class FakeRepository:
    def __init__(self, claims=()):
        self._claims = list(claims)
        self._lock = threading.Lock()
        self.claim_calls = 0
        self.claimed = []
        self.raise_after_claim = False
        self.progress_events = []
        self.finishes = []
        self.finish_failures = 0
        self.control_calls = 0
        self.fail_control_call = None
        self.fail_heartbeats = False
        self.hold_heartbeat = False
        self.heartbeat_entered = threading.Event()
        self.release_heartbeat = threading.Event()
        self.terminal = {}
        self.pause_heartbeat_control = False
        self.heartbeat_control_entered = threading.Event()
        self.release_heartbeat_control = threading.Event()
        self.cancel = set()
        self.registered = None
        self.stopped = []
        self.process_rows = []
        self.recovered = []

    def register_worker(self, worker_instance, host_id, pid, process_start_token):
        self.registered = (worker_instance, host_id, pid, process_start_token)

    def heartbeat_worker(self, worker_instance, *, lease_seconds):
        if self.hold_heartbeat:
            self.heartbeat_entered.set()
            self.release_heartbeat.wait(2)
        if self.fail_heartbeats:
            raise ConnectionError("synthetic heartbeat outage")
        return None

    def claim_next(self, worker_instance):
        with self._lock:
            self.claim_calls += 1
            if not self._claims:
                return None
            claim = self._claims.pop(0)
            self.claimed.append(claim["task_id"])
            if self.raise_after_claim:
                raise ConnectionError("synthetic committed claim response loss")
            return claim

    def execution_control(self, worker_instance, task_id, attempt_token):
        if (task_id == "A" and threading.current_thread().name == "rag-worker-heartbeat"
                and self.pause_heartbeat_control):
            self.heartbeat_control_entered.set()
            self.release_heartbeat_control.wait(2)
        with self._lock:
            self.control_calls += 1
            if self.fail_control_call == self.control_calls:
                raise ConnectionError("synthetic database outage")
            status = self.terminal.get(
                task_id, "cancelling" if task_id in self.cancel else "running")
            return {"cancel_requested": task_id in self.cancel, "status": status}

    def progress(self, worker_instance, task_id, attempt_token, stage, detail=""):
        self.progress_events.append((task_id, attempt_token, stage, detail))

    def finish(self, worker_instance, task_id, attempt_token, status, *, result=None, error=None):
        with self._lock:
            if self.finish_failures:
                self.finish_failures -= 1
                raise ConnectionError("synthetic finish outage")
            self.finishes.append((task_id, attempt_token, status, result, error))
            self.terminal[task_id] = status
            if task_id == "first":
                self.fail_heartbeats = False
        return True

    def stop_worker(self, worker_instance):
        self.stopped.append(worker_instance)

    def workers(self):
        return list(self.process_rows)

    def recover_confirmed_exit(self, worker_instance, process_start_token):
        self.recovered.append((worker_instance, process_start_token))
        return True


IDENTITY = ProcessIdentity("host-a", 1234, "100.125000")


def _runtime(repository, **kwargs):
    # These legacy runner tests exercise the retained synchronous integration.
    # Production async dispatch has its own async-only integration tests.
    kwargs.setdefault("async_execution", False)
    return WorkerRuntime(
        repository,
        identity=IDENTITY,
        poll_interval=0.005,
        heartbeat_interval=0.02,
        lease_seconds=3,
        **kwargs,
    )


class WorkerRuntimeTests(unittest.TestCase):
    def test_claims_only_when_local_thread_slot_is_free(self):
        started = threading.Event()
        release = threading.Event()

        def runner(request, progress, cancel_event):
            started.set()
            release.wait(2)
            return {"text": request["task_id"]}

        repository = FakeRepository([_claim("one"), _claim("two")])
        runtime = _runtime(repository, concurrency=1, runner=runner)
        thread = threading.Thread(target=lambda: runtime.run(max_tasks=2))
        thread.start()
        self.assertTrue(started.wait(1))
        time.sleep(0.04)
        self.assertEqual(repository.claimed, ["one"])
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(repository.claimed, ["one", "two"])
        self.assertEqual([row[2] for row in repository.finishes], ["completed", "completed"])

    def test_database_finish_retry_does_not_reexecute_runner(self):
        calls = []

        def runner(request, progress, cancel_event):
            calls.append(request["task_id"])
            return {"text": "result survives disconnect"}

        repository = FakeRepository([_claim()])
        repository.finish_failures = 1
        _runtime(repository, runner=runner).run(max_tasks=1)
        self.assertEqual(calls, ["task-1"])
        self.assertGreaterEqual(len(repository.finishes), 1)
        self.assertEqual(repository.finishes[-1][3],
                         {"text": "result survives disconnect", "simulation": False})
        self.assertEqual(repository.stopped, [repository.registered[0]])

    def test_ambiguous_committed_claim_fences_worker_without_running_or_reclaiming(self):
        calls = []
        repository = FakeRepository([_claim()])
        repository.raise_after_claim = True
        runtime = _runtime(repository, runner=lambda *args: calls.append(args))
        runtime.run(max_tasks=1)
        self.assertEqual(repository.claim_calls, 1)
        self.assertEqual(repository.claimed, ["task-1"])
        self.assertEqual(calls, [])
        self.assertEqual(repository.finishes, [])
        self.assertEqual(repository.stopped, [])
        self.assertTrue(runtime._fenced.is_set())

    def test_malformed_committed_claim_stops_without_busy_shutdown_loop(self):
        malformed = {"task_id": "ghost", "attempt_token": "token-ghost", "request": "invalid"}
        repository = FakeRepository([malformed])
        runtime = _runtime(repository, runner=lambda *_args: self.fail("runner must not run"))
        runtime.run(max_tasks=1)
        self.assertEqual(repository.claim_calls, 1)
        self.assertEqual(repository.finishes, [])
        self.assertEqual(repository.stopped, [])
        self.assertTrue(runtime._fenced.is_set())

    def test_control_plane_failure_blocks_model_request_and_is_not_cancel(self):
        calls = []

        def runner(request, progress, cancel_event):
            progress("model", "正在调用测试模型")
            calls.append("model")
            return {"text": "should not run"}

        repository = FakeRepository([_claim()])
        repository.fail_control_call = 2  # initial claim guard succeeds; model guard fails
        _runtime(repository, runner=runner).run(max_tasks=1)
        self.assertEqual(calls, [])
        self.assertEqual(repository.finishes[-1][2], "failed")
        self.assertEqual(json.loads(repository.finishes[-1][4])["code"],
                         "control_plane_unavailable")

    def test_stop_waits_for_runner_cleanup_before_finish_and_stopped(self):
        started = threading.Event()
        release = threading.Event()
        cleaned = threading.Event()

        def runner(request, progress, cancel_event):
            started.set()
            release.wait(2)
            return {"text": "finished after stop request"}

        class Store:
            def close(self):
                cleaned.set()

        repository = FakeRepository([_claim()])
        runtime = _runtime(repository, database=object(), runner=runner)
        # A test-owned cleanup hook mirrors the production DocumentStore.close.
        original_execute = runtime._execute_claim

        def execute_with_cleanup(claim, cancel_event):
            intent = original_execute(claim, cancel_event)
            cleaned.set()
            return intent

        runtime._execute_claim = execute_with_cleanup
        thread = threading.Thread(target=lambda: runtime.run(max_tasks=1))
        thread.start()
        self.assertTrue(started.wait(1))
        runtime.request_stop()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        self.assertEqual(repository.finishes, [])
        self.assertEqual(repository.stopped, [])
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(cleaned.is_set())
        self.assertEqual(repository.finishes[-1][2], "cancelled")
        self.assertEqual(repository.stopped, [repository.registered[0]])

    def test_real_runner_gets_workspace_owned_document_store_and_closes_before_finish(self):
        events = []
        backend = object()

        class Resources:
            pdf_backend = backend

            def close(self):
                events.append("resources_closed")

        class Store:
            def close(self):
                events.append("store_closed")

        def store_factory(root, workspace_id, pdf_backend, *, user_id, database):
            events.append((str(root), workspace_id, pdf_backend, user_id, database))
            return Store()

        class Repository(FakeRepository):
            def finish(self, *args, **kwargs):
                events.append("finish")
                return super().finish(*args, **kwargs)

            def stop_worker(self, worker_instance):
                events.append("worker_stopped")
                return super().stop_worker(worker_instance)

        repository = Repository([_claim()])
        database = object()
        with patch("rag_arch.web.chat_runner.run_chat",
                   return_value={"text": "tenant result"}) as run_chat:
            _runtime(repository, database=database, settings=object(), resources=Resources(),
                     document_store_factory=store_factory).run(max_tasks=1)
        created = events[0]
        self.assertEqual(created[1:], ("workspace-7", backend, 7, database))
        self.assertEqual(events[1:4], ["finish", "store_closed", "worker_stopped"])
        self.assertEqual(run_chat.call_args.kwargs["documents"].__class__.__name__, "Store")

    def test_real_run_chat_reuses_same_user_document_embeddings_and_isolates_users(self):
        from rag_arch.config import Settings
        from rag_arch.web.documents import DocumentStore
        from tests.helpers.tenant_audit_fixtures import FakeModel, reply

        class TrackingDocumentStore(DocumentStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                super().close()

        class FakeEmbeddings:
            def __init__(self):
                self.document_batches = []
                self.query_calls = 0

            def embed_documents(self, texts):
                self.document_batches.append(tuple(texts))
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                self.query_calls += 1
                return [1.0, 0.0]

        with TemporaryDirectory(prefix="rag-worker-tenant-cache-") as temp:
            root = Path(temp)
            backend = object()
            stores = {
                (7, "workspace-a"): TrackingDocumentStore(
                    root / "user-a", "workspace-a", backend, direct_text_max_tokens=0),
                (8, "workspace-b"): TrackingDocumentStore(
                    root / "user-b", "workspace-b", backend, direct_text_max_tokens=0),
            }
            try:
                asset_a = stores[(7, "workspace-a")].add(
                    "long.txt", ("混凝土强度应满足设计要求。\n" * 80).encode("utf-8"))
                asset_b = stores[(8, "workspace-b")].add(
                    "long.txt", ("钢筋连接应符合施工要求。\n" * 80).encode("utf-8"))
                factory_calls = []
                embeddings = FakeEmbeddings()

                def store_factory(project_root, workspace_id, pdf_backend, *, user_id, database):
                    key = (user_id, workspace_id)
                    factory_calls.append(key)
                    return stores[key]

                class Resources:
                    pdf_backend = backend

                class FinishRepository(FakeRepository):
                    def __init__(self, claims):
                        super().__init__(claims)
                        self.store_close_counts_at_finish = []

                    def finish(self, worker_instance, task_id, attempt_token, status,
                               *, result=None, error=None):
                        self.store_close_counts_at_finish.append(
                            (stores[(7, "workspace-a")].close_calls,
                             stores[(8, "workspace-b")].close_calls))
                        return super().finish(worker_instance, task_id, attempt_token, status,
                                              result=result, error=error)

                claims = [
                    _claim("a-first", user_id=7, workspace_id="workspace-a",
                           attachment_ids=[asset_a["asset_id"]]),
                    _claim("a-follow-up", user_id=7, workspace_id="workspace-a",
                           attachment_ids=[asset_a["asset_id"]]),
                    _claim("b-isolated", user_id=8, workspace_id="workspace-b",
                           attachment_ids=[asset_b["asset_id"]]),
                ]
                repository = FinishRepository(claims)
                settings = Settings(
                    chat_model="synthetic-model", chat_base_url="https://example.invalid/v1",
                    chat_api_key="synthetic-key", chat_max_tokens=256,
                    chat_context_budget_tokens=4096,
                )
                with patch("rag_arch.web.chat_runner.create_chat_model",
                           side_effect=lambda _settings: FakeModel(reply("模拟回答"))), \
                     patch("rag_arch.web.chat_runner.create_embeddings",
                           return_value=embeddings), \
                     patch("rag_arch.web.chat_runner.create_web_search_provider",
                           return_value=None), \
                     patch("rag_arch.runtime_audit.AUDIT_ROOT", root / "audit"):
                    runtime = _runtime(
                        repository, database=object(), settings=settings, resources=Resources(),
                        root=root, document_store_factory=store_factory,
                        concurrency=1, max_cached_document_stores=2,
                    )
                    runtime.run(max_tasks=3)

                self.assertEqual(factory_calls, [(7, "workspace-a"), (8, "workspace-b")])
                self.assertEqual(len(embeddings.document_batches), 2)
                self.assertGreaterEqual(embeddings.query_calls, 3)
                self.assertEqual(repository.store_close_counts_at_finish, [(0, 0)] * 3)
                self.assertEqual([row[2] for row in repository.finishes],
                                 ["completed", "completed", "completed"])
                self.assertEqual(stores[(7, "workspace-a")].close_calls, 1)
                self.assertEqual(stores[(8, "workspace-b")].close_calls, 1)
            finally:
                for store in stores.values():
                    store.close()

    def test_simulation_uses_only_run_demo(self):
        repository = FakeRepository([_claim(simulation=True, duration_seconds=0, fail_at_end=False)])
        with patch("rag_arch.web.demo.run_demo", return_value={"text": "demo", "simulation": True}) as demo:
            runtime = _runtime(repository, simulation=True)
            runtime.run(max_tasks=1)
        demo.assert_called_once()
        self.assertEqual(repository.finishes[-1][3], {"text": "demo", "simulation": True})

    def test_simulation_worker_rejects_model_task_without_running_runner(self):
        calls = []
        repository = FakeRepository([_claim(simulation=False)])
        _runtime(repository, simulation=True,
                 runner=lambda *args: calls.append(args)).run(max_tasks=1)
        self.assertEqual(calls, [])
        self.assertEqual(repository.finishes[-1][2], "failed")
        self.assertEqual(json.loads(repository.finishes[-1][4])["code"],
                         "simulation_mode_mismatch")

    def test_runner_result_is_validated_before_database_finish(self):
        invalid_results = ({}, {"text": 1}, {"text": "nan", "score": float("nan")},
                           {"text": "opaque", "payload": object()})
        for index, bad_result in enumerate(invalid_results):
            with self.subTest(index=index):
                repository = FakeRepository([_claim(f"invalid-{index}")])
                _runtime(repository, runner=lambda *_args, value=bad_result: value).run(max_tasks=1)
                self.assertEqual(repository.finishes[-1][2], "failed")
                self.assertIsNone(repository.finishes[-1][3])
                self.assertIsInstance(repository.finishes[-1][4], str)

    def test_db_gate_stops_following_embedding_batch_and_later_task_can_run(self):
        batches = []

        def runner(request, progress, cancel_event):
            if request["task_id"] == "first":
                for batch in range(2):
                    if cancel_event.is_set():
                        raise CancelledError()
                    batches.append(batch)
                    if batch == 0:
                        # This is the flag the independent heartbeat sets on a
                        # database failure while a provider call is in flight.
                        runtime._database_available.clear()
                    if cancel_event.is_set():
                        raise CancelledError()
                return {"text": "unexpected"}
            return {"text": "second task after reconnection"}

        repository = FakeRepository([_claim("first"), _claim("second")])
        runtime = _runtime(repository, concurrency=1, runner=runner)
        runtime.run(max_tasks=2)
        self.assertEqual(batches, [0])
        self.assertEqual([row[2] for row in repository.finishes], ["failed", "completed"])
        self.assertEqual(repository.finishes[-1][3]["text"], "second task after reconnection")

    def test_embedding_outage_blocks_actual_evidence_index_rerank_http(self):
        import httpx
        from rag_arch.evidence_index import EvidenceIndex
        from rag_arch.reranking import BailianReranker, QWEN3_RERANK
        from rag_arch.standards import StandardCorpus

        requests = []
        real_client = httpx.Client

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"request_id": "mock", "usage": {"total_tokens": 1},
                                             "output": {"results": [
                                                 {"index": 0, "relevance_score": 0.9},
                                                 {"index": 1, "relevance_score": 0.1},
                                             ]}}, request=request)

        def client_factory(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        repository = FakeRepository([_claim()])
        repository.hold_heartbeat = True
        runtime = _runtime(repository, runner=None)

        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                self_outage()
                return [1.0, 0.0]

        def self_outage():
            self.assertTrue(repository.heartbeat_entered.wait(1))
            repository.fail_heartbeats = True
            runtime._database_available.clear()

        corpus = StandardCorpus(clauses=[
            {"clause_id": "GB1/1.1", "std_id": "GB1", "std_no": "GB 1-2024",
             "clause_no": "1.1", "text": "混凝土强度应符合设计要求。"},
            {"clause_id": "GB1/1.2", "std_id": "GB1", "std_no": "GB 1-2024",
             "clause_no": "1.2", "text": "混凝土构件应满足耐久性要求。"},
        ], tables=[])
        index = EvidenceIndex(
            corpus, embeddings=FakeEmbeddings(),
            reranker=BailianReranker(base_url="https://mock.invalid",
                                     endpoint="https://mock.invalid/rerank",
                                     api_key="test-key", model=QWEN3_RERANK),
            candidate_pool=2,
        )

        def runner(request, progress, cancel_event):
            try:
                index.retrieve("混凝土", mode="dense", top_k=1, use_reranker=True)
            finally:
                repository.release_heartbeat.set()
            return {"text": "should not reach a completed answer"}

        runtime.runner = runner
        with patch("rag_arch.reranking.httpx.Client", side_effect=client_factory):
            runtime.run(max_tasks=1)
        self.assertEqual(requests, [])
        self.assertEqual(repository.finishes[-1][2], "failed")
        self.assertEqual(json.loads(repository.finishes[-1][4])["code"],
                         "control_plane_unavailable")

    def test_shared_resources_are_initialized_once_under_concurrent_first_use(self):
        entered = threading.Event()
        release = threading.Event()
        values = []
        factory_calls = []

        class Resources:
            pdf_backend = object()

        def factory(root):
            factory_calls.append(root)
            entered.set()
            release.wait(2)
            return Resources()

        runtime = _runtime(FakeRepository(), resource_factory=factory)
        first = threading.Thread(target=lambda: values.append(runtime._get_resources()))
        second = threading.Thread(target=lambda: values.append(runtime._get_resources()))
        first.start()
        self.assertTrue(entered.wait(1))
        second.start()
        time.sleep(0.04)
        self.assertEqual(len(factory_calls), 1)
        release.set()
        first.join(2)
        second.join(2)
        self.assertEqual(len(factory_calls), 1)
        self.assertIs(values[0], values[1])

    def test_active_tenant_store_is_never_evicted_and_overflow_store_is_task_local(self):
        created = []

        class Resources:
            pdf_backend = object()

        class Store:
            def __init__(self, key):
                self.key = key
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        def factory(root, workspace_id, pdf_backend, *, user_id, database):
            store = Store((user_id, workspace_id))
            created.append(store)
            return store

        runtime = _runtime(FakeRepository(), database=object(), resources=Resources(),
                           document_store_factory=factory,
                           max_cached_document_stores=1)
        first, first_lease = runtime._acquire_document_store(
            _claim("first", user_id=7, workspace_id="workspace-a")["request"])
        overflow, overflow_lease = runtime._acquire_document_store(
            _claim("overflow", user_id=8, workspace_id="workspace-b")["request"])
        self.assertEqual(len(created), 2)
        self.assertEqual(first.close_calls, 0)
        self.assertEqual(overflow.close_calls, 0)
        overflow_lease.release()
        self.assertEqual(overflow.close_calls, 1)
        self.assertEqual(first.close_calls, 0)
        first_lease.release()
        self.assertEqual(first.close_calls, 0)
        runtime._close_tenant_document_stores()
        self.assertEqual(first.close_calls, 1)

    def test_heartbeat_late_terminal_observation_does_not_fence_other_tasks(self):
        a_started = threading.Event()
        b_started = threading.Event()
        release_b = threading.Event()

        def runner(request, progress, cancel_event):
            if request["task_id"] == "A":
                a_started.set()
                if not repository.heartbeat_control_entered.wait(1):
                    raise AssertionError("heartbeat did not snapshot task A")
                return {"text": "A done"}
            b_started.set()
            release_b.wait(2)
            return {"text": "B done"}

        repository = FakeRepository([_claim("A"), _claim("B")])
        repository.pause_heartbeat_control = True
        runtime = _runtime(repository, concurrency=2, runner=runner)
        thread = threading.Thread(target=lambda: runtime.run(max_tasks=2))
        thread.start()
        self.assertTrue(a_started.wait(1))
        self.assertTrue(b_started.wait(1))
        self.assertTrue(repository.heartbeat_control_entered.wait(1))
        deadline = time.monotonic() + 1
        while not any(row[0] == "A" for row in repository.finishes) and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(any(row[0] == "A" for row in repository.finishes))
        repository.release_heartbeat_control.set()
        release_b.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        statuses = {row[0]: row[2] for row in repository.finishes}
        self.assertEqual(statuses, {"A": "completed", "B": "completed"})


class ProcessIdentityTests(unittest.TestCase):
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    def test_current_process_identity_matches_exact_os_pid_and_start_token(self):
        import psutil

        identity = current_process_identity(psutil_module=psutil)
        self.assertEqual(identity.pid, os.getpid())
        self.assertEqual(
            identity.process_start_token,
            process_start_token(psutil.Process(os.getpid()).create_time()),
        )

    class FakePsutil:
        def __init__(self, create_time=100.125, running=True, error=None):
            self.NoSuchProcess = ProcessIdentityTests.NoSuchProcess
            self.AccessDenied = ProcessIdentityTests.AccessDenied
            self.create_time_value = create_time
            self.running = running
            self.error = error

        def Process(self, pid):
            if self.error:
                raise self.error
            owner = self

            class Process:
                def create_time(self):
                    return owner.create_time_value

                def is_running(self):
                    return owner.running

            return Process()

    def test_exact_identity_probe_distinguishes_running_exit_pid_reuse_and_unknown(self):
        self.assertEqual(probe_process_identity(
            "host-a", 1234, "100.125000", local_host="host-a",
            psutil_module=self.FakePsutil()), ProcessProbeState.RUNNING)
        self.assertEqual(probe_process_identity(
            "host-a", 1234, "99.000000", local_host="host-a",
            psutil_module=self.FakePsutil()), ProcessProbeState.EXITED)
        self.assertEqual(probe_process_identity(
            "host-a", 1234, "100.125000", local_host="host-a",
            psutil_module=self.FakePsutil(running=False)), ProcessProbeState.EXITED)
        self.assertEqual(probe_process_identity(
            "host-a", 1234, "100.125000", local_host="host-a",
            psutil_module=self.FakePsutil(error=self.AccessDenied())), ProcessProbeState.UNKNOWN)
        self.assertEqual(probe_process_identity(
            "host-b", 1234, "100.125000", local_host="host-a",
            psutil_module=self.FakePsutil()), ProcessProbeState.REMOTE)

    def test_recovery_skips_remote_running_and_unknown_processes(self):
        repository = FakeRepository()
        repository.process_rows = [
            {"worker_instance": "dead-local", "host_id": "host-a", "pid": 1,
             "process_start_token": "gone", "state": "stale"},
            {"worker_instance": "live-local", "host_id": "host-a", "pid": 2,
             "process_start_token": "live", "state": "stale"},
            {"worker_instance": "remote", "host_id": "host-b", "pid": 3,
             "process_start_token": "remote", "state": "stale"},
        ]

        def probe(host_id, pid, token, *, local_host):
            return ProcessProbeState.EXITED if token == "gone" else ProcessProbeState.RUNNING

        with patch("rag_arch.web.task_worker.local_host_id", return_value="host-a"):
            rows = recover_local_workers(repository, probe=probe)
        self.assertEqual(repository.recovered, [("dead-local", "gone")])
        self.assertEqual([row["state"] for row in rows], [
            ProcessProbeState.EXITED, ProcessProbeState.RUNNING, ProcessProbeState.REMOTE])


class WorkerProcessCliTests(unittest.TestCase):
    def test_simulation_check_is_read_only_and_does_not_load_settings(self):
        class Result:
            def __init__(self, row=None, rows=None):
                self._row = row
                self._rows = rows or []

            def fetchone(self):
                return self._row

            def fetchall(self):
                return self._rows

        class Connection:
            def __init__(self):
                self.closed = False

            def execute(self, statement):
                if "current_schema() AS schema_name" in statement:
                    return Result({"schema_name": "worker_test"})
                if "information_schema.tables" in statement:
                    from rag_arch.web.readiness import REQUIRED_TABLES
                    return Result(rows=[{"table_name": name} for name in REQUIRED_TABLES])
                if statement == "SET TRANSACTION READ ONLY":
                    return Result()
                return Result(rows=[{"version": "001_multiuser.sql"},
                                    {"version": "002_task_workers.sql"},
                                    {"version": "003_shared_login_limits.sql"},
                                    {"version": "004_process_resources.sql"}])

            def close(self):
                self.closed = True

        connection = Connection()

        class Database:
            schema = "worker_test"

            def connect(self):
                return connection

        from rag_arch.web import worker_process
        with patch.object(worker_process, "_database", return_value=Database()), \
             patch("rag_arch.config.load_settings", side_effect=AssertionError("must not load")):
            output = StringIO()
            with redirect_stdout(output):
                result = worker_process.main(["check", "--simulation", "--schema", "worker_test"])
        self.assertEqual(result, 0)
        self.assertTrue(connection.closed)
        self.assertTrue(json.loads(output.getvalue())["ready"])

    def test_simulation_run_does_not_load_settings_and_accepts_temporary_root(self):
        from rag_arch.web import worker_process
        runtime = unittest.mock.Mock()
        temporary_root = Path(self.enterContext(TemporaryDirectory())) / "worker-root"
        with patch.object(worker_process, "_database", return_value=object()), \
             patch.object(worker_process, "_repository", return_value=object()), \
             patch.object(worker_process, "WorkerRuntime", return_value=runtime) as factory, \
             patch("rag_arch.config.load_settings", side_effect=AssertionError("must not load")):
            self.assertEqual(worker_process.main([
                "run", "--simulation", "--database-url", "postgresql://test/isolated",
                "--schema", "only_test", "--root", str(temporary_root),
                "--heartbeat-interval", "0.5", "--lease-seconds", "3", "--max-tasks", "1",
            ]), 0)
        kwargs = factory.call_args.kwargs
        self.assertTrue(kwargs["simulation"])
        self.assertIsNone(kwargs["settings"])
        self.assertEqual(kwargs["root"], temporary_root.resolve())
        runtime.run.assert_called_once_with(max_tasks=1)

    def test_recover_marks_stale_and_reports_model_tasks_kept(self):
        from rag_arch.web import worker_process
        repository = unittest.mock.Mock()
        with patch.object(worker_process, "_database", return_value=object()), \
             patch.object(worker_process, "_repository", return_value=repository), \
             patch("rag_arch.web.resource_recovery.recover_resources", return_value=[]) as recover:
            output = StringIO()
            with redirect_stdout(output):
                result = worker_process.main(["recover"])
        self.assertEqual(result, 0)
        recover.assert_called_once_with(repository, cleanup_owned=False, root=worker_process.ROOT)
        report = json.loads(output.getvalue())
        self.assertEqual(report["local_resource_checks"], [])
        self.assertIn("recovering", report["note"])

    def test_recover_does_not_accept_a_host_override(self):
        from rag_arch.web import worker_process
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            worker_process.main(["recover", "--host-id", "pretend-remote-host"])


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL,
                     "set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL instance")
class PostgreSQLWorkerFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:  # pragma: no cover - web dependency
            raise unittest.SkipTest("psycopg 3 is required for PostgreSQL worker tests") from exc
        cls.psycopg = psycopg
        cls.sql = sql

    def setUp(self):
        from rag_arch.web.db import Database
        from rag_arch.web.task_repository import TaskRepository

        self.schema = f"worker-test-{uuid4().hex}"
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("CREATE SCHEMA {}").format(
                self.sql.Identifier(self.schema)))
        self.addCleanup(self._drop_schema)
        self.database = Database(TEST_DATABASE_URL, schema=self.schema)
        self.database.migrate()
        with self.database.transaction() as connection:
            user = connection.execute(
                """
                INSERT INTO users(username, username_key, password_salt, password_hash,
                                  password_iterations, enabled, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, TRUE, 1, 1) RETURNING id
                """,
                (f"worker-{uuid4().hex[:10]}", uuid4().hex, b"salt", b"hash", 1000),
            ).fetchone()
        self.user_id = int(user["id"])
        self.repository = TaskRepository(self.database)

    def _drop_schema(self):
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                self.sql.Identifier(self.schema)))

    def test_runner_errors_and_invalid_json_results_fail_and_release_slots(self):
        prompts = ("raises", "missing-text", "non-finite", "custom-object")
        task_ids = []
        calls = []

        def runner(request, progress, cancel_event):
            calls.append(request["task_id"])
            if request["prompt"] == "raises":
                raise RuntimeError("synthetic runner failure")
            if request["prompt"] == "missing-text":
                return {}
            if request["prompt"] == "non-finite":
                return {"text": "bad", "score": float("nan")}
            return {"text": "bad", "opaque": object()}

        for index, prompt in enumerate(prompts):
            task = self.repository.submit(
                self.user_id, "chat", f"failure-{index}",
                {"prompt": prompt, "simulation": False},
            )
            task_ids.append(task["task_id"])
            runtime = WorkerRuntime(
                self.repository, database=self.database, runner=runner,
                identity=ProcessIdentity(IDENTITY.host_id, IDENTITY.pid + index,
                                         IDENTITY.process_start_token),
                worker_instance=f"worker-{uuid4().hex}",
                concurrency=1, poll_interval=0.005, heartbeat_interval=0.02,
                lease_seconds=3,
            )
            runtime.run(max_tasks=1)
        self.assertEqual(calls, task_ids)
        for task_id in task_ids:
            task = self.repository.get_task(self.user_id, task_id)
            self.assertEqual(task["status"], "failed")
            self.assertIsNone(task["result"])
            self.assertIsInstance(task["error"], str)
        with self.database.transaction() as connection:
            slot = connection.execute(
                "SELECT active_task_id FROM assistant_slots WHERE user_id = %s AND assistant_id = 'chat'",
                (self.user_id,),
            ).fetchone()
        self.assertIsNone(slot["active_task_id"])

    def test_claim_commit_with_lost_response_does_not_run_or_retry_unknown_task(self):
        from rag_arch.web.task_repository import TaskRepository

        task = self.repository.submit(
            self.user_id, "chat", "claim-response-lost",
            {"prompt": "must not rerun", "simulation": True,
             "duration_seconds": 0, "fail_at_end": False},
        )
        calls = []

        class LostClaimResponseRepository(TaskRepository):
            def __init__(self, database):
                super().__init__(database)
                self.claim_calls = 0

            def claim_next(inner_self, worker_instance):
                inner_self.claim_calls += 1
                claimed = super(LostClaimResponseRepository, inner_self).claim_next(worker_instance)
                if claimed is not None:
                    raise ConnectionError("synthetic lost database response after commit")
                return None

        lost_repo = LostClaimResponseRepository(self.database)
        runtime = WorkerRuntime(
            lost_repo, database=self.database,
            runner=lambda *args: calls.append(args),
            identity=IDENTITY, worker_instance=f"worker-{uuid4().hex}",
            concurrency=1, poll_interval=0.005, heartbeat_interval=0.02,
            lease_seconds=3,
        )
        runtime.run(max_tasks=1)
        persisted = self.repository.get_task(self.user_id, task["task_id"])
        self.assertEqual(lost_repo.claim_calls, 1)
        self.assertEqual(calls, [])
        self.assertEqual(persisted["status"], "running")
        with self.database.transaction() as connection:
            slot = connection.execute(
                "SELECT active_task_id FROM assistant_slots WHERE user_id = %s AND assistant_id = 'chat'",
                (self.user_id,),
            ).fetchone()
        self.assertEqual(slot["active_task_id"], task["task_id"])


if __name__ == "__main__":
    unittest.main()
