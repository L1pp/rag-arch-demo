"""Safety contracts for task-owned process registration and context propagation."""
import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import psutil

from rag_arch.async_support import AsyncTaskRuntime, run_local
from rag_arch.resource_supervision import (
    ResourceCleanupUncertain,
    TaskResourceSupervisor,
    begin_resource,
    probe_process,
    resource_supervision_scope,
)


class RecordingRepository:
    def __init__(self, *, attach_result=True, use_result=True, fail_begin=False):
        self.attach_result = attach_result
        self.use_result = use_result
        self.fail_begin = fail_begin
        self.begun = []
        self.attached = []
        self.uses = []
        self.unknown = []

    def begin_resource(self, worker_instance, kind, scope, *, task_id=None, owner_path=None,
                       resource_id=None, owner_descriptor_path=None):
        if self.fail_begin:
            raise ConnectionError("synthetic database outage")
        resource_id = resource_id or uuid4().hex
        self.begun.append((worker_instance, kind, scope, task_id, owner_path, resource_id))
        return {"resource_id": resource_id, "owner_descriptor_path": owner_descriptor_path,
                "owner_nonce": uuid4().hex}

    def attach_resource(self, worker_instance, resource_id, **identity):
        self.attached.append((worker_instance, resource_id, identity))
        return self.attach_result

    def use_resource(self, worker_instance, resource_id, task_id):
        self.uses.append((worker_instance, resource_id, task_id))
        return self.use_result

    def resource_unknown(self, worker_instance, resource_id, error_code):
        self.unknown.append((worker_instance, resource_id, error_code))
        return True

    def resource_exited(self, worker_instance, resource_id, *, pid=None,
                        process_start_token=None):
        return True


class ResourceSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag-resource-supervision-")
        self.addCleanup(self.temp.cleanup)
        self.repository = RecordingRepository()
        self.supervisor = TaskResourceSupervisor(
            self.repository, "worker-test", "task-test",
            SimpleNamespace(host_id="host-test", pid=1234,
                            process_start_token="100.125000"), root=self.temp.name)

    def test_attach_and_use_rejections_preserve_unknown_state(self):
        self.repository.attach_result = False
        rejected_attach = self.supervisor.begin("pdf", "task", None)
        with resource_supervision_scope(self.supervisor):
            with self.assertRaises(ResourceCleanupUncertain):
                rejected_attach.attach_process(os.getpid())
        self.assertTrue(self.supervisor.has_unconfirmed)
        self.assertTrue(self.supervisor.retire_requested)
        self.assertEqual(self.repository.unknown[-1][2], "resource_identity_registration_rejected")

        # A new supervisor isolates this second case from the first rejection.
        repository = RecordingRepository(use_result=False)
        supervisor = TaskResourceSupervisor(
            repository, "worker-test-2", "task-test-2",
            SimpleNamespace(host_id="host-test", pid=1234,
                            process_start_token="100.125000"), root=self.temp.name)
        handle = supervisor.begin("word", "task", None)
        with resource_supervision_scope(supervisor):
            with self.assertRaises(ResourceCleanupUncertain):
                handle.used()
        self.assertTrue(supervisor.has_unconfirmed)
        self.assertTrue(supervisor.retire_requested)
        self.assertEqual(len(repository.uses), 1)
        self.assertEqual(repository.uses[0][0], "worker-test-2")
        self.assertEqual(repository.uses[0][2], "task-test-2")
        self.assertEqual(repository.unknown[-1][2], "resource_use_registration_rejected")

    def test_resource_registration_failure_is_conservative(self):
        repository = RecordingRepository(fail_begin=True)
        supervisor = TaskResourceSupervisor(
            repository, "worker-test", "task-test",
            SimpleNamespace(host_id="host-test", pid=1234,
                            process_start_token="100.125000"), root=self.temp.name)

        with self.assertRaises(ResourceCleanupUncertain):
            supervisor.begin("pdf", "task", None)

        self.assertTrue(supervisor.has_unconfirmed)
        self.assertTrue(supervisor.retire_requested)

    def test_pid_reuse_proves_old_identity_exited_but_executable_mismatch_is_unknown(self):
        class Process:
            def create_time(self):
                return 200.0

            def exe(self):
                return str(Path(self_path) / "new-process.exe")

            def is_running(self):
                return True

            def status(self):
                return psutil.STATUS_RUNNING

        self_path = self.temp.name
        with patch.object(psutil, "Process", return_value=Process()):
            self.assertEqual(probe_process(12345, "100.000000"), "exited")
            self.assertEqual(
                probe_process(12345, "200.000000",
                              executable=str(Path(self.temp.name) / "old-process.exe")),
                "unknown",
            )

    def test_supervisor_context_reaches_async_loop_and_bounded_local_thread(self):
        runtime = AsyncTaskRuntime(local_workers=1)
        self.addCleanup(runtime.close)

        async def create_resources():
            on_loop = begin_resource("pdf", scope="task")
            on_local_thread = await run_local(begin_resource, "powershell", scope="task")
            return on_loop, on_local_thread

        with resource_supervision_scope(self.supervisor):
            handles = runtime.run(create_resources())

        self.assertEqual(len(self.repository.begun), 2)
        self.assertTrue(all(handle.supervisor is self.supervisor for handle in handles))
        self.assertEqual([row[1] for row in self.repository.begun], ["pdf", "powershell"])
        self.assertEqual([row[3] for row in self.repository.begun], ["task-test", "task-test"])


if __name__ == "__main__":
    unittest.main()
