"""PostgreSQL and filesystem tests for the local child-resource recovery path.

Tests use random PostgreSQL schemas and fake process probes. They never create,
terminate or signal real child processes and never call model providers.
"""
from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from types import SimpleNamespace
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from rag_arch.resource_supervision import (
    ResourceCleanupUncertain,
    TaskResourceSupervisor,
    resource_owner_descriptor_path,
)
from rag_arch.web.db import Database
from rag_arch.web.resource_recovery import recover_resources, resolve_task
from rag_arch.web.task_repository import TaskRepository
from rag_arch.web.task_service import BusyError
from rag_arch.web.task_worker import ProcessProbeState, local_host_id, process_start_token


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


class _Probe:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def __call__(self, host_id, pid, token, *, local_host=None, psutil_module=None):
        if host_id.casefold() != str(local_host).casefold():
            return ProcessProbeState.REMOTE
        return self.values.get((host_id, pid, token), ProcessProbeState.UNKNOWN)


class _FakeNoSuchProcess(Exception):
    pass


class _FakeAccessDenied(Exception):
    pass


class _FakeTimeoutExpired(Exception):
    pass


class _FakeChildProcess:
    def __init__(self, pid, create_time, ppid, executable):
        self.pid = pid
        self._create_time = create_time
        self._ppid = ppid
        self._executable = executable
        self.alive = True
        self.terminated = False

    def create_time(self):
        return self._create_time

    def ppid(self):
        return self._ppid

    def exe(self):
        return self._executable

    def is_running(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        if self.alive:
            raise _FakeTimeoutExpired()
        return 0


class _FakePsutil:
    NoSuchProcess = _FakeNoSuchProcess
    AccessDenied = _FakeAccessDenied
    ZombieProcess = type("ZombieProcess", (Exception,), {})
    TimeoutExpired = _FakeTimeoutExpired

    def __init__(self, child):
        self.child = child

    def Process(self, pid):
        if pid == self.child.pid:
            return self.child
        # The parent is confirmed exited. No child may be cleaned up if that
        # worker PID currently names a different live process.
        raise self.NoSuchProcess()


@unittest.skipUnless(TEST_DATABASE_URL, "set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL instance")
class PostgreSQLResourceRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise unittest.SkipTest("psycopg 3 is required for resource recovery integration tests") from exc
        cls.psycopg = psycopg
        cls.sql = sql

    def setUp(self):
        self.schema = "resource_recovery_" + uuid4().hex
        self.database = Database(TEST_DATABASE_URL, schema=self.schema)
        self.addCleanup(self._drop_schema)
        self.assertEqual(self.database.migrate(), [
            "001_multiuser.sql", "002_task_workers.sql",
            "003_shared_login_limits.sql", "004_process_resources.sql",
        ])
        self.repo = TaskRepository(self.database)
        username = "recovery-" + uuid4().hex[:10]
        with self.database.transaction() as connection:
            row = connection.execute(
                "INSERT INTO users(username, username_key, password_salt, password_hash, "
                "password_iterations, enabled, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 1000, TRUE, %s, %s) RETURNING id",
                (username, username.lower(),
                 b"salt", b"digest", time.time(), time.time()),
            ).fetchone()
            self.user_id = int(row["id"])
        self.temp_root = tempfile.TemporaryDirectory(prefix="rag-resource-recovery-")
        self.addCleanup(self.temp_root.cleanup)
        self.root = Path(self.temp_root.name).resolve()
        self.local_host = local_host_id()
        self.worker = self._new_worker()

    def _drop_schema(self):
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(
                self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    self.sql.Identifier(self.schema)
                )
            )

    def _new_worker(self):
        worker = {
            "worker_instance": uuid4().hex,
            "host_id": self.local_host if hasattr(self, "local_host") else local_host_id(),
            "pid": os.getpid(),
            "process_start_token": "worker-start-" + uuid4().hex,
        }
        self.repo.register_worker(**worker)
        return worker

    def _task(self, *, simulation=True, assistant="chat"):
        inputs = {
            "prompt": "resource recovery",
            "duration_seconds": 0.0 if simulation else None,
            "simulation": simulation,
            "attachment_ids": [],
            "registered_plan_id": None,
            "plan_action": "question",
            "use_embeddings": False,
            "max_steps": None,
            "max_tool_calls": None,
        }
        return self.repo.submit(self.user_id, assistant, uuid4().hex, inputs)

    def _begin(self, *, simulation=True, kind="pdf", owner_path=None):
        task = self._task(simulation=simulation)
        claim = self.repo.claim_next(self.worker["worker_instance"])
        self.assertEqual(claim["task_id"], task["task_id"])
        supervisor = TaskResourceSupervisor(
            self.repo, self.worker["worker_instance"], task["task_id"],
            SimpleNamespace(**{
                "host_id": self.worker["host_id"], "pid": self.worker["pid"],
                "process_start_token": self.worker["process_start_token"],
            }), root=self.root,
        )
        handle = supervisor.begin(kind, "task", owner_path)
        child_pid = self.worker["pid"] + 1000
        child_token = f"{123.456789 + len(task['task_id']) / 1_000_000:.6f}"
        executable_name = "WINWORD.EXE" if kind == "word" else "owned-child.exe"
        executable = str((self.root / "bin" / executable_name).absolute())
        descriptor_path = supervisor._write_owner_descriptor(
            handle, pid=child_pid, process_start_token=child_token,
            executable=executable, owned=True,
        )
        self.repo.resource_unknown(self.worker["worker_instance"], handle.resource_id, "attach_db_unavailable")
        return task, claim, handle, descriptor_path, child_pid, child_token, executable

    def _exit_parent(self):
        self.repo.mark_stale_workers(now=time.time() + 120)
        self.assertEqual(
            self.repo.recover_confirmed_exit(
                self.worker["worker_instance"], self.worker["process_start_token"],
            ),
            [],
        )

    def _probe(self, child_pid, child_token, result=ProcessProbeState.EXITED):
        return _Probe({
            (self.worker["host_id"], self.worker["pid"], self.worker["process_start_token"]): ProcessProbeState.EXITED,
            (self.worker["host_id"], child_pid, child_token): result,
        })

    def test_exact_descriptor_backfills_exit_identity_and_audits_file_evidence(self):
        task, _claim, handle, _path, pid, token, _exe = self._begin()
        self._exit_parent()
        outcomes = recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        worker_report = next(item for item in outcomes if item.get("kind") == "worker_resources")
        self.assertEqual(worker_report["resources"][0]["action"], "identity_backfilled_exited")
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual((resource["pid"], resource["process_start_token"], resource["state"]),
                         (pid, token, "exited"))
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "failed")
        with self.database.transaction() as connection:
            audit = connection.execute(
                "SELECT owner_descriptor_path, owner_nonce, probe_state, evidence "
                "FROM process_resource_identity_audit WHERE resource_id = %s",
                (handle.resource_id,),
            ).fetchone()
        self.assertEqual(audit["probe_state"], "exited")
        self.assertEqual(audit["owner_descriptor_path"], resource["owner_descriptor_path"])
        self.assertEqual(audit["owner_nonce"], resource["owner_nonce"])
        self.assertEqual(audit["evidence"]["descriptor"]["resource"]["pid"], pid)
        self.assertIn("file_identity", audit["evidence"])

    def test_parallel_first_descriptor_writes_create_private_directories_safely(self):
        self.repo.set_global_limit(2)
        tasks = [
            self._task(assistant="chat"),
            self._task(assistant="standards"),
        ]
        claims = [self.repo.claim_next(self.worker["worker_instance"]) for _ in range(2)]
        self.assertEqual({claim["task_id"] for claim in claims}, {task["task_id"] for task in tasks})
        handles = []
        supervisors = []
        for claim in claims:
            supervisor = TaskResourceSupervisor(
                self.repo, self.worker["worker_instance"], claim["task_id"],
                SimpleNamespace(**{
                    "host_id": self.worker["host_id"], "pid": self.worker["pid"],
                    "process_start_token": self.worker["process_start_token"],
                }), root=self.root,
            )
            handles.append(supervisor.begin("pdf", "task", None))
            supervisors.append(supervisor)

        def write(index):
            return supervisors[index]._write_owner_descriptor(
                handles[index], pid=self.worker["pid"] + 2000 + index,
                process_start_token=f"child-{index}-token",
                executable=str(self.root / "bin" / f"renderer-{index}.exe"),
                owned=True,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(write, range(2)))
        self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertNotEqual(paths[0], paths[1])

    def test_symlinked_owner_root_is_rejected_before_external_directory_side_effects(self):
        task = self._task()
        claim = self.repo.claim_next(self.worker["worker_instance"])
        self.assertEqual(claim["task_id"], task["task_id"])
        supervisor = TaskResourceSupervisor(
            self.repo, self.worker["worker_instance"], task["task_id"],
            SimpleNamespace(**{
                "host_id": self.worker["host_id"], "pid": self.worker["pid"],
                "process_start_token": self.worker["process_start_token"],
            }), root=self.root,
        )
        handle = supervisor.begin("pdf", "task", None)
        runtime = self.root / "data" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        outside = self.root / "outside-owner-root"
        outside.mkdir()
        trusted = runtime / "resource_owners"
        try:
            os.symlink(outside, trusted, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable on this host: {type(exc).__name__}")
        with self.assertRaises(ResourceCleanupUncertain):
            supervisor._write_owner_descriptor(
                handle, pid=self.worker["pid"] + 3000,
                process_start_token="child-token", executable=str(self.root / "renderer.exe"),
            )
        self.assertFalse((outside / self.worker["worker_instance"]).exists())

    def test_descriptor_cannot_claim_another_parent_identity(self):
        task, _claim, handle, path, pid, token, _exe = self._begin()
        descriptor = json.loads(Path(path).read_text(encoding="utf-8"))
        descriptor["worker"]["process_start_token"] = "different-parent"
        Path(path).write_text(json.dumps(descriptor), encoding="utf-8")
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertIsNone(resource["pid"])
        self.assertFalse(self.repo.task_has_unconfirmed_resources(task["task_id"]) is False)
        with self.database.transaction() as connection:
            count = connection.execute(
                "SELECT count(*) AS count FROM process_resource_identity_audit WHERE resource_id = %s",
                (handle.resource_id,),
            ).fetchone()["count"]
        self.assertEqual(count, 0)

    def test_descriptor_cannot_claim_remote_child_or_different_owner_path(self):
        task, _claim, handle, path, pid, token, _exe = self._begin()
        descriptor = json.loads(Path(path).read_text(encoding="utf-8"))
        descriptor["resource"]["host_id"] = "remote-host"
        Path(path).write_text(json.dumps(descriptor), encoding="utf-8")
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertIsNone(resource["pid"])

    def test_descriptor_owner_path_must_match_the_registered_owner_file(self):
        task, _claim, _handle, path, pid, token, _exe = self._begin()
        descriptor = json.loads(Path(path).read_text(encoding="utf-8"))
        descriptor["owner_path"] = str(self.root / "elsewhere.json")
        Path(path).write_text(json.dumps(descriptor), encoding="utf-8")
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertIsNone(resource["pid"])
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "recovering")

    def test_missing_descriptor_leaves_unknown_without_identity_or_slot_release(self):
        task = self._task(simulation=True)
        claim = self.repo.claim_next(self.worker["worker_instance"])
        self.assertEqual(claim["task_id"], task["task_id"])
        supervisor = TaskResourceSupervisor(
            self.repo, self.worker["worker_instance"], task["task_id"],
            SimpleNamespace(**{
                "host_id": self.worker["host_id"], "pid": self.worker["pid"],
                "process_start_token": self.worker["process_start_token"],
            }), root=self.root,
        )
        handle = supervisor.begin("word", "task", None)
        self.repo.resource_unknown(self.worker["worker_instance"], handle.resource_id, "unknown_child")
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(self.worker["pid"] + 1000, "missing"))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertIsNone(resource["pid"])
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "recovering")
        with self.assertRaises(BusyError):
            self._task()

    def test_running_child_identity_is_audited_but_not_released_without_cleanup(self):
        task, _claim, handle, _path, pid, token, _exe = self._begin()
        self._exit_parent()
        child = _FakeChildProcess(pid, float(token), self.worker["pid"], _exe)
        recover_resources(self.repo, root=self.root,
                          probe=self._probe(pid, token, ProcessProbeState.RUNNING),
                          psutil_module=_FakePsutil(child))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertEqual(resource["pid"], pid)
        self.assertTrue(self.repo.task_has_unconfirmed_resources(task["task_id"]))
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "recovering")
        with self.database.transaction() as connection:
            audit = connection.execute(
                "SELECT probe_state FROM process_resource_identity_audit WHERE resource_id = %s",
                (handle.resource_id,),
            ).fetchone()
        self.assertEqual(audit["probe_state"], "running")

    def test_unknown_probe_backfills_identity_but_waits_until_a_later_exact_exit(self):
        task, _claim, handle, _path, pid, token, _exe = self._begin()
        self._exit_parent()
        recover_resources(self.repo, root=self.root,
                          probe=self._probe(pid, token, ProcessProbeState.UNKNOWN))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "unknown")
        self.assertEqual(resource["pid"], pid)
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token, ProcessProbeState.EXITED))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        self.assertEqual(resource["state"], "exited")
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "failed")
        with self.database.transaction() as connection:
            count = connection.execute(
                "SELECT count(*) AS count FROM process_resource_identity_audit WHERE resource_id = %s",
                (handle.resource_id,),
            ).fetchone()["count"]
        self.assertEqual(count, 1)

    def test_cleanup_requires_owned_descriptor_exact_parent_child_and_word_owner_path(self):
        owner_file = self.root / "data" / "runtime" / "document_pages" / uuid4().hex / "pages.owner.json"
        owner_file.parent.mkdir(parents=True)
        task, _claim, handle, _path, pid, token, executable = self._begin(
            simulation=True, kind="word", owner_path=str(owner_file),
        )
        child = _FakeChildProcess(pid, float(token), self.worker["pid"], executable)
        ticks = str(int(Decimal(token) * Decimal(10000000)
                       + Decimal(621355968000000000)))
        owner_file.write_text(json.dumps({
            "pid": pid, "owned": True, "start_ticks": ticks,
            "executable": executable,
        }), encoding="utf-8")
        if os.name != "nt":
            owner_file.chmod(0o600)
        self._exit_parent()
        def probe(host_id, process_id, process_token_value, *, local_host=None, psutil_module=None):
            if process_id == self.worker["pid"] and process_token_value == self.worker["process_start_token"]:
                return ProcessProbeState.EXITED
            if process_id == pid and process_token_value == token:
                return ProcessProbeState.RUNNING if child.alive else ProcessProbeState.EXITED
            return ProcessProbeState.UNKNOWN

        def stop_exact(worker, resource, **_kwargs):
            self.assertEqual(worker["worker_instance"], self.worker["worker_instance"])
            self.assertEqual(resource["pid"], pid)
            self.assertEqual(resource["process_start_token"], token)
            self.assertEqual(resource["executable"], executable)
            child.terminated = True
            child.alive = False
            return ProcessProbeState.EXITED

        with patch("rag_arch.web.resource_recovery._terminate_exact_child", side_effect=stop_exact):
            outcomes = recover_resources(self.repo, cleanup_owned=True, root=self.root,
                                         probe=probe, psutil_module=_FakePsutil(child))
        worker_report = next(item for item in outcomes if item.get("kind") == "worker_resources")
        self.assertEqual(worker_report["resources"][0]["action"], "owned_child_terminated")
        self.assertTrue(child.terminated)
        self.assertEqual(self.repo.resources_for_worker(self.worker["worker_instance"])[0]["state"], "exited")
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "failed")

    def test_cleanup_unknown_native_exit_result_keeps_task_slot(self):
        task, _claim, handle, _path, pid, token, executable = self._begin()
        child = _FakeChildProcess(pid, float(token), self.worker["pid"] + 7, executable)
        self._exit_parent()
        probe = _Probe({
            (self.worker["host_id"], self.worker["pid"], self.worker["process_start_token"]): ProcessProbeState.EXITED,
            (self.worker["host_id"], pid, token): ProcessProbeState.RUNNING,
        })
        with patch("rag_arch.web.resource_recovery._terminate_exact_child",
                   return_value=ProcessProbeState.UNKNOWN):
            outcomes = recover_resources(self.repo, cleanup_owned=True, root=self.root,
                                         probe=probe, psutil_module=_FakePsutil(child))
        worker_report = next(item for item in outcomes if item.get("kind") == "worker_resources")
        self.assertEqual(worker_report["resources"][0]["action"], "cleanup_not_confirmed")
        self.assertFalse(child.terminated)
        self.assertEqual(self.repo.resources_for_worker(self.worker["worker_instance"])[0]["state"], "unknown")
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "recovering")

    def test_resolve_reprobes_worker_and_children_and_writes_only_failed_terminal_audit(self):
        task, claim, handle, _path, pid, token, _exe = self._begin(simulation=False)
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        proof = {
            "operator": "integration-admin",
            "reason": "external model request was inspected and confirmed ended",
            "external_work_confirmed_ended": True,
            "worker": self.worker,
            "resources": [{
                "resource_id": handle.resource_id, "host_id": resource["host_id"],
                "pid": resource["pid"], "process_start_token": resource["process_start_token"],
            }],
        }
        running_probe = self._probe(pid, token, ProcessProbeState.RUNNING)
        self.assertFalse(resolve_task(self.repo, task["task_id"], proof, root=self.root, probe=running_probe))
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "recovering")
        self.assertTrue(resolve_task(self.repo, task["task_id"], proof, root=self.root,
                                     probe=self._probe(pid, token)))
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "failed")
        with self.database.transaction() as connection:
            audit = connection.execute(
                "SELECT terminal_status, proof FROM task_resolution_audit WHERE task_id = %s",
                (task["task_id"],),
            ).fetchone()
        self.assertEqual(audit["terminal_status"], "failed")
        self.assertEqual(audit["proof"], proof)

    def test_resolve_uses_durable_descriptor_after_normal_word_owner_file_is_removed(self):
        owner_file = self.root / "data" / "runtime" / "document_pages" / uuid4().hex / "pages.owner.json"
        owner_file.parent.mkdir(parents=True)
        task, _claim, handle, _path, pid, token, executable = self._begin(
            simulation=False, kind="word", owner_path=str(owner_file),
        )
        ticks = str(int(Decimal(token) * Decimal(10000000)
                        + Decimal(621355968000000000)))
        owner_file.write_text(json.dumps({
            "pid": pid, "owned": True, "start_ticks": ticks,
            "executable": executable,
        }), encoding="utf-8")
        self._exit_parent()
        recover_resources(self.repo, root=self.root, probe=self._probe(pid, token))
        resource = self.repo.resources_for_worker(self.worker["worker_instance"])[0]
        owner_file.unlink()  # normal Word cleanup removes its transient descriptor
        proof = {
            "operator": "integration-admin",
            "reason": "external request checked and ended; Word owner was normally cleaned",
            "external_work_confirmed_ended": True,
            "worker": self.worker,
            "resources": [{
                "resource_id": handle.resource_id, "host_id": resource["host_id"],
                "pid": resource["pid"], "process_start_token": resource["process_start_token"],
            }],
        }
        self.assertTrue(resolve_task(self.repo, task["task_id"], proof, root=self.root,
                                     probe=self._probe(pid, token)))
        self.assertEqual(self.repo.get_task(self.user_id, task["task_id"])["status"], "failed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
