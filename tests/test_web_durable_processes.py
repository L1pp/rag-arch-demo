"""Black-box durable-task checks across real API and worker processes.

Run with the project's .venv-web Python and
RAG_ARCH_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:<ephemeral-port>/rag_arch_test?sslmode=disable.
Every test owns a random schema. API and worker subprocesses receive only that
test URL/schema, use a temporary project root, and run the local simulation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.parse import parse_qsl, unquote, urlsplit
from uuid import uuid4

import httpx

from rag_arch.web.auth import PostgresAuthStore
from rag_arch.web.db import Database
from rag_arch.web.task_repository import TaskRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSISTANTS = ("chat", "standards", "plan")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _validate_test_url(url: str) -> None:
    if not isinstance(url, str) or not url or url != url.strip():
        raise AssertionError("进程验收数据库URL必须是明确的测试连接串。")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        query_params = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise AssertionError("进程验收数据库URL格式无效。") from None
    if (parsed.scheme not in {"postgres", "postgresql"}
            or parsed.hostname != "127.0.0.1"
            or parsed.username != "postgres"
            or parsed.password is not None
            or port is None
            or not 1024 <= port <= 65535
            or parsed.path != "/rag_arch_test"
            or parsed.fragment != ""
            or "#" in url
            or query_params not in ([], [("sslmode", "disable")])):
        raise AssertionError(
            "进程验收只允许显式postgres无密码连接127.0.0.1上的rag_arch_test，"
            "且端口须合法、路径精确、查询仅可为空或单个sslmode=disable。"
        )


class DatabaseURLGuardTests(unittest.TestCase):
    def test_accepts_ci_url_and_the_current_ephemeral_port_url(self):
        _validate_test_url("postgresql://postgres@127.0.0.1:5432/rag_arch_test")
        _validate_test_url("postgresql://postgres@127.0.0.1:64276/rag_arch_test?sslmode=disable")

    def test_rejects_authority_override_query_parameters(self):
        for query in ("host=example.invalid", "dbname=postgres", "port=5432", "user=other"):
            with self.subTest(query=query), self.assertRaises(AssertionError):
                _validate_test_url("postgresql://postgres@127.0.0.1:64276/rag_arch_test?" + query)

    def test_rejects_repeated_or_additional_query_parameters(self):
        for query in (
            "sslmode=disable&sslmode=disable",
            "sslmode=disable&host=127.0.0.1",
            "application_name=test",
        ):
            with self.subTest(query=query), self.assertRaises(AssertionError):
                _validate_test_url("postgresql://postgres@127.0.0.1:64276/rag_arch_test?" + query)

    def test_rejects_nonexact_paths_fragments_and_passwords(self):
        for url in (
            "postgresql://postgres@127.0.0.1:64276//rag_arch_test",
            "postgresql://postgres@127.0.0.1:64276///rag_arch_test",
            "postgresql://postgres@127.0.0.1:64276/rag_arch_test#",
            "postgresql://postgres@127.0.0.1:64276/rag_arch_test#fragment",
            "postgresql://postgres:secret@127.0.0.1:64276/rag_arch_test",
        ):
            with self.subTest(url=url), self.assertRaises(AssertionError):
                _validate_test_url(url)

class DurableTaskProcessTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.children: list[dict] = []
        self.clients: list[httpx.Client] = []
        self.temp = TemporaryDirectory(prefix="rag-durable-process-root-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log_temp = TemporaryDirectory(prefix="rag-durable-process-logs-")
        self.addCleanup(self.log_temp.cleanup)
        self.log_root = Path(self.log_temp.name)

        self.url = os.environ.get("RAG_ARCH_TEST_DATABASE_URL", "").strip()
        if not self.url:
            self.skipTest(
                "设置 RAG_ARCH_TEST_DATABASE_URL 指向专用本机 PostgreSQL 后运行进程验收。"
            )
        _validate_test_url(self.url)
        try:
            import psutil
        except ImportError as exc:
            self.skipTest(f"真实进程验收需要psutil：{exc}")
        self.psutil = psutil
        self.schema = "process_" + uuid4().hex
        self.database = Database(self.url, schema=self.schema,
                                 connect_timeout=3, statement_timeout_ms=8000)
        self.repository = TaskRepository(self.database)
        self.addCleanup(self._drop_schema)
        # The parent test process is the only process allowed to migrate this
        # test schema. API and worker subprocesses must only use it as-is.
        self.database.migrate()
        self.auth = PostgresAuthStore(self.database, password_iterations=1500)
        self.auth.create_user("Alice", "alice-test-password-1")
        self.auth.create_user("Bob", "bob-test-password-1")
        self.user_ids = {
            "Alice": self.auth.verify_identity("alice", "alice-test-password-1").user_id,
            "Bob": self.auth.verify_identity("BOB", "bob-test-password-1").user_id,
        }
        # Register cleanup after schema cleanup so child processes are always
        # stopped and waited before the schema is dropped.
        self.addCleanup(self._cleanup_processes)

    def _drop_schema(self):
        import psycopg
        from psycopg import sql

        live = self._live_child_processes()
        if live:
            raise AssertionError(
                f"仍有测试子进程可能使用schema {self.schema}，保留schema不删除：{live}"
            )

        with psycopg.connect(self.url, autocommit=True, connect_timeout=3) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(self.schema)
                )
            )
            exists = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = %s)",
                (self.schema,),
            ).fetchone()[0]
        if exists:
            raise AssertionError(f"测试schema未能清理：{self.schema}")

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # These explicit values override any project or shell environment.
        # The helper and worker both validate/use this dedicated test target.
        env["RAG_ARCH_TEST_DATABASE_URL"] = self.url
        env["RAG_ARCH_DATABASE_URL"] = self.url
        env["RAG_ARCH_DATABASE_SCHEMA"] = self.schema
        env["RAG_ARCH_AUTH_ALLOWED_ORIGINS"] = ""
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _spawn(self, kind: str, name: str, args: list[str]) -> dict:
        log_path = self.log_root / f"{name}-{uuid4().hex}.log"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                args,
                cwd=PROJECT_ROOT,
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=flags,
            )
        child = {
            "kind": kind,
            "name": name,
            "process": process,
            "log": log_path,
            "processes": {},
            "command": list(args),
        }
        try:
            child["popen_identity"] = self._identity_for_pid(process.pid)
            self._remember_process(child, child["popen_identity"], "popen")
            for _ in range(40):
                self._capture_process_tree(child)
                if len(child["processes"]) > 1 or process.poll() is not None:
                    break
                time.sleep(0.005)
        except self.psutil.Error:
            # Very short-lived utilities may exit between Popen and psutil.
            pass
        self.children.append(child)
        return child

    def _identity_for_pid(self, pid: int) -> dict:
        process = self.psutil.Process(pid)
        from rag_arch.web.task_worker import process_start_token

        return {
            "pid": int(pid),
            "start_token": process_start_token(process.create_time()),
        }

    def _live_child_processes(self) -> list[tuple]:
        live = []
        for child in self.children:
            self._capture_process_tree(child)
            for identity in child.get("processes", {}).values():
                try:
                    actual = self._identity_for_pid(int(identity["pid"]))
                except self.psutil.NoSuchProcess:
                    continue
                except self.psutil.Error as exc:
                    live.append((child["kind"], child["name"], identity["pid"],
                                 "identity-unavailable", type(exc).__name__))
                    continue
                if actual["start_token"] == identity["start_token"]:
                    live.append((child["kind"], child["name"], identity["pid"],
                                 identity["start_token"]))
            process = child["process"]
            if process.poll() is None:
                live.append((child["kind"], child["name"], process.pid,
                             "popen-still-running"))
            if child["kind"] == "worker" and hasattr(self, "repository"):
                try:
                    row = self.repository.get_worker(child["worker_instance"])
                except Exception as exc:
                    live.append(("worker", child["worker_instance"],
                                 "identity-unavailable", type(exc).__name__))
                    continue
                if row is not None:
                    identity = {"pid": int(row["pid"]),
                                "start_token": str(row["process_start_token"])}
                    try:
                        actual = self._identity_for_pid(identity["pid"])
                    except self.psutil.NoSuchProcess:
                        continue
                    except self.psutil.Error as exc:
                        live.append(("worker", child["worker_instance"], identity["pid"],
                                     "identity-unavailable", type(exc).__name__))
                        continue
                    if actual["start_token"] == identity["start_token"]:
                        live.append(("worker", child["worker_instance"], identity["pid"],
                                     identity["start_token"]))
        return live

    def _remember_process(self, child: dict, identity: dict, role: str) -> None:
        if identity is None:
            return
        child["processes"].setdefault(
            int(identity["pid"]),
            {"pid": int(identity["pid"]), "start_token": identity["start_token"],
             "role": role},
        )

    def _same_process(self, identity: dict) -> bool:
        if not identity:
            return False
        try:
            actual = self._identity_for_pid(int(identity["pid"]))
        except self.psutil.Error:
            return False
        return actual["start_token"] == identity["start_token"]

    def _capture_process_tree(self, child: dict) -> None:
        """Remember only the exact Popen process and its live descendants."""
        identity = child.get("popen_identity")
        if not self._same_process(identity):
            return
        try:
            root_process = self.psutil.Process(int(identity["pid"]))
            descendants = root_process.children(recursive=True)
        except self.psutil.Error:
            return
        for process in descendants:
            try:
                process_identity = self._identity_for_pid(process.pid)
            except self.psutil.Error:
                continue
            self._remember_process(child, process_identity, "descendant")

    def _is_descendant_of_popen(self, child: dict, pid: int) -> bool:
        root_identity = child.get("popen_identity")
        if not root_identity:
            return False
        if int(root_identity["pid"]) == int(pid):
            return self._same_process(root_identity)
        remembered = child.get("processes", {}).get(int(pid))
        if remembered and remembered.get("role") in {"popen", "descendant"}:
            return self._same_process(remembered)
        try:
            process = self.psutil.Process(int(pid))
            from rag_arch.web.task_worker import process_start_token

            for parent in process.parents():
                if parent.pid != int(root_identity["pid"]):
                    continue
                if process_start_token(parent.create_time()) == root_identity["start_token"]:
                    return True
            return False
        except self.psutil.Error:
            return False

    def _remember_worker_identity(self, worker: dict, row: dict | None = None) -> dict:
        row = row or self.repository.get_worker(worker["worker_instance"])
        self.assertIsNotNone(row, f"worker未注册：{worker['worker_instance']}")
        identity = {
            "pid": int(row["pid"]),
            "start_token": str(row["process_start_token"]),
            "host_id": str(row["host_id"]),
        }
        actual = self._identity_for_pid(identity["pid"])
        self.assertEqual(
            actual["start_token"], identity["start_token"],
            f"worker数据库启动标识必须匹配真实OS进程：instance={worker['worker_instance']}, "
            f"DB PID={identity['pid']}, Popen launcher PID={worker['process'].pid}",
        )
        self._capture_process_tree(worker)
        if not self._is_descendant_of_popen(worker, identity["pid"]):
            process = self.psutil.Process(identity["pid"])
            raise AssertionError(
                f"拒绝把非此Popen后代的OS进程作为worker：instance={worker['worker_instance']}, "
                f"DB PID={identity['pid']}, parent PID={process.ppid()}, "
                f"Popen PID={worker['process'].pid}, tree={sorted(worker['processes'])}"
            )
        self._remember_process(worker, identity, "worker")
        worker["worker_identity"] = identity
        return identity

    def _assert_worker_descends_from_popen(self, worker: dict, row: dict) -> None:
        identity = self._remember_worker_identity(worker, row)
        self.assertTrue(
            self._is_descendant_of_popen(worker, identity["pid"]),
            f"DB worker不是此Popen启动的后代：instance={worker['worker_instance']}, "
            f"DB PID={identity['pid']}, Popen PID={worker['process'].pid}, "
            f"process tree={sorted(worker['processes'])}, log={self._read_log(worker)}",
        )

    def _find_api_server_identity(self, api: dict) -> dict | None:
        self.assertEqual(api.get("schema"), self.schema)
        self.assertIn(str(self.root), api["command"])
        self.assertIn(str(api["port"]), api["command"])
        self._capture_process_tree(api)
        port = int(api["port"])
        for pid, identity in list(api["processes"].items()):
            if not self._same_process(identity):
                continue
            try:
                connections = self.psutil.Process(pid).net_connections(kind="inet")
            except self.psutil.Error:
                continue
            if any(connection.status == self.psutil.CONN_LISTEN
                   and connection.laddr and connection.laddr.port == port
                   for connection in connections):
                api["server_identity"] = identity
                return identity
        return None

    def _terminate_exact_identity(self, identity: dict | None, timeout: float = 5) -> None:
        if not identity or not self._same_process(identity):
            return
        try:
            process = self.psutil.Process(int(identity["pid"]))
            actual = self._identity_for_pid(int(identity["pid"]))
            if actual["start_token"] != identity["start_token"]:
                return
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except self.psutil.TimeoutExpired:
                if self._same_process(identity):
                    exact = self.psutil.Process(int(identity["pid"]))
                    if self._identity_for_pid(int(identity["pid"]))["start_token"] == identity["start_token"]:
                        exact.kill()
                        exact.wait(timeout=timeout)
        except self.psutil.NoSuchProcess:
            return

    @staticmethod
    def _free_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_api(self) -> dict:
        port = self._free_loopback_port()
        child = self._spawn(
            "api", "api",
            [sys.executable, "-m", "tests.helpers.durable_api_server",
             "--root", str(self.root), "--port", str(port)],
        )
        child["port"] = port
        child["base_url"] = f"http://127.0.0.1:{port}"
        child["schema"] = self.schema
        api = child
        deadline = time.monotonic() + 12
        last_error = None
        while time.monotonic() < deadline:
            if child["process"].poll() is not None:
                self.fail(f"API进程提前退出，日志：\n{self._read_log(child)}")
            self._capture_process_tree(child)
            try:
                response = httpx.get(api["base_url"] + "/api/health", timeout=0.5)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    server = self._find_api_server_identity(api)
                    if server is not None and self._is_descendant_of_popen(api, server["pid"]):
                        api["server_identity"] = server
                        return api
                    last_error = "health已就绪，但没有确认由此Popen启动的监听PID"
                    time.sleep(0.05)
                    continue
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
            time.sleep(0.05)
        self.fail(f"API未在期限内就绪（{last_error}），日志：\n{self._read_log(child)}")

    def _stop_api(self, api: dict, timeout: float = 5) -> None:
        identity = api.get("server_identity") or self._find_api_server_identity(api)
        self._terminate_exact_identity(identity, timeout=timeout)
        process = api["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=timeout)
        else:
            process.wait(timeout=0)
        self._wait_until(
            lambda: not self._api_health_reachable(api), timeout,
            f"停止API监听PID后端口仍可访问：{api['port']}",
        )

    def _api_health_reachable(self, api: dict) -> bool:
        try:
            response = httpx.get(api["base_url"] + "/api/health", timeout=0.25)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _login(self, api: dict, username: str) -> tuple[httpx.Client, dict]:
        password = "alice-test-password-1" if username == "Alice" else "bob-test-password-1"
        client = httpx.Client(follow_redirects=False, timeout=httpx.Timeout(4.0))
        self.clients.append(client)
        origin = api["base_url"]
        response = client.post(
            origin + "/login",
            data={"username": username, "password": password},
            headers={"Origin": origin},
        )
        self.assertEqual(response.status_code, 303, response.text)
        workspace_response = client.get(origin + "/api/workspace")
        self.assertEqual(workspace_response.status_code, 200, workspace_response.text)
        return client, workspace_response.json()

    def _submit(self, api: dict, client: httpx.Client, workspace: dict, assistant: str,
                *, request_id: str | None = None, prompt: str = "local simulation",
                duration_seconds: float = 1.5, attachment_ids: list[str] | None = None):
        body = {
            "workspace_id": workspace["workspace_id"],
            "assistant_id": assistant,
            "request_id": request_id or uuid4().hex,
            "prompt": prompt,
            "duration_seconds": duration_seconds,
        }
        if attachment_ids is not None:
            body["attachment_ids"] = attachment_ids
        return client.post(
            api["base_url"] + "/api/tasks",
            json=body,
            headers={"Origin": api["base_url"]},
        )

    def _upload_plan(self, api: dict, client: httpx.Client, workspace: dict) -> str:
        response = client.post(
            api["base_url"] + "/api/uploads",
            params={"workspace_id": workspace["workspace_id"], "filename": "plan.txt"},
            content=b"Synthetic local simulation input. No model will read this file.",
            headers={"Origin": api["base_url"], "Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["asset_id"]

    def _start_worker(self, *, max_tasks: int | None = 1,
                      worker_instance: str | None = None) -> dict:
        worker_instance = worker_instance or "process-worker-" + uuid4().hex
        args = [
            sys.executable, "-m", "rag_arch.web.worker_process", "run",
            "--simulation", "--concurrency", "1", "--poll-interval", "0.05",
            "--heartbeat-interval", "0.5", "--lease-seconds", "3",
            "--root", str(self.root), "--database-url", self.url,
            "--schema", self.schema, "--worker-instance", worker_instance,
        ]
        if max_tasks is not None:
            args.extend(["--max-tasks", str(max_tasks)])
        child = self._spawn("worker", "worker", args)
        child["worker_instance"] = worker_instance
        child["schema"] = self.schema
        return child

    def _start_idle_workers(self, count: int) -> list[dict]:
        workers = [self._start_worker(max_tasks=1) for _ in range(count)]
        self._wait_until(
            lambda: all(self._worker_registered(item) for item in workers),
            8,
            "worker进程没有全部注册到数据库",
        )
        for worker in workers:
            row = self.repository.get_worker(worker["worker_instance"])
            self.assertEqual(row["state"], "active")
            self._assert_worker_descends_from_popen(worker, row)
        return workers

    def _worker_registered(self, worker: dict) -> bool:
        row = self.repository.get_worker(worker["worker_instance"])
        if row is None:
            return False
        try:
            self._assert_worker_descends_from_popen(worker, row)
        except (AssertionError, self.psutil.Error):
            return False
        return True

    def _wait_until(self, predicate, timeout: float, message: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(0.03)
        self.fail(message)

    def _task_row(self, task_id: str) -> dict:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT task_id, user_id, workspace_id, assistant_id, request_id, status, "
                "worker_instance, attempt_token, started_at, finished_at "
                "FROM tasks WHERE task_id = %s", (task_id,),
            ).fetchone()
        self.assertIsNotNone(row, f"task row missing: {task_id}")
        return dict(row)

    def _task_events(self, task_id: str) -> list[dict]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT seq, event_type, payload FROM workspace_events "
                "WHERE task_id = %s ORDER BY seq", (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _wait_for_terminal(self, task_id: str, timeout: float = 15) -> dict:
        user_id = self._task_row(task_id)["user_id"]
        self._wait_until(
            lambda: self.repository.get_task(user_id, task_id)["status"] in TERMINAL_STATUSES,
            timeout,
            f"task没有在{timeout}秒内进入终态：{task_id}",
        )
        return self._task_row(task_id)

    def _read_log(self, child: dict) -> str:
        try:
            return child["log"].read_text(encoding="utf-8", errors="replace")[-6000:]
        except OSError:
            return "<log unavailable>"

    def _stop_child(self, child: dict, timeout: float = 5) -> None:
        process = child["process"]
        self._capture_process_tree(child)
        if child["kind"] == "api":
            self._stop_api(child, timeout=timeout)
        elif child["kind"] == "worker":
            row = self.repository.get_worker(child["worker_instance"])
            if row is not None:
                try:
                    identity = self._remember_worker_identity(child, row)
                except (AssertionError, self.psutil.Error):
                    identity = None
                self._terminate_exact_identity(identity, timeout=timeout)

        # Terminate only descendants previously captured from this exact
        # Popen root or verified through the database worker identity. Tokens
        # prevent a reused PID from targeting an unrelated process.
        root_pid = (child.get("popen_identity") or {}).get("pid")
        for identity in reversed(list(child.get("processes", {}).values())):
            if identity["pid"] != root_pid:
                self._terminate_exact_identity(identity, timeout=timeout)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        else:
            process.wait(timeout=0)

    def _cleanup_processes(self):
        cleanup_errors = []
        for kind in ("worker", "utility", "api"):
            for child in reversed(self.children):
                if child["kind"] == kind:
                    try:
                        self._stop_child(child)
                    except Exception as exc:
                        cleanup_errors.append((child["kind"], child["name"],
                                                type(exc).__name__, str(exc)))
                        for identity in reversed(list(child.get("processes", {}).values())):
                            try:
                                self._terminate_exact_identity(identity)
                            except self.psutil.NoSuchProcess:
                                pass
                        process = child["process"]
                        if process.poll() is None:
                            try:
                                process.terminate()
                                process.wait(timeout=5)
                            except (OSError, subprocess.TimeoutExpired) as fallback_exc:
                                cleanup_errors.append((child["kind"], child["name"],
                                                        "popen-fallback", type(fallback_exc).__name__))
                                try:
                                    process.kill()
                                    process.wait(timeout=5)
                                except Exception as kill_exc:
                                    cleanup_errors.append((child["kind"], child["name"],
                                                            "popen-kill", type(kill_exc).__name__))
        survivors = self._live_child_processes()
        for child in self.children:
            if child["process"].poll() is not None:
                child["process"].wait(timeout=0)
        if survivors:
            raise AssertionError(f"测试结束后仍有精确子进程存活：{survivors}")
        if cleanup_errors:
            raise AssertionError(f"子进程cleanup触发恢复路径：{cleanup_errors}")
        for client in self.clients:
            client.close()

    def _global_peak_running(self, task_ids: list[str]) -> int:
        if not task_ids:
            return 0
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT task_id, event_type, payload, created_at FROM workspace_events "
                "WHERE task_id = ANY(%s) ORDER BY created_at, seq",
                (task_ids,),
            ).fetchall()
        starts = {}
        finishes = {}
        for row in rows:
            if row["event_type"] != "task.updated":
                continue
            payload = row["payload"]
            if payload.get("status") == "running" and payload.get("stage") == "starting":
                starts[row["task_id"]] = float(row["created_at"])
            elif payload.get("status") in TERMINAL_STATUSES:
                finishes[row["task_id"]] = float(row["created_at"])
        edges = []
        for task_id in task_ids:
            if task_id not in starts or task_id not in finishes:
                continue
            # Event creation occurs while committing the claim/terminal update;
            # the task started_at value is captured before its scheduler lock
            # and can predate a queued worker's real start by lock-wait time.
            edges.append((starts[task_id], 1))
            edges.append((finishes[task_id], -1))
        active = peak = 0
        for _timestamp, change in sorted(edges):
            active += change
            peak = max(peak, active)
        return peak

    def test_two_api_processes_race_idempotency_and_user_assistant_slot(self):
        first_api = self._start_api()
        second_api = self._start_api()
        self.assertNotEqual(first_api["server_identity"]["pid"],
                            second_api["server_identity"]["pid"],
                            "两个HTTP API必须由不同的真实监听OS进程提供")
        first_client, first_workspace = self._login(first_api, "Alice")
        second_client, second_workspace = self._login(second_api, "Alice")
        self.assertEqual(first_workspace["workspace_id"], second_workspace["workspace_id"])

        request_id = "same-process-race-" + uuid4().hex
        def submit_same(pair):
            api, client, workspace = pair
            return self._submit(api, client, workspace, "chat", request_id=request_id,
                                prompt="same payload", duration_seconds=15)

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(submit_same, (
                (first_api, first_client, first_workspace),
                (second_api, second_client, second_workspace),
            )))
        self.assertEqual([response.status_code for response in responses], [202, 202],
                         [response.text for response in responses])
        task_ids = {response.json()["task_id"] for response in responses}
        self.assertEqual(len(task_ids), 1, "同一幂等键必须跨API进程映射到一个任务")
        task_id = next(iter(task_ids))

        occupied_id = "occupied-process-race-" + uuid4().hex
        def submit_occupied(pair):
            api, client, workspace = pair
            return self._submit(api, client, workspace, "chat", request_id=occupied_id,
                                prompt="different payload", duration_seconds=15)

        with ThreadPoolExecutor(max_workers=2) as pool:
            occupied_responses = list(pool.map(submit_occupied, (
                (first_api, first_client, first_workspace),
                (second_api, second_client, second_workspace),
            )))
        self.assertEqual(sorted(response.status_code for response in occupied_responses), [409, 409],
                         [response.text for response in occupied_responses])

        # With a free assistant slot, two distinct requests racing through the
        # two API processes must still create at most one active task.
        competing_ids = ["free-slot-race-" + uuid4().hex for _ in range(2)]
        def submit_free_slot(pair_and_id):
            pair, free_request_id = pair_and_id
            api, client, workspace = pair
            return self._submit(api, client, workspace, "standards",
                                request_id=free_request_id,
                                prompt="competing on a free slot", duration_seconds=15)

        with ThreadPoolExecutor(max_workers=2) as pool:
            free_responses = list(pool.map(submit_free_slot, zip((
                (first_api, first_client, first_workspace),
                (second_api, second_client, second_workspace),
            ), competing_ids)))
        self.assertEqual(sorted(response.status_code for response in free_responses), [202, 409],
                         [response.text for response in free_responses])
        with self.database.transaction() as connection:
            same_key_count = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE user_id = %s "
                "AND assistant_id = 'chat' AND request_id = %s",
                (self.user_ids["Alice"], request_id),
            ).fetchone()["count"]
            occupied_key_count = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE user_id = %s "
                "AND assistant_id = 'chat' AND request_id = %s",
                (self.user_ids["Alice"], occupied_id),
            ).fetchone()["count"]
            free_slot_count = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE user_id = %s "
                "AND assistant_id = 'standards'",
                (self.user_ids["Alice"],),
            ).fetchone()["count"]
        self.assertEqual(same_key_count, 1)
        self.assertEqual(occupied_key_count, 0)
        self.assertEqual(free_slot_count, 1)
        self.assertEqual(self._task_row(task_id)["status"], "queued")

    def test_two_worker_processes_compete_but_claim_and_start_one_attempt(self):
        api = self._start_api()
        client, workspace = self._login(api, "Alice")
        workers = self._start_idle_workers(2)

        response = self._submit(api, client, workspace, "standards",
                                request_id="two-worker-race-" + uuid4().hex,
                                duration_seconds=1.8)
        self.assertEqual(response.status_code, 202, response.text)
        task_id = response.json()["task_id"]
        row = self._wait_for_terminal(task_id, timeout=12)
        self.assertEqual(row["status"], "completed")
        self.assertIn(row["worker_instance"], {item["worker_instance"] for item in workers})
        self.assertTrue(row["attempt_token"])

        events = self._task_events(task_id)
        running_events = [event for event in events
                          if event["event_type"] == "task.updated"
                          and event["payload"].get("status") == "running"
                          and event["payload"].get("stage") == "starting"]
        self.assertEqual(len(running_events), 1, "两个OS worker只能有一次starting领取事件")
        self.assertEqual(sum(event["payload"].get("status") == "completed"
                             for event in events), 1)
        self.assertTrue(all(event["event_type"] == "task.updated" for event in events))

        winner = row["worker_instance"]
        losers = [item for item in workers if item["worker_instance"] != winner]
        self.assertEqual(len(losers), 1)
        self._stop_child(losers[0])
        self.assertEqual(self.repository.get_task(self.user_ids["Alice"], task_id)["status"],
                         "completed")

    def test_two_users_have_three_slots_each_and_workers_obey_global_limit(self):
        first_api = self._start_api()
        second_api = self._start_api()
        alice, alice_workspace = self._login(first_api, "Alice")
        bob, bob_workspace = self._login(second_api, "Bob")
        alice_plan = self._upload_plan(first_api, alice, alice_workspace)
        bob_plan = self._upload_plan(second_api, bob, bob_workspace)
        self.repository.set_global_limit(2)

        submitted = []
        for username, api, client, workspace, plan_asset in (
            ("Alice", first_api, alice, alice_workspace, alice_plan),
            ("Bob", second_api, bob, bob_workspace, bob_plan),
        ):
            for assistant in ASSISTANTS:
                attachments = [plan_asset] if assistant == "plan" else None
                response = self._submit(
                    api, client, workspace, assistant,
                    request_id=f"{username}-{assistant}-{uuid4().hex}",
                    prompt=f"{username} {assistant} simulation", duration_seconds=2.4,
                    attachment_ids=attachments,
                )
                self.assertEqual(response.status_code, 202, response.text)
                task = response.json()
                submitted.append((username, api, client, workspace, task))

        for username, api, client, workspace, task in submitted:
            attachments = task["attachment_ids"] if task["assistant_id"] == "plan" else None
            duplicate = self._submit(
                api, client, workspace, task["assistant_id"],
                request_id="extra-" + uuid4().hex,
                prompt="must be busy", duration_seconds=2.4,
                attachment_ids=attachments,
            )
            self.assertEqual(duplicate.status_code, 409, duplicate.text)

        for username in ("Alice", "Bob"):
            active_slots = self.repository.snapshot(
                self.user_ids[username], {name: "simulation" for name in ASSISTANTS}
            )
            self.assertEqual(
                sum(item["active_task_id"] is not None for item in active_slots["assistants"]),
                3,
                f"{username}应同时占用三个不同助手槽位",
            )

        workers = [self._start_worker(max_tasks=1) for _ in range(6)]
        task_ids = [item[4]["task_id"] for item in submitted]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            rows = [self._task_row(task_id) for task_id in task_ids]
            if all(row["status"] in TERMINAL_STATUSES for row in rows):
                break
            time.sleep(0.05)
        else:
            rows = [self._task_row(task_id) for task_id in task_ids]
            worker_logs = "\n".join(self._read_log(item) for item in workers)
            self.fail(f"六个任务未在期限内全部完成：{rows}\nworker日志：\n{worker_logs}")

        rows = [self._task_row(task_id) for task_id in task_ids]
        self.assertTrue(all(row["status"] == "completed" for row in rows), rows)
        self.assertEqual(len({row["worker_instance"] for row in rows}), 6,
                         "六个max_tasks=1 worker进程各领取一个任务")
        worker_rows = [self.repository.get_worker(row["worker_instance"]) for row in rows]
        self.assertTrue(all(worker_rows), "每个任务都应持久记录领取它的worker")
        self.assertEqual(len({int(item["pid"]) for item in worker_rows}), 6,
                         "六条任务必须由六个不同的真实worker OS进程领取")
        self.assertEqual(len({item["process_start_token"] for item in worker_rows}), 6)
        peak = self._global_peak_running(task_ids)
        self.assertLessEqual(peak, 2, f"跨worker执行峰值{peak}超过全站限额2")
        self.assertEqual(peak, 2, "应观察到至少两个用户任务真正并发执行")

        for username in ("Alice", "Bob"):
            snapshot = self.repository.snapshot(
                self.user_ids[username], {name: "simulation" for name in ASSISTANTS}
            )
            self.assertTrue(all(item["active_task_id"] is None for item in snapshot["assistants"]))

    def test_api_restart_keeps_queued_task_available_to_independent_worker(self):
        first_api = self._start_api()
        client, workspace = self._login(first_api, "Alice")
        response = self._submit(first_api, client, workspace, "chat",
                                request_id="api-restart-" + uuid4().hex,
                                prompt="survives API process exit", duration_seconds=1.5)
        self.assertEqual(response.status_code, 202, response.text)
        task_id = response.json()["task_id"]
        self.assertEqual(self._task_row(task_id)["status"], "queued")

        self._stop_child(first_api)
        self.assertIsNotNone(first_api["process"].poll())
        worker = self._start_worker(max_tasks=1)
        self._wait_until(lambda: worker["process"].poll() is not None, 12,
                         f"worker未独立完成任务，日志：\n{self._read_log(worker)}")
        self.assertEqual(worker["process"].returncode, 0, self._read_log(worker))
        row = self._task_row(task_id)
        self.assertEqual(row["status"], "completed")

        restarted_api = self._start_api()
        restarted_client, restarted_workspace = self._login(restarted_api, "Alice")
        self.assertEqual(restarted_workspace["workspace_id"], workspace["workspace_id"])
        response = restarted_client.get(
            restarted_api["base_url"] + f"/api/tasks/{task_id}",
            params={"workspace_id": restarted_workspace["workspace_id"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()
        self.assertEqual(snapshot["status"], "completed")
        self.assertTrue(snapshot["result"]["simulation"])
        self.assertIn("survives API process exit", snapshot["result"]["text"])

    def test_dead_worker_lease_keeps_slot_until_exact_process_recovery(self):
        try:
            import psutil
        except ImportError as exc:  # pragma: no cover - project web runtime dependency
            self.skipTest(f"psutil is required for process identity verification: {exc}")

        from rag_arch.web.task_worker import ProcessProbeState, process_start_token, probe_process_identity

        api = self._start_api()
        client, workspace = self._login(api, "Alice")
        worker = self._start_worker(max_tasks=None)
        self._wait_until(
            lambda: (self.repository.get_worker(worker["worker_instance"]) is not None
                     and float(self.repository.get_worker(
                         worker["worker_instance"])["lease_seconds"]) == 3.0),
            8,
            f"worker没有应用3秒测试租约，日志：\n{self._read_log(worker)}",
        )
        response = self._submit(api, client, workspace, "chat",
                                request_id="lease-recovery-" + uuid4().hex,
                                prompt="long local simulation to interrupt", duration_seconds=60)
        self.assertEqual(response.status_code, 202, response.text)
        task_id = response.json()["task_id"]
        self._wait_until(lambda: (self._task_row(task_id)["status"] == "running"
                                  and self.repository.get_worker(worker["worker_instance"]) is not None),
                         8, f"worker没有领取长模拟任务，日志：\n{self._read_log(worker)}")
        worker_row = self.repository.get_worker(worker["worker_instance"])
        self._assert_worker_descends_from_popen(worker, worker_row)
        expected_start = worker_row["process_start_token"]
        self.assertEqual(worker["worker_identity"]["pid"], int(worker_row["pid"]))
        running_row = self._task_row(task_id)
        first_attempt = running_row["attempt_token"]
        self.assertTrue(first_attempt)

        # This is the explicitly requested crash scenario: kill only the exact
        # registered worker interpreter, verified by PID and create-time token.
        actual_worker = psutil.Process(worker["worker_identity"]["pid"])
        self.assertEqual(process_start_token(actual_worker.create_time()), expected_start)
        actual_worker.kill()
        actual_worker.wait(timeout=5)
        if worker["process"].poll() is None:
            try:
                worker["process"].wait(timeout=2)
            except subprocess.TimeoutExpired:
                # The launcher is not the worker. It is handled independently
                # during cleanup after the dead worker has been verified.
                pass
        try:
            current_token = process_start_token(psutil.Process(worker_row["pid"]).create_time())
        except psutil.NoSuchProcess:
            current_token = None
        self.assertNotEqual(current_token, expected_start)
        self.assertEqual(
            probe_process_identity(worker_row["host_id"], int(worker_row["pid"]), expected_start,
                                   psutil_module=psutil),
            ProcessProbeState.EXITED,
        )

        worker_row_after_exit = self.repository.get_worker(worker["worker_instance"])
        self.assertEqual(worker_row_after_exit["process_start_token"], expected_start)
        lease_deadline = (float(worker_row_after_exit["heartbeat_at"])
                          + float(worker_row_after_exit["lease_seconds"]))
        self._wait_until(lambda: time.time() >= lease_deadline, 5,
                         "worker租约未在有界时间内过期")
        stale_instances = []
        def mark_expired_worker_stale():
            stale_instances.extend(self.repository.mark_stale_workers())
            return worker["worker_instance"] in stale_instances

        self._wait_until(mark_expired_worker_stale, 3,
                         "租约过期后没有将worker标记为stale/recovering")
        stale = stale_instances
        self.assertIn(worker["worker_instance"], stale)
        recovering = self._task_row(task_id)
        self.assertEqual(recovering["status"], "recovering")
        self.assertEqual(recovering["attempt_token"], first_attempt)
        with self.database.transaction() as connection:
            slot = connection.execute(
                "SELECT active_task_id FROM assistant_slots WHERE user_id = %s AND assistant_id = 'chat'",
                (self.user_ids["Alice"],),
            ).fetchone()
        self.assertEqual(slot["active_task_id"], task_id,
                         "租约过期只能转recovering，仍需保留用户助手槽")

        # A second, real worker sees no queued work and cannot replay a
        # recovering attempt. It is stopped by its exact Popen handle below.
        idle_worker = self._start_worker(max_tasks=1)
        self._wait_until(lambda: self.repository.get_worker(idle_worker["worker_instance"]) is not None,
                         6, "第二个worker未注册")
        time.sleep(0.35)
        still_recovering = self._task_row(task_id)
        self.assertEqual(still_recovering["status"], "recovering")
        self.assertEqual(still_recovering["attempt_token"], first_attempt)
        self._stop_child(idle_worker)

        recovery = self._spawn(
            "utility", "recover",
            [sys.executable, "-m", "rag_arch.web.worker_process", "recover",
             "--database-url", self.url, "--schema", self.schema],
        )
        self._wait_until(lambda: recovery["process"].poll() is not None, 10,
                         f"worker recover命令未退出，日志：\n{self._read_log(recovery)}")
        self.assertEqual(recovery["process"].returncode, 0, self._read_log(recovery))
        recovery_report = json.loads(self._read_log(recovery))
        outcomes = recovery_report["local_worker_checks"]
        self.assertTrue(any(item.get("worker_instance") == worker["worker_instance"]
                            and item.get("state") == ProcessProbeState.EXITED
                            and item.get("recovered") for item in outcomes), outcomes)

        failed = self._task_row(task_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempt_token"], first_attempt)
        with self.database.transaction() as connection:
            slot = connection.execute(
                "SELECT active_task_id FROM assistant_slots WHERE user_id = %s AND assistant_id = 'chat'",
                (self.user_ids["Alice"],),
            ).fetchone()
            task_count = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE user_id = %s AND assistant_id = 'chat' "
                "AND request_id = %s",
                (self.user_ids["Alice"], failed["request_id"]),
            ).fetchone()["count"]
        self.assertIsNone(slot["active_task_id"])
        self.assertEqual(task_count, 1, "恢复失败不自动重放或创建第二个任务")
        events = self._task_events(task_id)
        self.assertEqual(sum(event["payload"].get("status") == "running"
                             and event["payload"].get("stage") == "starting"
                             for event in events), 1)
        self.assertEqual(sum(event["payload"].get("status") == "failed" for event in events), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
