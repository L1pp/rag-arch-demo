"""Durable task execution in an independent, bounded worker process.

The API process owns task submission and user-facing snapshots.  This module
owns only task execution: it claims one persisted task when a local thread is
free, keeps that claim leased from a separate heartbeat thread, and commits a
terminal result only after the runner and its resource cleanup
have returned.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import asyncio
import inspect
import json
import logging
import math
import os
import socket
import threading
import time
from typing import Any, Callable
from uuid import uuid4
from ..async_support import TaskResourceCleanupError
from ..resource_supervision import (TaskResourceSupervisor, ResourceCleanupUncertain,
                                    resource_supervision_scope)
from .task_service import BusyError


ROOT = Path(__file__).resolve().parents[2]
ASSISTANTS = frozenset({"chat", "standards", "plan"})
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_LOG = logging.getLogger(__name__)


class TaskControlUnavailable(RuntimeError):
    """The database control plane failed; do not start another paid request."""


class StaleWorkerAttempt(RuntimeError):
    """The database no longer recognizes this worker's attempt token."""


class SimulationModeMismatch(RuntimeError):
    """An explicit simulation worker cannot execute a production-model task."""


class ProcessProbeState:
    RUNNING = "running"
    EXITED = "exited"
    UNKNOWN = "unknown"
    REMOTE = "remote"


@dataclass(frozen=True)
class ProcessIdentity:
    host_id: str
    pid: int
    process_start_token: str


@dataclass(frozen=True)
class _FinishIntent:
    task_id: str
    attempt_token: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _ActiveTask:
    claim: dict[str, Any]
    cancel_event: threading.Event
    future: Future


@dataclass
class _TenantDocumentStore:
    key: tuple[int, str]
    store: Any
    borrowers: int = 0


class _DocumentStoreLease:
    def __init__(self, store, release):
        self.store = store
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()


def local_host_id() -> str:
    """Return the local hostname in a stable comparison form."""
    value = socket.gethostname().strip().rstrip(".").casefold()
    if not value:
        raise RuntimeError("无法确定本机 host_id。")
    return value


def process_start_token(create_time: float) -> str:
    """Canonicalize psutil's OS process creation time for persisted fencing."""
    return f"{float(create_time):.6f}"


def current_process_identity(*, psutil_module=None) -> ProcessIdentity:
    """Read an exact local process identity; never fall back to PID alone."""
    if psutil_module is None:
        try:
            import psutil as psutil_module
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("worker 进程核验需要 psutil。") from exc
    pid = os.getpid()
    process = psutil_module.Process(pid)
    return ProcessIdentity(
        host_id=local_host_id(),
        pid=pid,
        process_start_token=process_start_token(process.create_time()),
    )


def probe_process_identity(host_id: str, pid: int, process_start: str, *,
                           local_host: str | None = None, psutil_module=None) -> str:
    """Classify a persisted process identity without guessing remote status.

    A PID that exists with a different creation token is PID reuse and proves
    that the old process exited. Access denied or malformed identity remains
    unknown, so callers must leave its task slots untouched.
    """
    expected_host = local_host or local_host_id()
    if not isinstance(host_id, str) or host_id.casefold() != expected_host.casefold():
        return ProcessProbeState.REMOTE
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return ProcessProbeState.UNKNOWN
    if not isinstance(process_start, str) or not process_start:
        return ProcessProbeState.UNKNOWN
    if psutil_module is None:
        try:
            import psutil as psutil_module
        except ImportError:  # pragma: no cover - packaging failure
            return ProcessProbeState.UNKNOWN

    try:
        process = psutil_module.Process(pid)
        actual_start = process_start_token(process.create_time())
        running = process.is_running()
    except psutil_module.NoSuchProcess:
        return ProcessProbeState.EXITED
    except getattr(psutil_module, "ZombieProcess", ()):  # pragma: no cover - OS dependent
        return ProcessProbeState.EXITED
    except psutil_module.AccessDenied:
        return ProcessProbeState.UNKNOWN
    except (OSError, ValueError, TypeError):
        return ProcessProbeState.UNKNOWN

    if actual_start != process_start:
        return ProcessProbeState.EXITED
    return ProcessProbeState.RUNNING if running else ProcessProbeState.EXITED


def recover_local_workers(repository, *,
                          probe: Callable[..., str] = probe_process_identity) -> list[dict[str, Any]]:
    """Fail stale tasks only after proving the exact local worker has exited.

    Remote workers and local processes that are still running or inaccessible
    are reported as skipped. Lease expiry by itself never releases a slot.
    """
    current_host = local_host_id()
    outcomes: list[dict[str, Any]] = []
    for worker in repository.workers():
        worker_id = worker.get("worker_instance")
        state = str(worker.get("state", "")).lower()
        if not worker_id or state == "stopped":
            continue
        worker_host = worker.get("host_id")
        if not isinstance(worker_host, str) or worker_host.casefold() != current_host.casefold():
            outcomes.append({"worker_instance": worker_id, "state": ProcessProbeState.REMOTE})
            continue
        token = worker.get("process_start_token")
        process_state = probe(worker_host, worker.get("pid"), token,
                              local_host=current_host)
        if process_state == ProcessProbeState.EXITED:
            changed = repository.recover_confirmed_exit(worker_id, token)
            outcomes.append({"worker_instance": worker_id,
                             "state": ProcessProbeState.EXITED,
                             "recovered": bool(changed)})
        else:
            outcomes.append({"worker_instance": worker_id,
                             "state": process_state,
                             "recovered": False})
    return outcomes


def _is_external_work_start(stage: str, detail: str) -> bool:
    """Identify runner callbacks immediately before model/tool/network work."""
    stage = str(stage).strip().lower()
    detail = str(detail).strip()
    if stage == "model":
        return True
    if stage == "embedding":
        return "正在准备规范向量" in detail
    if stage == "files":
        # Chat attachment preparation may make embedding and reranker calls.
        return "正在读取附件全文" in detail
    if stage == "tool":
        return detail.startswith("正在调用")
    if stage in {"web", "web_search", "search", "reranker", "rerank"}:
        return not any(marker in detail for marker in ("已返回", "完成", "结束"))
    return False


def _safe_error(exc: BaseException, assistant_id: str) -> str:
    if isinstance(exc, TaskResourceCleanupError):
        value = exc.as_error()
    elif isinstance(exc, TaskControlUnavailable):
        value = {"code": "control_plane_unavailable",
                 "message": "数据库控制面暂时不可用，worker 已停止本轮后续外部请求。"}
    elif isinstance(exc, StaleWorkerAttempt):
        value = {"code": "stale_attempt",
                 "message": "任务执行权已变化，当前 worker 已停止本轮任务。"}
    elif isinstance(exc, SimulationModeMismatch):
        value = {"code": "simulation_mode_mismatch",
                 "message": "模拟worker不能执行真实模型任务；任务已拒绝。"}
    else:
        # Runner exceptions are deliberately not copied verbatim: provider
        # responses can include URLs or other operational details.
        value = {"code": "worker_execution_failed",
                 "message": f"{assistant_id}任务执行失败（{type(exc).__name__}）。"}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _ControlEvent:
    """Event interface that also interrupts work after a DB outage is observed."""

    def __init__(self, user_cancel: threading.Event, database_available: threading.Event,
                 *, on_cleanup_failure=None):
        self._user_cancel = user_cancel
        self._database_available = database_available
        self._on_cleanup_failure = on_cleanup_failure

    def mark_resource_cleanup_failed(self):
        if self._on_cleanup_failure is not None:
            self._on_cleanup_failure()

    def is_set(self) -> bool:
        return self._user_cancel.is_set() or not self._database_available.is_set()

    def set(self) -> None:
        self._user_cancel.set()

    def clear(self) -> None:
        self._user_cancel.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Wake on user cancellation or a cleared DB-health gate."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.is_set()
                self._user_cancel.wait(min(0.05, remaining))
            else:
                self._user_cancel.wait(0.05)
        return True


class WorkerRuntime:
    """Bounded task ownership with native async runners and sync compatibility."""

    def __init__(self, repository, *, database=None, settings=None, resources=None,
                 root: str | Path = ROOT, worker_instance: str | None = None,
                 concurrency: int = 3, poll_interval: float = 0.5,
                 heartbeat_interval: float = 5.0, lease_seconds: int = 15,
                 simulation: bool = False, runner=None, document_store_factory=None,
                 max_cached_document_stores: int = 8,
                 identity: ProcessIdentity | None = None,
                 resource_factory=None, async_execution: bool = True,
                 task_timeout_seconds: float = 3600):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency须为正整数。")
        if (isinstance(poll_interval, bool) or isinstance(heartbeat_interval, bool)
                or not math.isfinite(poll_interval) or not math.isfinite(heartbeat_interval)
                or poll_interval <= 0 or heartbeat_interval <= 0):
            raise ValueError("poll_interval和heartbeat_interval须大于0。")
        if type(lease_seconds) is not int or lease_seconds < 3:
            raise ValueError("lease_seconds须为不小于3的整数。")
        if type(simulation) is not bool:
            raise ValueError("simulation须为布尔值。")
        if type(max_cached_document_stores) is not int or max_cached_document_stores < 1:
            raise ValueError("max_cached_document_stores须为正整数。")
        if type(async_execution) is not bool:
            raise ValueError("async_execution须为布尔值。")
        if (isinstance(task_timeout_seconds, bool) or not isinstance(task_timeout_seconds, (int, float))
                or not math.isfinite(task_timeout_seconds) or task_timeout_seconds <= 0):
            raise ValueError("task_timeout_seconds须为有限正数。")

        self.repository = repository
        self.database = database
        self.settings = settings
        self.root = Path(root).resolve()
        self.worker_instance = worker_instance or uuid4().hex
        self.concurrency = concurrency
        self.poll_interval = float(poll_interval)
        self.heartbeat_interval = float(heartbeat_interval)
        self.lease_seconds = lease_seconds
        self.simulation = simulation
        self.runner = runner
        self.document_store_factory = document_store_factory
        self.max_cached_document_stores = max_cached_document_stores
        self.resource_factory = resource_factory
        self.async_execution = async_execution
        self.task_timeout_seconds = float(task_timeout_seconds)
        self._async_runtime = None
        self._async_runtime_lock = threading.Lock()
        self.identity = identity or current_process_identity()
        if self.heartbeat_interval >= self.lease_seconds:
            raise ValueError("heartbeat_interval必须小于lease_seconds。")

        self._resources = resources
        self._owns_resources = resources is None
        self._resource_lock = threading.RLock()
        self._tenant_store_lock = threading.RLock()
        self._tenant_stores: OrderedDict[tuple[int, str], _TenantDocumentStore] = OrderedDict()
        self._runtime_resources_closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="rag-web-task")
        self._active: dict[str, _ActiveTask] = {}
        self._pending_finishes: OrderedDict[str, _FinishIntent] = OrderedDict()
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._retire_requested = threading.Event()
        self._held_for_recovery = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._wake = threading.Event()
        self._database_available = threading.Event()
        self._fenced = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._registered = False
        self._closed = False

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def pending_finish_count(self) -> int:
        with self._lock:
            return len(self._pending_finishes)

    def request_stop(self) -> None:
        """Stop claiming and request cooperative cancellation of active work."""
        self._stop_requested.set()
        with self._lock:
            active = tuple(self._active.values())
        for item in active:
            item.cancel_event.set()
        self._wake.set()

    def run(self, *, max_tasks: int | None = None) -> None:
        """Register and run until stopped; ``max_tasks`` is useful for smoke tests."""
        if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
            raise ValueError("max_tasks须为正整数或None。")
        if self._registered:
            raise RuntimeError("同一个WorkerRuntime不能重复run。")

        registered_worker = self.repository.register_worker(
            self.worker_instance, self.identity.host_id, self.identity.pid,
            self.identity.process_start_token)
        if isinstance(registered_worker, dict):
            persisted_identity = (
                registered_worker.get("host_id"), registered_worker.get("pid"),
                registered_worker.get("process_start_token"),
            )
            supplied_identity = (
                self.identity.host_id, self.identity.pid,
                self.identity.process_start_token,
            )
            if persisted_identity != supplied_identity:
                raise RuntimeError("数据库worker进程身份与当前进程身份不一致，拒绝领取任务。")
        _LOG.info("worker registered instance=%s host=%s pid=%d",
                  self.worker_instance, self.identity.host_id, self.identity.pid)
        self._registered = True
        self._database_available.set()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="rag-worker-heartbeat", daemon=True)
        self._heartbeat_thread.start()

        claimed = 0
        try:
            while not self._stop_requested.is_set():
                self._collect_finished()
                if self._pending_finishes and not self._drain_finishes():
                    self._wake.wait(self.poll_interval)
                    self._wake.clear()
                    continue
                if self._fenced.is_set():
                    self.request_stop()
                    continue

                no_more_claims = self._retire_requested.is_set() or (max_tasks is not None and claimed >= max_tasks)
                with self._lock:
                    local_free = self.concurrency - len(self._active)
                if not no_more_claims and local_free > 0 and self._database_available.is_set():
                    try:
                        claim = self.repository.claim_next(self.worker_instance)
                    except Exception as exc:
                        # claim_next may have committed running state before a
                        # response was lost. We cannot safely assume it did
                        # not claim a task whose token we never received.
                        self._database_available.clear()
                        self._fenced.set()
                        _LOG.error("claim结果不确定；worker将停止以便安全核验（%s）",
                                   type(exc).__name__)
                        self.request_stop()
                    else:
                        self._database_available.set()
                        if claim is not None:
                            try:
                                self._submit_claim(claim)
                            except Exception as exc:
                                # A task has already been assigned in the DB;
                                # if the local executor cannot register it,
                                # treat the token as unknown and stop safely.
                                self._database_available.clear()
                                self._fenced.set()
                                _LOG.exception("已领取任务无法登记到本地执行器；停止以便安全核验（%s）",
                                               type(exc).__name__)
                                self.request_stop()
                                continue
                            claimed += 1
                            continue

                if no_more_claims and self.active_count == 0 and self.pending_finish_count == 0:
                    self.request_stop()
                    continue
                self._wake.wait(self.poll_interval)
                self._wake.clear()
        except KeyboardInterrupt:
            self.request_stop()
        finally:
            self._graceful_shutdown()

    def _submit_claim(self, claim: dict[str, Any]) -> None:
        if not isinstance(claim, dict):
            raise TypeError("claim_next必须返回任务对象或None。")
        task_id = claim.get("task_id")
        token = claim.get("attempt_token")
        request = claim.get("request")
        if not all(isinstance(value, str) and value for value in (task_id, token)):
            raise ValueError("任务claim缺少task_id或attempt_token。")
        if not isinstance(request, dict):
            raise ValueError("任务claim缺少request对象。")
        if request.get("assistant_id") not in ASSISTANTS:
            raise ValueError("任务claim包含未知assistant_id。")
        cancel_event = threading.Event()
        with self._lock:
            if task_id in self._active or task_id in self._pending_finishes:
                raise RuntimeError("repository重复领取了本地已登记任务。")
            future = self._executor.submit(self._execute_claim, claim, cancel_event)
            future.add_done_callback(lambda _future: self._wake.set())
            self._active[task_id] = _ActiveTask(claim, cancel_event, future)

    def _execute_claim(self, claim: dict[str, Any], cancel_event: threading.Event) -> _FinishIntent:
        task_id = claim["task_id"]
        token = claim["attempt_token"]
        request = dict(claim["request"])
        assistant_id = request.get("assistant_id", "task")
        result = None
        status = "failed"
        error = None
        store = None
        store_lease = None
        cleanup_uncertain = False
        supervisor = TaskResourceSupervisor(self.repository, self.worker_instance, task_id,
                                            self.identity, root=self.root)
        runner_event = _ControlEvent(cancel_event, self._database_available,
                                    on_cleanup_failure=self._retire_after_cleanup_failure)
        try:
            self._check_control(task_id, token, cancel_event)
            progress = self._progress_callback(task_id, token, cancel_event, runner_event)
            request_simulation = request.get("simulation", False)
            if type(request_simulation) is not bool:
                raise ValueError("任务simulation标记必须是布尔值。")
            if self.simulation and not request_simulation:
                raise SimulationModeMismatch()
            use_demo = request_simulation
            from ..request_control import request_control_scope

            def check_external_request():
                if cancel_event.is_set():
                    raise CancelledError()
                if not self._database_available.is_set():
                    raise TaskControlUnavailable()

            # Context-local provider guards share this task's raw cancellation
            # and database-health state. Scope cleanup is guaranteed even when
            # a runner or provider raises.
            with resource_supervision_scope(supervisor), request_control_scope(check_external_request):
                if use_demo:
                    # API-persisted simulation and the explicit test worker
                    # mode both route only to the local demonstration runner.
                    from .demo import run_demo
                    result = run_demo(request, progress, runner_event)
                elif self.runner is not None:
                    result = self.runner(request, progress, runner_event)
                    if inspect.isawaitable(result):
                        result = self._async_run(self._guard_async_runner(result, runner_event))
                else:
                    store, store_lease = self._acquire_document_store(request)
                    result = self._run_configured_runner(
                        assistant_id, request, progress, runner_event, store)
            if not isinstance(result, dict):
                raise TypeError("任务runner必须返回对象。")
            if not isinstance(result.get("text"), str):
                raise TypeError("任务runner结果必须包含字符串text。")
            # Validate the exact JSONB boundary before retaining a pending
            # finish. Non-finite or custom values become task failures rather
            # than an endless retry that holds the user's assistant slot.
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":"))
            result = json.loads(encoded)
            # Simulation behavior comes from the persisted server setting.
            result["simulation"] = use_demo
            status = "cancelled" if cancel_event.is_set() else "completed"
        except CancelledError:
            result = None
            if cancel_event.is_set():
                status = "cancelled"
            else:
                status = "failed"
                cause = (TaskControlUnavailable() if not self._database_available.is_set()
                         else RuntimeError("runner停止时没有用户取消请求。"))
                error = _safe_error(cause, assistant_id)
        except StaleWorkerAttempt as exc:
            self._fenced.set()
            status = "failed"
            result = None
            error = _safe_error(exc, assistant_id)
        except Exception as exc:
            result = None
            cleanup_uncertain = isinstance(exc, ResourceCleanupUncertain)
            if isinstance(exc, TaskResourceCleanupError):
                # No in-flight work remains for this task. Retire the process
                # after other users finish, rather than cancelling their work.
                self._retire_requested.set()
                self._wake.set()
            if cancel_event.is_set():
                status = "cancelled"
            else:
                status = "failed"
                error = _safe_error(
                    TaskControlUnavailable() if not self._database_available.is_set() else exc,
                    assistant_id)
        finally:
            # PDF/DOCX cleanup is part of execution. The caller cannot finish
            # the task until this method and close() have both returned.
            if store_lease is not None:
                try:
                    store_lease.release()
                except Exception as exc:
                    cleanup_uncertain = cleanup_uncertain or isinstance(exc, ResourceCleanupUncertain)
                    status = "failed"
                    result = None
                    error = json.dumps(
                        {"code": "resource_cleanup_failed",
                         "message": f"任务资源清理失败（{type(exc).__name__}）。"},
                        ensure_ascii=False, separators=(",", ":"))

        if supervisor.retire_requested or cleanup_uncertain:
            self._retire_after_cleanup_failure()
        if supervisor.has_unconfirmed or cleanup_uncertain:
            status, result = "recovering", None
            error = json.dumps({"code": "resource_cleanup_uncertain",
                                "message": "本地进程资源未确认退出，任务槽和执行容量保留，等待管理员核验。"},
                               ensure_ascii=False, separators=(",", ":"))

        return _FinishIntent(task_id=task_id, attempt_token=token,
                             status=status, result=result, error=error)

    def _new_document_store(self, request):
        user_id = request.get("user_id")
        workspace_id = request.get("workspace_id")
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValueError("任务缺少服务端user_id。")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("任务缺少workspace_id。")
        if self.database is None:
            raise RuntimeError("真实任务执行需要注入Database。")
        if self.document_store_factory is None:
            from .documents import DocumentStore
            factory = DocumentStore
        else:
            factory = self.document_store_factory
        resources = self._get_resources()
        return factory(self.root, workspace_id, resources.pdf_backend,
                       user_id=user_id, database=self.database)

    def _acquire_document_store(self, request):
        """Borrow a bounded user/workspace store; active stores cannot be evicted."""
        user_id = request.get("user_id")
        workspace_id = request.get("workspace_id")
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValueError("任务缺少服务端user_id。")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("任务缺少workspace_id。")
        key = (user_id, workspace_id)
        evicted = None
        with self._tenant_store_lock:
            entry = self._tenant_stores.get(key)
            if entry is not None:
                entry.borrowers += 1
                self._tenant_stores.move_to_end(key)
                return entry.store, _DocumentStoreLease(
                    entry.store, lambda: self._release_cached_store(key, entry))

            if len(self._tenant_stores) >= self.max_cached_document_stores:
                for candidate_key, candidate in tuple(self._tenant_stores.items()):
                    if candidate.borrowers == 0:
                        evicted = self._tenant_stores.pop(candidate_key)
                        break

            if len(self._tenant_stores) < self.max_cached_document_stores:
                if evicted is not None:
                    evicted.store.close()
                    evicted = None
                store = self._new_document_store(request)
                entry = _TenantDocumentStore(key=key, store=store, borrowers=1)
                self._tenant_stores[key] = entry
                return store, _DocumentStoreLease(
                    store, lambda: self._release_cached_store(key, entry))

            # All cached tenant stores are currently in use. Keep memory
            # bounded by using a task-local store that closes on release.
            transient = self._new_document_store(request)
            return transient, _DocumentStoreLease(transient, transient.close)

    def _release_cached_store(self, key, entry):
        with self._tenant_store_lock:
            current = self._tenant_stores.get(key)
            if current is not entry or entry.borrowers < 1:
                raise RuntimeError("DocumentStore借用计数不一致。")
            entry.borrowers -= 1

    def _close_tenant_document_stores(self):
        with self._tenant_store_lock:
            active = [key for key, entry in self._tenant_stores.items()
                      if entry.borrowers != 0]
            if active:
                raise RuntimeError("仍有任务借用DocumentStore，不能关闭worker资源。")
            entries = tuple(self._tenant_stores.values())
            self._tenant_stores.clear()
        first_error = None
        for entry in entries:
            try:
                entry.store.close()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def _get_resources(self):
        if self._resources is None:
            with self._resource_lock:
                if self._resources is None:
                    if self.resource_factory is None:
                        from .standards_resources import StandardsResources
                        factory = StandardsResources
                    else:
                        factory = self.resource_factory
                    resources = factory(self.root)
                    try:
                        if self.settings is not None:
                            resources.configure_catalog(self.settings)
                            resources.configure_embeddings(self.settings)
                    except BaseException:
                        close = getattr(resources, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                _LOG.exception("未发布的worker资源清理失败")
                        raise
                    self._resources = resources
        return self._resources

    def _run_configured_runner(self, assistant_id, request, progress, cancel_event, store):
        if self.settings is None:
            raise RuntimeError("真实任务执行需要加载Settings。")
        if self.async_execution:
            return self._async_run(self._guard_async_runner(
                self._arun_configured_runner(assistant_id, request, progress, cancel_event, store),
                cancel_event))
        resources = self._get_resources()
        if assistant_id == "chat":
            from .chat_runner import run_chat
            return run_chat(request, progress, cancel_event,
                            settings=self.settings, documents=store)
        if assistant_id == "standards":
            from .standards_runner import run_standards
            return run_standards(request, progress, cancel_event,
                                 settings=self.settings, resources=resources)
        if assistant_id == "plan":
            from .plan_runner import run_plan
            return run_plan(request, progress, cancel_event,
                            settings=self.settings, resources=resources, documents=store)
        raise ValueError("任务包含未知assistant_id。")

    def _async_run(self, coroutine):
        from ..async_support import AsyncTaskRuntime
        with self._async_runtime_lock:
            if self._async_runtime is None:
                self._async_runtime = AsyncTaskRuntime(local_workers=max(2, self.concurrency * 2))
            runtime = self._async_runtime
        # Never cancel this concurrent Future: its completion must mean the
        # coroutine, including any shielded local cleanup, actually returned.
        return runtime.run(coroutine)

    def _retire_after_cleanup_failure(self):
        self._retire_requested.set()
        self._wake.set()

    async def _guard_async_runner(self, coroutine, cancel_event):
        task = asyncio.create_task(coroutine)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.task_timeout_seconds
        try:
            while True:
                if task.done():
                    return await task
                if cancel_event.is_set():
                    task.cancel()
                    return await task
                if loop.time() >= deadline:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise TimeoutError("任务超过配置的执行时限。")
                await asyncio.wait((task,), timeout=min(0.05, max(0.001, deadline - loop.time())))
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _arun_configured_runner(self, assistant_id, request, progress, cancel_event, store):
        # _acquire_document_store prepared these resources on the bounded
        # task thread. Never initialize a corpus/PDF backend on the async loop.
        resources = self._resources
        if assistant_id == "chat":
            from .chat_runner import arun_chat
            return await arun_chat(request, progress, cancel_event,
                                   settings=self.settings, documents=store)
        if assistant_id == "standards":
            from .standards_runner import arun_standards
            return await arun_standards(request, progress, cancel_event,
                                        settings=self.settings, resources=resources)
        if assistant_id == "plan":
            from .plan_runner import arun_plan
            return await arun_plan(request, progress, cancel_event,
                                   settings=self.settings, resources=resources, documents=store)
        raise ValueError("任务包含未知assistant_id。")

    def _progress_callback(self, task_id: str, token: str,
                           cancel_event: threading.Event, runner_event: _ControlEvent):
        def publish(stage, detail=""):
            if cancel_event.is_set():
                raise CancelledError()
            if runner_event.is_set():
                if cancel_event.is_set():
                    raise CancelledError()
                raise TaskControlUnavailable()
            stage_text = str(stage)
            detail_text = str(detail)
            before_external_work = _is_external_work_start(stage_text, detail_text)
            if before_external_work:
                self._check_control(task_id, token, cancel_event)
            try:
                self.repository.progress(self.worker_instance, task_id, token,
                                         stage_text, detail_text)
            except Exception as exc:
                self._database_available.clear()
                if before_external_work:
                    raise TaskControlUnavailable() from None
                # A progress event may be lost during a short DB outage. The
                # next external-work guard will stop new requests, while an
                # already-running call may still return a reusable result.
                _LOG.warning("task progress暂未持久化（%s）", type(exc).__name__)
        return publish

    def _check_control(self, task_id: str, token: str,
                       cancel_event: threading.Event) -> dict[str, Any]:
        if cancel_event.is_set():
            raise CancelledError()
        try:
            control = self.repository.execution_control(
                self.worker_instance, task_id, token)
        except Exception as exc:
            if type(exc).__name__ == "StaleAttempt":
                self._fenced.set()
                self._database_available.clear()
                cancel_event.set()
                raise StaleWorkerAttempt() from None
            self._database_available.clear()
            raise TaskControlUnavailable() from None
        self._database_available.set()
        if not isinstance(control, dict):
            raise TaskControlUnavailable()
        if control.get("cancel_requested") or control.get("status") == "cancelling":
            cancel_event.set()
            raise CancelledError()
        if control.get("status") != "running":
            self._fenced.set()
            cancel_event.set()
            raise StaleWorkerAttempt()
        return control

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            if self._fenced.is_set():
                # A lost claim token cannot be leased selectively through the
                # repository API. Stop renewing this worker's task set and
                # exit after known executions have cooperatively drained.
                self._database_available.clear()
                return
            cycle_ok = True
            try:
                self.repository.heartbeat_worker(
                    self.worker_instance, lease_seconds=self.lease_seconds)
            except Exception as exc:
                cycle_ok = False
                _LOG.warning("worker heartbeat暂不可用（%s）", type(exc).__name__)
            if cycle_ok:
                with self._lock:
                    active = tuple(self._active.items())
                for task_id, item in active:
                    try:
                        control = self.repository.execution_control(
                            self.worker_instance, task_id,
                            item.claim["attempt_token"])
                    except Exception as exc:
                        if type(exc).__name__ == "StaleAttempt":
                            if item.future.done():
                                continue
                            cycle_ok = False
                            self._fenced.set()
                            self._database_available.clear()
                            item.cancel_event.set()
                        else:
                            cycle_ok = False
                            _LOG.warning("任务控制状态暂不可用（%s）", type(exc).__name__)
                        continue
                    if not isinstance(control, dict):
                        cycle_ok = False
                        continue
                    if control.get("cancel_requested") or control.get("status") == "cancelling":
                        item.cancel_event.set()
                    elif control.get("status") in _TERMINAL and item.future.done():
                        # The main loop may have committed this task after the
                        # heartbeat thread copied its active-task snapshot.
                        # That late terminal observation is expected.
                        continue
                    elif control.get("status") != "running":
                        cycle_ok = False
                        self._fenced.set()
                        self._database_available.clear()
                        item.cancel_event.set()
            if cycle_ok and not self._fenced.is_set():
                self._database_available.set()
            else:
                self._database_available.clear()
            self._wake.set()
            self._heartbeat_stop.wait(self.heartbeat_interval)

    def _collect_finished(self) -> None:
        with self._lock:
            done = [(task_id, item) for task_id, item in self._active.items()
                    if item.future.done()]
            for task_id, _item in done:
                self._active.pop(task_id, None)
        for task_id, item in done:
            try:
                intent = item.future.result()
            except BaseException as exc:  # executor-level failures still need a durable terminal state
                intent = _FinishIntent(
                    task_id=task_id,
                    attempt_token=item.claim["attempt_token"],
                    status="failed",
                    error=_safe_error(exc, item.claim["request"].get("assistant_id", "task")),
                )
            with self._lock:
                self._pending_finishes[task_id] = intent

    def _drain_finishes(self) -> bool:
        """Commit pending results once; never re-run a completed runner."""
        with self._lock:
            intents = tuple(self._pending_finishes.items())
        all_committed = True
        for task_id, intent in intents:
            preserving_slot = intent.status == "recovering"
            try:
                if intent.status == "recovering":
                    accepted = self.repository.keep_recovering(
                        self.worker_instance, intent.task_id, intent.attempt_token, intent.error)
                else:
                    accepted = self.repository.finish(
                        self.worker_instance, intent.task_id, intent.attempt_token,
                        intent.status, result=intent.result, error=intent.error)
                    if (accepted is False and callable(getattr(self.repository, "task_has_unconfirmed_resources", None))
                            and self.repository.task_has_unconfirmed_resources(intent.task_id)):
                        accepted = self.repository.keep_recovering(
                            self.worker_instance, intent.task_id, intent.attempt_token,
                            "资源状态未确认，任务保持恢复中。")
                        preserving_slot = True
                        self._retire_after_cleanup_failure()
            except Exception as exc:
                all_committed = False
                if type(exc).__name__ == "StaleAttempt":
                    self._fenced.set()
                    self._database_available.clear()
                    with self._lock:
                        self._pending_finishes.pop(task_id, None)
                    _LOG.error("finish被fencing拒绝，等待本机进程退出后恢复任务")
                else:
                    self._database_available.clear()
                    if isinstance(exc, (ValueError, TypeError, OverflowError)):
                        self._fenced.set()
                        with self._lock:
                            self._pending_finishes.pop(task_id, None)
                        _LOG.error("finish数据不符合持久化契约；worker停止并等待安全恢复")
                    else:
                        _LOG.warning("终态暂未持久化，将重试且不重跑runner（%s）",
                                     type(exc).__name__)
                continue
            if accepted:
                if preserving_slot:
                    self._held_for_recovery.set()
                    self._retire_after_cleanup_failure()
                self._database_available.set()
                with self._lock:
                    self._pending_finishes.pop(task_id, None)
            else:
                # A false result means this attempt cannot commit. Keep the
                # task in recovering and stop this process so exact process
                # exit can be proven by the explicit local recovery command.
                self._fenced.set()
                self._database_available.clear()
                with self._lock:
                    self._pending_finishes.pop(task_id, None)
                _LOG.error("finish确认当前attempt已失效；worker将停止以便安全恢复")
        return all_committed

    def _close_runtime_resources(self) -> bool:
        if self._runtime_resources_closed:
            return not self._fenced.is_set()
        cleanup_ok = True
        try:
            self._close_tenant_document_stores()
        except Exception:
            cleanup_ok = False
            _LOG.exception("worker租户DocumentStore清理失败")
        if self._owns_resources and self._resources is not None:
            try:
                async_close = getattr(self._resources, "aclose", None)
                if self._async_runtime is not None and callable(async_close):
                    self._async_run(async_close())
            except Exception:
                cleanup_ok = False
                _LOG.exception("worker异步网络资源清理失败")
            try:
                self._resources.close()
            except Exception:
                cleanup_ok = False
                _LOG.exception("worker共享规范资源清理失败")
        self._runtime_resources_closed = True
        if not cleanup_ok:
            self._fenced.set()
            self._database_available.clear()
        return cleanup_ok

    def _graceful_shutdown(self) -> None:
        if self._closed:
            return
        self._stop_requested.set()
        with self._lock:
            active = tuple(self._active.values())
        for item in active:
            item.cancel_event.set()

        # Do not cancel Futures: running model/PDF/DOCX calls must return and
        # clean up before a terminal state or stopped worker record is written.
        while True:
            self._collect_finished()
            pending_empty = self.pending_finish_count == 0
            active_empty = self.active_count == 0
            if pending_empty and active_empty:
                break
            if self._pending_finishes:
                self._drain_finishes()
            self._wake.wait(self.poll_interval)
            self._wake.clear()

        # No runner is borrowing a tenant store now. Close idle per-user caches
        # and the shared PDF/resource backend before publishing worker=stopped.
        cleanup_ok = self._close_runtime_resources()
        if (self._registered and cleanup_ok and not self._fenced.is_set()
                and not self._held_for_recovery.is_set()):
            while True:
                try:
                    self.repository.stop_worker(self.worker_instance)
                    break
                except BusyError:
                    # A held task/resource is a durable safety guard, not a
                    # transient connection failure. Leave its slot intact,
                    # stop heartbeats, and let the OS exit become provable.
                    self._held_for_recovery.set()
                    _LOG.warning("worker保留未确认资源或任务；停止心跳并退出，等待管理员核验")
                    break
                except Exception as exc:
                    _LOG.warning("worker stopped状态暂未提交，等待数据库恢复（%s）",
                                 type(exc).__name__)
                    self._wake.wait(self.poll_interval)
                    self._wake.clear()

        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._async_runtime is not None:
            self._async_runtime.close()
        self._closed = True
