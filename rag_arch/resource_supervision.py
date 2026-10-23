"""Owned process records shared by task threads and native async runners.

No process is killed here. Adapters own their subprocess handles; this module
only persists intent, verifies exact identities, and preserves uncertainty.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import re
import socket
import threading
from uuid import uuid4


_current = ContextVar("rag_process_supervisor", default=None)


def resource_owner_descriptor_path(root, worker_instance, resource_id):
    """Stable path for the durable per-resource owner descriptor."""
    worker_instance = str(worker_instance)
    resource_id = str(resource_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", worker_instance):
        raise ValueError("worker_instance不能安全用作resource owner目录名。")
    if not re.fullmatch(r"[a-f0-9]{32}", resource_id):
        raise ValueError("resource_id必须是32位小写hex UUID。")
    return (Path(root).resolve() / "data" / "runtime" / "resource_owners"
            / worker_instance / f"{resource_id}.json")


class ResourceCleanupUncertain(RuntimeError):
    """An external local process may still be running; retain the task slot."""

    def __init__(self, code="resource_cleanup_uncertain"):
        self.code = code if isinstance(code, str) and code.replace("_", "").isalnum() else "resource_cleanup_uncertain"
        super().__init__("本地进程资源退出状态未能确认，任务槽保留等待核验。")


def process_token(pid):
    import psutil
    return f"{psutil.Process(pid).create_time():.6f}"


def probe_process(pid, token, *, executable=None):
    """Return exited/running/unknown; PID reuse proves the old process ended."""
    import psutil
    try:
        process = psutil.Process(pid)
        if f"{process.create_time():.6f}" != token:
            return "exited"
        if executable and os.path.normcase(os.path.realpath(process.exe())) != os.path.normcase(os.path.realpath(executable)):
            return "unknown"
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "exited"
        return "running"
    except psutil.NoSuchProcess:
        return "exited"
    except (psutil.AccessDenied, OSError, ValueError):
        return "unknown"


class TaskResourceSupervisor:
    def __init__(self, repository, worker_instance, task_id, identity, *, root):
        self.repository = repository
        self.worker_instance = worker_instance
        self.task_id = task_id
        self.identity = identity
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._unknown = set()
        self._retire = False

    @property
    def has_unconfirmed(self):
        with self._lock:
            return bool(self._unknown)

    @property
    def retire_requested(self):
        with self._lock:
            return self._retire

    def _mark_unknown(self, resource_id):
        with self._lock:
            self._unknown.add(resource_id)
            self._retire = True

    def _mark_exited(self, resource_id):
        with self._lock:
            self._unknown.discard(resource_id)

    def begin(self, kind, scope, owner_path):
        if owner_path is not None:
            owner_path = str(Path(owner_path).resolve())
            if not Path(owner_path).is_relative_to(self.root):
                self._mark_unknown("invalid_owner_path")
                raise ResourceCleanupUncertain("invalid_owner_path")
        resource_id = uuid4().hex
        descriptor_path = resource_owner_descriptor_path(
            self.root, self.worker_instance, resource_id,
        )
        begin_method = self.repository.begin_resource
        parameters = inspect.signature(begin_method).parameters
        supports_descriptor = ("resource_id" in parameters
                              or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                     for p in parameters.values()))
        begin_kwargs = {
            "task_id": self.task_id if scope == "task" else None,
            "owner_path": owner_path,
        }
        try:
            if supports_descriptor:
                row = begin_method(
                    self.worker_instance, kind, scope,
                    **begin_kwargs, resource_id=resource_id,
                    owner_descriptor_path=str(descriptor_path),
                )
            else:  # compatibility with narrow in-memory test repositories
                row = begin_method(self.worker_instance, kind, scope, **begin_kwargs)
        except Exception:
            self._mark_unknown("resource_registration_failed")
            raise ResourceCleanupUncertain("resource_registration_failed") from None
        registered_id = row["resource_id"]
        if supports_descriptor:
            registered_path = row.get("owner_descriptor_path")
            registered_nonce = row.get("owner_nonce")
            if (registered_id != resource_id or registered_path != str(descriptor_path)
                    or not isinstance(registered_nonce, str) or not registered_nonce):
                self._mark_unknown("resource_descriptor_registration_incomplete")
                raise ResourceCleanupUncertain("resource_descriptor_registration_incomplete")
        else:
            # Narrow fakes predating descriptor fields receive a local-only
            # handle; production repositories must return their persisted path
            # and nonce above.
            descriptor_path = resource_owner_descriptor_path(
                self.root, self.worker_instance, registered_id,
            )
            registered_nonce = uuid4().hex
        return ResourceHandle(
            kind, scope, owner_path, self, registered_id,
            owner_descriptor_path=str(descriptor_path),
            owner_nonce=registered_nonce,
        )

    def _write_owner_descriptor(self, handle, *, pid, process_start_token,
                                executable, owned=True):
        descriptor_path = Path(handle.owner_descriptor_path).absolute()
        expected = resource_owner_descriptor_path(
            self.root, self.worker_instance, handle.resource_id,
        ).absolute()
        if descriptor_path != expected:
            raise ResourceCleanupUncertain("owner_descriptor_path_mismatch")
        root = self.root.resolve()
        if not descriptor_path.is_relative_to(root / "data" / "runtime" / "resource_owners"):
            raise ResourceCleanupUncertain("owner_descriptor_outside_runtime")
        parent = descriptor_path.parent
        trusted_root = root / "data" / "runtime" / "resource_owners"
        directory_chain = [
            root / "data", root / "data" / "runtime", trusted_root, parent,
        ]
        for directory in directory_chain:
            if directory.exists() or directory.is_symlink():
                if directory.is_symlink():
                    raise ResourceCleanupUncertain("owner_descriptor_directory_symlink")
                resolved = directory.resolve(strict=True)
                if os.path.normcase(os.path.abspath(resolved)) != os.path.normcase(os.path.abspath(directory)):
                    raise ResourceCleanupUncertain("owner_descriptor_directory_untrusted")
                if not resolved.is_dir():
                    raise ResourceCleanupUncertain("owner_descriptor_directory_invalid")
            else:
                directory.mkdir(exist_ok=True)
                resolved = directory.resolve(strict=True)
                if os.path.normcase(os.path.abspath(resolved)) != os.path.normcase(os.path.abspath(directory)):
                    raise ResourceCleanupUncertain("owner_descriptor_directory_untrusted")
            if os.name != "nt" and directory in (trusted_root, parent):
                directory.chmod(0o700)
        resolved_parent = parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(trusted_root.resolve(strict=True)):
            raise ResourceCleanupUncertain("owner_descriptor_directory_untrusted")
        if descriptor_path.is_symlink():
            raise ResourceCleanupUncertain("owner_descriptor_symlink")
        if descriptor_path.exists():
            try:
                existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except Exception:
                raise ResourceCleanupUncertain("owner_descriptor_already_exists") from None
            if (existing.get("resource_id") == handle.resource_id
                    and existing.get("owner_nonce") == handle.owner_nonce
                    and existing.get("worker_instance") == self.worker_instance
                    and existing.get("resource") == {
                        "host_id": self.identity.host_id, "pid": pid,
                        "process_start_token": process_start_token,
                        "executable": executable,
                    }):
                return str(descriptor_path)
            raise ResourceCleanupUncertain("owner_descriptor_conflict")

        descriptor = {
            "schema_version": 1,
            "resource_id": handle.resource_id,
            "owner_nonce": handle.owner_nonce,
            "worker_instance": self.worker_instance,
            "owner_path": handle.owner_path,
            "owned": bool(owned),
            "worker": {
                "host_id": self.identity.host_id,
                "pid": self.identity.pid,
                "process_start_token": self.identity.process_start_token,
            },
            "resource": {
                "host_id": self.identity.host_id,
                "pid": pid,
                "process_start_token": process_start_token,
                "executable": executable,
            },
        }
        data = (json.dumps(descriptor, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        temporary = descriptor_path.with_name(f".{descriptor_path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, descriptor_path)
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    os.close(directory_fd)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return str(descriptor_path)


@dataclass
class ResourceHandle:
    kind: str
    scope: str
    owner_path: str | None = None
    supervisor: TaskResourceSupervisor | None = None
    resource_id: str | None = None
    owner_descriptor_path: str | None = None
    owner_nonce: str | None = None
    pid: int | None = None
    process_start_token: str | None = None
    executable: str | None = None
    _verified_exited: bool = False
    _lock: object = field(default_factory=threading.RLock, repr=False)

    def attach_process(self, pid, process_start_token=None, executable=None):
        if type(pid) is not int or pid < 1:
            self.unknown("invalid_process_identity")
        # Preserve local identity before attempting persistence. A failed DB
        # write must never erase the information needed by adapter cleanup.
        self.pid = pid
        try:
            observed = process_token(pid)
            if process_start_token is not None and observed != str(process_start_token):
                self.unknown("process_identity_changed_during_registration")
            self.process_start_token = observed
            if executable is None:
                import psutil
                executable = psutil.Process(pid).exe()
            self.executable = str(Path(executable).resolve()) if executable else None
            if self.supervisor:
                self.supervisor._write_owner_descriptor(
                    self, pid=pid, process_start_token=observed,
                    executable=self.executable, owned=True,
                )
                attached = self.supervisor.repository.attach_resource(
                    self.supervisor.worker_instance, self.resource_id,
                    host_id=self.supervisor.identity.host_id, pid=pid,
                    process_start_token=observed, executable=self.executable)
                if attached is False or attached is None:
                    self.unknown("resource_identity_registration_rejected")
        except ResourceCleanupUncertain as exc:
            self._record_unknown(exc.code)
            raise
        except Exception:
            self.unknown("process_identity_registration_failed")
        self.used()
        return self

    def used(self):
        current = _current.get()
        if current is None:
            return self
        if self.supervisor is None:
            current._mark_unknown("unregistered_worker_process_resource")
            raise ResourceCleanupUncertain("unregistered_worker_process_resource")
        if current.worker_instance != self.supervisor.worker_instance:
            current._mark_unknown("resource_worker_mismatch")
            raise ResourceCleanupUncertain("resource_worker_mismatch")
        try:
            accepted = current.repository.use_resource(current.worker_instance, self.resource_id, current.task_id)
            if accepted is False:
                self.unknown("resource_use_registration_rejected")
        except ResourceCleanupUncertain as exc:
            self._record_unknown(exc.code)
            raise
        except Exception:
            self.unknown("resource_use_registration_failed")
        return self

    def attach_exited_process(self, pid, process_start_token, executable=None):
        """Record a trusted creation descriptor after that process already ended.

        Used for a renderer that wrote its owner record and exited before its
        parent could observe it alive. It cannot certify a still-live identity.
        """
        if type(pid) is not int or pid < 1 or not isinstance(process_start_token, str):
            self.unknown("invalid_exited_process_identity")
        if probe_process(pid, process_start_token, executable=executable) != "exited":
            self.unknown("process_still_present_after_conversion")
        self.pid, self.process_start_token = pid, process_start_token
        self.executable = str(Path(executable).resolve()) if executable else None
        if self.supervisor:
            try:
                self.supervisor._write_owner_descriptor(
                    self, pid=pid, process_start_token=process_start_token,
                    executable=self.executable, owned=True,
                )
                attached = self.supervisor.repository.attach_resource(
                    self.supervisor.worker_instance, self.resource_id,
                    host_id=self.supervisor.identity.host_id, pid=pid,
                    process_start_token=process_start_token, executable=self.executable)
                if attached is False or attached is None:
                    self.unknown("exited_process_identity_rejected")
            except ResourceCleanupUncertain as exc:
                self._record_unknown(exc.code)
                raise
            except Exception:
                self.unknown("exited_process_identity_persistence_failed")
        return self.exited()

    def exited(self):
        if self.pid is not None:
            if not self.process_start_token:
                self.unknown("process_identity_missing")
            if probe_process(self.pid, self.process_start_token, executable=self.executable) != "exited":
                self.unknown("process_exit_not_confirmed")
        # With no attached PID, the caller explicitly certifies that it did
        # not start a process. An uncertain launch must call unknown instead.
        if self.supervisor:
            try:
                accepted = self.supervisor.repository.resource_exited(
                    self.supervisor.worker_instance, self.resource_id,
                    pid=self.pid, process_start_token=self.process_start_token)
                if accepted is False:
                    self.unknown("resource_exit_identity_rejected")
            except ResourceCleanupUncertain:
                raise
            except Exception:
                self.unknown("resource_exit_persistence_failed")
            self.supervisor._mark_exited(self.resource_id)
        current = _current.get()
        if current is not None:
            current._mark_exited(self.resource_id)
        self._verified_exited = True
        return True

    def unknown(self, code="resource_cleanup_uncertain"):
        self._record_unknown(code)
        raise ResourceCleanupUncertain(code)

    def _record_unknown(self, code):
        if self.supervisor:
            self.supervisor._mark_unknown(self.resource_id)
            try:
                self.supervisor.repository.resource_unknown(
                    self.supervisor.worker_instance, self.resource_id, code)
            except Exception:
                pass  # in-memory uncertainty still prevents terminal release
        current = _current.get()
        if current is not None:
            current._mark_unknown(self.resource_id or code)

    def retire_owner(self):
        for supervisor in (self.supervisor, _current.get()):
            if supervisor is not None:
                with supervisor._lock:
                    supervisor._retire = True


def begin_resource(kind, *, scope="task", owner_path=None):
    if kind not in {"pdf", "word", "powershell"} or scope not in {"worker", "task"}:
        raise ValueError("invalid process resource kind/scope")
    supervisor = _current.get()
    if supervisor:
        return supervisor.begin(kind, scope, owner_path)
    return ResourceHandle(kind, scope, str(owner_path) if owner_path is not None else None)


@contextmanager
def resource_supervision_scope(supervisor):
    token = _current.set(supervisor)
    try:
        yield supervisor
    finally:
        _current.reset(token)
