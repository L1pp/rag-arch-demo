"""Exact local recovery for worker-owned child process resources.

Recovery is deliberately conservative. Database lease expiry never proves a
child process exited. A PID-less record can be repaired only from the durable,
per-resource descriptor written before its database attachment, with a bound
worker identity, owner nonce, resource identity and local OS probe.
"""
from __future__ import annotations

from decimal import Decimal
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable

from ..resource_supervision import resource_owner_descriptor_path
from .task_worker import (
    ROOT,
    ProcessProbeState,
    local_host_id,
    probe_process_identity,
    recover_local_workers,
    process_start_token,
)
from .task_repository import ResourceIdentityConflict, WorkerIdentityConflict


_DESCRIPTOR_KEYS = {
    "schema_version", "resource_id", "owner_nonce", "worker_instance",
    "owner_path", "owned", "worker", "resource",
}
_IDENTITY_KEYS = {"host_id", "pid", "process_start_token"}
_RESOURCE_IDENTITY_KEYS = {"host_id", "pid", "process_start_token", "executable"}
_MAX_DESCRIPTOR_BYTES = 64 * 1024
_MAX_WORD_OWNER_BYTES = 64 * 1024


def _norm_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _file_identity(info: os.stat_result) -> dict[str, int | None]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "owner_uid": int(info.st_uid) if hasattr(info, "st_uid") else None,
    }


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, stop_at: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("owner descriptor路径包含符号链接。")
        if current == stop_at:
            return
        parent = current.parent
        if parent == current:
            raise ValueError("owner descriptor路径未位于受信任目录。")
        current = parent


def read_owner_descriptor(resource: dict, worker: dict, *, root: str | Path) -> tuple[dict, dict]:
    """Read one regular, private, canonical descriptor without following links."""
    project_root = Path(root).resolve(strict=True)
    trusted_root = project_root / "data" / "runtime" / "resource_owners"
    expected = resource_owner_descriptor_path(
        project_root, worker["worker_instance"], resource["resource_id"],
    ).absolute()
    stored = resource.get("owner_descriptor_path")
    if not isinstance(stored, str) or _norm_path(stored) != _norm_path(expected):
        raise ValueError("资源未预登记位于runtime/resource_owners内的descriptor路径。")
    candidate = Path(stored).absolute()
    if not _is_under(candidate, trusted_root):
        raise ValueError("owner descriptor不在持久runtime目录。")
    try:
        resolved_candidate = candidate.resolve(strict=True)
        if _norm_path(resolved_candidate) != _norm_path(candidate):
            raise ValueError("owner descriptor路径经过symlink或reparse point。")
        _reject_symlink_components(candidate, project_root)
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"owner descriptor无法读取：{type(exc).__name__}。") from None
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("owner descriptor必须是普通文件。")
    if before.st_size < 1 or before.st_size > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("owner descriptor大小不合法。")
    if os.name != "nt" and stat.S_IMODE(before.st_mode) & 0o022:
        raise ValueError("owner descriptor不能由组或其他用户写入。")
    if hasattr(os, "getuid") and hasattr(before, "st_uid") and before.st_uid != os.getuid():
        raise ValueError("owner descriptor文件属主与当前服务用户不一致。")

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(candidate, flags)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _stat_signature(opened) != _stat_signature(before):
                raise ValueError("owner descriptor在打开时发生替换。")
            raw = stream.read(_MAX_DESCRIPTOR_BYTES + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > _MAX_DESCRIPTOR_BYTES or _stat_signature(after) != _stat_signature(opened):
            raise ValueError("owner descriptor读取期间发生变化。")
        final = candidate.lstat()
        if _stat_signature(final) != _stat_signature(opened):
            raise ValueError("owner descriptor路径在读取期间发生替换。")
        descriptor = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"owner descriptor内容无效：{type(exc).__name__}。") from None
    if not isinstance(descriptor, dict) or set(descriptor) != _DESCRIPTOR_KEYS:
        raise ValueError("owner descriptor字段不符合schema v1。")
    return descriptor, _file_identity(opened)


def _identity_tuple(value: dict) -> tuple[Any, Any, Any]:
    return value.get("host_id"), value.get("pid"), value.get("process_start_token")


def validate_owner_descriptor(descriptor: dict, resource: dict, worker: dict,
                              *, local_host: str) -> dict:
    """Bind descriptor nonce, owner path and parent/child identities to DB rows."""
    if descriptor["schema_version"] != 1:
        raise ValueError("不支持owner descriptor版本。")
    if descriptor["resource_id"] != resource["resource_id"]:
        raise ResourceIdentityConflict("descriptor resource_id与数据库不一致。")
    if descriptor["worker_instance"] != worker["worker_instance"]:
        raise WorkerIdentityConflict("descriptor worker_instance与数据库不一致。")
    if descriptor["owner_nonce"] != resource.get("owner_nonce") or not resource.get("owner_nonce"):
        raise ResourceIdentityConflict("descriptor owner_nonce与数据库不一致。")
    if descriptor["owner_path"] != resource.get("owner_path"):
        raise ResourceIdentityConflict("descriptor owner_path与数据库不一致。")
    if descriptor["owned"] is not True:
        raise ResourceIdentityConflict("descriptor未声明该子进程由worker拥有。")

    parent = descriptor["worker"]
    child = descriptor["resource"]
    if not isinstance(parent, dict) or set(parent) != _IDENTITY_KEYS:
        raise ValueError("descriptor缺少精确parent身份。")
    if not isinstance(child, dict) or set(child) != _RESOURCE_IDENTITY_KEYS:
        raise ValueError("descriptor缺少精确child身份。")
    expected_parent = {
        "host_id": worker["host_id"], "pid": int(worker["pid"]),
        "process_start_token": worker["process_start_token"],
    }
    if parent != expected_parent:
        raise WorkerIdentityConflict("descriptor parent host/PID/start token与worker登记不一致。")
    if parent["host_id"].casefold() != local_host.casefold():
        raise ValueError("descriptor parent不是本机worker。")
    if not isinstance(child["host_id"], str) or child["host_id"].casefold() != local_host.casefold():
        raise ValueError("descriptor child不是本机进程。")
    if isinstance(child["pid"], bool) or not isinstance(child["pid"], int) or child["pid"] < 1:
        raise ValueError("descriptor child PID无效。")
    if not isinstance(child["process_start_token"], str) or not child["process_start_token"].strip():
        raise ValueError("descriptor child start token无效。")
    if not isinstance(child["executable"], str) or not child["executable"].strip():
        raise ValueError("descriptor child executable无效。")
    if not Path(child["executable"]).is_absolute():
        raise ValueError("descriptor child executable必须是绝对路径。")
    if resource.get("pid") is not None:
        registered = (resource.get("host_id"), int(resource["pid"]), resource.get("process_start_token"))
        if registered != _identity_tuple(child):
            raise ResourceIdentityConflict("descriptor child身份与已持久化PID身份不一致。")
    if resource.get("executable") not in (None, child["executable"]):
        raise ResourceIdentityConflict("descriptor executable与数据库不一致。")
    return child


def _probe(host_id: str, pid: int, token: str, executable: str | None, *,
           local_host: str, probe: Callable, psutil_module=None) -> str:
    try:
        outcome = probe(host_id, pid, token, local_host=local_host,
                        psutil_module=psutil_module)
    except TypeError:
        outcome = probe(host_id, pid, token, local_host=local_host)
    if outcome == ProcessProbeState.RUNNING and executable:
        if psutil_module is None:
            try:
                import psutil as psutil_module
            except ImportError:
                return ProcessProbeState.UNKNOWN
        try:
            process = psutil_module.Process(pid)
            actual = os.path.normcase(os.path.realpath(process.exe()))
            expected = os.path.normcase(os.path.realpath(executable))
            if actual != expected:
                return ProcessProbeState.UNKNOWN
        except Exception:
            return ProcessProbeState.UNKNOWN
    return outcome


def _read_word_owner(owner_path: str, child: dict, *, root: Path) -> bool:
    """Validate the secondary Word owner file and normalize its FILETIME token."""
    candidate = Path(owner_path).absolute()
    document_root = root / "data" / "runtime" / "document_pages"
    try:
        resolved = candidate.resolve(strict=True)
        if _norm_path(resolved) != _norm_path(candidate) or not _is_under(resolved, document_root):
            return False
        _reject_symlink_components(resolved, document_root)
        before = resolved.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > _MAX_WORD_OWNER_BYTES:
            return False
        if os.name != "nt" and stat.S_IMODE(before.st_mode) & 0o022:
            return False
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(resolved, flags)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _stat_signature(opened) != _stat_signature(before):
                return False
            raw = stream.read(_MAX_WORD_OWNER_BYTES + 1)
            after = os.fstat(stream.fileno())
        final = resolved.lstat()
        if (len(raw) > _MAX_WORD_OWNER_BYTES
                or _stat_signature(opened) != _stat_signature(after)
                or _stat_signature(final) != _stat_signature(opened)):
            return False
        owner = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(owner, dict):
            return False
        pid = owner.get("pid")
        ticks = owner.get("start_ticks")
        executable = owner.get("executable")
        if owner.get("owned") is not True or pid != child["pid"]:
            return False
        if isinstance(pid, bool) or not isinstance(pid, int):
            return False
        if (not isinstance(ticks, str) or not ticks.isascii() or not ticks.isdigit()):
            return False
        try:
            unix_start = float(
                (Decimal(ticks) - Decimal(621355968000000000)) / Decimal(10000000))
        except Exception:
            return False
        if not math.isfinite(unix_start) or unix_start < 0:
            return False
        token = process_start_token(unix_start)
        if token != child["process_start_token"]:
            return False
        if Path(executable).name.casefold() != "winword.exe":
            return False
        if not isinstance(executable, str) or os.path.normcase(os.path.realpath(executable)) != os.path.normcase(
                os.path.realpath(child["executable"])):
            return False
        return True
    except Exception:
        return False


def _terminate_exact_child(worker: dict, child: dict, *, psutil_module=None,
                            timeout: float = 5.0) -> str:
    """Use a native stable process handle; never signal a bare PID."""
    try:
        from ..owned_process_exit import terminate_exact_process
        outcome = terminate_exact_process(
            child["pid"], child["process_start_token"], child["executable"],
            timeout_seconds=timeout,
        )
    except Exception:
        return ProcessProbeState.UNKNOWN
    return ProcessProbeState.EXITED if outcome == ProcessProbeState.EXITED else ProcessProbeState.UNKNOWN


def _mark_exited(repository, worker_instance: str, resource: dict) -> bool:
    return repository.resource_exited(
        worker_instance, resource["resource_id"], pid=resource.get("pid"),
        process_start_token=resource.get("process_start_token"),
    )


def _recover_office_resource(repository, worker: dict, resource: dict, *, root: Path,
                             local_host: str) -> dict:
    """A launcher exit alone does not prove that its Office process group exited."""
    worker_instance = worker["worker_instance"]
    resource_id = resource["resource_id"]
    try:
        from ..office_conversion import probe_office_owner
        owner_path = Path(resource["owner_path"]).absolute()
        expected = (root / "data/runtime/office_conversion/active.json").absolute()
        if _norm_path(owner_path) != _norm_path(expected):
            raise ValueError("office owner path is outside the conversion directory")
        descriptor, file_identity = read_owner_descriptor(resource, worker, root=root)
        child = validate_owner_descriptor(descriptor, resource, worker, local_host=local_host)
        if child["host_id"].casefold() != local_host.casefold():
            return {"resource_id": resource_id, "state": "unknown", "action": "remote_child"}
        outcome = probe_office_owner(
            owner_path, pid=child["pid"], token=child["process_start_token"],
            executable=child["executable"])
        # Unknown descendants must never be converted into an exited identity by
        # the database backfill, even when the launcher itself has disappeared.
        proven_exited = outcome == ProcessProbeState.EXITED
        if resource.get("pid") is None:
            updated = repository.backfill_unknown_resource_identity(
                worker_instance, resource_id,
                owner_descriptor_path=resource["owner_descriptor_path"],
                descriptor=descriptor, descriptor_file_identity=file_identity,
                probe_state="exited" if proven_exited else "unknown")
            if proven_exited:
                return {"resource_id": resource_id, "state": updated["state"],
                        "action": "office_group_exit_backfilled"}
        elif proven_exited:
            accepted = _mark_exited(repository, worker_instance, resource)
            return {"resource_id": resource_id,
                    "state": "exited" if accepted else "unknown",
                    "action": "office_group_exited" if accepted else "exit_identity_rejected"}
    except Exception as exc:
        if resource["state"] != "unknown":
            repository.resource_unknown(worker_instance, resource_id, "office_owner_untrusted")
        return {"resource_id": resource_id, "state": "unknown",
                "action": "office_owner_untrusted", "reason": type(exc).__name__}
    if resource["state"] != "unknown":
        repository.resource_unknown(worker_instance, resource_id, "office_group_exit_unconfirmed")
    return {"resource_id": resource_id, "state": "unknown", "action": "office_group_exit_unconfirmed"}


def _recover_one_resource(repository, worker: dict, resource: dict, *, root: Path,
                          cleanup_owned: bool, local_host: str, probe: Callable,
                          psutil_module=None) -> dict:
    worker_instance = worker["worker_instance"]
    resource_id = resource["resource_id"]
    state = resource["state"]
    pid = resource.get("pid")
    token = resource.get("process_start_token")
    host_id = resource.get("host_id")
    executable = resource.get("executable")

    if state == "exited":
        return {"resource_id": resource_id, "state": "exited", "action": "already_exited"}
    if (resource.get("kind") == "word" and resource.get("owner_path")
            and Path(resource["owner_path"]).name == "active.json"):
        return _recover_office_resource(repository, worker, resource, root=root,
                                        local_host=local_host)
    if state == "starting" and pid is None:
        repository.resource_unknown(worker_instance, resource_id, "worker_exited_before_resource_attach")
        resource = {**resource, "state": "unknown"}
        state = "unknown"
    if pid is None:
        try:
            descriptor, file_identity = read_owner_descriptor(resource, worker, root=root)
            child = validate_owner_descriptor(descriptor, resource, worker, local_host=local_host)
        except Exception as exc:
            if state != "unknown":
                repository.resource_unknown(worker_instance, resource_id, "owner_descriptor_untrusted")
            return {"resource_id": resource_id, "state": "unknown",
                    "action": "identity_unavailable", "reason": type(exc).__name__}
        child_host = child["host_id"]
        child_pid = int(child["pid"])
        child_token = child["process_start_token"]
        child_executable = child["executable"]
        outcome = _probe(child_host, child_pid, child_token, child_executable,
                         local_host=local_host, probe=probe, psutil_module=psutil_module)
        if outcome == ProcessProbeState.REMOTE:
            return {"resource_id": resource_id, "state": "unknown", "action": "remote_child"}
        updated = repository.backfill_unknown_resource_identity(
            worker_instance, resource_id,
            owner_descriptor_path=resource["owner_descriptor_path"],
            descriptor=descriptor, descriptor_file_identity=file_identity,
            probe_state=outcome if outcome in ("running", "exited") else "unknown",
        )
        if outcome == ProcessProbeState.EXITED:
            return {"resource_id": resource_id, "state": updated["state"],
                    "action": "identity_backfilled_exited"}
        if outcome == ProcessProbeState.RUNNING and cleanup_owned:
            if resource.get("kind") == "word" and resource.get("owner_path"):
                if not _read_word_owner(resource["owner_path"], child, root=root):
                    return {"resource_id": resource_id, "state": "unknown",
                            "action": "word_owner_not_trusted"}
            stopped = _terminate_exact_child(worker, child, psutil_module=psutil_module)
            after = _probe(child_host, child_pid, child_token, child_executable,
                           local_host=local_host, probe=probe, psutil_module=psutil_module)
            if stopped == ProcessProbeState.EXITED and after == ProcessProbeState.EXITED:
                accepted = repository.resource_exited(
                    worker_instance, resource_id, pid=child_pid,
                    process_start_token=child_token,
                )
                return {"resource_id": resource_id,
                        "state": "exited" if accepted else "unknown",
                        "action": "owned_child_terminated" if accepted else "exit_identity_rejected"}
            return {"resource_id": resource_id, "state": "unknown", "action": "cleanup_not_confirmed"}
        return {"resource_id": resource_id, "state": "unknown",
                "action": "child_running" if outcome == ProcessProbeState.RUNNING else "child_probe_unknown"}

    if not isinstance(host_id, str) or not isinstance(pid, int) or not isinstance(token, str):
        repository.resource_unknown(worker_instance, resource_id, "resource_identity_incomplete")
        return {"resource_id": resource_id, "state": "unknown", "action": "identity_incomplete"}
    if host_id.casefold() != local_host.casefold():
        return {"resource_id": resource_id, "state": "unknown", "action": "remote_child"}
    outcome = _probe(host_id, pid, token, executable,
                     local_host=local_host, probe=probe, psutil_module=psutil_module)
    if outcome == ProcessProbeState.EXITED:
        accepted = _mark_exited(repository, worker_instance, resource)
        return {"resource_id": resource_id,
                "state": "exited" if accepted else "unknown",
                "action": "exact_identity_exited" if accepted else "exit_identity_rejected"}
    if state != "unknown":
        repository.resource_unknown(worker_instance, resource_id,
                                    "resource_process_still_running" if outcome == ProcessProbeState.RUNNING
                                    else "resource_process_probe_unknown")
    if outcome == ProcessProbeState.RUNNING and cleanup_owned:
        try:
            descriptor, _ = read_owner_descriptor(resource, worker, root=root)
            child = validate_owner_descriptor(descriptor, resource, worker, local_host=local_host)
        except Exception as exc:
            return {"resource_id": resource_id, "state": "unknown",
                    "action": "cleanup_descriptor_untrusted", "reason": type(exc).__name__}
        if resource.get("kind") == "word" and resource.get("owner_path"):
            if not _read_word_owner(resource["owner_path"], child, root=root):
                return {"resource_id": resource_id, "state": "unknown",
                        "action": "word_owner_not_trusted"}
        stopped = _terminate_exact_child(worker, child, psutil_module=psutil_module)
        after = _probe(host_id, pid, token, executable,
                       local_host=local_host, probe=probe, psutil_module=psutil_module)
        if stopped == ProcessProbeState.EXITED and after == ProcessProbeState.EXITED:
            accepted = _mark_exited(repository, worker_instance, resource)
            return {"resource_id": resource_id,
                    "state": "exited" if accepted else "unknown",
                    "action": "owned_child_terminated" if accepted else "exit_identity_rejected"}
    return {"resource_id": resource_id, "state": "unknown",
            "action": "child_running" if outcome == ProcessProbeState.RUNNING else "child_probe_unknown"}


def recover_resources(repository, *, cleanup_owned: bool = False,
                      root: str | Path = ROOT, probe: Callable = probe_process_identity,
                      psutil_module=None) -> list[dict[str, Any]]:
    """Expire leases, prove parent exits, inspect child identities and retry simulation recovery."""
    if type(cleanup_owned) is not bool:
        raise ValueError("cleanup_owned必须是bool。")
    project_root = Path(root).resolve(strict=True)
    local_host = local_host_id()
    marked_stale = repository.mark_stale_workers()
    def parent_probe(host_id, pid, token, *, local_host=None):
        return _probe(host_id, pid, token, None, local_host=local_host or local_host_id(),
                      probe=probe, psutil_module=psutil_module)

    parent_checks = recover_local_workers(repository, probe=parent_probe)
    outcomes: list[dict[str, Any]] = [{
        "kind": "lease_check", "marked_stale_workers": marked_stale,
        "parent_checks": parent_checks,
    }]
    for worker in repository.workers():
        worker_instance = worker.get("worker_instance")
        if not worker_instance:
            continue
        state = str(worker.get("state", "")).lower()
        worker_host = worker.get("host_id")
        if (state != "exited" or not isinstance(worker_host, str)
                or worker_host.casefold() != local_host.casefold()):
            continue
        parent_state = _probe(
            worker["host_id"], int(worker["pid"]), worker["process_start_token"],
            None, local_host=local_host, probe=probe, psutil_module=psutil_module,
        )
        if parent_state != ProcessProbeState.EXITED:
            outcomes.append({"worker_instance": worker_instance,
                             "state": parent_state, "resources": [],
                             "recovered_tasks": []})
            continue
        resources = repository.resources_for_worker(worker_instance)
        resource_outcomes = [
            _recover_one_resource(
                repository, worker, resource, root=project_root,
                cleanup_owned=cleanup_owned, local_host=local_host,
                probe=probe, psutil_module=psutil_module,
            )
            for resource in resources
        ]
        recovered_tasks = repository.recover_confirmed_exit(
            worker_instance, worker["process_start_token"],
        )
        outcomes.append({
            "kind": "worker_resources",
            "worker_instance": worker_instance,
            "state": ProcessProbeState.EXITED,
            "resources": resource_outcomes,
            "recovered_tasks": recovered_tasks,
        })
    return outcomes


def resolve_task(repository, task_id: str, proof: dict, *, root: str | Path = ROOT,
                 probe: Callable = probe_process_identity,
                 psutil_module=None) -> bool:
    """Re-probe local worker/children, then write the administrator resolution audit."""
    if not isinstance(proof, dict) or not isinstance(proof.get("worker"), dict):
        raise ValueError("proof必须包含worker身份。")
    worker_proof = proof["worker"]
    worker_instance = worker_proof.get("worker_instance")
    if not isinstance(worker_instance, str):
        raise ValueError("proof.worker.worker_instance无效。")
    worker = repository.get_worker(worker_instance)
    if worker is None or worker.get("state") != "exited":
        return False
    local_host = local_host_id()
    parent_state = _probe(
        worker["host_id"], int(worker["pid"]), worker["process_start_token"],
        None, local_host=local_host, probe=probe, psutil_module=psutil_module,
    )
    if parent_state != ProcessProbeState.EXITED:
        return False
    resources = repository.resources_for_worker(worker_instance)
    proof_resources = proof.get("resources")
    if not isinstance(proof_resources, list):
        raise ValueError("proof.resources必须是数组。")
    by_id = {}
    for entry in proof_resources:
        if not isinstance(entry, dict) or not isinstance(entry.get("resource_id"), str):
            raise ValueError("proof.resources缺少resource_id。")
        if entry["resource_id"] in by_id:
            raise ValueError("proof.resources存在重复resource_id。")
        by_id[entry["resource_id"]] = entry
    if set(by_id) != {row["resource_id"] for row in resources}:
        raise ResourceIdentityConflict("proof必须精确覆盖该worker的所有资源。")
    for resource in resources:
        if resource["state"] != "exited":
            return False
        entry = by_id[resource["resource_id"]]
        expected = (resource.get("host_id"), resource.get("pid"), resource.get("process_start_token"))
        supplied = (entry.get("host_id"), entry.get("pid"), entry.get("process_start_token"))
        if expected != supplied:
            raise ResourceIdentityConflict("proof资源身份与数据库不一致。")
        if resource.get("pid") is None:
            if expected != (None, None, None):
                raise ResourceIdentityConflict("无PID资源身份列不一致。")
            continue
        descriptor = None
        if resource.get("owner_descriptor_path"):
            descriptor, _ = read_owner_descriptor(resource, worker, root=root)
            child = validate_owner_descriptor(
                descriptor, resource, worker, local_host=local_host,
            )
            if _identity_tuple(child) != expected:
                raise ResourceIdentityConflict("proof对应的child身份不匹配owner descriptor。")
        if resource["host_id"].casefold() != local_host.casefold():
            return False
        outcome = _probe(
            resource["host_id"], int(resource["pid"]), resource["process_start_token"],
            resource.get("executable"), local_host=local_host,
            probe=probe, psutil_module=psutil_module,
        )
        if outcome != ProcessProbeState.EXITED:
            return False
    return repository.resolve_confirmed_task_by_proof(task_id, proof)


def maintain_recovery(repository, *, cleanup_owned: bool = True,
                      cooldown_seconds: float = 120.0, root: str | Path = ROOT,
                      probe: Callable = probe_process_identity,
                      psutil_module=None) -> list[dict[str, Any]]:
    """Isolate interrupted attempts after proving all local execution has ended.

    Remote request completion is deliberately not inferred from a timeout or
    local process exit. The cooldown only spaces recovery and subsequent work;
    audit records continue to say that external work is unknown. This path never
    retries the interrupted task and does not weaken explicit administrator
    resolution. A running, inaccessible or remote local resource still blocks
    release rather than being labelled exited.
    """
    if (isinstance(cooldown_seconds, bool) or not isinstance(cooldown_seconds, (int, float))
            or not math.isfinite(cooldown_seconds) or cooldown_seconds < 0):
        raise ValueError("cooldown_seconds 必须是有限非负数。")
    outcomes = recover_resources(
        repository, cleanup_owned=cleanup_owned, root=root, probe=probe,
        psutil_module=psutil_module,
    )
    local_host = local_host_id()
    for worker in repository.workers():
        worker_instance = worker.get("worker_instance")
        if not worker_instance or worker.get("state") != "exited":
            continue
        worker_host = worker.get("host_id")
        if not isinstance(worker_host, str) or worker_host.casefold() != local_host.casefold():
            continue
        tasks = repository.recovery_tasks_for_worker(worker_instance)
        if not tasks:
            continue
        report = {"kind": "task_isolation", "worker_instance": worker_instance,
                  "isolated_tasks": [], "blocked": [],
                  "external_work_state": "unknown", "automatic_retry": False}
        outcomes.append(report)
        parent_state = _probe(
            worker_host, int(worker["pid"]), worker["process_start_token"], None,
            local_host=local_host, probe=probe, psutil_module=psutil_module,
        )
        if parent_state != ProcessProbeState.EXITED:
            report["blocked"].append("worker_exit_unconfirmed")
            continue
        exited_at = worker.get("exited_at")
        if exited_at is None or time.time() < float(exited_at) + cooldown_seconds:
            report["blocked"].append("cooldown")
            if exited_at is not None:
                report["eligible_after"] = float(exited_at) + cooldown_seconds
            continue
        resources = repository.resources_for_worker(worker_instance)
        for resource in resources:
            if resource.get("state") != "exited":
                report["blocked"].append("resource_exit_unconfirmed")
                break
            identity = (resource.get("host_id"), resource.get("pid"),
                        resource.get("process_start_token"))
            if identity == (None, None, None):
                # Exited-before-attach is accepted only after the existing
                # resource lifecycle has durably established no child exists.
                continue
            child_state = _probe(
                resource.get("host_id"), resource.get("pid"),
                resource.get("process_start_token"), resource.get("executable"),
                local_host=local_host, probe=probe, psutil_module=psutil_module,
            )
            if child_state != ProcessProbeState.EXITED:
                report["blocked"].append("resource_exit_unconfirmed")
                break
        if report["blocked"]:
            continue
        proof = {
            "operator": "scheduled-maintenance",
            "reason": "Exact local worker and all registered resources exited; isolate without retry.",
            "resolution_kind": "local_exit_isolation", "external_work_state": "unknown",
            "automatic_retry": False,
            "worker": {key: worker[key] for key in (
                "worker_instance", "host_id", "pid", "process_start_token")},
            "resources": [{key: resource.get(key) for key in (
                "resource_id", "host_id", "pid", "process_start_token")}
                for resource in resources],
        }
        for task in tasks:
            accepted = repository.isolate_confirmed_local_exit(
                worker_instance, task["task_id"], task["attempt_token"], proof,
                cooldown_seconds=cooldown_seconds,
            )
            if accepted:
                report["isolated_tasks"].append(task["task_id"])
            else:
                report["blocked"].append("task_state_changed_or_resources_unconfirmed")
    return outcomes


__all__ = [
    "read_owner_descriptor", "recover_resources", "maintain_recovery", "resolve_task",
    "validate_owner_descriptor",
]
