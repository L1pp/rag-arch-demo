"""Bounded, supervised LibreOffice conversions for the Linux web service.

The on-disk lock is shared by the API and worker.  A separate, durable owner
record survives a process crash; an unconfirmed record blocks the next run.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from .owned_process_exit import terminate_exact_process
from .request_control import check_request_control
from .resource_supervision import (
    ResourceCleanupUncertain, begin_resource, probe_process, process_token,
)


MAX_OFFICE_BYTES = 128 * 1024 * 1024
_POLL_SECONDS = 0.1
_CLEANUP_SECONDS = 5.0
_MAX_LOG_BYTES = 64 * 1024
_MARKER_NAME = "active.json"
_SCHEMA = "rag-arch-libreoffice-owner-v1"
_RESULT_SCHEMA = "rag-arch-libreoffice-result-v1"
_LOG = logging.getLogger(__name__)

# A fresh profile has no trusted locations.  These settings explicitly disable
# script execution, OLE/DDE active content and automatic update of linked
# documents; they do not turn LibreOffice into an OS security sandbox.
_PROFILE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
 <item oor:path="/org.openoffice.Office.Common/Security/Scripting">
  <prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop>
  <prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop>
  <prop oor:name="DisableActiveContent" oor:op="fuse"><value>true</value></prop>
  <prop oor:name="BlockUntrustedRefererLinks" oor:op="fuse"><value>true</value></prop>
 </item>
 <item oor:path="/org.openoffice.Office.Writer/Content/Update">
  <prop oor:name="Link" oor:op="fuse"><value>2</value></prop>
  <prop oor:name="Field" oor:op="fuse"><value>false</value></prop>
  <prop oor:name="Chart" oor:op="fuse"><value>false</value></prop>
 </item>
</oor:items>
"""


class _PrivateJSONError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _not_link(path: Path) -> bool:
    return not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction())


def _private_directory(root: Path) -> Path:
    current = root
    for part in ("data", "runtime", "office_conversion"):
        current = current / part
        # API and worker may prepare this tree for the first time together.
        # mkdir(exist_ok=True) tolerates that race; every result is then checked.
        current.mkdir(mode=0o700, exist_ok=True)
        info = current.lstat()
        if not _not_link(current) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("文档转换目录包含不安全的链接或非目录。")
    if os.name == "posix":
        if current.stat().st_uid != os.getuid():
            raise ValueError("文档转换目录不属于当前服务账户。")
        current.chmod(0o700)
    return current


def _atomic_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError("文档转换状态文件不能是链接。")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_marker(path: Path, value: dict) -> None:
    """Publish a new owner record without ever replacing an existing marker."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def _load_private_json(path: Path) -> dict:
    if not _not_link(path):
        raise _PrivateJSONError("symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _PrivateJSONError("not_regular")
        if info.st_size > 65536:
            raise _PrivateJSONError("oversized")
        if os.name == "posix" and info.st_uid != os.getuid():
            raise _PrivateJSONError("owner_mismatch")
        if os.name == "posix" and info.st_mode & 0o077:
            raise _PrivateJSONError("permissions")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            value = json.load(stream)
        if not isinstance(value, dict):
            raise _PrivateJSONError("not_object")
        return value
    finally:
        if fd >= 0:
            os.close(fd)


def _descendants() -> list[dict]:
    """Snapshot children of this dedicated supervisor, including adopted ones."""
    import psutil

    probe_pid = None
    probe_stage = "children"
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
        found = []
        for child in children:
            probe_pid = child.pid
            try:
                probe_stage = "status"
                if child.status() == psutil.STATUS_ZOMBIE:
                    continue
                probe_stage = "executable"
                executable = child.exe()
                if not executable:
                    raise OSError("missing executable")
                probe_stage = "start_token"
                token = process_token(child.pid)
                found.append({"pid": child.pid, "token": token,
                              "executable": str(Path(executable).resolve())})
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return found
    except (psutil.AccessDenied, OSError) as exc:
        error = ResourceCleanupUncertain("office_descendant_probe_failed")
        error.probe_details = {
            "stage": probe_stage,
            "pid": probe_pid if type(probe_pid) is int else None,
            "error_type": type(exc).__name__[:64],
            "errno": exc.errno if type(getattr(exc, "errno", None)) is int else None,
            "reason": ("empty_executable" if probe_stage == "executable"
                       and type(exc) is OSError and str(exc) == "missing executable"
                       else "identity_unavailable"),
        }
        raise error from exc


def _enable_subreaper() -> None:
    """Make every orphaned conversion descendant a child of this supervisor."""
    if not sys.platform.startswith("linux"):
        raise OSError("Linux child subreaper is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise OSError("PR_GET_CHILD_SUBREAPER did not confirm the setting")


def _reap_remaining_children() -> bool:
    """Return True while an adopted child is still alive or not yet waitable."""
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        except InterruptedError:
            continue
        if pid == 0:
            return True


def _supervise(run: Path, command: list[str]) -> int:
    """Run as a dedicated Linux child so adopted descendants can be waited for.

    A completion record is written only after waitpid proves there are no
    remaining children.  The parent retains the active marker without it.
    """
    result_path = run / "result.json"
    cancel_path = run / "cancel.json"
    owner = {"schema": _RESULT_SCHEMA, "pid": os.getpid(),
             "token": process_token(os.getpid()), "descendants_exited": False,
             "descendant_probe_failures": 0}
    child = None
    cancellation_at = None
    status = "start_failed"
    exit_code = None
    captured = bytearray()
    reader = None
    try:
        _enable_subreaper()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=str(run))
        def drain_output():
            while True:
                block = child.stdout.read(4096)
                if not block:
                    return
                remaining = max(0, _MAX_LOG_BYTES - len(captured))
                captured.extend(block[:remaining])
        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        previous_children = None
        while True:
            cancel_requested = cancel_path.exists()
            if cancel_requested:
                if cancellation_at is None:
                    cancellation_at = time.monotonic()
            try:
                children = _descendants()
            except ResourceCleanupUncertain as exc:
                if exc.code != "office_descendant_probe_failed":
                    raise
                # This snapshot is for observation and exact cancellation, not
                # an exit proof.  Keep the last trusted list and keep the
                # subreaper alive; waitpid remains the only completion proof.
                owner["descendant_probe_failures"] += 1
                owner["last_descendant_probe_failure"] = getattr(exc, "probe_details", {
                    "stage": "unknown", "pid": None, "error_type": "unknown",
                    "errno": None, "reason": "identity_unavailable",
                })
                children = None
            if children is not None and children != previous_children:
                _atomic_json(run / "children.json", {"children": children})
                previous_children = children
            if cancel_requested and children is not None:
                # Stop the parent first, then any adopted children, using exact
                # PID/start/executable identity rather than process names.
                children.sort(key=lambda item: item["pid"] != child.pid)
                for item in children:
                    stopped = terminate_exact_process(item["pid"], item["token"],
                                                      item["executable"], timeout_seconds=0.5)
                    if stopped != "exited" and probe_process(item["pid"], item["token"],
                                                              executable=item["executable"]) != "exited":
                        owner["failed_process"] = item
                        raise ResourceCleanupUncertain("office_descendant_cleanup_unconfirmed")
            exit_code = child.poll()
            if exit_code is not None and not _reap_remaining_children():
                reader.join(timeout=1)
                if reader.is_alive():
                    raise ResourceCleanupUncertain("office_log_pipe_unconfirmed")
                owner["descendants_exited"] = True
                status = "cancelled" if cancellation_at is not None else (
                    "success" if exit_code == 0 else "conversion_failed")
                break
            if cancellation_at is not None and time.monotonic() - cancellation_at > _CLEANUP_SECONDS:
                raise ResourceCleanupUncertain("office_descendant_cleanup_unconfirmed")
            time.sleep(_POLL_SECONDS)
    except OSError as exc:
        if child is None:  # no external process was launched
            owner["descendants_exited"] = True
        status = "start_failed"
        owner["error_type"] = type(exc).__name__[:64]
        owner["error_code"] = f"errno_{exc.errno}" if type(exc.errno) is int else "os_error"
        owner["detail"] = owner["error_code"]
    except BaseException as exc:
        status = "cleanup_uncertain"
        owner["error_type"] = type(exc).__name__[:64]
        owner["error_code"] = exc.code if isinstance(exc, ResourceCleanupUncertain) else "supervisor_error"
        owner["detail"] = owner["error_code"]
    owner.update(status=status, exit_code=exit_code)
    try:
        (run / "converter.log").write_bytes(bytes(captured))
        _atomic_json(result_path, owner)
    except OSError:
        return 2
    return 0 if owner["descendants_exited"] else 2


def _read_completion(run: Path, *, pid: int, token: str,
                     diagnostic: dict | None = None) -> dict | None:
    def reject(reason: str, **fields):
        if diagnostic is not None:
            diagnostic.clear()
            diagnostic.update(reason=reason, **fields)
        return None

    try:
        result = _load_private_json(run / "result.json")
    except FileNotFoundError:
        return reject("missing")
    except PermissionError:
        return reject("permissions")
    except _PrivateJSONError as exc:
        return reject(exc.reason)
    except json.JSONDecodeError:
        return reject("invalid_json")
    except OSError as exc:
        return reject("read_error", error_type=type(exc).__name__)
    except ValueError:
        return reject("invalid_value")
    required = ("schema", "pid", "token", "descendants_exited", "status")
    missing = [key for key in required if key not in result]
    if missing:
        return reject("missing_fields", fields=missing)
    if result["schema"] != _RESULT_SCHEMA:
        return reject("schema_mismatch")
    observed_pid = result["pid"]
    if type(observed_pid) is not int or observed_pid != pid:
        return reject("pid_mismatch", expected_pid=pid,
                      observed_pid=observed_pid if type(observed_pid) is int else None)
    observed_token = result["token"]
    if not isinstance(observed_token, str) or observed_token != token:
        return reject("token_mismatch", expected_token=token,
                      observed_token=observed_token if isinstance(observed_token, str)
                      and re.fullmatch(r"\d{1,20}\.\d{6}", observed_token, re.ASCII) else None)
    if result["descendants_exited"] is not True:
        return reject("descendants_unconfirmed")
    observed_status = result["status"]
    if not isinstance(observed_status, str) or observed_status not in {
            "success", "conversion_failed", "start_failed", "cancelled"}:
        return reject("invalid_status", observed_status=observed_status
                      if isinstance(observed_status, str)
                      and re.fullmatch(r"[a-z_]{1,64}", observed_status, re.ASCII) else None)
    return result


def probe_office_owner(path: str | Path, *, pid: int, token: str,
                       executable: str) -> str:
    """Read-only recovery probe for a recorded Linux office conversion.

    A missing/invalid marker, another host, PID reuse, an inaccessible process,
    or an unrecognised platform is intentionally reported as ``unknown``.
    """
    try:
        marker = Path(path).absolute()
        if (marker.name != _MARKER_NAME or marker.parent.name != "office_conversion"
                or marker.parent.parent.name != "runtime"
                or marker.parent.parent.parent.name != "data"
                or not _not_link(marker.parent) or not _not_link(marker)):
            return "unknown"
        if not sys.platform.startswith("linux"):
            return "unknown"
        data = _load_private_json(marker)
        if (data.get("schema") != _SCHEMA or data.get("host") != socket.gethostname()
                or data.get("pid") != pid or data.get("token") != token
                or os.path.realpath(data.get("executable", "")) != os.path.realpath(executable)
                or data.get("group_id") != pid):
            return "unknown"
        run = Path(data.get("run_directory", "")).absolute()
        runs = marker.parent / "runs"
        if (not _not_link(runs) or not _not_link(run) or run.parent != runs
                or not run.is_dir()):
            return "unknown"
        leader = probe_process(pid, token, executable=executable)
        if leader == "unknown":
            return "unknown"
        if leader == "running":
            return "running"
        # A dead supervisor alone says nothing about orphaned descendants.
        return "exited" if _read_completion(run, pid=pid, token=token) else "unknown"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            ResourceCleanupUncertain):
        return "unknown"


class OfficeConverter:
    def __init__(self, project_root: str | Path, *, timeout: float = 120,
                 executable: str | None = None):
        self.root = Path(project_root).resolve()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("文档转换超时必须为有限正数。")
        self.timeout = float(timeout)
        self.executable = executable

    def _executable(self) -> str:
        if not sys.platform.startswith("linux"):
            raise ValueError("当前 DOC 转换只支持 Linux；Windows 用户请先将 DOC 另存为 DOCX 后上传。")
        configured = self.executable or os.environ.get("RAG_ARCH_LIBREOFFICE_EXECUTABLE")
        selected = configured or shutil.which("soffice") or shutil.which("libreoffice")
        if not selected:
            raise ValueError("未找到 LibreOffice，请安装 LibreOffice Writer 或配置转换程序路径。")
        selected = shutil.which(selected) or selected
        resolved = Path(selected).expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError("LibreOffice 转换程序不存在或不可执行。")
        # Keep LibreOffice's official launcher: it prepares the first profile
        # and may restart/spawn the actual binary.  The dedicated Linux
        # subreaper supervises and waits for its complete descendant tree.
        return str(resolved)

    @contextmanager
    def _lock(self, directory: Path, deadline: float):
        lock_path = directory / "conversion.lock"
        if not _not_link(lock_path):
            raise ValueError("文档转换锁不能是链接。")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("文档转换锁不是普通文件。")
            if os.name == "nt":
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
            else:
                import fcntl
            acquired = False
            while not acquired:
                check_request_control()
                if time.monotonic() >= deadline:
                    raise TimeoutError("等待文档转换队列超时。")
                try:
                    if os.name == "nt":
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except (BlockingIOError, PermissionError):
                    time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
            try:
                yield
            finally:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _check_output(path: Path, output_format: str) -> None:
        try:
            if (not path.is_file() or not _not_link(path)
                    or not 0 < path.stat().st_size <= MAX_OFFICE_BYTES):
                raise ValueError("转换结果为空或超过大小限制。")
            if output_format == "docx":
                with ZipFile(path) as package:
                    names = package.namelist()
                    if ("[Content_Types].xml" not in names or "word/document.xml" not in names
                            or sum(item.file_size for item in package.infolist()) > MAX_OFFICE_BYTES):
                        raise ValueError("转换结果不是有效的 DOCX。")
                # Converted files must satisfy the same XML/active-content
                # checks that the page service requires before rendering.
                from .document_pages import _check_docx
                _check_docx(path)
            else:
                import pymupdf
                with pymupdf.open(path) as document:
                    if not document.is_pdf or document.needs_pass or len(document) < 1:
                        raise ValueError("转换结果不是可读取的 PDF。")
        except (OSError, BadZipFile, RuntimeError) as exc:
            raise ValueError("转换结果损坏或格式不符。") from exc

    def _target(self, target: str | Path) -> Path:
        target = Path(target).absolute()
        runtime = self.root / "data" / "runtime"
        try:
            relative = target.relative_to(runtime)
        except ValueError:
            raise ValueError("文档转换输出必须位于项目运行时目录内。")
        current = runtime
        for part in relative.parts:
            current = current / part
            if not _not_link(current):
                raise ValueError("文档转换输出路径不能包含链接。")
        if not target.parent.resolve().is_relative_to(runtime.resolve()):
            raise ValueError("文档转换输出必须位于项目运行时目录内。")
        return target

    @staticmethod
    def _source_stamp(source: Path) -> dict:
        info = source.stat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_OFFICE_BYTES:
            raise ValueError("文档来源不是可转换的普通文件，或超过大小限制。")
        return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}

    def ensure_docx(self, source: str | Path, target: str | Path) -> None:
        self._ensure(source, target, source_suffix=".doc", output_format="docx")

    def ensure_pdf(self, source: str | Path, target: str | Path) -> None:
        self._ensure(source, target, source_suffix=".docx", output_format="pdf")

    def _ensure(self, source: str | Path, target: str | Path, *, source_suffix: str,
                output_format: str) -> None:
        source = Path(source).resolve(strict=True)
        target = self._target(target)
        if source.suffix.lower() != source_suffix or target.suffix.lower() != "." + output_format:
            raise ValueError("文档转换来源或目标格式不符。")
        directory = _private_directory(self.root)
        deadline = time.monotonic() + self.timeout
        with self._lock(directory, deadline):
            stamp = self._source_stamp(source)
            if source_suffix == ".docx":
                from .document_pages import _check_docx
                _check_docx(source)
            sidecar = target.with_name(target.name + ".source.json")
            if not _not_link(sidecar):
                raise ValueError("文档转换状态文件不能是链接。")
            if target.exists() and sidecar.exists():
                try:
                    if _load_private_json(sidecar) == stamp:
                        self._check_output(target, output_format)
                        if self._source_stamp(source) != stamp:
                            raise ValueError("转换期间文档来源发生变化，请重试。")
                        return
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            # A stale owner prevents new processes, not reuse of an already
            # validated result produced before that owner appeared.
            self._check_marker(directory)
            self._convert_locked(source, target, output_format, directory, deadline, stamp)
            _atomic_json(sidecar, stamp)

    def normalized_docx(self, source: str | Path, target: str | Path) -> Path:
        self.ensure_docx(source, target)
        return Path(target)

    def render_pdf(self, source: str | Path, target: str | Path) -> None:
        self.convert(source, target, output_format="pdf")

    def convert(self, source: str | Path, target: str | Path, *, output_format: str) -> None:
        source = Path(source).resolve(strict=True)
        target = self._target(target)
        if output_format not in {"pdf", "docx"} or target.suffix.lower() != "." + output_format:
            raise ValueError("文档转换输出格式无效。")
        if source.suffix.lower() not in {".doc", ".docx"}:
            raise ValueError("文档转换仅支持 DOC 和 DOCX。")
        directory = _private_directory(self.root)
        deadline = time.monotonic() + self.timeout
        with self._lock(directory, deadline):
            self._check_marker(directory)
            self._convert_locked(source, target, output_format, directory, deadline,
                                 self._source_stamp(source))

    @staticmethod
    def _check_marker(directory: Path) -> None:
        marker = directory / _MARKER_NAME
        if marker.exists() or marker.is_symlink():
            # This request has not launched a child.  The old owner stays in
            # its own recovery record; inventing a new unknown resource here
            # would permanently retain an unrelated task slot.
            raise ValueError("上一次文档转换的进程状态尚未核验，请先处理后重试。")

    def _convert_locked(self, source: Path, target: Path, output_format: str,
                        directory: Path, deadline: float, stamp: dict) -> None:
        if source.suffix.lower() == ".docx":
            # Use the same active-content checks as the existing page service.
            from .document_pages import _check_docx
            _check_docx(source)
        executable = self._executable()
        runs = directory / "runs"
        if runs.exists() and (not _not_link(runs) or not runs.is_dir()):
            raise ValueError("文档转换工作目录不安全。")
        runs.mkdir(mode=0o700, exist_ok=True)
        run = runs / uuid4().hex
        run.mkdir(mode=0o700)
        marker = directory / _MARKER_NAME
        resource = None
        process = None
        confirmed = False
        marker_created = False
        succeeded = False
        identity_attached = False
        try:
            input_dir, output_dir = run / "input", run / "output"
            input_dir.mkdir(mode=0o700)
            output_dir.mkdir(mode=0o700)
            staged = input_dir / ("source" + source.suffix.lower())
            shutil.copyfile(source, staged)
            if self._source_stamp(source) != stamp:
                raise ValueError("转换期间文档来源发生变化，请重试。")
            profile = run / "profile"
            profile_user = profile / "user"
            profile_user.mkdir(parents=True, mode=0o700)
            (profile_user / "registrymodifications.xcu").write_text(_PROFILE_XML, encoding="utf-8")
            check_request_control()
            if time.monotonic() >= deadline:
                raise TimeoutError("等待文档转换队列超时。")
            resource = begin_resource("word", owner_path=marker)
            data = {"schema": _SCHEMA, "host": socket.gethostname(), "phase": "starting",
                    "pid": None, "token": None, "executable": None, "group_id": None,
                    "children": [], "run_directory": str(run),
                    "resource_id": getattr(resource, "resource_id", None)}
            _create_marker(marker, data)
            marker_created = True
            command = [executable, "-env:UserInstallation=" + profile.as_uri(),
                       "--headless", "--nologo", "--nodefault", "--norestore",
                       "--convert-to", "pdf:writer_pdf_Export" if output_format == "pdf"
                       else "docx:Office Open XML Text", "--outdir", str(output_dir), str(staged)]
            # Keep the virtualenv launcher path for execution: resolving it
            # first would silently select the system Python and its packages.
            supervisor_launcher = os.path.abspath(sys.executable)
            supervisor_executable = str(Path(supervisor_launcher).resolve())
            supervisor_command = [supervisor_launcher, "-m", "rag_arch.office_conversion",
                                  "--supervise", str(run), *command]
            inherited = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "LD_LIBRARY_PATH", "TZ")
            child_env = {name: os.environ[name] for name in inherited if name in os.environ}
            # LibreOffice and its launcher may consult these even with an
            # explicit profile.  Keep them writable and private under this
            # conversion rather than relying on a service account's home or
            # systemd's per-service private /tmp.
            for variable, name in (("HOME", "home"), ("XDG_CONFIG_HOME", "xdg-config"),
                                   ("XDG_CACHE_HOME", "xdg-cache"), ("XDG_RUNTIME_DIR", "xdg-runtime"),
                                   ("TMPDIR", "tmp")):
                folder = run / name
                folder.mkdir(mode=0o700)
                child_env[variable] = str(folder)
            child_env["PYTHONPATH"] = os.pathsep.join(
                (str(self.root), str(Path(__file__).resolve().parents[1])))
            try:
                process = subprocess.Popen(supervisor_command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           cwd=str(run), env=child_env, start_new_session=True)
            except OSError as exc:
                resource.exited()  # Popen failed before a child was created
                marker.unlink()
                confirmed = True
                raise ValueError("LibreOffice 无法启动。") from exc
            try:
                token = process_token(process.pid)
                data.update(phase="running", pid=process.pid, token=token,
                            executable=supervisor_executable, group_id=process.pid,
                            converter_executable=executable)
                _atomic_json(marker, data)
                resource.attach_process(process.pid, token, executable=supervisor_executable)
                identity_attached = True
            except BaseException as exc:
                # Signal the already-started guard in the outer cleanup before
                # recording our own uncertainty; attach_process itself may
                # already have recorded uncertainty while propagating failure.
                raise ResourceCleanupUncertain("office_process_identity_unconfirmed") from exc
            while True:
                check_request_control()
                if time.monotonic() >= deadline:
                    raise TimeoutError("LibreOffice 文档转换超时。")
                children_file = run / "children.json"
                if children_file.exists():
                    try:
                        children = _load_private_json(children_file).get("children")
                        if isinstance(children, list) and children != data["children"]:
                            data["children"] = children
                            _atomic_json(marker, data)
                    except (OSError, ValueError, json.JSONDecodeError):
                        resource.unknown("office_child_record_unconfirmed")
                status = process.poll()
                if status is not None:
                    break
                time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
            completion_diagnostic = {}
            result = _read_completion(run, pid=process.pid, token=token,
                                      diagnostic=completion_diagnostic)
            if result is None:
                data["supervisor_returncode"] = status
                data["completion_diagnostic"] = completion_diagnostic
                _atomic_json(marker, data)
                resource.unknown("office_descendant_exit_unconfirmed")
            probe_failures = result.get("descendant_probe_failures", 0)
            if type(probe_failures) is int and probe_failures > 0:
                _LOG.warning("LibreOffice 后代身份快照 %d 次不可用；已按进程回收证明确认退出。",
                             probe_failures)
            resource.exited()
            marker.unlink()
            confirmed = True
            if status != 0 or result["status"] != "success":
                raise ValueError(f"LibreOffice 文档转换失败，日志编号 {run.name}。")
            produced = output_dir / ("source." + output_format)
            self._check_output(produced, output_format)
            if self._source_stamp(source) != stamp:
                raise ValueError("转换期间文档来源发生变化，请重试。")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("." + target.name + "." + uuid4().hex + ".tmp")
            try:
                shutil.copyfile(produced, temporary)
                os.replace(temporary, target)
                succeeded = True
            finally:
                temporary.unlink(missing_ok=True)
        except BaseException:
            if process is not None and not confirmed and resource is not None:
                try:
                    # The guard already exists.  Notify it using our own trusted
                    # run directory before inspecting a possibly missing,
                    # damaged, or still-starting shared owner marker.
                    if (not _not_link(runs) or not _not_link(run) or not run.is_dir()
                            or run.resolve().parent != runs.resolve()):
                        raise ValueError("文档转换工作目录不可信。")
                    info = run.stat()
                    if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                        raise ValueError("文档转换工作目录权限不可信。")
                    _atomic_json(run / "cancel.json", {"cancel": True})
                except BaseException:
                    resource.unknown("office_cancel_signal_failed")
                try:
                    if not identity_attached:
                        resource.unknown("office_process_identity_unconfirmed")
                    data = _load_private_json(marker)
                    if not isinstance(data.get("token"), str):
                        resource.unknown("office_process_identity_unconfirmed")
                    try:
                        process.wait(timeout=_CLEANUP_SECONDS + 1)
                    except subprocess.TimeoutExpired:
                        resource.unknown("office_cleanup_unconfirmed")
                    if _read_completion(run, pid=process.pid, token=data["token"]) is None:
                        resource.unknown("office_cleanup_unconfirmed")
                    resource.exited()
                    marker.unlink()
                    confirmed = True
                except ResourceCleanupUncertain:
                    raise
                except BaseException:
                    resource.unknown("office_cleanup_unconfirmed")
            elif process is None and resource is not None and not confirmed:
                resource.exited()
                if marker_created:
                    marker.unlink()
                confirmed = True
            raise
        finally:
            if confirmed or resource is None:
                if confirmed and not succeeded:
                    self._retain_failure_log(directory, run)
                # Only remove this newly created directory after checking its
                # resolved location; uncertain resources retain their evidence.
                if (run.exists() and _not_link(run)
                        and run.resolve().parent == runs.resolve()):
                    shutil.rmtree(run)

    @staticmethod
    def _retain_failure_log(directory: Path, run: Path) -> None:
        try:
            failures = directory / "failures"
            if failures.exists() and (not _not_link(failures) or not failures.is_dir()):
                return
            failures.mkdir(mode=0o700, exist_ok=True)
            output = failures / (run.name + ".log")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(output, flags, 0o600)
            with os.fdopen(fd, "wb") as stream:
                try:
                    result = _load_private_json(run / "result.json")
                    summary = {key: result.get(key) for key in ("status", "exit_code", "detail")}
                    header = (json.dumps(summary, ensure_ascii=False) + "\n").encode("utf-8")[:2048]
                except (OSError, ValueError, json.JSONDecodeError):
                    header = b"No supervisor result was captured.\n"
                stream.write(header)
                try:
                    with (run / "converter.log").open("rb") as log:
                        stream.write(log.read(max(0, _MAX_LOG_BYTES - len(header))))
                except OSError:
                    stream.write(b"No converter output was captured.\n"[:max(0, _MAX_LOG_BYTES - len(header))])
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--supervise":
        sys.exit(_supervise(Path(sys.argv[2]), sys.argv[3:]))
    sys.exit("This module is an internal LibreOffice supervisor.")
