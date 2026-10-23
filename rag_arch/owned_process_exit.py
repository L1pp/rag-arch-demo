"""Terminate one process only after pinning and verifying its exact identity.

Callers must first authorize cleanup from the durable owner descriptor and
worker state. This module provides the final OS-level PID-reuse race barrier.
"""
from __future__ import annotations

import ctypes
import errno
import math
import ntpath
import os
import re
import select
import signal
import time
from typing import Literal


_EXITED = "exited"
_UNKNOWN = "unknown"
_TOKEN = re.compile(r"\d+\.\d{6}\Z", re.ASCII)
_FILETIME_UNIX_EPOCH = 116444736000000000
_FILETIME_TICKS_PER_SECOND = 10_000_000


def _platform_name() -> str:
    return os.name


def _psutil_module():
    import psutil
    return psutil


def _valid_request(pid: int, process_start_token: str, executable: str,
                   timeout_seconds: float, platform: str) -> bool:
    if type(pid) is not int or pid < 1:
        return False
    if not isinstance(process_start_token, str) or not _TOKEN.fullmatch(process_start_token):
        return False
    if not isinstance(executable, str) or not executable:
        return False
    path_is_absolute = ntpath.isabs(executable) if platform == "nt" else os.path.isabs(executable)
    if not path_is_absolute:
        return False
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        return False
    try:
        timeout = float(timeout_seconds)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(timeout) and timeout > 0


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                ("dwHighDateTime", ctypes.c_uint32)]


def _windows_api():
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return None
    return loader("kernel32", use_last_error=True)


def _last_error() -> int:
    getter = getattr(ctypes, "get_last_error", None)
    return int(getter()) if getter is not None else 0


def _windows_filetime_token(filetime_value: int) -> str:
    """Use the same FILETIME -> float epoch -> six-decimal path as psutil."""
    # Subtract in integer space first. Converting the 1601-based 17-digit
    # FILETIME to float before subtraction loses sub-microsecond precision.
    seconds = float(filetime_value - _FILETIME_UNIX_EPOCH) / float(_FILETIME_TICKS_PER_SECOND)
    return f"{seconds:.6f}"


def _windows_executable_matches(actual: str, expected: str) -> bool:
    try:
        return ntpath.normcase(ntpath.normpath(actual)) == ntpath.normcase(ntpath.normpath(expected))
    except (TypeError, ValueError):
        return False


def _windows_terminate(pid: int, token: str, executable: str,
                       timeout_seconds: float) -> Literal["exited", "unknown"]:
    try:
        kernel = _windows_api()
        if kernel is None:
            return _UNKNOWN
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        process_terminate = 0x0001
        synchronize = 0x00100000
        wait_object_0 = 0x00000000
        wait_timeout = 0x00000102
        error_invalid_parameter = 87

        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileTime),
                                          ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
                                          ctypes.POINTER(_FileTime)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                       wintypes.LPWSTR,
                                                       ctypes.POINTER(wintypes.DWORD)]
        kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL

        access = process_query_limited_information | process_terminate | synchronize
        try:
            handle = kernel.OpenProcess(access, False, pid)
        except Exception:
            return _UNKNOWN
        if not handle:
            return _EXITED if _last_error() == error_invalid_parameter else _UNKNOWN

        outcome: Literal["exited", "unknown"] = _UNKNOWN
        try:
            outcome = _windows_terminate_pinned(
                kernel, handle, token, executable, timeout_seconds,
                wait_object_0=wait_object_0, wait_timeout=wait_timeout,
            )
        except BaseException:
            try:
                outcome = (_EXITED if kernel.WaitForSingleObject(handle, 0) == wait_object_0
                           else _UNKNOWN)
            except Exception:
                outcome = _UNKNOWN
        try:
            closed = bool(kernel.CloseHandle(handle))
        except Exception:
            closed = False
        return _UNKNOWN if not closed and outcome == _EXITED else outcome
    except BaseException:
        return _UNKNOWN


def _windows_terminate_pinned(kernel, handle, token, executable, timeout_seconds, *,
                              wait_object_0: int, wait_timeout: int) -> Literal["exited", "unknown"]:
    creation = _FileTime()
    exit_time = _FileTime()
    kernel_time = _FileTime()
    user_time = _FileTime()
    if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        return _EXITED if kernel.WaitForSingleObject(handle, 0) == wait_object_0 else _UNKNOWN

    filetime_value = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
    if _windows_filetime_token(filetime_value) != token:
        # OpenProcess pinned the currently named PID. A different start token
        # proves the descriptor's old process has exited; never kill the PID reuse.
        return _EXITED

    from ctypes import wintypes
    image_buffer = ctypes.create_unicode_buffer(32768)
    image_length = wintypes.DWORD(len(image_buffer))
    if not kernel.QueryFullProcessImageNameW(handle, 0, image_buffer,
                                              ctypes.byref(image_length)):
        return _EXITED if kernel.WaitForSingleObject(handle, 0) == wait_object_0 else _UNKNOWN
    actual_executable = image_buffer.value[:int(image_length.value)]
    if not _windows_executable_matches(actual_executable, executable):
        return _UNKNOWN

    if not kernel.TerminateProcess(handle, 1):
        return _EXITED if kernel.WaitForSingleObject(handle, 0) == wait_object_0 else _UNKNOWN

    max_milliseconds = 0xFFFFFFFE
    timeout_ms = min(float(timeout_seconds), max_milliseconds / 1000.0) * 1000.0
    milliseconds = max(1, math.ceil(timeout_ms))
    wait_state = kernel.WaitForSingleObject(handle, milliseconds)
    return _EXITED if wait_state == wait_object_0 else _UNKNOWN


def _linux_pidfd_alive(pidfd: int) -> Literal["running", "exited", "unknown"]:
    try:
        signal.pidfd_send_signal(pidfd, 0, None, 0)
        return "running"
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "unknown"
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return "exited"
        return "unknown"
    except Exception:
        return "unknown"


def _linux_signal(pidfd: int, signum: int) -> Literal["sent", "exited", "unknown"]:
    try:
        signal.pidfd_send_signal(pidfd, signum, None, 0)
        return "sent"
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "unknown"
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return "exited"
        return "unknown"
    except Exception:
        return "unknown"


def _wait_pidfd(pidfd: int, timeout_seconds: float) -> Literal["ready", "timeout", "unknown"]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            ready, _write, _errors = select.select([pidfd], [], [], remaining)
            return "ready" if ready else "timeout"
        except InterruptedError:
            if time.monotonic() >= deadline:
                return "timeout"
        except Exception:
            return "unknown"


def _posix_executable_matches(actual: str, expected: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(actual)) == os.path.normcase(os.path.realpath(expected))
    except (OSError, TypeError, ValueError):
        return False


def _linux_terminate(pid: int, token: str, executable: str,
                     timeout_seconds: float) -> Literal["exited", "unknown"]:
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        return _UNKNOWN

    try:
        pidfd = pidfd_open(pid, 0)
    except ProcessLookupError:
        return _EXITED
    except PermissionError:
        return _UNKNOWN
    except OSError as exc:
        return _EXITED if exc.errno == errno.ESRCH else _UNKNOWN
    except Exception:
        return _UNKNOWN

    try:
        try:
            outcome = _linux_terminate_pidfd(pidfd, pid, token, executable, timeout_seconds)
        except BaseException:
            outcome = _UNKNOWN
        try:
            os.close(pidfd)
            return outcome
        except Exception:
            return _UNKNOWN
    except BaseException:
        try:
            os.close(pidfd)
        except Exception:
            pass
        return _UNKNOWN


def _linux_terminate_pidfd(pidfd: int, pid: int, token: str, executable: str,
                           timeout_seconds: float) -> Literal["exited", "unknown"]:
    initial = _linux_pidfd_alive(pidfd)
    if initial == "exited":
        return _EXITED
    if initial != "running":
        return _UNKNOWN

    try:
        psutil = _psutil_module()
    except Exception:
        return _UNKNOWN
    try:
        process = psutil.Process(pid)
        observed_token = f"{float(process.create_time()):.6f}"
    except psutil.NoSuchProcess:
        return _EXITED
    except Exception:
        return _UNKNOWN

    if observed_token != token:
        # The pidfd is pinned to the process currently named by pid; a different
        # token means the descriptor's original process has already exited.
        return _EXITED
    try:
        actual_executable = process.exe()
    except Exception:
        return _UNKNOWN
    if not _posix_executable_matches(actual_executable, executable):
        return _UNKNOWN
    try:
        if process.status() == psutil.STATUS_ZOMBIE or not process.is_running():
            return _EXITED
    except Exception:
        return _UNKNOWN

    # Recheck that the pinned process still exists after PID-based metadata
    # reads. Every signal below uses pidfd, never the reusable numeric PID.
    still_alive = _linux_pidfd_alive(pidfd)
    if still_alive == "exited":
        return _EXITED
    if still_alive != "running":
        return _UNKNOWN

    sent = _linux_signal(pidfd, signal.SIGTERM)
    if sent == "exited":
        return _EXITED
    if sent != "sent":
        return _UNKNOWN

    timeout = float(timeout_seconds)
    term_wait = timeout / 2.0
    wait_state = _wait_pidfd(pidfd, term_wait)
    if wait_state == "ready":
        return _EXITED
    if wait_state != "timeout":
        return _UNKNOWN

    sent = _linux_signal(pidfd, signal.SIGKILL)
    if sent == "exited":
        return _EXITED
    if sent != "sent":
        return _UNKNOWN
    wait_state = _wait_pidfd(pidfd, timeout - term_wait)
    return _EXITED if wait_state == "ready" else _UNKNOWN


def terminate_exact_process(pid: int, process_start_token: str, executable: str,
                            timeout_seconds: float = 5) -> Literal["exited", "unknown"]:
    """Terminate one exact process through a pinned Windows handle or Linux pidfd.

    The caller must already have validated the worker/resource owner descriptor
    and decided that cleanup is authorized. This function never signals by a
    bare PID and never falls back when the platform lacks pidfd support.
    """
    platform = _platform_name()
    if not _valid_request(pid, process_start_token, executable, timeout_seconds, platform):
        return _UNKNOWN
    if platform == "nt":
        return _windows_terminate(pid, process_start_token, executable, float(timeout_seconds))
    if platform == "posix" and sys_platform_is_linux():
        return _linux_terminate(pid, process_start_token, executable, float(timeout_seconds))
    return _UNKNOWN


def sys_platform_is_linux() -> bool:
    # Avoid importing platform at module import for a small, auditable helper.
    import sys
    return sys.platform.startswith("linux")
