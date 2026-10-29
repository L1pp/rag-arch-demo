"""Bounded PDF worker lifecycle checks using only synthetic files/processes."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from unittest.mock import patch

import pymupdf

from rag_arch.pdf_backend import (
    DEFAULT_PDF_TIMEOUT_SECONDS,
    PDF_TIMEOUT_ENV,
    ProcessPdfBackend,
)
from rag_arch.request_control import request_control_scope
from rag_arch.resource_supervision import ResourceCleanupUncertain


def _close_backend(backend):
    try:
        backend.close()
    except ResourceCleanupUncertain:
        # The unknown-identity tests intentionally leave this state latched.
        pass


def _synthetic_pdf() -> bytes:
    with pymupdf.open() as document:
        document.new_page().insert_text((30, 40), "supervised synthetic page")
        return document.tobytes()


def _blocking_worker_call(_operation, payload):
    """Test child: it waits for an explicit marker owned by the test."""
    started = Path(payload["started_path"])
    release = Path(payload["release_path"])
    temporary = started.with_suffix(".tmp")
    temporary.write_text(str(os.getpid()), encoding="ascii")
    os.replace(temporary, started)
    deadline = time.monotonic() + 15
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not release.exists():
        raise TimeoutError("test release marker was not created")
    return os.getpid(), {"released": True}


class _RecordingResourceHandle:
    def __init__(self):
        self.pid = None
        self.token = None
        self.executable = None
        self.exited_count = 0
        self.retired = False
        self.unknown_codes = []

    def attach_process(self, pid, process_start_token=None, executable=None):
        self.pid = pid
        self.token = process_start_token
        return self

    def used(self):
        return self

    def exited(self):
        self.exited_count += 1
        return True

    def unknown(self, code):
        self.unknown_codes.append(code)
        raise ResourceCleanupUncertain(code)

    def retire_owner(self):
        self.retired = True


class PdfSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.temp_root = self.root / "pdf-owned-temp"
        self.addCleanup(self.directory.cleanup)

    def make_backend(self, **kwargs):
        backend = ProcessPdfBackend(temp_root=self.temp_root, **kwargs)
        self.addCleanup(_close_backend, backend)
        return backend

    def block_payload(self):
        return {
            "started_path": str(self.root / "worker-started.txt"),
            "release_path": str(self.root / "release-worker.txt"),
        }

    def wait_for_file(self, path, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if Path(path).exists():
                return
            time.sleep(0.01)
        self.fail(f"test worker did not create {Path(path).name}")

    def test_synthetic_pdf_keeps_sync_results_and_cleans_owned_temp(self):
        backend = self.make_backend(call_timeout=5)
        pdf = _synthetic_pdf()

        self.assertEqual(backend.validate(pdf), 1)
        document = backend.extract_document("synthetic.pdf", pdf)
        self.assertEqual(document.filename, "synthetic.pdf")
        self.assertTrue(document.units)
        self.assertEqual(len(backend.worker_pids), 1)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_default_timeout_is_configurable_and_rejects_unbounded_values(self):
        default_backend = self.make_backend()
        self.assertEqual(default_backend.call_timeout, DEFAULT_PDF_TIMEOUT_SECONDS)
        with patch.dict(os.environ, {PDF_TIMEOUT_ENV: "17.5"}):
            configured_backend = self.make_backend()
            self.assertEqual(configured_backend.call_timeout, 17.5)
        with self.assertRaises(ValueError):
            ProcessPdfBackend(call_timeout=float("inf"))

    def test_close_stops_an_idle_owned_worker(self):
        backend = self.make_backend(call_timeout=5, cleanup_timeout=0.5)
        backend.validate(_synthetic_pdf())
        process = backend._process

        started = time.monotonic()
        backend.close()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 2.0)
        self.assertFalse(process.is_alive())
        self.assertIsNone(backend._process)

    def test_blocking_call_times_out_and_terminates_only_its_child(self):
        backend = self.make_backend(
            call_timeout=3.0, cleanup_timeout=0.5,
            _worker_callable=_blocking_worker_call,
        )
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown, wait=True, cancel_futures=True)
        started = time.monotonic()
        future = pool.submit(backend._call, "block", self.block_payload())
        self.wait_for_file(self.root / "worker-started.txt")
        process = backend._process
        with self.assertRaises(TimeoutError):
            future.result(timeout=5)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 4.5)
        self.assertFalse(process.is_alive())
        self.assertEqual(len(backend.worker_pids), 1)
        self.assertEqual(list(self.temp_root.iterdir()), [])
        self.assertTrue(backend._resource is None)
        self.assertTrue(backend._closed)

    def test_close_waiting_for_blocked_call_has_a_finite_upper_bound(self):
        backend = ProcessPdfBackend(
            call_timeout=3.0, cleanup_timeout=0.5, temp_root=self.temp_root,
            _worker_callable=_blocking_worker_call,
        )
        self.addCleanup(_close_backend, backend)
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown, wait=True, cancel_futures=True)
        future = pool.submit(backend._call, "block", self.block_payload())
        self.wait_for_file(self.root / "worker-started.txt")
        process = backend._process

        started = time.monotonic()
        backend.close()
        close_elapsed = time.monotonic() - started
        with self.assertRaises(TimeoutError):
            future.result(timeout=1)

        self.assertLess(close_elapsed, 4.5)
        self.assertFalse(process.is_alive())
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_pid_token_mismatch_never_terminates_a_process_by_reused_pid(self):
        handle = _RecordingResourceHandle()
        backend = ProcessPdfBackend(
            call_timeout=3.0, cleanup_timeout=0.5, temp_root=self.temp_root,
            _worker_callable=_blocking_worker_call,
            _process_probe=lambda *_args, **_kwargs: "exited",
        )
        self.addCleanup(_close_backend, backend)
        with patch("rag_arch.pdf_backend.begin_resource", return_value=handle):
            future_pool = ThreadPoolExecutor(max_workers=1)
            self.addCleanup(future_pool.shutdown, wait=True, cancel_futures=True)
            future = future_pool.submit(backend._call, "block", self.block_payload())
            self.wait_for_file(self.root / "worker-started.txt")
            pid = int((self.root / "worker-started.txt").read_text(encoding="ascii"))
            process = backend._process

            with self.assertRaises(TimeoutError):
                future.result(timeout=5)

            self.assertTrue(process.is_alive())
            self.assertEqual(handle.exited_count, 1)
            self.assertTrue(handle.retired)
            # The fake probe models a PID whose current token belongs to another
            # process. We release our still-owned child explicitly and verify it
            # exits without the backend killing by integer PID.
            (self.root / "release-worker.txt").write_text("release", encoding="ascii")
            process.join(timeout=2)
            self.assertFalse(process.is_alive())
            self.assertEqual(pid, handle.pid)

    def test_unknown_process_identity_raises_and_preserves_input_directory(self):
        handle = _RecordingResourceHandle()
        backend = ProcessPdfBackend(
            call_timeout=3.0, cleanup_timeout=0.5, temp_root=self.temp_root,
            _worker_callable=_blocking_worker_call,
            _process_probe=lambda *_args, **_kwargs: "unknown",
        )
        self.addCleanup(_close_backend, backend)
        with patch("rag_arch.pdf_backend.begin_resource", return_value=handle):
            future_pool = ThreadPoolExecutor(max_workers=1)
            self.addCleanup(future_pool.shutdown, wait=True, cancel_futures=True)
            future = future_pool.submit(backend._call, "block", self.block_payload())
            self.wait_for_file(self.root / "worker-started.txt")
            process = backend._process

            with self.assertRaises(ResourceCleanupUncertain):
                future.result(timeout=5)

            self.assertTrue(process.is_alive())
            self.assertEqual(len(handle.unknown_codes), 1)
            self.assertEqual(len(backend._preserved_temp_dirs), 1)
            self.assertTrue(Path(backend._preserved_temp_dirs[0]).exists())
            (self.root / "release-worker.txt").write_text("release", encoding="ascii")
            process.join(timeout=5)
            if process.is_alive():
                # The harness, which owns this exact multiprocessing.Process
                # object, reaps the deliberately unconfirmed child after the
                # assertions above have proved the backend did not kill it.
                process.terminate()
                process.join(timeout=1)
            self.assertFalse(process.is_alive())

    def test_temp_cleanup_failure_releases_lock_and_retires_backend(self):
        backend = self.make_backend(call_timeout=5, cleanup_timeout=0.5)
        remove_directory = __import__(
            "rag_arch.pdf_backend", fromlist=["_remove_owned_directory"]
        )._remove_owned_directory
        attempts = []

        def fail_once(path):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError("private path omitted")
            return remove_directory(path)

        with patch("rag_arch.pdf_backend._remove_owned_directory", side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, "临时文件清理失败"):
                backend.validate(_synthetic_pdf())

        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "PDF worker 已关闭"):
            backend.validate(_synthetic_pdf())
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(backend._closed)
        self.assertIsNone(backend._process)

    def test_stage_and_cleanup_failure_before_spawn_keeps_backend_reusable(self):
        backend = self.make_backend(call_timeout=5, cleanup_timeout=0.5)
        with patch.object(ProcessPdfBackend, "_stage_request", side_effect=OSError):
            with patch(
                "rag_arch.pdf_backend._remove_owned_directory",
                side_effect=PermissionError("synthetic cleanup denial"),
            ):
                with self.assertRaisesRegex(RuntimeError, "临时文件清理失败"):
                    backend._call("validate", {"source": _synthetic_pdf()})

        self.assertFalse(backend._closed)
        self.assertIsNone(backend._process)
        self.assertEqual(len(backend._preserved_temp_dirs), 1)
        self.assertEqual(backend.validate(_synthetic_pdf()), 1)
        self.assertEqual(len(backend.worker_pids), 1)

    def test_request_cancellation_terminates_the_inflight_pdf_child(self):
        backend = self.make_backend(
            call_timeout=15, cleanup_timeout=0.5,
            _worker_callable=_blocking_worker_call,
        )
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown, wait=True, cancel_futures=True)
        cancelled = threading.Event()

        def run_cancelled_call():
            def check_cancel():
                if cancelled.is_set():
                    raise CancelledError("test cancellation")

            with request_control_scope(check_cancel):
                return backend._call("block", self.block_payload())

        future = pool.submit(run_cancelled_call)
        self.wait_for_file(self.root / "worker-started.txt")
        process = backend._process
        cancelled.set()

        started = time.monotonic()
        with self.assertRaises(CancelledError):
            future.result(timeout=3)

        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(process.is_alive())
        self.assertTrue(backend._closed)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_cancellation_while_queued_does_not_kill_other_call_process(self):
        backend = self.make_backend(
            call_timeout=10, cleanup_timeout=0.5,
            _worker_callable=_blocking_worker_call,
        )
        pool = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(pool.shutdown, wait=True, cancel_futures=True)
        first = pool.submit(backend._call, "block", self.block_payload())
        self.wait_for_file(self.root / "worker-started.txt")
        process = backend._process
        cancelled = threading.Event()

        def run_queued_call():
            def check_cancel():
                if cancelled.is_set():
                    raise CancelledError("queued call cancelled")

            with request_control_scope(check_cancel):
                return backend._call("block", self.block_payload())

        second = pool.submit(run_queued_call)
        time.sleep(0.15)
        cancelled.set()
        with self.assertRaises(CancelledError):
            second.result(timeout=1)
        self.assertTrue(process.is_alive())

        (self.root / "release-worker.txt").write_text("release", encoding="ascii")
        self.assertEqual(first.result(timeout=3), {"released": True})
        self.assertTrue(process.is_alive())
        backend.close()
        self.assertFalse(process.is_alive())


if __name__ == "__main__":
    unittest.main()
