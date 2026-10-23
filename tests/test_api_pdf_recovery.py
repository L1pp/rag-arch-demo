"""Bounded recovery for an API-owned shared PDF backend."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from rag_arch.pdf_backend import ProcessPdfBackend
from rag_arch.resource_supervision import ResourceCleanupUncertain
from rag_arch.web.documents import DocumentStore
from rag_arch.web.standards_resources import StandardsResources


def _constant_pdf_worker(operation, payload):
    """Synthetic worker result; this test never opens a real document."""
    return os.getpid(), 7


class _ProcessHandle:
    pid = 424242

    def join(self, timeout=None):
        return None


class _ResourceHandle:
    executable = None

    def __init__(self, supervisor=None):
        self.supervisor = supervisor
        self.exited_calls = 0
        self.retired = False

    def exited(self):
        self.exited_calls += 1

    def retire_owner(self):
        self.retired = True
        if self.supervisor is not None:
            self.supervisor.retire_requested = True

    def unknown(self, code):
        raise ResourceCleanupUncertain(code)


class _Supervisor:
    def __init__(self):
        self.retire_requested = False


class ApiPdfRecoveryTests(unittest.TestCase):
    def _mark_old_child_exited(self, backend, resource=None):
        backend._process_probe = lambda *args, **kwargs: "exited"
        backend._process = _ProcessHandle()
        backend._resource = resource or _ResourceHandle()
        backend._process_token = "synthetic-process-token"

    def test_confirmed_abort_reopens_shared_backend_for_next_consumer(self):
        with tempfile.TemporaryDirectory() as root:
            backend = ProcessPdfBackend(
                temp_root=root,
                _worker_callable=_constant_pdf_worker,
                reopen_after_confirmed_abort=True,
            )
            try:
                with patch("rag_arch.pdf_backend.ProcessPdfBackend", return_value=backend) as factory:
                    resources = StandardsResources(
                        root=root, recover_pdf_after_confirmed_abort=True)
                    shared_backend = resources.pdf_backend
                    factory.assert_called_once_with(reopen_after_confirmed_abort=True)
                first = DocumentStore(root=root, workspace_id="first", pdf_backend=shared_backend)
                second = DocumentStore(root=root, workspace_id="second", pdf_backend=shared_backend)

                resource_handle = _ResourceHandle()
                self._mark_old_child_exited(backend, resource_handle)
                backend._abort_process("pdf_worker_call_timeout", retire_owner=True)
                self.assertEqual(1, resource_handle.exited_calls)
                self.assertFalse(backend._closed)
                self.assertFalse(backend._broken)

                # The failed call is not replayed. A separate consumer can
                # start a fresh local child through the same shared object.
                self.assertEqual(7, second.pdf_backend.validate(b"%PDF synthetic fixture"))
                self.assertIs(first.pdf_backend, shared_backend)
                self.assertIs(second.pdf_backend, shared_backend)

                first.close()
                first.close()
                second.close()
                second.close()
                resources.close()
                resources.close()
                self.assertTrue(backend._permanently_closed)
            finally:
                try:
                    backend.close()
                except ResourceCleanupUncertain:
                    pass

    def test_unknown_cleanup_remains_fail_closed(self):
        backend = ProcessPdfBackend(reopen_after_confirmed_abort=True)
        backend._process_probe = lambda *args, **kwargs: "unknown"
        backend._process = _ProcessHandle()
        backend._resource = _ResourceHandle()
        backend._process_token = "synthetic-process-token"

        with self.assertRaises(ResourceCleanupUncertain):
            backend._abort_process("pdf_worker_call_timeout", retire_owner=True)
        self.assertTrue(backend._closed)
        self.assertIsNotNone(backend._cleanup_error)
        with self.assertRaises(ResourceCleanupUncertain):
            backend.validate(b"%PDF synthetic fixture")

    def test_worker_default_and_supervised_retirement_do_not_reopen(self):
        worker_default = ProcessPdfBackend()
        self._mark_old_child_exited(worker_default)
        worker_default._abort_process("pdf_worker_call_timeout", retire_owner=True)
        self.assertTrue(worker_default._closed)

        supervisor = _Supervisor()
        supervised_api_backend = ProcessPdfBackend(reopen_after_confirmed_abort=True)
        resource = _ResourceHandle(supervisor=supervisor)
        self._mark_old_child_exited(supervised_api_backend, resource)
        supervised_api_backend._abort_process("pdf_worker_call_timeout", retire_owner=True)
        self.assertTrue(supervised_api_backend._closed)
        self.assertTrue(supervisor.retire_requested)

    def test_close_fence_prevents_confirmed_abort_from_reopening(self):
        backend = ProcessPdfBackend(reopen_after_confirmed_abort=True)
        backend.close()
        self._mark_old_child_exited(backend)

        backend._abort_process("pdf_worker_call_timeout", retire_owner=True)

        self.assertTrue(backend._permanently_closed)
        self.assertTrue(backend._closed)


if __name__ == "__main__":
    unittest.main()
