import asyncio
from decimal import Decimal
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZipFile

from rag_arch.document_pages import DocumentPageService
from rag_arch.resource_supervision import (
    ResourceCleanupUncertain, TaskResourceSupervisor, begin_resource,
    probe_process, resource_supervision_scope,
)
from rag_arch.request_control import request_control_scope


WINDOWS_EPOCH_TICKS = 621355968000000000
START_TICKS = str(WINDOWS_EPOCH_TICKS + 1_700_000_000 * 10_000_000 + 1_234_560)
START_TOKEN = format((Decimal(START_TICKS) - Decimal(WINDOWS_EPOCH_TICKS))
                     / Decimal(10_000_000), ".6f")
WORD_EXE = str(Path(tempfile.gettempdir()) / "synthetic-office" / "WINWORD.EXE")
POWERSHELL_EXE = str(Path(tempfile.gettempdir()) / "synthetic-powershell" / "powershell.exe")


class FakeResourceHandle:
    def __init__(self, kind, events, *, exit_error=None, attach_error=None):
        self.kind = kind
        self.events = events
        self.exit_error = exit_error
        self.attach_error = attach_error
        self.attached = None
        self._verified_exited = False
        self.unknown_codes = []

    def attach_process(self, pid, process_start_token=None, executable=None):
        self.events.append(("attach", self.kind, pid))
        self.attached = (pid, process_start_token, executable)
        if self.attach_error:
            self.unknown(self.attach_error)
        return self

    def attach_exited_process(self, pid, process_start_token, executable=None):
        self.events.append(("attach_exited", self.kind, pid))
        self.attached = (pid, process_start_token, executable)
        self._verified_exited = True
        return self

    def used(self):
        self.events.append(("used", self.kind))
        return self

    def exited(self):
        self.events.append(("exited", self.kind))
        if self.exit_error:
            raise ResourceCleanupUncertain(self.exit_error)
        self._verified_exited = True
        return True

    def unknown(self, code):
        self.events.append(("unknown", self.kind, code))
        self.unknown_codes.append(code)
        raise ResourceCleanupUncertain(code)


class FakePopen:
    def __init__(self, pid, events, *, stdout="", stderr="", returncode=0,
                 first_communicate_error=None, on_start=None):
        self.pid = pid
        self.events = events
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = None
        self.final_returncode = returncode
        self.first_communicate_error = first_communicate_error
        self.first_call = True
        if on_start:
            on_start(self)

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.events.append(("communicate", self.pid, timeout))
        if self.first_communicate_error is not None and (
                self.first_call or (isinstance(self.first_communicate_error,
                                               subprocess.TimeoutExpired)
                                    and self.returncode is None)):
            self.first_call = False
            error = self.first_communicate_error
            if isinstance(error, subprocess.TimeoutExpired):
                error.output = error.output or self._stdout
                error.stderr = error.stderr or self._stderr
            raise error
        self.first_call = False
        self.returncode = self.final_returncode
        return self._stdout, self._stderr

    def terminate(self):
        self.events.append(("terminate", self.pid))
        self.returncode = -15

    def kill(self):
        self.events.append(("kill", self.pid))
        self.returncode = -9


class StateCheckingResourceRepository:
    """Exercise real ResourceHandle ordering without a database or live processes."""
    def __init__(self):
        self.resources = {}
        self.calls = []
        self.counter = 0

    def begin_resource(self, worker, kind, scope, *, task_id=None, owner_path=None,
                       resource_id=None, owner_descriptor_path=None):
        self.counter += 1
        resource_id = resource_id or f"{self.counter:032x}"
        self.resources[resource_id] = {"kind": kind, "state": "starting", "pid": None,
                                       "token": None, "scope": scope, "task_id": task_id}
        self.calls.append(("begin", kind, scope))
        return {"resource_id": resource_id,
                "owner_descriptor_path": owner_descriptor_path,
                "owner_nonce": uuid4().hex}

    def attach_resource(self, worker, resource_id, *, host_id, pid,
                        process_start_token, executable):
        item = self.resources[resource_id]
        item.update(state="active", pid=pid, token=process_start_token)
        self.calls.append(("attach", item["kind"], pid))
        return True

    def use_resource(self, worker, resource_id, task_id):
        item = self.resources[resource_id]
        self.calls.append(("use", item["kind"], item["state"]))
        if item["state"] != "active":
            return False
        return True

    def resource_exited(self, worker, resource_id, *, pid, process_start_token):
        item = self.resources[resource_id]
        self.calls.append(("exit", item["kind"], pid))
        if item["pid"] != pid or item["token"] != process_start_token:
            return False
        item["state"] = "exited"
        return True

    def resource_unknown(self, worker, resource_id, code):
        self.resources[resource_id]["state"] = "unknown"
        self.calls.append(("unknown", self.resources[resource_id]["kind"], code))


class WordSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "render_docx_pages.ps1").write_text("# fake", encoding="utf-8")
        self.service = DocumentPageService(self.root, conversion_timeout=0.25)
        self.events = []
        self.handles = []
        self.popen_kwargs = []
        self.word_probe_status = "running"

    def _begin_resource(self, *, kind, scope, owner_path):
        self.events.append(("begin", kind, scope, owner_path))
        handle = FakeResourceHandle(kind, self.events)
        self.handles.append(handle)
        return handle

    def _probe_process(self, pid, token, *, executable=None):
        self.events.append(("probe", pid, token, executable))
        return self.word_probe_status

    def _owner_record(self, *, pid=4001):
        return {"pid": pid, "owned": True, "start_ticks": START_TICKS,
                "executable": WORD_EXE}

    def _fake_processes(self, *, main_error=None, main_returncode=0,
                        main_stdout="DOCX_PAGE_RENDER_STAGE:create_word\nDOCX_PAGE_RENDER_OK\n",
                        cleanup_returncode=0, cleanup_stdout="DOCX_PAGE_WORD_CLEANUP_VERIFIED\n",
                        owner_record=True):
        processes = []

        def popen(command, **kwargs):
            self.events.append(("popen", tuple(command)))
            self.popen_kwargs.append(kwargs)
            if "-CleanupOnly" in command:
                process = FakePopen(4202, self.events, stdout=cleanup_stdout,
                                    returncode=cleanup_returncode)
            else:
                owner_file = Path(command[command.index("-OwnerFile") + 1])
                if owner_record is not None:
                    value = self._owner_record() if owner_record is True else owner_record
                    owner_file.write_text(json.dumps(value), encoding="utf-8")
                process = FakePopen(4201, self.events, stdout=main_stdout,
                                    returncode=main_returncode,
                                    first_communicate_error=main_error)
            process.args = command
            processes.append(process)
            return process

        return popen, processes

    def _invoke(self, *, popen, output=None, resource_api=None, resource_scope=None):
        output = output or (self.root / "data" / "runtime" / "document_pages" / "pages.pdf")
        output.parent.mkdir(parents=True, exist_ok=True)
        source = output.with_suffix(".docx")
        api = resource_api or (ResourceCleanupUncertain, self._begin_resource,
                               self._probe_process)
        begin_patch = patch.object(DocumentPageService, "_resource_api", return_value=api)
        popen_patch = patch("rag_arch.document_pages.subprocess.Popen", side_effect=popen)
        which_patch = patch("rag_arch.document_pages.shutil.which", return_value=POWERSHELL_EXE)
        with begin_patch, popen_patch, which_patch:
            if resource_scope is None:
                self.service._render_with_word(source, output)
            else:
                with resource_scope:
                    self.service._render_with_word(source, output)
        return output

    def test_success_registers_exact_word_identity_and_confirms_all_processes(self):
        popen, processes = self._fake_processes()
        output = self._invoke(popen=popen)
        owner = output.with_suffix(".owner.json")
        self.assertFalse(owner.exists())
        self.assertEqual(len(processes), 2)
        word = next(handle for handle in self.handles if handle.kind == "word")
        self.assertEqual(word.attached, (4001, START_TOKEN, WORD_EXE))
        self.assertTrue(word._verified_exited)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))
        self.assertEqual([event[1] for event in self.events if event[0] == "begin"],
                         ["word", "powershell", "powershell"])
        main_call, cleanup_call = [event[1] for event in self.events if event[0] == "popen"]
        self.assertIn("-InputPath", main_call)
        self.assertIn("-CleanupOnly", cleanup_call)
        cleanup_created = next(i for i, event in enumerate(self.events)
                               if event[0] == "popen" and "-CleanupOnly" in event[1])
        self.assertLess(cleanup_created,
                        next(i for i, event in enumerate(self.events)
                             if event[0] == "exited" and event[1] == "word"))
        child_env = self.popen_kwargs[0]["env"]
        temp_dir = Path(child_env["TEMP"])
        self.assertEqual(child_env["TMP"], str(temp_dir))
        self.assertTrue(temp_dir.name.startswith("word-temp-"))
        self.assertFalse(temp_dir.exists())
        self.assertEqual(self.popen_kwargs[1]["env"], child_env)
        self.assertNotIn("shell", self.popen_kwargs[0])
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            self.assertEqual(self.popen_kwargs[0]["creationflags"], subprocess.CREATE_NO_WINDOW)

    def test_success_after_a_poll_timeout_is_not_a_conversion_timeout(self):
        self.service.conversion_timeout = 5
        popen, processes = self._fake_processes()

        def finish_after_one_poll(command, **kwargs):
            process = popen(command, **kwargs)
            if "-CleanupOnly" not in command:
                original = process.communicate
                first_poll = True

                def communicate(timeout=None):
                    nonlocal first_poll
                    if first_poll:
                        first_poll = False
                        self.events.append(("communicate", process.pid, timeout))
                        raise subprocess.TimeoutExpired(command, timeout,
                            output="DOCX_PAGE_RENDER_STAGE:create_word\n")
                    return original(timeout=timeout)

                process.communicate = communicate
            return process

        output = self._invoke(popen=finish_after_one_poll)
        self.assertEqual(processes[0].returncode, 0)
        self.assertEqual(len([event for event in self.events
                              if event[0] == "communicate" and event[1] == 4201]), 2)
        self.assertNotIn(("terminate", 4201), self.events)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))
        self.assertFalse(output.with_suffix(".owner.json").exists())
        self.assertNotIn("Word 排版超时", output.with_suffix(".log").read_text(encoding="utf-8"))

    def test_timeout_cleans_owned_word_before_terminating_converter_powershell(self):
        timeout = subprocess.TimeoutExpired("render", 0.25,
                                            output=b"DOCX_PAGE_RENDER_STAGE:create_word\n",
                                            stderr=b"Word waiting\xff")
        popen, processes = self._fake_processes(main_error=timeout)
        output = self.root / "data" / "runtime" / "document_pages" / "timedout.pdf"
        with self.assertRaisesRegex(ValueError, "超时") as raised:
            self._invoke(popen=popen, output=output)
        self.assertIs(raised.exception.__cause__, timeout)
        cleanup_index = next(i for i, event in enumerate(self.events)
                             if event[0] == "popen" and "-CleanupOnly" in event[1])
        terminate_index = next(i for i, event in enumerate(self.events)
                               if event == ("terminate", 4201))
        self.assertLess(cleanup_index, terminate_index)
        self.assertFalse(output.with_suffix(".owner.json").exists())
        log = output.with_suffix(".log").read_text(encoding="utf-8")
        self.assertIn("DOCX_PAGE_RENDER_STAGE:create_word", log)
        self.assertIn("Word waiting", log)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))
        self.assertEqual(len(processes), 2)

    def test_cancellation_still_cleans_word_then_reaps_powershell_and_reraises(self):
        cancel = asyncio.CancelledError("task cancelled")
        popen, _ = self._fake_processes(main_error=cancel)
        with self.assertRaises(asyncio.CancelledError) as raised:
            self._invoke(popen=popen, output=self.root / "data" / "runtime" /
                         "document_pages" / "cancel.pdf")
        self.assertIs(raised.exception, cancel)
        cleanup_index = next(i for i, event in enumerate(self.events)
                             if event[0] == "popen" and "-CleanupOnly" in event[1])
        terminate_index = next(i for i, event in enumerate(self.events)
                               if event == ("terminate", 4201))
        self.assertLess(cleanup_index, terminate_index)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))

    def test_task_request_control_cancellation_enters_word_cleanup_path(self):
        cancel = asyncio.CancelledError("task cancellation observed by request guard")
        popen, _ = self._fake_processes()

        def cancel_check():
            raise cancel

        with request_control_scope(cancel_check):
            with self.assertRaises(asyncio.CancelledError) as raised:
                self._invoke(popen=popen, output=self.root / "data" / "runtime" /
                             "document_pages" / "guard-cancel.pdf")
        self.assertIs(raised.exception, cancel)
        cleanup_index = next(i for i, event in enumerate(self.events)
                             if event[0] == "popen" and "-CleanupOnly" in event[1])
        terminate_index = next(i for i, event in enumerate(self.events)
                               if event == ("terminate", 4201))
        self.assertLess(cleanup_index, terminate_index)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))

    def test_zero_cleanup_exit_code_does_not_override_unconfirmed_word_exit(self):
        popen, _ = self._fake_processes()
        original_begin = self._begin_resource

        def begin(*, kind, scope, owner_path):
            handle = original_begin(kind=kind, scope=scope, owner_path=owner_path)
            if kind == "word":
                handle.exit_error = "process_exit_not_confirmed"
            return handle

        with patch.object(DocumentPageService, "_resource_api",
                          return_value=(ResourceCleanupUncertain, begin,
                                        self._probe_process)), \
                patch("rag_arch.document_pages.subprocess.Popen", side_effect=popen), \
                patch("rag_arch.document_pages.shutil.which", return_value=POWERSHELL_EXE):
            output = self.root / "data" / "runtime" / "document_pages" / "uncertain.pdf"
            output.parent.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(ResourceCleanupUncertain):
                self.service._render_with_word(output.with_suffix(".docx"), output)
        self.assertTrue(output.with_suffix(".owner.json").exists())
        self.assertIn(("unknown", "word", "owned_word_exit_unverified"), self.events)

    def test_word_that_exited_before_first_poll_uses_trusted_descriptor_without_cleanup(self):
        self.word_probe_status = "exited"
        popen, processes = self._fake_processes()
        output = self._invoke(popen=popen)
        word = next(handle for handle in self.handles if handle.kind == "word")
        self.assertTrue(word._verified_exited)
        self.assertEqual(word.attached, (4001, START_TOKEN, WORD_EXE))
        self.assertTrue(any(event[0] == "attach_exited" and event[1] == "word"
                            for event in self.events))
        self.assertEqual(len(processes), 1)
        self.assertFalse(output.with_suffix(".owner.json").exists())

    def test_fast_word_exit_uses_real_resource_handle_without_starting_usage(self):
        repository = StateCheckingResourceRepository()
        supervisor = TaskResourceSupervisor(
            repository, "worker", "task",
            SimpleNamespace(host_id="local", pid=1234,
                             process_start_token=START_TOKEN), root=self.root)
        popen, processes = self._fake_processes()

        def exited_probe(pid, token, *, executable=None):
            return "exited"

        api = (ResourceCleanupUncertain, begin_resource, exited_probe)
        with patch("rag_arch.resource_supervision.process_token", return_value=START_TOKEN):
            output = self._invoke(
                popen=popen, resource_api=api,
                resource_scope=resource_supervision_scope(supervisor))
        word_rows = [row for row in repository.resources.values() if row["kind"] == "word"]
        self.assertEqual(len(word_rows), 1)
        self.assertEqual(word_rows[0]["state"], "exited")
        self.assertFalse(any(call[0] == "use" and call[1] == "word"
                             for call in repository.calls))
        self.assertEqual(len(processes), 1)
        self.assertFalse(output.with_suffix(".owner.json").exists())

    def test_malformed_owner_is_unknown_and_never_invokes_cleanup_script(self):
        bad_record = {"pid": 4001, "owned": True, "start_ticks": "bad",
                      "process_start_token": START_TOKEN, "executable": WORD_EXE}
        popen, processes = self._fake_processes(owner_record=bad_record)
        output = self.root / "data" / "runtime" / "document_pages" / "malformed.pdf"
        with self.assertRaises(ResourceCleanupUncertain):
            self._invoke(popen=popen, output=output)
        self.assertEqual(len(processes), 1)
        self.assertTrue(output.with_suffix(".owner.json").exists())
        self.assertTrue(any(event[0] == "unknown" and event[1] == "word"
                            for event in self.events))

    def test_converter_that_fails_before_word_creation_marks_word_not_started(self):
        popen, _ = self._fake_processes(main_returncode=1, main_stdout="validation failed\n",
                                        owner_record=None)
        output = self.root / "data" / "runtime" / "document_pages" / "early-fail.pdf"
        with self.assertRaisesRegex(ValueError, "后台排版失败"):
            self._invoke(popen=popen, output=output)
        word = next(handle for handle in self.handles if handle.kind == "word")
        self.assertIsNone(word.attached)
        self.assertTrue(word._verified_exited)
        self.assertFalse(output.with_suffix(".owner.json").exists())
        self.assertEqual(len([event for event in self.events if event[0] == "popen"]), 1)

    def test_resource_cleanup_uncertain_is_not_wrapped_as_a_regular_render_failure(self):
        source = BytesIO()
        with ZipFile(source, "w") as package:
            package.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
        expected = ResourceCleanupUncertain("owned_word_exit_unverified")

        def uncertain_renderer(input_path, output_path):
            raise expected

        service = DocumentPageService(self.root, docx_renderer=uncertain_renderer)
        service.register("plan", "plan.docx", data=source.getvalue())
        with self.assertRaises(ResourceCleanupUncertain) as raised:
            service.list_pages("plan")
        self.assertIs(raised.exception, expected)

    def test_word_started_without_trusted_owner_stays_unknown_and_preserves_owner_evidence(self):
        # An unreadable identity file is not evidence that the started process
        # has exited, and must never authorize cleanup of an arbitrary PID.
        popen, _ = self._fake_processes(owner_record={"pid": True, "owned": True})
        output = self.root / "data" / "runtime" / "document_pages" / "bad-identity.pdf"
        with self.assertRaises(ResourceCleanupUncertain):
            self._invoke(popen=popen, output=output)
        self.assertTrue(output.with_suffix(".owner.json").exists())
        self.assertEqual(len([event for event in self.events if event[0] == "popen"]), 1)

    def test_word_owner_process_token_is_derived_from_start_ticks(self):
        owner = self.root / "owner.json"
        record = self._owner_record()
        record["process_start_token"] = "untrusted duplicate value"
        owner.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(DocumentPageService._read_word_owner(owner)["process_start_token"],
                         START_TOKEN)
        boundary_ticks = (WINDOWS_EPOCH_TICKS + 1_767_225_600 * 10_000_000 + 1_234_565)
        record["start_ticks"] = str(boundary_ticks)
        owner.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(DocumentPageService._read_word_owner(owner)["process_start_token"],
                         "1767225600.123456")

    def test_process_start_failure_confirms_no_word_was_created(self):
        def fail_start(command, **kwargs):
            raise OSError("synthetic PowerShell start failure")

        output = self.root / "data" / "runtime" / "document_pages" / "start-fail.pdf"
        with self.assertRaisesRegex(OSError, "synthetic PowerShell"):
            self._invoke(popen=fail_start, output=output)
        self.assertTrue(all(handle._verified_exited for handle in self.handles))
        self.assertFalse(output.with_suffix(".owner.json").exists())


if __name__ == "__main__":
    unittest.main()
