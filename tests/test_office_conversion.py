import json
from io import BytesIO
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from rag_arch.office_conversion import (
    OfficeConverter, _SCHEMA, _PrivateJSONError, _descendants, _private_directory, _read_completion,
    _supervise, probe_office_owner,
)
from rag_arch.resource_supervision import process_token
from rag_arch.request_control import request_control_scope
from rag_arch.owned_process_exit import terminate_exact_process
from rag_arch.resource_supervision import (
    ResourceCleanupUncertain, TaskResourceSupervisor, resource_supervision_scope,
)
import rag_arch.office_conversion as office_module


def _write_docx(path):
    with ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/document.xml", "<document/>")


def _write_completion(run, *, status="success"):
    path = run / "result.json"
    path.write_text(json.dumps({"schema": "rag-arch-libreoffice-result-v1", "pid": 12345,
                                "token": "12.345678", "descendants_exited": True,
                                "status": status, "exit_code": 0}), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)


def _lock_contender(root, ready, result):
    converter = OfficeConverter(root, timeout=5)
    directory = _private_directory(Path(root))
    ready.set()
    try:
        with converter._lock(directory, time.monotonic() + 5):
            result.put("acquired")
    except BaseException as exc:
        result.put(type(exc).__name__)


def _directory_contender(root, ready, start, result):
    ready.put(True)
    start.wait(5)
    try:
        result.put(str(_private_directory(Path(root))))
    except BaseException as exc:
        result.put(type(exc).__name__ + ": " + str(exc))


class _Resource:
    resource_id = None

    def __init__(self):
        self.attached = []
        self.exits = 0
        self.unknowns = []

    def attach_process(self, pid, token, executable=None):
        self.attached.append((pid, token, executable))
        return self

    def exited(self):
        self.exits += 1
        return True

    def unknown(self, code):
        self.unknowns.append(code)
        raise ResourceCleanupUncertain(code)


class _Process:
    def __init__(self, command, *, result="ok", on_start=None, **kwargs):
        self.pid = 12345
        self.command = command
        self.kwargs = kwargs
        self.done = result != "waiting"
        self.returncode = 0
        if on_start:
            on_start(command)

    def poll(self):
        return self.returncode if self.done else None

    def wait(self, timeout=None):
        self.done = True
        return self.returncode


class OfficeConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "data" / "runtime" / "uploads"
        self.runtime.mkdir(parents=True)
        self.source = self.runtime / "source.doc"
        self.source.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1synthetic")
        self.target = self.runtime / "normalized.docx"
        # WSL cold imports from a Windows-mounted checkout can take seconds;
        # they are not part of the queue/termination behaviour under test.
        self.converter = OfficeConverter(self.root, timeout=10)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ensure_docx_reuses_valid_result_and_rebuilds_changed_source(self):
        calls = []

        def convert(source, target, output_format, directory, deadline, stamp):
            calls.append(dict(stamp))
            _write_docx(target)

        with patch.object(self.converter, "_convert_locked", side_effect=convert):
            self.converter.ensure_docx(self.source, self.target)
            self.converter.ensure_docx(self.source, self.target)
            self.assertEqual(len(calls), 1)
            self.source.write_bytes(self.source.read_bytes() + b"changed")
            self.converter.ensure_docx(self.source, self.target)
            self.assertEqual(len(calls), 2)
            self.target.write_bytes(b"damaged")
            self.converter.ensure_docx(self.source, self.target)
            self.assertEqual(len(calls), 3)
        self.converter._check_output(self.target, "docx")

    def test_existing_unknown_owner_blocks_conversion_and_is_preserved(self):
        directory = _private_directory(self.root)
        marker = directory / "active.json"
        marker.write_text('{"phase":"starting"}', encoding="utf-8")
        class Repository:
            calls = 0

            def begin_resource(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("a blocked request must not create a resource")

        repository = Repository()
        supervisor = TaskResourceSupervisor(repository, "worker", "task", object(), root=self.root)
        with resource_supervision_scope(supervisor):
            with self.assertRaisesRegex(ValueError, "上一次"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertEqual(repository.calls, 0)
        self.assertFalse(supervisor.has_unconfirmed)
        self.assertFalse(supervisor.retire_requested)
        self.assertTrue(marker.exists())
        self.assertFalse(self.target.exists())

    def test_existing_valid_cache_remains_readable_with_unconfirmed_owner(self):
        calls = []

        def convert(source, target, output_format, directory, deadline, stamp):
            calls.append(1)
            _write_docx(target)

        with patch.object(self.converter, "_convert_locked", side_effect=convert):
            self.converter.ensure_docx(self.source, self.target)
            marker = self.root / "data/runtime/office_conversion/active.json"
            marker.write_text('{"phase":"starting"}', encoding="utf-8")
            self.converter.ensure_docx(self.source, self.target)
            self.assertEqual(len(calls), 1)
            self.source.write_bytes(self.source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "上一次"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertTrue(marker.exists())

    def test_lock_serializes_separate_processes(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        result = context.Queue()
        directory = _private_directory(self.root)
        child = context.Process(target=_lock_contender, args=(str(self.root), ready, result))
        with self.converter._lock(directory, time.monotonic() + 5):
            child.start()
            self.assertTrue(ready.wait(5))
            with self.assertRaises(queue.Empty):
                result.get(timeout=0.35)
        self.assertEqual(result.get(timeout=5), "acquired")
        child.join(5)
        self.assertEqual(child.exitcode, 0)

    def test_first_directory_creation_is_safe_across_processes(self):
        context = multiprocessing.get_context("spawn")
        ready, result = context.Queue(), context.Queue()
        start = context.Event()
        root = self.root / "new-project"
        root.mkdir()
        children = [context.Process(target=_directory_contender,
                                    args=(str(root), ready, start, result)) for _ in range(3)]
        try:
            for child in children:
                child.start()
            for _ in children:
                self.assertTrue(ready.get(timeout=8))
            start.set()
            expected = str(root / "data/runtime/office_conversion")
            self.assertEqual([result.get(timeout=8) for _ in children], [expected] * len(children))
        finally:
            start.set()
            for child in children:
                child.join(8)
        self.assertTrue(all(child.exitcode == 0 for child in children))

    def test_cancellation_before_lock_does_not_leave_marker(self):
        def cancel():
            raise RuntimeError("request cancelled")

        with request_control_scope(cancel):
            with self.assertRaisesRegex(RuntimeError, "request cancelled"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertFalse((self.root / "data/runtime/office_conversion/active.json").exists())

    def test_rejects_output_path_and_sidecar_symlinks(self):
        try:
            sidecar = self.target.with_name(self.target.name + ".source.json")
            sidecar.symlink_to(self.source)
        except OSError:
            self.skipTest("creating a symbolic link is unavailable")
        with self.assertRaisesRegex(ValueError, "链接"):
            self.converter.ensure_docx(self.source, self.target)
        with self.assertRaisesRegex(ValueError, "运行时"):
            self.converter.ensure_docx(self.source, self.root / "outside.docx")

    @unittest.skipUnless(os.name == "posix", "executable mode is required")
    def test_official_script_launcher_is_not_replaced_by_soffice_binary(self):
        launcher = self.root / "soffice"
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o700)
        sibling = self.root / "soffice.bin"
        sibling.write_bytes(b"synthetic binary")
        sibling.chmod(0o700)
        with patch("rag_arch.office_conversion.sys.platform", "linux"):
            converter = OfficeConverter(self.root, executable=str(launcher))
            self.assertEqual(converter._executable(), str(launcher.resolve()))

    def test_windows_legacy_doc_error_explains_docx_workaround(self):
        with patch("rag_arch.office_conversion.sys.platform", "win32"):
            with self.assertRaisesRegex(ValueError, "Windows.*另存为 DOCX"):
                self.converter._executable()

    def test_valid_conversion_uses_private_profile_and_removes_marker(self):
        resource = _Resource()
        observed = {}

        def start(command):
            observed["command"] = command
            # Test profile contents through cwd on both Windows and Linux.
            run = Path(command[-1]).parent.parent
            xml = (run / "profile/user/registrymodifications.xcu").read_text(encoding="utf-8")
            observed["profile"] = xml
            _write_docx(Path(command[-2]) / "source.docx")
            _write_completion(run)

        def create_process(command, **kwargs):
            run = Path(command[4])
            observed["environment"] = dict(kwargs["env"])
            observed["private_dirs"] = {
                key: (Path(kwargs["env"][key]).parent == run
                      and Path(kwargs["env"][key]).is_dir())
                for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "TMPDIR")
            }
            if os.name == "posix":
                observed["private_modes"] = {
                    key: Path(kwargs["env"][key]).stat().st_mode & 0o777
                    for key in observed["private_dirs"]
                }
            return _Process(command, on_start=start, **kwargs)

        with (patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch.dict(os.environ, {"RAG_CHAT_API_KEY": "synthetic-private-key"}),
              patch("rag_arch.office_conversion.subprocess.Popen",
                    side_effect=create_process)):
            self.converter.ensure_docx(self.source, self.target)
        self.assertTrue(self.target.is_file())
        self.assertEqual(resource.exits, 1)
        self.assertIn("DisableMacrosExecution", observed["profile"])
        self.assertIn("DisableActiveContent", observed["profile"])
        self.assertIn('<prop oor:name="Link" oor:op="fuse"><value>2</value></prop>', observed["profile"])
        self.assertEqual(observed["command"][0], os.path.abspath(sys.executable))
        self.assertIn("docx:Office Open XML Text", observed["command"])
        self.assertTrue(all(observed["private_dirs"].values()))
        self.assertNotIn("RAG_CHAT_API_KEY", observed["environment"])
        if os.name == "posix":
            self.assertTrue(all(mode == 0o700 for mode in observed["private_modes"].values()))
        self.assertFalse((self.root / "data/runtime/office_conversion/active.json").exists())

    def test_invalid_result_is_not_published(self):
        resource = _Resource()

        def start(command):
            (Path(command[-2]) / "source.docx").write_bytes(b"not a zip")
            _write_completion(Path(command[-1]).parent.parent)

        with (patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion.subprocess.Popen",
                    side_effect=lambda *a, **kw: _Process(*a, on_start=start, **kw))):
            with self.assertRaises(ValueError):
                self.converter.ensure_docx(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(resource.exits, 1)
        self.assertFalse((self.root / "data/runtime/office_conversion/active.json").exists())

    def test_invalid_docx_xml_is_rejected(self):
        with ZipFile(self.target, "w") as package:
            package.writestr("[Content_Types].xml", "<Types/>")
            package.writestr("word/document.xml", "<invalid")
        with self.assertRaises(ValueError):
            self.converter._check_output(self.target, "docx")

    def test_cancel_exit_race_requires_confirmed_process_exit(self):
        run = self.runtime / "synthetic-supervisor"
        run.mkdir()
        (run / "cancel.json").write_text('{"cancel":true}', encoding="utf-8")
        child = _Process([], result="ok")
        child.stdout = BytesIO(b"")
        member = {"pid": child.pid, "token": "12.345678", "executable": "/fake/child"}
        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", return_value=[member]),
              patch("rag_arch.office_conversion._reap_remaining_children", return_value=False),
              patch("rag_arch.office_conversion.terminate_exact_process", return_value="unknown"),
              patch("rag_arch.office_conversion.probe_process", return_value="exited") as probe,
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 0)
            probe.assert_called_once_with(child.pid, "12.345678", executable="/fake/child")
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["descendants_exited"])
        child.stdout = BytesIO(b"")
        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", return_value=[member]),
              patch("rag_arch.office_conversion.terminate_exact_process", return_value="unknown"),
              patch("rag_arch.office_conversion.probe_process", return_value="running"),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 2)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cleanup_uncertain")
        self.assertFalse(result["descendants_exited"])

    def test_short_lived_and_inaccessible_descendants_are_distinguished(self):
        import psutil

        class Child:
            pid = 12345

            def __init__(self, exe_result):
                self.exe_result = exe_result

            def status(self):
                return "running"

            def exe(self):
                if isinstance(self.exe_result, BaseException):
                    raise self.exe_result
                return self.exe_result

        class Parent:
            def __init__(self, child):
                self.child = child

            def children(self, recursive):
                return [self.child]

        with patch("psutil.Process", return_value=Parent(Child(psutil.NoSuchProcess(12345)))):
            self.assertEqual(_descendants(), [])
        for exe_result, expected_type, expected_reason in (
                ("", "OSError", "empty_executable"),
                (psutil.AccessDenied(12345), "AccessDenied", "identity_unavailable")):
            with self.subTest(reason=expected_reason):
                with patch("psutil.Process", return_value=Parent(Child(exe_result))):
                    with self.assertRaises(ResourceCleanupUncertain) as raised:
                        _descendants()
                details = raised.exception.probe_details
                self.assertEqual(details["stage"], "executable")
                self.assertEqual(details["pid"], 12345)
                self.assertEqual(details["error_type"], expected_type)
                self.assertEqual(details["reason"], expected_reason)

    def test_transient_probe_failure_waits_for_all_descendants_before_success(self):
        run = self.runtime / "normal-probe-transient"
        run.mkdir()
        child = _Process([], result="ok")
        child.stdout = BytesIO(b"")
        fault = ResourceCleanupUncertain("office_descendant_probe_failed")
        fault.probe_details = {"stage": "executable", "pid": 12345, "error_type": "OSError",
                               "errno": None, "reason": "empty_executable"}
        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", side_effect=[fault, []]),
              patch("rag_arch.office_conversion._reap_remaining_children", side_effect=[True, False]) as reap,
              patch("rag_arch.office_conversion._POLL_SECONDS", 0),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 0)
        self.assertEqual(reap.call_count, 2)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertTrue(result["descendants_exited"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["descendant_probe_failures"], 1)
        self.assertEqual(result["last_descendant_probe_failure"]["reason"], "empty_executable")

    def test_probe_failure_preserves_the_last_trusted_child_snapshot(self):
        run = self.runtime / "probe-keeps-children"
        run.mkdir()
        child = _Process([], result="waiting")
        child.stdout = BytesIO(b"")
        polls = iter((None, None, 0))
        child.poll = lambda: next(polls)
        member = {"pid": child.pid, "token": "12.345678", "executable": "/fake/child"}
        calls = 0

        def probe():
            nonlocal calls
            calls += 1
            if calls == 1:
                return [member]
            current = json.loads((run / "children.json").read_text(encoding="utf-8"))
            self.assertEqual(current["children"], [member])
            if calls == 2:
                raise ResourceCleanupUncertain("office_descendant_probe_failed")
            return []

        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", side_effect=probe),
              patch("rag_arch.office_conversion._reap_remaining_children", return_value=False),
              patch("rag_arch.office_conversion._POLL_SECONDS", 0),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 0)
        self.assertEqual(calls, 3)

    def test_cancel_waits_for_identity_after_transient_probe_failure(self):
        run = self.runtime / "cancel-probe-transient"
        run.mkdir()
        (run / "cancel.json").write_text('{"cancel":true}', encoding="utf-8")
        child = _Process([], result="waiting")
        child.stdout = BytesIO(b"")
        fault = ResourceCleanupUncertain("office_descendant_probe_failed")
        fault.probe_details = {"stage": "executable", "pid": child.pid, "error_type": "AccessDenied",
                               "errno": None, "reason": "identity_unavailable"}
        member = {"pid": child.pid, "token": "12.345678", "executable": "/fake/child"}

        def stop(*args, **kwargs):
            child.done = True
            return "exited"

        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", side_effect=[fault, [member]]),
              patch("rag_arch.office_conversion._reap_remaining_children", return_value=False),
              patch("rag_arch.office_conversion.terminate_exact_process", side_effect=stop) as terminate,
              patch("rag_arch.office_conversion._POLL_SECONDS", 0),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 0)
        terminate.assert_called_once_with(child.pid, "12.345678", "/fake/child", timeout_seconds=0.5)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["descendants_exited"])
        self.assertEqual(result["last_descendant_probe_failure"]["error_type"], "AccessDenied")

    def test_cancel_with_persistent_probe_failure_stays_unknown(self):
        run = self.runtime / "cancel-probe-persistent"
        run.mkdir()
        (run / "cancel.json").write_text('{"cancel":true}', encoding="utf-8")
        child = _Process([], result="waiting")
        child.stdout = BytesIO(b"")
        fault = ResourceCleanupUncertain("office_descendant_probe_failed")
        with (patch("rag_arch.office_conversion._enable_subreaper"),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._descendants", side_effect=fault),
              patch("rag_arch.office_conversion.terminate_exact_process") as terminate,
              patch("rag_arch.office_conversion._CLEANUP_SECONDS", 0.03),
              patch("rag_arch.office_conversion._POLL_SECONDS", 0.01),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=child)):
            self.assertEqual(_supervise(run, ["fake"]), 2)
        terminate.assert_not_called()
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cleanup_uncertain")
        self.assertFalse(result["descendants_exited"])
        self.assertGreater(result["descendant_probe_failures"], 0)

    def test_completion_rejection_reports_precise_bounded_reason(self):
        run = self.runtime / "completion-reasons"
        run.mkdir()
        diagnostic = {}
        self.assertIsNone(_read_completion(run, pid=12345, token="12.345678", diagnostic=diagnostic))
        self.assertEqual(diagnostic["reason"], "missing")
        with patch("rag_arch.office_conversion._load_private_json", side_effect=PermissionError()):
            self.assertIsNone(_read_completion(run, pid=12345, token="12.345678", diagnostic=diagnostic))
        self.assertEqual(diagnostic["reason"], "permissions")
        with patch("rag_arch.office_conversion._load_private_json", side_effect=_PrivateJSONError("owner_mismatch")):
            self.assertIsNone(_read_completion(run, pid=12345, token="12.345678", diagnostic=diagnostic))
        self.assertEqual(diagnostic["reason"], "owner_mismatch")
        result_file = run / "result.json"
        result_file.write_text("{broken", encoding="utf-8")
        if os.name == "posix":
            result_file.chmod(0o600)
        self.assertIsNone(_read_completion(run, pid=12345, token="12.345678", diagnostic=diagnostic))
        self.assertEqual(diagnostic["reason"], "invalid_json")
        _write_completion(run)
        baseline = json.loads(result_file.read_text(encoding="utf-8"))
        cases = [({}, "missing_fields"), ({**baseline, "schema": "wrong"}, "schema_mismatch"),
                 ({**baseline, "pid": 12346}, "pid_mismatch"),
                 ({**baseline, "token": "98.765432"}, "token_mismatch"),
                 ({**baseline, "descendants_exited": False}, "descendants_unconfirmed"),
                 ({**baseline, "status": ["invalid"]}, "invalid_status")]
        for value, reason in cases:
            with self.subTest(reason=reason):
                result_file.write_text(json.dumps(value), encoding="utf-8")
                self.assertIsNone(_read_completion(run, pid=12345, token="12.345678", diagnostic=diagnostic))
                self.assertEqual(diagnostic["reason"], reason)
        result_file.write_text(json.dumps(baseline), encoding="utf-8")
        self.assertIsNotNone(_read_completion(run, pid=12345, token="12.345678"))

    def test_unconfirmed_completion_persists_returncode_and_token_mismatch(self):
        resource = _Resource()

        def start(command):
            run = Path(command[-1]).parent.parent
            _write_completion(run)
            result_path = run / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["token"] = "98.765432"
            result_path.write_text(json.dumps(result), encoding="utf-8")

        with (patch.object(self.converter, "_executable", return_value="/fake/soffice"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion.subprocess.Popen",
                    side_effect=lambda *a, **kw: _Process(*a, on_start=start, **kw))):
            with self.assertRaises(ResourceCleanupUncertain):
                self.converter.ensure_docx(self.source, self.target)
        marker = self.root / "data/runtime/office_conversion/active.json"
        data = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(data["supervisor_returncode"], 0)
        self.assertEqual(data["completion_diagnostic"]["reason"], "token_mismatch")
        self.assertEqual(data["completion_diagnostic"]["expected_token"], "12.345678")
        self.assertEqual(data["completion_diagnostic"]["observed_token"], "98.765432")

    def test_supervisor_exception_records_type_and_safe_error_code(self):
        run = self.runtime / "supervisor-exception"
        run.mkdir()
        with (patch("rag_arch.office_conversion._enable_subreaper", side_effect=RuntimeError("private detail")),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678")):
            self.assertEqual(_supervise(run, ["fake"]), 2)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error_code"], "supervisor_error")
        self.assertNotIn("private detail", json.dumps(result))
        self.assertFalse(result["descendants_exited"])

    def test_failed_conversion_keeps_a_bounded_diagnostic_log(self):
        resource = _Resource()

        def start(command):
            run = Path(command[-1]).parent.parent
            _write_completion(run, status="conversion_failed")
            (run / "converter.log").write_bytes(b"E" * (70 * 1024))

        with (patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion.subprocess.Popen",
                    side_effect=lambda *a, **kw: _Process(*a, on_start=start, **kw))):
            with self.assertRaisesRegex(ValueError, "日志编号"):
                self.converter.ensure_docx(self.source, self.target)
        logs = list((self.root / "data/runtime/office_conversion/failures").glob("*.log"))
        self.assertEqual(len(logs), 1)
        self.assertLessEqual(logs[0].stat().st_size, 64 * 1024)
        self.assertIn(b"conversion_failed", logs[0].read_bytes())

    def test_source_modified_during_conversion_does_not_publish_result(self):
        resource = _Resource()

        def start(command):
            run = Path(command[-1]).parent.parent
            _write_docx(Path(command[-2]) / "source.docx")
            _write_completion(run)
            self.source.write_bytes(self.source.read_bytes() + b"changed")

        with (patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion.subprocess.Popen",
                    side_effect=lambda *a, **kw: _Process(*a, on_start=start, **kw))):
            with self.assertRaisesRegex(ValueError, "发生变化"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(resource.exits, 1)

    def test_cancellation_after_start_confirms_exit_before_releasing_marker(self):
        resource = _Resource()
        process = _Process([], result="waiting")
        count = 0

        def cancel_after_start():
            nonlocal count
            count += 1
            if count >= 3:
                raise RuntimeError("request cancelled")

        with (request_control_scope(cancel_after_start),
              patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._read_completion", return_value={"status": "cancelled"}),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=process)):
            with self.assertRaisesRegex(RuntimeError, "request cancelled"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertEqual(resource.exits, 1)
        self.assertFalse((self.root / "data/runtime/office_conversion/active.json").exists())
        self.assertFalse(self.target.exists())

    def test_startup_identity_failures_signal_cancel_before_own_unknown(self):
        for failure in ("token", "marker_write", "attach"):
            with self.subTest(failure=failure):
                root = self.root / ("startup-" + failure)
                upload = root / "data/runtime/uploads"
                upload.mkdir(parents=True)
                source = upload / "source.doc"
                source.write_bytes(self.source.read_bytes())
                target = upload / "normalized.docx"
                converter = OfficeConverter(root, timeout=10)
                marker = root / "data/runtime/office_conversion/active.json"
                observed = {"run": None, "events": []}

                class FaultResource(_Resource):
                    def attach_process(inner, pid, token, executable=None):
                        if failure == "attach":
                            raise RuntimeError("synthetic attach failure")
                        return super().attach_process(pid, token, executable=executable)

                    def unknown(inner, code):
                        observed["events"].append(("unknown", code))
                        self.assertTrue((observed["run"] / "cancel.json").is_file())
                        return super().unknown(code)

                resource = FaultResource()
                process = _Process([], result="waiting")
                original_atomic = office_module._atomic_json

                def write_json(path, data):
                    path = Path(path)
                    if path == marker and data.get("phase") == "running" and failure == "marker_write":
                        marker.write_text("{corrupted", encoding="utf-8")
                        raise OSError("synthetic marker write failure")
                    if path.name == "cancel.json":
                        observed["events"].append(("cancel", None))
                    return original_atomic(path, data)

                def start_process(command, **kwargs):
                    observed["run"] = Path(command[4])
                    return process

                token_error = RuntimeError("synthetic token failure") if failure == "token" else None
                with (patch.object(converter, "_executable", return_value="/fake/soffice"),
                      patch("rag_arch.office_conversion.begin_resource", return_value=resource),
                      patch("rag_arch.office_conversion.process_token", side_effect=token_error,
                            return_value="12.345678"),
                      patch("rag_arch.office_conversion._atomic_json", side_effect=write_json),
                      patch("rag_arch.office_conversion.subprocess.Popen", side_effect=start_process)):
                    with self.assertRaises(ResourceCleanupUncertain):
                        converter.ensure_docx(source, target)
                self.assertEqual([event[0] for event in observed["events"]], ["cancel", "unknown"])
                self.assertEqual(resource.exits, 0)
                self.assertTrue(marker.is_file())
                self.assertFalse(target.exists())
                if failure == "marker_write":
                    self.assertEqual(marker.read_text(encoding="utf-8"), "{corrupted")

    def test_corrupt_marker_after_attach_still_signals_cancel_and_preserves_unknown(self):
        marker = self.root / "data/runtime/office_conversion/active.json"
        observed = {"run": None, "events": [], "checks": 0}

        class FaultResource(_Resource):
            def unknown(inner, code):
                observed["events"].append("unknown")
                self.assertTrue((observed["run"] / "cancel.json").is_file())
                return super().unknown(code)

        resource = FaultResource()
        process = _Process([], result="waiting")
        original_atomic = office_module._atomic_json

        def write_json(path, data):
            if Path(path).name == "cancel.json":
                observed["events"].append("cancel")
            return original_atomic(path, data)

        def start_process(command, **kwargs):
            observed["run"] = Path(command[4])
            return process

        def corrupt_after_attach():
            observed["checks"] += 1
            if observed["checks"] >= 3:
                marker.write_text("{corrupted", encoding="utf-8")
                raise RuntimeError("synthetic request cancellation")

        with (request_control_scope(corrupt_after_attach),
              patch.object(self.converter, "_executable", return_value="/fake/soffice"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion._atomic_json", side_effect=write_json),
              patch("rag_arch.office_conversion.subprocess.Popen", side_effect=start_process)):
            with self.assertRaises(ResourceCleanupUncertain):
                self.converter.ensure_docx(self.source, self.target)
        self.assertEqual(observed["events"], ["cancel", "unknown"])
        self.assertEqual(resource.exits, 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "{corrupted")
        self.assertFalse(self.target.exists())

    def test_unknown_cleanup_keeps_owner_and_blocks_later_conversion(self):
        resource = _Resource()
        process = _Process([], result="waiting")
        self.converter.timeout = 1.0
        with (patch.object(self.converter, "_executable", return_value="/fake/soffice.bin"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", return_value="12.345678"),
              patch("rag_arch.office_conversion.subprocess.Popen", return_value=process)):
            with self.assertRaises(ResourceCleanupUncertain):
                self.converter.ensure_docx(self.source, self.target)
            with self.assertRaisesRegex(ValueError, "上一次"):
                self.converter.ensure_docx(self.source, self.target)
        self.assertTrue((self.root / "data/runtime/office_conversion/active.json").is_file())
        self.assertFalse(self.target.exists())

    def test_recovery_probe_requires_matching_identity_and_completion(self):
        directory = _private_directory(self.root)
        marker = directory / "active.json"
        run = directory / "runs" / "test-run"
        run.mkdir(parents=True)
        _write_completion(run)
        data = {"schema": _SCHEMA, "host": __import__("socket").gethostname(),
                "pid": 12345, "token": "12.345678", "executable": "/fake/soffice.bin",
                "group_id": 12345, "children": [], "run_directory": str(run)}
        marker.write_text(json.dumps(data), encoding="utf-8")
        if os.name == "posix":
            marker.chmod(0o600)
        with (patch("rag_arch.office_conversion.sys.platform", "linux"),
              patch("rag_arch.office_conversion.probe_process", return_value="exited")):
            self.assertEqual(probe_office_owner(marker, pid=12345, token="12.345678",
                                                executable="/fake/soffice.bin"), "exited")
            self.assertEqual(probe_office_owner(marker, pid=12345, token="different",
                                                executable="/fake/soffice.bin"), "unknown")
            (run / "result.json").unlink()
            self.assertEqual(probe_office_owner(marker, pid=12345, token="12.345678",
                                                executable="/fake/soffice.bin"), "unknown")

    def test_ensure_pdf_reuses_valid_result_and_rebuilds_after_source_changes(self):
        import pymupdf

        source = self.runtime / "source.docx"
        target = self.runtime / "pages.pdf"
        _write_docx(source)
        calls = []

        def convert(source_path, target_path, output_format, directory, deadline, stamp):
            calls.append(dict(stamp))
            with pymupdf.open() as pdf:
                pdf.new_page()
                pdf.save(target_path)

        with patch.object(self.converter, "_convert_locked", side_effect=convert):
            self.converter.ensure_pdf(source, target)
            self.converter.ensure_pdf(source, target)
            self.assertEqual(len(calls), 1)
            with ZipFile(source, "a") as package:
                package.writestr("extra.txt", "changed")
            self.converter.ensure_pdf(source, target)
            self.assertEqual(len(calls), 2)

    def test_active_docx_is_rejected_before_libreoffice_starts(self):
        source = self.runtime / "unsafe.docx"
        with ZipFile(source, "w") as package:
            package.writestr("[Content_Types].xml", "<Types/>")
            package.writestr("word/document.xml", "<document/>")
            package.writestr("word/vbaProject.bin", b"macro")
        with patch.object(self.converter, "_executable") as executable:
            with self.assertRaisesRegex(ValueError, "宏"):
                self.converter.convert(source, self.runtime / "pages.pdf", output_format="pdf")
            executable.assert_not_called()


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux subreaper is required")
class LinuxSubreaperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rag-arch-office-test-")
        self.root = Path(self.temporary.name)
        self.directory = _private_directory(self.root)
        self.run = self.directory / "runs" / "test-run"
        self.run.mkdir(parents=True, mode=0o700)
        self.ready = self.run / "descendant-ready"
        self.supervisor = None
        self.descendant_identity = None

    def tearDown(self):
        if self.supervisor is not None and self.supervisor.poll() is None:
            (self.run / "cancel.json").write_text('{"cancel":true}', encoding="utf-8")
            try:
                self.supervisor.wait(timeout=7)
            except subprocess.TimeoutExpired:
                self.fail("synthetic subreaper did not terminate safely")
        if self.descendant_identity is not None:
            pid, token, executable = self.descendant_identity
            self.assertEqual(terminate_exact_process(pid, token, executable, timeout_seconds=3),
                             "exited", "synthetic test descendant must be confirmed exited")
        if self.supervisor is not None:
            for stream in (self.supervisor.stdout, self.supervisor.stderr):
                if stream:
                    stream.close()
        self.temporary.cleanup()

    def _start_detached_descendant(self, sleep_seconds):
        ready_temporary = self.run / "descendant-ready.tmp"
        script = (
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            f"    with open({str(ready_temporary)!r}, 'w') as stream: stream.write(str(os.getpid()))\n"
            f"    os.replace({str(ready_temporary)!r}, {str(self.ready)!r})\n"
            f"    time.sleep({sleep_seconds!r})\n"
            "    os._exit(0)\n"
            f"while not os.path.exists({str(self.ready)!r}): time.sleep(0.01)\n"
            "os._exit(0)\n"
        )
        self.supervisor = subprocess.Popen(
            [sys.executable, "-m", "rag_arch.office_conversion", "--supervise", str(self.run),
             sys.executable, "-c", script],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        token = process_token(self.supervisor.pid)
        deadline = time.monotonic() + 6
        while not self.ready.exists() and time.monotonic() < deadline:
            if self.supervisor.poll() is not None:
                self.fail("synthetic supervisor exited before child start: " +
                          self.supervisor.stderr.read().decode("utf-8", "replace"))
            time.sleep(0.02)
        self.assertTrue(self.ready.exists(), "synthetic descendant did not start")
        descendant_pid = int(self.ready.read_text(encoding="utf-8"))
        import psutil
        self.descendant_identity = (descendant_pid, process_token(descendant_pid),
                                    psutil.Process(descendant_pid).exe())
        return token, descendant_pid

    def test_detached_grandchild_must_exit_before_completion(self):
        token, descendant_pid = self._start_detached_descendant(1.5)
        self.assertIsNone(self.supervisor.poll(), "supervisor must wait for detached grandchild")
        marker = self.directory / "active.json"
        marker.write_text(json.dumps({"schema": _SCHEMA, "host": __import__("socket").gethostname(),
                          "pid": self.supervisor.pid, "token": token,
                          "executable": str(Path(sys.executable).resolve()),
                          "group_id": self.supervisor.pid, "run_directory": str(self.run)}),
                          encoding="utf-8")
        marker.chmod(0o600)
        self.assertEqual(probe_office_owner(marker, pid=self.supervisor.pid, token=token,
                                           executable=str(Path(sys.executable).resolve())), "running")
        exit_code = self.supervisor.wait(timeout=8)
        result = json.loads((self.run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0, str(result))
        self.assertTrue(result["descendants_exited"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(probe_office_owner(marker, pid=self.supervisor.pid, token=token,
                                           executable=str(Path(sys.executable).resolve())), "exited")
        with self.assertRaises(ProcessLookupError):
            os.kill(descendant_pid, 0)

    def test_cancel_terminates_detached_grandchild_before_completion(self):
        _token, descendant_pid = self._start_detached_descendant(20)
        self.assertIsNone(self.supervisor.poll())
        (self.run / "cancel.json").write_text('{"cancel":true}', encoding="utf-8")
        exit_code = self.supervisor.wait(timeout=8)
        result = json.loads((self.run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0, str(result))
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["descendants_exited"])
        with self.assertRaises(ProcessLookupError):
            os.kill(descendant_pid, 0)

    def _real_guard_with_startup_fault(self, failure):
        import psutil

        upload = self.root / "data/runtime/uploads"
        upload.mkdir(parents=True, exist_ok=True)
        source = upload / "source.doc"
        source.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1synthetic")
        target = upload / "normalized.docx"
        marker = self.directory / "active.json"
        converter = OfficeConverter(self.root, timeout=30)
        observed = {"run": None, "checks": 0, "unknown_after_cancel": False}

        class FaultResource(_Resource):
            def unknown(inner, code):
                observed["unknown_after_cancel"] = (observed["run"] / "cancel.json").is_file()
                return super().unknown(code)

        resource = FaultResource()
        original_popen = subprocess.Popen
        child_identity = process_token

        def start_guard(command, **kwargs):
            run = Path(command[4])
            observed["run"] = run
            self.run = run
            ready = run / "synthetic-child-ready.json"
            temporary = run / "synthetic-child-ready.tmp"
            script = (
                "import json,os,time\n"
                "pid=os.fork()\n"
                "if pid == 0:\n"
                "    os.setsid()\n"
                "    keys=('HOME','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_RUNTIME_DIR','TMPDIR')\n"
                "    payload={'pid':os.getpid(),'environment':{key:os.environ.get(key) for key in keys}}\n"
                f"    with open({str(temporary)!r}, 'w') as stream: json.dump(payload, stream)\n"
                f"    os.replace({str(temporary)!r}, {str(ready)!r})\n"
                "    time.sleep(20)\n"
                "    os._exit(0)\n"
                f"while not os.path.exists({str(ready)!r}): time.sleep(0.01)\n"
                "time.sleep(20)\n"
            )
            guarded = [*command[:5], sys.executable, "-c", script]
            self.supervisor = original_popen(guarded, **kwargs)
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                if self.supervisor.poll() is not None:
                    self.fail("synthetic guard exited before its child started")
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "synthetic guard did not start a child")
            record = json.loads(ready.read_text(encoding="utf-8"))
            pid = record["pid"]
            self.descendant_identity = (pid, child_identity(pid), psutil.Process(pid).exe())
            for value in record["environment"].values():
                self.assertEqual(Path(value).parent, run)
                self.assertTrue(Path(value).is_dir())
            return self.supervisor

        def fail_or_token(pid):
            if failure == "token":
                raise RuntimeError("synthetic token failure")
            return child_identity(pid)

        def corrupt_after_attach():
            observed["checks"] += 1
            if failure == "corrupt_marker" and observed["checks"] >= 3:
                marker.write_text("{corrupted", encoding="utf-8")
                raise RuntimeError("synthetic request cancellation")

        with (request_control_scope(corrupt_after_attach),
              patch.object(converter, "_executable", return_value="/fake/soffice"),
              patch("rag_arch.office_conversion.begin_resource", return_value=resource),
              patch("rag_arch.office_conversion.process_token", side_effect=fail_or_token),
              patch("rag_arch.office_conversion.subprocess.Popen", side_effect=start_guard)):
            with self.assertRaises(ResourceCleanupUncertain):
                converter.ensure_docx(source, target)

        self.assertTrue(observed["unknown_after_cancel"])
        self.assertEqual(resource.exits, 0)
        self.assertTrue(marker.is_file())
        self.assertTrue((observed["run"] / "cancel.json").is_file())
        self.assertFalse(target.exists())
        if failure == "corrupt_marker":
            self.assertEqual(marker.read_text(encoding="utf-8"), "{corrupted")
        self.assertEqual(self.supervisor.wait(timeout=10), 0)
        result = json.loads((observed["run"] / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["descendants_exited"])
        with self.assertRaises(ProcessLookupError):
            os.kill(self.descendant_identity[0], 0)

    def test_token_failure_signals_real_guard_and_reaps_detached_child(self):
        self._real_guard_with_startup_fault("token")

    def test_corrupt_marker_signals_real_guard_and_reaps_detached_child(self):
        self._real_guard_with_startup_fault("corrupt_marker")

    def test_supervisor_drains_output_without_unbounded_log(self):
        self.supervisor = subprocess.Popen(
            [sys.executable, "-m", "rag_arch.office_conversion", "--supervise", str(self.run),
             sys.executable, "-c", "import sys; sys.stdout.write('x'*200000); sys.stdout.flush()"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.assertEqual(self.supervisor.wait(timeout=8), 0)
        result = json.loads((self.run / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "success")
        self.assertEqual((self.run / "converter.log").stat().st_size, 64 * 1024)


if __name__ == "__main__":
    unittest.main()
