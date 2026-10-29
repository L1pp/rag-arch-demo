from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_arch.web.resource_recovery import _recover_one_resource


class OfficeResourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.worker = {"worker_instance": "worker"}
        self.resource = {
            "resource_id": "office", "state": "running", "kind": "word",
            "pid": 123, "process_start_token": "token", "host_id": "local",
            "executable": "/usr/lib/libreoffice/program/soffice.bin",
            "owner_path": str(self.root / "data/runtime/office_conversion/active.json"),
            "owner_descriptor_path": "trusted-owner.json",
        }
        self.child = {key: self.resource[key] for key in (
            "pid", "process_start_token", "host_id", "executable")}
        self.repository = Mock()
        self.repository.resource_exited.return_value = True
        self.repository.backfill_unknown_resource_identity.return_value = {"state": "unknown"}
        self.probe = Mock(return_value="exited")
        self.descriptor = self.enterContext(patch(
            "rag_arch.web.resource_recovery.read_owner_descriptor",
            return_value=({"identity": "trusted"}, {"file": "trusted"})))
        self.enterContext(patch(
            "rag_arch.web.resource_recovery.validate_owner_descriptor",
            return_value=self.child))
        self.office_probe = self.enterContext(patch(
            "rag_arch.office_conversion.probe_office_owner", return_value="unknown"))
        # The probe is a fake here; the marker remains a real temporary file.
        import json
        import os
        marker = Path(self.resource["owner_path"])
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"resource_id": self.resource["resource_id"],
            "pid": self.child["pid"], "token": self.child["process_start_token"],
            "executable": self.child["executable"], "group_id": self.child["pid"]}), encoding="utf-8")
        if os.name == "posix":
            marker.chmod(0o600)

    def recover(self, resource=None):
        return _recover_one_resource(
            self.repository, self.worker, resource or self.resource,
            root=self.root, cleanup_owned=True, local_host="local", probe=self.probe)

    def test_launcher_exit_does_not_release_unknown_office_descendants(self):
        result = self.recover()
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["action"], "office_group_exit_unconfirmed")
        self.repository.resource_exited.assert_not_called()
        self.probe.assert_not_called()

    def test_only_full_office_exit_proof_marks_resource_exited(self):
        self.office_probe.return_value = "exited"
        result = self.recover()
        self.assertEqual(result["state"], "exited")
        self.repository.resource_exited.assert_called_once_with(
            "worker", "office", pid=123, process_start_token="token")

    def test_backfilled_launcher_does_not_become_exited_without_group_proof(self):
        result = self.recover({**self.resource, "pid": None, "state": "unknown"})
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(self.repository.backfill_unknown_resource_identity.call_args.kwargs[
            "probe_state"], "unknown")
        self.repository.resource_exited.assert_not_called()

    def test_untrusted_owner_never_releases_a_slot(self):
        result = self.recover({**self.resource, "owner_path": str(self.root / "active.json")})
        self.assertEqual(result["action"], "office_owner_untrusted")
        self.office_probe.assert_not_called()
        self.repository.resource_exited.assert_not_called()

    def test_remote_office_resource_is_not_probed_on_this_host(self):
        self.child["host_id"] = "other-host"
        result = self.recover()
        self.assertEqual(result["action"], "remote_child")
        self.office_probe.assert_not_called()


class OfficeRecoveryMarkerTests(unittest.TestCase):
    """Only synthetic identities and result markers are created in temp storage."""

    def setUp(self):
        import json
        import os
        import socket
        import sys
        from rag_arch.office_conversion import _RESULT_SCHEMA, _SCHEMA, _private_directory
        from rag_arch.resource_supervision import resource_owner_descriptor_path

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.host = socket.gethostname()
        self.worker = {"worker_instance": "synthetic-worker", "host_id": self.host,
                       "pid": 222, "process_start_token": "90.000001"}
        self.child = {"host_id": self.host, "pid": 12345, "process_start_token": "12.345678",
                      "executable": str(Path(sys.executable).resolve())}
        directory = _private_directory(self.root)
        run = directory / "runs" / "synthetic-run"
        run.mkdir(parents=True, mode=0o700)
        self.marker = directory / "active.json"
        self.resource = {"resource_id": "a" * 32, "state": "running", "kind": "word",
                         **self.child, "owner_path": str(self.marker), "owner_nonce": "synthetic-nonce"}
        descriptor = resource_owner_descriptor_path(self.root, self.worker["worker_instance"],
                                                    self.resource["resource_id"])
        descriptor.parent.mkdir(parents=True, mode=0o700)
        self.resource["owner_descriptor_path"] = str(descriptor)
        descriptor.write_text(json.dumps({"schema_version": 1, "resource_id": self.resource["resource_id"],
            "owner_nonce": self.resource["owner_nonce"], "worker_instance": self.worker["worker_instance"],
            "owner_path": str(self.marker), "owned": True,
            "worker": {k: self.worker[k] for k in ("host_id", "pid", "process_start_token")},
            "resource": self.child}), encoding="utf-8")
        self.owner = {"schema": _SCHEMA, "host": self.host, "phase": "running",
                      "pid": self.child["pid"], "token": self.child["process_start_token"],
                      "executable": self.child["executable"], "group_id": self.child["pid"],
                      "run_directory": str(run), "resource_id": self.resource["resource_id"]}
        self.marker.write_text(json.dumps(self.owner), encoding="utf-8")
        completion = run / "result.json"
        completion.write_text(json.dumps({"schema": _RESULT_SCHEMA, "pid": self.child["pid"],
            "token": self.child["process_start_token"], "descendants_exited": True,
            "status": "success"}), encoding="utf-8")
        if os.name == "posix":
            for path in (descriptor, self.marker, completion):
                path.chmod(0o600)
        self.repository = Mock()
        self.repository.resource_exited.return_value = True
        self.repository.backfill_unknown_resource_identity.return_value = {"state": "exited"}

    def recover(self, resource=None):
        with patch("rag_arch.office_conversion.sys.platform", "linux"), \
             patch("rag_arch.office_conversion.probe_process", return_value="exited"):
            return _recover_one_resource(self.repository, self.worker, resource or self.resource,
                root=self.root, cleanup_owned=True, local_host=self.host, probe=Mock())

    def test_confirmed_recovery_allows_next_conversion(self):
        from rag_arch.office_conversion import OfficeConverter
        outcome = self.recover()
        self.assertEqual(outcome["state"], "exited")
        self.assertFalse(self.marker.exists())
        upload = self.root / "data/runtime/uploads"
        upload.mkdir(parents=True)
        source, target = upload / "fictional.doc", upload / "fictional.docx"
        source.write_bytes(b"fictional-legacy-document")
        converter = OfficeConverter(self.root, timeout=5)
        with patch.object(converter, "_convert_locked") as conversion:
            converter.ensure_docx(source, target)
        conversion.assert_called_once()


    def test_backfilled_identity_retires_confirmed_marker(self):
        resource = {**self.resource, "state": "unknown", "pid": None,
                    "process_start_token": None, "host_id": None, "executable": None}
        outcome = self.recover(resource)
        self.assertEqual((outcome["state"], outcome["action"]),
                         ("exited", "office_group_exit_backfilled"))
        self.assertEqual(self.repository.backfill_unknown_resource_identity.call_args.kwargs["probe_state"], "exited")
        self.assertFalse(self.marker.exists())
        self.repository.resource_exited.assert_not_called()

    def test_rejected_database_exit_or_backfill_keeps_marker(self):
        for backfill in (False, True):
            with self.subTest(backfill=backfill):
                self.repository.reset_mock()
                self.repository.resource_exited.return_value = False
                self.repository.backfill_unknown_resource_identity.return_value = {"state": "unknown"}
                resource = ({**self.resource, "state": "unknown", "pid": None,
                             "process_start_token": None, "host_id": None, "executable": None}
                            if backfill else self.resource)
                outcome = self.recover(resource)
                self.assertEqual((outcome["state"], outcome["action"]), ("unknown", "exit_identity_rejected"))
                self.assertTrue(self.marker.exists())

    def test_marker_identity_mismatches_and_unknown_exit_remain_blocked(self):
        import json
        changed_values = {
            "resource_id": "b" * 32, "pid": 54321, "token": "99.999999",
            "group_id": 54321, "executable": str(self.root / "other-executable"),
            "run_directory": str(self.root / "outside"),
        }
        for field, value in changed_values.items():
            with self.subTest(field=field):
                self.repository.reset_mock()
                altered = {**self.owner, field: value}
                self.marker.write_text(json.dumps(altered), encoding="utf-8")
                result = self.recover()
                self.assertEqual(result["state"], "unknown")
                self.assertEqual(json.loads(self.marker.read_text(encoding="utf-8")), altered)
                self.repository.resource_exited.assert_not_called()
        self.marker.write_text(json.dumps(self.owner), encoding="utf-8")
        result_file = Path(self.owner["run_directory"]) / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        result["descendants_exited"] = False
        result_file.write_text(json.dumps(result), encoding="utf-8")
        self.repository.reset_mock()
        outcome = self.recover()
        self.assertEqual(outcome["state"], "unknown")
        self.assertTrue(self.marker.exists())
        self.repository.resource_exited.assert_not_called()

    def test_new_owner_appearing_during_exit_commit_is_never_deleted(self):
        import json
        replacement = {**self.owner, "resource_id": "b" * 32, "pid": 54321,
                       "token": "99.999999", "group_id": 54321}
        def replace_owner(*args, **kwargs):
            self.marker.write_text(json.dumps(replacement), encoding="utf-8")
            return True
        self.repository.resource_exited.side_effect = replace_owner
        outcome = self.recover()
        self.assertEqual((outcome["state"], outcome["action"]), ("exited", "office_marker_changed"))
        self.assertEqual(json.loads(self.marker.read_text(encoding="utf-8")), replacement)

    def test_marker_cleanup_failure_is_retried_after_database_exit(self):
        with patch.object(Path, "unlink", side_effect=PermissionError("synthetic denial")):
            outcome = self.recover()
        self.assertEqual((outcome["state"], outcome["action"]), ("exited", "office_marker_cleanup_pending"))
        self.assertTrue(self.marker.exists())
        self.repository.reset_mock()
        outcome = self.recover({**self.resource, "state": "exited"})
        self.assertEqual((outcome["state"], outcome["action"]), ("exited", "office_marker_retired_after_exit"))
        self.assertFalse(self.marker.exists())
        self.repository.resource_exited.assert_not_called()

    def test_exit_commit_unknown_outcome_keeps_marker_until_exited_row_is_rechecked(self):
        self.repository.resource_exited.side_effect = ConnectionError("synthetic lost response")
        self.repository.resource_unknown.return_value = False  # The exit may already have committed.
        outcome = self.recover()
        self.assertEqual(outcome["state"], "unknown")
        self.assertTrue(self.marker.exists())
        self.repository.reset_mock()
        outcome = self.recover({**self.resource, "state": "exited"})
        self.assertEqual((outcome["state"], outcome["action"]),
                         ("exited", "office_marker_retired_after_exit"))
        self.assertFalse(self.marker.exists())
        self.repository.resource_exited.assert_not_called()

    def test_changed_probe_before_commit_keeps_marker_and_capacity(self):
        with patch("rag_arch.office_conversion.probe_office_owner", side_effect=["exited", "unknown"]):
            outcome = self.recover()
        self.assertEqual((outcome["state"], outcome["action"]),
                         ("unknown", "office_group_exit_unconfirmed"))
        self.assertTrue(self.marker.exists())
        self.repository.resource_exited.assert_not_called()

    def test_busy_conversion_lock_does_not_touch_marker_or_database(self):
        from contextlib import contextmanager
        from rag_arch.office_conversion import OfficeConverter
        @contextmanager
        def unavailable(*args, **kwargs):
            raise TimeoutError("synthetic busy lock")
            yield
        with patch.object(OfficeConverter, "_lock", new=unavailable):
            outcome = self.recover()
        self.assertEqual((outcome["state"], outcome["action"]), ("unknown", "office_marker_lock_unavailable"))
        self.assertTrue(self.marker.exists())
        self.repository.resource_exited.assert_not_called()
        self.repository.backfill_unknown_resource_identity.assert_not_called()

    def test_database_exit_and_marker_removal_are_under_conversion_lock(self):
        from contextlib import contextmanager
        from rag_arch.office_conversion import OfficeConverter
        original_lock = OfficeConverter._lock
        original_unlink = Path.unlink
        held = []
        @contextmanager
        def tracked_lock(converter, directory, deadline):
            with original_lock(converter, directory, deadline):
                held.append(True)
                try:
                    yield
                finally:
                    held.pop()
        def guarded_unlink(path, *args, **kwargs):
            if path == self.marker:
                self.assertEqual(held, [True])
            return original_unlink(path, *args, **kwargs)
        def database_exit(*args, **kwargs):
            self.assertEqual(held, [True])
            return True
        self.repository.resource_exited.side_effect = database_exit
        with patch.object(OfficeConverter, "_lock", new=tracked_lock), \
             patch.object(Path, "unlink", new=guarded_unlink):
            outcome = self.recover()
        self.assertEqual(outcome["state"], "exited")
        self.assertFalse(self.marker.exists())

    def test_exited_old_resource_does_not_remove_new_owners_marker(self):
        import json
        replacement = {**self.owner, "resource_id": "b" * 32}
        self.marker.write_text(json.dumps(replacement), encoding="utf-8")
        result = self.recover({**self.resource, "state": "exited"})
        self.assertEqual((result["state"], result["action"]), ("exited", "already_exited"))
        self.assertEqual(json.loads(self.marker.read_text(encoding="utf-8")), replacement)
        self.repository.resource_exited.assert_not_called()
        self.repository.resource_unknown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
