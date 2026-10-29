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


if __name__ == "__main__":
    unittest.main()
