"""Operator recovery flags use mock repositories and never open a database."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rag_arch.web import worker_process


class RecoveryCommandTests(unittest.TestCase):
    def test_default_recover_preserves_existing_behavior_and_output(self):
        repository = object()
        outcome = [{"kind": "lease_check", "marked_stale_workers": ["worker"],
                    "parent_checks": [{"state": "exited"}]}]
        with TemporaryDirectory() as root, \
                patch.object(worker_process, "_database", return_value=object()), \
                patch.object(worker_process, "_repository", return_value=repository), \
                patch("rag_arch.web.resource_recovery.recover_resources", return_value=outcome) as recover, \
                patch("rag_arch.web.resource_recovery.maintain_recovery") as maintain, \
                redirect_stdout(StringIO()) as output:
            self.assertEqual(worker_process.main(["recover", "--root", root]), 0)
        recover.assert_called_once_with(repository, cleanup_owned=False, root=Path(root).resolve())
        maintain.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(set(report), {"marked_stale_workers", "local_worker_checks",
                                      "local_resource_checks", "cleanup_owned_resources", "note"})
        self.assertEqual(report["marked_stale_workers"], ["worker"])
        self.assertFalse(report["cleanup_owned_resources"])
        with redirect_stdout(StringIO()) as help_output, self.assertRaises(SystemExit) as exit_code:
            worker_process.build_parser().parse_args(["recover", "--help"])
        self.assertEqual(exit_code.exception.code, 0)
        self.assertIn("LibreOffice进程组只核验退出，不由此命令终止", "".join(help_output.getvalue().split()))

    def test_isolation_is_explicit_and_keeps_remote_state_unknown(self):
        repository = object()
        outcome = [{"kind": "task_isolation", "isolated_tasks": ["task"],
                    "external_work_state": "unknown", "automatic_retry": False}]
        with TemporaryDirectory() as root, \
                patch.object(worker_process, "_database", return_value=object()), \
                patch.object(worker_process, "_repository", return_value=repository), \
                patch("rag_arch.web.resource_recovery.recover_resources") as recover, \
                patch("rag_arch.web.resource_recovery.maintain_recovery", return_value=outcome) as maintain, \
                redirect_stdout(StringIO()) as output:
            self.assertEqual(worker_process.main(["recover", "--root", root,
                                                 "--isolate-interrupted"]), 0)
        maintain.assert_called_once_with(repository, cleanup_owned=False, root=Path(root).resolve(),
                                        cooldown_seconds=120.0)
        recover.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["local_resource_checks"], outcome)
        self.assertTrue(report["isolate_interrupted"])
        self.assertEqual(report["cooldown_seconds"], 120.0)
        self.assertIn("远端调用结果仍未知", report["note"])
        self.assertIn("不会自动重试", report["note"])

    def test_explicit_cleanup_and_cooldown_are_forwarded(self):
        repository = object()
        with TemporaryDirectory() as root, \
                patch.object(worker_process, "_database", return_value=object()), \
                patch.object(worker_process, "_repository", return_value=repository), \
                patch("rag_arch.web.resource_recovery.maintain_recovery", return_value=[]) as maintain, \
                redirect_stdout(StringIO()):
            self.assertEqual(worker_process.main(["recover", "--root", root,
                "--isolate-interrupted", "--cooldown-seconds", "7.5", "--cleanup-owned-resources"]), 0)
        maintain.assert_called_once_with(repository, cleanup_owned=True, root=Path(root).resolve(),
                                        cooldown_seconds=7.5)

    def test_invalid_cooldown_never_opens_database(self):
        for value in ("nan", "inf", "-inf", "-1", "abc"):
            with self.subTest(value=value), \
                    patch.object(worker_process, "_database") as database, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                worker_process.main(["recover", "--isolate-interrupted", "--cooldown-seconds=" + value])
            database.assert_not_called()
        with patch.object(worker_process, "_database") as database, redirect_stderr(StringIO()):
            self.assertEqual(worker_process.main(["recover", "--cooldown-seconds", "0"]), 1)
        database.assert_not_called()


if __name__ == "__main__":
    unittest.main()
