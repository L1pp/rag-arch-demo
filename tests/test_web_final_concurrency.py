"""Final real-process checks for shared auth and bounded multi-user queues.

Run with the project's .venv-web Python and
RAG_ARCH_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:<ephemeral-port>/rag_arch_test?sslmode=disable.
All API/worker processes use the disposable random schema created by the
inherited process-test harness. No model, Office process, or external service
is involved.
"""
from __future__ import annotations

import unittest
from uuid import uuid4

import httpx

from tests.test_web_durable_processes import (
    ASSISTANTS,
    TERMINAL_STATUSES,
    DurableTaskProcessTests,
)


class FinalMultiUserConcurrencyTests(DurableTaskProcessTests):
    """Reuse the exact-PID process harness while running only final scenarios."""

    def _environment(self) -> dict[str, str]:
        env = super()._environment()
        env["RAG_ARCH_TEST_API_ONLY"] = "1"
        return env

    def _json_client(self) -> httpx.Client:
        client = httpx.Client(timeout=httpx.Timeout(5.0))
        self.clients.append(client)
        return client

    def _json_login(self, api: dict, client: httpx.Client, username: str,
                    password: str = "alice-test-password-1") -> httpx.Response:
        return client.post(
            api["base_url"] + "/api/auth/login",
            json={"username": username, "password": password},
            headers={"Origin": api["base_url"]},
        )

    def _login(self, api: dict, username: str) -> tuple[httpx.Client, dict]:
        password = "alice-test-password-1" if username == "Alice" else "bob-test-password-1"
        client = self._json_client()
        response = self._json_login(api, client, username, password)
        self.assertEqual(response.status_code, 200, response.text)
        workspace_response = client.get(api["base_url"] + "/api/workspace")
        self.assertEqual(workspace_response.status_code, 200, workspace_response.text)
        return client, workspace_response.json()

    def _start_api(self) -> dict:
        api = super()._start_api()
        for path in ("/", "/login", "/assets"):
            response = httpx.get(api["base_url"] + path, timeout=1.0)
            self.assertEqual(response.status_code, 404,
                             f"API-only进程必须关闭静态路由 {path}: {response.text}")
        return api

    def test_two_api_json_login_budgets_and_shared_session_revocation(self):
        first_api = self._start_api()
        second_api = self._start_api()
        self.assertNotEqual(first_api["server_identity"]["pid"],
                            second_api["server_identity"]["pid"])

        # The JSON login cookie resolves on either independent API process.
        alice_first = self._json_client()
        login = self._json_login(first_api, alice_first, "Alice")
        self.assertEqual(login.status_code, 200, login.text)
        self.assertTrue(login.json()["authenticated"])
        session_token = alice_first.cookies.get("rag_arch_session")
        self.assertTrue(session_token)

        alice_second = self._json_client()
        alice_second.cookies.set("rag_arch_session", session_token,
                                 domain="127.0.0.1", path="/")
        session = alice_second.get(second_api["base_url"] + "/api/auth/session")
        self.assertEqual(session.status_code, 200, session.text)
        self.assertEqual(session.json()["owner_user_id"], self.user_ids["Alice"])

        # Revoking through API A invalidates the still-present cookie at API B.
        logout = alice_first.post(
            first_api["base_url"] + "/api/auth/logout",
            headers={"Origin": first_api["base_url"]},
        )
        self.assertEqual(logout.status_code, 200, logout.text)
        self.assertFalse(logout.json()["authenticated"])
        revoked = alice_second.get(second_api["base_url"] + "/api/auth/session")
        self.assertEqual(revoked.status_code, 401, revoked.text)

        # Nine failures for Alice are shared across APIs. Bob's correct login
        # uses the tenth client slot and clears the IP-wide window, while
        # Alice's username+client failures remain. One more Alice failure
        # reaches her target budget without exhausting the shared IP bucket.
        shared_client = self._json_client()
        for index in range(9):
            target_name = ("ALICE" if index % 3 == 0 else
                           " alice " if index % 3 == 1 else "Alice")
            failed = self._json_login(
                first_api, shared_client, target_name, "incorrect-alice-password"
            )
            self.assertEqual(failed.status_code, 401, failed.text)
        bob_success = self._json_login(
            second_api, shared_client, "Bob", "bob-test-password-1"
        )
        self.assertEqual(bob_success.status_code, 200, bob_success.text)
        failed = self._json_login(
            first_api, shared_client, " alice ", "incorrect-alice-password"
        )
        self.assertEqual(failed.status_code, 401, failed.text)
        bob_success = self._json_login(
            second_api, shared_client, "Bob", "bob-test-password-1"
        )
        self.assertEqual(bob_success.status_code, 200, bob_success.text)
        limited = self._json_login(second_api, shared_client, "Alice")
        self.assertEqual(limited.status_code, 429, limited.text)
        self.assertGreaterEqual(int(limited.headers.get("Retry-After", "0")), 1)

        # The IP-wide budget is a separate bucket. Distinct unknown usernames
        # consume it across API A/B without filling any username+client bucket.
        def fail_unknown_accounts(count: int, prefix: str) -> None:
            for index in range(count):
                api = first_api if index % 2 == 0 else second_api
                failed = self._json_login(
                    api, shared_client, f"{prefix}-{uuid4().hex}", "invalid-password"
                )
                self.assertEqual(failed.status_code, 401,
                                 f"{prefix} attempt {index + 1}: {failed.text}")

        # A correct login is admitted as the tenth request and clears the shared
        # IP window. Repeating that cycle proves failures do not accumulate
        # across successful login through another API process.
        for cycle in ("reset-a", "reset-b"):
            fail_unknown_accounts(9, cycle)
            successful = self._json_login(
                second_api, shared_client, "Bob", "bob-test-password-1"
            )
            self.assertEqual(successful.status_code, 200, successful.text)

        fail_unknown_accounts(10, "ip-budget")
        address_limited = self._json_login(
            second_api, shared_client, "Bob", "bob-test-password-1"
        )
        self.assertEqual(address_limited.status_code, 429, address_limited.text)
        self.assertGreaterEqual(int(address_limited.headers.get("Retry-After", "0")), 1)

    def test_two_users_queue_cancel_and_complete_without_duplicate_runs(self):
        first_api = self._start_api()
        second_api = self._start_api()
        alice, alice_workspace = self._login(first_api, "Alice")
        bob, bob_workspace = self._login(second_api, "Bob")
        plan_assets = {
            "Alice": self._upload_plan(first_api, alice, alice_workspace),
            "Bob": self._upload_plan(second_api, bob, bob_workspace),
        }
        self.repository.set_global_limit(2)

        submitted: list[tuple[str, dict, httpx.Client, dict, dict]] = []

        def submit_wave(wave: int) -> None:
            for username, api, client, workspace in (
                ("Alice", first_api, alice, alice_workspace),
                ("Bob", second_api, bob, bob_workspace),
            ):
                for assistant in ASSISTANTS:
                    attachments = ([plan_assets[username]]
                                   if assistant == "plan" else None)
                    response = self._submit(
                        api, client, workspace, assistant,
                        request_id=f"wave-{wave}-{username}-{assistant}-{uuid4().hex}",
                        prompt=f"wave {wave} {username} {assistant}",
                        duration_seconds=1.25,
                        attachment_ids=attachments,
                    )
                    self.assertEqual(response.status_code, 202, response.text)
                    submitted.append((username, api, client, workspace, response.json()))

        # Queue six tasks before starting any worker, then cancel Bob's queued
        # plan and immediately prove that its assistant slot can be reused.
        submit_wave(1)
        cancelled_item = next(item for item in submitted
                              if item[0] == "Bob" and item[4]["assistant_id"] == "plan")
        cancelled_task = cancelled_item[4]
        cancel = bob.post(
            second_api["base_url"] + f"/api/tasks/{cancelled_task['task_id']}/cancel",
            json={"workspace_id": bob_workspace["workspace_id"]},
            headers={"Origin": second_api["base_url"]},
        )
        self.assertEqual(cancel.status_code, 200, cancel.text)
        self.assertEqual(cancel.json()["status"], "cancelled")
        self.assertEqual(
            sum(event["payload"].get("status") == "running"
                and event["payload"].get("stage") == "starting"
                for event in self._task_events(cancelled_task["task_id"])),
            0,
            "排队任务取消后不得被worker领取",
        )
        replacement = self._submit(
            second_api, bob, bob_workspace, "plan",
            request_id=f"replacement-plan-{uuid4().hex}",
            prompt="replacement after queued cancellation", duration_seconds=1.25,
            attachment_ids=[plan_assets["Bob"]],
        )
        self.assertEqual(replacement.status_code, 202, replacement.text)
        submitted.append(("Bob", second_api, bob, bob_workspace, replacement.json()))

        workers = [self._start_worker(max_tasks=None) for _ in range(3)]
        self._wait_until(
            lambda: all(self._worker_registered(worker) for worker in workers),
            10,
            "三个独立worker没有全部注册",
        )
        first_wave_ids = [item[4]["task_id"] for item in submitted]
        self._wait_until(
            lambda: all(self._task_row(task_id)["status"] in TERMINAL_STATUSES
                        for task_id in first_wave_ids),
            25,
            "第一波多用户任务没有在期限内完成",
        )
        for username, api, client, workspace, _task in submitted:
            snapshot = self.repository.snapshot(
                self.user_ids[username],
                {name: "simulation" for name in ASSISTANTS},
            )
            self.assertTrue(all(slot["active_task_id"] is None
                                for slot in snapshot["assistants"]))

        # Reuse the released per-user slots for a second wave. This keeps total
        # database/process load small while covering thirteen submissions and
        # twelve executions.
        submit_wave(2)
        all_task_ids = [item[4]["task_id"] for item in submitted]
        self._wait_until(
            lambda: all(self._task_row(task_id)["status"] in TERMINAL_STATUSES
                        for task_id in all_task_ids),
            25,
            "第二波多用户任务没有在期限内完成",
        )

        rows = {task_id: self._task_row(task_id) for task_id in all_task_ids}
        self.assertEqual(rows[cancelled_task["task_id"]]["status"], "cancelled")
        self.assertEqual(sum(row["status"] == "completed" for row in rows.values()), 12)
        self.assertEqual(len(rows), 13)
        for task_id in all_task_ids:
            events = self._task_events(task_id)
            starts = [event for event in events
                      if event["payload"].get("status") == "running"
                      and event["payload"].get("stage") == "starting"]
            terminals = [event for event in events
                         if event["payload"].get("status") in TERMINAL_STATUSES]
            if task_id == cancelled_task["task_id"]:
                self.assertEqual(starts, [])
                self.assertEqual(len(terminals), 1)
            else:
                self.assertEqual(len(starts), 1, f"task重复启动或未启动：{task_id}")
                self.assertEqual(len(terminals), 1, f"task重复完成：{task_id}")
                self.assertEqual(rows[task_id]["status"], "completed")

        peak = self._global_peak_running(all_task_ids)
        self.assertLessEqual(peak, 2, f"全站执行峰值{peak}超过配置限额2")
        self.assertEqual(peak, 2, "应观察到两个独立worker并发执行")
        for username in ("Alice", "Bob"):
            snapshot = self.repository.snapshot(
                self.user_ids[username],
                {name: "simulation" for name in ASSISTANTS},
            )
            self.assertTrue(all(slot["active_task_id"] is None
                                for slot in snapshot["assistants"]))


def load_tests(loader, standard_tests, pattern):
    """Select only this file's new tests, not the inherited step-3 suite."""
    suite = unittest.TestSuite()
    for name in (
        "test_two_api_json_login_budgets_and_shared_session_revocation",
        "test_two_users_queue_cancel_and_complete_without_duplicate_runs",
    ):
        suite.addTest(FinalMultiUserConcurrencyTests(name))
    return suite


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
