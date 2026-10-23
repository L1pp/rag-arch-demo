import asyncio
import json
import os
import threading
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.sse import ServerSentEvent
from fastapi.testclient import TestClient
from psycopg import sql
from starlette.concurrency import run_in_threadpool

from rag_arch.web.app import _event_session_check, create_app, event_stream
from rag_arch.web.auth import PostgresAuthStore, SESSION_COOKIE_NAME
from rag_arch.web.db import Database
from rag_arch.web.events import SSEConnectionLimiter, resume_cursor
from rag_arch.web.persistent_tasks import DatabaseTaskService
from rag_arch.web.task_repository import TaskRepository


class DurableSSEIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")
        if not self.url:
            self.skipTest("dedicated RAG_ARCH_TEST_DATABASE_URL required")
        self.database = Database(self.url, schema="sse_" + uuid4().hex)
        self.database.migrate()
        self.addCleanup(self._drop_schema)
        self.auth = PostgresAuthStore(self.database, password_iterations=1500)
        self.repository = TaskRepository(self.database)
        self.temp = TemporaryDirectory(prefix="rag-sse-test-")
        self.addCleanup(self.temp.cleanup)
        self.temp_root = self.temp.name
        self.alice, self.alice_token, self.alice_service = self._new_service("Alice")

    def _drop_schema(self):
        import psycopg
        with psycopg.connect(self.url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(self.database.schema)))

    def _new_service(self, username):
        self.auth.create_user(username, "SSE-test-password-1")
        token, _ = self.auth.create_session(username)
        identity = self.auth.resolve_identity(token)
        workspace_id = self.database.ensure_workspace(identity.user_id)
        service = DatabaseTaskService(
            self.repository, user_id=identity.user_id, workspace_id=workspace_id,
            execution_modes={name: "simulation" for name in ("chat", "standards", "plan")})
        return identity, token, service

    def _append_events(self, service, count, *, start=0):
        with self.database.transaction() as connection:
            for number in range(start, start + count):
                TaskRepository._append_event(
                    connection, service.user_id, service.workspace_id,
                    "test.event", {"number": number})

    def _session_check(self, token, user_id):
        async def check():
            identity = await run_in_threadpool(self.auth.resolve_identity, token)
            return identity is not None and identity.user_id == user_id
        return check

    def test_cursor_parsing_and_expired_ahead_or_invalid_resync_reasons(self):
        service = self.alice_service
        self._append_events(service, 3)
        self.assertEqual(resume_cursor("0", f"{service.server_instance_id}:2",
                                       service.server_instance_id), (2, None))
        self.assertEqual(resume_cursor("bad", None, service.server_instance_id),
                         (0, "invalid_cursor"))
        self.assertEqual(resume_cursor("-1", None, service.server_instance_id),
                         (0, "invalid_cursor"))
        self.assertEqual(resume_cursor("9223372036854775808", None,
                                       service.server_instance_id), (0, "invalid_cursor"))
        self.assertEqual(resume_cursor("0", f"other-workspace:1",
                                       service.server_instance_id), (0, "server_restarted"))

        events, reason = self.repository.events_after(service.user_id, -1)
        self.assertEqual((events, reason), ([], "invalid_cursor"))
        events, reason = self.repository.events_after(service.user_id, 4)
        self.assertEqual((events, reason), ([], "cursor_ahead"))

        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM workspace_events WHERE user_id = %s AND workspace_id = %s",
                (service.user_id, service.workspace_id))
        events, reason = self.repository.events_after(service.user_id, 0)
        self.assertEqual((events, reason), ([], "cursor_expired"))
        # A client exactly at the persisted high-water mark has missed nothing.
        events, reason = self.repository.events_after(service.user_id, 3)
        self.assertEqual((events, reason), ([], None))

        stream = event_stream(service, "bad")
        item = asyncio.run(stream.__anext__())
        self.assertEqual(item.event, "snapshot_required")
        self.assertEqual(item.data["reason"], "invalid_cursor")
        asyncio.run(stream.aclose())

    def test_replay_is_owner_scoped_and_rejects_another_workspace_event_id(self):
        bob, _, bob_service = self._new_service("Bob")
        self._append_events(self.alice_service, 1, start=10)
        self._append_events(bob_service, 1, start=90)
        self.assertNotEqual(self.alice_service.workspace_id, bob_service.workspace_id)
        alice_events, alice_reason = self.repository.events_after(self.alice.user_id, 0)
        bob_events, bob_reason = self.repository.events_after(bob.user_id, 0)
        self.assertIsNone(alice_reason)
        self.assertIsNone(bob_reason)
        self.assertEqual(alice_events[0]["data"], {"number": 10})
        self.assertEqual(bob_events[0]["data"], {"number": 90})

        async def read_wrong_workspace():
            stream = event_stream(
                self.alice_service, "0",
                last_event_id=f"{bob_service.server_instance_id}:1")
            try:
                return await stream.__anext__()
            finally:
                await stream.aclose()

        item = asyncio.run(read_wrong_workspace())
        self.assertEqual(item.event, "snapshot_required")
        self.assertEqual(item.data["reason"], "server_restarted")

    def test_slow_page_rechecks_revocation_between_yields_with_monotonic_clock(self):
        service = self.alice_service
        self._append_events(service, 2)
        now = [100.0]
        valid = [True]
        checks = []

        async def check_session():
            checks.append(now[0])
            return valid[0]

        async def read_two():
            stream = event_stream(service, "0", session_check=check_session,
                                  monotonic=lambda: now[0])
            try:
                first = await stream.__anext__()
                self.assertEqual(first.event, "test.event")
                valid[0] = False
                now[0] += 0.51
                second = await stream.__anext__()
                return first, second
            finally:
                await stream.aclose()

        first, second = asyncio.run(read_two())
        self.assertEqual(first.id, f"{service.server_instance_id}:1")
        self.assertEqual(second.event, "auth_required")
        self.assertIsNone(second.data)
        # Initial poll and pre-send batch checks; the next identity check occurs
        # only after the fake monotonic clock crosses the 0.5-second interval.
        self.assertEqual(len(checks), 3)
        self.assertEqual(service.events._subscriptions, {})

    def test_replays_across_bounded_pages_and_snapshot_to_subscribe_gap(self):
        service = self.alice_service
        self._append_events(service, 205)

        async def consume_pages():
            stream = event_stream(service, "0")
            try:
                return [await asyncio.wait_for(stream.__anext__(), timeout=3)
                        for _ in range(205)]
            finally:
                await stream.aclose()

        replayed = asyncio.run(consume_pages())
        numbers = [event.data["number"] for event in replayed]
        self.assertEqual(numbers, list(range(205)))
        self.assertEqual(replayed[199].id, f"{service.server_instance_id}:200")
        self.assertEqual(replayed[204].id, f"{service.server_instance_id}:205")

        snapshot = service.snapshot()
        self.assertEqual(snapshot["cursor"], 205)
        # This event commits after the snapshot and before the SSE generator
        # subscribes.  Durable replay must still deliver it.
        self._append_events(service, 1, start=205)

        async def replay_after_snapshot():
            stream = event_stream(service, str(snapshot["cursor"]))
            try:
                return await asyncio.wait_for(stream.__anext__(), timeout=3)
            finally:
                await stream.aclose()

        pending = asyncio.run(replay_after_snapshot())
        self.assertEqual(pending.data, {"number": 205})
        self.assertEqual(pending.id, f"{service.server_instance_id}:206")

    def test_reset_and_task_events_replay_in_sequence_from_snapshot_boundary(self):
        service = self.alice_service
        task = service.submit("chat", "request-1", "synthetic input", duration_seconds=5)
        snapshot = service.snapshot()
        self.assertEqual(snapshot["cursor"], task["updated_seq"])
        service.cancel(task["task_id"])
        service.reset("chat")

        async def replay():
            stream = event_stream(service, str(snapshot["cursor"]))
            try:
                first = await stream.__anext__()
                second = await stream.__anext__()
                return first, second
            finally:
                await stream.aclose()

        first, second = asyncio.run(replay())
        self.assertEqual((first.event, second.event), ("task.updated", "assistant.reset"))
        self.assertEqual([first.id, second.id], [
            f"{service.server_instance_id}:2", f"{service.server_instance_id}:3"])
        self.assertEqual(service.snapshot()["cursor"], 3)

    def test_repository_detects_interior_sequence_gaps(self):
        service = self.alice_service
        self._append_events(service, 4)
        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM workspace_events WHERE user_id = %s AND workspace_id = %s AND seq = 3",
                (service.user_id, service.workspace_id))

        first_page, reason = self.repository.events_after(service.user_id, 0, limit=2)
        self.assertIsNone(reason)
        self.assertEqual([event["seq"] for event in first_page], [1, 2])
        next_page, reason = self.repository.events_after(service.user_id, 2, limit=2)
        self.assertEqual((next_page, reason), ([], "cursor_expired"))

    def test_session_revocation_reset_and_disable_after_read_prevent_batch_delivery(self):
        cases = (
            ("LogoutUser", lambda token: self.auth.revoke_session(token)),
            ("PasswordUser", lambda token: self.auth.set_password(
                "PasswordUser", "replacement-password-1")),
            ("DisabledUser", lambda token: self.auth.disable_user("DisabledUser")),
        )
        for index, (username, revoke) in enumerate(cases):
            with self.subTest(action=username):
                identity, token, service = self._new_service(username)
                self._append_events(service, 1, start=100 + index)
                original_read = service.events.read_after

                def read_then_revoke(cursor, *, _read=original_read,
                                     _token=token, _revoke=revoke):
                    page = _read(cursor)
                    _revoke(_token)
                    return page

                service.events.read_after = read_then_revoke
                stream = event_stream(
                    service, "0", session_check=self._session_check(token, identity.user_id))

                async def receive():
                    try:
                        return await asyncio.wait_for(stream.__anext__(), timeout=3)
                    finally:
                        await stream.aclose()

                item = asyncio.run(receive())
                self.assertEqual(item.event, "auth_required")
                self.assertIsNone(item.data)
                self.assertEqual(service.events._subscriptions, {})

    def test_logout_is_detected_on_poll_after_empty_page_and_cleans_subscription(self):
        token = self.alice_token
        service = self.alice_service
        loop_thread = threading.get_ident()
        observed_threads = []
        read_finished = threading.Event()
        original_read = service.events.read_after

        def observed_read(cursor):
            observed_threads.append(threading.get_ident())
            result = original_read(cursor)
            read_finished.set()
            return result

        service.events.read_after = observed_read

        async def run():
            stream = event_stream(
                service, str(service.events.cursor),
                session_check=self._session_check(token, self.alice.user_id))
            pending = asyncio.create_task(stream.__anext__())
            try:
                await asyncio.wait_for(asyncio.to_thread(read_finished.wait, 3), timeout=4)
                await run_in_threadpool(self.auth.revoke_session, token)
                return await asyncio.wait_for(pending, timeout=3)
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                await stream.aclose()

        item = asyncio.run(run())
        self.assertEqual(item.event, "auth_required")
        self.assertIsNone(item.data)
        self.assertTrue(observed_threads)
        self.assertNotIn(loop_thread, observed_threads)
        self.assertEqual(service.events._subscriptions, {})

    def test_identity_store_failure_closes_without_business_events(self):
        self._append_events(self.alice_service, 1)

        async def unavailable():
            raise RuntimeError("auth database unavailable")

        async def run():
            stream = event_stream(self.alice_service, "0", session_check=unavailable)
            try:
                with self.assertRaises(StopAsyncIteration):
                    await stream.__anext__()
            finally:
                await stream.aclose()

        asyncio.run(run())
        self.assertEqual(self.alice_service.events._subscriptions, {})

    def test_legacy_shared_service_session_check_keeps_principal_and_token_binding(self):
        class Auth:
            identity = type("Identity", (), {"user_id": 7})()

            def resolve_identity(self, token):
                self.seen_token = token
                return self.identity

        authentication = Auth()
        check = _event_session_check(authentication, "original-token", 7, None)
        self.assertTrue(asyncio.run(check()))
        self.assertEqual(authentication.seen_token, "original-token")
        authentication.identity = type("Identity", (), {"user_id": 8})()
        self.assertFalse(asyncio.run(check()))

    def test_connection_limiter_bounds_owner_and_process_and_releases_idempotently(self):
        limiter = SSEConnectionLimiter(max_total=2, max_per_owner=1)
        alice = limiter.reserve(1)
        bob = limiter.reserve(2)
        self.assertIsNotNone(alice)
        self.assertIsNotNone(bob)
        self.assertIsNone(limiter.reserve(1))
        self.assertIsNone(limiter.reserve(3))
        self.assertEqual(limiter.active_total, 2)
        self.assertEqual(limiter.active_for(1), 1)
        self.assertTrue(limiter.release(alice))
        self.assertFalse(limiter.release(alice))
        carol = limiter.reserve(3)
        self.assertIsNotNone(carol)
        self.assertEqual(limiter.active_total, 2)
        limiter.release(bob)
        limiter.release(carol)
        self.assertEqual(limiter.active_total, 0)

    def test_connection_limit_returns_429_and_releases_on_end_or_exception(self):
        app = create_app(
            root=self.temp_root, database=self.database, auth_store=self.auth,
            auth_cookie_secure=False, sse_max_connections=1,
            sse_max_connections_per_user=1,
            execution_modes={name: "simulation" for name in ("chat", "standards", "plan")})
        with TestClient(app) as client:
            login = client.post(
                "/login", data={"username": "Alice", "password": "SSE-test-password-1"},
                headers={"Origin": "http://testserver"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            alice_token = client.cookies.get(SESSION_COOKIE_NAME)
            alice_workspace = client.get("/api/workspace").json()
            owner_id = alice_workspace["owner_user_id"]
            limiter = app.state.sse_connection_limiter
            held_lease = limiter.reserve(owner_id)
            self.assertIsNotNone(held_lease)

            async def should_not_start(*args, **kwargs):
                if False:
                    yield None
                raise AssertionError("limited connection reached the stream generator")

            with patch("rag_arch.web.app.event_stream", new=should_not_start):
                rejected = client.get(
                    "/api/events",
                    params={"workspace_id": alice_workspace["workspace_id"], "after": "0"},
                    cookies={SESSION_COOKIE_NAME: alice_token})
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(rejected.headers.get("Retry-After"), "2")
            self.assertEqual(limiter.active_total, 1)
            limiter.release(held_lease)

            async def finite_stream(*args, **kwargs):
                yield ServerSentEvent(event="test.event", data={"sent": True})

            with patch("rag_arch.web.app.event_stream", new=finite_stream):
                accepted = client.get(
                    "/api/events",
                    params={"workspace_id": alice_workspace["workspace_id"], "after": "0"},
                    cookies={SESSION_COOKIE_NAME: alice_token})
            self.assertEqual(accepted.status_code, 200)
            self.assertIn("event: test.event", accepted.text)
            self.assertEqual(limiter.active_total, 0)

            async def broken_stream(*args, **kwargs):
                yield ServerSentEvent(event="test.event", data={"sent": True})
                raise RuntimeError("controlled stream failure")

            with patch("rag_arch.web.app.event_stream", new=broken_stream):
                with self.assertRaises(Exception):
                    client.get(
                        "/api/events",
                        params={"workspace_id": alice_workspace["workspace_id"], "after": "0"},
                        cookies={SESSION_COOKIE_NAME: alice_token})
            self.assertEqual(app.state.sse_connection_limiter.active_total, 0)

    def test_http_negative_cursor_returns_explicit_snapshot_required(self):
        app = create_app(
            root=self.temp_root, database=self.database, auth_store=self.auth,
            auth_cookie_secure=False,
            execution_modes={name: "simulation" for name in ("chat", "standards", "plan")})
        with TestClient(app) as client:
            login = client.post(
                "/login", data={"username": "Alice", "password": "SSE-test-password-1"},
                headers={"Origin": "http://testserver"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            workspace = client.get("/api/workspace").json()
            response = client.get(
                "/api/events", params={"workspace_id": workspace["workspace_id"], "after": "-1"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("event: snapshot_required", response.text)
            payloads = [json.loads(line[len("data: "):]) for line in response.text.splitlines()
                        if line.startswith("data: ")]
            self.assertTrue(any(payload.get("reason") == "invalid_cursor"
                                for payload in payloads))


if __name__ == "__main__":
    unittest.main()
