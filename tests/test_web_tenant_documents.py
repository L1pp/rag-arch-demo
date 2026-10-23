"""PostgreSQL-backed tenant attachment and page-route isolation checks.

Set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL database.
Every test owns a random schema and uses synthetic users and fake page rendering.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from threading import Event, Lock, Thread, current_thread
import unittest
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_arch.standards import StandardCorpus
from rag_arch.web.auth import PostgresAuthStore, SESSION_COOKIE_NAME
from rag_arch.web.db import Database
from rag_arch.web.documents import DocumentStore
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.app import create_app


TEST_DATABASE_URL = os.environ.get("RAG_ARCH_TEST_DATABASE_URL")


class FakePageSources:
    """Small page-service double; the uploaded PDF bytes are intentionally fake."""

    image_bytes = b"synthetic-png"

    def __init__(self, root, *, collection_id="", pdf_backend=None, **kwargs):
        self.collection_id = collection_id
        self._sources = []

    def add(self, source_ref, filename, **kwargs):
        self._sources.append({"source_ref": source_ref, "filename": filename})

    def sources(self):
        return list(self._sources)

    def list_pages(self, source_ref, start=1, limit=20):
        if not any(row["source_ref"] == source_ref for row in self._sources):
            raise KeyError(source_ref)
        if start != 1:
            raise ValueError("page out of range")
        return {"page_count": 1, "pages": [{"page": 1}]}

    def read_page(self, source_ref, page, scale=1.5):
        if page != 1 or not any(row["source_ref"] == source_ref for row in self._sources):
            raise KeyError(source_ref)
        return {"image_bytes": self.image_bytes}


@unittest.skipUnless(TEST_DATABASE_URL, "set RAG_ARCH_TEST_DATABASE_URL to a dedicated disposable PostgreSQL instance")
class TenantDocumentIsolationTests(unittest.TestCase):
    def setUp(self):
        try:
            import psycopg
            from psycopg import sql
        except ImportError as exc:  # pragma: no cover - opt-in integration dependency
            raise unittest.SkipTest("psycopg 3 is required for PostgreSQL integration tests") from exc
        self.psycopg = psycopg
        self.sql = sql
        self.schema = f"test_documents_{uuid4().hex}"
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("CREATE SCHEMA {}").format(self.sql.Identifier(self.schema)))
        self.addCleanup(self._drop_schema)
        self.database = Database(TEST_DATABASE_URL, schema=self.schema)
        self.assertEqual(self.database.migrate(), [
            "001_multiuser.sql", "002_task_workers.sql",
            "003_shared_login_limits.sql", "004_process_resources.sql",
        ])

        self.temp = tempfile.TemporaryDirectory(prefix="rag-tenant-documents-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.auth = PostgresAuthStore(self.database, session_ttl_seconds=3600,
                                      password_iterations=1000)
        self.auth.create_user("alice", "test-password-a")
        self.auth.create_user("bob", "test-password-b")
        self.alice = self.auth.verify_identity("alice", "test-password-a")
        self.bob = self.auth.verify_identity("bob", "test-password-b")
        self.alice_cookie, _ = self.auth.create_session(self.alice)
        self.bob_cookie, _ = self.auth.create_session(self.bob)

        resources = StandardsResources(self.root, corpus=StandardCorpus([], []))
        self.app = create_app(
            root=self.root,
            resources=resources,
            auth_enabled=True,
            auth_store=self.auth,
            auth_cookie_secure=False,
            database=self.database,
            task_backend="memory",
            execution_modes={name: "simulation" for name in ("chat", "standards", "plan")},
            task_runner=lambda request, progress, cancel: {"text": "synthetic"},
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def _drop_schema(self):
        with self.psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                self.sql.Identifier(self.schema)))

    @staticmethod
    def _headers(cookie, *, unsafe=False):
        headers = {"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"}
        if unsafe:
            headers["Origin"] = "http://testserver"
        return headers

    def _workspace(self, user_id):
        return self.app.state.tenants.get(user_id).workspace_id

    def _assert_private_no_store(self, response):
        directives = {part.strip() for part in response.headers["cache-control"].split(",")}
        self.assertIn("private", directives)
        self.assertIn("no-store", directives)

    def test_persisted_metadata_private_paths_and_page_routes_are_tenant_scoped(self):
        with patch("rag_arch.web.documents.PageSources", FakePageSources):
            alice_ws = self._workspace(self.alice.user_id)
            bob_ws = self._workspace(self.bob.user_id)
            alice_headers = self._headers(self.alice_cookie)
            bob_headers = self._headers(self.bob_cookie)

            alice_upload = self.client.post(
                "/api/uploads", params={"workspace_id": alice_ws, "filename": "same.pdf"},
                content=b"fake-pdf-alice", headers=self._headers(self.alice_cookie, unsafe=True))
            bob_upload = self.client.post(
                "/api/uploads", params={"workspace_id": bob_ws, "filename": "same.pdf"},
                content=b"fake-pdf-bob", headers=self._headers(self.bob_cookie, unsafe=True))
            self.assertEqual(alice_upload.status_code, 201, alice_upload.text)
            self.assertEqual(bob_upload.status_code, 201, bob_upload.text)
            alice_asset = alice_upload.json()["asset_id"]
            bob_asset = bob_upload.json()["asset_id"]
            self.assertNotEqual(alice_asset, bob_asset)
            self.assertNotIn("path", alice_upload.json())

            alice_store = DocumentStore(self.root, alice_ws, user_id=self.alice.user_id,
                                        database=self.database)
            bob_store = DocumentStore(self.root, bob_ws, user_id=self.bob.user_id,
                                      database=self.database)
            self.addCleanup(alice_store.close)
            self.addCleanup(bob_store.close)
            alice_record = alice_store.get(alice_asset)
            bob_record = bob_store.get(bob_asset)
            expected_root = self.root / "data/runtime/web_uploads"
            self.assertTrue(alice_record.path.is_relative_to(
                expected_root / str(self.alice.user_id) / alice_ws))
            self.assertTrue(bob_record.path.is_relative_to(
                expected_root / str(self.bob.user_id) / bob_ws))
            self.assertEqual(alice_record.path.read_bytes(), b"fake-pdf-alice")
            self.assertEqual(bob_record.path.read_bytes(), b"fake-pdf-bob")

            # A reconstructed tenant store sees durable metadata; the other
            # owner cannot enumerate or resolve the same attachment.
            reopened_alice = DocumentStore(self.root, alice_ws, user_id=self.alice.user_id,
                                           database=self.database)
            self.addCleanup(reopened_alice.close)
            self.assertEqual(reopened_alice.get(alice_asset).path.read_bytes(), b"fake-pdf-alice")
            self.assertEqual([row["asset_id"] for row in reopened_alice.list()], [alice_asset])
            self.assertEqual([row["asset_id"] for row in bob_store.list()], [bob_asset])
            with self.assertRaises(KeyError):
                bob_store.get(alice_asset)
            with self.assertRaises(KeyError):
                bob_store.pages_for_asset(alice_asset)

            alice_listing = self.client.get("/api/uploads", params={"workspace_id": alice_ws},
                                            headers=alice_headers)
            bob_listing = self.client.get("/api/uploads", params={"workspace_id": bob_ws},
                                          headers=bob_headers)
            self.assertEqual([row["asset_id"] for row in alice_listing.json()["uploads"]], [alice_asset])
            self.assertEqual([row["asset_id"] for row in bob_listing.json()["uploads"]], [bob_asset])

            own_original = self.client.get(
                f"/api/uploads/{alice_asset}/download", params={"workspace_id": alice_ws},
                headers=alice_headers)
            cross_original = self.client.get(
                f"/api/uploads/{alice_asset}/download", params={"workspace_id": bob_ws},
                headers=bob_headers)
            wrong_workspace = self.client.get(
                f"/api/uploads/{bob_asset}/download", params={"workspace_id": alice_ws},
                headers=bob_headers)
            self.assertEqual(own_original.content, b"fake-pdf-alice")
            self._assert_private_no_store(own_original)
            self.assertEqual(cross_original.status_code, 404)
            self.assertEqual(wrong_workspace.status_code, 404)

            own_pages = self.client.get(
                "/api/documents/pages", params={"asset": alice_asset, "workspace_id": alice_ws},
                headers=alice_headers)
            cross_pages = self.client.get(
                "/api/documents/pages", params={"asset": alice_asset, "workspace_id": bob_ws},
                headers=bob_headers)
            self.assertEqual(own_pages.status_code, 200, own_pages.text)
            self.assertEqual(cross_pages.status_code, 404)
            self._assert_private_no_store(own_pages)

            own_image = self.client.get(own_pages.json()["pages"][0]["image_url"], headers=alice_headers)
            cross_image = self.client.get(
                "/api/documents/page-image",
                params={"source_ref": f"upload:{alice_asset}", "page": 1,
                        "asset": alice_asset, "workspace_id": bob_ws},
                headers=bob_headers)
            self.assertEqual(own_image.content, FakePageSources.image_bytes)
            self._assert_private_no_store(own_image)
            self.assertEqual(cross_image.status_code, 404)

    def test_pending_attachment_is_never_listed_as_ready(self):
        workspace_id = self.database.ensure_workspace(self.alice.user_id)
        store = DocumentStore(self.root, workspace_id, user_id=self.alice.user_id,
                              database=self.database)
        self.addCleanup(store.close)
        with patch.object(store, "_update_attachment_status", side_effect=RuntimeError("commit uncertain")):
            with self.assertRaisesRegex(RuntimeError, "commit uncertain"):
                store.add("uncertain.txt", b"private")
        self.assertEqual(store.list(), [])
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT status FROM attachments WHERE user_id = %s AND workspace_id = %s",
                (self.alice.user_id, workspace_id),
            ).fetchall()
        self.assertEqual([row["status"] for row in rows], ["processing"])

    def test_cache_limits_bound_parsed_and_page_entries(self):
        workspace_id = self.database.ensure_workspace(self.alice.user_id)
        store = DocumentStore(self.root, workspace_id, user_id=self.alice.user_id,
                              database=self.database, cache_max_entries=2,
                              cache_max_bytes=64)
        self.addCleanup(store.close)
        asset_ids = [store.add(f"{index}.txt", b"x" * 8)["asset_id"] for index in range(4)]

        class Loaded:
            pass

        with patch("rag_arch.web.documents.load_document", side_effect=lambda *args, **kwargs: Loaded()):
            for asset_id in asset_ids:
                store.load(asset_id)
        self.assertLessEqual(len(store._cache), 2)
        self.assertLessEqual(store._cache_bytes, 64)
        self.assertNotIn(("loaded", asset_ids[0]), store._cache)
        pdf_ids = [store.add(f"{index}.pdf", b"fake-pdf")["asset_id"] for index in range(4)]
        with patch("rag_arch.web.documents.PageSources", FakePageSources):
            for asset_id in pdf_ids:
                self.assertIsNotNone(store.pages_for_asset(asset_id))
        self.assertLessEqual(len(store._cache), 2)
        self.assertLessEqual(store._cache_bytes, 64)


class DocumentStoreSingleFlightRaceTests(unittest.TestCase):
    def test_stale_cache_miss_rechecks_before_reserving_second_build(self):
        with tempfile.TemporaryDirectory(prefix="rag-singleflight-race-") as temp:
            store = DocumentStore(temp, "workspace-race")
            self.addCleanup(store.close)
            initial_cache_get = store._cache_get
            first_miss_seen = Event()
            allow_first_to_continue = Event()
            second_finished = Event()
            calls = 0
            calls_lock = Lock()
            results = []
            errors = []

            def controlled_cache_get(namespace, key):
                result = initial_cache_get(namespace, key)
                if current_thread().name == "stale-miss-reader" and result is None:
                    first_miss_seen.set()
                    if not allow_first_to_continue.wait(5):
                        raise TimeoutError("first cache-miss reader was not released")
                return result

            def build():
                nonlocal calls
                with calls_lock:
                    calls += 1
                return object()

            def invoke():
                try:
                    results.append(store._memoized("race", "same-key", build, lambda _: 1))
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(store, "_cache_get", side_effect=controlled_cache_get):
                first = Thread(target=invoke, name="stale-miss-reader")
                first.start()
                self.assertTrue(first_miss_seen.wait(5), "first thread did not reach the controlled cache miss")

                second = Thread(target=lambda: (invoke(), second_finished.set()), name="first-builder")
                second.start()
                self.assertTrue(second_finished.wait(5), "second thread did not finish the first build")
                allow_first_to_continue.set()
                first.join(5)
                second.join(5)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(calls, 1)
            self.assertEqual(len(results), 2)
            self.assertIs(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
