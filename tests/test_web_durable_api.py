"""HTTP integration for DB-owned tasks; no worker or model starts in the API."""
import asyncio
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_arch.web.app import create_app, event_stream
from rag_arch.web.auth import PostgresAuthStore
from rag_arch.web.db import Database
from rag_arch.web.task_repository import TaskRepository
from scripts.configure_task_limits import main as limits_main


class DurableAPIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ.get('RAG_ARCH_TEST_DATABASE_URL')
        if not self.url:
            self.skipTest('dedicated RAG_ARCH_TEST_DATABASE_URL required')
        self.temp = TemporaryDirectory(prefix='rag-durable-api-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = Database(self.url, schema='api_task_' + uuid4().hex)
        self.addCleanup(self.drop_schema)
        self.database.migrate()
        self.auth = PostgresAuthStore(self.database, password_iterations=1500)
        self.auth.create_user('Alice', 'test-password-1')
        self.repository = TaskRepository(self.database)

    def drop_schema(self):
        import psycopg
        from psycopg import sql
        with psycopg.connect(self.url) as connection:
            connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(
                sql.Identifier(self.database.schema)))

    def client(self):
        app = create_app(root=self.root, database=self.database, auth_store=self.auth,
                         auth_cookie_secure=False,
                         execution_modes={key: 'simulation' for key in ('chat', 'standards', 'plan')})
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        response = client.post('/login', data={'username': 'Alice', 'password': 'test-password-1'},
                               headers={'Origin': 'http://testserver'}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        return app, client, client.get('/api/workspace').json()

    def submit(self, client, workspace, **extra):
        return client.post('/api/tasks', json={'workspace_id': workspace['workspace_id'],
            'assistant_id': 'chat', 'request_id': 'one-request', 'prompt': 'synthetic input',
            'duration_seconds': 1, **extra}, headers={'Origin': 'http://testserver'})

    def test_two_api_instances_share_idempotency_and_api_close_does_not_cancel(self):
        app_a, a, workspace = self.client()
        _, b, second = self.client()
        self.assertEqual(workspace['workspace_id'], second['workspace_id'])
        self.assertEqual(workspace['server_instance_id'], second['server_instance_id'])
        first = self.submit(a, workspace)
        self.assertEqual(first.status_code, 202, first.text)
        task = first.json()
        duplicate = self.submit(b, second)
        self.assertEqual(duplicate.status_code, 202, duplicate.text)
        self.assertEqual(duplicate.json()['task_id'], task['task_id'])
        self.assertEqual(self.submit(b, second, prompt='changed').status_code, 409)
        self.assertEqual(self.submit(b, second, request_id='second-request').status_code, 409)
        app_a.state.tenants.close()
        response = b.get(f"/api/tasks/{task['task_id']}", params={'workspace_id': second['workspace_id']})
        self.assertEqual(response.json()['status'], 'queued')
        self.assertEqual(b.get('/api/health').json()['task_storage'], 'postgresql')

    def test_queue_cancel_and_history_reset_are_durable(self):
        _, a, workspace = self.client()
        task = self.submit(a, workspace).json()
        _, b, second = self.client()
        cancelled = b.post(f"/api/tasks/{task['task_id']}/cancel", json={
            'workspace_id': second['workspace_id']}, headers={'Origin': 'http://testserver'})
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()['status'], 'cancelled')
        reset = b.post('/api/assistants/chat/reset', json={'workspace_id': second['workspace_id']},
                       headers={'Origin': 'http://testserver'})
        self.assertEqual(reset.status_code, 200)
        _, c, third = self.client()
        self.assertEqual(third['assistants'][0]['tasks'], [])
        self.assertEqual(c.get(f"/api/tasks/{task['task_id']}/download", params={
            'workspace_id': third['workspace_id'], 'format': 'json'}).status_code, 200)

    def test_operator_limit_is_explicit_and_api_startup_does_not_overwrite_it(self):
        env = {'RAG_ARCH_DATABASE_URL': self.url, 'RAG_ARCH_DATABASE_SCHEMA': self.database.schema}
        with patch.dict(os.environ, env), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(limits_main(['--limit', '2']), 0)
        self.assertIn('global_execution_limit=2', output.getvalue())
        self.client()
        self.client()
        with patch.dict(os.environ, env), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(limits_main([]), 0)
        self.assertEqual(output.getvalue().strip(), 'global_execution_limit=2')

    def test_database_sse_reads_are_offloaded_from_event_loop(self):
        app, client, workspace = self.client()
        task = self.submit(client, workspace).json()
        service = app.state.tenants.get(workspace['owner_user_id']).tasks
        threads = []
        original = service.repository.events_after

        def observed(*args, **kwargs):
            threads.append(threading.get_ident())
            return original(*args, **kwargs)

        service.repository.events_after = observed

        async def check():
            loop_thread = threading.get_ident()
            stream = event_stream(service, 0)
            try:
                event = await stream.__anext__()
                self.assertEqual(event.data['task_id'], task['task_id'])
                self.assertTrue(threads)
                self.assertNotIn(loop_thread, threads)
            finally:
                await stream.aclose()
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
