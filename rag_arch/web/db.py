"""Explicit PostgreSQL access and schema migration helpers for the web app.

Importing this module has no database side effects.  The authentication SQLite
store remains a separate legacy component; this module only manages the
PostgreSQL schema used by the multi-user web runtime.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any, Iterator
from uuid import uuid4


_MIGRATION_LOCK_NAMESPACE = 0x52414741  # "RAGA", reserved for schema migrations.


class Database:
    """Small psycopg3 connection manager for one PostgreSQL schema.

    ``connect()`` returns a raw ``psycopg.Connection`` configured with
    ``dict_row`` and the selected schema as its search path.  The caller owns
    and must close that connection.  ``transaction()`` owns a short-lived
    connection, yields that same connection, commits on success, rolls back
    on error, and closes it in either case.

    Construction only stores configuration.  It neither reads environment
    variables nor opens a connection.  Call ``migrate()`` explicitly to
    create the schema and apply bundled SQL migrations.
    """

    def __init__(self, url: str, schema: str = "public", *, connect_timeout: int = 5,
                 statement_timeout_ms: int = 15000) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty PostgreSQL connection string")
        if not isinstance(schema, str) or not schema:
            raise ValueError("schema must be a non-empty string")
        if len(schema.encode("utf-8")) > 63:
            raise ValueError("schema name must be at most 63 bytes in UTF-8")
        if type(connect_timeout) is not int or connect_timeout < 1:
            raise ValueError("connect_timeout must be a positive integer")
        if type(statement_timeout_ms) is not int or statement_timeout_ms < 0:
            raise ValueError("statement_timeout_ms must be a nonnegative integer")
        self.url = url
        self.schema = schema
        self.connect_timeout = connect_timeout
        self.statement_timeout_ms = statement_timeout_ms

    @staticmethod
    def _psycopg() -> tuple[Any, Any, Any]:
        try:
            import psycopg
            from psycopg import sql
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise RuntimeError(
                "PostgreSQL support requires psycopg 3; install the web runtime dependencies."
            ) from exc
        return psycopg, sql, dict_row

    def _set_search_path(self, connection: Any, sql: Any) -> None:
        # Schema names are identifiers, never interpolated as SQL text or bound
        # as values.  Quoting also permits deployment-defined names with dashes.
        connection.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema))
        )
        connection.execute("SELECT set_config('statement_timeout', %s, false)",
                           (str(self.statement_timeout_ms),))

    def connect(self) -> Any:
        """Open a dict-row psycopg connection scoped to ``self.schema``.

        The returned connection is not wrapped in a context manager.  Callers
        that need transaction ownership should prefer ``transaction()``.
        """
        psycopg, sql, dict_row = self._psycopg()
        connection = psycopg.connect(self.url, row_factory=dict_row, connect_timeout=self.connect_timeout)
        try:
            self._set_search_path(connection, sql)
            # Persist the session-level search_path before handing out the raw
            # connection.  Otherwise a caller's rollback could undo this SET.
            connection.commit()
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Yield one short-lived connection and commit or roll back atomically."""
        connection = self.connect()
        try:
            # psycopg's connection context commits on success, rolls back on
            # exception, and closes the connection on context exit.
            with connection:
                yield connection
        finally:
            if not connection.closed:
                connection.close()

    def ensure_workspace(self, user_id: int) -> str:
        """Return the user's stable workspace ID and ensure its three slots.

        Concurrent calls serialize on the unique ``workspaces.user_id``
        constraint.  The winner creates the workspace; all callers then read
        that same ID and idempotently insert the chat, standards, and plan
        slots in one short transaction.  The user row must already exist.
        """
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValueError("user_id must be a positive integer")

        now = time.time()
        candidate_workspace_id = uuid4().hex
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO workspaces(workspace_id, user_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (candidate_workspace_id, user_id, now, now),
            )
            workspace = connection.execute(
                "SELECT workspace_id FROM workspaces WHERE user_id = %s",
                (user_id,),
            ).fetchone()
            if workspace is None:
                # Normally the FK on workspaces makes an unknown user fail at
                # insert.  Keep this guard in case the schema changes later.
                raise LookupError(f"user {user_id} has no workspace")

            connection.execute(
                """
                INSERT INTO assistant_slots(user_id, assistant_id, updated_at)
                VALUES (%s, 'chat', %s), (%s, 'standards', %s), (%s, 'plan', %s)
                ON CONFLICT (user_id, assistant_id) DO NOTHING
                """,
                (user_id, now, user_id, now, user_id, now),
            )
            return workspace["workspace_id"]

    def migrate(self) -> list[str]:
        """Create ``schema`` and apply unapplied SQL migrations in filename order.

        A transaction-scoped PostgreSQL advisory lock serializes migration
        startup across processes.  The lock is database-wide by design because
        migrations are short and deployments should not race schema changes.
        """
        migrations_dir = Path(__file__).with_name("migrations")
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))
        if not migration_files:
            raise RuntimeError(f"no SQL migrations found in {migrations_dir}")

        psycopg, sql, dict_row = self._psycopg()
        applied_now: list[str] = []

        # Acquire the database-wide lock before CREATE SCHEMA so concurrent
        # first-start migrations cannot race on the same schema.  CREATE
        # SCHEMA receives a quoted identifier; no database or schema is dropped.
        connection = psycopg.connect(self.url, row_factory=dict_row, connect_timeout=self.connect_timeout)
        try:
            with connection:
                connection.execute(
                    "SELECT pg_advisory_xact_lock(%s, %s)",
                    (_MIGRATION_LOCK_NAMESPACE, 0),
                )
                connection.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    )
                )
                self._set_search_path(connection, sql)
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at DOUBLE PRECISION NOT NULL
                    )
                    """
                )
                for migration_path in migration_files:
                    version = migration_path.name
                    existing = connection.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = %s",
                        (version,),
                    ).fetchone()
                    if existing is not None:
                        continue
                    migration_sql = migration_path.read_text(encoding="utf-8")
                    connection.execute(migration_sql)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) "
                        "VALUES (%s, EXTRACT(EPOCH FROM clock_timestamp()))",
                        (version,),
                    )
                    applied_now.append(version)
        finally:
            if not connection.closed:
                connection.close()
        return applied_now
