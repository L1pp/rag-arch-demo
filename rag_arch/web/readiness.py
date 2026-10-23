"""Public, read-only readiness checks for the durable multi-user service."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

REQUIRED_MIGRATIONS = frozenset({
    "001_multiuser.sql",
    "002_task_workers.sql",
    "003_shared_login_limits.sql",
    "004_process_resources.sql",
})

REQUIRED_TABLES = frozenset({
    "schema_migrations",
    "users",
    "sessions",
    "workspaces",
    "assistant_slots",
    "tasks",
    "attachments",
    "task_attachments",
    "workspace_events",
    "scheduler_state",
    "worker_instances",
    "login_rate_limit_failures",
    "login_rate_limit_attempts",
    "process_resources",
    "process_resource_uses",
    "process_resource_identity_audit",
    "task_resolution_audit",
})


class ReadinessSnapshot(BaseModel):
    """The only fields disclosed by the unauthenticated readiness endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    database: bool
    schema_ready: bool = Field(alias="schema")
    worker: bool
    ready: bool


def check_readiness(database, *, auth_enabled: bool, task_backend: str) -> dict[str, bool]:
    """Check durable API prerequisites using a read-only database transaction.

    This does not load model configuration, run migrations, or inspect tenant,
    task, worker identity, or connection details for its public response.
    """
    state = {"database": False, "schema": False, "worker": False, "ready": False}
    if database is not None:
        connection = None
        try:
            connection = database.connect()
            connection.execute("SET TRANSACTION READ ONLY")
            connection.execute("SELECT 1").fetchone()
            state["database"] = True

            try:
                table_rows = connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
                ).fetchall()
                table_names = {str(row["table_name"]) for row in table_rows}
                migrations = set()
                if "schema_migrations" in table_names:
                    migration_rows = connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                    migrations = {str(row["version"]) for row in migration_rows}
                state["schema"] = (
                    REQUIRED_TABLES.issubset(table_names)
                    and REQUIRED_MIGRATIONS.issubset(migrations)
                )
                if state["schema"]:
                    row = connection.execute(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM worker_instances "
                        "WHERE state = 'active' "
                        "AND lease_seconds > 0 "
                        "AND lease_seconds < 'Infinity'::double precision "
                        "AND heartbeat_at > '-Infinity'::double precision "
                        "AND heartbeat_at < 'Infinity'::double precision "
                        "AND heartbeat_at + lease_seconds > EXTRACT(EPOCH FROM clock_timestamp())"
                        ") AS worker_ready"
                    ).fetchone()
                    state["worker"] = bool(row and row["worker_ready"])
            except Exception:
                # A reachable database with an incomplete/inaccessible schema
                # is not ready. Never expose database exception text publicly.
                state["schema"] = False
                state["worker"] = False
        except Exception:
            # Connection and permission errors are intentionally indistinguishable
            # to anonymous callers. No migration or retry side effect is made.
            pass
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    state["ready"] = bool(
        auth_enabled is True
        and task_backend == "durable"
        and database is not None
        and state["database"]
        and state["schema"]
        and state["worker"]
    )
    return state
