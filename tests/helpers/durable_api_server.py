"""Disposable real-process API entry point for durable task integration tests.

This helper deliberately accepts only the dedicated loopback test database. It
does not run migrations, read the project's .env file, or create a worker.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def _test_database_configuration() -> tuple[str, str]:
    url = os.environ.get("RAG_ARCH_DATABASE_URL", "").strip()
    test_url = os.environ.get("RAG_ARCH_TEST_DATABASE_URL", "").strip()
    schema = os.environ.get("RAG_ARCH_DATABASE_SCHEMA", "").strip()
    if not url or url != test_url:
        raise RuntimeError("API helper 仅接受显式相同的专用测试数据库连接串。")
    parsed = urlsplit(url)
    database_name = unquote(parsed.path.lstrip("/"))
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("测试数据库必须使用 PostgreSQL URL。")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("进程测试数据库必须只连接本机。")
    if database_name != "rag_arch_test":
        raise RuntimeError("进程测试数据库名称必须是 rag_arch_test。")
    if not re.fullmatch(r"process_[a-f0-9]{32}", schema):
        raise RuntimeError("进程测试必须使用随机隔离的 process_<uuid> schema。")
    return url, schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args(argv)

    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    url, schema = _test_database_configuration()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Imports happen only after the test DB guard has passed.  No migration is
    # run here: the parent test process owns schema creation and cleanup.
    from rag_arch.web.app import create_app
    from rag_arch.web.auth import PostgresAuthStore
    from rag_arch.web.db import Database
    import uvicorn

    database = Database(url, schema=schema, connect_timeout=3,
                        statement_timeout_ms=8000)
    auth_store = PostgresAuthStore(database, password_iterations=1500)
    execution_modes = {name: "simulation" for name in ("chat", "standards", "plan")}
    app = create_app(
        database=database,
        auth_store=auth_store,
        root=root,
        auth_cookie_secure=False,
        serve_frontend=os.environ.get("RAG_ARCH_TEST_API_ONLY") != "1",
        execution_modes=execution_modes,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port,
                access_log=False, log_level="warning", lifespan="on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
