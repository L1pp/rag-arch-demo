"""Explicit operator command for the persistent task execution capacity.

Does not load dotenv, apply migrations, or start API/worker processes. Running
without --limit only reads the current database setting. Application startup
must never reset this shared value.
"""
import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="maximum simultaneous executions across all workers")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    url = os.environ.get("RAG_ARCH_DATABASE_URL")
    if not url:
        parser.error("RAG_ARCH_DATABASE_URL is required")
    try:
        from rag_arch.web.db import Database
        from rag_arch.web.task_repository import TaskRepository
        database = Database(url, schema=os.environ.get("RAG_ARCH_DATABASE_SCHEMA", "public"))
        repository = TaskRepository(database)
        if args.limit is not None:
            limit = repository.set_global_limit(args.limit)
        else:
            with database.transaction() as connection:
                row = connection.execute(
                    "SELECT global_execution_limit FROM scheduler_state WHERE singleton_key='global'"
                ).fetchone()
                if row is None:
                    raise RuntimeError("scheduler configuration missing")
                limit = row["global_execution_limit"]
        print(f"global_execution_limit={limit}")
        return 0
    except Exception as exc:
        print(f"Task capacity configuration failed ({type(exc).__name__}); connection details omitted.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
