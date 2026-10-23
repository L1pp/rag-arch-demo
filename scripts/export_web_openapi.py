"""Export the backend OpenAPI schema without starting a server or its lifespan."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import gettempdir


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def _isolated_app_environment():
    """Prevent module-level ASGI app construction from reading live deployment config."""
    safe_values = {
        "RAG_ARCH_DATABASE_URL": None,
        "RAG_ARCH_DATABASE_SCHEMA": "public",
        "RAG_ARCH_AUTH_DB": str(Path(gettempdir()) / "rag-arch-openapi-no-db.sqlite3"),
        "RAG_ARCH_AUTH_SESSION_HOURS": "336",
        "RAG_ARCH_AUTH_COOKIE_SECURE": "false",
        "RAG_ARCH_AUTH_COOKIE_SAMESITE": "lax",
        "RAG_ARCH_AUTH_ALLOWED_ORIGINS": "",
        "RAG_ARCH_FRONTEND_ORIGINS": "",
        "RAG_ARCH_SERVE_FRONTEND": "false",
    }
    previous = {key: value for key, value in os.environ.items() if key.startswith("RAG_")}
    try:
        for key in tuple(os.environ):
            if key.startswith("RAG_"):
                os.environ.pop(key, None)
        for key, value in safe_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith("RAG_"):
                os.environ.pop(key, None)
        os.environ.update(previous)


def export_schema() -> dict:
    # Do not load dotenv, call app.router.startup(), or resolve Settings here.
    # Importing app.py creates its ASGI global at module scope, so temporarily
    # isolate the relevant process environment before importing or building the
    # schema app. AuthStore is lazy and its isolated placeholder path is never opened.
    with _isolated_app_environment():
        from rag_arch.web.app import create_app

        app = create_app(
            auth_enabled=False,
            serve_frontend=False,
            frontend_origins=[],
            auth_cookie_samesite="lax",
        )
        return app.openapi()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON here; defaults to stdout")
    args = parser.parse_args(argv)

    schema = export_schema()
    rendered = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
