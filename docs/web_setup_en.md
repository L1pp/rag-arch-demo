# Run the React + FastAPI Workbench Locally

[English](web_setup_en.md) | [简体中文](web_setup.md) | [Architecture](architecture_en.md)

The public workbench runs a React frontend, FastAPI API, PostgreSQL database, and separate AI worker processes. It is a source showcase, not a packaged hosted service. You provide PostgreSQL, model credentials, a corpus, and source documents for any workflow that needs to open original pages.

## Requirements

- Python 3.12.
- Node.js 22.12 or newer. The frontend lockfile also supports Node 20.19 or newer in the 20.x line.
- A PostgreSQL instance accessible from the API, worker, migration, and account-management processes.
- Model settings and corpus files supplied by the operator.
- A local browser.

The application scripts do not create a PostgreSQL server or import a corpus. Keep database credentials and model keys out of Git and logs.

## Install Python and frontend dependencies

From the repository root:

~~~~powershell
python -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
python -m pip install --no-deps -e .
Push-Location frontend
npm ci
Pop-Location
~~~~

For Linux or macOS, activate .venv-web with source .venv-web/bin/activate. The Python package, API, and worker use the same environment.

## Configure PostgreSQL and apply migrations

Set the database URL and schema in the process environment used by migration, account-management, API, and worker commands. These RAG_ARCH_DATABASE_* variables are not loaded from the model-settings .env file.

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<application-user>:<password>@<host>:5432/<database>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
python -m scripts.migrate_web schema
~~~~

Provision the database and user before running the explicit migration. Starting the API does not create tables or migrate old data. Apply all pending migrations before starting API or worker processes. Do not put a real connection string in a committed file.

For a local development instance, add an account after migration:

~~~~powershell
python -m rag_arch.web.auth_cli add YOUR_USERNAME
~~~~

The command prompts for the password twice. Set the same database environment variables in each new terminal that will run the account command, API, or worker.

## Provide model settings and corpus

Copy .env.example to .env and fill the service URLs, model names, and API keys you intend to use. The chat model must support the tools used by the selected workflow. Embedding and reranking services are needed for those features. Settings and provider capabilities vary; this repository does not include working credentials or claim that a live provider has been validated.

Standards Q&A expects the project-format files data/corpus/clauses.jsonl and data/corpus/tables.jsonl. Add catalog, status, version, and source mappings as needed. Original files are required only for workflows that open source pages. Plans can be uploaded through the UI. Keep private materials and runtime files out of Git.

The code supports read-only mmap packages for the shared corpus, BM25 index, and dense vectors. For a large corpus, build and deploy matching packages from your own data:

~~~~powershell
python -m scripts.build_core_corpus_mmap
python -m scripts.build_core_bm25
~~~~

The vector conversion command converts an existing compatible embedding cache; it does not create embeddings or call a model:

~~~~powershell
python -m scripts.convert_core_vectors
~~~~

Use the same source corpus and matching package versions for the API and every worker. If the source corpus changes, rebuild the derived packages and restart workers to load them. The full internal data and its packages are not included in this repository.

## Start the API, worker, and React frontend

Open three terminals from the repository root. Set the database URL and schema in each terminal. For local HTTP development, configure the API origin and cookie settings, then start the API:

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<application-user>:<password>@<host>:5432/<database>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
$env:RAG_ARCH_AUTH_COOKIE_SECURE = 'false'
$env:RAG_ARCH_FRONTEND_ORIGINS = 'http://127.0.0.1:5173'
.\start_web.ps1
~~~~

In the second terminal, set the same database variables and run the worker:

~~~~powershell
.\start_worker.ps1
~~~~

In the third terminal, start the frontend dev server:

~~~~powershell
Push-Location frontend
npm run dev
~~~~

Open http://127.0.0.1:5173 and sign in. The API listens on 127.0.0.1:8000. The Vite dev server forwards API requests to it. The database stores task state and events; API and worker processes must use the same schema. For production, build the frontend and put it behind an HTTPS gateway with the API; review cookie, origin, database-role, worker, and data-directory settings before deployment.

## Offline CLI sample

The authored CLI can run the small offline example without the web database, live API credentials, or private corpus:

~~~~powershell
python -m demo --output output
~~~~

It writes Markdown and JSON results under output/. Deterministic stand-ins cover only the authored fictional examples; this is not a general plan-review service.
