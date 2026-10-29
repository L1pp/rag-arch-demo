# Run the React + FastAPI Workbench Locally

[English](setup.md) | [简体中文](zh-CN/setup.md) | [Architecture](architecture.md)

This public workbench is a selected snapshot of an evolving project. It shows the main React, FastAPI, PostgreSQL, and worker implementation, but is not a complete delivery. Users provide PostgreSQL, model credentials, a corpus and any source documents they need, and complete the required environment integrations.

## Requirements

- Python 3.12.
- Node.js 22.12 or newer. The frontend lockfile also supports Node 20.19 or newer in the 20.x line.
- A PostgreSQL instance accessible from the API, worker, migration, and account-management processes.
- Model settings and corpus files supplied by the operator.
- A local browser.
- On Linux, LibreOffice Writer for DOC conversion and Word page previews. Install `soffice` on `PATH` or set `RAG_ARCH_LIBREOFFICE_EXECUTABLE` in the process environment. On Windows, use DOCX; page previews require Microsoft Word.

The application scripts do not create a PostgreSQL server or import a corpus. Keep database credentials and model keys out of Git and logs.

## Install Python and frontend dependencies

From the repository root:

~~~~powershell
python -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements/web.txt
python -m pip install --no-deps -e .
Push-Location frontend
npm ci
Pop-Location
~~~~

On Linux, use the same environment and install the frontend dependencies from a POSIX shell:

~~~~sh
python -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r requirements/web.txt
python -m pip install --no-deps -e .
cd frontend
npm ci
cd ..
~~~~

The Python package, API, and worker use the same environment.

## Configure PostgreSQL and apply migrations

Set the database URL and schema in the process environment used by migration, account-management, API, and worker commands. These RAG_ARCH_DATABASE_* variables are not loaded from the model-settings .env file.

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<application-user>:<password>@<host>:5432/<database>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
python -m scripts.migrate_web schema
~~~~

On Linux, set the same values with `export` before the migration:

~~~~sh
export RAG_ARCH_DATABASE_URL='postgresql://<application-user>:<password>@<host>:5432/<database>'
export RAG_ARCH_DATABASE_SCHEMA='rag_arch'
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

Use the same source corpus and matching package versions for the API and every worker. Opening a corpus mmap package does not check whether its original source directory exists or still matches. A structurally valid stale package can load, so the operator must manage matching versions. After changing the source, rebuild the matching corpus, BM25, and vector packages and restart workers to load them. The full internal data and its packages are not included in this repository.

## Optional page OCR and visual review

Cloud OCR is disabled by default. To enable it, configure `RAG_ARCH_OCR_ENABLED=true`, `RAG_OCR_BASE_URL`, and `RAG_OCR_API_KEY` in the worker's environment or local `.env`. The current adapter uses the fixed `qwen-vl-ocr-2025-11-20` model and accepts Aliyun Beijing ordinary pay-as-you-go HTTPS endpoints; it does not use Token Plan. Blank OCR endpoint or key values fall back to the configured chat endpoint or key, which must also satisfy the OCR service requirements. `RAG_ARCH_OCR_TIMEOUT_SECONDS` controls the request timeout (default 120 seconds, range 1–300).

Only a claimed worker task can submit candidate pages to OCR. `RAG_ARCH_OCR_MAX_PAGES_PER_TASK` is read from the worker process environment (default 100, range 1–500). Recognition remains unverified (`recognition_verified=false`) and the UI surfaces page warnings. Enabling OCR sends page images to the configured service and may incur provider charges.

Visual plan review can deliver whole pages or selected regions to a compatible vision model with bounded image budgets. The result records whether an image was acquired, sent, returned, or cited; this does not guarantee correct visual interpretation. The `deepseek-v4.1-flash-1017` snapshot has provider-specific Responses parameter handling; configure a compatible provider and supply its credentials yourself.

OCR caches also retain failed results; changing configuration does not automatically retry an existing asset. For an explicit new attempt, first verify the previous request state, correct settings, safely finish active work and restart the relevant workers, then upload a new asset and select it for a new task. The new request may incur charges. Do not automatically delete unknown-state markers to resend requests.

## Start the API, worker, and React frontend

On Windows, open three PowerShell terminals from the repository root. Set the database URL and schema in each terminal. For local HTTP development, configure the API origin and cookie settings, then start the API:

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<application-user>:<password>@<host>:5432/<database>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
$env:RAG_ARCH_AUTH_COOKIE_SECURE = 'false'
$env:RAG_ARCH_FRONTEND_ORIGINS = 'http://127.0.0.1:5173'
.\scripts\start_web.ps1
~~~~

In the second terminal, set the same database variables and run the worker:

~~~~powershell
.\scripts\start_worker.ps1
~~~~

In the third terminal, start the frontend dev server:

~~~~powershell
Push-Location frontend
npm run dev
~~~~

On Linux, activate `.venv-web` and export the same database URL and schema in each terminal. Set the API origin and cookie variables as above using `export`, then start the API and worker in separate terminals:

~~~~sh
python -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 5
python -m rag_arch.web.worker_process run --concurrency 3
~~~~

In a third terminal run `npm run dev` from `frontend/`. Open http://127.0.0.1:5173 and sign in. The API listens on 127.0.0.1:8000; the Vite dev server forwards API requests to it. The database stores task state and events, and API and worker processes must use the same schema. For production, build the frontend and put it behind an HTTPS gateway with the API; review cookie, origin, database-role, worker, and data-directory settings before deployment.

## Recover an interrupted task

For a local worker that has exited, the operator can run the following one-time maintenance command on that worker's host with the database environment configured:

~~~~sh
python -m rag_arch.web.worker_process recover --isolate-interrupted
~~~~

`--cleanup-owned-resources` permits cleanup of some identity-verified child processes. LibreOffice process groups are only checked for confirmed exit; this command does not terminate them. If exit cannot be confirmed, their state and task slot remain reserved.

The default cooldown is 120 seconds. The operation isolates an interrupted task only after verifying that the local worker and its registered child resources have exited. It does not automatically retry the task, and the status of any external work remains unknown.

## Offline CLI sample

The authored CLI can run the small offline example without the web database, live API credentials, or private corpus:

~~~~powershell
python -m demo --output output
~~~~

It writes Markdown and JSON results under output/. Deterministic stand-ins cover only the authored fictional examples; this is not a general plan-review service.
