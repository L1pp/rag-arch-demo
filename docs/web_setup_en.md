# Run the React + FastAPI Workbench Locally

[English](web_setup_en.md) | [简体中文](web_setup.md) | [Architecture](architecture_en.md)

This guide starts the React/FastAPI workbench included in the repository. The public tree does not contain a complete standards corpus, real source documents, API keys, or runtime database. These are local setup steps after you provide suitable resources; cloning the repository alone does not provide working real-document Q&A or plan review.

## Requirements

- Python 3.12.
- Node.js 22.12 or newer. The lockfile also supports Node 20.19 or newer within the 20.x line.
- A local browser.
- Your own model configuration and standards materials. Vector retrieval and reranking also require the corresponding services and usable vector data.

## Install dependencies and build the frontend

Run from the repository root:

~~~~powershell
python -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
Push-Location frontend
npm ci
npm run build
Pop-Location
~~~~

On macOS/Linux, activate the environment with source .venv-web/bin/activate. FastAPI serves the React page and assets from frontend/dist/ after the build.

## Configure model services

Copy the template and fill in your own service URLs, model names, and API keys:

~~~~powershell
Copy-Item .env.example .env
~~~~

Edit .env and configure at least the chat service you intend to use. The chat model must support tool calling. Configure Embedding and Reranker services as well if you want vector retrieval and reranking. Service URLs, capabilities, and parameter limits depend on your provider. This repository includes no working API credentials and does not claim that online services have been validated.

Do not commit a real .env, API keys, or service logs.

## Prepare your own standards materials

Standards Q&A requires at least these project-format files:

- data/corpus/clauses.jsonl
- data/corpus/tables.jsonl

Inspect rag_arch/standards.py and rag_arch/evidence_index.py for field formats and loading rules. Add catalog summaries, catalog entries, version relations, and status records as needed. To navigate from evidence to an original page, provide the corresponding originals and source mappings expected by the code; records without originals cannot open a source page. See rag_arch/web/standards_resources.py, rag_arch/page_sources.py, and rag_arch/standard_status.py for loading and location handling.

If you use the preparation/history page, also provide data/meta/history/active_preparation.json in the expected format. Plan documents can be uploaded through the page. The authentication database, attachments, history, indexes, and caches are maintained in local data directories. Do not commit runtime files containing private materials.

DOCX text extraction does not require Word. Original DOCX page previews use `scripts/render_docx_pages.ps1` on Windows with Microsoft Word installed; that optional path was not exercised in this validation.

## Create a local account and start the app

Create an account for the local instance. The command prompts for the password twice and does not put it in shell history:

~~~~powershell
python -m rag_arch.web.auth_cli add YOUR_USERNAME
~~~~

For local HTTP, disable Secure Cookie in the same process environment before starting the backend. This setting is read from the process environment, not from the model-settings .env file:

~~~~powershell
$env:RAG_ARCH_AUTH_COOKIE_SECURE = 'false'
python -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port 8000 --workers 1
~~~~

Open http://127.0.0.1:8000 and sign in. On Windows, you can also run .\start_web.ps1 -Port 8000 after dependencies, the frontend build, and environment variables are ready.

Authentication controls local login only. Accounts share the same workspace and runtime data; the current code does not isolate data between accounts. The example binds to loopback for local use.

## Retained sample entry points

To inspect the authored offline CLI path without a real corpus:

~~~~powershell
python -m demo --output output
~~~~

The separate no-key Streamlit sample requires the additional requirements-demo.txt; its command is in the [README](../README.md#separate-sample-paths). Both samples use deterministic stand-ins and cover only their authored inputs.
