# Architecture and Code Reading Guide

[English](architecture.md) | [简体中文](zh-CN/architecture.md) | [Web setup](setup.md)

The current public implementation separates the React frontend, FastAPI API, PostgreSQL state, and AI workers. Users operate one deployment with account-scoped data. The repository does not include the internal corpus, original documents, database, model credentials, or generated index packages.

~~~~mermaid
flowchart LR
    U[Browser / React] --> G[HTTPS gateway or Vite dev server]
    G --> A[FastAPI API processes]
    A --> D[(PostgreSQL users, sessions, tasks, events, limits, audit)]
    D --> W[Independent AI workers]
    W --> M[Async model, search, embedding, reranking calls]
    W --> L[Bounded local processing and supervised child processes]
    U --> E[Resumable task event stream]
    A --> E
    W --> E
    C[User-provided corpus and source files] --> W
~~~~

## Request and task lifecycle

FastAPI handles authentication, account-scoped API requests, task submission, upload and download boundaries, and event streaming. It does not execute the durable AI task in the request process. PostgreSQL stores task state and ordered events; one or more workers claim queued work and persist progress and terminal state. A client reconnects with its event cursor to resume delivery. Per-account assistant slots and the deployment-wide execution cap are enforced through shared database state, so adding API or worker processes does not bypass those limits.

Each account's tasks, attachments, history, and audit records are scoped to that account. The deployment operator still administers the database and host; this code does not claim to provide a hosted multi-tenant service or a managed identity platform.

The worker keeps the application workflows and retrieval code in the same project. Network calls to chat, web search, embeddings, and reranking are orchestrated asynchronously. Synchronous CPU, file, and document-conversion work stays behind bounded local execution resources or supervised child processes. The offline CLI remains a separate synchronous sample.

The recovery helper can isolate an interrupted attempt only after confirming that the local worker and its registered resources have exited and the cooldown has elapsed. It does not automatically retry the task or assume that an external request has stopped. Maintenance must be invoked explicitly.

## Shared retrieval data

The retrieval path combines BM25 and dense-vector candidates with RRF, then applies configured reranking and evidence expansion. Public corpus records, BM25 statistics, and dense vectors support immutable read-only memory-mapped packages. Processes can share operating-system file pages instead of each constructing a separate full Python copy; full corpus records and location details are decoded on demand. This keeps the existing retrieval behavior visible while changing how shared data is loaded.

The mmap code is included, but the corpus and packages are not. Users prepare data in the expected format and build packages from their own corpus and compatible embedding cache. Loaders retain format, dimension, offset, mapping, and boundary validation; standalone legacy integrity checks were removed. Opening a corpus mmap package does not check whether the original source directory still exists or matches the package; a structurally valid stale package can still load. Operators must manage matching versions and rebuild the corpus, BM25, and vector packages after source updates before restarting workers. The web setup guide describes the public commands and data boundaries.

## Page evidence and optional OCR

Plan review can inspect a whole source page or a selected region. Initial page selection and image delivery are bounded; the visual ledger records acquisition, request, response, and citation states without storing image bytes in the ledger. An image sent to a model is evidence of delivery, not proof that it was read correctly.

Cloud OCR is disabled by default. When explicitly enabled, a claimed worker task may supplement candidate pages with OCR text, retaining page locations and an unverified status. The current adapter uses a specified Aliyun VL-OCR service; see [setup](setup.md) for its configuration and limits. Linux uses supervised LibreOffice conversion for document pages; Windows users convert legacy DOC files to DOCX before upload.

## Source map

| Layer | Main files | What to inspect |
|---|---|---|
| React frontend | frontend/src/ | Login boundary, workspace state, task requests, event reconnection, results and downloads |
| API and authentication | rag_arch/web/app.py, auth.py, contracts.py | API-only startup, sessions, authorization, request validation, health/readiness |
| PostgreSQL state | rag_arch/web/db.py, task_repository.py, persistent_tasks.py | Explicit schema migration, ownership-scoped queries, durable task transitions |
| Events and workers | rag_arch/web/events.py, task_worker.py, worker_process.py | Ordered events, cursor resume, leasing, execution and recovery |
| Attachments and resources | rag_arch/web/documents.py, document_routes.py, standards_resources.py | Per-account attachment ownership and shared read-only corpus resources |
| Page review and OCR | rag_arch/page_regions.py, visual_review_ledger.py, cloud_ocr.py, web/ocr_preparation.py | Page regions, image evidence state, and optional unverified OCR |
| Async provider boundary | rag_arch/async_support.py, providers.py, responses_chat.py, web_search.py | Async calls, Responses handling, provider-specific parameters, and bounded local work |
| Retrieval and evidence | rag_arch/retrieval.py, binary_bm25.py, binary_vectors.py, binary_corpus.py, evidence_index.py | Hybrid candidates, mmap packages, evidence lookup and on-demand decoding |
| Agents and reports | rag_arch/standard_tools.py, graph_agent.py, plan_review_agent.py, plan_output.py | Evidence tool flow, plan quotation checks, citation checks, and report output |
| Independent web UI | frontend/ | React development server and production build |

## Data and operating boundaries

A working deployment needs PostgreSQL, model settings supplied by the operator, a corpus in the expected format, and source documents for source-page viewing. Users can prepare data/corpus/clauses.jsonl and data/corpus/tables.jsonl plus any catalog, status, and source mappings their workflows require. API credentials and runtime files stay local.

The documentation index links screenshots captured on 2026-10-13 with a fictional offline fixture. Those images show an earlier React interface and do not validate the current worker lifecycle or a live deployment.
