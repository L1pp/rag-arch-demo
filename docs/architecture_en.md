# Architecture and Code Reading Guide

[English](architecture_en.md) | [简体中文](architecture.md) | [Web setup](web_setup_en.md)

The repository now presents the React + FastAPI workbench as its primary entry. The full standards corpus, original source documents, runtime database, and model credentials are not included. The interface source can be inspected, while real document Q&A and source-page viewing require compatible resources supplied by the user.

~~~~mermaid
flowchart LR
    U[Browser / React] --> A[FastAPI pages and authentication]
    A --> B[Workspace, task, SSE, cancel, download APIs]
    B --> C[TaskService / assistant runner]
    C --> D[Standards Q&A, document chat, plan-review runners]
    D --> E[BM25 + vector retrieval]
    E --> F[RRF fusion / reranking]
    F --> G[Evidence tools and Agent]
    G --> H[Source and plan-quotation validation]
    H --> I[Markdown / JSON reports]
    J[User-provided corpus, metadata, originals] --> D
~~~~

| Stage | Main source files | What to inspect |
|---|---|---|
| React UI | frontend/src/App.tsx, api.ts, store.ts, WorkspaceContext.tsx | Workspace state, assistant tasks, event subscription, report display, and request boundaries |
| FastAPI entry | rag_arch/web/app.py, contracts.py | Pages and static assets, login, workspace/task APIs, SSE, cancellation, and downloads |
| Tasks and events | rag_arch/web/task_service.py | Single-process task lifecycle, progress events, cancellation, and current workspace snapshot |
| Upload and source pages | rag_arch/web/document_routes.py, documents.py, standards_routes.py | Uploaded file handling, source-page viewing, document catalog, and status routes |
| History | rag_arch/web/history.py | Local history pages and record-reading routes |
| Backend workflows | rag_arch/web/chat_runner.py, standards_runner.py, plan_runner.py | How requests enter chat, standards Q&A, and plan review |
| Document processing | rag_arch/loaders.py, splitting.py, plan_documents.py | TXT/MD/PDF/DOCX reading, line/page locations, scanned-page gaps, and splitting |
| Retrieval and reranking | rag_arch/retrieval.py, evidence_index.py, reranking.py, providers.py | Chinese and numeric tokenization, BM25, vector normalization, RRF candidates, and reranking services |
| Evidence and Q&A | rag_arch/standard_tools.py, graph_agent.py, page_qa.py | Search/read evidence tools, LangGraph tool loop, and final citation constraints |
| Plan review | rag_arch/plan_review_agent.py, plan_review_context.py, plan_review_evidence.py | Full-text-first review, batching, plan quotation location, evidence assembly, and coverage |
| Validation and output | rag_arch/plan_output.py, plan_review_report.py, model_input_view.py | Separating citation checks from engineering judgments, source traceability, and Markdown/JSON reports |

At startup, FastAPI reads settings and creates the task service. It serves the built React files and API from the same origin. The API includes task state, SSE progress, cancellation, and file downloads. This code is a local workbench, not a hosted or multi-tenant service. See the [Web setup guide](web_setup_en.md) for build and launch steps.

Standards Q&A requires at least user-provided data/corpus/clauses.jsonl and data/corpus/tables.jsonl in the project’s expected format. Add catalog summaries, catalog entries, version relations, and status records as needed. To navigate from evidence to an original page, provide the corresponding originals and source mappings expected by the code; records without originals cannot open a source page. Users also supply model, embedding, and reranking configuration. These resources are not included; the [Web setup guide](web_setup_en.md) describes the minimum setup.

demo/ and examples/ retain the authored offline CLI sample. The Streamlit page is now a separate no-key sample. Offline stand-ins cover only topics designed into the examples; they do not demonstrate review of arbitrary engineering plans. Existing screenshots show the initial Streamlit page, not the current React workbench.
