# RAG Arch Demo | React + FastAPI Building Document Assistant

[English](README.md) | [简体中文](README_CN.md)

This repository presents the current React + FastAPI implementation of a building-document assistant. The React frontend, FastAPI API, PostgreSQL task store, and AI workers run as separate components. The code shows how account-scoped tasks move through document processing, hybrid retrieval, evidence citations, plan review, resumable progress events, and report export.

The public tree contains application code and authored fictional examples. It does not include the complete standards corpus, original source documents, API credentials, database, or generated corpus/vector packages. Users provide PostgreSQL, model settings, their own corpus, and any originals needed for source-page viewing. The web workbench is an engineering showcase, not a turnkey hosted service. See the [setup guide](docs/web_setup_en.md).

## Current web architecture

Each account has its own task, attachment, history, and audit records within a deployment. PostgreSQL persists authentication and task/event state; independent API processes accept requests while AI workers claim and execute durable tasks. Clients can reconnect to task events and continue from a cursor. The database enforces per-account assistant slots and the deployment-wide execution limit.

The workers retain the retrieval and agent flows. Dense vectors, BM25 indexes, and corpus records can be shared through read-only memory-mapped packages, with full corpus records decoded as needed. Model, search, embedding, and reranking network orchestration runs asynchronously in the worker; bounded local processing remains under process and resource supervision. The [architecture guide](docs/architecture_en.md) maps these paths to source files.

## Run the local workbench

Use the [step-by-step setup guide](docs/web_setup_en.md). It covers Python and frontend dependencies, explicit PostgreSQL migrations, account creation, and starting the API, worker, and React development server. The interface requires user-provided model settings and materials for real document workflows.

The authored offline CLI remains available for a small, deterministic example:

~~~~bash
python -m demo --output output
~~~~

It writes Markdown and JSON results under output/. The stand-ins demonstrate only the authored examples and do not review arbitrary plans.

## Contribution and scope

I led the implementation of the building-document assistant and am authorized to publish the selected code in this repository. The source includes document loading and source locations, BM25 and vector retrieval, RRF fusion, reranking, evidence tools, agent tool use, plan quotation checks, citation validation, and report generation. Course-derived design notes and third-party component information remain in the [notices](THIRD_PARTY_NOTICES.md).

The public sample data is fictional. The complete internal corpus, customer materials, runtime state, private evaluation inputs, and derived packages tied to those materials are not included. Public historical evaluation figures are summarized in the [evaluation notes](docs/evaluation_en.md); the original inputs and reports are not published.

## Earlier React UI illustration

These screenshots were captured on 2026-10-13 using a fictional offline fixture. They illustrate an earlier React interface and do not show the current account, PostgreSQL task, or worker flows. They are not evidence of live model or production validation.

![Earlier React workbench illustration](docs/screenshots/react-01-workbench.png)
![Earlier evidence view with authored offline results](docs/screenshots/react-02-evidence.png)
![Earlier plan review view with authored offline results](docs/screenshots/react-03-plan-review.png)

## Repository guide

- frontend/: React application.
- rag_arch/web/: FastAPI API, authentication, persistence, tasks, events, uploads, and downloads.
- rag_arch/: document processing, retrieval, evidence, agents, plan review, and reports.
- scripts/: explicit database migrations and local corpus/index preparation.
- demo/, examples/: authored offline CLI sample and fictional inputs.
- docs/: architecture, setup, evaluation boundaries, publication scope, historical UI screenshots, and validation records.

Read the [architecture guide](docs/architecture_en.md), [setup guide](docs/web_setup_en.md), and [publication notes](docs/publication_en.md). The 2026-10-12 and 2026-10-13 validation pages are historical records for earlier code paths. The current migration validation record is [here](docs/web_validation_20261023_en.md) and will be completed from the final test and review evidence before release.

## License

Original project code is licensed under [MIT](LICENSE). Third-party dependencies follow their own licenses; commercial use of dependencies and outputs is not guaranteed, and users should check the terms applicable to their intended use. See [third-party notices](THIRD_PARTY_NOTICES.md).
