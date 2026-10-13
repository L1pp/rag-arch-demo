# RAG Arch Demo | React + FastAPI Building Document Assistant

[English](README.md) | [简体中文](README_CN.md)

This repository is a code showcase for a building-document assistant. Its primary web entry is a React single-page app served by a FastAPI backend. Follow the implementation from document upload and retrieval through evidence citations, plan review, task progress, cancellation, source-page viewing, and report export.

The public tree contains selected application code and fictional examples. It does not include the private standards corpus, original source files, credentials, or runtime database. The web workbench is therefore not a turnkey public service: users must provide compatible model settings and their own corpus and source materials for the workflows they want to run. See the [web setup guide](docs/web_setup_en.md).

## Main web workbench

The React interface calls FastAPI routes for workspace state, assistant tasks, server-sent progress events, task cancellation, document upload, source pages, history, and report downloads. The backend connects those requests to the retained retrieval and agent code. The repository keeps the real application flow visible while omitting proprietary source material.

The login feature manages local accounts, but accounts share the same local workspace and runtime data; it does not provide per-account data isolation. The included launch instructions bind to localhost for local review.

<a id="run-the-demo"></a>

## Separate sample paths

The small, authored offline CLI sample is still available:

~~~~bash
python -m demo --output output
~~~~

It writes a Markdown report and JSON results under output/. Deterministic offline stand-ins replace model, embedding, and reranking calls. They demonstrate the flow for the authored examples only; they do not review arbitrary plans.

The separate no-key Streamlit sample remains available as streamlit_demo.py:

~~~~bash
python -m pip install -r requirements-demo.txt
python -m streamlit run streamlit_demo.py --server.address 127.0.0.1 --server.port 8501
~~~~

This is a small standalone sample, not the primary React/FastAPI workbench. Its screenshots below are from the initial Streamlit release and are included as historical sample material.

## Core implementation and contribution

I led the implementation of the building-document assistant and am authorized to publish the selected code in this repository. The source shows document loading and source locations, BM25 and vector retrieval, RRF fusion, reranking, evidence tools, the agent tool loop, plan quotation checks, citation validation, and report delivery. Course-derived design notes and third-party component information remain in the [notices](THIRD_PARTY_NOTICES.md).

The [architecture guide](docs/architecture_en.md) maps the web and retrieval paths to source files. The [web setup guide](docs/web_setup_en.md) explains the user-provided configuration and data expected by the web app.

## Historical sample screenshots

These screenshots show the fictional, offline Streamlit sample from the initial release. They do not depict the current React interface.

![Historical Streamlit sample: authored materials](docs/screenshots/01-library.png)
![Historical Streamlit sample: retrieval and citations](docs/screenshots/02-qa.png)
![Historical Streamlit sample: plan review report](docs/screenshots/03-review.png)

## Existing results and limitations

In a historical 60-question retrieval regression, adding reranking to the same RRF candidate pool raised Recall@8 from 0.8444 to 0.8722 and increased complete evidence matches from 44/60 to 47/60. At Top-4, complete evidence matches fell from 40/60 to 39/60. On a fixed 240-question AECBench QA set, accuracy on decodable answer choices rose from 146/240 to 184/240.

The five-plan whole-plan evaluation used 59 fixed development reference items. These results measure reference coverage, not full-report accuracy; they do not establish that RAG consistently improves technical findings across whole plans. The [evaluation notes](docs/evaluation_en.md) describe the scope, native-review comparison, unfavorable results, and changes that cannot be attributed to RAG. The public examples do not reproduce those results.

Citation checks can verify sources and quoted text. They cannot automatically establish engineering applicability, correctness of conclusions, or completeness of issue discovery. Text extraction gaps, charts, and complex calculations still require manual review.

## Repository guide

- frontend/: React application and build configuration.
- rag_arch/web/: FastAPI app, authentication, tasks, upload, source, and history routes.
- rag_arch/: retrieval, evidence, agents, plan review, and report implementation.
- demo/, examples/: offline CLI sample and authored fictional inputs.
- streamlit_demo.py: independent small Streamlit sample.
- docs/: setup, architecture, evaluation, publication scope, and historical screenshots.

The [initial validation record](docs/validation_en.md) refers to the original Streamlit sample; it is not a validation report for the current React/FastAPI workbench. The [publication notes](docs/publication_en.md) describe the scope of the public source selection.

[Web migration validation](docs/web_validation_20261013_en.md) records the bounded checks for this source update.

## License

Original project code is licensed under [MIT](LICENSE). Third-party dependencies follow their own licenses; commercial use of dependencies and outputs is not guaranteed, and users should check the terms applicable to their intended use. See [third-party notices](THIRD_PARTY_NOTICES.md).
