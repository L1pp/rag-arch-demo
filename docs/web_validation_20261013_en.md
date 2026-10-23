# React/FastAPI Source Migration Validation — 2026-10-13

[English](web_validation_20261013_en.md) | [简体中文](web_validation_20261013.md)

Historical scope note (updated 2026-10-23): This page records the first React/FastAPI source migration, which retained Streamlit as a separate sample and used the earlier single-process workbench. Streamlit has since been retired and removed. The tests below do not validate the current PostgreSQL-backed multi-user architecture. See the [current migration validation record](web_validation_20261023_en.md).

At the time, the default entry exposed the React/FastAPI workbench. streamlit_demo.py, the offline CLI, and authored examples remained separate. No original Git history or private corpus was imported.

- Installed the web requirements and editable package in an independent Python 3.12.10 environment. Added the optional Streamlit dependencies for sample tests; pip check passed.
- Generated frontend API types from the public backend contract. Eight frontend suites covering 47 tests passed; five suites passed initially, and the three affected suites passed after correcting a JSX closure. TypeScript checking and Vite production build passed.
- All 18 existing sample regressions passed in the new environment. The offline CLI produced Markdown/JSON output with zero attempted socket connections.
- Three FastAPI tests passed with injected in-memory resources: plan configuration, workspace/task validation-state transport and failed-task state. The validation-state test also checks that Markdown downloads retain the rejection warning.
- Source/configuration checks found no confirmed keys, private host addresses or personal paths. The specific history-source directory was replaced with the generic operator-supplied data/meta/history/active_preparation.json path.
- Original Word pagination helper source is included; Word rendering was not exercised.

These checks verify source compilation, selected API contracts and the authored sample paths. They do not verify live model services, a populated production corpus, login in a real browser, original-document viewing, or production deployment. The screenshots associated with that release were historical Streamlit samples; no React screenshot or browser takeover was performed in that migration.
