# React/FastAPI migration validation — 2026-10-23

[English](web_validation_20261023_en.md) | [简体中文](web_validation_20261023.md)

This records local validation of the independent public repository's PR candidate. Actual GitHub CI and review evidence belong on the PR; this does not claim the change is merged or deployed.

## Checks performed

| Check | Observed result |
|---|---|
| Python dependencies | Installed in the dedicated `.venv-web`; `pip check` passed; Streamlit uninstalled |
| Backend integration | 169 tests across 22 modules passed with no failures, errors or skips, using a new loopback PostgreSQL 18 test database and random schemas |
| Connection-string guard | Four additional database-free regressions passed, rejecting target overrides through query parameters, duplicates, malformed paths, fragments and passwords |
| Frontend | 72 tests in 13 files passed; generated API types, TypeScript and Vite build passed |
| Offline CLI | Report and JSON output completed with socket connections intercepted; zero connection attempts |
| Python package | Wheel built with all four SQL migrations, worker and CLI, excluding private runtime resources |
| Publication scope | Rule scan of 224 staged files passed; deployment templates, synthetic fixtures and three existing fictional screenshots were also inspected |

The first backend run had seven failures caused by a copied test guard fixed to the old port. After adapting it to an explicit loopback test URL and valid port, the full integration run passed. A subsequent focused check hardened the guard against libpq query-parameter overrides. Tests use synthetic documents and fake models, with no real provider model calls. Independent GPT validation subsets overlap and are not added to the coverage total.

## Follow-up regressions from PR review

- Word conversion no longer treats an earlier polling timeout as a final timeout after successful completion. Both repositories passed 14 synthetic supervision tests; cancellation and total-deadline behavior remain intact.
- API-owned shared PDF clients reopen for a later request only after confirmed cleanup. Four new tests passed in each repository; the public API/worker/tenant subset passed 41 with five database-gated skips. Uncertain cleanup, supervised-worker retirement and permanent shutdown remain conservative.
- Controlled mmap lookup supports the lazy Mapping without materializing every Chunk. Each repository passed 17 new tests; the internal StandardTools suite passed 18 and both existing mmap-corpus suites passed four. Version and ambiguity boundaries remain enforced.
- The first Linux CI run exposed a Windows-only path fixture. It now uses a temporary directory; four focused CLI tests and the subsequent CI run passed. The PR CI runs the final combined regression set.

## Internal fixes

Relocated quotations now use the matched unit's actual source anchor while preserving the model's original anchor. Rejected-citation state survives the backend response schema into the page and Markdown download. The plan client uses its configured context budget. Internal checks passed: 23 targeted backend tests, 16 related frontend tests and TypeScript. Internal Git history, unrelated worktree changes and runtime material are not part of this PR.

## Review and limits

The main session evaluates normal independent GPT review and the additional third-party read-only review; their actual results are recorded on the PR. A GitHub automated review counts only if an actual record appears. The CI PostgreSQL service is disposable test infrastructure, not a deployment template.

No real-account migration, production deployment, online model-quality evaluation or real Office conversion was performed. Reference corpora, customer plans, credentials, databases, vector packages, history snapshots and runtime logs are excluded. Existing React screenshots are fictional offline UI illustrations from October 13; they do not validate the new authentication, persistence or multi-user features.
