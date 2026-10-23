# Public Demo Regression Fixes — 2026-10-12

[English](review-fixes-2026-10-12.md) | [简体中文](../zh-CN/validation/review-fixes-2026-10-12.md)

This addresses the four reproducible findings reviewed at `e5fca44`. The documentation-only commit `956f0e7` had not changed those runtime paths. The changes repair workflow and citation boundaries without expanding product features.

| Finding | Corrected behavior | Regression coverage |
|---|---|---|
| Rejected Q&A citations looked like normal answers | The app displays the termination and validation state; unverified model text is labeled and rendered as plain text | Invalid citation, missing evidence, malformed output, and valid citation |
| Relocated quotation retained the wrong source anchor | `source_ref` comes from the actual matched unit; `model_source_ref` retains the submitted anchor | S001→S002, unique match from an unknown anchor, valid citation, duplicate and missing quotations |
| Default web review dropped the second sample unit | The editor loads all units; the complete edited text goes through the text loader with fresh line locations | Two-unit web/CLI input parity and preservation of the edited second line |
| Plan-specific configuration was ignored | Plan clients receive plan output, timeout, thinking and context budgets; Q&A retains chat settings | Plan 4321/19/16000 versus chat 1234/7, and offline operation without live settings |

Validation used the existing Python 3.12.10 environment, 18 bounded tests and one offline CLI run. Program network connections were intercepted, with zero attempts. No real online model, new public benchmark, or real-plan evaluation was run.

The original comparison of 58 core files was an author-side local check and cannot be independently verified against the private source from this repository. `plan_review_agent.py` now contains public-demo fixes and is marked `adapted=true` in the manifest. Initial validation and screenshots remain historical records; the current app loads both sample plan units by default.
