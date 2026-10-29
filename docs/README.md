# Documentation

[简体中文](zh-CN/README.md) | [Project overview](../README.md)

This index groups the public demo's architecture, setup, evaluation, publication, and validation material.

## Guides

- [Architecture and code reading map](architecture.md) · [中文](zh-CN/architecture.md)
- [Run the web workbench](setup.md) · [中文](zh-CN/setup.md)
- [Existing results and metric boundaries](evaluation.md) · [中文](zh-CN/evaluation.md)
- [Publication scope and review notes](publication.md) · [中文](zh-CN/publication.md)
- [Third-party sources and license notes](third_party_notices.md)

## Validation history

- [2026-10-29: evolving implementation snapshot](validation/2026-10-29.md) · [中文](zh-CN/validation/2026-10-29.md)

- [2026-10-23: PR candidate checks and post-merge CI](validation/2026-10-23.md) · [中文](zh-CN/validation/2026-10-23.md)
- [2026-10-13: first React/FastAPI migration](validation/2026-10-13.md) · [中文](zh-CN/validation/2026-10-13.md)
- [2026-10-12: initial offline sample](validation/2026-10-12.md) · [中文](zh-CN/validation/2026-10-12.md)
- [2026-10-12: public demo regression fixes](validation/review-fixes-2026-10-12.md) · [中文](zh-CN/validation/review-fixes-2026-10-12.md)

Each record keeps the scope and results from its own run. Earlier test counts and screenshots describe earlier code; the post-merge CI result is recorded separately in the 2026-10-23 page.

## Interface previews

These screenshots were captured on 2026-10-29 using fictional inputs and a simulated local API. They show the interface without validating live OCR or model results.

- [Workbench](screenshots/react-20261029-workbench.png) (also shown in the root README)
- [Page and region viewer](screenshots/react-20261029-page.png)

## Screenshot archive

All three images were captured on 2026-10-13 with an authored fictional offline fixture. They illustrate an earlier React interface and do not validate current multi-account behavior, live model services, or deployment.

- [Earlier workbench](screenshots/react-01-workbench.png)
- [Evidence results](screenshots/react-02-evidence.png)
- [Earlier plan review](screenshots/react-03-plan-review.png)

See [core_manifest.json](core_manifest.json) for the source map and selected file notes.
