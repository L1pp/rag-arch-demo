# Publication Scope and Review Notes

[English](publication.md) | [简体中文](zh-CN/publication.md)

This repository was created in a new directory with independent Git history. It is not linked to the private backup repository. The public project now presents the React frontend, FastAPI API, PostgreSQL-backed account and task state, independent AI workers, retrieval and agent code, and an authored offline CLI sample. Streamlit is retired and removed as a project entry point.

The repository is a code showcase for engineering review and interviews, not a turnkey deployment bundle. The complete internal corpus, original standards documents, real plans, account database, service credentials, runtime logs, attachment history, and generated packages based on private data are not published. Users provide their own PostgreSQL instance, model configuration, corpus, original files, and derived retrieval packages required by their chosen workflows.

The screenshot archive includes images captured on 2026-10-13 with a fictional offline sample. Those images show an earlier React interface and do not validate current worker execution or live model behavior.

The authored DEMO identifiers, clauses, values, and plans are fictional. Public historical evaluation results are aggregate summaries only; source questions, reference answers, plans, complete reports, and request logs remain private. Conclusion-control and citation-recovery experiments are not presented as production features.

The [2026-10-23 validation record](validation/2026-10-23.md) separates the original local PR-candidate checks from the post-merge GitHub CI result. The merged CI passed 208 backend tests, 72 frontend tests, and the 224-file publication scan. Earlier pages retain results for their historical code paths.

The repository includes a public-tree checker at scripts/verify_public_tree.py. Its configured patterns are a useful scan, not proof that all newly added files are safe to publish or that company/course rights have been independently confirmed. Review new code, examples, screenshots, logs, caches, and Git changes before release. Course references and component boundaries are listed in [third-party notices](third_party_notices.md).

Original project code is licensed under [MIT](../LICENSE). The license covers project-owned code; third-party components keep their own licenses.
