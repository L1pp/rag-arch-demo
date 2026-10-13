# Publication Scope and Review Record

[English](publication_en.md) | [简体中文](publication.md)

This repository was assembled in a new directory with an independent Git history; it does not import the original repository history or connect to the private backup. The selected public source now includes the React frontend, FastAPI workbench, retrieval and Agent core, CLI sample, and a separate no-key Streamlit sample.

This is a source showcase for code review and interviews, not a complete deployment bundle. Users supply the standards corpus, source documents and service credentials, and create local accounts; runtime databases are generated locally. Without those external resources, the corresponding real-document workflows in the main web workbench cannot run. The new README screenshots were captured in an isolated temporary screenshot environment by connecting offline sample results to the current React pages; the materials are authored fiction. Normal use of the main workbench still requires users to configure API settings, a corpus, and source documents. The images do not verify an online model service or production deployment. Screenshots from the initial Streamlit release remain in docs/screenshots/. The CLI and Streamlit samples also use authored fictional inputs.

Not included: original project configuration, complete standards materials, full third-party datasets, real plans and customer attachments, private evaluation inputs, runtime logs, vector caches, model weights, authentication database, attachments and history, backup archives, or old Git objects.

The public-tree checker is python scripts/verify_public_tree.py; it also supports --tracked to inspect the Git index. Its rules cover only configured sensitive file types, personal paths, and credential patterns. They do not replace manual review of new files and screenshots, or establish company permissions, course permissions, or copyright ownership. Course sources and third-party license boundaries are listed in the [third-party notices](../THIRD_PARTY_NOTICES.md).

The DEMO standard identifiers, thresholds, objects, and plans are authored fiction. They do not cite or replace actual engineering standards. Only aggregate historical evaluation figures are public; original questions, reference answers, plans, full reports, and request records are not included. The repository examples do not reproduce those results.

The project’s own-code license is in [LICENSE](../LICENSE) and covers only project-owned material. Third-party components remain under their own licenses.
