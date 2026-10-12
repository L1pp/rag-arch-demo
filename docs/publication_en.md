# Publication Scope and Review Record

[English](publication_en.md) | [简体中文](publication.md)

This demo started in a new directory with an independently initialized Git repository, without importing the original commit history. Its creation did not use clone, fork, bundle, cherry-pick from old commits, or an old object database; the new repository is independent of any private backup repository.

Published files: core source code, demo entry point, configuration placeholder template, dependency declaration, authored sample materials and plan, generated authored sample report, demo screenshots, evaluation summary figures, and accompanying notes.

Excluded files: original `.env`, complete standards library, full third-party datasets, real plans and customer attachments, personal résumé, original screenshots, logs, vector cache, model weights, database, runtime directories, backup archives, old Git history, and remote configuration.

Review tools: `python scripts/verify_public_tree.py` scans the tree prepared for publication; `python scripts/verify_public_tree.py --tracked` scans files in the Git index. The checks look for sensitive file types, personal absolute paths, known credential formats, and private-key headers, and reject symbolic links. The report outputs only file locations and issue categories to avoid displaying suspected secrets again.

Automated scanning can only detect issues covered by its rules. It cannot confirm company agreements, course permissions, or copyright ownership. The publisher should confirm the right to make the selected core code public. This project’s code is licensed under MIT; the repository retains the course-source attribution, and third-party dependencies remain subject to their respective licenses.

The DEMO standard identifiers, thresholds, objects, and plans in the sample materials are authored fiction; they do not cite or replace actual engineering standards. Existing evaluations publish aggregate figures only; original questions, reference answers, plans, full reports, and request records are not public.