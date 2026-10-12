# Initial Release Validation

[English](validation_en.md) | [简体中文](validation.md)

Date: 2026-10-12. This records validation of the initial commit `403cb3d`. No new public benchmark, real-plan evaluation, or online model call was run. See the later [regression fixes](review_fixes_20261012_en.md).

- Created an independent Python 3.12.10 environment and installed the requirements and project as documented in the README. `pip check` found no dependency conflicts.
- Five bounded offline tests passed: tool-based evidence retrieval and citations, report changes after editing an input, no invented citations when evidence is missing, no remote clients constructed by default, and the CLI output bundle.
- `python -m demo --output output` produced the Markdown report and JSON files. The output records `offline` mode and `network_model_calls=0`.
- Playwright opened the local app and ran the Q&A and plan demos. Three screenshots were visually reviewed and contain only fictional sample material.
- Local documentation links and image paths existed.
- The 58 core source files in the manifest matched their selected source files byte for byte. No original Git objects or history were imported.
- The public tree and Git index passed publication checks. An index-content regression check confirmed that a simulated credential remains detectable after the working-tree file is cleaned.

The verified runtime scope is the default offline demo. Online mode retains the original service adapters and requires the user's own configuration. No paid or real API request was made, so online-service availability was not marked as verified. The screenshot server was used only for this validation and was not registered for automatic startup.
