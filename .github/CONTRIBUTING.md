# Contributing

Issues and pull requests in English or Chinese are welcome. Keep the change focused and describe its user-visible effect and public scope.

## Local setup

Use Python 3.12, Node.js 22, and PostgreSQL for the full backend suite. Create the repository-local `.venv-web` environment: the API type generator uses its Python executable.

```sh
python -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r requirements/web.txt
python -m pip install --no-deps -e .
```

On Windows PowerShell, activate with `.\.venv-web\Scripts\Activate.ps1` instead of `source`. See the [web setup guide](../docs/setup.md) for database and local startup steps.

Install and check the frontend from `frontend/`:

```sh
npm ci
npm test
npm run check:api
npm run build
```

The backend suite uses `unittest`, not pytest. Run it with:

```sh
python -m unittest discover tests
```

Database-backed tests use `RAG_ARCH_TEST_DATABASE_URL` when it points to a dedicated disposable PostgreSQL database; those tests skip when it is unset. Never point this variable at a production or personal database. The GitHub Actions job supplies a temporary PostgreSQL service and synthetic test data.

## Branch and pull request flow

1. Make a focused branch and open a pull request against `main`; public changes go through a formal PR.
2. Describe the behavior, affected files or boundaries, privacy implications, and exact validation results. Include relevant checks that were not run.
3. Let the PR's configured CI finish and address failures. Link to the actual CI run in the PR review evidence.
4. If an automated AI review actually appears on that PR, link its result and address relevant findings. Otherwise state that it did not run. Repository or account settings alone are not evidence of a completed review. Do not manually mention a review bot or change account-level review settings as part of contribution setup.
5. Merge after the required CI and review checks for the PR are complete.

## Public data and documentation

Keep fixtures synthetic and authored for this repository. Do not add internal corpora, customer plans, source documents, private logs, caches, API keys, tokens, or personal paths. Users provide their own API credentials and source materials where needed. Update the English and Chinese documentation together when the documented behavior changes.
