# Phase 4: Python SDK, CI and developer experience

## What is available

- A standalone Python 3.10+ package in `sdk/python`, imported as `llmforge`, with only HTTPX as a runtime dependency.
- Fetch production/staging or pinned prompt versions, compile Mustache/restricted brace templates locally, and revalidate cached snapshots with ETags.
- Evaluation-scoped project keys for dataset reads, bounded server-side runs, polling/cancellation, and external output/check/metric submissions.
- A CLI quality gate, JSON report export, a reusable GitHub Actions workflow, and SDK test/wheel-build jobs in repository CI.
- Public `/docs`, an updated landing page describing the actual prompt workflow, and runnable Python examples.

The package is built and installed locally. It has not been published to PyPI, and no deployment or remote GitHub workflow run is implied by local verification.

## Setup

Run the schema migration before using SDK keys:

```powershell
cd backend
.\venv\Scripts\python.exe -m alembic upgrade head
cd ..
.\backend\venv\Scripts\python.exe -m pip install -e ./sdk/python
```

Open Settings in your project. A normal key grants `prompts:read`. For CI or evaluation submissions, check **Allow evaluations for CI** when creating a new key. Copy it once into an environment variable or your CI secret store. Migration `l5m6n7o8p9q0` adds a scopes column and preserves all existing keys as read-only. Revocation applies to the next request, including conditional prompt fetches.

```powershell
$env:LLMFORGE_URL = 'http://localhost:8000/api/v1'
$env:LLMFORGE_API_KEY = '<key copied from Settings>'
.\backend\venv\Scripts\python.exe -m llmforge --help
```

These variables belong to the SDK process; the dashboard continues using Clerk. A hosted API root should use HTTPS. No project ID header is needed: the key determines its project. Keys cannot authorize Clerk-protected dashboard writes.

## End-to-end demonstration

1. Follow the [evaluation demo](PHASE_3_EVALUATIONS.md): `Echo demo` v1 uses `{{query}}`; `Greetings` revision 1 contains Hello/Goodbye references. Promote v1 to production.
2. Run `sdk/python/examples/fetch_prompt.py` with a read-only key. It fetches production and compiles Hello.
3. Set an evaluation-enabled key and run:

```powershell
.\backend\venv\Scripts\python.exe -m llmforge evaluate --prompt 'Echo demo' --version 1 --dataset Greetings --dataset-version 1 --min-pass-rate 1 --output artifacts/evaluation.json
```

4. Run the same command with prompt v2 (`Reply: {{query}}`). The exact-match checks fail and the CLI exits 1. This is a prompt contract regression, not an estimate of real model quality.
5. Inspect both runs in Evaluations, or run `llmforge check artifacts/evaluation.json` offline.
6. Run `sdk/python/examples/submit_results.py` to submit application-computed outputs/checks. The grid marks them **External evaluation**. Optional numeric metrics appear in the run summary.

For a real use case, replace local echo with your application's support-ticket classifier, use a saved JSON-label dataset and JSON-path assertions, and compare the candidate with a prior prompt version. Optional judges call a separate configured provider and return a rubric score/reason. Judge scores remain model opinions.

See [SDK reference](../sdk/python/README.md) for method signatures, wait behavior, secret handling and examples. Submission payloads are limited to 1 MB, 100 results and 20 finite named metrics. Each case needs exactly one result, with an output/check set or error. The backend derives pass/error counts from the submitted checks; external scores and metrics remain self-reported.

## CI/CD integration

Copy [evaluation-gate.yml](examples/evaluation-gate.yml) to your application's `.github/workflows`. Configure a reachable persistent backend, `LLMFORGE_URL` as a repository variable, and `LLMFORGE_API_KEY` as a repository secret with evaluation access. Set the SDK checkout `ref` to a verified commit once these changes are committed/pushed. The example skips fork PRs because they cannot access repository secrets.

The workflow installs the SDK from source, evaluates pinned versions, and uploads the JSON report even when the quality gate fails. Exit codes: 0 pass, 1 quality regression/case errors, 2 operational/configuration failure. A gate never accepts provider errors just because a lower pass-rate threshold was chosen. For live providers add `LLMFORGE_PROVIDER_API_KEY` as a secret and select provider/model explicitly.

Repository CI validates the SDK on Python 3.10/3.12, builds and installs its wheel, applies migrations to a disposable PostgreSQL database, and runs the SDK/CLI smoke test. Frontend CI checks types, lint, tests, and the production build. A push to `main` syncs the backend to Hugging Face only after these jobs succeed; the deployed database still needs its own migration procedure. The SDK is not published to PyPI.

## Architecture and interview discussion

The useful problem is separating a prompt's mutable release label from an immutable testable snapshot. Dataset revisions and prompt versions make a run reproducible. The results grid explains which cases failed; CI can enforce an output contract before a release label moves. An application can use the same prompt source and submit its own evaluation checks.

Scoped high-entropy keys are stored only as hashes. Prompt reads default to a limited capability, and evaluation access must be explicitly selected. A conditional cache still authenticates with the server, so revocation is observed. HTTP mutations are not automatically retried, avoiding duplicate evaluation runs. Provider secrets are used in memory, never in run configuration.

The runner intentionally uses sequential FastAPI background work with persisted progress for this personal project's scale. Restarted jobs fail visibly instead of silently resuming paid calls. This is a known operational boundary: use a persistent backend process, and discuss moving the runner to a durable queue if the project's requirements change.

Local verification covers package installation/build, compiler semantics, cache/promotion/revocation, key scopes and tenant isolation, external submissions, CLI exit codes, backend regression tests and frontend checks. Browser visual verification is separate from automated component tests.

With the API running locally and the SDK installed, `python -m scripts.smoke_sdk` from `backend` tests the actual HTTP SDK and CLI against Postgres using a disposable project. It checks passing/failing prompt gates, external metrics, read-only capability enforcement and revocation, then removes the temporary workspace. It doesn't call a model.
