# Running the regression checks

From the repository root, install the backend development dependencies and SDK test extra:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\backend\venv\Scripts\python.exe -m pip install -e './sdk/python[test]'
```

Run the backend suite in a shell configured for mock inference and a non-production database URL:

```powershell
cd backend
$env:INFERENCE_ENGINE = 'mock'
$env:DATABASE_URL = 'postgresql+asyncpg://test:test@127.0.0.1:1/test'
.\venv\Scripts\python.exe -m pytest tests -q
```

The workspace, prompt and evaluation integration tests use the shared `api` fixture in `backend/tests/conftest.py`, with an isolated SQLite database and fresh application per test. Both request sessions and background evaluation sessions use that database. `support.py` handles test sign-in/provisioning; `factories.py` creates checked resource scenarios. Test modules do not import each other's fixtures or setup functions.

The old mocked benchmark-route tests explicitly request `legacy_route_identity` using `pytestmark`. This override is limited to those modules, restored afterwards, and does not apply to the workspace/security integration tests. The application has no test authentication bypass.

Run the SDK suite separately from the root:

```powershell
.\backend\venv\Scripts\python.exe -m pytest sdk/python/tests -q
```

SDK tests are grouped into HTTP client/cache behavior, template compilation, and CLI quality gates. Their fixtures provide fresh response payloads and mock HTTP clients. No Clerk or provider credentials are required.

Frontend checks from `frontend`:

```powershell
npm test
npx tsc --noEmit
npm run lint
npm run build
```

Stop `next dev` before building because both use `.next`. Production builds and component tests verify compilation and behavior; they do not constitute browser visual acceptance testing.

For a database-backed smoke test, use a separate shell with the normal backend environment, apply migrations, and run the API locally. Then run `python -m scripts.smoke_sdk` from `backend`. It creates a disposable project, exercises the actual SDK and CLI over HTTP, checks passing/failing gates and revocation, and removes its own workspace. It doesn't call a model. Do not use the deliberately invalid test database URL for this smoke test.


## Durable execution and API contracts

Real locking tests are separate from the default SQLite suite. Configure `LLMFORGE_TEST_POSTGRES_URL` to a dedicated development PostgreSQL database (CI uses its PostgreSQL service), then run `python -m scripts.check_postgres` and `python -m scripts.check_migrations` from `backend`. Explicit script invocation can also use the configured development URL. Checks create uniquely named temporary schemas; ORM tables are schema-qualified to remain isolated behind connection poolers. Migration checks run the complete chain inside one transaction and roll it back. Never point these checks at production.

Recovery tests interrupt after generation and between evaluators, preserve error-stage checkpoints, fence late writes, check global/project admission, and enforce SDK scopes/tenant isolation. Structured-log tests discard secret/context fields. Frontend tests retain dispatch identity after an uncertain resume response and clear credentials after acceptance. A browser walkthrough should show the same run progressing from interrupted to completed with a new attempt and preserved case content. Fully checkpointed fixtures can exercise the UI without paid calls; provider crash semantics are covered by injected interruption tests.

Regenerate contracts with `python -m scripts.export_openapi` followed by `npm run contracts:generate` from `frontend`. Repeating generation must produce identical files. CI checks the committed schema and generated frontend types before deployment sync; SDK contract tests use the same exported schema.

## Provider-free feedback benchmark

`python -m scripts.benchmark_evaluations --runs 10 --output artifacts/evaluation-benchmark.json`
uses the actual SDK, HTTP routes and lifecycle-managed runner on four fixed cases.
It requires **LLMFORGE_TEST_POSTGRES_URL pointing to localhost PostgreSQL**; it
refuses hosted targets and never falls back to DATABASE_URL. It creates/removes
only a uniquely named test schema and binds its temporary HTTP server to loopback.
Dashboard setup uses a test identity; SDK authentication/scopes remain real.
No generation model or judge provider is called.

The report separates the first run and ten warm samples for inline assertions and
a saved builtin, with SDK/lifecycle durations, p50/p95, SQL statement/commit counts
and stage timings. Counts include polling and runner activity; they are not exact
per-case transaction budgets. The isolated regression test enforces the durable
write budget and retained checkpoints deterministically. Local timing is evidence
for this fixture/environment, not a hosted SLA. Use repeated measurements on the
same machine to establish a timing threshold; investigate hosted delay through
deployment metrics before promising a production latency improvement.

Once established, pass `--max-p95-ms` to fail a local regression run while
retaining its measurements. The October 9 reference checks use a provisional
1,000 ms local warm-p95 budget; this is not an agreed hosted service target.
