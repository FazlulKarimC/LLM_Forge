# Development and operations

Use this guide to run, verify and deploy LLMForge. [Architecture](ARCHITECTURE.md) owns system/domain contracts and limits; [DESIGN_SYSTEM.md](../DESIGN_SYSTEM.md) owns frontend design. The [root README](../README.md) is the product overview and quick start. The [SDK package reference](../sdk/python/README.md) documents client method signatures.

## Local setup

The backend uses Python 3.12 in CI; the frontend needs Node.js 20.9+; the standalone SDK supports Python 3.10+. Use PostgreSQL and a Clerk application. Redis, Qdrant and provider credentials are optional for the Echo evaluation demo.

From a PowerShell terminal:

```powershell
git clone https://github.com/FazlulKarimC/LLM_Forge.git
cd LLM_Forge
python -m venv backend/venv
.\backend\venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt -c backend/constraints.txt
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env.local
```

Edit the copied files before starting either process. Keep real secrets out of Git. Backend minimum configuration:

```dotenv
DATABASE_URL=postgresql://user:password@localhost:5432/llmforge
INFERENCE_ENGINE=mock
QUEUE_BACKEND_MODE=inline
CLERK_ISSUER_URL=https://YOUR_APPLICATION.clerk.accounts.dev
CLERK_AUTHORIZED_PARTIES=http://localhost:3000
CORS_ORIGINS=http://localhost:3000
```

Use an existing dedicated development database and its actual connection string. Leave optional Redis/provider settings unset instead of using the example file's placeholder endpoints/tokens. `QUEUE_BACKEND_MODE=inline` makes local benchmark dispatch explicit; evaluation execution uses its own runner.

Frontend configuration, from the same Clerk application:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_REPLACE_ME
CLERK_SECRET_KEY=sk_test_REPLACE_ME
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
```

### Authentication

Create a Clerk app, enable the desired sign-in method, and copy its publishable/secret keys to the frontend. Copy its actual Frontend API URL to backend `CLERK_ISSUER_URL`; do not guess the hostname. No JWT template or Clerk Organizations setup is needed: LLMForge owns organizations/projects in PostgreSQL.

Set `CLERK_AUTHORIZED_PARTIES` to exact frontend origins, comma-separated. For this repository's hosted frontend the configured origin is `https://llmforge.vercel.app`, with no trailing slash/path. CORS normalizes trailing slashes; token `azp` comparison uses exact equality. Configure frontend domains/URLs in Clerk as well. The backend does not need the Clerk secret key; optional `CLERK_JWT_PUBLIC_KEY` avoids JWKS fetches and `CLERK_AUDIENCE` applies only when the token has that audience.

Missing frontend keys show setup guidance; invalid backend credentials deny access. A 401 can use **Refresh session** / **Sign in again**. If a fresh session is rejected, inspect issuer/origin configuration instead of repeatedly retrying transport.

### Migrate and launch

Review migration history before applying it to a database with existing data. Migration `i2j3k4l5m6n7` deliberately truncates legacy benchmark/workspace-reset tables; downgrade cannot restore discarded data. Never use a production database for local tests.

From the activated repository shell:

```powershell
cd backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --port 8000
```

In a second terminal, from the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000), sign in and select a project. Initial authenticated bootstrap creates a personal organization/default project; repeated requests reuse them. Settings manages additional organizations/projects and project keys. Restart after environment changes. Public frontend variables are build-time values in production; include `/api/v1` in the API URL or requests can fall back to localhost.

## Local SDK evaluations

Create a saved text prompt named `Echo` with template `{{query}}`. In Settings create a project key with evaluation access. Install the SDK from the repository (not PyPI), configure `LLMFORGE_URL` ending in `/api/v1` and `LLMFORGE_API_KEY`, then run:

```python
from llmforge import LLMForge

cases = [
    {"id": "greeting", "inputs": {"query": "Hello"},
     "expected_output": "Hello"},
]

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo", version=1)
    report = forge.evaluate_local(
        prompt, cases,
        task=lambda compiled: compiled,  # Echo only; replace with your model
        evaluators={"exact_match": lambda output, case:
                    output == case["expected_output"]},
        scoring_version="exact-match-v1",
        dataset_name="Local greetings",
        model="local-echo-demo",
        upload=True,
        report_path="artifacts/evaluation.json",
    )
    print(report.id, report.pass_rate)
```

Task callbacks receive the compiled text/chat prompt and must return a string. Evaluators receive output and the local case and return bool. Application/provider clients, including Gemini or LangGraph, run entirely in your process; no new hosted provider adapter is required. Provider credentials never enter the summary upload. Callback errors count as case errors and cannot pass the gate.

Cases can be a list or local JSON/JSONL path. Each needs a unique stable `id` and string-valued `inputs`; `expected_output` and JSON `metadata` are optional. The SDK fingerprints the actual selected cases including references/metadata, sorted by ID so reordering does not change identity. Changing cases or selected subsets changes the fingerprint. Change `scoring_version` whenever evaluator code/settings change; its hash also includes evaluator names. These are SDK declarations, not independently verified execution evidence.

Full inputs, outputs and checks are written locally before upload. Only prompt identity, dataset/scoring fingerprints, display names, counts and optional aggregate metrics reach the server; no dataset or case rows are created. `upload=False` skips submission. A failed upload leaves the full local artifact and its `submission_key` so a caller can retry the same summary/intent with `submit_local_evaluation(..., idempotency_key=...)`. Mutations do not retry automatically.

The current reporting mode is summary-only: the dashboard displays coverage, counts, metrics and compatible reference summaries. It cannot infer per-case regressions, re-score absent outputs or resume local execution. All external values remain self-reported. Summary comparisons require matching data/scoring fingerprints; numeric per-score deltas remain conservative.

Provider-free CLI from the configured SDK environment:

```powershell
python -m llmforge evaluate-local --prompt Echo --version 1 --dataset ./cases.json --upload --output artifacts/evaluation.json
python -m llmforge check artifacts/evaluation.json
```

`evaluate-local` deliberately uses an echo task and exact match; use Python callbacks for real models/evaluators. Without `--upload` its report stays local. CLI exit codes are 0 pass, 1 quality regression/case errors, 2 operational/configuration failure. Pin prompt versions and keep full reports in your application's chosen artifact store.

Project keys are created in Settings. Normal `prompts:read` keys can fetch releases; uploads need the explicit `evaluations:write` scope. Hosted SDK consumption requires no local FastAPI/PostgreSQL. Use `https://faz-ai-llm-forge.hf.space/api/v1` for the configured hosted backend or `http://localhost:8000/api/v1` for your own.

For application CI copy [evaluation-gate.yml](examples/evaluation-gate.yml), configure a reachable API and evaluation-enabled key, and pin the SDK checkout SHA. The example uses local cases without a provider call and skips fork PRs without secrets.

## Compatibility and migrations

Apply migration `p9q0r1s2t3u4` before deploying local reporting. It makes the legacy dataset revision reference nullable; existing records/foreign keys remain intact. Downgrade refuses to proceed while local reports exist. No hosted database migration is performed by repository CI.

Dataset management and old dataset demo creation are retired and return 410 for writes by default. `LEGACY_DATASET_WRITES=true` explicitly enables authenticated compatibility authoring for older clients or disposable tests; it is not required by the current SDK journey. New UI routes do not expose dataset management. Historical inspection/recovery APIs remain for pre-existing hosted runs; local reports are terminal summaries, without hosted case content or execution attempts.

## Testing

The bounded hosted benchmark profile defaults to `QUEUE_BACKEND_MODE=inline`, 20 cases, 20 generation calls, a 300-second new-call deadline and a 30-second HTTP timeout. `BENCHMARK_*` settings are documented in `backend/.env.example`; they are application limits, not provider quotas. ReAct defaults to three iterations and at most five hosted cases. Concurrent batch requests execute with effective batch size one; optional remote scoring and new RAG execution are disabled in this profile. Drafts/historical records retain their original configuration. The frontend reads actual budgets and bundled dataset counts from `/api/v1/capabilities`.

Install backend test dependencies and SDK test extras from the repository root:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\backend\venv\Scripts\python.exe -m pip install -e './sdk/python[test]'
```

### Isolated regression suites

From `backend`, set mock inference and a deliberately unreachable test URL; the default API fixtures use fresh isolated SQLite databases:

```powershell
$env:INFERENCE_ENGINE = 'mock'
$env:DATABASE_URL = 'postgresql+asyncpg://test:test@127.0.0.1:1/test'
.\venv\Scripts\python.exe -m pytest tests -q
```

From the repository root run `.\backend\venv\Scripts\python.exe -m pytest sdk/python/tests -q`. Test helpers live in `backend/tests/conftest.py`, `support.py` and `factories.py`; test modules do not import one another's setup. Hosted-flow fixtures explicitly enable the compatibility authoring switch; current-report tests assert the default write rejection. The legacy mocked route identity override is confined to its tests and does not create an application auth bypass. SDK tests mock HTTP and need no Clerk/provider account.

Frontend checks from `frontend`:

```powershell
npm test
npx tsc --noEmit
npm run lint
npm run build
```

Stop `next dev` before building because both use `.next`. These checks validate behavior/compilation, not authenticated browser or live-provider acceptance. Browser layout/theme/focus checks follow the single [design guide](../DESIGN_SYSTEM.md#verification-and-maintenance). Documentation-only edits require link/source checks rather than a fresh application build.

### PostgreSQL and HTTP checks

`tests/test_benchmark_execution.py` exercises durable per-case writes, provider-free saved-prompt batches, interruption, cancellation, expiry and shared model-job admission. Set `LLMFORGE_TEST_POSTGRES_URL` to a disposable target to run its real-lock Stop and competing-project tests. CI provides that variable against its existing PostgreSQL service. Without it, those tests skip; SQLite success does not establish lock behavior.

Use a separate shell with disposable localhost PostgreSQL, not the unreachable unit-test URL. From `backend`, set both database variables explicitly so configuration cannot select a hosted `.env` URL:

```powershell
$env:ENVIRONMENT = 'development'
$env:DATABASE_URL = 'postgresql+asyncpg://test:test@127.0.0.1:5432/llmforge_test'
$env:LLMFORGE_TEST_POSTGRES_URL = $env:DATABASE_URL
python -m scripts.check_postgres
python -m scripts.check_migrations
```

Replace the example credentials/database with those of your disposable local instance. These checks create uniquely named temporary schemas. `check_migrations` requires `LLMFORGE_TEST_POSTGRES_URL` on localhost, refuses hosted targets and never falls back to `DATABASE_URL`; it rehearses the migration chain and seeded legacy-history preservation inside a transaction that is rolled back. `check_postgres` can fall back to configured `DATABASE_URL`, so keep the explicit local variables above. SQLite cannot prove locking guarantees. The complete backend suite with `LLMFORGE_TEST_POSTGRES_URL` also exercises benchmark Stop/admission and pin-versus-purge races; `check_postgres` alone runs the legacy evaluation locking checks.

With the local API running, migrated PostgreSQL and the SDK installed, `python -m scripts.smoke_sdk` exercises real HTTP/SDK/CLI, passing/failing gates, external results, scope enforcement and revocation, then removes its disposable workspace. `python -m scripts.smoke_evaluations` checks the evaluation flow in a temporary project. The SDK smoke uses the current local report path; the legacy evaluation smoke requires explicit compatibility authoring. Both are provider-free by default; `--live-groq` on the latter explicitly performs a generation and judge call using its configured key.

### Legacy feedback timing and benchmark profiling

From `backend`:

```powershell
python -m scripts.benchmark_evaluations --runs 10 --output artifacts/evaluation-benchmark.json
```

This legacy hosted-run benchmark also requires `LEGACY_DATASET_WRITES=true` and `LLMFORGE_TEST_POSTGRES_URL` pointing to localhost PostgreSQL, refuses hosted targets and never falls back to `DATABASE_URL`. It runs fixed four-case fixtures over actual SDK/HTTP/lifecycle paths, removes its generated schema and binds its temporary server to loopback. It calls no model.

Reports separate first/warm samples, SDK/lifecycle durations, p50/p95, SQL counts and stage timings. Counts include polling/runner activity; local results are not hosted latency guarantees. `--max-p95-ms` can enforce a locally established budget. For reasoning benchmark combinations use [run_benchmark.py](../scripts/run_benchmark.py) explicitly; it can call configured providers. Runtime cache/batch/profiling fields are in [architecture](ARCHITECTURE.md#benchmarks-and-optional-infrastructure).

## Contract generation

Pydantic models define wire responses. Export without starting API lifespan or contacting dependencies:

```powershell
cd backend
python -m scripts.export_openapi
cd ../frontend
npm run contracts:generate
```

Commit `contracts/openapi.json` and `frontend/src/lib/generated/api-schema.ts` together. Repeated generation must be deterministic. Keep frontend view-model adapters and SDK contract tests aligned; do not hand-edit generated types to hide drift.

## Deployment and diagnosis

Apply additive migrations through `r1s2t3u4v5w6` before deploying benchmark admission and attempt history. The migrations add execution reservations and attempt records, preserve existing experiments/cases, and keep old aggregates at unbound attempt zero. Do not delete history or guess the old result attempt. Drain existing model jobs before changing execution versions; old workers do not implement the new admission contract. Keep the additive schema on rollback. Once several result attempts exist, older code that assumes one result per experiment needs a compatibility patch; do not deploy it unchanged. Downgrade refuses to remove the table while unexpired active reservations exist. No migration, deployment or hosted configuration change is performed by local tests.

[Repository CI](../.github/workflows/ci.yml) runs on pushes/PRs targeting main. It tests/builds/installs the SDK wheel on Python 3.10/3.12, tests backend on Python 3.12/PostgreSQL 15 with migrations/locking/HTTP smoke checks, and validates generated contracts plus frontend lint/types/tests/build on Node 20. After all three jobs pass, a main push calls [HF sync](../.github/workflows/hf-sync.yml); PRs do not publish the backend.

HF sync stages only `backend/` at the Space root, embeds the upstream source/build identity and uses `backend/Dockerfile.hf`. The root Dockerfile also copies backend contents. A separate configured frontend deployment serves Next.js; backend sync does not deploy it. No workflow here migrates the hosted database. Apply migrations to that intended database before launching the new image, and preserve durable records when rolling back.

Use the direct Neon endpoint in `DATABASE_URL` for the migration command, then restore the application connection URL before starting the API. Runtime defaults to a two-connection pool plus one overflow, pre-ping, and 10-second connect/checkout timeouts. Production requires PostgreSQL and TLS. Prefer `sslmode=verify-full`; the parser preserves other query options instead of stripping them. An unsupported `channel_binding=require` fails explicitly: configure a supported TLS policy rather than assuming the driver honored channel binding. Local disposable PostgreSQL tests can use a development URL without TLS. Never point the test URL at Neon.

### Existing free hosting profile

The personal deployment uses [Vercel](https://llmforge.vercel.app) for Next.js, [Hugging Face Spaces](https://faz-ai-llm-forge.hf.space) for FastAPI and the existing Neon PostgreSQL database. Redis/RQ, Qdrant, an additional worker/server and an object store are not required for the current hosted profile.

| Backend setting | Hosted value / purpose |
| --- | --- |
| `ENVIRONMENT` | `production` |
| `DATABASE_URL` | Existing Neon application/pooler endpoint with `sslmode=verify-full`; keep credentials in Space secrets |
| `QUEUE_BACKEND_MODE` | `inline` |
| `DB_POOL_SIZE` | Up to `2` retained connections |
| `DB_MAX_OVERFLOW` | `3` transient connections; maximum five total, overriding the repository default of one overflow |
| `DB_POOL_TIMEOUT` | `10` seconds; connect timeout also defaults to 10 seconds |

Use the same Clerk application and exact frontend origin for backend authorized parties/CORS. The Vercel build uses `NEXT_PUBLIC_API_URL=https://faz-ai-llm-forge.hf.space/api/v1`. Keep the direct Neon URL scoped to migration commands, then use the application/pooler URL for runtime. Backend environment changes require a restart; frontend public-variable changes require a rebuild.

Hosted smoke testing exposed a read transaction retained by the Run request while its inline task executed. The route now commits that lookup before returning its response; the worker uses its own session. Do not remove that boundary or hold provider calls inside a transaction. The five-connection cap accommodates initial concurrent detail-page reads without adding infrastructure. If reads time out, inspect pool checkout failures and transaction lifetimes before treating the database as unavailable.

### Rollout verification

For each release, run CI and the disposable PostgreSQL migration/locking checks, drain active hosted work, inspect the current database revision and apply only the intended migration path. A rehearsal is not a data backup: decide how existing data will be protected before migration. Keep additive schemas when rolling back code, and verify historical provenance rather than assigning old aggregates to an inferred attempt.

After deployment, compare frontend `/api/version` and backend `/api/v1/version` against the intended source commit, then check `/health` and `/ready`. With a signed-in test project, use a clearly labeled mock model to check completion, Stop, rerun, historical selection, a baseline pinned to its original attempt, comparisons and complete JSONL export. Confirm that interrupted cases and earlier result/case identities remain unchanged, execution reservations release and logs show no pool/request/execution failures. Remove only the temporary smoke records after checking exports and references. A mock smoke test establishes workflow behavior, not live-provider availability or model quality.

The **19 October 2026 rollout** merged [implementation PR #1](https://github.com/FazlulKarimC/LLM_Forge/pull/1) and [smoke-fix PR #2](https://github.com/FazlulKarimC/LLM_Forge/pull/2). Both hosted services were verified at source commit `75a9c27faa27d8604e47cc24962e29f009fa67fc`, with Neon at migration `r1s2t3u4v5w6`. That release's [CI run](https://github.com/FazlulKarimC/LLM_Forge/actions/runs/39201662925) passed 646 backend tests, six PostgreSQL locking checks, 71 frontend tests and 40 SDK tests on each supported CI Python version, plus contract/lint/type/build and SDK HTTP checks. These are dated verification results, not continuously current counts. The two additive migrations preserved existing records. Signed-in mock smoke checks passed, temporary test records were removed and no hosted infrastructure was added. New hosted RAG execution remains paused, and root dataset deduplication remains deferred.

### Retention and diagnosis

Verify an attempt export before purging it. JSON exports are bounded; use `?attempt=N&format=jsonl` for a complete historical export, including traces, scoring/provider metadata and retention-truncation markers. Active, pinned or referenced attempts are protected. New case storage and project attempt caps are documented in [architecture](ARCHITECTURE.md#benchmarks-and-optional-infrastructure). Idle runner verification and provider-free rerun/export/tenant/gate regressions live in `backend/tests/test_benchmark_attempts.py`. PostgreSQL locking and migration checks require the disposable CI/local PostgreSQL service; SQLite does not prove row locking.

Set frontend build-time Clerk/API URL values and backend database/Clerk/CORS/optional-service values for their respective targets. Use HTTPS for a hosted API root. Provider validity and model availability still require an explicit live check; `INFERENCE_ENGINE` configures the benchmark default, not the new evaluation provider contract.

| Symptom / endpoint | Check |
| --- | --- |
| `/health` passes but work fails | It only checks process liveness; inspect `/ready`, request ID and safe run/attempt status |
| `/ready` fails | Check database connectivity, required schema and the capacity row. Optional Redis/Qdrant/providers and an idle runner never gate readiness |
| Fresh browser 401 | Matching Clerk app, issuer, exact authorized-party origin, token expiry and frontend URL configuration |
| Browser API URL failure | Built `NEXT_PUBLIC_API_URL` includes HTTPS host and `/api/v1`; CORS includes the frontend origin |
| Unexpected provider behavior | Inspect deployed `/api/v1/capabilities`; benchmark and evaluation provider registries differ |
| Slow mock evaluation | Correlate admission, queue wait, claim, fence, checkpoint/finalization timings; missing optional RQ alone does not diagnose evaluation latency |
| Interrupted evaluation | Inspect preserved checkpoints and recovery metadata; explicit live resume needs fresh keys |
| Stranded benchmark | Inspect its attempt and use **Stop run**, then rerun when appropriate; it is a separate execution system |
| Build mismatch | Compare backend `/api/v1/version` and frontend `/api/version` separately; SDK package/source is another identity |

Backend metadata reads embedded source info or explicitly configured non-secret `BUILD_COMMIT` / `BUILD_ID`. Frontend metadata uses Vercel/source build values. Missing/invalid source identity is unknown; it does not establish deployed parity. Never include secrets or environment dumps in release metadata/logs. Optional Sentry reporting uses the existing sanitizers.

Maintain these two guides with code changes. Completed task evidence belongs in Git history; avoid another phase, redesign or audit Markdown file competing with current architecture/development guidance.
