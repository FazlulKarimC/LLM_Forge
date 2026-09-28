# LlmForge backend

FastAPI API for authenticated projects, versioned prompts and datasets, evaluations, and the legacy reasoning-benchmark workflow.

The prompt workflow saves immutable versions, runs fixed cases with assertions or an optional judge, and exposes released prompts to the Python SDK. The experiment workflow runs reasoning methods through configured providers and records per-sample results, routing, cost, metrics, and regression checks.

---

## 🛠️ Technology Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12 in CI)
- **Database:** PostgreSQL via [NeonDB](https://neon.tech/) (Async SQLAlchemy + Alembic)
- **Validation:** Pydantic v2
- **Vector DB:** Qdrant Cloud (for RAG and embeddings retrieval)
- **Task Execution:** PostgreSQL-backed state, FastAPI `BackgroundTasks` for best-effort inline execution, optional Upstash Redis + RQ worker
- **Observability:** optional Sentry error reporting and request IDs
- **Math/Stats:** NumPy, statsmodels (for Bootstrap CIs, McNemar's tests)

---

## ⚙️ Core Modules

These robust systems handle the complex workflows of systematic evaluation:

### Adaptive Multi-Provider Engine
Provides epsilon-greedy auto-routing across `HF Inference API`, `OpenRouter`, `Groq`, and any `OpenAI-compatible endpoints`. The `ProviderStatsTracker` tracks success rates, latency, and costs to continuously refine routing strategies based on chosen policies (e.g., `cheapest_first`, `fastest_first`).

### Trajectory Regression Gates
Each candidate run is passed through a comprehensive `GraderEngine` employing deterministic bounds checks such as specific token/latency budgets, explicit tool dependencies, or hard F1-score floors. The system isolates and flags regressions against pinned baseline experiments to ensure deployment safety.

### Reliability & Error Tracking
Experiment and job metadata is stored in PostgreSQL. Inline execution is best-effort: a backend restart can leave a benchmark queued or running. Startup does not change a run another process might own. Use **Stop run** on its detail page, or `POST /api/v1/experiments/{id}/interrupt`, to mark it failed and keep partial results, then rerun it. Dispatched jobs include an attempt number so an older queued job cannot take over a rerun; pre-upgrade RQ jobs without an attempt number are ignored and may need **Stop run**. A running worker checks for interruption between examples; an in-flight provider call may finish. Final metrics and completion are committed under the same experiment lock. Evaluation runs are marked failed when their next authorized read finds 120 seconds without progress. RAG experiments preflight collections before running.

---

## 📁 Project Structure

```text
backend/
├── alembic/            # Database schema migrations
├── app/
│   ├── api/            # Workspace, prompt, dataset, evaluation, benchmark, and health routes
│   ├── core/           # Configuration, auth/tenancy, task dispatch, and error handling
│   ├── models/         # SQLAlchemy ORM definitions
│   ├── schemas/        # Pydantic validation DTOs (slim & full patterns)
│   ├── services/       # Domain logic and benchmark execution
│   │   ├── inference/  # Provider handlers and the Adaptive Router
│   │   └── grader_service.py # Evaluation gates & heuristics
│   └── main.py         # App factory & Sentry initialization
├── tests/              # API, service, dispatch, and regression tests
├── constraints.txt     # Versions pinned for repeatable CI/deployment builds
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (the version used by backend CI)
- A valid PostgreSQL URI string (`DATABASE_URL`)
- Clerk issuer and authorized frontend origin for authenticated routes
- Provider keys only for live inference; `INFERENCE_ENGINE=mock` supports a provider-free demo

### Installation

```bash
python -m venv venv
```

Activate it with `.\venv\Scripts\Activate.ps1` in PowerShell or `source venv/bin/activate` on macOS/Linux, then install dependencies:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

Copy `.env.example` to `.env` and set `DATABASE_URL`, `CLERK_ISSUER_URL`, and `CLERK_AUTHORIZED_PARTIES`. The frontend needs the same Clerk application. Set `INFERENCE_ENGINE=mock` for the sample evaluation and local checks. Redis, Qdrant, provider credentials, and Sentry are optional for the workflows that use them. Do not point local tests at production data.

### Database & Launch

Prepare local or dev DB tables and run the server:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

`GET /health` reports process liveness. `GET /ready` checks the database and task dispatch, with optional provider, vector-store, Redis, and worker status shown separately. The dashboard surfaces the readiness result. Sign in through the frontend and select a project before calling project-scoped API routes; they require a Clerk bearer token and `X-Project-ID`.

Apply migrations to the intended deployment database before starting a new
backend image. CI applies migrations only to its disposable PostgreSQL service;
it does not migrate the hosted database. Review historical migrations before
using a database with existing data: the workspace reset migration
`i2j3k4l5m6n7` intentionally truncates legacy benchmark tables.

---

## 🔄 Task Dispatch & Queue Architecture

LlmForge uses a resilient task dispatch system designed for free-tier hosting:

- **Neon/Postgres is required** — all durable state (experiments, results, job metadata) lives here
- **Upstash Redis + RQ worker is optional** — provides background execution acceleration
- **`QUEUE_BACKEND_MODE=auto`** is the recommended production setting

### How `auto` mode works

1. If `REDIS_URL` is not set → inline execution
2. If the Upstash circuit breaker is open → inline execution
3. If Redis health probe fails → circuit opens for 30 minutes, inline execution
4. If no RQ worker heartbeat exists (within 90s) → inline execution
5. Otherwise → enqueue via RQ for background processing

### Circuit Breaker

The circuit breaker protects against repeated calls to archived/dead Upstash instances:

- **Opens for 30 minutes** on: `ConnectionError`, `TimeoutError`, `AuthenticationError`, or "archived/gone" style responses
- **Does NOT open** for application bugs (bad arguments, serialization errors)
- **Half-open probe** after 30 minutes: one test request allowed; success closes the circuit

### Worker Heartbeat

The RQ worker writes a heartbeat to the `worker_heartbeats` table every 30 seconds. The API checks this before enqueuing work. If no fresh heartbeat exists, dispatch goes inline immediately — avoiding the silent failure where Redis is up but no worker is processing jobs.

### Running the Worker (optional)

```bash
python worker.py
```

The worker is optional. Without it, experiments run inline via FastAPI `BackgroundTasks`. This fallback avoids a Redis dependency, but it is not a durable queue: if the API process restarts mid-job, the in-flight work is lost. Use the explicit **Stop run** action to release a stranded status; a worker still running will stop before its next example, while an in-flight call may finish.

## Tests and CI

From the repository root, install `backend/requirements-dev.txt` and the SDK test extra, then run `python -m pytest backend/tests sdk/python/tests -q` with mock inference and a non-production `DATABASE_URL`. The isolated API fixtures use SQLite; the [testing guide](../docs/TESTING.md) has the exact commands. GitHub Actions additionally applies migrations to PostgreSQL and runs a real HTTP/SDK smoke test. The local provider-free demo is documented in the [evaluation walkthrough](../docs/PHASE_3_EVALUATIONS.md).
