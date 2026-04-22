# LlmForge - Backend Services

> The FastAPI inference and evaluation engine for LlmForge.

This directory contains the Python backend that drives LlmForge. It acts as the core controller for executing experiments, handling multi-provider LLM routing, orchestrating trajectory regression gates, and compiling the statistical metrics needed for evaluation.

---

## 🛠️ Technology Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+)
- **Database:** PostgreSQL via [NeonDB](https://neon.tech/) (Async SQLAlchemy + Alembic)
- **Validation:** Pydantic v2
- **Vector DB:** Qdrant Cloud (for RAG and embeddings retrieval)
- **Task Execution:** Postgres-backed experiment/job state + optional Upstash Redis (RQ) acceleration
- **Observability:** Sentry for distributed tracing and crash reporting
- **Math/Stats:** NumPy, statsmodels (for Bootstrap CIs, McNemar's tests)

---

## ⚙️ Core Modules

These robust systems handle the complex workflows of systematic evaluation:

### Adaptive Multi-Provider Engine
Provides epsilon-greedy auto-routing across `HF Inference API`, `OpenRouter`, `Groq`, and any `OpenAI-compatible endpoints`. The `ProviderStatsTracker` tracks success rates, latency, and costs to continuously refine routing strategies based on chosen policies (e.g., `cheapest_first`, `fastest_first`).

### Trajectory Regression Gates
Each candidate run is passed through a comprehensive `GraderEngine` employing deterministic bounds checks such as specific token/latency budgets, explicit tool dependencies, or hard F1-score floors. The system isolates and flags regressions against pinned baseline experiments to ensure deployment safety.

### Reliability & Error Tracking
To support execution in constrained serverless setups, durable state for experiments and background-job metadata is stored in Postgres. Actual execution is still best-effort: experiments run inline via FastAPI `BackgroundTasks` unless RQ is available, and interrupted in-process jobs are marked failed on restart rather than resumed automatically. The latest integration funnels failure stack traces down to Sentry. RAG experiments include intelligent preflight collections checks to fail fast gracefully.

---

## 📁 Project Structure

```text
backend/
├── alembic/            # Database schema migrations
├── app/
│   ├── api/            # API routers / view endpoints (experiments, results)
│   ├── core/           # Configuration, exception handlers, background_jobs
│   ├── models/         # SQLAlchemy ORM definitions
│   ├── schemas/        # Pydantic validation DTOs (slim & full patterns)
│   ├── services/       # Core domain logic
│   │   ├── inference/  # Provider handlers and the Adaptive Router
│   │   └── grader_service.py # Evaluation gates & heuristics
│   └── main.py         # App factory & Sentry initialization
├── tests/              # Extensive Pytest validation suite
└── requirements.txt    
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A valid PostgreSQL URI string (`DATABASE_URL`)
- Desirable Inference API keys (`HF_TOKEN`, `OPENROUTER_API_KEY`, etc)

### Installation

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate 
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Database & Launch

Prepare local or dev DB tables and run the server:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

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

The worker is optional. Without it, experiments run inline via FastAPI `BackgroundTasks`. This fallback is resilient for free-tier hosting, but it is not a durable queue: if the API process restarts mid-job, the in-flight work is lost and the corresponding records are marked failed on startup.
