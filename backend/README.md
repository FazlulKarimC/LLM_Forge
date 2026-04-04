# LlmForge - Backend Services

> The FastAPI inference and evaluation engine for LlmForge.

This directory contains the Python backend that drives LlmForge. It acts as the core controller for executing experiments, handling multi-provider LLM routing, orchestrating trajectory regression gates, and compiling the statistical metrics needed for evaluation.

---

## 🛠️ Technology Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+)
- **Database:** PostgreSQL via [NeonDB](https://neon.tech/) (Async SQLAlchemy + Alembic)
- **Validation:** Pydantic v2
- **Vector DB:** Qdrant Cloud (for RAG and embeddings retrieval)
- **Task Queue:** Durable Postgres-backed custom queue + Upstash Redis (RQ)
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
To support execution in potentially constrained serverless setups, backend queues are now durable via Postgres caching to resume natively without data-loss across cold starts. The latest integration seamlessly funnels failure stack traces down to Sentry. RAG experiments include intelligent preflight collections checks to fail fast gracefully.

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
