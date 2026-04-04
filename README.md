<div align="center">

# LlmForge

**A full-stack experimentation platform for systematically evaluating LLM reasoning strategies**

[![CI](https://github.com/FazlulKarimC/LLM_Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/FazlulKarimC/LLM_Forge/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white)](https://react.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4-38bdf8?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-NeonDB-336791?logo=postgresql&logoColor=white)](https://neon.tech)
[![Redis](https://img.shields.io/badge/Redis-Upstash-dc382d?logo=redis&logoColor=white)](https://upstash.com)

*Compare Naive Prompting, Chain-of-Thought, RAG, and ReAct Agents side-by-side with metrics, statistical significance testing, and a research-grade dashboard.*

</div>

---

## Overview

LlmForge is a config-driven platform for designing, executing, and comparing LLM experiments. Each experiment combines a **reasoning method**, **dataset**, **model**, **inference provider**, and **hyperparameters** -- all version-controlled in a database. After execution, the platform computes quality, performance, and cost metrics, surfacing them through an interactive dashboard with per-sample inspection, latency distributions, and side-by-side statistical comparison.

Built to answer a core question: *How do different LLM reasoning strategies trade off accuracy, latency, and token cost on real QA benchmarks?*

---

## Key Features

### Four Reasoning Strategies
- **Naive Prompting** -- Direct question-answer with zero-shot or few-shot templates
- **Chain-of-Thought (CoT)** -- Step-by-step reasoning elicitation before the final answer
- **RAG** -- Dense, hybrid, or reranked retrieval-augmented generation over indexed knowledge bases
- **ReAct Agent** -- Dynamic multi-step tool calling (Wikipedia search, Calculator, Retrieval) with observation-action traces

### Adaptive Multi-Provider Routing
Route experiments through **HuggingFace Inference API**, **OpenRouter**, **Groq**, or any **OpenAI-compatible endpoint**. An epsilon-greedy **Adaptive Provider Router** handles provider selection based on active policies (`cheapest_first`, `fastest_first`, or `adaptive` via composite score) and tracks routing telemetry.

### Trajectory Regression Gates
Pin completed experiments as **Baselines**. New candidate runs are automatically evaluated against pinned baselines using a deterministic 7-rule **Grader Engine** (max turns, required tools, token/latency budgets, F1 score). Prevent prompt or model regressions before they happen with clear pass/fail/skip verdicts and inline configuration diffing.

### Comprehensive Metrics and Evaluation

| Category | Metrics |
|----------|---------|
| **Accuracy** | Exact match, substring match, F1 score |
| **Latency** | p50, p95, p99, per-sample histogram (10-bucket distribution) |
| **Cost** | Token usage breakdown, estimated USD, cost-per-correct-answer |
| **Quality** | LLM-as-Judge scoring (coherence, helpfulness, factuality) -- budget-capped |
| **Safety** | Robustness score against prompt injection, jailbreak, and edge-case datasets |
| **Statistical** | Bootstrap 95% CI, McNemar's chi-squared test, pass@k, multi-trial variance |

### Side-by-Side Comparison Workspace
Compare any two experiments with:
- McNemar's chi-squared test for paired statistical significance
- Bootstrap confidence intervals on accuracy deltas
- Agreement / disagreement distribution bars
- Per-example output diffs with correctness annotations

### Filmstrip Evaluator
A colour-coded correctness grid at the per-sample level. Click any cell to inspect the full prompt, model output, expected answer, retrieved context chunks (RAG), and complete agent traces (ReAct).

### Safety and Robustness Testing
Three adversarial datasets (`prompt_injection`, `jailbreak`, `edge_cases`) with a deterministic rule-based robustness scorer -- no additional API calls required.

### Inference Optimization
- **Batch Execution** -- Parallelized API calls via thread pools with per-phase profiling
- **Prompt Caching** -- LRU cache for deterministic runs to avoid redundant API calls
- **Rate Limiting and Retry** -- Exponential backoff with jitter, global concurrency gating

### Export and Reporting
Download results as **JSON** or a formatted **Markdown report** directly from the UI. Reports include configuration, metrics summary, and per-run correctness tables.

---

## Architecture

```
Frontend (Next.js 16)
  Landing | Dashboard | Experiment Builder | Detail | Comparison
  Framer Motion, TanStack Query, Sonner, Lucide, Dark Theme
                        |
                    REST API
                        |
Backend (FastAPI)
  +----------------+  +---------------+  +----------------------+
  | Experiment     |  | Inference     |  | Evaluation Pipeline  |
  | Service        |  | Engines       |  |                      |
  |                |  | Adaptive Route|  | Metrics, LLM Judge   |
  | CRUD, Jobs     |  | HFAPIEngine   |  | Regression Gates     |
  | Batch/Cache    |  | OpenAIEngine  |  | Robustness, Stats    |
  | Profiling      |  | MockEngine    |  | Cost, Safety         |
  +-------+--------+  +------+--------+  +--------+-------------+
          |                   |                    |
  +-------v-------------------v--------------------v-------------+
  | Rate Limit, Retry, Pricing, Background Jobs, Sentry          |
  +------+----------+----------+--------------+----------+-------+
         |          |          |              |           |
         v          v          v              v           v
    PostgreSQL  Upstash    Qdrant       HF Inference  OpenRouter/
     (NeonDB)  Redis+RQ  (Vectors)        API       Groq/Custom
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, FastAPI, SQLAlchemy (async), Alembic, Pydantic v2, statsmodels, NumPy |
| **Frontend** | Next.js 16, TypeScript, React 19, Tailwind CSS v4, Framer Motion, TanStack Query, Sonner, Lucide |
| **Database** | PostgreSQL via NeonDB (serverless) |
| **Vector Store** | Qdrant Cloud (RAG document retrieval) |
| **Task Queue** | Durable Postgres-backed background jobs, Upstash Redis + RQ |
| **Inference** | HuggingFace Inference API, OpenRouter, Groq, OpenAI-compatible endpoints |
| **Observability**| Sentry (Full-stack distributed error tracking) |
| **Embeddings** | sentence-transformers (CPU-friendly) |
| **CI/CD** | GitHub Actions |

---

## Frontend Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page -- animated hero, live comparison preview, feature overview |
| `/dashboard` | Operational dashboard -- KPI cards, system readiness checks, experiment queue with inline actions |
| `/experiments` | Experiment catalog -- filterable list with baseline pinning, regression status pills, and run/delete controls |
| `/experiments/new` | Experiment builder -- model/dataset/provider selectors, preset templates, complexity indicator |
| `/experiments/[id]` | Experiment detail -- lifecycle metadata, progressively-loaded metrics, regression/routing panels, filmstrip evaluator, latency histogram, run profiler, export |
| `/experiments/compare` | Comparison workspace -- metric deltas, statistical significance, agreement bars, per-example diffs |

---

## Design System

The UI follows a custom **dark-first editorial tech** design language:

- **OKLCH colour palette** with semantic tokens for consistent theming
- **Glass morphism** surfaces with `color-mix` tints and layered depth
- **Framer Motion** micro-animations on metrics, page transitions, and data renders
- **Reusable component library** -- `PageHeader`, `Panel`, `MetricCard`, `StatusPill`, `AnimatedNumber`, `MetricBar`, `EmptyState`, `SkeletonBlock`
- **Collapsible sidebar** with persistent state and theme toggle (dark / light)
- **Accessible** -- focus rings, keyboard navigation, colour-independent status icons, screen-reader utilities

Full reference: [`DESIGN_SYSTEM.md`](./DESIGN_SYSTEM.md)

---

## Supported Datasets

| Dataset | Category | Description |
|---------|----------|-------------|
| `sample` | Smoke Test | Built-in mixed QA for quick validation |
| `trivia_qa` | Factual QA | Single-hop open-domain factual recall |
| `commonsense_qa` | Reasoning | Everyday logic and commonsense reasoning |
| `multi_hop` | Reasoning | Composite multi-fact bridging questions |
| `math_reasoning` | Math | GSM8K-style word problems |
| `react_bench` | Agent | Tool-use questions requiring search + calculation |
| `knowledge_base` | RAG | Grounded QA answerable from indexed articles |
| `prompt_injection` | Safety | Tests instruction override resistance |
| `jailbreak` | Safety | Tests DAN-style jailbreak resistance |
| `edge_cases` | Safety | Tests unusual or malformed inputs |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [NeonDB](https://neon.tech) PostgreSQL connection string (free tier)
- [Upstash](https://upstash.com) Redis connection string (free tier -- production task queue)
- Inference API token (HuggingFace, OpenRouter, Groq, or custom endpoint)

### Backend

```bash
git clone https://github.com/FazlulKarimC/LLM_Forge.git
cd LLM_Forge/backend

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Create `.env` in `/backend`:

```env
DATABASE_URL=postgresql+asyncpg://<user>:<pass>@<host>/neondb
HF_TOKEN=hf_...
INFERENCE_ENGINE=hf_api
HF_PROVIDER=novita
REDIS_URL=redis://...         # Upstash Redis URL
ENVIRONMENT=production        # "development" to skip Redis
```

Run migrations and start the server:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open **http://localhost:3000**

---

## Running an Experiment

### Via the UI

1. Navigate to **Experiments > New Experiment**
2. Select a reasoning strategy (Naive / CoT / ReAct), dataset, model, and inference provider
3. Optionally choose a **preset configuration** or provide a custom LLM endpoint
4. Enable **Batching** or **Caching** under the optimization section
5. Click **Create and Run** -- the detail page auto-polls until execution completes
6. Inspect results via the **filmstrip evaluator**, **metrics cards**, and **latency histogram**
7. Navigate to the **Comparison workspace** to run a statistical A/B test against another experiment
8. **Export** results as JSON or Markdown

### Via the API

```bash
# Create
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cot_vs_naive_multihop",
    "config": {
      "model_name": "meta-llama/Llama-3.2-1B-Instruct",
      "reasoning_method": "cot",
      "dataset_name": "multi_hop",
      "provider": "auto",
      "num_samples": 20,
      "hyperparameters": { "temperature": 0.1, "max_tokens": 512 }
    }
  }'

# Run
curl -X POST "http://localhost:8000/api/v1/experiments/{id}/run"

# Metrics
curl http://localhost:8000/api/v1/results/{id}/metrics

# Statistical comparison
curl "http://localhost:8000/api/v1/results/compare/statistical?experiment_a={id_a}&experiment_b={id_b}"

# Export
curl http://localhost:8000/api/v1/results/{id}/export
```

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

Test coverage spans: API routes, experiment lifecycle, metrics computation, prompting strategies, RAG retrieval, agent execution, optimization profiling, statistical comparison, and integration tests.

---

## License

MIT -- See [LICENSE](./LICENSE)

---

<p align="center">
  <b>Config-driven experiments. Multi-provider inference. Statistical rigor.</b>
</p>
