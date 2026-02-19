# LlmForge

> A config-driven experimentation platform for systematically comparing LLM reasoning strategies — Naive Prompting, Chain-of-Thought, RAG, and ReAct Agents — with full metrics tracking and a research-grade dashboard.

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs)](https://nextjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-NeonDB-336791?logo=postgresql&logoColor=white)](https://neon.tech)

---

## Overview

LlmForge lets you design, run, and compare LLM experiments through a web UI or REST API. Each experiment is a combination of a **reasoning method**, **dataset**, **model**, and **hyperparameters** — all version-controlled in a database. After a run completes, the platform computes quality, performance, and cost metrics and surfaces them in an interactive dashboard.

**Built as a personal learning project** to deeply understand how different LLM reasoning strategies trade off accuracy, latency, and token cost on real QA benchmarks.

---

## Features

- **4 reasoning methods** — Naive, Chain-of-Thought, RAG (vector/BM25/hybrid retrieval), ReAct Agent with tools
- **End-to-end pipeline** — create experiment → queue → inference → metrics → results, entirely from the UI
- **Inference Providers API** — uses HuggingFace's Inference Providers (novita) — no local GPU needed
- **Async-safe execution** — all blocking inference calls run in thread pools via `asyncio.to_thread()`, keeping the API responsive during long runs
- **Metrics dashboard** — Accuracy, F1, Latency p50/p95, Throughput, per-run correctness grid
- **Experiment comparison** — side-by-side metric cards across runs
- **Inference optimization** — in-memory prompt caching (LRU), batch execution, per-section profiling
- **ReAct agent** — iterative Thought → Action → Observation loop with Wikipedia search and Calculator tools
- **RAG pipeline** — Qdrant vector store, configurable retrieval (dense, BM25, hybrid), faithfulness scoring

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Next.js 16 Frontend                    │
│  Dashboard │ New Experiment │ Results │ Comparison        │
└───────────────────────────┬──────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                        │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Experiment │  │  Inference   │  │   Evaluation    │  │
│  │  Service    │  │  Engine      │  │   Pipeline      │  │
│  │             │  │              │  │                 │  │
│  │ CRUD + exec │  │ HFAPIEngine  │  │ Accuracy / F1   │  │
│  │ BackgroundT │  │ MockEngine   │  │ Faithfulness    │  │
│  │ asyncio.t_t │  │ Prompts      │  │ Latency p50/p95 │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
└─────────┼────────────────┼────────────────────┼──────────┘
          │                │                    │
          ▼                ▼                    ▼
   ┌────────────┐   ┌─────────────┐   ┌──────────────────┐
   │ PostgreSQL │   │   Qdrant    │   │  HuggingFace     │
   │ (NeonDB)   │   │ (Vector DB) │   │  Inference API   │
   │ Experiments│   │ RAG Chunks  │   │  (novita)        │
   │ Runs       │   └─────────────┘   └──────────────────┘
   │ Results    │
   └────────────┘
```

---

## Reasoning Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Naive** | Direct `Question → Answer` prompting | Factual recall baselines |
| **Chain-of-Thought** | Few-shot examples + `"Let's think step by step"` | Multi-step reasoning, math |
| **RAG** | Retrieve context chunks → inject into prompt | Knowledge-grounded QA |
| **ReAct Agent** | Thought → Action → Observation loop with tools | Multi-hop, tool-requiring tasks |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, SQLAlchemy (async), Alembic, Pydantic v2 |
| **Frontend** | Next.js 16, TypeScript, React 19, shadcn/ui, TanStack Query |
| **Database** | PostgreSQL via NeonDB (serverless free tier) |
| **Vector DB** | Qdrant (for RAG document retrieval) |
| **Inference** | HuggingFace Inference Providers API (novita provider) |
| **Embeddings** | sentence-transformers (CPU, no GPU required) |
| **Task Queue** | FastAPI BackgroundTasks + asyncio thread pool |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [NeonDB](https://neon.tech) PostgreSQL connection string (free tier)
- A [HuggingFace](https://huggingface.co) API token with Inference API access

### 1. Backend

```bash
git clone https://github.com/yourusername/LlmForge.git
cd LlmForge/backend

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file:

```env
DATABASE_URL=postgresql+asyncpg://<user>:<pass>@<host>/neondb
HF_TOKEN=hf_...
INFERENCE_ENGINE=hf_api       # or "mock" for offline testing
HF_PROVIDER=novita            # HF Inference Provider (default: novita)
```

Run migrations and start:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Running an Experiment

### Via the UI

1. Go to **Experiments → New Experiment**
2. Pick a model, reasoning method, and dataset
3. Click **Create & Run** — the experiment runs in the background
4. Results appear on the experiment detail page with live status polling

### Via REST API

```bash
# Create
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cot-vs-naive-trivia",
    "config": {
      "model_name": "meta-llama/Llama-3.2-1B-Instruct",
      "reasoning_method": "cot",
      "dataset_name": "trivia_qa",
      "num_samples": 10,
      "hyperparameters": { "temperature": 0.1, "max_tokens": 512 }
    }
  }'

# Run
curl -X POST http://localhost:8000/api/v1/experiments/{id}/run

# Results
curl http://localhost:8000/api/v1/results/{id}/metrics
```

---

## Datasets

| Dataset | Task | Samples |
|---------|------|---------|
| `sample` | Mixed QA (smoke test) | 5 |
| `trivia_qa` | Open-domain factual QA | 100 |
| `commonsense_qa` | Everyday reasoning | 30 |
| `multi_hop` | Multi-step reasoning | 40 |
| `math_reasoning` | GSM8K-style word problems | 40 |
| `react_bench` | Tool-use evaluation | 20 |
| `knowledge_base` | RAG-focused grounded QA | 50 |

---

## Project Structure

```
LlmForge/
├── backend/
│   ├── app/
│   │   ├── api/            # REST endpoints (experiments, results, health)
│   │   ├── core/           # Config, database, Redis
│   │   ├── models/         # SQLAlchemy ORM (Experiment, Result, Run)
│   │   ├── schemas/        # Pydantic v2 schemas
│   │   └── services/
│   │       ├── inference/  # HFAPIEngine, MockEngine, prompt templates
│   │       ├── rag_service.py      # RAG pipeline + faithfulness scorer
│   │       ├── agent_service.py    # ReAct agent + tools
│   │       ├── experiment_service.py  # Execution orchestrator
│   │       ├── metrics_service.py
│   │       ├── dataset_service.py
│   │       └── optimization.py     # Cache, profiler, batch runner
│   ├── configs/            # cot_examples.json, dataset JSONs
│   └── alembic/            # DB migrations
├── frontend/
│   └── src/
│       ├── app/            # Next.js pages (dashboard, experiments, compare)
│       ├── components/     # Shared UI components
│       └── lib/            # API client (api.ts), type definitions
├── PHASES.md               # Development history by phase
├── DESIGN_SYSTEM.md        # Frontend design guidelines
└── README.md
```

---

## Metrics Collected

| Category | Metrics |
|----------|---------|
| **Quality** | Exact Match, Substring Match, Token-level F1 (with alias support) |
| **Performance** | Latency p50 / p95, Throughput (queries/sec) |
| **Cost** | Total input tokens, output tokens, per-run token breakdown |
| **RAG** | Faithfulness score (NLI-based answer grounding) |
| **Agent** | Tool call count, iteration count, termination reason |

---

## Development Notes

- **`asyncio.to_thread()`** — HuggingFace's `InferenceClient` is synchronous (`requests`-based). Calling it directly in an `async` FastAPI handler blocks the event loop. All inference calls are wrapped in `asyncio.to_thread()` to keep the server responsive.
- **HF Inference Providers** — The old serverless endpoint (`api-inference.huggingface.co/models/`) is deprecated (returns 410). The platform uses the new `InferenceClient(provider="novita")` API. Provider is configurable via `HF_PROVIDER` env var.
- **Mock engine** — Set `INFERENCE_ENGINE=mock` in `.env` to run the full pipeline offline without HF API calls. Useful for frontend development and CI.

---

## Documentation

| File | Description |
|------|-------------|
| [PHASES.md](./PHASES.md) | What was built in each development phase |
| [DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) | Frontend color palette, typography, component patterns |
| `http://localhost:8000/docs` | Auto-generated FastAPI Swagger docs |

---

## License

MIT — See [LICENSE](./LICENSE)

---

<p align="center">
  <i>Config-driven experiments. Reproducible results. No GPU required.</i>
</p>
