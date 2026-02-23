# LlmForge

> A config-driven experimentation platform for systematically comparing LLM reasoning strategies — Naive Prompting, Chain-of-Thought, RAG, and ReAct Agents — with full metrics tracking, a research-grade dashboard, and support for both HuggingFace and Custom Hosted OpenAI-compatible models.

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs)](https://nextjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-NeonDB-336791?logo=postgresql&logoColor=white)](https://neon.tech)

---

## Overview

LlmForge lets you design, run, and compare LLM experiments through a polished web UI or REST API. Each experiment is a combination of a **reasoning method**, **dataset**, **model**, **target endpoint**, and **hyperparameters** — all version-controlled in a database. After a run completes, the platform computes quality, performance, and cost metrics, surfacing them in an interactive, clickable dashboard.

Built to deeply understand how different LLM reasoning strategies trade off accuracy, latency, and token cost on real QA benchmarks.

---

## Features

- **4 Reasoning Methods** — Naive, Chain-of-Thought, RAG (vector/BM25/hybrid retrieval), and ReAct Agent with dynamic tool calling (Wikipedia / Calculator).
- **End-to-end Pipeline** — Create, queue, execute, and evaluate experiments entirely from a sleek, glassmorphic UI featuring auto-polling and unified global navigation.
- **Flexible Inference Engines** — Run experiments against **HuggingFace Inference Providers** (hosted serverless) *or* **Custom Hosted LLMs** (any OpenAI-compatible endpoint like local vLLM, Ollama, or third-party providers).
- **Inference Optimization** — Built-in support for **Batch Execution** (parallel API calls via thread pools) and **Prompt Caching** (LRU cache for identical runs) to drastically reduce api latency and cost.
- **Async-Safe Execution** — All blocking inference calls run reliably in thread pools via `asyncio.to_thread()`, keeping the FASTAPI event loop buttery smooth during heavy loads.
- **Robust Error Handling** — Global exception middleware handles validation errors seamlessly, assigning unique Request IDs and gracefully rendering user-friendly alerts.
- **Metrics Dashboard** — Compare Accuracy, Token-level F1, Latency (p50/p95), Throughput, and view a per-run correctness grid for granular output analysis.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Next.js 16 Frontend                   │
│  Dashboard │ New Experiment │ Results │ Comparison       │
└───────────────────────────┬──────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Experiment  │  │  Inference   │  │   Evaluation    │  │
│  │ Service     │  │  Engines     │  │   Pipeline      │  │
│  │             │  │              │  │                 │  │
│  │ CRUD + Queue│  │ HFAPIEngine  │  │ Accuracy / F1   │  │
│  │ Async/Batch │  │ OpenAIEngine │  │ Faithfulness    │  │
│  │ Cache/Prof  │  │ MockEngine   │  │ Latency p50/p95 │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
└─────────┼────────────────┼────────────────────┼──────────┘
          │                │                    │
          ▼                ▼                    ▼
   ┌────────────┐   ┌─────────────┐   ┌──────────────────┐
   │ PostgreSQL │   │   Qdrant    │   │  HuggingFace API │
   │ (NeonDB)   │   │ (Vector DB) │   │  / Custom Models │
   └────────────┘   └─────────────┘   └──────────────────┘
```

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, SQLAlchemy (async), Alembic, Pydantic v2 |
| **Frontend** | Next.js 16, TypeScript, React 19, Tailwind v4, shadcn/ui, TanStack Query |
| **Database** | PostgreSQL via NeonDB (serverless free tier) |
| **Vector DB** | Qdrant (for RAG document retrieval) |
| **Inference** | HuggingFace Inference API (novita) **OR** OpenAI-compatible APIs |
| **Embeddings** | sentence-transformers (CPU-friendly) |
| **Task Queue** | FastAPI BackgroundTasks + `asyncio` ThreadPoolExecutors |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [NeonDB](https://neon.tech) PostgreSQL connection string (free tier)
- An Inference API token (HuggingFace token OR any Custom LLM provider key)

### 1. Backend Setup

```bash
git clone https://github.com/yourusername/LlmForge.git
cd LlmForge/backend

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file in the `/backend` directory:

```env
DATABASE_URL=postgresql+asyncpg://<user>:<pass>@<host>/neondb
HF_TOKEN=hf_...
INFERENCE_ENGINE=hf_api       # OR "mock" for offline frontend dev
HF_PROVIDER=novita            # HF Inference Provider
```

Run migrations and start the server:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd ../frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Running an Experiment

### Via the User Interface

1. Navigate to **Experiments → New Experiment** via the global navbar.
2. Select your reasoning strategy: **Naive, CoT, RAG, or ReAct**.
3. *[Optional]* Provide a **Custom Base URL & API Key** if you are benchmarking an external LLM (e.g., local Ollama, vLLM instance).
4. *[Optional]* Enable **Batching** or **Caching** in the Optimization section for rapid datasets.
5. Click **Create & Run**. The UI will auto-poll backend background tasks.
6. Once complete, click the row to dive into the Results grid and Optimization timings.

### Via REST API

```bash
# Create Custom Endpoint Experiment
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_llm_benchmark",
    "config": {
      "model_name": "llama-3-8b",
      "reasoning_method": "naive",
      "dataset_name": "trivia_qa",
      "num_samples": 10,
      "hyperparameters": { "temperature": 0.1, "max_tokens": 512 }
    }
  }'

# Run directly pointing to a custom URL
curl -X POST "http://localhost:8000/api/v1/experiments/{id}/run?custom_base_url=http://localhost:11434/v1&custom_api_key=sk-123"

# Fetch Computed Results & Optimization Profile
curl http://localhost:8000/api/v1/results/{id}/metrics
```

---

## Supported Datasets

| Dataset | Task Category | Samples |
|---------|---------------|---------|
| `sample` | Mixed QA (Smoke Testing) | 5 |
| `trivia_qa` | Open-domain Factual QA | 100 |
| `commonsense_qa` | Everyday Logic & Reasoning | 30 |
| `multi_hop` | Composite/Bridged Reasoning | 40 |
| `math_reasoning` | GSM8K-style Word Problems | 40 |
| `react_bench` | Tool-use and Iterative Evaluation | 20 |
| `knowledge_base` | RAG-focused Grounded QA | 50 |

---

## Internal Code Documentation

- **[DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md)** — Frontend component UI rules, Tailwind v4 specs, 4-color unified palette.
- **[OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md)** — Core architectures regarding `PromptCache`, batch `ThreadPoolExecutor` parallelization, and `ProfilerContext` bottlenecks.
- **[PHASES.md](./PHASES.md)** — Historical context log of the project's entire build genesis.
- **Swagger Docs:** `http://localhost:8000/docs`

---

## License

MIT — See [LICENSE](./LICENSE)

---
<p align="center">
  <i>Config-driven experiments. Built for scale. Deep insights.</i>
</p>
