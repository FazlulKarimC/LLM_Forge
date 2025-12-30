# LLM Research Platform

> A config-driven experimentation platform for studying LLM reasoning, retrieval, alignment, and inference optimization.

## 🎯 Overview

This platform enables controlled, reproducible experiments on:
- **Reasoning Methods**: Naive, Chain-of-Thought (CoT), ReAct
- **Retrieval Strategies**: No RAG, Naive RAG, Hybrid RAG, Reranked RAG
- **Alignment**: Direct Preference Optimization (DPO)
- **Inference Optimizations**: Batching, KV Cache

## 📊 Metrics

| Category | Metrics |
|----------|---------|
| Quality | Accuracy, Faithfulness, Hallucination Rate |
| Performance | Latency (p50/p95/p99), Throughput |
| Cost | Token count, GPU time |

## 🏗️ Project Structure

```
LlmForge/
├── backend/           # FastAPI (Python 3.11+)
│   ├── app/
│   │   ├── api/       # REST endpoints
│   │   ├── core/      # Config, database
│   │   ├── models/    # SQLAlchemy ORM
│   │   ├── schemas/   # Pydantic schemas
│   │   └── services/  # Business logic & ML inference
│   └── tests/
├── frontend/          # Next.js 15 (TypeScript)
│   └── src/
│       ├── app/       # Page routes
│       └── lib/       # API client, types
├── configs/           # Experiment configs (YAML)
└── docker-compose.yml # Qdrant vector database
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)

### Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env     # Configure your settings
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Start Qdrant (Vector Database)
```bash
docker compose up -d qdrant
```

## 📋 Development Roadmap

### Iteration 1: Runnable Baseline
- [ ] Implement experiment CRUD
- [ ] Basic inference with Phi-2
- [ ] Naive vs CoT comparison
- [ ] Simple metrics display

### Iteration 2: Research-Grade
- [ ] RAG pipeline (naive, hybrid, reranked)
- [ ] Faithfulness evaluation
- [ ] ReAct agent implementation
- [ ] Full metrics dashboard

### Iteration 3: Optimization
- [ ] Batching inference
- [ ] vLLM integration
- [ ] DPO fine-tuning
- [ ] Performance benchmarks

## 🧪 Running Experiments

```bash
# Create experiment via API
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d @configs/reasoning/naive_vs_cot.yaml
```

## 📖 Documentation

- [Project Blueprint](./PROJECT_BLUEPRINT.md) - Full implementation plan
- API Docs: http://localhost:8000/docs (when backend is running)

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Frontend | Next.js 15, TypeScript, Tailwind |
| Database | PostgreSQL (NeonDB) |
| Vector DB | Qdrant |
| ML | Transformers, vLLM |

---

*Config-driven experimentation. Reproducibility by default.*
