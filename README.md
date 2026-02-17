# LlmForge

> **A config-driven experimentation platform for systematically comparing LLM reasoning strategies, retrieval methods, and alignment techniques.**

---

## Why This Project?

Large Language Models can solve problems through multiple strategies—direct answering, step-by-step reasoning, retrieving context, or using tools. But **which approach works best, and at what cost?**

Most LLM applications choose a single strategy without rigorous comparison. This platform enables **controlled, reproducible experiments** to measure the trade-offs between accuracy, latency, cost, and reliability across different methods.

### The Research Gap

| Problem | What's Missing |
|---------|----------------|
| Reasoning methods are chosen arbitrarily | No side-by-side comparison on same dataset |
| RAG is assumed to always help | Hallucination rates rarely measured |
| Agents are expensive but "better" | Cost-benefit analysis missing |
| Alignment improves helpfulness | Factuality trade-off undocumented |

**LlmForge fills this gap** by providing infrastructure to run experiments, collect metrics, and generate evidence-backed findings.

---

## Research Questions

This platform investigates four core hypotheses:

| # | Hypothesis | Method |
|---|------------|--------|
| **H1** | Chain-of-Thought improves accuracy on reasoning tasks at the cost of higher latency | Compare Naive vs CoT on TriviaQA/HotpotQA |
| **H2** | Reranking in RAG reduces hallucinations compared to naive retrieval | Measure faithfulness across RAG variants |
| **H3** | Tool-using agents achieve higher accuracy but at 3-5× token cost | Compare ReAct vs CoT vs RAG on multi-hop QA |
| **H4** | DPO alignment improves helpfulness but reduces factual accuracy | Evaluate base vs fine-tuned on both metrics |

### What We Measure

| Category | Metrics |
|----------|---------|
| **Quality** | Exact Match Accuracy, F1 Score, Faithfulness (NLI-based) |
| **Performance** | Latency (p50/p95/p99), Throughput (queries/sec) |
| **Cost** | Input/Output tokens, GPU time, Memory usage |
| **Reliability** | Hallucination rate, Tool failure rate, Loop detection |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│   Dashboard │ Experiment Creation │ Results Comparison │ Traces  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend (FastAPI)                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Experiment    │   Inference     │      Evaluation             │
│   Service       │   Engine        │      Pipeline               │
│                 │                 │                             │
│ • CRUD ops      │ • TransformersEngine  │ • Accuracy metrics   │
│ • Config mgmt   │ • vLLM Engine   │ • Faithfulness (NLI)       │
│ • Run logging   │ • Prompt strategies   │ • Latency percentiles │
└────────┬────────┴────────┬────────┴────────┬────────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PostgreSQL  │    │   Qdrant    │    │  HuggingFace│
│ (NeonDB)    │    │ (Vectors)   │    │  Hub        │
│ Experiments │    │ RAG Chunks  │    │ Models/Data │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Layer Responsibilities

| Layer | Purpose |
|-------|---------|
| **Frontend** | Create experiments, visualize results, compare methods |
| **API** | REST endpoints for experiments, runs, metrics |
| **Services** | Business logic: experiment execution, inference, evaluation |
| **Inference** | Model loading, prompt formatting, generation |
| **RAG** | Document ingestion, embedding, retrieval (dense + BM25) |
| **Agent** | ReAct loop, tool execution, trace logging |
| **Evaluation** | Accuracy, faithfulness, latency computation |

---

## Methodology

### Models

| Environment | Model | Parameters | Use Case |
|-------------|-------|------------|----------|
| Local (GTX 1650) | Phi-2 | 2.7B | Development, Phase 1-3 |
| Colab (T4) | Mistral-7B-Instruct | 7B | Research experiments, DPO |

### Baselines

| Strategy | Description |
|----------|-------------|
| **Naive** | Direct prompting: `Question: {q}\nAnswer:` |
| **Chain-of-Thought** | Add "Let's think step by step" trigger |
| **Naive RAG** | Top-5 vector retrieval → concatenate → generate |
| **Hybrid RAG** | Dense + BM25 retrieval → merge top-5 |
| **Reranked RAG** | Hybrid → cross-encoder rerank → top-5 |
| **ReAct Agent** | Thought/Action/Observation loop with tools |

### Datasets

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| TriviaQA | Factual QA | 100-500 samples | HuggingFace |
| HotpotQA | Multi-hop reasoning | 100-500 samples | HuggingFace |
| Simple Wikipedia | RAG knowledge base | ~200K articles | HuggingFace |
| Anthropic HH-RLHF | DPO training | 800 train / 100 test | HuggingFace |

### Why These Choices?

- **Phi-2**: Fits in 4GB VRAM, sufficient for methodology validation
- **TriviaQA**: Standard factual QA benchmark with clear ground truth
- **Simple Wikipedia**: Smaller than full Wikipedia, faster to embed
- **Cross-encoder reranking**: Industry standard for RAG quality improvement

---

## Expected Results

> ⚠️ **Note**: These are projected outcomes. Actual results will be documented as experiments complete.

### Hypothesis 1: CoT vs Naive

| Method | Accuracy | Latency p50 | Tokens |
|--------|----------|-------------|--------|
| Naive | ~42% | ~280ms | ~5K |
| CoT | ~58% | ~450ms | ~13K |

**Expected Finding**: CoT improves accuracy by ~16% on reasoning tasks, with 60% latency increase.

### Hypothesis 2: RAG Variants

| Method | Accuracy | Faithfulness | Hallucination Rate |
|--------|----------|--------------|-------------------|
| No RAG | ~45% | N/A | ~35% |
| Naive RAG | ~52% | 0.65 | ~28% |
| Hybrid RAG | ~55% | 0.72 | ~22% |
| Reranked RAG | ~58% | 0.87 | ~12% |

**Expected Finding**: Reranking reduces hallucinations by ~18% compared to naive retrieval.

### Hypothesis 3: Agent Cost-Benefit

| Method | Accuracy | Tokens | Cost Multiplier |
|--------|----------|--------|-----------------|
| CoT | ~58% | ~13K | 1× |
| RAG | ~55% | ~18K | 1.4× |
| ReAct Agent | ~65% | ~52K | 4× |

**Expected Finding**: Agents achieve +7-12% accuracy but at 4× token cost.

### Hypothesis 4: Alignment Trade-off

| Model | Helpfulness | Factuality |
|-------|-------------|------------|
| Base Mistral-7B | Baseline | ~62% |
| DPO-tuned | +15% preferred | ~56% |

**Expected Finding**: DPO improves helpfulness by 15% but reduces factuality by 6%.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- NVIDIA GPU with 4GB+ VRAM (or CPU fallback)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/yourusername/LlmForge.git
cd LlmForge/backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your NeonDB connection string

# Run backend
uvicorn app.main:app --reload
```

### 2. Setup Frontend

```bash
cd ../frontend
npm install
npm run dev
```

### 3. Start Vector Database

```bash
cd ..
docker compose up -d qdrant
```

### 4. Verify Installation

- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:3000
- Qdrant: http://localhost:6333/dashboard

---

## Running Experiments

### Create Experiment (API)

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "naive_vs_cot_trivia",
    "method": "chain_of_thought",
    "model": "microsoft/phi-2",
    "dataset": "trivia_qa",
    "num_samples": 100,
    "seed": 42
  }'
```

### Run Experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiments/{id}/run
```

### View Results

Navigate to `http://localhost:3000/experiments/{id}` to see:
- Accuracy, latency, token metrics
- Per-example breakdown (correct/incorrect)
- Comparison with other experiments

---

## Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Config-driven experiments** | Reproducibility, version control | More setup for simple tests |
| **Phi-2 for local dev** | Fits in 4GB VRAM | Limited capability vs 7B+ |
| **PostgreSQL over SQLite** | Production-ready, NeonDB free tier | More complex setup |
| **Qdrant over pgvector** | Better vector search performance | Additional service to manage |
| **Float16 inference** | Fits larger models in VRAM | Minor precision loss |
| **Automatic CPU fallback** | Graceful degradation | Slower inference |

---

## Limitations

### Current

- **Single model at a time**: No parallel model comparison (memory constraint)
- **Local GPU limited**: 4GB VRAM restricts model size
- **Colab session limits**: Larger experiments require session management
- **No production deployment yet**: Focus on research, not serving

### Methodological

- **Dataset size**: 100-500 samples may not capture full distribution
- **Single seed**: Results may vary with different random seeds
- **English only**: No multilingual evaluation
- **Simplified faithfulness**: NLI-based metric, not human judgment

---

## Roadmap

### Phase 1-3: Foundation (Current)
- [x] Project scaffold
- [x] Database CRUD operations
- [x] Basic inference with Phi-2
- [ ] Evaluation metrics pipeline

### Phase 4-6: Research Core
- [ ] Chain-of-Thought implementation
- [ ] RAG pipeline (3 variants)
- [ ] ReAct agent with tools

### Phase 7-8: Advanced
- [ ] DPO fine-tuning (Colab)
- [ ] Inference optimization (vLLM, batching)

### Phase 9: Polish
- [ ] Comprehensive README with results
- [ ] Demo video
- [ ] Deployment (Vercel + Railway)

---

## Project Structure

```
LlmForge/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # REST endpoints
│   │   ├── core/           # Config, database
│   │   ├── models/         # SQLAlchemy ORM
│   │   ├── schemas/        # Pydantic validation
│   │   └── services/       # Business logic
│   │       ├── inference/  # LLM engines
│   │       ├── rag/        # Retrieval pipeline
│   │       ├── agent/      # ReAct implementation
│   │       └── evaluation/ # Metrics computation
│   └── tests/
├── frontend/                # Next.js 15 dashboard
│   └── src/
│       ├── app/            # Page routes
│       ├── components/     # UI components (shadcn/ui)
│       └── lib/            # API client, utils
├── configs/                 # Experiment configurations
├── data/                    # Datasets, embeddings
├── notebooks/               # Colab notebooks (DPO, eval)
└── docs/
    ├── PROJECT_BLUEPRINT.md
    ├── PHASES.md
    └── DESIGN_SYSTEM.md
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **Backend** | FastAPI | Async, auto-docs, ML ecosystem |
| **Frontend** | Next.js 16 + shadcn/ui | SSR, great DX, elegant UI |
| **Database** | PostgreSQL (NeonDB) | Serverless, free tier |
| **Vector DB** | Qdrant | Best OSS performance |
| **ML** | Transformers, vLLM | Industry standard |
| **Embeddings** | sentence-transformers | Fast, reliable |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PROJECT_BLUEPRINT.md](./PROJECT_BLUEPRINT.md) | Full technical specification |
| [PHASES.md](./PHASES.md) | Development phases with tasks |
| [DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) | Frontend styling guide |
| API Docs | http://localhost:8000/docs |

---

## Contributing

This is a personal research project, but suggestions are welcome:

1. Open an issue for bugs or feature ideas
2. Fork and submit PRs for improvements
3. Star the repo if you find it useful

---

## License

MIT License - See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <i>Config-driven experimentation. Reproducibility by default.</i>
</p>
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload