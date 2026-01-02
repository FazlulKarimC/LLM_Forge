# LLM Research Engineering Platform - Project Blueprint

> **Version**: 1.0 | **Created**: 2025-12-30 | **Status**: Planning

## Executive Summary

A config-driven, reproducible LLM experimentation platform for studying:
- **Reasoning Methods**: Naive, Chain-of-Thought (CoT), ReAct
- **Retrieval Strategies**: No RAG, Naive RAG, Hybrid RAG
- **Alignment**: Direct Preference Optimization (DPO)
- **Inference Optimizations**: Batching, KV Cache, Prompt Caching

### Metrics Framework
| Category | Metrics |
|----------|---------|
| Quality | Accuracy, Faithfulness, Hallucination Rate |
| Performance | Latency (p50/p95/p99), Throughput |
| Cost | Token count, GPU time, Memory usage |

---

## Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Backend** | FastAPI (Python 3.11+) | Async, auto-docs, ML ecosystem |
| **Frontend** | Next.js 16 (TypeScript) | SSR, modern React, great DX |
| **Database** | PostgreSQL (NeonDB) | Serverless, free tier, reliable |
| **Task Queue** | RQ + Upstash Redis | Simple, reliable background jobs |
| **Vector DB** | Qdrant Cloud | 1GB free, excellent performance |
| **Inference** | HuggingFace Inference API | Free tier, no local GPU needed |
| **ML Training** | Transformers, TRL (Colab only) | Industry standard, cloud compute |
| **Embeddings** | sentence-transformers | Fast, reliable |
| **Containerization** | Docker | Qdrant local dev |

### Free Vector DB Options for Deployment
| Provider | Free Tier | Best For |
|----------|-----------|----------|
| **Qdrant Cloud** | 1GB storage | Recommended - best performance |
| **Pinecone** | 100K vectors | Good alternative |
| **Weaviate Cloud** | 14-day sandbox | Testing only |
| **NeonDB + pgvector** | Within Postgres limits | Single DB solution |

**Recommendation**: Use Qdrant Cloud - native vector DB with better performance than pgvector.

---

## Reference Projects & Inspiration

> This project is a **"Config-driven LLM Experimentation & Evaluation Platform"** - combining elements of LLMOps, ML benchmarking, and AI research infrastructure.

### LLM Evaluation Frameworks (Most Similar)

| Project | GitHub | Stars | Description |
|---------|--------|-------|-------------|
| **lm-evaluation-harness** | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 12k+ | Industry standard for LLM evaluation. Config-driven, supports 200+ tasks. Study their task YAML structure. |
| **RAGAS** | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | 7k+ | RAG evaluation with faithfulness, context precision/recall metrics. Use for Phase 5/6 metric ideas. |
| **DeepEval** | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | 3k+ | Pytest-like LLM testing, 14+ metrics, CI/CD integration. Good for testing patterns. |
| **Opik** | [comet-ml/opik](https://github.com/comet-ml/opik) | 2k+ | Open-source LLM evaluation, tracing, debugging. Study their trace visualization. |

### MLOps & Experiment Tracking

| Project | GitHub | Stars | Description |
|---------|--------|-------|-------------|
| **MLflow** | [mlflow/mlflow](https://github.com/mlflow/mlflow) | 20k+ | General ML experiment tracking with recent LLM support. Good for experiment logging patterns. |
| **Langfuse** | [langfuse/langfuse](https://github.com/langfuse/langfuse) | 7k+ | Open-source LLM observability, tracing, prompt management. Great UI inspiration for traces. |
| **Arize Phoenix** | [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | 4k+ | AI observability, LLM-as-a-judge evaluation. Study their evaluation patterns. |
| **W&B Weave** | [wandb/weave](https://github.com/wandb/weave) | 1k+ | LLM tracing and evaluation from Weights & Biases. |

### Agent & Reasoning Frameworks

| Project | GitHub | Stars | Description |
|---------|--------|-------|-------------|
| **LangChain** | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 100k+ | Build LLM apps with chains, RAG, agents. Reference for ReAct implementation patterns. |
| **LangSmith** | [smith.langchain.com](https://smith.langchain.com/) | N/A | Debugging & tracing for LangChain (commercial, free tier available). |

### What Makes LlmForge Different

1. **Research-focused** - Compares methods (CoT vs RAG vs Agent) rather than just running benchmarks
2. **Full-stack with UI** - Most frameworks are CLI-only; we have a Next.js dashboard
3. **DPO alignment included** - Few projects cover fine-tuning with preference learning
4. **Portfolio-oriented** - Designed to demonstrate ML engineering skills for interviews

### Key Repos to Study

| For Phase | Study This Repo | Learn |
|-----------|-----------------|-------|
| Phase 3 (Metrics) | lm-evaluation-harness | Task config YAML structure |
| Phase 5 (RAG) | RAGAS | Faithfulness/hallucination metrics |
| Phase 6 (Agent) | Langfuse | Trace visualization UI |
| Phase 7 (DPO) | TRL library docs | DPO training loop patterns |

---

## Hardware Constraints & Strategy

### Architecture: Remote Inference + Cloud Training

> **Key Decision**: No local GPU dependencies. All inference and training via cloud services.

| Component | Environment | Technology |
|-----------|-------------|------------|
| **Development** | Local PC (no GPU) | FastAPI, Next.js, lightweight deps only |
| **Inference** | HuggingFace API | Remote API calls (Phases 2-4) |
| **Training** | Google Colab (T4 GPU) | Transformers, TRL (Phase 7) |
| **Model Storage** | HuggingFace Hub | Trained models, LoRA adapters |

### Model Strategy by Phase
| Phase | Inference Method | Models |
|-------|------------------|--------|
| 1-2 (Weeks 1-4) | HF Inference API (free tier) | Phi-2 (2.7B), TinyLlama (1.1B) |
| 3-5 (Weeks 5-10) | HF API or Colab inference server | Mistral-7B, LLaMA-3-8B |
| 6 (Week 11) | Colab inference server | Agent evaluation |
| 7 (Week 12) | Colab training + HF Hub | DPO fine-tuning |

### Why Remote Inference?
- ✅ No local disk space needed (saves 10-15GB)
- ✅ Works on any computer (no GPU required)
- ✅ Faster development setup (5 min vs 2 hours)
- ✅ Production-ready architecture
- ✅ Matches industry practices (development ≠ training)

---

## Project Structure

```
LlmForge/
├── backend/                    # FastAPI Application
│   ├── app/
│   │   ├── api/               # API routes
│   │   │   ├── experiments.py
│   │   │   ├── results.py
│   │   │   └── metrics.py
│   │   ├── core/              # Core configuration
│   │   │   ├── config.py
│   │   │   └── database.py
│   │   ├── models/            # SQLAlchemy models
│   │   │   ├── experiment.py
│   │   │   ├── result.py
│   │   │   └── run.py
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # Business logic
│   │   │   ├── inference/     # LLM inference engines
│   │   │   ├── rag/           # RAG pipeline
│   │   │   ├── agent/         # ReAct agent
│   │   │   └── evaluation/    # Metrics computation
│   │   └── main.py
│   ├── experiments/           # Experiment configs (YAML/JSON)
│   ├── tests/
│   └── requirements.txt
│
├── frontend/                   # Next.js Application
│   ├── app/
│   │   ├── page.tsx           # Dashboard
│   │   ├── experiments/
│   │   │   ├── page.tsx       # List experiments
│   │   │   ├── new/page.tsx   # Create experiment
│   │   │   └── [id]/page.tsx  # Experiment detail
│   │   └── results/
│   ├── components/
│   └── lib/
│
├── notebooks/                  # Colab notebooks
│   ├── training/              # DPO training
│   └── evaluation/            # Large-scale eval
│
├── configs/                    # Versioned experiment configs
│   ├── reasoning/
│   ├── rag/
│   ├── agent/
│   └── alignment/
│
├── data/                       # Processed datasets
├── docs/                       # Documentation
├── docker-compose.yml
└── PROJECT_BLUEPRINT.md        # This file
```

---

## Phase-by-Phase Implementation

### Phase 1: Foundation & Infrastructure (Weeks 1-2)

**Goal**: Working end-to-end pipeline with simplest components

#### Week 1: Backend Core

**Day 1-3: Experiment Configuration System**
- [ ] Design experiment schema (JSON/YAML)
  ```yaml
  experiment:
    id: "exp_001"
    name: "naive_vs_cot_trivia"
    model: "microsoft/phi-2"
    method: "chain_of_thought"  # naive, cot, react
    dataset: "trivia_qa"
    hyperparameters:
      temperature: 0.7
      max_tokens: 256
      top_p: 0.9
    seed: 42
  ```
- [ ] Build config validator (Pydantic)
- [ ] Create experiment ID generator

**Day 3-4: Database Schema**
```sql
-- Experiments: stores experiment configs
experiments (id, name, config_json, status, created_at, completed_at)

-- Results: aggregated metrics per experiment
results (id, experiment_id, accuracy, faithfulness, latency_p50, tokens_used)

-- Runs: individual LLM calls
runs (id, experiment_id, input_text, output_text, tokens_in, tokens_out, latency_ms, correct)
```

**Day 4-5: Minimal Inference Engine**
- [ ] HuggingFace Transformers wrapper
- [ ] Load Phi-2 locally (GTX 1650)
- [ ] Basic generation: prompt → text → logs
- [ ] Log: tokens, latency, GPU memory

**Day 6-7: First Experiment**
- [ ] Download TriviaQA (100 samples)
- [ ] Implement Naive prompting
- [ ] Implement CoT prompting
- [ ] Run comparison, save to DB
- [ ] First README table with real numbers

#### Week 2: Frontend & API

**Day 1-3: Frontend Pages**
- [ ] Dashboard: experiment list, summary stats
- [ ] Experiment Detail: config, metrics, per-example results
- [ ] Create Experiment: form with model/method/dataset selection

**Day 4-5: API Routes**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/experiments` | POST | Create experiment |
| `/experiments` | GET | List experiments |
| `/experiments/{id}` | GET | Get experiment details |
| `/experiments/{id}/run` | POST | Execute experiment |
| `/experiments/{id}/metrics` | GET | Get computed metrics |

**Phase 1 Deliverables**:
- ✅ Config system working
- ✅ 2+ experiments logged
- ✅ Frontend displaying results
- ✅ README table: "Naive vs CoT on TriviaQA"

---

### Phase 2: RAG Pipeline (Weeks 3-5)

**Goal**: Build retrieval-augmented generation, measure faithfulness

#### Week 3: Knowledge Base & Embeddings

**Day 1-2: Document Processing**
- [ ] Download Wikipedia subset (~10K articles)
- [ ] Chunk documents (256 tokens, 50 overlap)
- [ ] Store chunks with metadata

**Day 3-5: Vector Database**
- [ ] Setup Qdrant (Docker local, Cloud for prod)
- [ ] Embed chunks with `all-MiniLM-L6-v2`
- [ ] Upload to Qdrant collection
- [ ] Validate retrieval quality

#### Week 4: RAG Methods

**Day 1-3: Implement Three Strategies**

| Method | Description |
|--------|-------------|
| **Naive RAG** | Query → embed → top-5 chunks → generate |
| **Hybrid RAG** | Dense (vector) + Sparse (BM25) → merge → generate |
| **Reranked RAG** | Hybrid → cross-encoder rerank → top-5 → generate |

**Day 4-5: Evaluation Metrics**
| Metric | How to Compute |
|--------|----------------|
| Accuracy | Exact match, F1 score |
| Faithfulness | NLI entailment (BART-MNLI) |
| Answer Relevance | Semantic similarity |
| Hallucination Rate | % with faithfulness < 0.5 |

#### Week 5: RAG Ablations

| Ablation | Variable | Values |
|----------|----------|--------|
| Retrieval Method | method | naive, hybrid, reranked |
| Top-K | k | 1, 3, 5, 10 |
| Chunk Size | tokens | 128, 256, 512 |

**Phase 2 Deliverables**:
- ✅ 3 RAG variants implemented
- ✅ Faithfulness metrics working
- ✅ 3 ablation studies complete
- ✅ Finding: "Reranking reduces hallucinations by X%"

---

### Phase 3: ReAct Agent (Weeks 6-7)

**Goal**: Implement tool-using agent, measure efficiency

#### Week 6: Tool Framework

**Day 1-2: Tool Interface**
```python
class Tool:
    name: str
    description: str
    input_schema: dict
    
    def execute(self, **kwargs) -> ToolResult:
        ...
```

**Tools to Implement**:
| Tool | Input | Output |
|------|-------|--------|
| `wikipedia_search` | query: str | First paragraph |
| `calculator` | expression: str | Numerical result |
| `retrieval` | query: str, k: int | Concatenated chunks |

**Day 3-5: ReAct Loop**
```
Thought: [reasoning]
Action: tool_name(arguments)
Observation: [tool result]
... repeat ...
Thought: I know the answer
Answer: [final answer]
```

#### Week 7: Agent Evaluation

**Metrics**:
| Metric | Definition |
|--------|------------|
| Success Rate | Correct final answer |
| Tool Efficiency | Avg tools per question |
| Failed Calls | % of tool errors |
| Cost Proxy | Total tokens used |

**Comparison**: Naive vs CoT vs RAG vs ReAct Agent

**Phase 3 Deliverables**:
- ✅ ReAct agent with 3 tools
- ✅ Full agent traces logged
- ✅ Comparison table
- ✅ Finding: "Agents improve accuracy by X% but cost Y× more"

---

### Phase 4: DPO Alignment (Weeks 8-9)

**Goal**: Fine-tune with preferences, measure alignment tax

#### Week 8: Training Setup

**Day 1-2: Dataset**
- [ ] Download Anthropic HH-RLHF (1000 pairs)
- [ ] Split: 800 train / 100 val / 100 test

**Day 3-5: DPO Training (Colab)**
```python
# Config
model: "mistralai/Mistral-7B-Instruct-v0.2"
beta: 0.1
learning_rate: 5e-7
epochs: 1
batch_size: 4
lora_rank: 16
```

#### Week 9: Alignment Evaluation

| Evaluation | Dataset | Metric |
|------------|---------|--------|
| Helpfulness | HH test set | Human preference |
| Factuality | TriviaQA | Exact match |
| Verbosity | All | Avg response length |

**Ablations**:
- Beta temperature: 0.05, 0.1, 0.2
- Dataset size: 200, 500, 800 pairs

**Phase 4 Deliverables**:
- ✅ DPO-tuned model saved
- ✅ Base vs DPO comparison
- ✅ Finding: "DPO improves helpfulness by X% but reduces factuality by Y%"

---

### Phase 5: Inference Optimization (Weeks 10-11)

**Goal**: Optimize performance, create Pareto analysis

#### Week 10: Optimizations

| Optimization | Implementation |
|--------------|----------------|
| **Batching** | Process 4/8/16 prompts together |
| **KV Cache** | vLLM with PagedAttention |
| **Prompt Caching** | Cache repeated context prefixes |

#### Week 11: Benchmarking

**Full Matrix**:
- Methods: Naive, CoT, RAG, Agent
- Optimizations: Sequential, Batched, vLLM, Cached
- Metrics: Accuracy, Latency (p50/p95/p99), Throughput, Memory

**Visualizations**:
- Pareto frontier: Accuracy vs Latency
- Bar chart: Speedup from each optimization
- Heatmap: Method × Optimization → Performance

**Phase 5 Deliverables**:
- ✅ All optimizations implemented
- ✅ Comprehensive benchmarks
- ✅ Pareto charts
- ✅ Finding: "vLLM + batching gives 4× throughput"

---

### Phase 6: Polish & Presentation (Week 12)

**Goal**: Interview-ready presentation

#### Deliverables Checklist

**Documentation**:
- [ ] README with results tables
- [ ] Architecture diagram
- [ ] 5+ visualizations
- [ ] Limitations section
- [ ] Future work

**Code Quality**:
- [ ] Docstrings on all functions
- [ ] Type hints everywhere
- [ ] 10-15 unit tests
- [ ] No hardcoded values

**Deployment**:
| Component | Platform |
|-----------|----------|
| Frontend | Vercel |
| Backend | Railway / Render |
| Database | NeonDB |
| Vector DB | Qdrant Cloud |

**Demo**:
- [ ] 5-7 minute video walkthrough
- [ ] Live demo URL in README

---

## Operating Principles

### 1. Think Like Big Tech
- Build infrastructure first, intelligence second
- Measure before optimizing
- Prefer boring, reliable solutions

### 2. Iteration Over Completion
- Ship minimal working versions
- Improve through controlled experiments
- Refactor only when justified by data

### 3. Config-Driven Everything
- No hardcoded experiments
- Every run reproducible from config
- Version configs like code

### 4. Research Integrity
- Every claim backed by logged metrics
- Document negative results
- Avoid cherry-picking outcomes

---

## Architecture Principles

> These principles were established after debugging background task failures. They ensure reliability in async experiment execution.

### 1. Status Is a Persisted State Machine

`ExperimentStatus` is a database-backed state machine, not an in-memory flag.

**State Transitions**:
```
PENDING → RUNNING → COMPLETED
                 ↘ FAILED
```

**Rules**:
- No skipping states (PENDING cannot go directly to COMPLETED)
- No backward transitions (COMPLETED cannot go back to RUNNING)
- A transition is not real until it is **committed**

```python
# Correct pattern
await update_status(experiment_id, COMPLETED)
await db.commit()  # Transition is now permanent
```

### 2. One Layer Owns Transactions

Exactly one layer owns `commit()` and `rollback()`:

| Layer | Responsibility |
|-------|----------------|
| API | Request validation, enqueue job |
| Background Task (RQ) | Execution wrapper, session creation |
| **Service** | **OWNS transactions + business logic** |
| Database | Source of truth |

**Anti-patterns**:
- ❌ Commit in API routes
- ❌ Commit in background task wrapper
- ❌ Commit in helper functions like `update_status()`

### 3. Background Tasks Are Fire-and-Forget Workers

We use RQ (Redis Queue) with Upstash for reliable background execution.

**Why RQ over FastAPI BackgroundTasks**:
| FastAPI BackgroundTasks | RQ Workers |
|-------------------------|------------|
| Tied to API process lifecycle | Independent process |
| Silent failures possible | Explicit failure handling |
| No retry mechanism | Built-in retry support |
| In-memory queue | Persistent Redis queue |

**Worker Rules**:
- Catch all exceptions
- Persist failure state to database
- Log aggressively
- Never crash silently

### 4. Think in Execution Phases

Each experiment execution has distinct phases:

```
1. Load inputs (experiment config, questions)
2. Validate config
3. Prepare execution context (engine, services)
4. Run inference loop
5. Persist outputs (runs to database)
6. Finalize experiment (update status)
```

Each phase:
- Logs start/completion
- Can fail independently
- Commits only at defined boundaries

---



## Success Metrics

### Technical
- [ ] 15+ experiments logged
- [ ] 3+ ablation studies
- [ ] 5+ visualizations
- [ ] Fully reproducible

### Learning
- [ ] Can explain transformers whiteboard-style
- [ ] Understand reasoning/RAG trade-offs
- [ ] Comfortable with alignment challenges
- [ ] Know inference optimization techniques

### Career
- [ ] GitHub stars > 10
- [ ] Positive recruiter responses
- [ ] Can explain in 2 minutes
- [ ] Confident in ML systems interviews

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Colab limits | Develop locally with Phi-2 first |
| Scope creep | Stick strictly to roadmap |
| Shallow evaluation | Prioritize metrics over features |
| No insights | Every experiment needs hypothesis |
| Code quality | Refactor in Week 12 |

---

## Interview Talking Points

### 30-Second Pitch
> "I built an LLM experimentation platform to study reasoning, retrieval, and alignment. I implemented Chain-of-Thought, ReAct agents, and RAG with multiple strategies. I fine-tuned a 7B model using DPO and found that alignment improves helpfulness by 15% but reduces factual accuracy by 6%. I optimized inference using batching and KV caching, achieving 4× throughput improvement."

### Key Questions Prepared
1. **How did you evaluate faithfulness?** → NLI with BART-MNLI
2. **Why did DPO hurt factuality?** → Alignment-capability trade-off
3. **How would you improve this?** → Iterative refinement, more tools, MoE routing
4. **Biggest challenge?** → Designing reproducible experiments

---

## Appendix: Useful Commands

```bash
# Backend development
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend development
cd frontend
npm install
npm run dev

# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run experiment
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d @configs/reasoning/naive_vs_cot.json
```

---

*This blueprint is a living document. Update as the project evolves.*
