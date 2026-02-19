# LlmForge — Development Phases

| Phase | Name | Status |
|-------|------|--------|
| 0 | Infrastructure Foundation | ✅ Complete |
| 1 | Core Platform | ✅ Complete |
| 2 | Basic Inference | ✅ Complete |
| 3 | Evaluation & Metrics | ✅ Complete |
| 4 | Chain-of-Thought | ✅ Complete |
| 5 | RAG Pipeline | ✅ Complete |
| 6 | ReAct Agent | ✅ Complete |
| 7 | Inference Optimization | ✅ Complete |
| 8 | Polish & Frontend | ✅ Complete |

---

## Phase 0 — Infrastructure Foundation ✅

Set up the full project scaffold before writing any real logic.

- FastAPI backend with layered architecture (`api/`, `core/`, `models/`, `schemas/`, `services/`)
- Next.js 16 frontend with TypeScript, React 19, shadcn/ui, Tailwind CSS v4
- SQLAlchemy ORM models: `Experiment`, `Result`, `Run`
- Pydantic schemas with validation
- Docker Compose for Qdrant (vector DB)
- `DESIGN_SYSTEM.md` — 4-color palette, Typography (Instrument Serif + Inter), component patterns

---

## Phase 1 — Core Platform ✅

Wired up the database and built a working CRUD API.

- NeonDB (PostgreSQL) connection via async SQLAlchemy
- Alembic migrations
- Full experiment CRUD: create, list (paginated + filtered), get by ID, soft delete
- Frontend pages: Dashboard, Experiments list, New Experiment form, Experiment detail
- TanStack Query for data fetching, loading/error states

---

## Phase 2 — Basic Inference ✅

Ran first real LLM inference through the pipeline.

- `HFAPIEngine` — calls HuggingFace Inference Providers API (novita provider, `InferenceClient`)
- `MockEngine` — deterministic fake responses for testing without API quota
- `GenerationConfig` / `GenerationResult` base types
- `INFERENCE_ENGINE` setting to switch between `hf_api` and `mock`
- Experiment `execute()` method: loads dataset → generates → saves runs → computes metrics → marks complete
- Redis + `BackgroundTasks` for async experiment execution
- Fixed: wrapped all blocking `engine.generate()` calls with `asyncio.to_thread()` to prevent event loop starvation

---

## Phase 3 — Evaluation & Metrics ✅

Built the metrics pipeline and results dashboard.

- Metrics: exact match, substring match, token-level F1 (with alias support)
- `MetricsService.compute_and_save()` — aggregates per-run scores into a `Result` row
- Results API (`/api/v1/results/<id>/metrics`) returning quality + performance + cost breakdown
- Frontend `ResultsDashboard`: Accuracy, F1, Latency, Throughput cards
- Per-run correctness grid with pagination (10 rows/page)
- Export results as JSON

---

## Phase 4 — Chain-of-Thought ✅

Added CoT reasoning as a second inference method.

- `CoTPromptTemplate` — few-shot examples loaded from `configs/cot_examples.json`
- Zero-shot CoT fallback if examples file missing
- `CoTPromptTemplate.parse_response()` — extracts final answer after "Answer:" marker
- Auto-increases `max_tokens` to 512 for CoT experiments
- Experiment comparison page — side-by-side metric cards for selected experiments

---

## Phase 5 — RAG Pipeline ✅

Added retrieval-augmented generation for knowledge-base questions.

- `RAGPipeline` — loads knowledge base, chunks text, indexes into Qdrant
- Three retrieval methods: `vector` (semantic), `bm25` (keyword), `hybrid`
- `RAGPromptTemplate` — injects retrieved context chunks into prompt
- `FaithfulnessScorer` — measures answer grounding in retrieved context
- `RetrievalTool` — wraps RAG pipeline as a ReAct-compatible tool
- `knowledge_base` dataset for RAG-focused evaluation

---

## Phase 6 — ReAct Agent ✅

Built a tool-using agent with the ReAct (Reason + Act) loop.

- `ReActAgent` — iterative Thought → Action → Observation loop (configurable max iterations)
- Tools: `WikipediaSearchTool`, `CalculatorTool`, `RetrievalTool`
- `ReActPromptTemplate` — formats the multi-turn tool-use conversation
- Agent trace stored per run (thoughts, tool calls, observations)
- `react_bench` dataset for agent evaluation
- Agent config in experiment schema: `tools`, `max_iterations`

---

## Phase 7 — Inference Optimization ✅

Added caching, batching, and profiling to speed up experiment runs.

- `PromptCache` — in-memory LRU cache keyed by (prompt, model, hyperparams)
- `ProfilerContext` — per-section timing (prompt_build, api_call, parsing, metrics)
- `OptimizationReport` — wall time, cache hit rate, batch stats, saved to `raw_metrics`
- Batched execution path: groups prompts, runs `generate_batch()` with thread pool
- `OptimizationConfig` schema: `enable_caching`, `enable_batching`, `batch_size`, `cache_max_size`, `enable_profiling`

---

## Phase 8 — Polish & Frontend ✅

Production-ready UI and backend hardening.

- Full frontend audit and fixes (Zod form validation, model list from backend, Run/Delete buttons)
- Status badges: `pending`, `queued`, `running`, `completed`, `failed`
- "Create & Run" button on New Experiment form
- "Run Experiment" button on detail page with auto-polling
- Results grid pagination (10 rows/page)
- Experiment comparison page
- HF Inference API fix: switched from deprecated serverless endpoint to novita provider (`InferenceClient(provider="novita")`)
- `asyncio.to_thread()` fix — prevents blocking event loop during HF API calls
