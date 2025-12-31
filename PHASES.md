# LLM Research Platform - Development Phases

> Real-world AI system development roadmap

---

## Overview

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| 0 | Infrastructure Foundation | ✅ Complete | Project scaffold |
| 1 | Core Platform | 🔲 Pending | Working CRUD + API |
| 2 | Basic Inference | 🔲 Pending | LLM generates text |
| 3 | Evaluation & Metrics | 🔲 Pending | Metrics dashboard |
| 4 | Chain-of-Thought | 🔲 Pending | First comparison |
| 5 | RAG Pipeline | 🔲 Pending | Retrieval system |
| 6 | ReAct Agent | 🔲 Pending | Tool-using agent |
| 7 | DPO Alignment | 🔲 Pending | Fine-tuned model |
| 8 | Inference Optimization | 🔲 Pending | Performance benchmarks |
| 9 | Polish & Deployment | 🔲 Pending | Live demo |

---

## Phase 0: Infrastructure Foundation ✅

**Status**: Complete

### What We Built
- [x] Project folder structure (backend/frontend/configs)
- [x] FastAPI backend scaffold with proper layering
- [x] Next.js 16 frontend with TypeScript & React 19
- [x] SQLAlchemy models (Experiment, Result, Run)
- [x] Pydantic schemas with validation
- [x] Service interfaces with TODOs
- [x] Docker Compose for Qdrant
- [x] Example experiment config
- [x] Frontend Design System (DESIGN_SYSTEM.md)
  - 4-color palette (Dark Brown, Off-White, Taupe Gray, Light Gray)
  - Typography system (Instrument Serif + Inter)
  - shadcn/ui component patterns
  - Tailwind CSS v4 configuration

### Files Created
```
backend/app/
├── api/         → experiments.py, results.py, health.py
├── core/        → config.py, database.py
├── models/      → experiment.py, result.py, run.py
├── schemas/     → experiment.py, result.py, run.py
└── services/    → experiment_service.py, inference/

frontend/src/
├── app/         → Dashboard, Experiments pages
└── lib/         → api.ts, types.ts

DESIGN_SYSTEM.md  → Frontend design guidelines & component patterns
```

---

## Phase 1: Core Platform

**Goal**: Make the platform functional with database operations

### Tasks
- [ ] **1.1 Database Connection**
  - Configure NeonDB connection string
  - Test async SQLAlchemy connection
  - Create database tables

- [ ] **1.2 Alembic Migrations**
  - Initialize Alembic
  - Create initial migration
  - Set up auto-migration workflow

- [ ] **1.3 Implement Experiment CRUD**
  - `create()` - Save experiment config to DB
  - `get()` - Retrieve by ID
  - `list()` - Paginated listing with filters
  - `delete()` - Soft delete

- [ ] **1.4 Frontend API Integration**
  - Connect Dashboard to real stats
  - Connect Experiments list to API
  - Connect New Experiment form
  - Handle loading/error states

- [ ] **1.5 End-to-End Test**
  - Create experiment via UI
  - Verify it appears in list
  - View experiment details

- [ ] **1.6 Apply Design System**
  - Install shadcn/ui components (button, card, badge, table, tabs, form, input, skeleton)
  - Configure Tailwind with DESIGN_SYSTEM.md color palette
  - Apply 4-color system to all pages
  - Use Instrument Serif for headings, Inter for body

### Deliverables
- ✅ Create experiment → appears in database
- ✅ List experiments with pagination
- ✅ Filter by status, method

### Exit Criteria
- [ ] Create experiment via UI → appears in database within 2 seconds
- [ ] Filter experiments by status returns correct subset
- [ ] Error handling works (test with invalid config)
- [ ] Pagination works (create 20+ experiments, test navigation)
- [ ] Can view experiment details without errors
- [ ] UI follows DESIGN_SYSTEM.md guidelines (4-color palette, typography, spacing)
- [ ] All shadcn/ui components styled per design system

### Technical Notes
```python
# NeonDB connection string format
DATABASE_URL=postgresql://user:pass@ep-xxx.region.neon.tech/dbname?sslmode=require
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| NeonDB SSL errors | Add `?sslmode=require` to connection string |
| Async/await confusion | All DB operations must use async/await, don't mix sync/async |
| Frontend CORS errors | Ensure FastAPI has CORS middleware in `main.py`, check allowed origins |

---

## Phase 2: Basic Inference Engine

**Goal**: Run actual LLM inference locally on GTX 1650

### Tasks
- [ ] **2.1 Model Loading**
  - Install transformers, torch
  - Load Phi-2 (2.7B) - fits in 4GB VRAM
  - Implement `load_model()` in TransformersEngine
  - Handle GPU memory errors gracefully

- [ ] **2.2 Text Generation**
  - Implement `generate()` method
  - Track input/output tokens
  - Measure latency
  - Return GenerationResult

- [ ] **2.3 Naive Prompting**
  - Format: `Question: {q}\nAnswer:`
  - Parse generated response
  - Handle edge cases (empty, too long)

- [ ] **2.4 Run Logging**
  - Save each LLM call to `runs` table
  - Log: input, output, tokens, latency
  - Associate runs with experiments

- [ ] **2.5 Execution Pipeline**
  - Implement `ExperimentService.execute()`
  - Load config → Load model → Run inference → Save results

- [ ] **2.6 GPU Memory Management**
  - Log GPU memory before/after model load
  - Implement `torch.cuda.empty_cache()` after runs
  - Add warning if memory >90% used
  - Create automatic fallback to CPU if OOM

### Deliverables
- ✅ Hit Phi-2 with a prompt, get response
- ✅ Every call logged with metrics
- ✅ Can run experiment from UI

### Exit Criteria
- [ ] Run 10 consecutive inferences without crashes
- [ ] Token counts match expected (input + output = total)
- [ ] Latency is reasonable (<2s GPU, <10s CPU)
- [ ] Runs table has 10 entries with all non-null required fields
- [ ] GPU memory clears properly between runs

### Technical Notes
```python
# Phi-2 fits on GTX 1650 with float16
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| OOM errors | Use `torch_dtype=torch.float16`, `device_map="auto"`, auto-fallback to CPU |
| Slow CPU generation | Expected locally; use small batches (1-5), reserve large runs for Colab |
| Token count mismatch | Use `tokenizer.encode()`, count prompt + generated separately |

---

## Phase 3: Evaluation & Metrics

**Goal**: Measure experiment quality with proper metrics

### Tasks
- [ ] **3.1 Dataset Loading**
  - Download TriviaQA from HuggingFace
  - Create dataset abstraction
  - Sample N examples with seed
  - Cache locally in `data/datasets/triviaqa/`
  - Save sampled subset as JSON with seed for reproducibility

- [ ] **3.2 Accuracy Metrics**
  - Exact string match (case-insensitive)
  - Substring containment
  - F1 token overlap score

- [ ] **3.3 Latency Metrics**
  - Collect all run latencies
  - Compute p50, p95, p99 percentiles
  - Calculate throughput (runs/second)

- [ ] **3.4 Cost Proxies**
  - Total input tokens
  - Total output tokens
  - Estimated GPU seconds

- [ ] **3.5 Results Dashboard (Enhanced)**
  - Display metrics in card layout (Accuracy, Latency, Cost)
  - Show latency histogram using Recharts
  - Highlight top-5 fastest/slowest examples
  - Add "Export Results" button (JSON download)
  - Show grid view: correct (green) vs incorrect (red) examples

### Deliverables
- ✅ Accuracy: 45.2% exact match
- ✅ Latency: p50=320ms, p95=580ms
- ✅ Tokens: 15,234 input, 8,921 output

### Exit Criteria
- [ ] At least one full experiment with 50+ examples
- [ ] All metrics (accuracy, latency, tokens) computed correctly
- [ ] Results appear in frontend dashboard
- [ ] Can export results to JSON
- [ ] Metrics make intuitive sense (accuracy 0-100%, latency >0)

### Expected Baseline (Don't Panic!)
| Metric | Expected Range | Why |
|--------|---------------|-----|
| Phi-2 Accuracy | 30-45% | Small models struggle on TriviaQA |
| Latency | 200ms - 2s | Varies by output length |
| F1 > Exact Match | Normal | F1 captures partial credit |

### Technical Notes
```python
# F1 score calculation
def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| Slow dataset downloads | Use `cache_dir="./data/cache"` in `load_dataset()` |
| Metrics don't match expectations | Validate on 5-10 examples manually, check tokenization |
| Dashboard shows wrong numbers | Verify DB queries, check aggregation logic (mean, percentiles) |

---

## Phase 4: Chain-of-Thought Reasoning

**Goal**: Implement and compare reasoning strategies

### Tasks
- [ ] **4.1 CoT Prompt Template**
  - Add "Let's think step by step" trigger
  - Format reasoning chain
  - Extract final answer

- [ ] **4.2 Few-Shot Examples**
  - Create 3-5 high-quality CoT examples manually
  - Store in `configs/cot_examples.json`
  - Test: zero-shot vs few-shot CoT
  - Document which performs better

- [ ] **4.3 Answer Parsing**
  - Parse "Answer:" or "Therefore:" patterns
  - Handle multi-line reasoning
  - Fallback to last sentence if no pattern found

- [ ] **4.4 Ablation: Naive vs CoT**
  - Run same 100 samples with both methods
  - Use identical seed for reproducibility
  - Compare accuracy and latency

- [ ] **4.5 Statistical Validation**
  - Compute confidence intervals using bootstrap
  - Run McNemar's test for paired accuracy comparison
  - Document: Is improvement statistically significant?

- [ ] **4.6 Experiment Comparison View**
  - Side-by-side metrics table
  - Highlight improvements (green) and regressions (red)
  - Show per-example differences

- [ ] **4.7 Document Findings**
  - Update README with results table
  - Write interpretation of findings

### Deliverables
- ✅ First research finding documented
- ✅ Table: Naive 42% → CoT 58% (+16%)
- ✅ Comparison view working

### Exit Criteria
- [ ] Both Naive and CoT methods run successfully on same dataset
- [ ] Results show measurable difference in accuracy or latency
- [ ] Statistical significance computed (p-value)
- [ ] Comparison view displays side-by-side results
- [ ] README updated with first research finding table

### Expected Results
| Method | Accuracy | Latency p50 | Tokens |
|--------|----------|-------------|--------|
| Naive | 42.0% | 280ms | 5,200 |
| CoT | 58.0% | 450ms | 12,800 |

### Technical Notes
```python
# McNemar's test for paired accuracy comparison
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

def compute_significance(naive_correct: list, cot_correct: list):
    """Compare two methods on same examples."""
    # Build contingency table
    # [both correct, naive only, cot only, both wrong]
    b = sum(n and not c for n, c in zip(naive_correct, cot_correct))  # naive correct, cot wrong
    c = sum(c and not n for n, c in zip(naive_correct, cot_correct))  # cot correct, naive wrong
    
    result = mcnemar([[0, b], [c, 0]], exact=True)
    return {"p_value": result.pvalue, "significant": result.pvalue < 0.05}
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| CoT doesn't improve accuracy | Try few-shot instead of zero-shot, test on reasoning-heavy dataset (HotpotQA) |
| Answer parsing fails | Model doesn't follow format; add explicit examples, use regex fallbacks |
| Results too similar | Need larger sample size (200+), try harder dataset |

---

## Phase 5: RAG Pipeline

**Goal**: Build retrieval-augmented generation system

### Tasks
- [ ] **5.1 Document Ingestion**
  - Download Simple Wikipedia from HuggingFace (`wikipedia/20220301.simple`)
  - Chunk into 256-token segments with 50-token overlap
  - Store chunks with metadata (title, section, URL)
  - Save processed chunks to `data/knowledge_base/`

- [ ] **5.2 Embedding Pipeline**
  - Use `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
  - Batch embed all chunks (`batch_size=100`)
  - Monitor progress (print every 1000 chunks)
  - Upload to Qdrant collection with metadata

- [ ] **5.3 Naive RAG**
  - Query → Embed → Top-5 retrieval
  - Concatenate chunks as context
  - Generate with context prepended

- [ ] **5.4 Hybrid RAG**
  - Install `rank-bm25` library
  - Build BM25 index over chunks (keyword search)
  - For each query: get top-10 dense + top-10 BM25
  - Merge results (deduplicate, keep top-5)

- [ ] **5.5 Reranked RAG**
  - Use cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - After hybrid retrieval (top-10 candidates)
  - Score each (query, chunk) pair, rerank, select top-5

- [ ] **5.6 Faithfulness Metric**
  - Use NLI model `facebook/bart-large-mnli`
  - Format input: `{context} </s> {answer}`
  - Compute faithfulness score (0-1)
  - Hallucination rate = % with faithfulness < 0.5

- [ ] **5.7 RAG Ablations**
  - Compare: No RAG vs Naive vs Hybrid vs Reranked
  - Measure: Accuracy, Faithfulness, Hallucination Rate, Latency
  - Test chunk size: 128, 256, 512 tokens
  - Test top-k: 1, 3, 5, 10 chunks

### Deliverables
- ✅ RAG pipeline with 3 variants
- ✅ Faithfulness: 0.72 → 0.87 with reranking
- ✅ Finding: "Reranking reduces hallucinations by 18%"

### Exit Criteria
- [ ] Qdrant running and accessible (test with simple query)
- [ ] All chunks embedded and uploaded successfully
- [ ] Retrieval returns relevant documents (manual spot-check 10 queries)
- [ ] All 3 RAG variants implemented and working
- [ ] Faithfulness metric validated on known entailment pairs
- [ ] At least 2 ablation studies completed with documented results

### Architecture
```
Query → [Embed] → [Vector Search] → [BM25 Search] → [Merge] → [Rerank] → [Generate]
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| Qdrant connection fails | Check `docker ps`, restart with `docker-compose restart qdrant` |
| Embeddings take forever | Use batch processing, expected 1-2 hours for 100K chunks on CPU |
| Retrieval returns irrelevant docs | Check chunk quality, adjust similarity threshold (0.3-0.7) |
| NLI always returns 1.0 or 0.0 | Check input format: must be `premise </s> hypothesis` |
| BM25 index errors | Ensure chunks are tokenized, handle empty chunks |

---

## Phase 6: ReAct Agent

**Goal**: Implement tool-using agent with traces

### Tasks
- [ ] **6.1 Tool Interface**
  - Define `Tool` base class with name, description, execute()
  - Tools return: result string, success boolean, execution time
  - Add error handling for tool failures

- [ ] **6.2 Implement Tools**
  - `wikipedia_search(query)` → First paragraph via Wikipedia API
  - `calculator(expression)` → Safe eval using `numexpr` (NOT `eval()`)
  - `retrieval(query, k)` → Search Qdrant, return chunks

- [ ] **6.3 Tool Unit Tests**
  - Test wikipedia_search with known queries
  - Test calculator with edge cases (division by zero, invalid syntax)
  - Test retrieval with empty query
  - Document: What % of tool calls fail normally?

- [ ] **6.4 ReAct Loop Implementation**
  - Parse Thought/Action/Observation format
  - Extract tool name and arguments from "Action:" line
  - Execute tool, inject "Observation:" back into prompt
  - Loop until "Answer:" appears or max iterations (5)

- [ ] **6.5 Loop Safety & Detection**
  - Detect if agent repeats same action 3+ times
  - Force termination with "stuck" status
  - Log: Why did agent get stuck?
  - Add to metrics: % of runs that got stuck

- [ ] **6.6 Agent Tracing**
  - Log full trace: every thought, action, observation
  - Count: successful tool calls, failed calls, total iterations
  - Store traces in JSONB column in database
  - Create trace visualization in frontend (formatted text log with highlights)

- [ ] **6.7 Error Handling**
  - Tool execution failures (network timeout, invalid input)
  - Parsing failures (model doesn't follow format)
  - Max iteration reached (agent doesn't conclude)
  - Graceful degradation: return partial result

- [ ] **6.8 Agent Evaluation**
  - Compare: Naive vs CoT vs RAG vs ReAct Agent
  - Datasets: HotpotQA (multi-hop), GSM8K (math)
  - Metrics: Success rate, tool efficiency, cost proxy, latency

### Deliverables
- ✅ Agent with 3 working tools
- ✅ Full traces logged
- ✅ Finding: "Agents +12% accuracy, 4× cost"

### Exit Criteria
- [ ] All 3 tools working independently (unit tests pass)
- [ ] Agent successfully completes 5+ multi-hop questions
- [ ] Full traces logged and viewable in UI
- [ ] Loop detection prevents infinite runs
- [ ] Comparison table shows agent vs non-agent methods
- [ ] At least 20% of agent runs succeed on hard questions

### ReAct Format
```
Thought: I need to find when Paris was founded.
Action: wikipedia_search("Paris history founding")
Observation: Paris was founded in the 3rd century BC...
Thought: I now know the answer.
Answer: Paris was founded in the 3rd century BC.
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| Agent gets stuck in loops | Implement loop detection (same action 3× → terminate) |
| Tool parsing fails constantly | Add explicit examples in prompt, use regex fallbacks |
| Wikipedia API rate limiting | Add 0.5s delays, cache results locally, retry with backoff |
| Calculator security concerns | Use `numexpr` library (safe), never use `eval()` |
| Agent too expensive | Expected (3-5× more tokens); document cost-benefit trade-off |

---

## Phase 7: DPO Alignment

**Goal**: Fine-tune model with preference learning

### Colab-Specific Setup

> All training happens in Colab - never locally.

**Initial Setup (First Session):**
1. Open new Colab notebook
2. Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Create project folder: `/content/drive/MyDrive/llmforge-training/`
4. Install dependencies: `!pip install transformers trl peft accelerate datasets`
5. Download dataset once, save to Drive
6. Download base model once, save to Drive

**Checkpoint Strategy:**
- Save checkpoint every 100 training steps → Google Drive
- On disconnect: resume from last checkpoint
- Final model: upload to HuggingFace Hub (private repo)

### Tasks
- [ ] **7.1 Preference Dataset Preparation**
  - Download Anthropic HH-RLHF (helpfulness subset)
  - Format: prompt + chosen_response + rejected_response
  - Split: 800 train / 100 validation / 100 test
  - Save to Google Drive as `.parquet` or `.json`
  - Manually review 20 pairs to ensure quality

- [ ] **7.2 Training Configuration**
  - Base model: `mistralai/Mistral-7B-Instruct-v0.2` (Colab T4 only)
  - Use LoRA for efficiency (don't full fine-tune)
  - LoRA config: `rank=16, target_modules=["q_proj", "v_proj"]`
  - DPO config: `beta=0.1, learning_rate=5e-7`
  - Training: 1 epoch, `batch_size=4`, `gradient_accumulation=4`

- [ ] **7.3 Training Execution in Colab**
  - Load base model with 4-bit quantization (fits in 16GB)
  - Initialize DPOTrainer from `trl` library
  - Train for ~2 hours on T4 GPU
  - Monitor loss curves (should decrease steadily)
  - Save checkpoints to Google Drive every 100 steps

- [ ] **7.4 Model Upload**
  - Merge LoRA weights with base model
  - Upload to HuggingFace Hub: `your-username/mistral-7b-dpo-v1`
  - Set repository to private during development
  - Save model card with training details

- [ ] **7.5 Evaluation Setup**
  - Download fine-tuned model from HF Hub
  - Compare: Base model vs DPO-tuned model
  - Use same evaluation infrastructure from Phase 3

- [ ] **7.6 Helpfulness Evaluation**
  - Use Anthropic HH test set (100 prompts)
  - Generate responses from both models
  - Human evaluation: judge 50 pairs (which is more helpful?)
  - Expected: DPO model preferred 60-70% of time

- [ ] **7.7 Factuality Evaluation**
  - Use TriviaQA (200 questions)
  - Measure exact match accuracy for both models
  - Expected: DPO model slightly worse (2-8% drop)

- [ ] **7.8 Verbosity Analysis**
  - Measure average response length (tokens)
  - Compare: base vs DPO on same prompts
  - Expected: DPO model 10-20% longer responses

- [ ] **7.9 Ablation Studies**
  - Ablation 1 - Beta: Train with `beta = 0.05, 0.1, 0.2`
  - Ablation 2 - Data: Train with 200, 500, 800 pairs
  - Measure: Helpfulness score vs Factuality accuracy
  - Find optimal hyperparameters

### Deliverables
- ✅ DPO-tuned model on HF Hub
- ✅ Finding: "Helpfulness +15%, Factuality -6%"
- ✅ Alignment-capability trade-off documented

### Exit Criteria
- [ ] Fine-tuned model successfully uploaded to HuggingFace Hub
- [ ] Base vs DPO comparison completed on 2+ metrics
- [ ] Human evaluation shows measurable preference improvement
- [ ] Factuality trade-off documented with numbers
- [ ] At least 1 ablation study completed
- [ ] README updated with alignment findings

### Training Config
```python
training_args = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)
```

### Colab Session Management
- Checkpoint every 100 steps → Google Drive
- If session disconnects: reload from last checkpoint
- Total training: ~800 steps (1 epoch on 800 pairs)
- Each session: aim for 200-300 steps, then save

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| Colab disconnects during training | Save checkpoints every 100 steps, resume with `trainer.train(resume_from_checkpoint=...)` |
| Out of memory during training | Use 4-bit quantization, reduce batch_size to 2, increase gradient accumulation |
| Loss doesn't decrease | Check learning rate, verify dataset format, monitor gradient norms |
| DPO makes model worse | Review dataset quality, try lower beta (0.05), may need better base model |
| Upload to HuggingFace fails | Run `huggingface-cli login`, check repo exists, verify internet in Colab |

---

## Phase 8: Inference Optimization

**Goal**: Production-grade performance

### Tasks
- [ ] **8.1 Batching Implementation**
  - Implement `generate_batch()` method
  - Group prompts into batches of 4, 8, 16
  - Use padding for variable-length inputs
  - Measure: throughput (prompts/second), latency per prompt

- [ ] **8.2 vLLM Integration (Colab Only)**
  - Install vLLM in Colab: `!pip install vllm==0.2.7`
  - Load Mistral-7B with vLLM engine
  - Compare: Transformers vs vLLM on same workload
  - Measure: tokens/second, first token latency, memory usage

- [ ] **8.3 Prompt Caching**
  - Detect repeated context prefixes (common in RAG)
  - Cache embedding of context chunks
  - Reuse cached embeddings for same context + different questions
  - Measure: cache hit rate, latency savings per hit

- [ ] **8.4 Memory Profiling**
  - Measure peak GPU memory per method
  - Test: max batch size before OOM
  - Document: Method × Batch Size → Memory usage
  - Create memory usage heatmap

- [ ] **8.5 Comprehensive Benchmark**
  - Matrix: All methods × All optimizations
  - Methods: Naive, CoT, RAG, Agent, DPO
  - Optimizations: Sequential, Batched, vLLM, Cached
  - Metrics: Accuracy, Latency (p50/p95), Throughput, Memory

- [ ] **8.6 Optimization Decision Framework**
  - Document when to use each optimization
  - Create decision tree based on use case
  - Include cost-benefit analysis

### Deliverables
- ✅ 4× throughput with vLLM + batching
- ✅ Pareto frontier visualization
- ✅ Production recommendations

### Exit Criteria
- [ ] Batching implemented and shows 2-3× throughput improvement
- [ ] vLLM tested on Colab (T4 GPU) with performance gains documented
- [ ] Prompt caching shows measurable latency reduction for RAG
- [ ] Full benchmark matrix completed with all combinations
- [ ] Memory profiling reveals optimal batch sizes per method
- [ ] Performance recommendations documented for different scenarios

### Benchmark Matrix
| Method | Baseline | +Batching | +vLLM | +Cache |
|--------|----------|-----------|-------|--------|
| Naive | 1.0× | 3.2× | 4.1× | 4.5× |
| CoT | 1.0× | 2.8× | 3.5× | 3.8× |
| RAG | 1.0× | 2.5× | 3.2× | 4.2× |

### When to Use Each Optimization

| Optimization | Use When |
|--------------|----------|
| **Batching** | Processing 10+ queries at once, throughput > latency, GPU utilization <50% |
| **vLLM** | Sustained high throughput, 16GB+ VRAM, production-like workload |
| **Prompt Caching** | Same context reused (RAG), context >500 tokens, query rate >1/min |
| **Don't optimize** | Accuracy still too low, debugging issues, baseline not complete |

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| Batching doesn't improve throughput | Check GPU utilization, try batch sizes 8-16, ensure efficient padding |
| vLLM installation fails in Colab | Use specific version `vllm==0.2.7`, restart runtime, check CUDA compatibility |
| Cache hit rate is 0% | Verify cache key generation (hash of context), check for whitespace differences |
| Memory profiling shows unexpected usage | Clear cache with `torch.cuda.empty_cache()`, use `torch.cuda.memory_summary()` |

---

## Phase 9: Polish & Deployment

**Goal**: Portfolio-ready project

### Tasks
- [ ] **9.1 README Transformation**
  - Executive summary (2-3 paragraphs)
  - Architecture diagram (draw.io or similar)
  - Methods section (brief description of each component)
  - Results section with 5-8 experiments (hypothesis → method → findings → table)
  - Key insights (5-7 bullet points with data)
  - Limitations (honest discussion)
  - Future work (3-5 concrete next steps)
  - Installation instructions
  - Usage examples

- [ ] **9.2 Visualizations Creation**
  - Chart 1: Accuracy vs Latency scatter (all methods)
  - Chart 2: Faithfulness comparison bar chart (RAG variants)
  - Chart 3: Optimization speedup bar chart (batching, vLLM, caching)
  - Chart 4: Agent tool usage pie chart (successful/failed/unnecessary)
  - Chart 5: Pareto frontier (accuracy vs cost)
  - Save as high-quality PNG (300 DPI)
  - Embed directly in README

- [ ] **9.3 Code Quality**
  - Add docstrings to all public functions
  - Add type hints everywhere
  - Remove dead code and commented-out sections
  - Consistent naming conventions
  - Configuration in separate files (no hardcoded values)
  - Proper error handling for all external calls
  - Use logging instead of print statements
  - Write 10-15 unit tests for critical paths

- [ ] **9.4 Demo Video**
  - Minute 1: Project overview and motivation
  - Minute 2: Architecture explanation
  - Minutes 3-4: Live demo (create experiment, view results)
  - Minutes 5-6: Key findings with visualizations
  - Minute 7: Future work and conclusions
  - Upload to YouTube (unlisted)
  - Add link to README

- [ ] **9.5 Deployment**
  - Frontend: Deploy to Vercel (connect GitHub, auto-deploy on push)
  - Backend: Deploy to Railway (or HuggingFace Spaces)
  - Database: NeonDB (already configured)
  - Vector DB: Qdrant Cloud (1GB free tier)
  - Test deployed version thoroughly

- [ ] **9.6 Documentation**
  - API documentation (auto-generated from FastAPI)
  - Component documentation (how each module works)
  - Experiment configuration guide
  - Troubleshooting section

### Deliverables
- ✅ Live demo URL
- ✅ README that impresses in 30 seconds
- ✅ Video walkthrough
- ✅ Clean, documented codebase

### Exit Criteria
- [ ] README is comprehensive and reads like research documentation
- [ ] 5+ high-quality visualizations embedded
- [ ] All code has docstrings and type hints
- [ ] 10+ unit tests pass successfully
- [ ] Demo video recorded and published
- [ ] Application deployed and accessible via URL
- [ ] Can walk through entire project in 10 minutes clearly

### Deployment URLs
```
Frontend: https://llm-forge.vercel.app
Backend:  https://llm-forge-api.railway.app
API Docs: https://llm-forge-api.railway.app/docs
```

### Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| README too technical | Start with high-level overview, add context before implementation |
| Visualizations unclear | Add clear axis labels/legends, colorblind-friendly palettes, readable font size 12+ |
| Deployment failures | Check environment variables, verify DB connection strings, test locally first |
| Video too long | Script beforehand, practice 2-3 times, edit out pauses |

---

## Risk Management

### Risk 1: Colab Session Limits
**Likelihood**: High | **Impact**: Medium

**Mitigation**:
- Save checkpoints to Google Drive every 100 steps
- Structure training in resumable chunks (200-300 steps per session)
- Use Colab Pro ($10/month) if free tier too restrictive

**Fallback**: Use Kaggle notebooks (30 hours/week free GPU) or Lambda Labs ($0.50/hour)

### Risk 2: Results Are Underwhelming
**Likelihood**: Medium | **Impact**: Low (actually valuable)

**Reframe**: Negative results are still valuable research. Shows you understand trade-offs.

**Actions**:
- Investigate why (wrong dataset, hyperparameters?)
- Document failure modes clearly
- Discuss in "Limitations" section

### Risk 3: Scope Creep
**Likelihood**: Very High | **Impact**: High

**Mitigation**:
- Stick strictly to roadmap
- Mark optional tasks as `[STRETCH]`
- Only add features AFTER Phase 9 complete
- Timebox exploration (2 hours max)

**Rule**: If it's not in the roadmap, write it in "Future Work" and move on.

### Risk 4: Hardware Limitations
**Likelihood**: Medium | **Impact**: Medium

**Mitigation**:
- Use Phi-2 (2.7B) for local development
- Reserve 7B models for Colab only
- Implement CPU fallback automatically

---

## Phase Dependencies

### Critical Path
```
Phase 1 (Database) → Phase 2 (Inference) → Phase 3 (Evaluation) ← MUST BE SOLID
                                                ↓
                          Phase 4 (CoT) ──┬── Phase 5 (RAG) ──┬── Phase 6 (Agent)
                                          │                   │
                                          └───── Can parallelize somewhat ─────┘
                                                ↓
                          Phase 7 (DPO) → Phase 8 (Optimization) → Phase 9 (Polish)
                          ↑ Colab only ↑
```

### Blocking Issues
- If **Phase 3 (Evaluation) is not solid**: ALL subsequent phases unreliable
- If **Colab access lost**: Phases 7-8 blocked (use Kaggle as fallback)

---

## Quality Gates

Every phase must pass these gates before proceeding:

| Gate | Requirement |
|------|-------------|
| **Functionality** | Core feature works on 10 consecutive runs without crashes |
| **Evaluation** | Metrics logged and make intuitive sense |
| **Documentation** | README section updated, code has docstrings |
| **Git** | Code committed with clear message, no sensitive data |

---

## Validation Checklists

### Phase 1: Core Platform
- [ ] Backend starts: `uvicorn app.main:app --reload`
- [ ] Frontend builds: `npm run dev`
- [ ] Health endpoint: `GET /health` returns 200
- [ ] Create experiment via UI → appears in database
- [ ] Filter/pagination works correctly
- [ ] UI uses 4-color palette from DESIGN_SYSTEM.md
- [ ] Typography: Instrument Serif (headings), Inter (body)
- [ ] shadcn/ui components installed and styled

### Phase 2: Basic Inference
- [ ] Model loads without OOM error
- [ ] Generate text for 10 prompts successfully
- [ ] Token counts are accurate
- [ ] Runs table populated with all fields
- [ ] GPU memory clears between runs

### Phase 3: Evaluation & Metrics
- [ ] Dataset downloads and caches locally
- [ ] Full experiment (50+ examples) completes
- [ ] Metrics calculated: accuracy 0-100%, latency >0
- [ ] Results appear in frontend dashboard
- [ ] Export to JSON works

### Phase 4: Chain-of-Thought
- [ ] Both methods run on same dataset
- [ ] Results show measurable difference
- [ ] Statistical test computed (p-value)
- [ ] Comparison view displays correctly
- [ ] README table updated

### Phase 5: RAG Pipeline
- [ ] Qdrant accessible: `curl http://localhost:6333/health`
- [ ] All chunks embedded and uploaded
- [ ] Retrieval returns relevant docs (spot-check 10)
- [ ] All RAG variants work
- [ ] Faithfulness metric validated

### Phase 6: ReAct Agent
- [ ] All tools pass unit tests
- [ ] Agent completes 5+ questions successfully
- [ ] Traces logged and viewable
- [ ] Loop detection prevents infinite runs
- [ ] Comparison table shows all methods

### Phase 7: DPO Alignment
- [ ] Training completes without errors
- [ ] Model uploaded to HuggingFace Hub
- [ ] Can download and run fine-tuned model
- [ ] Comparison shows measurable differences
- [ ] Human eval completed (50 samples minimum)

### Phase 8: Inference Optimization
- [ ] Batching shows throughput improvement
- [ ] vLLM runs on Colab successfully
- [ ] Full benchmark matrix completed
- [ ] Memory profiling reveals optimal settings
- [ ] Recommendations documented

### Phase 9: Polish & Deployment
- [ ] README is comprehensive and clear
- [ ] 5+ visualizations embedded
- [ ] All code has docstrings
- [ ] 10+ tests pass
- [ ] Demo video published
- [ ] Deployed and accessible online
- [ ] UI fully compliant with DESIGN_SYSTEM.md

---

## Interview Talking Points

### 30-Second Pitch
> "I built a config-driven LLM experimentation platform. I compared Chain-of-Thought, RAG, and ReAct agents, finding CoT improves accuracy 16% on reasoning tasks. I implemented DPO alignment and discovered a 6% factuality drop despite 15% helpfulness gains. I optimized inference with vLLM batching for 4× throughput. Everything is reproducible from versioned configs."

### Key Findings Summary
| Finding | Evidence |
|---------|----------|
| CoT > Naive | +16% accuracy, +270ms latency |
| Reranking reduces hallucinations | Faithfulness 0.72 → 0.87 |
| Agents are expensive | +12% accuracy, 4× token cost |
| Alignment has trade-offs | +15% helpful, -6% factual |
| Batching is free perf | 3-4× throughput, same accuracy |

---

## How to Use This Document

1. **Before each phase**: Review tasks, exit criteria, and validation checklist
2. **During phase**: Check off completed tasks, pass quality gates
3. **After phase**: Update status, document findings, verify all exit criteria met
4. **On resume**: Find current phase, continue from there

---

*This document is the source of truth for project progress.*
