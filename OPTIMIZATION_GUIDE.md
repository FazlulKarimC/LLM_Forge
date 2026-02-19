# Optimization Guide — Phase 8

## Overview

LlmForge provides three optimization knobs for inference performance:

| Feature | What It Does | Best For |
|---------|-------------|----------|
| **Batching** | Parallelizes N API calls via ThreadPoolExecutor | High-throughput experiments with many samples |
| **Caching** | LRU cache keyed by (prompt + params) hash | Datasets with repeated/similar prompts |
| **Profiling** | Times each phase (prompt_build, api_call, parsing, metrics) | Identifying bottlenecks |

---

## Decision Framework

```
             ┌──────────────────────┐
             │  Many samples (>20)? │
             └──────┬───────────────┘
                    │
              Yes ──┤── No → Baseline is fine
                    │
        ┌───────────▼───────────────┐
        │ Using ReAct Agent method? │
        └───────────┬───────────────┘
                    │
              Yes ──┤── No → Enable Batching (batch_size=8)
                    │
                    └→ Batching disabled automatically.
                       Consider Caching if tool calls repeat.
```

### When to Enable Batching
- **Always** for Naive and CoT methods with >20 samples
- `batch_size=8` is a good default for free-tier HF API
- Higher batch sizes (16-32) may hit rate limits
- **Not available** for ReAct agent (iterative tool-calling)

### When to Enable Caching
- Datasets with duplicate or near-duplicate prompts
- Re-running experiments with identical configs (reproducibility testing)
- Low-temperature settings (deterministic outputs)
- **Not useful** for unique prompts with high temperature

### Profiling
- Always enabled by default (minimal overhead)
- Helps identify if bottleneck is API calls vs. prompt building vs. metrics
- Data stored in `Result.raw_metrics["optimization"]`

---

## Configuration

### Via Frontend
1. Go to **New Experiment** page
2. Scroll to **⚡ Optimization** section
3. Toggle **Enable Batching** and/or **Enable Caching**
4. Adjust batch size (1-32) and cache size (16-2048)

### Via API
```json
{
  "config": {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "reasoning_method": "naive",
    "dataset_name": "trivia_qa",
    "num_samples": 100,
    "optimization": {
      "enable_batching": true,
      "batch_size": 8,
      "enable_caching": true,
      "cache_max_size": 256,
      "enable_profiling": true
    }
  }
}
```

---

## Viewing Results

After running an optimized experiment:

1. Go to the experiment detail page
2. Scroll below the Results section
3. The **⚡ Optimization Profile** section shows:
   - **Total Wall Time** — end-to-end execution duration
   - **Cache Hit Rate** — percentage of prompts served from cache
   - **Batches Processed** — number of concurrent batch groups
   - **Timing Breakdown** — per-phase mean/p50/p95 latencies

### API Endpoint
```
GET /api/v1/results/{experiment_id}/profile
```

---

## Benchmarking

Run the benchmark matrix to compare strategies:

```bash
cd LlmForge
python scripts/run_benchmark.py --samples 20 --output benchmark_results.json
```

This generates experiments with all combinations of methods (naive, cot) and optimizations (baseline, batched_4, batched_8, cached, batched+cached).

---

## Expected Performance Gains

| Optimization | Expected Improvement | Notes |
|-------------|---------------------|-------|
| Batching (8) | 2-3x throughput | Parallelizes API latency |
| Caching | Up to 10x for repeat prompts | Zero cost for cache hits |
| Both | 3-5x overall | Compound benefits |

> **Note**: Actual gains depend on API rate limits, network latency, and prompt uniqueness.
