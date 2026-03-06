# ADR 002: In-Process LRU Cache over Redis

**Status**: Accepted  
**Date**: 2024-05-15

### Context
When running extensive evaluations (like RAG or Chain-of-Thought), the pipeline frequently makes identical sub-queries to the LLM (e.g., identical routing prompts or extraction formatting). Caching these API calls is essential to preserve free-tier request limits and accelerate experiment times.

Standard practice for distributed caching is Redis. However, adding a Redis dependency violates our goal of a zero-cost, easily deployable portfolio project.

### Decision
We implemented an in-memory `functools.lru_cache` (configured in the DB via `OptimizationConfig`) directly inside the API engine adapters (`hf_api_engine.py`, `openai_engine.py`). 

### Consequences
- **Positive:** Zero added infrastructure cost.
- **Positive:** Zero network latency for cache hits (reads are nanosecond-fast).
- **Negative:** The cache does not survive application restarts or HF Space sleep cycles. 
- **Negative:** The cache cannot be shared if the application is ever scaled to multiple workers.

### Constraints Addressed
Given the Free Tier limits of Groq and Hugging Face Inference API, saving redundant calls is paramount. The transient nature of the cache is an acceptable tradeoff since experiments generally consume the most redundant prompts within a single contiguous execution window.
