# ADR 001: Thread Pools over Celery for Experiment Execution

**Status**: Accepted  
**Date**: 2024-05-15

### Context
LLMForge evaluates hundreds of LLM generations per experiment. Typically, ML orchestration systems (like MLflow or Promptfoo's heavier modes) use distributed task queues like Celery with Redis/RabbitMQ to manage these long-running jobs.

However, a core constraint of this project is to run on **100% free-tier infrastructure**.
1. **Frontend**: Vercel Serverless (strict 10-15s timeouts, kills background threads immediately upon HTTP response).
2. **Backend**: Hugging Face Spaces (Docker Space — 2 vCPU, 16GB RAM constraints).

Deploying a Redis instance plus a Celery worker on free tiers adds significant operational overhead, introduces points of failure, and often requires paid databases for reliable hosting.

### Decision
We use Python's built-in `asyncio.to_thread()` combined with FastAPI's `BackgroundTasks` to execute experiments within the same single Docker container hosted on Hugging Face Spaces.

The Vercel frontend only triggers the job via a fast, short-lived HTTP POST `/api/v1/experiments/{id}/execute` endpoint and immediately receives a `202 Accepted` response. The heavy lifting happens exclusively on the HF Space backend.

### Consequences
- **Positive:** Zero external infrastructure dependencies. The entire app ships as a single container, making it trivial for reviewers to deploy a clone.
- **Positive:** Saves free-tier database connections (Celery requires persistent broker connections).
- **Negative:** If the HF Space container sleeps or restarts mid-experiment, the in-memory queue is lost.
- **Negative:** We cannot easily scale horizontally across multiple instances (though HF free tier doesn't support scaling anyway).

### Constraints Addressed
- **Vercel Timeout Landmine**: By strictly separating the UI (Vercel) from the execution engine (HF Spaces), we avoid Vercel's serverless thread-killing behavior entirely. Long-evaluations run safely inside the persistent Docker container.
