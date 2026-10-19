"""
Health Check Endpoints

Provides endpoints for:
- Basic health check (is the server running?)
- Readiness check (are dependencies connected?)
- Liveness check (is the server responsive?)

Used by:
- Load balancers (Render, Railway)
- Kubernetes probes
- Monitoring systems
"""

import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

_READY_TASK_DISPATCH_STATUSES = {"healthy", "inline_only", "fallback_inline"}
_CRITICAL_READY_VALUES = {"healthy"}


@router.get("/")
async def root():
    """
    Root endpoint — satisfies the Hugging Face Spaces health probe.
    HF Spaces sends GET / to check if the container is alive; without
    this route it gets a 404 and marks the Space as unhealthy.
    """
    return {"status": "healthy", "service": "llmforge-backend"}


@router.get("/health")
async def health_check():
    """
    Basic health check.
    
    Returns:
        Simple status indicating the server is running.
        Does NOT check dependencies.
    """
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Check the core database/schema; optional services never gate readiness."""
    try:
        database = await asyncio.wait_for(_check_database(db), timeout=12)
    except Exception:
        database = "unhealthy: dependency check failed"
    from app.services.evaluation_runner import runner
    from app.core.config import settings
    checks = {"database": database, "evaluation_runner": runner.health(),
              "models": await _check_models(), "vector_db": "optional",
              "task_dispatch": "inline_only" if settings.QUEUE_BACKEND_MODE == "inline" else "optional"}
    ready = database == "healthy"
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ready" if ready else "not_ready", "mode": "healthy" if ready else "down", "checks": checks})


async def _check_database(db: AsyncSession) -> str:
    """Check database connectivity."""
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        capacity = await db.scalar(text("SELECT id FROM evaluation_capacity WHERE id = 1"))
        if capacity != 1:
            return "unhealthy: evaluation capacity missing"
        await db.execute(text("SELECT experiment_id, number, storage_bytes FROM experiment_attempts LIMIT 0"))
        await db.execute(text("SELECT experiment_id, attempt FROM results LIMIT 0"))
        await db.commit()
        return "healthy"
    except Exception:
        await db.rollback()
        return "unhealthy: dependency check failed"


async def _check_vector_db() -> str:
    """Check Qdrant vector database connectivity."""
    try:
        from app.core.config import settings
        if not settings.QDRANT_API_KEY:
            return "not_configured"

        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=5,
        )
        # Run blocking call in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client.get_collections)
        return "healthy"
    except Exception as e:
        error_msg = str(e).lower()
        if "archived" in error_msg or "paused" in error_msg:
            return "archived (inactive >14 days)"
        return "unhealthy: dependency check failed"


async def _check_models() -> str:
    """Report server provider configuration without a potentially slow API call."""
    from app.core.config import settings

    return "configured" if any(
        (settings.HF_TOKEN, settings.OPENROUTER_API_KEY, settings.GROQ_API_KEY)
    ) else "not_configured"


async def _check_dispatch() -> dict:
    """
    Get dispatch readiness snapshot.
    
    Returns a dict with keys: task_dispatch, upstash, rq_worker.
    """
    try:
        from app.core.task_dispatch import get_dispatch_readiness_snapshot
        return await get_dispatch_readiness_snapshot()
    except Exception:
        logger.warning("Dispatch readiness check failed")
        return {
            "task_dispatch": "unhealthy: dependency check failed",
            "upstash": "unknown",
            "rq_worker": "unknown",
        }
