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
    """
    Readiness check for dependencies.
    
    Checks (run concurrently):
        - Database connection (NeonDB)
        - Vector database connection (Qdrant Cloud)
        - Model API availability (HuggingFace token)
        - Task dispatch / Upstash / RQ worker status
    
    Returns:
        Status of each dependency.
    """
    # Run all checks concurrently to minimize latency
    db_result, vector_result, model_result, dispatch_result = await asyncio.gather(
        _check_database(db),
        _check_vector_db(),
        _check_models(),
        _check_dispatch(),
        return_exceptions=True,
    )

    checks = {}

    # Database
    if isinstance(db_result, Exception):
        checks["database"] = f"unhealthy: {str(db_result)[:120]}"
    else:
        checks["database"] = db_result

    # Vector DB
    if isinstance(vector_result, Exception):
        checks["vector_db"] = f"unhealthy: {str(vector_result)[:120]}"
    else:
        checks["vector_db"] = vector_result

    # Models
    if isinstance(model_result, Exception):
        checks["models"] = f"unhealthy: {str(model_result)[:120]}"
    else:
        checks["models"] = model_result

    # Dispatch (task_dispatch, upstash, rq_worker)
    if isinstance(dispatch_result, Exception):
        checks["task_dispatch"] = f"unhealthy: {str(dispatch_result)[:120]}"
        checks["upstash"] = "unknown"
        checks["rq_worker"] = "unknown"
    else:
        checks.update(dispatch_result)

    database_ready = checks.get("database") in _CRITICAL_READY_VALUES
    models_ready = checks.get("models") in _CRITICAL_READY_VALUES
    dispatch_ready = checks.get("task_dispatch") in _READY_TASK_DISPATCH_STATUSES
    ready = database_ready and models_ready and dispatch_ready

    optional_degraded = any(
        checks.get(key) not in ("healthy", "not_configured")
        for key in ("upstash", "rq_worker")
    )
    critical_degraded = checks.get("task_dispatch") in {"inline_only", "fallback_inline"}
    degraded = ready and (optional_degraded or critical_degraded)

    return {
        "status": "ready" if ready else "not_ready",
        "mode": "degraded" if degraded else ("healthy" if ready else "down"),
        "checks": checks,
    }


# ── Individual check coroutines ─────────────────────────────────────────


async def _check_database(db: AsyncSession) -> str:
    """Check database connectivity."""
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        return "healthy"
    except Exception as e:
        return f"unhealthy: {str(e)[:120]}"


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
        return f"unhealthy: {str(e)[:120]}"


async def _check_models() -> str:
    """
    Check whether at least one configured provider path is usable.

    Hugging Face gets a live token validation because the SDK is already present.
    OpenRouter and Groq are treated as configured/usable when their API keys are
    present so /ready does not incorrectly fail a deployment that intentionally
    routes away from Hugging Face.
    """
    try:
        from app.core.config import settings as _settings

        configured_non_hf = any(
            (_settings.OPENROUTER_API_KEY, _settings.GROQ_API_KEY)
        )
        if _settings.HF_TOKEN:
            from huggingface_hub import HfApi

            api = HfApi(token=_settings.HF_TOKEN)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, api.whoami)
            return "healthy"

        if configured_non_hf:
            return "healthy"

        if not _settings.HF_TOKEN:
            return "not_configured"
    except Exception as e:
        return f"unhealthy: {str(e)[:120]}"


async def _check_dispatch() -> dict:
    """
    Get dispatch readiness snapshot.
    
    Returns a dict with keys: task_dispatch, upstash, rq_worker.
    """
    try:
        from app.core.task_dispatch import get_dispatch_readiness_snapshot
        return await get_dispatch_readiness_snapshot()
    except Exception as e:
        logger.warning("Dispatch readiness check failed: %s", e)
        return {
            "task_dispatch": f"unhealthy: {str(e)[:120]}",
            "upstash": "unknown",
            "rq_worker": "unknown",
        }
