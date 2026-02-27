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

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


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
    
    Checks:
        - Database connection (NeonDB)
        - Vector database connection (Qdrant Cloud)
        - Model API availability (HuggingFace token)
    
    Returns:
        Status of each dependency.
    """
    checks = {
        "database": "unknown",
        "vector_db": "unknown",
        "models": "unknown",
    }
    
    # --- Database (NeonDB) ---
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # --- Vector Database (Qdrant Cloud) ---
    try:
        from app.core.config import settings
        if not settings.QDRANT_API_KEY:
            checks["vector_db"] = "not_configured"
        else:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=5,
            )
            client.get_collections()
            checks["vector_db"] = "healthy"
    except Exception as e:
        error_msg = str(e).lower()
        if "archived" in error_msg or "paused" in error_msg:
            checks["vector_db"] = "archived (inactive >14 days)"
        else:
            checks["vector_db"] = f"unhealthy: {str(e)[:120]}"
    
    # --- Models (HuggingFace API) ---
    try:
        from app.core.config import settings as _settings
        if not _settings.HF_TOKEN:
            checks["models"] = "not_configured"
        else:
            from huggingface_hub import HfApi
            api = HfApi(token=_settings.HF_TOKEN)
            api.whoami()
            checks["models"] = "healthy"
    except Exception as e:
        checks["models"] = f"unhealthy: {str(e)[:120]}"
    
    all_healthy = all(
        v == "healthy"
        for v in checks.values()
        if v not in ("not_configured",)
    )
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
    }
