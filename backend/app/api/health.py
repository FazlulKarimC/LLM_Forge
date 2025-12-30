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
        - Database connection
        - Vector database connection (TODO)
        - Model availability (TODO)
    
    Returns:
        Status of each dependency.
    
    TODO (Iteration 1): Add database ping
    TODO (Iteration 2): Add vector DB check
    TODO (Iteration 3): Add model availability check
    """
    checks = {
        "database": "unknown",
        "vector_db": "not_implemented",
        "models": "not_implemented",
    }
    
    # TODO: Implement actual checks
    try:
        # Placeholder - will ping database
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    all_healthy = all(v == "healthy" for v in checks.values() if v != "not_implemented")
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
    }
