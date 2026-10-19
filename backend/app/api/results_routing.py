"""Routing telemetry routes for experiment results."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.result import Result
from app.services.attempts import result_at

router = APIRouter()


@router.get("/{experiment_id}/routing")
async def get_routing_telemetry(
    experiment_id: UUID,
    attempt: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Get routing telemetry for an experiment."""
    db_result = await result_at(db, experiment_id, attempt)

    if not db_result:
        raise HTTPException(status_code=404, detail="No results found for experiment")

    raw = db_result.raw_metrics or {}
    routing = raw.get("routing")

    if routing is None:
        raise HTTPException(status_code=404, detail="No routing telemetry for this experiment")

    return routing
