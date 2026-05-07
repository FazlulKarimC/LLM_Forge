"""Routing telemetry routes for experiment results."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.result import Result

router = APIRouter()


@router.get("/{experiment_id}/routing")
async def get_routing_telemetry(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get routing telemetry for an experiment."""
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()

    if not db_result:
        raise HTTPException(status_code=404, detail="No results found for experiment")

    raw = db_result.raw_metrics or {}
    routing = raw.get("routing")

    if routing is None:
        raise HTTPException(status_code=404, detail="No routing telemetry for this experiment")

    return routing
