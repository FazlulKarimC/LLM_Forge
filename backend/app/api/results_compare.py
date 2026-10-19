"""Comparison routes for experiment results."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.results_common import result_to_metrics_response
from app.core.database import get_db
from app.models.experiment import Experiment
from app.models.result import Result
from app.services.attempts import result_at
from app.schemas.result import ComparisonResponse, ExperimentComparison

router = APIRouter()


@router.get("/compare", response_model=ComparisonResponse)
async def compare_experiments(
    experiment_ids: List[str] = Query(..., description="List of experiment IDs to compare"),
    attempt_a: int | None = Query(None, ge=0),
    attempt_b: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Compare metrics across multiple experiments."""
    if len(experiment_ids) != 2 or len(set(experiment_ids)) != 2:
        raise HTTPException(
            status_code=400,
            detail="Exactly 2 distinct experiments required for comparison",
        )

    try:
        parsed_ids = [UUID(eid) for eid in experiment_ids]
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")

    comparisons = []
    accuracy_values = []
    f1_values = []
    latency_p50_values = []

    for index, exp_id in enumerate(parsed_ids):
        exp_query = select(Experiment).where(Experiment.id == exp_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {exp_id} not found",
            )

        db_result = await result_at(db, exp_id, (attempt_a, attempt_b)[index])

        if not db_result:
            raise HTTPException(
                status_code=404,
                detail=f"No results for experiment {exp_id}",
            )

        metrics = result_to_metrics_response(db_result)

        from app.models.experiment_attempt import ExperimentAttempt
        snapshot = await db.get(ExperimentAttempt, (exp_id, db_result.attempt))
        config = (snapshot.config if snapshot else None) or {}
        comparisons.append(
            ExperimentComparison(
                experiment_id=exp_id,
                experiment_name=experiment.name,
                method=config.get("reasoning_method", "unknown"),
                model=config.get("model_name", "unknown"),
                metrics=metrics,
            )
        )

        accuracy_values.append(db_result.accuracy_exact)
        f1_values.append(db_result.accuracy_f1)
        latency_p50_values.append(db_result.latency_p50)

    return ComparisonResponse(
        experiments=comparisons,
        comparison_metrics={
            "accuracy_exact": accuracy_values,
            "accuracy_f1": f1_values,
            "latency_p50": latency_p50_values,
        },
    )


@router.get("/compare/statistical")
async def statistical_comparison(
    experiment_a: str = Query(..., description="First experiment ID"),
    experiment_b: str = Query(..., description="Second experiment ID"),
    attempt_a: int | None = Query(None, ge=0),
    attempt_b: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Statistical comparison between two experiments."""
    from app.services.statistical_service import StatisticalService

    try:
        parsed_a = UUID(experiment_a)
        parsed_b = UUID(experiment_b)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")

    stat_service = StatisticalService(db)

    try:
        return await stat_service.compare_experiments(parsed_a, parsed_b, attempt_a, attempt_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
