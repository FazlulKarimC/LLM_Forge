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
from app.schemas.result import ComparisonResponse, ExperimentComparison

router = APIRouter()


@router.get("/compare", response_model=ComparisonResponse)
async def compare_experiments(
    experiment_ids: List[str] = Query(..., description="List of experiment IDs to compare"),
    db: AsyncSession = Depends(get_db),
):
    """Compare metrics across multiple experiments."""
    if len(experiment_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 experiments required for comparison",
        )

    try:
        parsed_ids = [UUID(eid) for eid in experiment_ids]
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")

    comparisons = []
    accuracy_values = []
    f1_values = []
    latency_p50_values = []

    for exp_id in parsed_ids:
        exp_query = select(Experiment).where(Experiment.id == exp_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {exp_id} not found",
            )

        res_query = select(Result).where(Result.experiment_id == exp_id)
        res_result = await db.execute(res_query)
        db_result = res_result.scalar_one_or_none()

        if not db_result:
            raise HTTPException(
                status_code=404,
                detail=f"No results for experiment {exp_id}",
            )

        metrics = result_to_metrics_response(db_result)

        config = experiment.config or {}
        comparisons.append(
            ExperimentComparison(
                experiment_id=exp_id,
                experiment_name=experiment.name,
                method=experiment.method or config.get("reasoning_method", "unknown"),
                model=experiment.model_name or config.get("model_name", "unknown"),
                metrics=metrics,
            )
        )

        accuracy_values.append(db_result.accuracy_exact or 0.0)
        f1_values.append(db_result.accuracy_f1 or 0.0)
        latency_p50_values.append(db_result.latency_p50 or 0.0)

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
        return await stat_service.compare_experiments(parsed_a, parsed_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
