"""
Results API Routes

Endpoints for experiment results and metrics:
- Get results for an experiment
- Get aggregated metrics
- Compare experiments
- Export results

TODO (Iteration 1): Implement basic results retrieval
TODO (Iteration 2): Add metric computation
TODO (Iteration 3): Add export (CSV, JSON)
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.result import (
    ResultResponse,
    MetricsResponse,
    ComparisonResponse,
)

router = APIRouter()


@router.get("/{experiment_id}", response_model=ResultResponse)
async def get_results(
    experiment_id: UUID,
    include_runs: bool = Query(False, description="Include individual run logs"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get results for a specific experiment.
    
    Args:
        experiment_id: UUID of the experiment
        include_runs: Whether to include per-example run logs
    
    Returns:
        Aggregated results with optional run details
    
    TODO (Iteration 1): Implement basic retrieval
    TODO (Iteration 2): Add pagination for runs
    """
    # TODO: Implement
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{experiment_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get computed metrics for an experiment.
    
    Metrics include:
    - Accuracy (exact match, F1)
    - Faithfulness (NLI-based)
    - Latency (p50, p95, p99)
    - Cost proxies (tokens, GPU time)
    
    TODO (Iteration 1): Implement accuracy metrics
    TODO (Iteration 2): Add faithfulness scoring
    TODO (Iteration 3): Add latency percentiles
    """
    # TODO: Implement
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/compare", response_model=ComparisonResponse)
async def compare_experiments(
    experiment_ids: List[UUID] = Query(..., description="List of experiment IDs to compare"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare metrics across multiple experiments.
    
    Useful for:
    - Ablation studies
    - Method comparisons
    - Hyperparameter tuning
    
    Returns:
        Side-by-side metrics for all specified experiments
    
    TODO (Iteration 2): Implement comparison logic
    TODO (Iteration 3): Add statistical significance testing
    """
    if len(experiment_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 experiments required for comparison"
        )
    
    # TODO: Implement
    raise HTTPException(status_code=501, detail="Not implemented")
