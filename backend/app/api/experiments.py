"""
Experiment API Routes

CRUD operations for experiments:
- Create experiment from config
- List experiments with filtering
- Get experiment details
- Run experiment (trigger inference)
- Delete experiment

TODO (Iteration 1): Implement create and list
TODO (Iteration 2): Implement run with background tasks
TODO (Iteration 3): Add experiment comparison endpoint
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatus,
)
from app.services.experiment_service import ExperimentService

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    experiment: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new experiment from configuration.
    
    The experiment config defines:
    - Model to use (e.g., microsoft/phi-2)
    - Reasoning method (naive, cot, react)
    - Dataset for evaluation
    - Hyperparameters (temperature, max_tokens, etc.)
    - Random seed for reproducibility
    
    Args:
        experiment: Experiment configuration
    
    Returns:
        Created experiment with generated ID
    
    TODO (Iteration 1): Implement with database persistence
    TODO (Iteration 2): Add config validation against model capabilities
    """
    service = ExperimentService(db)
    return await service.create(experiment)


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    List experiments with optional filtering.
    
    Filters:
        - status: pending, running, completed, failed
        - method: naive, cot, react, rag
        - model: model identifier
    
    Pagination:
        - skip: offset for pagination
        - limit: max results (1-100)
    
    Returns:
        Paginated list of experiments
    
    TODO (Iteration 1): Implement basic listing
    TODO (Iteration 2): Add sorting options
    TODO (Iteration 3): Add date range filtering
    """
    service = ExperimentService(db)
    return await service.list(
        status=status,
        method=method,
        model=model,
        skip=skip,
        limit=limit,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get experiment details by ID.
    
    Returns:
        Full experiment config and results if available
    
    Raises:
        404: Experiment not found
    
    TODO (Iteration 1): Implement with database lookup
    """
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment


@router.post("/{experiment_id}/run", response_model=ExperimentResponse)
async def run_experiment(
    experiment_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger experiment execution.
    
    This endpoint:
    1. Validates experiment exists and is runnable
    2. Updates status to 'running'
    3. Queues inference job in background
    4. Returns immediately with updated status
    
    The actual inference runs asynchronously.
    Poll GET /experiments/{id} for completion.
    
    Args:
        experiment_id: UUID of experiment to run
        background_tasks: FastAPI background task manager
    
    Returns:
        Updated experiment with 'running' status
    
    TODO (Iteration 1): Implement synchronous execution
    TODO (Iteration 2): Add background task processing
    TODO (Iteration 3): Add job queue (Redis/RabbitMQ)
    """
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Experiment already running")
    
    # TODO: Queue background task
    # background_tasks.add_task(service.execute, experiment_id)
    
    return await service.update_status(experiment_id, ExperimentStatus.RUNNING)


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an experiment.
    
    Also deletes associated results and run logs.
    
    TODO (Iteration 1): Implement soft delete
    TODO (Iteration 2): Add cascade delete for results
    """
    service = ExperimentService(db)
    deleted = await service.delete(experiment_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")
