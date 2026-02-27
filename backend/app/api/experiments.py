"""
Experiment API Routes

CRUD operations for experiments:
- Create experiment from config
- List experiments with filtering
- Get experiment details
- Run experiment (trigger inference)
- Delete experiment
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatus,
)
from app.core.custom_exceptions import ResourceNotFoundException, ValidationException
from app.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)

router = APIRouter()


async def _execute_inline(
    experiment_id: UUID, 
    custom_base_url: Optional[str] = None, 
    custom_api_key: Optional[str] = None
) -> None:
    """
    Execute experiment inline using a fresh DB session.
    Used as fallback when Redis/RQ is unavailable.
    """
    from app.core.database import async_session_maker
    from app.core.rate_limit import decrement_active_runs
    
    try:
        logger.info(f"[INLINE] Running experiment {experiment_id} inline (no Redis)")
        async with async_session_maker() as session:
            svc = ExperimentService(session)
            await svc.execute(
                experiment_id, 
                custom_base_url=custom_base_url, 
                custom_api_key=custom_api_key
            )
    finally:
        decrement_active_runs()


def _enqueue_or_fallback(
    background_tasks: BackgroundTasks,
    experiment_id: UUID,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
) -> str:
    """
    Try to enqueue via RQ (Redis). If Redis is unavailable or
    we are in development mode, fall back to FastAPI BackgroundTasks
    (async inline execution).
    
    Returns:
        'rq' if enqueued to Redis, 'inline' if using BackgroundTasks
    """
    from app.core.config import settings
    
    # In development, always use inline execution (no RQ worker running)
    if settings.ENVIRONMENT == "development":
        logger.info("Development mode: using inline execution (no RQ worker)")
        background_tasks.add_task(_execute_inline, experiment_id, custom_base_url, custom_api_key)
        return "inline"
    
    try:
        from app.core.redis import get_queue
        from app.tasks.experiment_tasks import run_experiment_task
        queue = get_queue()
        queue.enqueue(
            run_experiment_task, 
            str(experiment_id), 
            custom_base_url=custom_base_url, 
            custom_api_key=custom_api_key
        )
        return "rq"
    except Exception as e:
        logger.warning(f"Redis enqueue failed ({e}), falling back to inline execution")
        background_tasks.add_task(_execute_inline, experiment_id, custom_base_url, custom_api_key)
        return "inline_fallback"


@router.get("/stats")
async def get_experiment_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregated experiment counts by status."""
    service = ExperimentService(db)
    return await service.get_stats()


@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    experiment: ExperimentCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new experiment from configuration."""
    from app.core.rate_limit import check_create_rate_limit
    await check_create_rate_limit(request)
    
    service = ExperimentService(db)
    return await service.create(experiment)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List experiments with optional filtering and pagination."""
    service = ExperimentService(db)
    return await service.list(
        status=status,
        method=method,
        model=model,
        skip=skip,
        limit=limit,
    )


@router.get("/models")
async def list_available_models():
    """
    List available LLM models for experiments.

    Returns curated free-tier models available via HuggingFace Inference API.
    Centralised here so the frontend does not need hardcoded model lists.
    """
    return {
        "models": [
            {
                "value": "meta-llama/Llama-3.2-1B-Instruct",
                "label": "Llama 3.2 (1B)",
                "description": "Fast, efficient — default",
            },
            {
                "value": "meta-llama/Llama-3.2-3B-Instruct",
                "label": "Llama 3.2 (3B)",
                "description": "Stronger small model",
            },
            {
                "value": "meta-llama/Llama-3.1-8B-Instruct",
                "label": "Llama 3.1 (8B)",
                "description": "High capability",
            },
            {
                "value": "Qwen/Qwen2.5-7B-Instruct",
                "label": "Qwen 2.5 (7B)",
                "description": "Powerful 7B model (Free-tier safe)",
            },
        ]
    }


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get experiment details by ID."""
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise ResourceNotFoundException(resource_type="Experiment", resource_id=experiment_id)
    
    return experiment


from fastapi import Header

@router.post("/{experiment_id}/run", response_model=ExperimentResponse)
async def run_experiment(
    experiment_id: UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    x_custom_llm_base: Optional[str] = Header(None, alias="X-Custom-LLM-Base"),
    x_custom_llm_key: Optional[str] = Header(None, alias="X-Custom-LLM-Key"),
):
    """
    Trigger experiment execution.
    
    Tries RQ (Redis) first for production-grade background processing.
    Falls back to FastAPI BackgroundTasks if Redis is unavailable.
    """
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise ResourceNotFoundException(resource_type="Experiment", resource_id=experiment_id)
    
    if experiment.status in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]:
        raise ValidationException(message="Experiment already queued or running")
    
    # Rate limit check
    from app.core.rate_limit import check_run_rate_limit
    await check_run_rate_limit(request)
        
    from app.core.config import settings
    if x_custom_llm_base:
        # Validate URL scheme to prevent SSRF (file://, ftp://, etc.)
        from urllib.parse import urlparse
        parsed = urlparse(x_custom_llm_base)
        if parsed.scheme not in ("http", "https"):
            raise ValidationException(message="Custom LLM base URL must use http:// or https://")
        if not parsed.hostname:
            raise ValidationException(message="Custom LLM base URL must include a valid hostname")
        if settings.ENVIRONMENT != "development":
            raise ValidationException(message="Custom LLM execution is only allowed in development mode")
    
    await service.update_status(experiment_id, ExperimentStatus.QUEUED)
    await db.commit()
    
    try:
        _enqueue_or_fallback(
            background_tasks, 
            experiment_id, 
            custom_base_url=x_custom_llm_base, 
            custom_api_key=x_custom_llm_key
        )
    except Exception as e:
        logger.error("Enqueue failed, rolling back to FAILED: %s", e)
        await service.update_status(
            experiment_id, ExperimentStatus.FAILED,
            error_message="Failed to start execution: task queue unavailable"
        )
        await db.commit()
        raise
    
    return await service.get(experiment_id)



@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete an experiment (soft delete)."""
    service = ExperimentService(db)
    deleted = await service.delete(experiment_id)
    
    if not deleted:
        raise ResourceNotFoundException(resource_type="Experiment", resource_id=experiment_id)
