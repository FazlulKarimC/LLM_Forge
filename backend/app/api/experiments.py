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
    """
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Experiment already running")
    
    # Queue background task with standalone execution function
    # This creates its own database session to avoid session lifecycle issues
    background_tasks.add_task(execute_experiment_background, experiment_id)
    
    updated_experiment = await service.update_status(experiment_id, ExperimentStatus.RUNNING)
    await db.commit()  # Ensure status change is persisted before returning
    
    return updated_experiment


async def execute_experiment_background(experiment_id: UUID):
    """
    Background task execution function.
    
    Creates its own database session to avoid lifecycle issues.
    
    DEBUGGING: Enhanced logging to track execution flow.
    """
    import logging
    import traceback
    import sys
    
    # Configure logging to ensure output is visible
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"="*60)
    logger.info(f"🚀 BACKGROUND TASK STARTED")
    logger.info(f"   Experiment ID: {experiment_id}")
    logger.info(f"="*60)
    print(f"\n{'='*60}")
    print(f"🚀 BACKGROUND TASK STARTED")
    print(f"   Experiment ID: {experiment_id}")
    print(f"{'='*60}")
    
    from app.core.database import async_session_maker
    
    try:
        logger.info(f"[STEP 1] Creating database session...")
        print(f"[STEP 1] Creating database session...")
        
        async with async_session_maker() as db:
            logger.info(f"✓ Database session created successfully")
            print(f"✓ Database session created successfully")
            
            logger.info(f"[STEP 2] Creating ExperimentService...")
            print(f"[STEP 2] Creating ExperimentService...")
            service = ExperimentService(db)
            logger.info(f"✓ ExperimentService created")
            print(f"✓ ExperimentService created")
            
            logger.info(f"[STEP 3] Calling service.execute()...")
            print(f"[STEP 3] Calling service.execute()...")
            
            await service.execute(experiment_id)
            
            logger.info(f"[STEP 4] Committing database changes...")
            print(f"[STEP 4] Committing database changes...")
            await db.commit()
            
            logger.info(f"="*60)
            logger.info(f"✅ BACKGROUND TASK COMPLETED SUCCESSFULLY!")
            logger.info(f"="*60)
            print(f"{'='*60}")
            print(f"✅ BACKGROUND TASK COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        logger.error(f"="*60)
        logger.error(f"❌ BACKGROUND TASK FAILED!")
        logger.error(f"   Error Type: {error_type}")
        logger.error(f"   Error Message: {error_msg}")
        logger.error(f"="*60)
        print(f"{'='*60}")
        print(f"❌ BACKGROUND TASK FAILED!")
        print(f"   Error Type: {error_type}")
        print(f"   Error Message: {error_msg}")
        print(f"{'='*60}")
        
        # Print full traceback
        logger.error(f"Full Traceback:")
        traceback.print_exc()
        
        # Update status to failed in a NEW session (important!)
        logger.info(f"[RECOVERY] Updating experiment status to FAILED...")
        print(f"[RECOVERY] Updating experiment status to FAILED...")
        
        try:
            async with async_session_maker() as db:
                service = ExperimentService(db)
                await service.update_status(
                    experiment_id,
                    ExperimentStatus.FAILED,
                    error_message=f"{error_type}: {error_msg}"
                )
                await db.commit()  # CRITICAL: Commit the failed status!
                
                logger.info(f"✓ Status updated to FAILED and committed")
                print(f"✓ Status updated to FAILED and committed")
                
        except Exception as update_error:
            logger.error(f"❌ CRITICAL: Could not update status to FAILED: {update_error}")
            print(f"❌ CRITICAL: Could not update status to FAILED: {update_error}")
            traceback.print_exc()


@router.post("/{experiment_id}/execute", status_code=202)
async def execute_experiment_alias(
    experiment_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Alias endpoint for /run.
    
    Some frontends may call /execute instead of /run.
    This provides compatibility for both conventions.
    
    Returns:
        202 Accepted with message indicating background execution started
    """
    service = ExperimentService(db)
    experiment = await service.get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Experiment already running")
    
    # Queue background task
    background_tasks.add_task(execute_experiment_background, experiment_id)
    
    # Update status to running
    await service.update_status(experiment_id, ExperimentStatus.RUNNING)
    await db.commit()
    
    return {
        "message": "Experiment execution started in background",
        "experiment_id": str(experiment_id),
        "status": "running"
    }


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
