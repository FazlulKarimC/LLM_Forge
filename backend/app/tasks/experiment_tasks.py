"""
Experiment Tasks for RQ

Background task functions for experiment execution.
These run in the RQ worker process, separate from the API.

Architecture:
- API validates and enqueues job
- RQ worker picks up job
- Task creates own DB session
- ExperimentService.execute() owns all commits/rollbacks
"""

import asyncio
import logging
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def run_experiment_task(
    experiment_id: str,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None
) -> None:
    """
    RQ task wrapper for experiment execution.
    
    This is the entry point for RQ worker. It:
    1. Creates its own database session
    2. Calls ExperimentService.execute()
    3. Service handles all commits and status updates
    
    Args:
        experiment_id: UUID string of experiment to run
    
    Note:
        RQ doesn't support async functions directly, so we use
        asyncio.run() to execute the async code.
    """
    logger.info(f"[RQ TASK] Starting experiment: {experiment_id}")
    print(f"[RQ TASK] Starting experiment: {experiment_id}")
    
    try:
        asyncio.run(_run_experiment_async(
            UUID(experiment_id), 
            custom_base_url=custom_base_url, 
            custom_api_key=custom_api_key
        ))
        logger.info(f"[RQ TASK] Completed experiment: {experiment_id}")
        print(f"[RQ TASK] ✅ Completed experiment: {experiment_id}")
    except Exception as e:
        logger.error(f"[RQ TASK] Failed experiment {experiment_id}: {e}")
        print(f"[RQ TASK] ❌ Failed experiment {experiment_id}: {e}")
        raise  # Let RQ handle the failure


async def _run_experiment_async(
    experiment_id: UUID,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None
) -> None:
    """
    Async implementation of experiment execution.
    
    Creates database session and delegates to ExperimentService.
    Service owns all transaction management.
    """
    # Import here to avoid circular imports at module load time
    from app.core.database import async_session_maker
    from app.services.experiment_service import ExperimentService
    
    async with async_session_maker() as db:
        service = ExperimentService(db)
        await service.execute(
            experiment_id, 
            custom_base_url=custom_base_url, 
            custom_api_key=custom_api_key
        )
