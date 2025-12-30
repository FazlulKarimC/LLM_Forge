"""
Experiment Service

Business logic for experiment management.
Handles CRUD operations and experiment execution orchestration.

TODO (Iteration 1): Implement CRUD operations
TODO (Iteration 2): Add execution orchestration
TODO (Iteration 3): Add experiment cloning and versioning
"""

from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatus,
)


class ExperimentService:
    """
    Service for experiment management.
    
    Responsibilities:
    - Create experiments from config
    - List and filter experiments
    - Update experiment status
    - Orchestrate experiment execution
    - Delete experiments (soft delete)
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize with database session.
        
        Args:
            db: Async database session from dependency injection
        """
        self.db = db
    
    async def create(self, data: ExperimentCreate) -> ExperimentResponse:
        """
        Create a new experiment.
        
        Args:
            data: Experiment creation request
        
        Returns:
            Created experiment with generated ID
        
        TODO (Iteration 1): Implement database persistence
        """
        # TODO: Implement
        # experiment = Experiment(
        #     name=data.name,
        #     description=data.description,
        #     config=data.config.model_dump(),
        #     method=data.config.reasoning_method.value,
        #     model_name=data.config.model_name,
        #     dataset_name=data.config.dataset_name,
        #     status=ExperimentStatus.PENDING,
        # )
        # self.db.add(experiment)
        # await self.db.flush()
        # return ExperimentResponse.model_validate(experiment)
        raise NotImplementedError("Iteration 1: Implement create")
    
    async def get(self, experiment_id: UUID) -> Optional[ExperimentResponse]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: UUID of experiment
        
        Returns:
            Experiment or None if not found
        
        TODO (Iteration 1): Implement database lookup
        """
        # TODO: Implement
        # result = await self.db.execute(
        #     select(Experiment).where(Experiment.id == experiment_id)
        # )
        # experiment = result.scalar_one_or_none()
        # if experiment:
        #     return ExperimentResponse.model_validate(experiment)
        # return None
        raise NotImplementedError("Iteration 1: Implement get")
    
    async def list(
        self,
        status: Optional[ExperimentStatus] = None,
        method: Optional[str] = None,
        model: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> ExperimentListResponse:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status
            method: Filter by reasoning method
            model: Filter by model name
            skip: Pagination offset
            limit: Max results
        
        Returns:
            Paginated list of experiments
        
        TODO (Iteration 1): Implement basic listing
        TODO (Iteration 2): Add sorting options
        """
        # TODO: Implement
        raise NotImplementedError("Iteration 1: Implement list")
    
    async def update_status(
        self,
        experiment_id: UUID,
        status: ExperimentStatus,
        error_message: Optional[str] = None,
    ) -> Optional[ExperimentResponse]:
        """
        Update experiment status.
        
        Args:
            experiment_id: UUID of experiment
            status: New status
            error_message: Error details if failed
        
        Returns:
            Updated experiment
        
        TODO (Iteration 1): Implement status update
        """
        # TODO: Implement
        raise NotImplementedError("Iteration 1: Implement update_status")
    
    async def delete(self, experiment_id: UUID) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: UUID of experiment
        
        Returns:
            True if deleted, False if not found
        
        TODO (Iteration 1): Implement soft delete
        """
        # TODO: Implement
        raise NotImplementedError("Iteration 1: Implement delete")
    
    async def execute(self, experiment_id: UUID) -> None:
        """
        Execute an experiment.
        
        This is the main orchestration method:
        1. Load experiment config
        2. Initialize inference engine
        3. Load dataset
        4. Run inference for each sample
        5. Log results
        6. Compute metrics
        7. Update status
        
        Should be run as a background task.
        
        TODO (Iteration 2): Implement execution orchestration
        """
        # TODO: Implement
        raise NotImplementedError("Iteration 2: Implement execute")
