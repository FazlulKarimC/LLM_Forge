"""
Experiment Service

Business logic for experiment management.
Handles CRUD operations and experiment execution orchestration.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatus,
    ExperimentConfig,
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
    
    def _to_response(self, experiment: Experiment) -> ExperimentResponse:
        """Convert database model to response schema."""
        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            config=ExperimentConfig(**experiment.config),
            status=experiment.status,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
            error_message=experiment.error_message,
        )
    
    async def create(self, data: ExperimentCreate) -> ExperimentResponse:
        """
        Create a new experiment.
        
        Args:
            data: Experiment creation request
        
        Returns:
            Created experiment with generated ID
        """
        experiment = Experiment(
            name=data.name,
            description=data.description,
            config=data.config.model_dump(),
            method=data.config.reasoning_method.value,
            model_name=data.config.model_name,
            dataset_name=data.config.dataset_name,
            status=ExperimentStatus.PENDING,
        )
        self.db.add(experiment)
        await self.db.flush()
        await self.db.refresh(experiment)
        return self._to_response(experiment)
    
    async def get(self, experiment_id: UUID) -> Optional[ExperimentResponse]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: UUID of experiment
        
        Returns:
            Experiment or None if not found (excludes soft-deleted)
        """
        result = await self.db.execute(
            select(Experiment).where(
                and_(
                    Experiment.id == experiment_id,
                    Experiment.deleted_at.is_(None)  # Exclude soft-deleted
                )
            )
        )
        experiment = result.scalar_one_or_none()
        if experiment:
            return self._to_response(experiment)
        return None
    
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
        """
        # Build base query (exclude soft-deleted)
        conditions = [Experiment.deleted_at.is_(None)]
        
        # Apply filters
        if status:
            conditions.append(Experiment.status == status)
        if method:
            conditions.append(Experiment.method == method)
        if model:
            conditions.append(Experiment.model_name.ilike(f"%{model}%"))
        
        # Count total matching
        count_query = select(func.count(Experiment.id)).where(and_(*conditions))
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Fetch paginated results
        query = (
            select(Experiment)
            .where(and_(*conditions))
            .order_by(Experiment.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        experiments = result.scalars().all()
        
        return ExperimentListResponse(
            total=total,
            experiments=[self._to_response(exp) for exp in experiments],
            skip=skip,
            limit=limit,
        )
    
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
            Updated experiment or None if not found
        """
        result = await self.db.execute(
            select(Experiment).where(
                and_(
                    Experiment.id == experiment_id,
                    Experiment.deleted_at.is_(None)
                )
            )
        )
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            return None
        
        # Update status
        experiment.status = status
        
        # Set timestamps based on status
        now = datetime.now(timezone.utc)
        if status == ExperimentStatus.RUNNING and experiment.started_at is None:
            experiment.started_at = now
        elif status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            experiment.completed_at = now
        
        # Set error message if provided
        if error_message:
            experiment.error_message = error_message
        
        await self.db.flush()
        await self.db.refresh(experiment)
        return self._to_response(experiment)
    
    async def delete(self, experiment_id: UUID) -> bool:
        """
        Soft delete an experiment.
        
        Args:
            experiment_id: UUID of experiment
        
        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            select(Experiment).where(
                and_(
                    Experiment.id == experiment_id,
                    Experiment.deleted_at.is_(None)
                )
            )
        )
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            return False
        
        # Soft delete: set deleted_at timestamp
        experiment.deleted_at = datetime.now(timezone.utc)
        await self.db.flush()
        return True
    
    async def execute(self, experiment_id: UUID) -> None:
        """
        Execute an experiment.
        
        This is the main orchestration method:
        1. Load experiment config
        2. Initialize inference engine
        3. Load sample questions
        4. Run inference for each sample
        5. Log runs to database
        6. Update status
        
        Should be run as a background task.
        
        DEBUGGING: Enhanced logging to track failures.
        """
        import json
        import os
        import logging
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        from app.schemas.experiment import ExperimentStatus
        from app.services.inference.base import GenerationConfig
        from app.services.inference.hf_api_engine import HFAPIEngine
        from app.services.inference.mock_engine import MockEngine
        from app.services.inference.prompting import NaivePromptTemplate
        from app.services.run_service import RunService
        
        logger.info(f"[EXECUTE] Starting execution for experiment: {experiment_id}")
        print(f"[EXECUTE] Starting execution for experiment: {experiment_id}")
        
        # Step 1: Get experiment
        logger.info(f"[EXECUTE STEP 1] Getting experiment from database...")
        print(f"[EXECUTE STEP 1] Getting experiment from database...")
        experiment_response = await self.get(experiment_id)
        if not experiment_response:
            error = f"Experiment {experiment_id} not found"
            logger.error(f"[EXECUTE] ❌ {error}")
            print(f"[EXECUTE] ❌ {error}")
            raise ValueError(error)
        logger.info(f"[EXECUTE STEP 1] ✓ Found experiment: {experiment_response.name}")
        print(f"[EXECUTE STEP 1] ✓ Found experiment: {experiment_response.name}")
        
        try:
            # Step 1.5: Update status to RUNNING (inside try for error handling)
            logger.info(f"[EXECUTE] Updating status to RUNNING...")
            print(f"[EXECUTE] Updating status to RUNNING...")
            await self.update_status(experiment_id, ExperimentStatus.RUNNING)
            await self.db.commit()
            logger.info(f"[EXECUTE] ✓ Status: RUNNING")
            print(f"[EXECUTE] ✓ Status: RUNNING")
            
            # Step 2: Initialize inference engine
            logger.info(f"[EXECUTE STEP 2] Initializing inference engine...")
            print(f"[EXECUTE STEP 2] Initializing inference engine...")
            engine_type = os.getenv("INFERENCE_ENGINE", "mock")  # mock or hf_api
            logger.info(f"[EXECUTE STEP 2] Engine type: {engine_type}")
            print(f"[EXECUTE STEP 2] Engine type: {engine_type}")
            
            if engine_type == "hf_api":
                engine = HFAPIEngine(model_name=experiment_response.config.model_name)
            else:
                engine = MockEngine()
            
            engine.load_model(experiment_response.config.model_name)
            logger.info(f"[EXECUTE STEP 2] ✓ Engine loaded: {experiment_response.config.model_name}")
            print(f"[EXECUTE STEP 2] ✓ Engine loaded: {experiment_response.config.model_name}")
            
            # Step 3: Load sample questions
            logger.info(f"[EXECUTE STEP 3] Loading sample questions...")
            print(f"[EXECUTE STEP 3] Loading sample questions...")
            questions_path = Path(__file__).parent.parent.parent.parent / "configs" / "sample_questions.json"
            logger.info(f"[EXECUTE STEP 3] Path: {questions_path}")
            print(f"[EXECUTE STEP 3] Path: {questions_path}")
            logger.info(f"[EXECUTE STEP 3] Path exists: {questions_path.exists()}")
            print(f"[EXECUTE STEP 3] Path exists: {questions_path.exists()}")
            
            if not questions_path.exists():
                raise FileNotFoundError(f"Sample questions file not found: {questions_path}")
            
            with open(questions_path, "r") as f:
                questions = json.load(f)
            logger.info(f"[EXECUTE STEP 3] ✓ Loaded {len(questions)} questions")
            print(f"[EXECUTE STEP 3] ✓ Loaded {len(questions)} questions")
            
            # Step 4: Prepare generation config
            logger.info(f"[EXECUTE STEP 4] Preparing generation config...")
            print(f"[EXECUTE STEP 4] Preparing generation config...")
            gen_config = GenerationConfig(
                max_tokens=experiment_response.config.hyperparameters.max_tokens,
                temperature=experiment_response.config.hyperparameters.temperature,
                top_p=experiment_response.config.hyperparameters.top_p,
            )
            logger.info(f"[EXECUTE STEP 4] ✓ Config: max_tokens={gen_config.max_tokens}, temp={gen_config.temperature}")
            print(f"[EXECUTE STEP 4] ✓ Config: max_tokens={gen_config.max_tokens}, temp={gen_config.temperature}")
            
            # Step 5: Initialize run service
            logger.info(f"[EXECUTE STEP 5] Initializing RunService...")
            print(f"[EXECUTE STEP 5] Initializing RunService...")
            run_service = RunService(self.db)
            logger.info(f"[EXECUTE STEP 5] ✓ RunService ready")
            print(f"[EXECUTE STEP 5] ✓ RunService ready")
            
            # Step 6: Run inference for each question
            logger.info(f"[EXECUTE STEP 6] Running inference for {len(questions)} questions...")
            print(f"[EXECUTE STEP 6] Running inference for {len(questions)} questions...")
            
            for i, item in enumerate(questions):
                logger.info(f"[EXECUTE STEP 6] Processing question {i+1}/{len(questions)}: {item['id']}")
                print(f"[EXECUTE STEP 6] Processing question {i+1}/{len(questions)}: {item['id']}")
                
                # Format prompt using naive template
                prompt = NaivePromptTemplate.format(item["question"])
                
                # Generate response
                result = engine.generate(prompt, gen_config)
                
                # Parse response
                parsed_answer = NaivePromptTemplate.parse_response(result.text)
                
                # Check correctness (simple exact match for Phase 2)
                is_correct = parsed_answer.lower().strip() == item["answer"].lower().strip()
                
                # Log run to database
                await run_service.create_run(
                    experiment_id=experiment_id,
                    example_id=item["id"],
                    input_text=prompt,
                    output_text=parsed_answer,
                    expected_output=item["answer"],
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    tokens_input=result.tokens_input,
                    tokens_output=result.tokens_output,
                    latency_ms=result.latency_ms,
                    gpu_memory_mb=result.gpu_memory_mb,
                )
                logger.info(f"[EXECUTE STEP 6] ✓ Run logged for question {item['id']}")
                print(f"[EXECUTE STEP 6] ✓ Run logged for question {item['id']}")
            
            # Step 7: Commit all runs
            logger.info(f"[EXECUTE STEP 7] Committing all runs to database...")
            print(f"[EXECUTE STEP 7] Committing all runs to database...")
            await self.db.commit()
            logger.info(f"[EXECUTE STEP 7] ✓ Runs committed")
            print(f"[EXECUTE STEP 7] ✓ Runs committed")
            
            # Step 8: Cleanup
            logger.info(f"[EXECUTE STEP 8] Cleanup...")
            print(f"[EXECUTE STEP 8] Cleanup...")
            engine.unload_model()
            logger.info(f"[EXECUTE STEP 8] ✓ Engine unloaded")
            print(f"[EXECUTE STEP 8] ✓ Engine unloaded")
            
            # Step 9: Update status to COMPLETED
            logger.info(f"[EXECUTE STEP 9] Updating status to COMPLETED...")
            print(f"[EXECUTE STEP 9] Updating status to COMPLETED...")
            await self.update_status(experiment_id, ExperimentStatus.COMPLETED)
            await self.db.commit()  # IMPORTANT: Commit the status change!
            logger.info(f"[EXECUTE STEP 9] ✓ Status updated and committed")
            print(f"[EXECUTE STEP 9] ✓ Status updated and committed")
            
            logger.info(f"[EXECUTE] ✅ EXECUTION COMPLETED SUCCESSFULLY")
            print(f"[EXECUTE] ✅ EXECUTION COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            # Log error and update status to FAILED
            logger.error(f"[EXECUTE] ❌ EXECUTION FAILED: {type(e).__name__}: {str(e)}")
            print(f"[EXECUTE] ❌ EXECUTION FAILED: {type(e).__name__}: {str(e)}")
            
            import traceback
            traceback.print_exc()
            
            error_message = f"Execution failed: {type(e).__name__}: {str(e)}"
            await self.update_status(
                experiment_id,
                ExperimentStatus.FAILED,
                error_message=error_message
            )
            await self.db.commit()  # IMPORTANT: Commit the failed status!
            logger.info(f"[EXECUTE] Status updated to FAILED and committed")
            print(f"[EXECUTE] Status updated to FAILED and committed")
            
            raise
