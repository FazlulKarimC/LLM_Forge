"""
Experiment Service

Business logic for experiment management.
Handles CRUD operations and experiment execution orchestration.
"""

import json
import logging
import time as _time

from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentListItem,
    ExperimentSlimListResponse,
    ExperimentStatus,
    ExperimentConfig,
    RegressionStatus,
)
from app.services.experiment_runtime import sanitize_error_message as _sanitize_error_message

logger = logging.getLogger(__name__)


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
            tags=experiment.tags,
            run_manifest=experiment.run_manifest,
            is_baseline=getattr(experiment, 'is_baseline', False),
            regression_status=RegressionStatus(
                getattr(experiment, 'regression_status', None) or RegressionStatus.NOT_CHECKED.value
            ),
            regression_passed=getattr(experiment, 'regression_passed', None),
        )

    def _to_list_item(self, experiment: Experiment) -> ExperimentListItem:
        """Convert database model to slim list item schema."""
        config = experiment.config or {}
        return ExperimentListItem(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            status=experiment.status,
            created_at=experiment.created_at,
            completed_at=experiment.completed_at,
            is_baseline=getattr(experiment, 'is_baseline', False),
            regression_status=RegressionStatus(
                getattr(experiment, 'regression_status', None) or RegressionStatus.NOT_CHECKED.value
            ),
            provider=config.get('provider', 'auto'),
            reasoning_method=config.get('reasoning_method', 'naive'),
            model_name=config.get('model_name', ''),
            dataset_name=config.get('dataset_name', ''),
            num_samples=config.get('num_samples'),
        )
    
    async def create(self, data: ExperimentCreate) -> ExperimentResponse:
        """
        Create a new experiment.
        
        Args:
            data: Experiment creation request
        
        Returns:
            Created experiment with generated ID
        """
        import hashlib
        
        config_dict = data.config.model_dump()
        prompt_version_manifest = None
        if data.config.prompt_version_id:
            from app.models.prompt_version import PromptVersion

            prompt_result = await self.db.execute(
                select(PromptVersion).where(PromptVersion.id == data.config.prompt_version_id)
            )
            prompt_version = prompt_result.scalar_one_or_none()
            if not prompt_version:
                from app.core.custom_exceptions import ValidationException

                raise ValidationException(f"PromptVersion {data.config.prompt_version_id} not found")
            prompt_version_manifest = {
                "id": str(prompt_version.id),
                "name": prompt_version.name,
                "version": prompt_version.version,
                "sha256_hash": prompt_version.sha256_hash,
            }
        
        # Build immutable run manifest for reproducibility
        manifest_data = {
            "dataset_name": data.config.dataset_name,
            "model_name": data.config.model_name,
            "provider": data.config.provider.value if data.config.provider else "auto",
            "reasoning_method": data.config.reasoning_method.value,
            "hyperparameters": config_dict.get("hyperparameters", {}),
            "num_samples": data.config.num_samples,
            "rag": config_dict.get("rag"),
            "agent": config_dict.get("agent"),
            "optimization": config_dict.get("optimization"),
            "graders": config_dict.get("graders"),
            "regression": config_dict.get("regression"),
            "routing": config_dict.get("routing"),
            "prompt_version_id": str(data.config.prompt_version_id) if hasattr(data.config, 'prompt_version_id') and data.config.prompt_version_id else None,
            "prompt_version": prompt_version_manifest,
        }
        manifest_json = json.dumps(manifest_data, sort_keys=True, default=str)
        manifest_data["manifest_hash"] = hashlib.sha256(manifest_json.encode()).hexdigest()
        
        experiment = Experiment(
            name=data.name,
            description=data.description,
            config=config_dict,
            method=data.config.reasoning_method.value,
            model_name=data.config.model_name,
            dataset_name=data.config.dataset_name,
            status=ExperimentStatus.PENDING,
            tags=data.tags or [],
            run_manifest=manifest_data,
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
    
    def _build_list_conditions(
        self,
        status: Optional[ExperimentStatus],
        method: Optional[str],
        model: Optional[str],
        tag: Optional[str],
    ) -> List[Any]:
        """Build shared WHERE conditions for list and list_slim queries."""
        conditions: List[Any] = [Experiment.deleted_at.is_(None)]
        if status:
            conditions.append(Experiment.status == status)
        if method:
            conditions.append(Experiment.method == method)
        if model:
            conditions.append(Experiment.model_name.ilike(f"%{model}%"))
        if tag:
            from sqlalchemy.dialects.postgresql import JSONB
            conditions.append(Experiment.tags.cast(JSONB).contains([tag]))
        return conditions

    async def _paginated_experiments(
        self,
        conditions: list,
        skip: int,
        limit: int,
    ):
        """Execute count + paginated fetch for a set of WHERE conditions."""
        count_query = select(func.count(Experiment.id)).where(and_(*conditions))
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        query = (
            select(Experiment)
            .where(and_(*conditions))
            .order_by(Experiment.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        experiments = result.scalars().all()
        return total, experiments

    async def list(
        self,
        status: Optional[ExperimentStatus] = None,
        method: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> ExperimentListResponse:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status
            method: Filter by reasoning method
            model: Filter by model name
            tag: Filter by tag (experiments containing this tag)
            skip: Pagination offset
            limit: Max results
        
        Returns:
            Paginated list of experiments
        """
        conditions = self._build_list_conditions(status, method, model, tag)
        total, experiments = await self._paginated_experiments(conditions, skip, limit)
        
        return ExperimentListResponse(
            total=total,
            experiments=[self._to_response(exp) for exp in experiments],
            skip=skip,
            limit=limit,
        )
    
    async def list_slim(
        self,
        status: Optional[ExperimentStatus] = None,
        method: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> ExperimentSlimListResponse:
        """List experiments as slim list items (no full config, run_manifest, etc.)."""
        conditions = self._build_list_conditions(status, method, model, tag)
        total, experiments = await self._paginated_experiments(conditions, skip, limit)

        return ExperimentSlimListResponse(
            total=total,
            experiments=[self._to_list_item(exp) for exp in experiments],
            skip=skip,
            limit=limit,
        )
    
    async def get_stats(self) -> dict:
        """
        Get aggregated experiment counts by status.
        
        Uses a single GROUP BY query instead of fetching all rows.
        Returns dict with total, completed, running, pending, queued, failed counts.
        """
        query = (
            select(Experiment.status, func.count(Experiment.id))
            .where(Experiment.deleted_at.is_(None))
            .group_by(Experiment.status)
        )
        result = await self.db.execute(query)
        rows = result.all()
        
        # Build counts dict
        counts = {status.value: count for status, count in rows}
        total = sum(counts.values())
        
        return {
            "total": total,
            "completed": counts.get("completed", 0),
            "running": counts.get("running", 0),
            "pending": counts.get("pending", 0),
            "queued": counts.get("queued", 0),
            "failed": counts.get("failed", 0),
        }
    
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
        if status == ExperimentStatus.QUEUED:
            experiment.started_at = None
            experiment.completed_at = None
        elif status == ExperimentStatus.RUNNING:
            experiment.started_at = now
            experiment.completed_at = None
        elif status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            experiment.completed_at = now
        
        # Set error message if provided (or clear it if rerun passing None/"")
        experiment.error_message = error_message if error_message is not None else ""
        
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

    async def _prepare_execution_attempt(self, experiment_id: UUID):
        """Create a new non-destructive attempt and mark the experiment as running."""
        from sqlalchemy import select as _sel, func as _fn
        from app.models.run import Run as _Run
        from app.models.experiment import Experiment as _Exp

        max_attempt_q = await self.db.execute(
            _sel(_fn.coalesce(_fn.max(_Run.attempt), 0)).where(_Run.experiment_id == experiment_id)
        )
        current_attempt = (max_attempt_q.scalar() or 0) + 1

        exp_row = await self.db.execute(_sel(_Exp).where(_Exp.id == experiment_id))
        exp_obj = exp_row.scalar_one_or_none()
        if exp_obj:
            exp_obj.current_attempt = current_attempt
            exp_obj.regression_status = RegressionStatus.NOT_CHECKED.value
            exp_obj.regression_passed = None

        await self.update_status(experiment_id, ExperimentStatus.RUNNING, error_message="")
        await self.db.commit()
        logger.info("[EXECUTE] Status: RUNNING (attempt %s)", current_attempt)
        return current_attempt, exp_obj

    def _runtime_helper(self):
        helper = getattr(self, "_runtime", None)
        if helper is None:
            from app.services.experiment_runtime import ExperimentRuntimeBuilder

            helper = ExperimentRuntimeBuilder(getattr(self, "db", None))
            self._runtime = helper
        return helper

    def _runner_helper(self):
        helper = getattr(self, "_runner", None)
        if helper is None:
            from app.services.experiment_runner import ExperimentRunExecutor

            helper = ExperimentRunExecutor()
            self._runner = helper
        return helper

    def _postprocessor_helper(self):
        helper = getattr(self, "_postprocessor", None)
        if helper is None:
            from app.services.experiment_postprocessing import ExperimentPostProcessor

            helper = ExperimentPostProcessor(getattr(self, "db", None))
            self._postprocessor = helper
        return helper

    def _create_optimization_runtime(self, experiment_response: ExperimentResponse):
        return self._runtime_helper().create_optimization_runtime(experiment_response)

    def _initialize_engine(
        self,
        experiment_response: ExperimentResponse,
        custom_base_url: Optional[str],
        custom_api_key: Optional[str],
    ):
        return self._runtime_helper().initialize_engine(
            experiment_response,
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
        )

    def _initialize_rag_runtime(self, experiment_response: ExperimentResponse):
        return self._runtime_helper().initialize_rag_runtime(experiment_response)

    async def _load_examples(self, experiment_response: ExperimentResponse, exp_obj: Optional[Experiment]):
        return await self._runtime_helper().load_examples(experiment_response, exp_obj)

    async def _load_cot_examples(self, reasoning_method: str):
        return await self._runtime_helper().load_cot_examples(reasoning_method)

    async def _resolve_prompt_templates(self, **kwargs):
        return await self._runtime_helper().resolve_prompt_templates(**kwargs)

    def _build_generation_config(self, experiment_response: ExperimentResponse, reasoning_method: str):
        return self._runtime_helper().build_generation_config(experiment_response, reasoning_method)

    async def _persist_effective_execution_manifest(self, **kwargs) -> None:
        await self._runtime_helper().persist_effective_execution_manifest(**kwargs)

    def _create_react_agent(self, experiment_response: ExperimentResponse, rag_pipeline, engine, gen_config):
        return self._runtime_helper().create_react_agent(experiment_response, rag_pipeline, engine, gen_config)

    def _generation_seed(self, gen_config) -> Any:
        return self._runner_helper().generation_seed(gen_config)

    def _uses_robustness_scoring(self, dataset_name: str) -> bool:
        return self._runner_helper().uses_robustness_scoring(dataset_name)

    def _score_response(self, **kwargs) -> dict[str, Any]:
        return self._runner_helper().score_response(**kwargs)

    async def _execute_batched_runs(self, **kwargs) -> dict[str, int]:
        return await self._runner_helper().execute_batched_runs(**kwargs)

    async def _execute_sequential_runs(self, **kwargs) -> dict[str, int]:
        return await self._runner_helper().execute_sequential_runs(**kwargs)

    async def _apply_graders(self, **kwargs) -> None:
        await self._postprocessor_helper().apply_graders(**kwargs)

    async def _save_optimization_report(self, **kwargs) -> None:
        await self._postprocessor_helper().save_optimization_report(**kwargs)

    async def _run_auto_regression_check(self, experiment_id: UUID) -> None:
        await self._postprocessor_helper().run_auto_regression_check(experiment_id)

    async def execute(
        self, 
        experiment_id: UUID, 
        custom_base_url: Optional[str] = None, 
        custom_api_key: Optional[str] = None
    ) -> None:
        """
        Execute an experiment.
        
        This is the main orchestration method:
        1. Load experiment config
        2. Initialize inference engine
        3. Load dataset (TriviaQA or sample)
        4. Run inference for each sample (with optional batching/caching)
        5. Compute per-run metrics (F1, exact match)
        6. Log runs to database
        7. Compute aggregate metrics and save Result
        8. Update status
        9. Store optimization report (profiling, cache stats, batch stats)
        
        Should be run as a background task.
        """
        logger.info("[EXECUTE] Starting execution for experiment: %s", experiment_id)
        
        # Step 1: Get experiment
        experiment_response = await self.get(experiment_id)
        if not experiment_response:
            error = f"Experiment {experiment_id} not found"
            logger.error("[EXECUTE] %s", error)
            raise ValueError(error)
        logger.info("[EXECUTE] Found experiment: %s", experiment_response.name)
        
        try:
            from app.services.inference.prompting import (
                NaivePromptTemplate, CoTPromptTemplate, RAGPromptTemplate, ReActPromptTemplate
            )
            from app.services.run_service import RunService
            from app.services.metrics_service import MetricsService

            # Step 2: Initialize services for a new execution attempt.
            run_service = RunService(self.db)
            metrics_svc = MetricsService(self.db)
            current_attempt, exp_obj = await self._prepare_execution_attempt(experiment_id)
            reasoning_method = experiment_response.config.reasoning_method.value
            logger.info("[EXECUTE] Reasoning method: %s", reasoning_method)
            wall_start, opt_config, cache, profiler, opt_report = self._create_optimization_runtime(experiment_response)
            engine, engine_type = self._initialize_engine(
                experiment_response,
                custom_base_url=custom_base_url,
                custom_api_key=custom_api_key,
            )
            rag_config, use_rag, rag_pipeline, faithfulness_scorer = self._initialize_rag_runtime(experiment_response)

            # Step 4: Load dataset.
            examples = await self._load_examples(experiment_response, exp_obj)
            cot_examples = await self._load_cot_examples(reasoning_method)
            use_robustness_scoring = self._uses_robustness_scoring(
                experiment_response.config.dataset_name
            )
            if use_robustness_scoring:
                logger.info("[EXECUTE] Robustness scoring enabled for adversarial dataset")
            naive_prompt_template, cot_prompt_template, rag_prompt_template = await self._resolve_prompt_templates(
                experiment_response=experiment_response,
                reasoning_method=reasoning_method,
                use_rag=use_rag,
                naive_prompt_template=NaivePromptTemplate,
                cot_prompt_template=CoTPromptTemplate,
                rag_prompt_template=RAGPromptTemplate,
            )

            # Step 5: Prepare generation runtime.
            gen_config, max_tokens = self._build_generation_config(experiment_response, reasoning_method)
            react_agent = self._create_react_agent(
                experiment_response,
                rag_pipeline=rag_pipeline,
                engine=engine,
                gen_config=gen_config,
            )

            # Step 6: Run inference.
            logger.info("[EXECUTE] Running inference for %s examples...", len(examples))
            use_batching = (
                opt_config.enable_batching
                and reasoning_method != "react"
            )
            await self._persist_effective_execution_manifest(
                experiment_response=experiment_response,
                exp_obj=exp_obj,
                current_attempt=current_attempt,
                engine_type=engine_type,
                gen_config=gen_config,
                max_tokens=max_tokens,
                examples=examples,
                use_rag=use_rag,
                use_batching=use_batching,
                opt_config=opt_config,
            )

            if use_batching and not use_rag:
                batch_stats = await self._execute_batched_runs(
                    experiment_id=experiment_id,
                    experiment_response=experiment_response,
                    examples=examples,
                    reasoning_method=reasoning_method,
                    cot_examples=cot_examples,
                    gen_config=gen_config,
                    max_tokens=max_tokens,
                    engine=engine,
                    cache=cache,
                    profiler=profiler,
                    metrics_svc=metrics_svc,
                    run_service=run_service,
                    current_attempt=current_attempt,
                    batch_size=opt_config.batch_size,
                    naive_prompt_template=naive_prompt_template,
                    cot_prompt_template=cot_prompt_template,
                    use_robustness_scoring=use_robustness_scoring,
                )
            else:
                batch_stats = await self._execute_sequential_runs(
                    experiment_id=experiment_id,
                    experiment_response=experiment_response,
                    examples=examples,
                    reasoning_method=reasoning_method,
                    cot_examples=cot_examples,
                    gen_config=gen_config,
                    max_tokens=max_tokens,
                    engine=engine,
                    cache=cache,
                    profiler=profiler,
                    metrics_svc=metrics_svc,
                    run_service=run_service,
                    current_attempt=current_attempt,
                    use_batching=use_batching,
                    use_rag=use_rag,
                    use_robustness_scoring=use_robustness_scoring,
                    rag_config=rag_config,
                    rag_pipeline=rag_pipeline,
                    faithfulness_scorer=faithfulness_scorer,
                    react_agent=react_agent,
                    naive_prompt_template=naive_prompt_template,
                    cot_prompt_template=cot_prompt_template,
                    rag_prompt_template=rag_prompt_template,
                    react_prompt_template=ReActPromptTemplate,
                )

            logger.info("[EXECUTE] Committing %s runs to database...", len(examples))
            await self.db.commit()
            await self._apply_graders(
                experiment_id=experiment_id,
                experiment_response=experiment_response,
                current_attempt=current_attempt,
            )

            logger.info("[EXECUTE] Computing aggregate metrics...")
            wall_ms = (_time.perf_counter() - wall_start) * 1000
            await metrics_svc.compute_and_save(experiment_id, wall_clock_ms=wall_ms)
            await self._save_optimization_report(
                experiment_id=experiment_id,
                engine=engine,
                opt_report=opt_report,
                batch_stats=batch_stats,
                cache=cache,
                profiler=profiler,
                wall_start=wall_start,
            )

            engine.unload_model()
            await self.update_status(experiment_id, ExperimentStatus.COMPLETED)
            await self.db.commit()
            await self._run_auto_regression_check(experiment_id)
            logger.info(
                "[EXECUTE] Execution completed (wall time: %.0fms)",
                opt_report.total_wall_time_ms,
            )

        except Exception as e:
            logger.exception("[EXECUTE] Execution failed: %s: %s", type(e).__name__, e)
            error_message = _sanitize_error_message(e)
            await self.update_status(
                experiment_id,
                ExperimentStatus.FAILED,
                error_message=error_message
            )
            await self.db.commit()
            
            raise


