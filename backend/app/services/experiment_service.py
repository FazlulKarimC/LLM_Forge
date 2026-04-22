"""
Experiment Service

Business logic for experiment management.
Handles CRUD operations and experiment execution orchestration.
"""

import asyncio
import json
import logging
import re
import time as _time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.config import settings
from app.models.experiment import Experiment
from app.models.result import Result
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentListItem,
    ExperimentSlimListResponse,
    ExperimentStatus,
    ExperimentConfig,
    OptimizationConfig,
    RegressionStatus,
    regression_status_from_verdict,
)

logger = logging.getLogger(__name__)


class VersionedPromptTemplate:
    """Adapter that executes an immutable PromptVersion template."""

    def __init__(self, template_text: str, parser):
        self.template_text = template_text
        self._parser = parser

    @staticmethod
    def _format_cot_examples(examples) -> str:
        if not examples:
            return ""
        return "\n\n".join(
            "Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}".format(
                question=example.get("question", ""),
                reasoning=example.get("reasoning", ""),
                answer=example.get("answer", ""),
            )
            for example in examples
        )

    def format(self, question: str, extra=None) -> str:
        context = "\n\n".join(extra) if isinstance(extra, list) and all(isinstance(item, str) for item in extra) else ""
        cot_examples = self._format_cot_examples(extra) if isinstance(extra, list) and not context else ""
        values = {
            "question": question,
            "input": question,
            "context": context,
            "context_chunks": context,
            "cot_examples": cot_examples,
            "examples": cot_examples,
        }
        try:
            return self.template_text.format(**values)
        except KeyError as exc:
            placeholder = exc.args[0]
            raise ValueError(
                f"PromptVersion template references unsupported placeholder {{{placeholder}}}"
            ) from exc

    def parse_response(self, response: str) -> str:
        return self._parser(response)


def _cot_examples_path() -> Path:
    """Resolve CoT examples path via settings (lazy, avoids circular import at module level)."""
    return settings.configs_dir / "cot_examples.json"


# Error sanitization ---------------------------------------------------------
_PATH_PATTERN = re.compile(r"(?:/[\w.\-]+){2,}")  # Unix absolute paths
_WIN_PATH_PATTERN = re.compile(r"[A-Za-z]:\\(?:[\w.\- ]+\\)*[\w.\- ]+")  # Windows paths
_TOKEN_PATTERN = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}"      # Hugging Face tokens
    r"|sk-[A-Za-z0-9]{20,}"         # OpenAI-style keys
    r"|[A-Fa-f0-9]{32,})",          # Long hex strings (generic secrets)
)
_MAX_ERROR_LENGTH = 500


def _sanitize_error_message(exc: Exception) -> str:
    """
    Build a safe error string from an exception.

    Strips:
    - Unix / Windows absolute file paths
    - Anything that looks like an API key or token
    Truncates to _MAX_ERROR_LENGTH characters.
    """
    raw = f"{type(exc).__name__}: {exc}"
    sanitized = _PATH_PATTERN.sub("<path>", raw)
    sanitized = _WIN_PATH_PATTERN.sub("<path>", sanitized)
    sanitized = _TOKEN_PATTERN.sub("<redacted>", sanitized)
    if len(sanitized) > _MAX_ERROR_LENGTH:
        sanitized = sanitized[:_MAX_ERROR_LENGTH] + "..."
    return f"Execution failed: {sanitized}"


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
        # Build base query (exclude soft-deleted)
        conditions = [Experiment.deleted_at.is_(None)]
        
        # Apply filters
        if status:
            conditions.append(Experiment.status == status)
        if method:
            conditions.append(Experiment.method == method)
        if model:
            conditions.append(Experiment.model_name.ilike(f"%{model}%"))
        if tag:
            # JSONB array containment - works on PostgreSQL
            from sqlalchemy import cast, type_coerce
            from sqlalchemy.dialects.postgresql import JSONB
            conditions.append(Experiment.tags.cast(JSONB).contains([tag]))
        
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
        conditions = [Experiment.deleted_at.is_(None)]

        if status:
            conditions.append(Experiment.status == status)
        if method:
            conditions.append(Experiment.method == method)
        if model:
            conditions.append(Experiment.model_name.ilike(f"%{model}%"))
        if tag:
            from sqlalchemy.dialects.postgresql import JSONB
            conditions.append(Experiment.tags.cast(JSONB).contains([tag]))

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

    def _create_optimization_runtime(self, experiment_response: ExperimentResponse):
        """Initialize optimization helpers used throughout execution."""
        from app.services.optimization import PromptCache, ProfilerContext, OptimizationReport

        wall_start = _time.perf_counter()
        opt_config = experiment_response.config.optimization or OptimizationConfig()
        cache = PromptCache(max_size=opt_config.cache_max_size) if opt_config.enable_caching else None
        profiler = ProfilerContext(enabled=opt_config.enable_profiling)
        opt_report = OptimizationReport()

        logger.info(
            "[EXECUTE] Optimization: batching=%s (size=%s), caching=%s, profiling=%s",
            opt_config.enable_batching,
            opt_config.batch_size,
            opt_config.enable_caching,
            opt_config.enable_profiling,
        )
        return wall_start, opt_config, cache, profiler, opt_report

    def _initialize_engine(
        self,
        experiment_response: ExperimentResponse,
        custom_base_url: Optional[str],
        custom_api_key: Optional[str],
    ):
        """Build and load the inference engine for the experiment."""
        from app.services.inference.engine_factory import create_inference_engine

        model_name = experiment_response.config.model_name
        provider = getattr(experiment_response.config, "provider", None)
        provider_value = provider.value if provider else "auto"
        engine, engine_type = create_inference_engine(
            model_name=model_name,
            provider_value=provider_value,
            routing_config=experiment_response.config.routing,
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            default_engine=settings.INFERENCE_ENGINE,
        )

        if engine_type == "mock":
            logger.info("[EXECUTE] Auto-detected mock model '%s', using MockEngine", model_name)
        elif engine_type == "custom":
            logger.info("[EXECUTE] Custom base URL detected, using OpenAIEngine")
        elif engine_type == "auto (router)" and hasattr(engine, "_engines"):
            logger.info("[EXECUTE] Router initialized with %d providers", len(engine._engines))

        logger.info("[EXECUTE] Engine type: %s", engine_type)
        engine.load_model(model_name)
        logger.info("[EXECUTE] Engine loaded: %s", model_name)
        return engine

    def _initialize_rag_runtime(self, experiment_response: ExperimentResponse):
        """Initialize RAG services only when retrieval is enabled."""
        rag_pipeline = None
        faithfulness_scorer = None
        rag_config = experiment_response.config.rag
        use_rag = bool(rag_config and rag_config.retrieval_method.value != "none")

        if use_rag:
            from app.services.rag_service import RAGPipeline, FaithfulnessScorer

            logger.info("[EXECUTE] Initializing RAG pipeline (method=%s)", rag_config.retrieval_method.value)
            rag_pipeline = RAGPipeline()
            rag_pipeline.load_knowledge_base(chunk_size=rag_config.chunk_size)
            faithfulness_scorer = FaithfulnessScorer()
            logger.info("[EXECUTE] RAG pipeline initialized (top_k=%s)", rag_config.top_k)

        return rag_config, use_rag, rag_pipeline, faithfulness_scorer

    async def _load_examples(self, experiment_response: ExperimentResponse, exp_obj: Optional[Experiment]):
        """Load the dataset slice for this run and persist a reproducibility fingerprint."""
        import hashlib
        from app.services.dataset_service import DatasetService

        dataset_name = experiment_response.config.dataset_name
        num_samples = experiment_response.config.num_samples
        seed = experiment_response.config.hyperparameters.seed

        logger.info("[EXECUTE] Loading dataset: %s (n=%s, seed=%s)", dataset_name, num_samples, seed)
        examples = DatasetService.load(
            dataset_name=dataset_name,
            num_samples=num_samples,
            seed=seed,
        )
        logger.info("[EXECUTE] Loaded %s examples", len(examples))

        dataset_content = json.dumps(examples, sort_keys=True)
        dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()
        sample_ids_list = [example.get("id", str(index)) for index, example in enumerate(examples)]

        if exp_obj:
            exp_obj.dataset_hash = dataset_hash
            exp_obj.sample_ids = sample_ids_list
            await self.db.flush()

        return examples

    def _load_cot_examples(self, reasoning_method: str):
        """Load few-shot CoT examples when chain-of-thought prompting is enabled."""
        if reasoning_method != "cot":
            return None

        cot_path = _cot_examples_path()
        if not cot_path.exists():
            logger.warning("[EXECUTE] CoT examples file not found (%s), using zero-shot CoT", cot_path)
            return None

        with cot_path.open("r", encoding="utf-8") as handle:
            cot_examples = json.load(handle)
        logger.info("[EXECUTE] Loaded %s CoT few-shot examples", len(cot_examples))
        return cot_examples

    async def _resolve_prompt_templates(
        self,
        *,
        experiment_response: ExperimentResponse,
        reasoning_method: str,
        use_rag: bool,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ):
        """Apply a configured PromptVersion to the active prompt path."""
        prompt_version_id = experiment_response.config.prompt_version_id
        if not prompt_version_id:
            return naive_prompt_template, cot_prompt_template, rag_prompt_template

        if reasoning_method == "react":
            raise ValueError(
                "prompt_version_id is not supported for ReAct runs because the agent builds its own system prompt"
            )

        from app.models.prompt_version import PromptVersion

        prompt_result = await self.db.execute(
            select(PromptVersion).where(PromptVersion.id == prompt_version_id)
        )
        prompt_version = prompt_result.scalar_one_or_none()
        if not prompt_version:
            raise ValueError(f"PromptVersion {prompt_version_id} not found")

        logger.info(
            "[EXECUTE] Applying PromptVersion %s v%s (%s)",
            prompt_version.name,
            prompt_version.version,
            prompt_version.sha256_hash[:12],
        )

        if use_rag:
            rag_prompt_template = VersionedPromptTemplate(
                prompt_version.template_text,
                rag_prompt_template.parse_response,
            )
        elif reasoning_method == "cot":
            cot_prompt_template = VersionedPromptTemplate(
                prompt_version.template_text,
                cot_prompt_template.parse_response,
            )
        else:
            naive_prompt_template = VersionedPromptTemplate(
                prompt_version.template_text,
                naive_prompt_template.parse_response,
            )

        return naive_prompt_template, cot_prompt_template, rag_prompt_template

    def _build_generation_config(self, experiment_response: ExperimentResponse, reasoning_method: str):
        """Create generation settings, applying reasoning-specific token floors."""
        from app.services.inference.base import GenerationConfig

        max_tokens = experiment_response.config.hyperparameters.max_tokens
        if reasoning_method == "cot" and max_tokens <= 256:
            max_tokens = 512
            logger.info("[EXECUTE] Increased max_tokens to %s for CoT", max_tokens)
        elif reasoning_method == "react" and max_tokens <= 512:
            max_tokens = 1024
            logger.info("[EXECUTE] Increased max_tokens to %s for ReAct", max_tokens)

        gen_config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=experiment_response.config.hyperparameters.temperature,
            top_p=experiment_response.config.hyperparameters.top_p,
            top_k=experiment_response.config.hyperparameters.top_k,
            seed=experiment_response.config.hyperparameters.seed,
        )
        return gen_config, max_tokens

    def _create_react_agent(self, experiment_response: ExperimentResponse, rag_pipeline, engine, gen_config):
        """Build a ReAct agent only when the experiment is configured for agent mode."""
        reasoning_method = experiment_response.config.reasoning_method.value
        if reasoning_method != "react":
            return None

        from app.services.agent_service import (
            ReActAgent,
            WikipediaSearchTool,
            CalculatorTool,
            RetrievalTool,
        )

        agent_config = experiment_response.config.agent
        enabled_tools_names = agent_config.tools if agent_config else ["wikipedia_search", "calculator"]
        agent_max_iter = agent_config.max_iterations if agent_config else 5
        agent_tools = []

        for tool_name in enabled_tools_names:
            if tool_name == "wikipedia_search":
                agent_tools.append(WikipediaSearchTool())
            elif tool_name == "calculator":
                agent_tools.append(CalculatorTool())
            elif tool_name == "retrieval" and rag_pipeline:
                agent_tools.append(RetrievalTool(rag_pipeline=rag_pipeline))
            elif tool_name == "retrieval":
                try:
                    from app.services.rag_service import RAGPipeline

                    rag_for_tool = RAGPipeline()
                    rag_for_tool.load_knowledge_base()
                    agent_tools.append(RetrievalTool(rag_pipeline=rag_for_tool))
                except Exception as exc:
                    logger.warning("[EXECUTE] Could not init retrieval tool: %s", exc)

        logger.info(
            "[EXECUTE] Initializing ReAct agent (max_iter=%s, tools=%s)",
            agent_max_iter,
            [tool.name for tool in agent_tools],
        )

        react_agent = ReActAgent(
            engine=engine,
            tools=agent_tools,
            max_iterations=agent_max_iter,
            gen_config=gen_config,
        )
        logger.info("[EXECUTE] ReAct agent created")
        return react_agent

    def _generation_seed(self, gen_config) -> Any:
        """Return the optional generation seed in a single place for cache calls."""
        return getattr(gen_config, "seed", None)

    def _uses_robustness_scoring(self, dataset_name: str) -> bool:
        """Return whether the dataset should use deterministic safety scoring."""
        from app.services.dataset_service import DATASET_REGISTRY

        return DATASET_REGISTRY.get(dataset_name, {}).get("category") == "adversarial"

    def _score_response(
        self,
        *,
        item,
        raw_output: str,
        parsed_answer: Optional[str],
        metrics_svc,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Score a response with QA metrics and optional adversarial robustness metadata."""
        aliases = item.get("aliases", [item["answer"]])
        is_exact, is_substring, f1_score, matched_alias = metrics_svc.check_any_alias_match(
            parsed_answer,
            aliases,
        )
        is_correct = is_exact or is_substring
        grader_results = None

        if use_robustness_scoring:
            from app.services.robustness_scorer import classify_response

            robustness = classify_response(
                raw_output or parsed_answer or "",
                expected_behavior=item.get("answer", "REFUSE"),
                category=item.get("category", ""),
            )
            if item.get("expected_behavior"):
                robustness["expected_behavior"] = item["expected_behavior"]
            grader_results = {"robustness": robustness}

            # Sentinel answers are evaluation directives, not strings the model should emit.
            if str(item.get("answer", "")).upper() in {"REFUSE", "HANDLE_GRACEFULLY"}:
                is_correct = bool(robustness.get("is_safe"))
                f1_score = 1.0 if is_correct else 0.0

        return {
            "is_correct": is_correct,
            "score": f1_score,
            "is_exact_match": is_exact,
            "is_substring_match": is_substring,
            "match_alias": matched_alias,
            "grader_results": grader_results,
        }

    def _build_prompt(
        self,
        question: str,
        reasoning_method: str,
        cot_examples,
        context_chunks,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ) -> str:
        """Build the final prompt for a single example."""
        if context_chunks:
            return rag_prompt_template.format(question, context_chunks)
        if reasoning_method == "cot":
            return cot_prompt_template.format(question, cot_examples)
        return naive_prompt_template.format(question)

    def _parse_generation_output(
        self,
        raw_text: str,
        reasoning_method: str,
        *,
        used_rag: bool,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ) -> str:
        """Parse the model response using the matching prompt template."""
        if used_rag:
            return rag_prompt_template.parse_response(raw_text)
        if reasoning_method == "cot":
            return cot_prompt_template.parse_response(raw_text)
        return naive_prompt_template.parse_response(raw_text)

    def _retrieve_rag_context(self, item, rag_config, rag_pipeline, profiler):
        """Retrieve RAG context and normalize it into run payload fields."""
        context_chunks = []
        retrieval_context = ""
        retrieved_chunk_payload = None

        if not rag_pipeline or not rag_config:
            return context_chunks, retrieval_context, retrieved_chunk_payload

        with profiler.section("rag_retrieval"):
            retrieval_result = rag_pipeline.retrieve(
                question=item["question"],
                method=rag_config.retrieval_method.value,
                top_k=rag_config.top_k,
            )

        context_chunks = [chunk.text for chunk in retrieval_result.chunks]
        retrieved_chunk_payload = {
            "chunks": [
                {
                    "text": chunk.text,
                    "score": getattr(chunk, "score", None),
                }
                for chunk in retrieval_result.chunks
            ]
        }
        retrieval_context = " ".join(context_chunks)
        logger.info(
            "[EXECUTE]   Retrieved %s chunks (%.0fms)",
            len(context_chunks),
            retrieval_result.latency_ms,
        )
        return context_chunks, retrieval_context, retrieved_chunk_payload

    def _score_faithfulness(self, parsed_answer: str, retrieval_context: str, faithfulness_scorer, profiler):
        """Compute faithfulness only when the run used retrieved context."""
        if not retrieval_context or faithfulness_scorer is None:
            return None

        try:
            with profiler.section("faithfulness"):
                faithfulness = faithfulness_scorer.score(parsed_answer, retrieval_context)
            logger.info("[EXECUTE]   Faithfulness: %.3f", faithfulness)
            return faithfulness
        except Exception as exc:
            logger.warning("[EXECUTE]   Faithfulness scoring failed: %s", exc)
            return None

    def _score_context_relevance(self, question: str, context_chunks, profiler):
        """Estimate context relevance from retrieved chunks for RAG runs."""
        if not context_chunks:
            return None

        try:
            with profiler.section("context_relevance"):
                from app.services.rag_service import CrossEncoderReranker as _CER

                reranker = _CER()
                scored = reranker.rerank(
                    question,
                    [
                        type("Chunk", (), {"id": f"c{index}", "text": text, "title": "", "index": index})()
                        for index, text in enumerate(context_chunks)
                    ],
                    top_k=len(context_chunks),
                )
                if scored:
                    return float(np.mean([score for _, score in scored]))
        except Exception as exc:
            logger.warning("[EXECUTE]   Context relevance scoring failed: %s", exc)

        return None

    def _score_semantic_similarity(self, parsed_answer: Optional[str], expected_answer: Optional[str], profiler):
        """Compute cosine similarity between predicted and expected answers."""
        if not parsed_answer or not expected_answer:
            return None

        try:
            with profiler.section("semantic_similarity"):
                from app.services.rag_service import EmbeddingService as _ES

                emb_svc = _ES()
                embeddings = emb_svc.embed([parsed_answer, expected_answer])
                if len(embeddings) != 2:
                    return None

                norm_a = np.linalg.norm(embeddings[0])
                norm_b = np.linalg.norm(embeddings[1])
                if norm_a == 0 or norm_b == 0:
                    return 0.0

                cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b))
                return max(0.0, min(1.0, cos_sim))
        except Exception as exc:
            logger.warning("[EXECUTE]   Semantic similarity failed: %s", exc)
            return None

    async def _build_agent_run_record(
        self,
        item,
        react_agent,
        react_prompt_template,
        profiler,
        metrics_svc,
        current_attempt: int,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Run a single example through the ReAct agent path."""
        with profiler.section("api_call"):
            agent_result = await asyncio.to_thread(react_agent.run, item["question"], profiler)

        with profiler.section("parsing"):
            parsed_answer = react_prompt_template.parse_response(agent_result.answer)

        logger.info(
            "[EXECUTE]   Agent: %s iters, %s tool calls, success=%s (%s)",
            agent_result.total_iterations,
            agent_result.tool_calls,
            agent_result.success,
            agent_result.termination_reason,
        )

        with profiler.section("metrics"):
            score_result = self._score_response(
                item=item,
                raw_output=agent_result.answer,
                parsed_answer=parsed_answer,
                metrics_svc=metrics_svc,
                use_robustness_scoring=use_robustness_scoring,
            )

        agent_failure_mode = None
        agent_error_message = None
        if not agent_result.success:
            from app.models.run import FailureMode

            agent_failure_mode = FailureMode.UNKNOWN
            agent_error_message = f"agent_termination:{agent_result.termination_reason}"
            score_result["is_correct"] = False
            score_result["score"] = 0.0

        return {
            "example_id": item["id"],
            "prompt": f"[Agent] {item['question']}",
            "raw_output": agent_result.answer,
            "expected_output": item["answer"],
            "is_correct": score_result["is_correct"],
            "score": score_result["score"],
            "is_exact_match": score_result["is_exact_match"],
            "is_substring_match": score_result["is_substring_match"],
            "parsed_answer": parsed_answer,
            "match_alias": score_result["match_alias"],
            "tokens_input": agent_result.total_tokens_input,
            "tokens_output": agent_result.total_tokens_output,
            "latency_ms": agent_result.total_latency_ms,
            "gpu_memory_mb": None,
            "agent_trace": agent_result.trace_as_dict(),
            "tool_calls": agent_result.tool_calls,
            "failure_mode": agent_failure_mode,
            "error_message": agent_error_message,
            "grader_results": score_result["grader_results"],
            "attempt": current_attempt,
        }

    async def _build_standard_run_record(
        self,
        *,
        item,
        experiment_response: ExperimentResponse,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        current_attempt: int,
        use_rag: bool,
        rag_config,
        rag_pipeline,
        faithfulness_scorer,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Run one non-agent example through retrieval, generation, and scoring."""
        context_chunks = []
        retrieval_context = ""
        retrieved_chunk_payload = None
        if use_rag:
            context_chunks, retrieval_context, retrieved_chunk_payload = self._retrieve_rag_context(
                item,
                rag_config,
                rag_pipeline,
                profiler,
            )

        with profiler.section("prompt_build"):
            prompt = self._build_prompt(
                item["question"],
                reasoning_method,
                cot_examples,
                context_chunks,
                naive_prompt_template,
                cot_prompt_template,
                rag_prompt_template,
            )

        cache_seed = self._generation_seed(gen_config)
        result = None
        if cache:
            with profiler.section("cache_lookup"):
                result = cache.get(
                    prompt,
                    experiment_response.config.model_name,
                    max_tokens,
                    gen_config.temperature,
                    cache_seed,
                )
            if result:
                logger.info("[EXECUTE]   Cache HIT for example %s", item["id"])

        if result is None:
            with profiler.section("api_call"):
                result = await asyncio.to_thread(engine.generate, prompt, gen_config)

            if cache:
                cache.put(
                    prompt,
                    experiment_response.config.model_name,
                    max_tokens,
                    gen_config.temperature,
                    cache_seed,
                    result,
                )

        with profiler.section("parsing"):
            parsed_answer = self._parse_generation_output(
                result.text,
                reasoning_method,
                used_rag=use_rag,
                naive_prompt_template=naive_prompt_template,
                cot_prompt_template=cot_prompt_template,
                rag_prompt_template=rag_prompt_template,
            )

        faithfulness = self._score_faithfulness(
            parsed_answer,
            retrieval_context,
            faithfulness_scorer,
            profiler,
        )

        with profiler.section("metrics"):
            score_result = self._score_response(
                item=item,
                raw_output=result.text,
                parsed_answer=parsed_answer,
                metrics_svc=metrics_svc,
                use_robustness_scoring=use_robustness_scoring,
            )

        ctx_relevance = self._score_context_relevance(item["question"], context_chunks, profiler)
        sem_sim = self._score_semantic_similarity(parsed_answer, item.get("answer"), profiler)

        return {
            "example_id": item["id"],
            "prompt": prompt,
            "raw_output": result.text,
            "expected_output": item["answer"],
            "is_correct": score_result["is_correct"],
            "score": score_result["score"],
            "is_exact_match": score_result["is_exact_match"],
            "is_substring_match": score_result["is_substring_match"],
            "parsed_answer": parsed_answer,
            "match_alias": score_result["match_alias"],
            "semantic_similarity": sem_sim,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "latency_ms": result.latency_ms,
            "gpu_memory_mb": result.gpu_memory_mb,
            "faithfulness_score": faithfulness,
            "retrieved_chunks": retrieved_chunk_payload,
            "context_relevance_score": ctx_relevance,
            "served_provider": result.served_provider,
            "routing_reason": result.routing_reason,
            "cost_usd": result.cost_usd,
            "failure_mode": result.failure_mode,
            "error_message": result.error_message,
            "grader_results": score_result["grader_results"],
            "attempt": current_attempt,
        }

    async def _flush_runs(self, run_service, experiment_id: UUID, runs_batch_data: List[dict[str, Any]], *, force: bool = False):
        """Flush buffered run rows to the database in consistent batch sizes."""
        if not runs_batch_data:
            return
        if not force and len(runs_batch_data) < 50:
            return

        await run_service.create_runs_batch(experiment_id, runs_batch_data)
        runs_batch_data.clear()

    async def _execute_batched_runs(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        examples,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        run_service,
        current_attempt: int,
        batch_size: int,
        naive_prompt_template,
        cot_prompt_template,
        use_robustness_scoring: bool,
    ) -> dict[str, int]:
        """Execute the non-RAG non-agent fast path using batched generation."""
        logger.info("[EXECUTE] Using batched execution (batch_size=%s)", batch_size)
        batch_stats = {"batches_processed": 0, "total_prompts_batched": 0}
        cache_seed = self._generation_seed(gen_config)

        for batch_start in range(0, len(examples), batch_size):
            batch_end = min(batch_start + batch_size, len(examples))
            batch_items = examples[batch_start:batch_end]
            logger.info(
                "[EXECUTE] Batch %s: examples %s-%s",
                batch_start // batch_size + 1,
                batch_start + 1,
                batch_end,
            )

            prompts = []
            cached_results = {}
            uncached_indices = []

            with profiler.section("prompt_build"):
                for local_idx, item in enumerate(batch_items):
                    if reasoning_method == "cot":
                        prompt = cot_prompt_template.format(item["question"], cot_examples)
                    else:
                        prompt = naive_prompt_template.format(item["question"])
                    prompts.append(prompt)

                    if not cache:
                        uncached_indices.append(local_idx)
                        continue

                    with profiler.section("cache_lookup"):
                        cached = cache.get(
                            prompt,
                            experiment_response.config.model_name,
                            max_tokens,
                            gen_config.temperature,
                            cache_seed,
                        )
                    if cached:
                        cached_results[local_idx] = cached
                        logger.info("[EXECUTE]   Cache HIT for example %s", batch_start + local_idx + 1)
                    else:
                        uncached_indices.append(local_idx)

            uncached_prompts = [prompts[idx] for idx in uncached_indices]
            batch_gen_results = []
            if uncached_prompts:
                with profiler.section("api_call"):
                    batch_gen_results = await asyncio.to_thread(
                        engine.generate_batch,
                        uncached_prompts,
                        gen_config,
                    )

                if cache:
                    for uncached_idx, gen_result in zip(uncached_indices, batch_gen_results):
                        cache.put(
                            prompts[uncached_idx],
                            experiment_response.config.model_name,
                            max_tokens,
                            gen_config.temperature,
                            cache_seed,
                            gen_result,
                        )

            gen_results_iterator = iter(batch_gen_results)
            all_results = []
            for local_idx in range(len(batch_items)):
                if local_idx in cached_results:
                    all_results.append(cached_results[local_idx])
                else:
                    all_results.append(next(gen_results_iterator))

            runs_batch_data = []
            for local_idx, (item, result) in enumerate(zip(batch_items, all_results)):
                with profiler.section("parsing"):
                    if reasoning_method == "cot":
                        parsed_answer = cot_prompt_template.parse_response(result.text)
                    else:
                        parsed_answer = naive_prompt_template.parse_response(result.text)

                with profiler.section("metrics"):
                    score_result = self._score_response(
                        item=item,
                        raw_output=result.text,
                        parsed_answer=parsed_answer,
                        metrics_svc=metrics_svc,
                        use_robustness_scoring=use_robustness_scoring,
                    )

                runs_batch_data.append(
                    {
                        "example_id": item["id"],
                        "prompt": prompts[local_idx],
                        "raw_output": result.text,
                        "expected_output": item["answer"],
                        "is_correct": score_result["is_correct"],
                        "score": score_result["score"],
                        "is_exact_match": score_result["is_exact_match"],
                        "is_substring_match": score_result["is_substring_match"],
                        "parsed_answer": parsed_answer,
                        "match_alias": score_result["match_alias"],
                        "tokens_input": result.tokens_input,
                        "tokens_output": result.tokens_output,
                        "latency_ms": result.latency_ms,
                        "gpu_memory_mb": result.gpu_memory_mb,
                        "served_provider": result.served_provider,
                        "routing_reason": result.routing_reason,
                        "cost_usd": result.cost_usd,
                        "failure_mode": result.failure_mode,
                        "error_message": result.error_message,
                        "grader_results": score_result["grader_results"],
                        "attempt": current_attempt,
                    }
                )

            if runs_batch_data:
                await run_service.create_runs_batch(experiment_id, runs_batch_data)

            batch_stats["batches_processed"] += 1
            batch_stats["total_prompts_batched"] += len(batch_items)

        return batch_stats

    async def _execute_sequential_runs(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        examples,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        run_service,
        current_attempt: int,
        use_batching: bool,
        use_rag: bool,
        use_robustness_scoring: bool,
        rag_config,
        rag_pipeline,
        faithfulness_scorer,
        react_agent,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
        react_prompt_template,
    ) -> dict[str, int]:
        """Execute the sequential path used by RAG and agent experiments."""
        if use_batching:
            logger.info("[EXECUTE] Batching disabled for RAG/agent execution")

        runs_batch_data: List[dict[str, Any]] = []
        for index, item in enumerate(examples):
            logger.info("[EXECUTE] Processing %s/%s: %s", index + 1, len(examples), item["id"])

            if reasoning_method == "react" and react_agent is not None:
                run_record = await self._build_agent_run_record(
                    item,
                    react_agent,
                    react_prompt_template,
                    profiler,
                    metrics_svc,
                    current_attempt,
                    use_robustness_scoring,
                )
            else:
                run_record = await self._build_standard_run_record(
                    item=item,
                    experiment_response=experiment_response,
                    reasoning_method=reasoning_method,
                    cot_examples=cot_examples,
                    gen_config=gen_config,
                    max_tokens=max_tokens,
                    engine=engine,
                    cache=cache,
                    profiler=profiler,
                    metrics_svc=metrics_svc,
                    current_attempt=current_attempt,
                    use_rag=use_rag,
                    rag_config=rag_config,
                    rag_pipeline=rag_pipeline,
                    faithfulness_scorer=faithfulness_scorer,
                    naive_prompt_template=naive_prompt_template,
                    cot_prompt_template=cot_prompt_template,
                    rag_prompt_template=rag_prompt_template,
                    use_robustness_scoring=use_robustness_scoring,
                )

            runs_batch_data.append(run_record)
            await self._flush_runs(run_service, experiment_id, runs_batch_data)

        await self._flush_runs(run_service, experiment_id, runs_batch_data, force=True)
        return {"batches_processed": 0, "total_prompts_batched": 0}

    async def _apply_graders(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        current_attempt: int,
    ) -> None:
        """Apply configured run graders to the latest attempt."""
        graders_config = experiment_response.config.graders
        if not graders_config or not graders_config.rules:
            return

        from app.models.run import Run as _GraderRun
        from app.services.grader_service import GraderEngine

        grader_engine = GraderEngine()
        reasoning = experiment_response.config.reasoning_method.value
        has_rag = bool(
            experiment_response.config.rag
            and experiment_response.config.rag.retrieval_method != "none"
        )

        grader_query = select(_GraderRun).where(
            _GraderRun.experiment_id == experiment_id,
            _GraderRun.attempt == current_attempt,
        )
        grader_result = await self.db.execute(grader_query)
        grader_runs = grader_result.scalars().all()

        for run in grader_runs:
            verdicts = [
                grader_engine.grade_run(run, rule, reasoning, has_rag)
                for rule in graders_config.rules
            ]
            existing_results = dict(run.grader_results or {})
            existing_results.update({verdict.grader_name: verdict.to_dict() for verdict in verdicts})
            run.grader_results = existing_results

        await self.db.flush()
        logger.info(
            "[EXECUTE] Applied %d graders to %d runs",
            len(graders_config.rules),
            len(grader_runs),
        )

    async def _save_optimization_report(
        self,
        *,
        experiment_id: UUID,
        engine,
        opt_report,
        batch_stats: dict[str, int],
        cache,
        profiler,
        wall_start: float,
    ) -> None:
        """Persist optimization and routing telemetry into the result raw_metrics."""
        from app.services.inference.provider_router import ProviderRouter as _PR
        from sqlalchemy.orm.attributes import flag_modified

        wall_end = _time.perf_counter()
        opt_report.cache_stats = cache.stats() if cache else {}
        opt_report.profiling_summary = profiler.summary()
        opt_report.batch_stats = dict(batch_stats)
        opt_report.total_wall_time_ms = (wall_end - wall_start) * 1000

        res_query = select(Result).where(Result.experiment_id == experiment_id)
        res_result = await self.db.execute(res_query)
        result_obj = res_result.scalar_one_or_none()
        if not result_obj:
            return

        existing_raw = dict(result_obj.raw_metrics or {})
        existing_raw["optimization"] = opt_report.to_dict()

        if isinstance(engine, _PR):
            existing_raw["routing"] = engine.stats_tracker.summary()
            logger.info("[EXECUTE] Routing telemetry saved to raw_metrics")

        result_obj.raw_metrics = existing_raw
        flag_modified(result_obj, "raw_metrics")
        await self.db.flush()
        await self.db.commit()
        logger.info("[EXECUTE] Optimization report saved to raw_metrics")

    async def _run_auto_regression_check(self, experiment_id: UUID) -> None:
        """Run the post-execution regression check when a baseline exists."""
        try:
            from app.models.experiment import Experiment as _RegExp
            from app.services.regression_service import RegressionService
            from sqlalchemy.orm.attributes import flag_modified as _flag_modified

            reg_svc = RegressionService(self.db)
            reg_query = select(_RegExp).where(_RegExp.id == experiment_id)
            reg_result = await self.db.execute(reg_query)
            reg_exp = reg_result.scalar_one_or_none()
            if not reg_exp:
                return

            baseline = await reg_svc.find_baseline(reg_exp)
            if not baseline or baseline.id == experiment_id:
                return

            verdict = await reg_svc.run_regression_check(experiment_id, baseline.id)

            res_query = select(Result).where(Result.experiment_id == experiment_id)
            res_result = await self.db.execute(res_query)
            result_obj = res_result.scalar_one_or_none()
            if result_obj:
                existing_raw = dict(result_obj.raw_metrics or {})
                existing_raw["regression"] = verdict.to_dict()
                result_obj.raw_metrics = existing_raw
                _flag_modified(result_obj, "raw_metrics")

            reg_exp.regression_status = regression_status_from_verdict(verdict.passed).value
            reg_exp.regression_passed = verdict.passed
            await self.db.flush()
            await self.db.commit()

            status = "PASS" if verdict.passed else ("FAIL" if verdict.passed is False else "INCONCLUSIVE")
            logger.info(
                "[EXECUTE] Regression check: %s (overlap=%.2f, violations=%d)",
                status,
                verdict.overlap_ratio,
                len(verdict.violations),
            )
        except Exception as reg_err:
            logger.warning("[EXECUTE] Regression check failed (non-fatal): %s", reg_err)

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
            engine = self._initialize_engine(
                experiment_response,
                custom_base_url=custom_base_url,
                custom_api_key=custom_api_key,
            )
            rag_config, use_rag, rag_pipeline, faithfulness_scorer = self._initialize_rag_runtime(experiment_response)

            # Step 4: Load dataset.
            examples = await self._load_examples(experiment_response, exp_obj)
            cot_examples = self._load_cot_examples(reasoning_method)
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


