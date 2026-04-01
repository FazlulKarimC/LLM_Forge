"""
Experiment Service

Business logic for experiment management.
Handles CRUD operations and experiment execution orchestration.
"""

import asyncio
import json
import logging
import os
import re
import time as _time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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

def _cot_examples_path() -> Path:
    """Resolve CoT examples path via settings (lazy, avoids circular import at module level)."""
    return settings.configs_dir / "cot_examples.json"


# ── Error sanitization ──────────────────────────────────────────────────────
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
        sanitized = sanitized[:_MAX_ERROR_LENGTH] + "…"
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
            # JSONB array containment — works on PostgreSQL
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
            logger.error(f"[EXECUTE] ❌ {error}")
            raise ValueError(error)
        logger.info("[EXECUTE] ✓ Found experiment: %s", experiment_response.name)
        
        try:
            from app.services.inference.base import GenerationConfig
            from app.services.inference.hf_api_engine import HFAPIEngine
            from app.services.inference.mock_engine import MockEngine
            from app.services.inference.prompting import (
                NaivePromptTemplate, CoTPromptTemplate, RAGPromptTemplate, ReActPromptTemplate
            )
            from app.services.run_service import RunService
            from app.services.dataset_service import DatasetService
            from app.services.metrics_service import MetricsService
            from app.services.optimization import PromptCache, ProfilerContext, OptimizationReport

            # Step 2: Initialize services — non-destructive re-runs (P1 #8)
            run_service = RunService(self.db)
            metrics_svc = MetricsService(self.db)
            
            # Instead of deleting old runs, increment attempt counter
            from sqlalchemy import select as _sel, func as _fn
            from app.models.run import Run as _Run
            max_attempt_q = await self.db.execute(
                _sel(_fn.coalesce(_fn.max(_Run.attempt), 0)).where(
                    _Run.experiment_id == experiment_id
                )
            )
            current_attempt = (max_attempt_q.scalar() or 0) + 1
            
            # Update experiment's current_attempt
            from app.models.experiment import Experiment as _Exp
            exp_row = await self.db.execute(
                _sel(_Exp).where(_Exp.id == experiment_id)
            )
            exp_obj = exp_row.scalar_one_or_none()
            if exp_obj:
                exp_obj.current_attempt = current_attempt
                exp_obj.regression_status = RegressionStatus.NOT_CHECKED.value
                exp_obj.regression_passed = None
            
            # Clear old results (will be recomputed from latest attempt)
            await metrics_svc.clear_results(experiment_id)
            
            # Step 2b: Update status to RUNNING
            await self.update_status(experiment_id, ExperimentStatus.RUNNING, error_message="")
            await self.db.commit()
            logger.info("[EXECUTE] ✓ Status: RUNNING (attempt %s)", current_attempt)
            
            # ─── Optimization setup (Phase 8) ───
            wall_start = _time.perf_counter()
            opt_config = experiment_response.config.optimization or OptimizationConfig()
            
            cache = PromptCache(max_size=opt_config.cache_max_size) if opt_config.enable_caching else None
            profiler = ProfilerContext(enabled=opt_config.enable_profiling)
            opt_report = OptimizationReport()
            
            logger.info(
                "[EXECUTE] Optimization: batching=%s (size=%s), caching=%s, profiling=%s",
                opt_config.enable_batching, opt_config.batch_size,
                opt_config.enable_caching, opt_config.enable_profiling,
            )
            
            
            # Step 3: Initialize inference engine (Phase 6 — provider-aware)
            model_name = experiment_response.config.model_name
            provider = getattr(experiment_response.config, 'provider', None)
            provider_value = provider.value if provider else "auto"
            engine_type = settings.INFERENCE_ENGINE

            if provider_value == "custom" and not custom_base_url:
                raise ValueError(
                    "Custom provider runs require stored custom endpoint credentials for the configured model."
                )
            
            # Auto-detect mock models regardless of provider setting
            if "mock" in model_name.lower():
                engine_type = "mock"
                logger.info("[EXECUTE] Auto-detected mock model '%s', using MockEngine", model_name)
                engine = MockEngine()
            elif provider_value == "custom":
                engine_type = "custom"
                logger.info("[EXECUTE] Custom base URL detected, using OpenAIEngine")
                from app.services.inference.openai_engine import OpenAIEngine
                engine = OpenAIEngine(
                    base_url=custom_base_url, 
                    api_key=custom_api_key, 
                    model_name=model_name
                )
            elif provider_value == "openrouter":
                engine_type = "openrouter"
                from app.services.inference.openrouter_engine import OpenRouterEngine
                engine = OpenRouterEngine(model_name=model_name)
            elif provider_value == "groq":
                engine_type = "groq"
                from app.services.inference.groq_engine import GroqEngine
                engine = GroqEngine(model_name=model_name)
            elif provider_value == "auto":
                # Build fallback chain: primary engine + available alternatives
                engine_type = "auto (router)"
                from app.services.inference.provider_router import ProviderRouter
                
                engines = []
                # Primary: HF API (always available if token is set)
                if settings.HF_TOKEN:
                    engines.append(HFAPIEngine(model_name=model_name))
                # Fallback 1: OpenRouter
                if settings.OPENROUTER_API_KEY:
                    try:
                        from app.services.inference.openrouter_engine import OpenRouterEngine
                        engines.append(OpenRouterEngine(model_name=model_name))
                    except Exception as e:
                        logger.warning("[EXECUTE] Could not init OpenRouter: %s", e)
                # Fallback 2: Groq
                if settings.GROQ_API_KEY:
                    try:
                        from app.services.inference.groq_engine import GroqEngine
                        engines.append(GroqEngine(model_name=model_name))
                    except Exception as e:
                        logger.warning("[EXECUTE] Could not init Groq: %s", e)
                
                if not engines:
                    # No API keys configured — fall back to HF API without key
                    engines.append(HFAPIEngine(model_name=model_name))
                
                # Apply routing config if present
                from app.services.inference.provider_stats import RoutingPolicy as _RP
                _routing_cfg = experiment_response.config.routing
                _router_policy = _RP.FALLBACK_CHAIN
                _epsilon = 0.15
                _exp_window = 10
                if _routing_cfg:
                    try:
                        _router_policy = _RP(_routing_cfg.policy)
                    except ValueError:
                        _router_policy = _RP.FALLBACK_CHAIN
                    _epsilon = _routing_cfg.epsilon
                    _exp_window = _routing_cfg.exploration_window
                
                engine = ProviderRouter(
                    engines=engines,
                    policy=_router_policy,
                    epsilon=_epsilon,
                    exploration_window=_exp_window,
                )
                logger.info("[EXECUTE] Router initialized with %d providers (policy=%s)", len(engines), _router_policy.value)
            else:
                # Default: hf_api or settings-based
                if engine_type == "hf_api":
                    engine = HFAPIEngine(model_name=model_name)
                else:
                    engine = MockEngine()
            
            logger.info("[EXECUTE] Engine type: %s", engine_type)
            engine.load_model(experiment_response.config.model_name)
            logger.info("[EXECUTE] ✓ Engine loaded: %s", experiment_response.config.model_name)
            
            # Step 3b: Initialize RAG pipeline if configured
            rag_pipeline = None
            rag_config = experiment_response.config.rag
            use_rag = rag_config and rag_config.retrieval_method.value != "none"
            
            if use_rag:
                from app.services.rag_service import RAGPipeline, FaithfulnessScorer
                logger.info("[EXECUTE] Initializing RAG pipeline (method=%s)", rag_config.retrieval_method.value)
                rag_pipeline = RAGPipeline()
                rag_pipeline.load_knowledge_base(chunk_size=rag_config.chunk_size)
                faithfulness_scorer = FaithfulnessScorer()
                logger.info("[EXECUTE] ✓ RAG pipeline initialized (top_k=%s)", rag_config.top_k)
            
            # Step 3c: Determine reasoning method (needed for agent init)
            reasoning_method = experiment_response.config.reasoning_method.value
            logger.info("[EXECUTE] Reasoning method: %s", reasoning_method)
            
            # Step 3d: Initialize ReAct agent if configured
            react_agent = None
            if reasoning_method == "react":
                from app.services.agent_service import (
                    ReActAgent, WikipediaSearchTool, CalculatorTool, RetrievalTool,
                )
                agent_config = experiment_response.config.agent
                enabled_tools_names = agent_config.tools if agent_config else ["wikipedia_search", "calculator"]
                agent_max_iter = agent_config.max_iterations if agent_config else 5
                
                # Build tool list
                agent_tools = []
                for tool_name in enabled_tools_names:
                    if tool_name == "wikipedia_search":
                        agent_tools.append(WikipediaSearchTool())
                    elif tool_name == "calculator":
                        agent_tools.append(CalculatorTool())
                    elif tool_name == "retrieval" and rag_pipeline:
                        agent_tools.append(RetrievalTool(rag_pipeline=rag_pipeline))
                    elif tool_name == "retrieval":
                        # Initialize RAG pipeline for retrieval tool
                        try:
                            from app.services.rag_service import RAGPipeline
                            rag_for_tool = RAGPipeline()
                            rag_for_tool.load_knowledge_base()
                            agent_tools.append(RetrievalTool(rag_pipeline=rag_for_tool))
                        except Exception as e:
                            logger.warning(f"[EXECUTE] ⚠ Could not init retrieval tool: {e}")
                
                logger.info(
                    "[EXECUTE] Initializing ReAct agent (max_iter=%s, tools=%s)",
                    agent_max_iter, [t.name for t in agent_tools]
                )
                
                _agent_tools = agent_tools
                _agent_max_iter = agent_max_iter
            
            # Step 4: Load dataset
            dataset_name = experiment_response.config.dataset_name
            num_samples = experiment_response.config.num_samples
            seed = experiment_response.config.hyperparameters.seed
            
            logger.info("[EXECUTE] Loading dataset: %s (n=%s, seed=%s)", dataset_name, num_samples, seed)
            
            examples = DatasetService.load(
                dataset_name=dataset_name,
                num_samples=num_samples,
                seed=seed,
            )
            logger.info("[EXECUTE] ✓ Loaded %s examples", len(examples))
            
            # P1 #11: Store dataset hash and sample IDs for reproducibility
            import hashlib
            dataset_content = json.dumps(examples, sort_keys=True)
            dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()
            sample_ids_list = [e.get("id", str(i)) for i, e in enumerate(examples)]
            
            if exp_obj:
                exp_obj.dataset_hash = dataset_hash
                exp_obj.sample_ids = sample_ids_list
                await self.db.flush()
            
            # Step 5: Prepare prompt template based on reasoning method
            cot_examples = None
            if reasoning_method == "cot":
                cot_path = _cot_examples_path()
                if cot_path.exists():
                    with cot_path.open("r", encoding="utf-8") as f:
                        cot_examples = json.load(f)
                    logger.info("[EXECUTE] ✓ Loaded %s CoT few-shot examples", len(cot_examples))
                else:
                    logger.warning("[EXECUTE] ⚠ CoT examples file not found (%s), using zero-shot CoT", cot_path)
            
            # Step 6: Prepare generation config
            max_tokens = experiment_response.config.hyperparameters.max_tokens
            if reasoning_method == "cot" and max_tokens <= 256:
                max_tokens = 512
                logger.info("[EXECUTE] ✓ Increased max_tokens to %s for CoT", max_tokens)
            elif reasoning_method == "react" and max_tokens <= 512:
                max_tokens = 1024
                logger.info("[EXECUTE] ✓ Increased max_tokens to %s for ReAct", max_tokens)
            
            gen_config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=experiment_response.config.hyperparameters.temperature,
                top_p=experiment_response.config.hyperparameters.top_p,
            )
            
            # Step 6b: Create ReAct agent now that gen_config is ready
            if reasoning_method == "react" and react_agent is None:
                from app.services.agent_service import ReActAgent as _ReActAgent
                react_agent = _ReActAgent(
                    engine=engine,
                    tools=_agent_tools,
                    max_iterations=_agent_max_iter,
                    gen_config=gen_config,
                )
                logger.info("[EXECUTE] ✓ ReAct agent created")
            
            # Step 7: Initialize services
            run_service = RunService(self.db)
            metrics_svc = MetricsService(self.db)
            
            # Step 8: Run inference
            logger.info("[EXECUTE] Running inference for %s examples...", len(examples))
            
            # ─── Decide execution strategy ───
            use_batching = (
                opt_config.enable_batching
                and reasoning_method != "react"  # Agent needs iterative tool calling
            )
            
            batch_stats = {"batches_processed": 0, "total_prompts_batched": 0}
            
            if use_batching and not use_rag:
                # ═══════════════════════════════════════════════
                # BATCHED execution path (non-RAG, non-agent)
                # ═══════════════════════════════════════════════
                batch_size = opt_config.batch_size
                logger.info("[EXECUTE] Using BATCHED execution (batch_size=%s)", batch_size)
                
                for batch_start in range(0, len(examples), batch_size):
                    batch_end = min(batch_start + batch_size, len(examples))
                    batch_items = examples[batch_start:batch_end]
                    
                    logger.info(f"[EXECUTE] Batch {batch_start // batch_size + 1}: examples {batch_start+1}-{batch_end}")
                    
                    # Build prompts for batch
                    prompts = []
                    cached_results = {}  # idx -> GenerationResult
                    uncached_indices = []
                    
                    with profiler.section("prompt_build"):
                        for local_idx, item in enumerate(batch_items):
                            if reasoning_method == "cot":
                                prompt = CoTPromptTemplate.format(item["question"], cot_examples)
                            else:
                                prompt = NaivePromptTemplate.format(item["question"])
                            prompts.append(prompt)
                            
                            # Check cache
                            if cache:
                                with profiler.section("cache_lookup"):
                                    cached = cache.get(
                                        prompt,
                                        experiment_response.config.model_name,
                                        max_tokens,
                                        gen_config.temperature,
                                        gen_config.seed if hasattr(gen_config, 'seed') else None,
                                    )
                                if cached:
                                    cached_results[local_idx] = cached
                                    logger.info(f"[EXECUTE]   Cache HIT for example {batch_start + local_idx + 1}")
                                else:
                                    uncached_indices.append(local_idx)
                            else:
                                uncached_indices.append(local_idx)
                    
                    batch_gen_results = []
                    uncached_prompts = [prompts[idx] for idx in uncached_indices] if uncached_indices else []
                    
                    if uncached_prompts:
                        with profiler.section("api_call"):
                            # Run blocking sync HTTP calls in thread-pool to avoid
                            # starving the uvicorn event loop.
                            batch_gen_results = await asyncio.to_thread(
                                engine.generate_batch,
                                uncached_prompts, gen_config,
                            )
                        
                        # Store in cache
                        if cache:
                            for uidx, gen_result in zip(uncached_indices, batch_gen_results):
                                cache.put(
                                    prompts[uidx],
                                    experiment_response.config.model_name,
                                    max_tokens,
                                    gen_config.temperature,
                                    gen_config.seed if hasattr(gen_config, 'seed') else None,
                                    gen_result,
                                )
                    
                    # Merge cached + generated results
                    gen_results_iterator = iter(batch_gen_results)
                    all_results = []
                    for local_idx in range(len(batch_items)):
                        if local_idx in cached_results:
                            all_results.append(cached_results[local_idx])
                        else:
                            all_results.append(next(gen_results_iterator))
                    
                    # Process results
                    runs_batch_data = []
                    for local_idx, (item, result) in enumerate(zip(batch_items, all_results)):
                        global_idx = batch_start + local_idx
                        
                        with profiler.section("parsing"):
                            if reasoning_method == "cot":
                                parsed_answer = CoTPromptTemplate.parse_response(result.text)
                            else:
                                parsed_answer = NaivePromptTemplate.parse_response(result.text)
                        
                        with profiler.section("metrics"):
                            aliases = item.get("aliases", [item["answer"]])
                            is_exact, is_substring, f1_score, matched_alias = metrics_svc.check_any_alias_match(
                                parsed_answer, aliases
                            )
                        
                        runs_batch_data.append({
                            "example_id": item["id"],
                            "prompt": prompts[local_idx],
                            "raw_output": result.text,
                            "expected_output": item["answer"],
                            "is_correct": is_exact or is_substring,
                            "score": f1_score,
                            "is_exact_match": is_exact,
                            "is_substring_match": is_substring,
                            "parsed_answer": parsed_answer,
                            "match_alias": matched_alias,
                            "tokens_input": result.tokens_input,
                            "tokens_output": result.tokens_output,
                            "latency_ms": result.latency_ms,
                            "gpu_memory_mb": result.gpu_memory_mb,
                            "served_provider": result.served_provider,
                            "failure_mode": result.failure_mode,
                            "error_message": result.error_message,
                            "attempt": current_attempt,
                        })
                    
                    if runs_batch_data:
                        await run_service.create_runs_batch(experiment_id, runs_batch_data)
                    
                    batch_stats["batches_processed"] += 1
                    batch_stats["total_prompts_batched"] += len(batch_items)
            else:
                # ═══════════════════════════════════════════════
                # SEQUENTIAL execution path (original + cache/profiling)
                # ═══════════════════════════════════════════════
                if use_batching:
                    logger.info("[EXECUTE] Batching disabled for RAG/Agent (requires sequential processing)")
                
                runs_batch_data = []

                for i, item in enumerate(examples):
                    logger.info("[EXECUTE] Processing %s/%s: %s", i+1, len(examples), item['id'])
                    
                    # ReAct agent path
                    if reasoning_method == "react" and react_agent is not None:
                        with profiler.section("api_call"):
                            # react_agent.run() makes multiple sync HTTP calls;
                            # offload to thread-pool to keep the event loop free.
                            agent_result = await asyncio.to_thread(
                                react_agent.run, item["question"], profiler
                            )
                        
                        with profiler.section("parsing"):
                            parsed_answer = ReActPromptTemplate.parse_response(agent_result.answer)
                        prompt = f"[Agent] {item['question']}"
                        raw_output = agent_result.answer
                        
                        logger.info(
                            f"[EXECUTE]   Agent: {agent_result.total_iterations} iters, "
                            f"{agent_result.tool_calls} tool calls, "
                            f"success={agent_result.success} ({agent_result.termination_reason})"
                        )
                        
                        with profiler.section("metrics"):
                            aliases = item.get("aliases", [item["answer"]])
                            is_exact, is_substring, f1_score, matched_alias = metrics_svc.check_any_alias_match(
                                parsed_answer, aliases
                            )
                        
                        runs_batch_data.append({
                            "example_id": item["id"],
                            "prompt": prompt,
                            "raw_output": raw_output,
                            "expected_output": item["answer"],
                            "is_correct": is_exact or is_substring,
                            "score": f1_score,
                            "is_exact_match": is_exact,
                            "is_substring_match": is_substring,
                            "parsed_answer": parsed_answer,
                            "match_alias": matched_alias,
                            "tokens_input": agent_result.total_tokens_input,
                            "tokens_output": agent_result.total_tokens_output,
                            "latency_ms": agent_result.total_latency_ms,
                            "gpu_memory_mb": None,
                            "agent_trace": agent_result.trace_as_dict(),
                            "tool_calls": agent_result.tool_calls,
                            "attempt": current_attempt,
                        })

                        if len(runs_batch_data) >= 50:
                            await run_service.create_runs_batch(experiment_id, runs_batch_data)
                            runs_batch_data = []

                        continue
                    
                    # RAG retrieval (if enabled)
                    context_chunks = []
                    retrieval_context = ""
                    retrieved_chunk_payload = None
                    if use_rag and rag_pipeline:
                        with profiler.section("rag_retrieval"):
                            retrieval_result = rag_pipeline.retrieve(
                                question=item["question"],
                                method=rag_config.retrieval_method.value,
                                top_k=rag_config.top_k,
                            )
                        context_chunks = [c.text for c in retrieval_result.chunks]
                        retrieved_chunk_payload = {
                            "chunks": [
                                {
                                    "text": c.text,
                                    "score": getattr(c, "score", None),
                                }
                                for c in retrieval_result.chunks
                            ]
                        }
                        retrieval_context = " ".join(context_chunks)
                        logger.info(f"[EXECUTE]   Retrieved {len(context_chunks)} chunks ({retrieval_result.latency_ms:.0f}ms)")
                    
                    # Format prompt based on reasoning method and RAG
                    with profiler.section("prompt_build"):
                        if use_rag and context_chunks:
                            prompt = RAGPromptTemplate.format(item["question"], context_chunks)
                        elif reasoning_method == "cot":
                            prompt = CoTPromptTemplate.format(item["question"], cot_examples)
                        else:
                            prompt = NaivePromptTemplate.format(item["question"])
                    
                    # Check cache before API call
                    result = None
                    if cache:
                        with profiler.section("cache_lookup"):
                            result = cache.get(
                                prompt,
                                experiment_response.config.model_name,
                                max_tokens,
                                gen_config.temperature,
                                gen_config.seed if hasattr(gen_config, 'seed') else None,
                            )
                        if result:
                            logger.info(f"[EXECUTE]   Cache HIT for example {i+1}")
                    
                    # Generate response (on cache miss)
                    if result is None:
                        with profiler.section("api_call"):
                            # engine.generate() is sync (requests-based); run in
                            # thread-pool so the event loop stays unblocked.
                            result = await asyncio.to_thread(
                                engine.generate, prompt, gen_config
                            )
                        
                        # Store in cache
                        if cache:
                            cache.put(
                                prompt,
                                experiment_response.config.model_name,
                                max_tokens,
                                gen_config.temperature,
                                gen_config.seed if hasattr(gen_config, 'seed') else None,
                                result,
                            )
                    
                    # Parse response based on reasoning method
                    with profiler.section("parsing"):
                        if use_rag:
                            parsed_answer = RAGPromptTemplate.parse_response(result.text)
                        elif reasoning_method == "cot":
                            parsed_answer = CoTPromptTemplate.parse_response(result.text)
                        else:
                            parsed_answer = NaivePromptTemplate.parse_response(result.text)
                    
                    # Compute faithfulness score (RAG only)
                    faithfulness = None
                    if use_rag and retrieval_context:
                        try:
                            with profiler.section("faithfulness"):
                                faithfulness = faithfulness_scorer.score(parsed_answer, retrieval_context)
                            logger.info(f"[EXECUTE]   Faithfulness: {faithfulness:.3f}")
                        except Exception as e:
                            logger.warning(f"[EXECUTE]   Faithfulness scoring failed: {e}")
                    
                    with profiler.section("metrics"):
                        aliases = item.get("aliases", [item["answer"]])
                        is_exact, is_substring, f1_score, matched_alias = metrics_svc.check_any_alias_match(
                            parsed_answer, aliases
                        )
                    
                    # P2 #12: Context relevance via CrossEncoder (RAG only)
                    ctx_relevance = None
                    if use_rag and context_chunks:
                        try:
                            with profiler.section("context_relevance"):
                                # Average reranker score across retrieved chunks
                                from app.services.rag_service import CrossEncoderReranker as _CER
                                _reranker_for_eval = _CER()
                                scored = _reranker_for_eval.rerank(
                                    item["question"],
                                    [type('C', (), {'id': f'c{ci}', 'text': ct, 'title': '', 'index': ci})() for ci, ct in enumerate(context_chunks)],
                                    top_k=len(context_chunks),
                                )
                                if scored:
                                    ctx_relevance = float(np.mean([s for _, s in scored]))
                        except Exception as e:
                            logger.warning(f"[EXECUTE]   Context relevance scoring failed: {e}")
                    
                    # P1 #9: Semantic similarity via embeddings
                    sem_sim = None
                    try:
                        if parsed_answer and item.get("answer"):
                            with profiler.section("semantic_similarity"):
                                from app.services.rag_service import EmbeddingService as _ES
                                _emb_svc = _ES()
                                embs = _emb_svc.embed([parsed_answer, item["answer"]])
                                if len(embs) == 2:
                                    norm_a = np.linalg.norm(embs[0])
                                    norm_b = np.linalg.norm(embs[1])
                                    if norm_a > 0 and norm_b > 0:
                                        cos_sim = float(np.dot(embs[0], embs[1]) / (norm_a * norm_b))
                                        sem_sim = max(0.0, min(1.0, cos_sim))
                                    else:
                                        sem_sim = 0.0
                    except Exception as e:
                        logger.warning(f"[EXECUTE]   Semantic similarity failed: {e}")
                    
                    runs_batch_data.append({
                        "example_id": item["id"],
                        "prompt": prompt,
                        "raw_output": result.text,
                        "expected_output": item["answer"],
                        "is_correct": is_exact or is_substring,
                        "score": f1_score,
                        "is_exact_match": is_exact,
                        "is_substring_match": is_substring,
                        "parsed_answer": parsed_answer,
                        "match_alias": matched_alias,
                        "semantic_similarity": sem_sim,
                        "tokens_input": result.tokens_input,
                        "tokens_output": result.tokens_output,
                        "latency_ms": result.latency_ms,
                        "gpu_memory_mb": result.gpu_memory_mb,
                        "faithfulness_score": faithfulness,
                        "retrieved_chunks": retrieved_chunk_payload,
                        "context_relevance_score": ctx_relevance,
                        "served_provider": result.served_provider,
                        "failure_mode": result.failure_mode,
                        "error_message": result.error_message,
                        "attempt": current_attempt,
                    })

                    if len(runs_batch_data) >= 50:
                        await run_service.create_runs_batch(experiment_id, runs_batch_data)
                        runs_batch_data = []
            
            if not use_batching and runs_batch_data:
                await run_service.create_runs_batch(experiment_id, runs_batch_data)
            
            # Step 8: Commit all runs
            logger.info("[EXECUTE] Committing %s runs to database...", len(examples))
            await self.db.commit()

            # Step 8b: Apply graders (if configured)
            graders_config = experiment_response.config.graders
            if graders_config and graders_config.rules:
                from app.services.grader_service import GraderEngine
                grader_engine = GraderEngine()

                reasoning = experiment_response.config.reasoning_method.value
                has_rag = bool(
                    experiment_response.config.rag
                    and experiment_response.config.rag.retrieval_method != "none"
                )

                # Load latest-attempt runs for grading
                from app.models.run import Run as _GraderRun
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
                    run.grader_results = {v.grader_name: v.to_dict() for v in verdicts}

                await self.db.flush()
                logger.info(
                    "[EXECUTE] ✓ Applied %d graders to %d runs",
                    len(graders_config.rules), len(grader_runs),
                )
            
            # Step 9: Compute aggregate metrics and save Result
            logger.info("[EXECUTE] Computing aggregate metrics...")
            wall_end_for_metrics = _time.perf_counter()
            wall_ms = (wall_end_for_metrics - wall_start) * 1000
            await metrics_svc.compute_and_save(experiment_id, wall_clock_ms=wall_ms)
            
            # ─── Step 9b: Save optimization report into raw_metrics ───
            wall_end = _time.perf_counter()
            opt_report.cache_stats = cache.stats() if cache else {}
            opt_report.profiling_summary = profiler.summary()
            opt_report.batch_stats = dict(batch_stats)
            opt_report.total_wall_time_ms = (wall_end - wall_start) * 1000
            
            res_query = select(Result).where(Result.experiment_id == experiment_id)
            res_result = await self.db.execute(res_query)
            result_obj = res_result.scalar_one_or_none()
            
            if result_obj:
                existing_raw = dict(result_obj.raw_metrics or {})
                existing_raw["optimization"] = opt_report.to_dict()
                
                # Save routing telemetry if using router
                from app.services.inference.provider_router import ProviderRouter as _PR
                if isinstance(engine, _PR):
                    existing_raw["routing"] = engine.stats_tracker.summary()
                    logger.info("[EXECUTE] ✓ Routing telemetry saved to raw_metrics")
                
                result_obj.raw_metrics = existing_raw
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(result_obj, "raw_metrics")
                await self.db.flush()
                await self.db.commit()
                logger.info("[EXECUTE] ✓ Optimization report saved to raw_metrics")
            
            # Step 10: Cleanup
            engine.unload_model()
            
            # Step 11: Update status to COMPLETED
            await self.update_status(experiment_id, ExperimentStatus.COMPLETED)
            await self.db.commit()
            
            # Step 12: Auto-regression check (if baseline exists)
            try:
                from app.services.regression_service import RegressionService
                reg_svc = RegressionService(self.db)
                
                # Reload experiment for baseline lookup
                from app.models.experiment import Experiment as _RegExp
                reg_query = select(_RegExp).where(_RegExp.id == experiment_id)
                reg_result = await self.db.execute(reg_query)
                reg_exp = reg_result.scalar_one_or_none()
                
                if reg_exp:
                    baseline = await reg_svc.find_baseline(reg_exp)
                    if baseline and baseline.id != experiment_id:
                        verdict = await reg_svc.run_regression_check(experiment_id, baseline.id)
                        
                        # Merge verdict into raw_metrics
                        res_q2 = select(Result).where(Result.experiment_id == experiment_id)
                        res_r2 = await self.db.execute(res_q2)
                        result_obj2 = res_r2.scalar_one_or_none()
                        
                        if result_obj2:
                            existing_raw2 = dict(result_obj2.raw_metrics or {})
                            existing_raw2["regression"] = verdict.to_dict()
                            result_obj2.raw_metrics = existing_raw2
                            from sqlalchemy.orm.attributes import flag_modified as _fm2
                            _fm2(result_obj2, "raw_metrics")
                        
                        # Denormalize for list-view badges
                        reg_exp.regression_status = regression_status_from_verdict(verdict.passed).value
                        reg_exp.regression_passed = verdict.passed
                        
                        await self.db.flush()
                        await self.db.commit()
                        
                        status = "PASS" if verdict.passed else ("FAIL" if verdict.passed is False else "INCONCLUSIVE")
                        logger.info(
                            "[EXECUTE] ✓ Regression check: %s (overlap=%.2f, violations=%d)",
                            status, verdict.overlap_ratio, len(verdict.violations),
                        )
            except Exception as reg_err:
                # Regression check failure must not fail the experiment
                logger.warning("[EXECUTE] Regression check failed (non-fatal): %s", reg_err)
            
            logger.info(
                "[EXECUTE] ✅ EXECUTION COMPLETED (wall time: %.0fms)",
                opt_report.total_wall_time_ms
            )
            
        except Exception as e:
            logger.exception("[EXECUTE] ❌ EXECUTION FAILED: %s: %s", type(e).__name__, e)
            
            error_message = _sanitize_error_message(e)
            await self.update_status(
                experiment_id,
                ExperimentStatus.FAILED,
                error_message=error_message
            )
            await self.db.commit()
            
            raise


