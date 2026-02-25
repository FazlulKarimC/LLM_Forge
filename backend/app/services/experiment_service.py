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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.config import settings
from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatus,
    ExperimentConfig,
    OptimizationConfig,
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

            # Step 2: Initialize services and clear old data for re-runs
            run_service = RunService(self.db)
            metrics_svc = MetricsService(self.db)
            
            await run_service.clear_runs(experiment_id)
            await metrics_svc.clear_results(experiment_id)
            
            # Step 2b: Update status to RUNNING
            await self.update_status(experiment_id, ExperimentStatus.RUNNING, error_message="")
            await self.db.commit()
            logger.info("[EXECUTE] ✓ Status: RUNNING (and old data cleared)")
            
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
            
            
            # Step 3: Initialize inference engine
            model_name = experiment_response.config.model_name
            engine_type = settings.INFERENCE_ENGINE
            
            # Auto-detect mock models regardless of INFERENCE_ENGINE setting
            if "mock" in model_name.lower():
                engine_type = "mock"
                logger.info("[EXECUTE] Auto-detected mock model '%s', using MockEngine", model_name)
            elif custom_base_url:
                engine_type = "openai_compatible"
                logger.info("[EXECUTE] Custom base URL detected, using OpenAIEngine")
            
            logger.info("[EXECUTE] Engine type: %s", engine_type)
            
            if engine_type == "hf_api":
                engine = HFAPIEngine(model_name=model_name)
            elif engine_type == "openai_compatible":
                from app.services.inference.openai_engine import OpenAIEngine
                engine = OpenAIEngine(
                    base_url=custom_base_url, 
                    api_key=custom_api_key, 
                    model_name=model_name
                )
            else:
                engine = MockEngine()
            
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
                    for local_idx, (item, result) in enumerate(zip(batch_items, all_results)):
                        global_idx = batch_start + local_idx
                        
                        with profiler.section("parsing"):
                            if reasoning_method == "cot":
                                parsed_answer = CoTPromptTemplate.parse_response(result.text)
                            else:
                                parsed_answer = NaivePromptTemplate.parse_response(result.text)
                        
                        with profiler.section("metrics"):
                            aliases = item.get("aliases", [item["answer"]])
                            is_exact, is_substring, f1_score = metrics_svc.check_any_alias_match(
                                parsed_answer, aliases
                            )
                        
                        await run_service.create_run(
                            experiment_id=experiment_id,
                            example_id=item["id"],
                            input_text=prompts[local_idx],
                            output_text=result.text,
                            expected_output=item["answer"],
                            is_correct=is_exact or is_substring,
                            score=f1_score,
                            tokens_input=result.tokens_input,
                            tokens_output=result.tokens_output,
                            latency_ms=result.latency_ms,
                            gpu_memory_mb=result.gpu_memory_mb,
                        )
                    
                    batch_stats["batches_processed"] += 1
                    batch_stats["total_prompts_batched"] += len(batch_items)
            else:
                # ═══════════════════════════════════════════════
                # SEQUENTIAL execution path (original + cache/profiling)
                # ═══════════════════════════════════════════════
                if use_batching:
                    logger.info("[EXECUTE] Batching disabled for RAG/Agent (requires sequential processing)")
                
                for i, item in enumerate(examples):
                    logger.info("[EXECUTE] Processing %s/%s: %s", i+1, len(examples), item['id'])
                    
                    # ReAct agent path
                    if reasoning_method == "react" and react_agent is not None:
                        with profiler.section("api_call"):
                            # react_agent.run() makes multiple sync HTTP calls;
                            # offload to thread-pool to keep the event loop free.
                            agent_result = await asyncio.to_thread(
                                react_agent.run, item["question"]
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
                            is_exact, is_substring, f1_score = metrics_svc.check_any_alias_match(
                                parsed_answer, aliases
                            )
                        
                        await run_service.create_run(
                            experiment_id=experiment_id,
                            example_id=item["id"],
                            input_text=prompt,
                            output_text=raw_output,
                            expected_output=item["answer"],
                            is_correct=is_exact or is_substring,
                            score=f1_score,
                            tokens_input=agent_result.total_tokens_input,
                            tokens_output=agent_result.total_tokens_output,
                            latency_ms=agent_result.total_latency_ms,
                            gpu_memory_mb=None,
                            agent_trace=agent_result.trace_as_dict(),
                            tool_calls=agent_result.tool_calls,
                        )
                        continue
                    
                    # RAG retrieval (if enabled)
                    context_chunks = []
                    retrieval_context = ""
                    if use_rag and rag_pipeline:
                        with profiler.section("rag_retrieval"):
                            retrieval_result = rag_pipeline.retrieve(
                                question=item["question"],
                                method=rag_config.retrieval_method.value,
                                top_k=rag_config.top_k,
                            )
                        context_chunks = [c.text for c in retrieval_result.chunks]
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
                    
                    # Evaluate against aliases
                    with profiler.section("metrics"):
                        aliases = item.get("aliases", [item["answer"]])
                        is_exact, is_substring, f1_score = metrics_svc.check_any_alias_match(
                            parsed_answer, aliases
                        )
                    
                    # Log run to database
                    await run_service.create_run(
                        experiment_id=experiment_id,
                        example_id=item["id"],
                        input_text=prompt,
                        output_text=result.text,
                        expected_output=item["answer"],
                        is_correct=is_exact or is_substring,
                        score=f1_score,
                        tokens_input=result.tokens_input,
                        tokens_output=result.tokens_output,
                        latency_ms=result.latency_ms,
                        gpu_memory_mb=result.gpu_memory_mb,
                        faithfulness_score=faithfulness,
                        retrieved_chunks={"chunks": context_chunks} if use_rag else None,
                    )
            
            # Step 8: Commit all runs
            logger.info("[EXECUTE] Committing %s runs to database...", len(examples))
            await self.db.commit()
            
            # Step 9: Compute aggregate metrics and save Result
            logger.info("[EXECUTE] Computing aggregate metrics...")
            await metrics_svc.compute_and_save(experiment_id)
            await self.db.commit()
            logger.info("[EXECUTE] ✓ Metrics computed and saved")
            
            # ─── Step 9b: Save optimization report into raw_metrics ───
            wall_end = _time.perf_counter()
            opt_report.total_wall_time_ms = (wall_end - wall_start) * 1000
            opt_report.cache_stats = cache.stats() if cache else {}
            opt_report.profiling_summary = profiler.summary()
            opt_report.batch_stats = batch_stats
            
            # Update raw_metrics on the latest Result row
            from sqlalchemy import select
            from app.models.result import Result
            result_row = await self.db.execute(
                select(Result).where(Result.experiment_id == experiment_id)
                .order_by(Result.computed_at.desc()).limit(1)
            )
            result_obj = result_row.scalar_one_or_none()
            if result_obj:
                existing_raw = result_obj.raw_metrics or {}
                existing_raw["optimization"] = opt_report.to_dict()
                result_obj.raw_metrics = existing_raw
                await self.db.flush()
                await self.db.commit()
                logger.info("[EXECUTE] ✓ Optimization report saved to raw_metrics")
            
            # Step 10: Cleanup
            engine.unload_model()
            
            # Step 11: Update status to COMPLETED
            await self.update_status(experiment_id, ExperimentStatus.COMPLETED)
            await self.db.commit()
            
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


