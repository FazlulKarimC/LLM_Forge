"""Runtime setup helpers for experiment execution.

This module owns the mutable, environment-dependent setup work that happens
just before an experiment attempt runs: engines, datasets, prompts, RAG, agent
tools, generation settings, and effective execution provenance.
"""

import asyncio
import hashlib
import json
import logging
import re
import time as _time
from pathlib import Path
from typing import Optional, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentResponse, OptimizationConfig
from app.services.experiment_provenance import build_effective_execution_manifest_entry

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


def cot_examples_path() -> Path:
    """Resolve CoT examples path via settings."""
    return settings.configs_dir / "cot_examples.json"


_PATH_PATTERN = re.compile(r"(?:/[\w.\-]+){2,}")
_WIN_PATH_PATTERN = re.compile(r"[A-Za-z]:\\(?:[\w.\- ]+\\)*[\w.\- ]+")
_TOKEN_PATTERN = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|[A-Fa-f0-9]{32,})"
)
_MAX_ERROR_LENGTH = 500


def sanitize_error_message(exc: Exception) -> str:
    """Build a safe error string from an exception."""
    raw = f"{type(exc).__name__}: {exc}"
    sanitized = _PATH_PATTERN.sub("<path>", raw)
    sanitized = _WIN_PATH_PATTERN.sub("<path>", sanitized)
    sanitized = _TOKEN_PATTERN.sub("<redacted>", sanitized)
    if len(sanitized) > _MAX_ERROR_LENGTH:
        sanitized = sanitized[:_MAX_ERROR_LENGTH] + "..."
    return f"Execution failed: {sanitized}"


class ExperimentRuntimeBuilder:
    """Create runtime collaborators and persist attempt-level provenance."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def create_optimization_runtime(self, experiment_response: ExperimentResponse):
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

    def initialize_engine(
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
        return engine, engine_type

    def initialize_rag_runtime(self, experiment_response: ExperimentResponse):
        """Initialize RAG services only when retrieval is enabled."""
        rag_pipeline = None
        faithfulness_scorer = None
        rag_config = experiment_response.config.rag
        use_rag = bool(rag_config and rag_config.retrieval_method.value != "none")

        if use_rag:
            from app.services.rag_service import RAGPipeline, FaithfulnessScorer

            assert rag_config is not None  # guaranteed by use_rag check above
            logger.info("[EXECUTE] Initializing RAG pipeline (method=%s)", rag_config.retrieval_method.value)
            rag_pipeline = RAGPipeline()
            rag_pipeline.load_knowledge_base(chunk_size=rag_config.chunk_size)
            faithfulness_scorer = FaithfulnessScorer()
            logger.info("[EXECUTE] RAG pipeline initialized (top_k=%s)", rag_config.top_k)

        return rag_config, use_rag, rag_pipeline, faithfulness_scorer

    async def load_examples(self, experiment_response: ExperimentResponse, exp_obj: Optional[Experiment]):
        """Load the dataset slice for this run and persist a reproducibility fingerprint."""
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
            exp_obj.dataset_hash = dataset_hash  # type: ignore[assignment]
            exp_obj.sample_ids = sample_ids_list  # type: ignore[assignment]
            await self.db.flush()

        return examples

    async def load_cot_examples(self, reasoning_method: str):
        """Load few-shot CoT examples when chain-of-thought prompting is enabled."""
        if reasoning_method != "cot":
            return None

        cot_path = cot_examples_path()
        if not cot_path.exists():
            logger.warning("[EXECUTE] CoT examples file not found (%s), using zero-shot CoT", cot_path)
            return None

        def _read_cot():
            with cot_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        cot_examples = await asyncio.to_thread(_read_cot)
        logger.info("[EXECUTE] Loaded %s CoT few-shot examples", len(cot_examples))
        return cot_examples

    async def resolve_prompt_templates(
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
                cast(str, prompt_version.template_text),
                rag_prompt_template.parse_response,
            )
        elif reasoning_method == "cot":
            cot_prompt_template = VersionedPromptTemplate(
                cast(str, prompt_version.template_text),
                cot_prompt_template.parse_response,
            )
        else:
            naive_prompt_template = VersionedPromptTemplate(
                cast(str, prompt_version.template_text),
                naive_prompt_template.parse_response,
            )

        return naive_prompt_template, cot_prompt_template, rag_prompt_template

    def build_generation_config(self, experiment_response: ExperimentResponse, reasoning_method: str):
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

    async def persist_effective_execution_manifest(
        self,
        *,
        experiment_response: ExperimentResponse,
        exp_obj: Optional[Experiment],
        current_attempt: int,
        engine_type: str,
        gen_config,
        max_tokens: int,
        examples,
        use_rag: bool,
        use_batching: bool,
        opt_config: OptimizationConfig,
    ) -> None:
        """Persist the effective settings that actually governed this attempt."""
        if not exp_obj:
            return

        from sqlalchemy.orm.attributes import flag_modified

        configured_hp = experiment_response.config.hyperparameters.model_dump(mode="json")
        effective_hp = {
            "max_tokens": max_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "seed": gen_config.seed,
        }
        routing_config = experiment_response.config.routing
        provider = experiment_response.config.provider.value if experiment_response.config.provider else "auto"
        effective_execution, effective_manifest_hash = build_effective_execution_manifest_entry(
            attempt=current_attempt,
            engine_type=engine_type,
            provider=provider,
            routing_config=routing_config,
            configured_hyperparameters=configured_hp,
            effective_hyperparameters=effective_hp,
            dataset_hash=cast(Optional[str], exp_obj.dataset_hash),
            sample_ids=cast(Optional[list], exp_obj.sample_ids),
            sample_count=len(examples),
            execution_mode="batched" if use_batching and not use_rag else "sequential",
            rag_enabled=use_rag,
            optimization=opt_config,
        )
        manifest = dict(cast(dict, exp_obj.run_manifest or experiment_response.run_manifest or {}))
        manifest["effective_execution"] = effective_execution
        manifest["effective_manifest_hash"] = effective_manifest_hash
        exp_obj.run_manifest = manifest  # type: ignore[assignment]
        flag_modified(exp_obj, "run_manifest")
        await self.db.flush()

    def create_react_agent(self, experiment_response: ExperimentResponse, rag_pipeline, engine, gen_config):
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
