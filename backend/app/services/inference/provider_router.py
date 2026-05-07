"""
Multi-Provider Router

Routes LLM inference requests across multiple providers with:
- Configurable routing policies (fallback_chain, cheapest_first, fastest_first, adaptive)
- Automatic fallback on rate limits and transient provider failures
- Per-request provider tracking for cost attribution
- Thread-safe stats tracking for adaptive decisions
- Batch-path routing guard: per-prompt routing when multiple providers are available

Supported Providers:
- hf_api: HuggingFace Inference Providers API (default)
- openrouter: OpenRouter (free models available)
- groq: Groq (fast inference, strict rate limits)
- custom: User-provided OpenAI-compatible endpoint
"""

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from app.models.run import FailureMode
from app.services.inference.base import (
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)
from app.services.inference.provider_stats import ProviderStatsTracker, RoutingPolicy

logger = logging.getLogger(__name__)


class ProviderRouter(InferenceEngine):
    """
    Routes inference across providers with automatic fallback and
    policy-driven selection (cheapest, fastest, adaptive).
    """

    def __init__(
        self,
        engines: List[InferenceEngine],
        policy: RoutingPolicy = RoutingPolicy.FALLBACK_CHAIN,
        stats_tracker: Optional[ProviderStatsTracker] = None,
        epsilon: float = 0.15,
        exploration_window: int = 10,
        strict_comparison: bool = False,
    ):
        if not engines:
            raise ValueError("At least one engine is required")

        self._engines = engines
        self._policy = policy
        self._model_name: Optional[str] = None
        self._stats = stats_tracker or ProviderStatsTracker()
        self._epsilon = epsilon
        self._exploration_window = exploration_window
        self._strict_comparison = strict_comparison
        self._request_count = 0
        self._request_count_lock = threading.Lock()

        logger.info(
            "ProviderRouter initialized (policy=%s, engines=%s, epsilon=%.2f, strict=%s)",
            policy.value,
            [self._get_engine_name(e) for e in engines],
            epsilon,
            strict_comparison,
        )

    @property
    def stats_tracker(self) -> ProviderStatsTracker:
        """Access the stats tracker for summary/persistence."""
        return self._stats

    def load_model(self, model_name: str) -> None:
        """Load model on all engines (each engine maps it to provider-specific name)."""
        self._model_name = model_name
        for engine in self._engines:
            try:
                engine.load_model(model_name)
            except Exception as exc:
                logger.warning(
                    "Failed to load model %s on %s: %s",
                    model_name,
                    type(engine).__name__,
                    exc,
                )

    def _get_engine_name(self, engine: InferenceEngine) -> str:
        """Get a consistent name for an engine."""
        return getattr(engine, "provider_id", type(engine).__name__)

    def _get_available_engines(self) -> List[InferenceEngine]:
        """Get engines that have a model loaded."""
        return [engine for engine in self._engines if engine.is_loaded]

    @staticmethod
    def _is_retryable_result(result: GenerationResult) -> bool:
        """Whether a returned result should trigger provider fallback."""
        if result.failure_mode == FailureMode.TIMEOUT:
            return True
        if result.failure_mode != FailureMode.API_ERROR:
            return False

        err_lower = (result.error_message or "").lower()
        return any(
            keyword in err_lower
            for keyword in (
                "429",
                "rate limit",
                "too many requests",
                "connection",
                "timeout",
                "unavailable",
                "temporarily overloaded",
                "server error",
                "5xx",
            )
        )

    @staticmethod
    def _should_record_error(result: GenerationResult) -> bool:
        """Count provider-side failures in routing telemetry."""
        return result.failure_mode in {
            FailureMode.API_ERROR,
            FailureMode.TIMEOUT,
            FailureMode.CONTEXT_EXCEEDED,
        }

    def _record_result_stats(self, engine_name: str, result: GenerationResult) -> None:
        """Persist routing telemetry for a returned generation result."""
        self._stats.record(
            engine_name,
            result.latency_ms or 0.0,
            result.tokens_input or 0,
            result.tokens_output or 0,
            result.cost_usd or 0.0,
            self._should_record_error(result),
        )

    def _select_engine(self) -> InferenceEngine:
        """
        Select an engine based on current routing policy.

        Returns:
            Selected engine, falls back to first available on any issue.
        """
        available = self._get_available_engines()
        if not available:
            return self._engines[0]

        if self._strict_comparison or self._policy == RoutingPolicy.FALLBACK_CHAIN:
            return available[0]

        available_names = [self._get_engine_name(engine) for engine in available]
        engine_map: Dict[str, InferenceEngine] = {
            self._get_engine_name(engine): engine for engine in available
        }

        if self._policy == RoutingPolicy.ADAPTIVE:
            return self._select_adaptive(available, engine_map, available_names)

        recommended = self._stats.recommend(self._policy, available_names)
        if recommended and recommended in engine_map:
            return engine_map[recommended]

        return available[0]

    def _select_adaptive(
        self,
        available: List[InferenceEngine],
        engine_map: Dict[str, InferenceEngine],
        available_names: List[str],
    ) -> InferenceEngine:
        """Epsilon-greedy adaptive selection with exploration window."""
        with self._request_count_lock:
            count = self._request_count

        if count < self._exploration_window:
            idx = count % len(available)
            return available[idx]

        if random.random() < self._epsilon:
            return random.choice(available)

        recommended = self._stats.recommend(RoutingPolicy.ADAPTIVE, available_names)
        if recommended and recommended in engine_map:
            return engine_map[recommended]

        return available[0]

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text with policy-driven selection and fallback.

        Records stats for adaptive routing decisions.
        """
        selected = self._select_engine()

        with self._request_count_lock:
            self._request_count += 1

        if self._strict_comparison:
            engines_to_try = [selected]
        else:
            engines_to_try = [selected] + [
                engine for engine in self._engines if engine is not selected and engine.is_loaded
            ]
        last_error = None

        for index, engine in enumerate(engines_to_try):
            engine_name = self._get_engine_name(engine)
            start_time = time.perf_counter()

            try:
                result = engine.generate(prompt, config)
                if not result.latency_ms:
                    result.latency_ms = (time.perf_counter() - start_time) * 1000

                self._record_result_stats(engine_name, result)

                if self._is_retryable_result(result):
                    last_error = result.error_message or result.failure_mode.value
                    if not self._strict_comparison and index < len(engines_to_try) - 1:
                        logger.warning(
                            "Provider %s returned retryable failure (%s), trying next provider...",
                            engine_name,
                            result.error_message or result.failure_mode,
                        )
                        continue

                result.served_provider = engine_name
                reason = "selected" if engine is selected else f"fallback_{index}"
                if self._strict_comparison:
                    reason = "strict_selected"
                elif self._policy == RoutingPolicy.ADAPTIVE:
                    reason = f"adaptive_{reason}"
                result.routing_reason = reason
                if self._is_retryable_result(result) and not self._strict_comparison:
                    break
                return result

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._stats.record(engine_name, elapsed_ms, 0, 0, 0.0, True)

                last_error = str(exc)
                err_lower = str(exc).lower()
                is_retryable = any(
                    keyword in err_lower
                    for keyword in (
                        "429",
                        "rate limit",
                        "too many requests",
                        "connection",
                        "timeout",
                        "unavailable",
                    )
                )

                if not self._strict_comparison and is_retryable and index < len(engines_to_try) - 1:
                    logger.warning(
                        "Provider %s failed (%s), falling back...",
                        engine_name,
                        exc,
                    )
                    continue
                raise

        return GenerationResult(
            text="",
            tokens_input=len(prompt.split()),
            tokens_output=0,
            latency_ms=0,
            finish_reason="error",
            failure_mode=FailureMode.API_ERROR,
            error_message=f"All providers exhausted. Last error: {last_error}",
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        max_workers: int = 8,
    ) -> List[GenerationResult]:
        """
        Generate batch with routing-aware behavior.

        - Single-provider fallback chains: delegate whole batch to the engine
        - Multi-provider fallback chains: route each prompt through self.generate()
        - Other policies: per-prompt routing through self.generate()
        """
        available = self._get_available_engines()

        if self._strict_comparison:
            selected = self._select_engine()
            engine_name = self._get_engine_name(selected)
            try:
                results = selected.generate_batch(prompts, config, max_workers)
                for result in results:
                    result.served_provider = engine_name
                    result.routing_reason = "strict_batch_delegate"
                    self._record_result_stats(engine_name, result)
                return results
            except Exception as exc:
                logger.warning(
                    "Strict batch generation failed on %s: %s; falling back to strict sequential",
                    engine_name,
                    exc,
                )
                return [self.generate(prompt, config) for prompt in prompts]

        if self._policy == RoutingPolicy.FALLBACK_CHAIN and len(available) <= 1:
            for engine in self._engines:
                if not engine.is_loaded:
                    continue
                try:
                    results = engine.generate_batch(prompts, config, max_workers)
                    engine_name = self._get_engine_name(engine)
                    for result in results:
                        result.served_provider = engine_name
                        result.routing_reason = "batch_delegate"
                        self._record_result_stats(engine_name, result)
                    return results
                except Exception as exc:
                    logger.warning(
                        "Batch generation failed on %s: %s",
                        self._get_engine_name(engine),
                        exc,
                    )

            logger.warning("All engine batch methods failed, falling back to sequential")
            return [self.generate(prompt, config) for prompt in prompts]

        if self._policy == RoutingPolicy.FALLBACK_CHAIN and len(available) > 1:
            logger.info(
                "Using per-prompt router fallback for batch generation across %d providers",
                len(available),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda prompt: self.generate(prompt, config), prompts))

    def unload_model(self) -> None:
        """Unload model from all engines."""
        for engine in self._engines:
            try:
                engine.unload_model()
            except Exception:
                pass
        self._model_name = None

    @property
    def model_name(self) -> Optional[str]:
        """Currently loaded model name."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """True if at least one engine has a model loaded."""
        return any(engine.is_loaded for engine in self._engines)

    @property
    def active_engine_name(self) -> str:
        """Name of the engine that last served a request (from stats)."""
        summary = self._stats.summary()
        if summary:
            by_count = sorted(
                summary.items(),
                key=lambda item: item[1].get("total_requests", 0),
                reverse=True,
            )
            if by_count:
                return by_count[0][0]
        return "none"
