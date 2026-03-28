"""
Multi-Provider Router

Routes LLM inference requests across multiple providers with:
- Configurable routing policies (fallback_chain, cheapest_first, fastest_first, adaptive)
- Automatic fallback on rate limits (429) or connection errors
- Per-request provider tracking for cost attribution
- Thread-safe stats tracking for adaptive decisions
- Batch-path routing guard: per-prompt routing when policy != fallback_chain

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
from typing import List, Optional

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)
from app.services.inference.provider_stats import ProviderStatsTracker, RoutingPolicy
from app.models.run import FailureMode

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
    ):
        """
        Initialize the router.

        Args:
            engines: Ordered list of inference engines to try
            policy: Routing policy for provider selection
            stats_tracker: Optional pre-populated stats tracker
            epsilon: Exploration rate for ADAPTIVE policy (epsilon-greedy)
            exploration_window: Round-robin for first N requests before exploiting
        """
        if not engines:
            raise ValueError("At least one engine is required")

        self._engines = engines
        self._policy = policy
        self._model_name: Optional[str] = None
        self._stats = stats_tracker or ProviderStatsTracker()
        self._epsilon = epsilon
        self._exploration_window = exploration_window
        self._request_count = 0
        self._request_count_lock = threading.Lock()

        logger.info(
            "ProviderRouter initialized (policy=%s, engines=%s, epsilon=%.2f)",
            policy.value,
            [type(e).__name__ for e in engines],
            epsilon,
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
            except Exception as e:
                logger.warning(
                    "Failed to load model %s on %s: %s",
                    model_name, type(engine).__name__, e,
                )

    def _get_engine_name(self, engine: InferenceEngine) -> str:
        """Get a consistent name for an engine."""
        return type(engine).__name__

    def _get_available_engines(self) -> List[InferenceEngine]:
        """Get engines that have a model loaded."""
        return [e for e in self._engines if e.is_loaded]

    def _select_engine(self) -> InferenceEngine:
        """
        Select an engine based on current routing policy.
        
        Returns:
            Selected engine, falls back to first available on any issue.
        """
        available = self._get_available_engines()
        if not available:
            return self._engines[0]  # Will fail at generate(), but that's expected
        
        if self._policy == RoutingPolicy.FALLBACK_CHAIN:
            return available[0]
        
        available_names = [self._get_engine_name(e) for e in available]
        engine_map = {self._get_engine_name(e): e for e in available}
        
        if self._policy == RoutingPolicy.ADAPTIVE:
            return self._select_adaptive(available, engine_map, available_names)
        
        # CHEAPEST_FIRST or FASTEST_FIRST
        recommended = self._stats.recommend(self._policy, available_names)
        if recommended and recommended in engine_map:
            return engine_map[recommended]
        
        return available[0]

    def _select_adaptive(
        self,
        available: List[InferenceEngine],
        engine_map: dict,
        available_names: List[str],
    ) -> InferenceEngine:
        """Epsilon-greedy adaptive selection with exploration window."""
        with self._request_count_lock:
            count = self._request_count
        
        # Exploration phase: round-robin
        if count < self._exploration_window:
            idx = count % len(available)
            return available[idx]
        
        # Exploitation with epsilon-greedy
        if random.random() < self._epsilon:
            # Explore: random engine
            return random.choice(available)
        
        # Exploit: use cheapest (with tie-breaks)
        recommended = self._stats.recommend(RoutingPolicy.CHEAPEST_FIRST, available_names)
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
        
        # Try selected engine first, then fallback to others
        engines_to_try = [selected] + [e for e in self._engines if e is not selected and e.is_loaded]
        last_error = None

        for i, engine in enumerate(engines_to_try):
            engine_name = self._get_engine_name(engine)
            start_time = time.perf_counter()
            
            try:
                result = engine.generate(prompt, config)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Check if the result indicates a rate limit
                if result.failure_mode == FailureMode.API_ERROR and result.error_message:
                    err_lower = result.error_message.lower()
                    if "429" in err_lower or "rate limit" in err_lower:
                        self._stats.record(
                            engine_name, elapsed_ms,
                            result.tokens_input, result.tokens_output,
                            0.0, True,
                        )
                        logger.warning(
                            "Rate limited on %s, trying next provider...",
                            engine_name,
                        )
                        last_error = result.error_message
                        continue

                # Success — record stats
                self._stats.record(
                    engine_name, elapsed_ms,
                    result.tokens_input, result.tokens_output,
                    0.0,  # Cost computed per-run by pricing service
                    False,
                )
                
                # Tag result with routing info
                result.served_provider = engine_name
                reason = "selected" if engine is selected else f"fallback_{i}"
                if self._policy == RoutingPolicy.ADAPTIVE:
                    reason = f"adaptive_{reason}"
                result.routing_reason = reason
                
                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._stats.record(engine_name, elapsed_ms, 0, 0, 0.0, True)
                
                last_error = str(e)
                err_lower = str(e).lower()

                is_retryable = any(kw in err_lower for kw in [
                    "429", "rate limit", "too many requests",
                    "connection", "timeout", "unavailable",
                ])

                if is_retryable and i < len(engines_to_try) - 1:
                    logger.warning(
                        "Provider %s failed (%s), falling back...",
                        engine_name, e,
                    )
                    continue
                else:
                    raise

        # All providers failed
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
        
        - FALLBACK_CHAIN: delegate entire batch to first available engine
        - Other policies: per-prompt routing through self.generate()
        """
        if self._policy == RoutingPolicy.FALLBACK_CHAIN:
            # Existing behavior: delegate batch to first loaded engine
            for engine in self._engines:
                if not engine.is_loaded:
                    continue
                try:
                    results = engine.generate_batch(prompts, config, max_workers)
                    # Tag results with provider info
                    engine_name = self._get_engine_name(engine)
                    for r in results:
                        r.served_provider = engine_name
                        r.routing_reason = "batch_delegate"
                    return results
                except Exception as e:
                    logger.warning(
                        "Batch generation failed on %s: %s", self._get_engine_name(engine), e,
                    )
                    continue

            # Fallback: sequential via router
            logger.warning("All engine batch methods failed, falling back to sequential")
            return [self.generate(p, config) for p in prompts]
        else:
            # Per-prompt routing for adaptive/cheapest/fastest
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(lambda p: self.generate(p, config), prompts))

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
        return any(e.is_loaded for e in self._engines)

    @property
    def active_engine_name(self) -> str:
        """Name of the engine that last served a request (from stats)."""
        summary = self._stats.summary()
        if summary:
            # Return the provider with most requests
            by_count = sorted(summary.items(), key=lambda x: x[1].get("total_requests", 0), reverse=True)
            if by_count:
                return by_count[0][0]
        return "none"
