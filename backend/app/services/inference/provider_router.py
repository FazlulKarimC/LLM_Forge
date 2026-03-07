"""
Multi-Provider Router

Routes LLM inference requests across multiple providers with:
- Configurable routing policies (cheapest_first, fastest_first, fallback_chain)
- Automatic fallback on rate limits (429) or connection errors
- Per-request provider tracking for cost attribution

Supported Providers:
- hf_api: HuggingFace Inference Providers API (default)
- openrouter: OpenRouter (free models available)
- groq: Groq (fast inference, strict rate limits)
- custom: User-provided OpenAI-compatible endpoint
"""

import logging
from enum import Enum
from typing import List, Optional

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)
from app.models.run import FailureMode

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    """How to select which provider to try first."""
    CHEAPEST_FIRST = "cheapest_first"
    FASTEST_FIRST = "fastest_first"
    FALLBACK_CHAIN = "fallback_chain"  # Use providers in order given


class ProviderRouter(InferenceEngine):
    """
    Routes inference across providers with automatic fallback.

    If the primary provider fails with a rate limit or connection error,
    the router transparently retries on the next provider in the chain.
    """

    def __init__(
        self,
        engines: List[InferenceEngine],
        policy: RoutingPolicy = RoutingPolicy.FALLBACK_CHAIN,
    ):
        """
        Initialize the router.

        Args:
            engines: Ordered list of inference engines to try
            policy: Routing policy (currently only fallback_chain is implemented)
        """
        if not engines:
            raise ValueError("At least one engine is required")

        self._engines = engines
        self._policy = policy
        self._model_name: Optional[str] = None
        self._active_engine: Optional[InferenceEngine] = None

        logger.info(
            "ProviderRouter initialized (policy=%s, engines=%s)",
            policy.value,
            [type(e).__name__ for e in engines],
        )

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

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text with automatic provider fallback.

        Tries each engine in order. On rate limit or connection error,
        falls back to the next engine transparently.
        """
        last_error = None

        for i, engine in enumerate(self._engines):
            if not engine.is_loaded:
                continue

            try:
                result = engine.generate(prompt, config)
                self._active_engine = engine

                # Check if the result indicates a rate limit
                if result.failure_mode == FailureMode.API_ERROR and result.error_message:
                    err_lower = result.error_message.lower()
                    if "429" in err_lower or "rate limit" in err_lower:
                        logger.warning(
                            "Rate limited on %s, trying next provider...",
                            type(engine).__name__,
                        )
                        last_error = result.error_message
                        continue

                return result

            except Exception as e:
                last_error = str(e)
                err_lower = str(e).lower()

                # Only retry on retryable errors
                is_retryable = any(kw in err_lower for kw in [
                    "429", "rate limit", "too many requests",
                    "connection", "timeout", "unavailable",
                ])

                if is_retryable and i < len(self._engines) - 1:
                    logger.warning(
                        "Provider %s failed (%s), falling back to %s",
                        type(engine).__name__,
                        e,
                        type(self._engines[i + 1]).__name__,
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
        Generate batch with provider fallback.

        Tries the first available engine's batch method.
        Falls back to sequential generation if batch fails.
        """
        for engine in self._engines:
            if not engine.is_loaded:
                continue
            try:
                return engine.generate_batch(prompts, config, max_workers)
            except Exception as e:
                logger.warning(
                    "Batch generation failed on %s: %s", type(engine).__name__, e,
                )
                continue

        # Fallback: sequential generation through the router
        logger.warning("All engine batch methods failed, falling back to sequential")
        return [self.generate(p, config) for p in prompts]

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
        """Name of the engine that served the last request."""
        if self._active_engine:
            return type(self._active_engine).__name__
        return "none"
