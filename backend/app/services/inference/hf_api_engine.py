"""
HuggingFace Inference API Engine

Remote inference engine using HuggingFace Inference Providers API.
No local model loading, all inference happens via API calls.

Uses the new Inference Providers API (huggingface_hub >= 0.22).
The provider is configured via HF_PROVIDER env var (default: novita).

NOTE: The old /models/<name> serverless endpoint was deprecated in 2025
and now returns HTTP 410 Gone. The new provider-based API must be used.
"""

import logging
import time
import os
from typing import Optional, List

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)

logger = logging.getLogger(__name__)


# Providers confirmed working with novita as default.
# The old HF Serverless Inference endpoint (api-inference.huggingface.co/models/*)
# was deprecated mid-2025 and returns HTTP 410 Gone.
_DEFAULT_PROVIDER = "novita"

# Models confirmed working on novita provider (as of 2026-02)
NOVITA_SUPPORTED_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
}


class HFAPIEngine(InferenceEngine):
    """
    HuggingFace Inference Providers API engine.

    Uses the new HF Inference Providers API (provider= parameter) for
    text generation. No local GPU or model loading required.

    Configure provider via HF_PROVIDER env var (default: novita).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize the HF Inference Providers engine.

        Args:
            api_key: HuggingFace API token (from settings/.env if None)
            model_name: Model identifier on HuggingFace Hub
        """
        from app.core.config import settings
        self._api_key = api_key or settings.HF_TOKEN or os.getenv("HF_TOKEN")
        if not self._api_key:
            raise ValueError(
                "HuggingFace API token required. "
                "Set HF_TOKEN in .env file or pass api_key parameter."
            )

        # Provider: configurable via env var, default to novita (confirmed working)
        self._provider = os.getenv("HF_PROVIDER", _DEFAULT_PROVIDER)
        self._model_name = model_name
        self._client = self._make_client()
        self._is_loaded = True  # API doesn't need explicit loading
        logger.info(f"HFAPIEngine initialized: provider={self._provider}, model={self._model_name}")

    def _make_client(self) -> InferenceClient:
        """Create an InferenceClient with the configured provider."""
        return InferenceClient(provider=self._provider, api_key=self._api_key)
    
    def load_model(self, model_name: str) -> None:
        """
        Set the model to use for inference.

        Args:
            model_name: HuggingFace model identifier
        """
        self._model_name = model_name
        self._client = self._make_client()
        logger.info(f"HFAPIEngine: switched to model {model_name} (provider={self._provider})") 
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(ValueError)
    )
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text from a prompt via HF Inference API (chat_completion).
        
        Args:
            prompt: Input text
            config: Generation parameters
        
        Returns:
            GenerationResult with text and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Engine not initialized. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        try:
            # Use chat_completion with explicit model — provider routes it
            response = self._client.chat_completion(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=max(config.temperature, 0.01),  # API requires > 0
                top_p=config.top_p,
            )

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Measure latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Get token counts from response usage
            tokens_input = response.usage.prompt_tokens if response.usage else len(prompt.split())
            tokens_output = response.usage.completion_tokens if response.usage else len(generated_text.split())

            return GenerationResult(
                text=generated_text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason or "stop",
                gpu_memory_mb=None,  # N/A for API
            )

        except ValueError as e:
            # Re-raise ValueError immediately without retrying
            # so we bubble up "Model X is not supported" properly.
            logger.error(f"HF API rejected model {self._model_name} on provider {self._provider}: {e}")
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"HF API inference failed (provider={self._provider}, model={self._model_name}): {e}")
            raise RuntimeError(f"HF API inference failed: {str(e)}") from e
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        max_workers: int = 8,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts concurrently.
        
        Uses ThreadPoolExecutor to parallelize HTTP calls to HF API,
        reducing wall-clock time compared to sequential execution.
        
        Args:
            prompts: List of input texts
            config: Generation parameters (applied to all)
            max_workers: Max concurrent threads (default 8)
        
        Returns:
            List of GenerationResult in same order as prompts
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not prompts:
            return []
        
        workers = min(len(prompts), max_workers)
        results: List[GenerationResult] = [None] * len(prompts)  # type: ignore
        
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(self.generate, p, config): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except ValueError as e:
                    # Cancel pending futures and re-raise immediately
                    for f in future_to_idx:
                        f.cancel()
                    raise
        
        return results
    
    def unload_model(self) -> None:
        """
        Unload model (no-op for API).
        """
        logger.info(f"HFAPIEngine: released resources for {self._model_name}")
        self._is_loaded = False
    
    @property
    def model_name(self) -> Optional[str]:
        """Currently configured model name."""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Whether engine is ready to use."""
        return self._is_loaded
