import logging
import time
from typing import Optional, List

from openai import OpenAI, APIConnectionError, APIError, NotFoundError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)

logger = logging.getLogger(__name__)


class OpenAIEngine(InferenceEngine):
    """
    OpenAI-Compatible API Engine for testing custom hosted models.
    
    Routes text generation requests to any OpenAI-compatible endpoint.
    This allows users to test locally hosted models (e.g. Ollama, vLLM)
    or third-party apis (e.g. Together AI, DeepSeek) securely without
    saving API keys in the backend database.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = "dummy_key", model_name: str = "custom-model"):
        """
        Initialize the OpenAI-compatible engine.

        Args:
            base_url: The URL of the custom API endpoint (e.g., http://localhost:11434/v1)
            api_key: Optional API key for the endpoint (default: "dummy_key" to prevent client errors for unauthenticated endpoints)
            model_name: The identifier of the model to be tested
        """
        if not base_url:
            raise ValueError("Base URL is required for the OpenAI-Compatible engine.")

        self._base_url = base_url
        self._api_key = api_key or "dummy_key"
        self._model_name = model_name
        self._client = self._make_client()
        self._is_loaded = True
        logger.info(f"OpenAIEngine initialized: base_url={self._base_url}, model={self._model_name}")

    def _make_client(self) -> OpenAI:
        """Create an OpenAI SDK client pointed at the custom Base URL."""
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=120.0, # Generous timeout for local or unoptimized models
        )
    
    def load_model(self, model_name: str) -> None:
        """
        Set the model to use for inference.
        """
        self._model_name = model_name
        self._client = self._make_client()
        logger.info(f"OpenAIEngine: switched to model {model_name} (base_url={self._base_url})")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type((ValueError, NotFoundError))  # Don't retry on validation errors or 404s
    )
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text from a prompt via OpenAI API format.
        """
        if not self.is_loaded:
            raise RuntimeError("Engine not initialized. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        try:
            api_kwargs = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": max(config.temperature, 0.01),  # Prevent absolute 0
                "top_p": config.top_p,
            }
            if hasattr(config, "seed") and config.seed is not None:
                api_kwargs["seed"] = config.seed

            response = self._client.chat.completions.create(**api_kwargs)

            generated_text = response.choices[0].message.content or ""
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # OpenAI completion tokens usage
            tokens_input = response.usage.prompt_tokens if response.usage else len(prompt.split())
            tokens_output = response.usage.completion_tokens if response.usage else len(generated_text.split())
            finish_reason = response.choices[0].finish_reason if response.choices else "stop"

            return GenerationResult(
                text=generated_text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency_ms=latency_ms,
                finish_reason=str(finish_reason),
                gpu_memory_mb=None,
            )

        except NotFoundError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Model not found on custom endpoint (url={self._base_url}, model={self._model_name}): {e}")
            raise RuntimeError(f"Model '{self._model_name}' not found on the provided endpoint. Please verify the model ID.") from e
        except RateLimitError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Rate limit exceeded (url={self._base_url}, model={self._model_name}): {e}")
            raise RuntimeError(f"Rate limit exceeded on custom endpoint: {str(e)}") from e
        except (APIConnectionError, APIError) as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Custom OpenAI API Error (url={self._base_url}, model={self._model_name}): {e}")
            raise RuntimeError(f"API Error from custom endpoint: {str(e)}") from e
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Unexpected custom API inference failure (url={self._base_url}, model={self._model_name}): {e}")
            raise RuntimeError(f"Unexpected inference failure: {str(e)}") from e
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        max_workers: int = 8,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts concurrently.
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
                    for f in future_to_idx:
                        f.cancel()
                    raise
        
        return results
    
    def unload_model(self) -> None:
        """
        Unload model (no-op for API).
        """
        logger.info(f"OpenAIEngine: released resources for {self._model_name}")
        self._is_loaded = False
    
    @property
    def model_name(self) -> Optional[str]:
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
