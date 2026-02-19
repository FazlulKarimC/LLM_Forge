"""
HuggingFace Inference API Engine

Remote inference engine using HuggingFace Inference API.
No local model loading, all inference happens via API calls.

Uses chat_completion API (standard for instruction-tuned models).
"""

import time
import os
from typing import Optional, List

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)


class HFAPIEngine(InferenceEngine):
    """
    HuggingFace Inference API engine.
    
    Uses HF Inference API (chat_completion) for text generation.
    No local GPU or model loading required.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize the HF API engine.
        
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
        
        self._model_name = model_name
        self._client = InferenceClient(
            model=self._model_name,
            token=self._api_key,
        )
        self._is_loaded = True  # API doesn't need explicit loading
    
    def load_model(self, model_name: str) -> None:
        """
        Set the model to use for inference.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self._model_name = model_name
        self._client = InferenceClient(
            model=self._model_name,
            token=self._api_key,
        )
        print(f"✓ Set model to: {model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
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
            # Use chat_completion API (standard for instruction-tuned models)
            response = self._client.chat_completion(
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
        
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
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
                results[idx] = future.result()
        
        return results
    
    def unload_model(self) -> None:
        """
        Unload model (no-op for API).
        """
        print(f"✓ Released API resources for {self._model_name}")
        self._is_loaded = False
    
    @property
    def model_name(self) -> Optional[str]:
        """Currently configured model name."""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Whether engine is ready to use."""
        return self._is_loaded
