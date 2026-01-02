"""
HuggingFace Inference API Engine

Remote inference engine using HuggingFace Inference API.
No local model loading, all inference happens via API calls.
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
    
    Uses HF Inference API for text generation.
    No local GPU or model loading required.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "microsoft/phi-2"):
        """
        Initialize the HF API engine.
        
        Args:
            api_key: HuggingFace API token (from environment if None)
            model_name: Model identifier on HuggingFace Hub
        """
        self._api_key = api_key or os.getenv("HF_TOKEN")
        if not self._api_key:
            raise ValueError(
                "HuggingFace API token required. "
                "Set HF_TOKEN environment variable or pass api_key parameter."
            )
        
        self._client = InferenceClient(token=self._api_key)
        self._model_name = model_name
        self._is_loaded = True  # API doesn't need explicit loading
    
    def load_model(self, model_name: str) -> None:
        """
        Set the model to use for inference.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self._model_name = model_name
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
        Generate text from a prompt via HF Inference API.
        
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
            # Call HuggingFace Inference API
            response = self._client.text_generation(
                prompt,
                model=self._model_name,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
                return_full_text=False,  # Only return generated text
            )
            
            # Measure latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Estimate token counts (API doesn't always return exact counts)
            # For Phase 2, we'll use rough estimates
            tokens_input = len(prompt.split())  # Rough estimate
            tokens_output = len(response.split())  # Rough estimate
            
            return GenerationResult(
                text=response,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency_ms=latency_ms,
                finish_reason="stop",
                gpu_memory_mb=None,  # N/A for API
            )
        
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise RuntimeError(f"HF API inference failed: {str(e)}") from e
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts.
        
        For now, sequential. Batching optimization in Phase 8.
        """
        return [self.generate(p, config) for p in prompts]
    
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
