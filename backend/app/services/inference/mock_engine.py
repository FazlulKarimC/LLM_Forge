"""
Mock Inference Engine

Fake inference for local development and UI testing.
Returns deterministic responses without any external dependencies.
"""

import time
import random
from typing import Optional, List

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)


class MockEngine(InferenceEngine):
    """
    Mock inference engine for development.
    
    Returns fake but realistic responses instantly.
    No network calls, no GPU, perfect for UI testing.
    """
    
    def __init__(self, simulate_latency_ms: int = 300):
        """
        Initialize mock engine.
        
        Args:
            simulate_latency_ms: Fake latency to simulate in milliseconds
        """
        self._model_name: Optional[str] = None
        self._is_loaded = False
        self._simulate_latency_ms = simulate_latency_ms
        
        # Predefined fake responses for variety
        self._fake_responses = [
            "Based on the information provided, the answer is",
            "After careful consideration,",
            "The most likely answer would be",
            "According to common knowledge,",
            "In this case, the response is",
        ]
    
    def load_model(self, model_name: str) -> None:
        """
        'Load' a model (instant, no real loading).
        
        Args:
            model_name: Model identifier (stored but not used)
        """
        self._model_name = model_name
        self._is_loaded = True
        print(f"✓ [MOCK] Loaded fake model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate fake text for a prompt.
        
        Args:
            prompt: Input text
            config: Generation parameters (mostly ignored)
        
        Returns:
            Fake GenerationResult
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        # Simulate network/processing time
        time.sleep(self._simulate_latency_ms / 1000)
        
        # Generate fake but contextual response
        prompt_preview = prompt[:50].replace("\n", " ")
        random_prefix = random.choice(self._fake_responses)
        
        fake_output = (
            f"{random_prefix} related to: '{prompt_preview}...'. "
            f"This is a mock response for testing purposes."
        )
        
        # Fake but realistic token counts
        tokens_input = len(prompt.split())
        tokens_output = len(fake_output.split())
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResult(
            text=fake_output,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            finish_reason="stop",
            gpu_memory_mb=None,
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[GenerationResult]:
        """
        Generate fake text for multiple prompts.
        """
        return [self.generate(p, config) for p in prompts]
    
    def unload_model(self) -> None:
        """
        'Unload' model (instant).
        """
        print(f"✓ [MOCK] Unloaded fake model: {self._model_name}")
        self._model_name = None
        self._is_loaded = False
    
    @property
    def model_name(self) -> Optional[str]:
        """Currently loaded fake model name."""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Whether fake model is loaded."""
        return self._is_loaded
