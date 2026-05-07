"""
Base Inference Engine Interface

Abstract base class for LLM inference engines.
All inference implementations must follow this interface.

Supported Engines:
- HFAPIEngine: HuggingFace Inference Providers API
- OpenRouterEngine: OpenRouter (free models available)
- GroqEngine: Groq (fast inference)
- OpenAIEngine: Custom OpenAI-compatible endpoints
- MockEngine: Deterministic mock for testing
- ProviderRouter: Multi-provider routing with fallback
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from app.models.run import FailureMode


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Maps to model-agnostic generation parameters.
    """
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """
    Result of a single generation.
    
    Contains both the generated text and metadata for logging.
    """
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    finish_reason: str  # "stop", "length", "error"
    cost_usd: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    failure_mode: Optional[FailureMode] = None
    error_message: Optional[str] = None
    served_provider: Optional[str] = None
    routing_reason: Optional[str] = None


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    
    Defines the interface that all inference implementations must follow.
    This allows swapping between API-based providers and mock engines.
    """
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """
        Load a model into memory.
        
        Args:
            model_name: HuggingFace model identifier or path
        
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            config: Generation parameters
        
        Returns:
            Generated text with metadata
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        max_workers: int = 8,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts.
        
        More efficient than calling generate() multiple times.
        
        Args:
            prompts: List of input texts
            config: Generation parameters (applied to all)
        
        Returns:
            List of generation results
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model and free resources."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Currently loaded model name."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        pass

    @property
    def provider_id(self) -> str:
        """Stable provider identifier used in routing telemetry."""
        return type(self).__name__
