"""
Base Inference Engine Interface

Abstract base class for LLM inference engines.
All inference implementations must follow this interface.

Supported Engines:
- TransformersEngine: HuggingFace Transformers (local)
- VLLMEngine: vLLM for optimized inference (TODO)
- APIEngine: External API calls (TODO)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


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
    gpu_memory_mb: Optional[float] = None


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    
    Defines the interface that all inference implementations must follow.
    This allows swapping between local models, vLLM, and API-based inference.
    """
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """
        Load a model into memory.
        
        Args:
            model_name: HuggingFace model identifier or path
        
        Raises:
            ModelLoadError: If model cannot be loaded
        
        TODO (Iteration 1): Implement for Transformers
        TODO (Iteration 2): Add GPU memory management
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
        
        TODO (Iteration 1): Implement basic generation
        TODO (Iteration 2): Add batching support
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
        
        TODO (Iteration 2): Implement batching
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload model and free GPU memory.
        
        TODO (Iteration 2): Implement cleanup
        """
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
