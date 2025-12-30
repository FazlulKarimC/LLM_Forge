"""
Transformers Inference Engine

HuggingFace Transformers-based inference engine.
This is the primary engine for local development (GTX 1650).

Supported Models:
- microsoft/phi-2 (2.7B) - fits on GTX 1650
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 - fast iteration

TODO (Iteration 1): Implement basic generation
TODO (Iteration 2): Add batching and memory management
TODO (Iteration 3): Add quantization support (4-bit)
"""

import time
from typing import Optional, List

from app.services.inference.base import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)


class TransformersEngine(InferenceEngine):
    """
    HuggingFace Transformers inference engine.
    
    Uses AutoModelForCausalLM and AutoTokenizer.
    Supports GPU acceleration with automatic device mapping.
    """
    
    def __init__(self, device: str = "auto", dtype: str = "auto"):
        """
        Initialize the engine.
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16")
        """
        self._model = None
        self._tokenizer = None
        self._model_name: Optional[str] = None
        self._device = device
        self._dtype = dtype
    
    def load_model(self, model_name: str) -> None:
        """
        Load a model from HuggingFace.
        
        Args:
            model_name: HuggingFace model identifier
        
        Example:
            engine.load_model("microsoft/phi-2")
        
        TODO (Iteration 1): Implement with AutoModel
        TODO (Iteration 2): Add GPU memory tracking
        TODO (Iteration 3): Add quantization (bitsandbytes)
        """
        # TODO: Implement
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # 
        # self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map=self._device,
        #     torch_dtype=self._dtype,
        # )
        # self._model_name = model_name
        raise NotImplementedError("Iteration 1: Implement load_model")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate text from a single prompt.
        
        Args:
            prompt: Input text
            config: Generation parameters
        
        Returns:
            GenerationResult with text and metadata
        
        TODO (Iteration 1): Implement basic generation
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # TODO: Implement
        # start_time = time.perf_counter()
        # 
        # inputs = self._tokenizer(prompt, return_tensors="pt")
        # inputs = inputs.to(self._model.device)
        # 
        # outputs = self._model.generate(
        #     **inputs,
        #     max_new_tokens=config.max_tokens,
        #     temperature=config.temperature,
        #     top_p=config.top_p,
        #     do_sample=config.temperature > 0,
        # )
        # 
        # generated_text = self._tokenizer.decode(
        #     outputs[0][inputs.input_ids.shape[1]:],
        #     skip_special_tokens=True,
        # )
        # 
        # latency_ms = (time.perf_counter() - start_time) * 1000
        # 
        # return GenerationResult(
        #     text=generated_text,
        #     tokens_input=inputs.input_ids.shape[1],
        #     tokens_output=outputs.shape[1] - inputs.input_ids.shape[1],
        #     latency_ms=latency_ms,
        #     finish_reason="stop",
        # )
        raise NotImplementedError("Iteration 1: Implement generate")
    
    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts efficiently.
        
        Uses left-padding for batched generation.
        
        TODO (Iteration 2): Implement batching
        """
        # TODO: Implement with proper padding
        # For now, fall back to sequential
        return [self.generate(p, config) for p in prompts]
    
    def unload_model(self) -> None:
        """
        Unload model and free GPU memory.
        
        TODO (Iteration 2): Implement proper cleanup
        """
        # TODO: Implement
        # import torch
        # del self._model
        # del self._tokenizer
        # torch.cuda.empty_cache()
        # self._model = None
        # self._tokenizer = None
        # self._model_name = None
        raise NotImplementedError("Iteration 2: Implement unload_model")
    
    @property
    def model_name(self) -> Optional[str]:
        """Currently loaded model name."""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return self._model is not None
