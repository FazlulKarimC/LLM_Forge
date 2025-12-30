"""
Experiment Pydantic Schemas

Defines the structure for experiment configurations.
This is the heart of the config-driven approach.

TODO (Iteration 1): Add basic validation
TODO (Iteration 2): Add model-specific constraints
TODO (Iteration 3): Add config versioning
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ExperimentStatus(str, Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReasoningMethod(str, Enum):
    """Supported reasoning methods."""
    NAIVE = "naive"
    CHAIN_OF_THOUGHT = "cot"
    REACT = "react"


class RetrievalMethod(str, Enum):
    """Supported retrieval methods for RAG."""
    NONE = "none"
    NAIVE = "naive"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class HyperParameters(BaseModel):
    """
    Model hyperparameters for inference.
    
    These directly control the LLM generation behavior.
    """
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, higher=more random)"
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling (None=disabled)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class RAGConfig(BaseModel):
    """
    RAG-specific configuration.
    
    Only used when retrieval_method != NONE.
    """
    retrieval_method: RetrievalMethod = RetrievalMethod.NONE
    top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=256, ge=64, le=1024)
    rerank_model: Optional[str] = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )


class AgentConfig(BaseModel):
    """
    ReAct agent configuration.
    
    Only used when reasoning_method == REACT.
    """
    max_iterations: int = Field(default=5, ge=1, le=20)
    tools: List[str] = Field(
        default=["wikipedia_search", "calculator", "retrieval"],
        description="Enabled tool names"
    )


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.
    
    This schema captures everything needed to reproduce an experiment.
    Version this like code!
    """
    # Required fields
    model_name: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model identifier (e.g., microsoft/phi-2)"
    )
    reasoning_method: ReasoningMethod = Field(
        default=ReasoningMethod.NAIVE,
        description="Reasoning strategy to use"
    )
    dataset_name: str = Field(
        ...,
        min_length=1,
        description="Dataset identifier (e.g., trivia_qa, hotpot_qa)"
    )
    
    # Hyperparameters
    hyperparameters: HyperParameters = Field(default_factory=HyperParameters)
    
    # RAG settings (optional)
    rag: Optional[RAGConfig] = None
    
    # Agent settings (optional)
    agent: Optional[AgentConfig] = None
    
    # Dataset sampling
    num_samples: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of samples to evaluate"
    )
    
    @field_validator("agent")
    @classmethod
    def agent_requires_react(cls, v, info):
        """Ensure agent config is only set for ReAct method."""
        if v is not None and info.data.get("reasoning_method") != ReasoningMethod.REACT:
            raise ValueError("Agent config only valid for ReAct method")
        return v


class ExperimentCreate(BaseModel):
    """Request schema for creating an experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: ExperimentConfig


class ExperimentResponse(BaseModel):
    """Response schema for experiment details."""
    id: UUID
    name: str
    description: Optional[str]
    config: ExperimentConfig
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class ExperimentListResponse(BaseModel):
    """Response schema for experiment listing."""
    total: int
    experiments: List[ExperimentResponse]
    skip: int
    limit: int
