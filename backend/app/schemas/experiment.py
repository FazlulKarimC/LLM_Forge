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

from pydantic import BaseModel, Field, ConfigDict, field_validator

MAX_TEXT_FIELD_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 4000
MAX_TAG_LENGTH = 64
MAX_TAG_COUNT = 20
MAX_TOOL_NAME_LENGTH = 64
MAX_TOOL_COUNT = 10


class ExperimentStatus(str, Enum):
    """Experiment execution status."""
    PENDING = "pending"
    QUEUED = "queued"  # Set by API before enqueue (prevents race condition)
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


class InferenceProvider(str, Enum):
    """Supported inference providers."""
    AUTO = "auto"              # Router picks best available
    HF_API = "hf_api"          # HuggingFace Inference API
    OPENROUTER = "openrouter"  # OpenRouter (free models)
    GROQ = "groq"              # Groq (fast, rate-limited)
    CUSTOM = "custom"          # User-provided endpoint


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
        max_length=MAX_TEXT_FIELD_LENGTH,
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

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: List[str]) -> List[str]:
        """Bound the number and size of agent tool identifiers."""
        if len(value) > MAX_TOOL_COUNT:
            raise ValueError(f"At most {MAX_TOOL_COUNT} tools are allowed")

        cleaned_tools = []
        for tool in value:
            cleaned = tool.strip()
            if not cleaned:
                raise ValueError("Tool names must not be empty")
            if len(cleaned) > MAX_TOOL_NAME_LENGTH:
                raise ValueError(f"Tool names must be at most {MAX_TOOL_NAME_LENGTH} characters")
            cleaned_tools.append(cleaned)

        return cleaned_tools


class OptimizationConfig(BaseModel):
    """
    Inference optimization settings (Phase 8).
    
    Controls batching, caching, and profiling for experiment execution.
    """
    enable_batching: bool = Field(
        default=False,
        description="Batch prompts into concurrent API calls"
    )
    batch_size: int = Field(
        default=8, ge=1, le=32,
        description="Number of prompts per batch"
    )
    enable_caching: bool = Field(
        default=False,
        description="LRU cache for identical prompts"
    )
    cache_max_size: int = Field(
        default=256, ge=16, le=2048,
        description="Maximum cache entries"
    )
    enable_profiling: bool = Field(
        default=True,
        description="Time each execution phase"
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
        max_length=MAX_TEXT_FIELD_LENGTH,
        description="HuggingFace model identifier (e.g., microsoft/phi-2)"
    )
    reasoning_method: ReasoningMethod = Field(
        default=ReasoningMethod.NAIVE,
        description="Reasoning strategy to use"
    )
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_FIELD_LENGTH,
        description="Dataset identifier (e.g., trivia_qa, hotpot_qa)"
    )
    
    # Hyperparameters
    hyperparameters: HyperParameters = Field(default_factory=HyperParameters)
    
    # RAG settings (optional)
    rag: Optional[RAGConfig] = None
    
    # Agent settings (optional)
    agent: Optional[AgentConfig] = None
    
    # Optimization settings (Phase 8)
    optimization: Optional[OptimizationConfig] = None
    
    # Provider routing (Phase 6)
    provider: InferenceProvider = Field(
        default=InferenceProvider.AUTO,
        description="Inference provider (auto=router picks best available)"
    )
    
    # Dataset sampling
    num_samples: int = Field(
        default=100,
        ge=1,
        le=500,  # Matches frontend validation cap
        description="Number of dataset samples to evaluate"
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
    name: str = Field(..., min_length=1, max_length=MAX_TEXT_FIELD_LENGTH)
    description: Optional[str] = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)
    config: ExperimentConfig
    tags: Optional[List[str]] = Field(default=None, description="Free-form labels for organization")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        """Bound the number and size of free-form tags."""
        if value is None:
            return value
        if len(value) > MAX_TAG_COUNT:
            raise ValueError(f"At most {MAX_TAG_COUNT} tags are allowed")

        cleaned_tags = []
        for tag in value:
            cleaned = tag.strip()
            if not cleaned:
                raise ValueError("Tags must not be empty")
            if len(cleaned) > MAX_TAG_LENGTH:
                raise ValueError(f"Tags must be at most {MAX_TAG_LENGTH} characters")
            cleaned_tags.append(cleaned)

        return cleaned_tags


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
    tags: Optional[List[str]] = None
    run_manifest: Optional[Dict[str, Any]] = None
    
    # Pydantic v2 style — replaces deprecated inner `class Config`
    model_config = ConfigDict(from_attributes=True)


class ExperimentListResponse(BaseModel):
    """Response schema for experiment listing."""
    total: int
    experiments: List[ExperimentResponse]
    skip: int
    limit: int
