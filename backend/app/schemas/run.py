"""
Run Pydantic Schemas

Schemas for individual LLM inference runs.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from app.models.run import FailureMode


class AgentStep(BaseModel):
    """Single step in agent trace."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class AgentTrace(BaseModel):
    """Full agent execution trace."""
    steps: List[AgentStep]
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    success: Optional[bool] = None
    termination_reason: Optional[str] = None
    total_iterations: Optional[int] = None


class RetrievalInfo(BaseModel):
    """Information about retrieved chunks."""
    method: str  # naive, hybrid, reranked
    chunks: List[Dict[str, Any]]  # title, content, score
    retrieval_time_ms: float


class RunResponse(BaseModel):
    """Response schema for a single run."""
    id: UUID
    experiment_id: UUID
    example_id: Optional[str]
    
    # Input/Output
    prompt: str
    raw_output: Optional[str]
    expected_output: Optional[str]
    
    # Evaluation
    is_correct: Optional[bool]
    score: Optional[float] = Field(None, ge=0, le=1)
    is_exact_match: Optional[bool] = None
    is_substring_match: Optional[bool] = None
    parsed_answer: Optional[str] = None
    match_alias: Optional[str] = None
    semantic_similarity: Optional[float] = Field(None, ge=0, le=1)
    
    # Performance
    tokens_input: Optional[int]
    tokens_output: Optional[int]
    latency_ms: Optional[float]
    
    # Failure Tracking
    failure_mode: Optional[FailureMode] = None
    error_message: Optional[str] = None
    
    # Agent-specific
    agent_trace: Optional[AgentTrace] = None
    grader_results: Optional[Dict[str, Any]] = None

    # RAG-specific
    retrieved_chunks: Optional[Dict[str, Any]] = None
    retrieval_info: Optional[RetrievalInfo] = None
    faithfulness_score: Optional[float] = Field(None, ge=0, le=1)
    context_relevance_score: Optional[float] = Field(None, ge=0, le=1)

    # Routing-specific
    served_provider: Optional[str] = None
    routing_reason: Optional[str] = None
    cost_usd: Optional[float] = Field(None, ge=0)
    
    # Audit metadata
    run_metadata: Optional[Dict[str, Any]] = None

    # Attempt tracking
    attempt: Optional[int] = None
    
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
