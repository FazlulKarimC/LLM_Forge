"""
Run Pydantic Schemas

Schemas for individual LLM inference runs.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


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
    input_text: str
    output_text: Optional[str]
    expected_output: Optional[str]
    
    # Evaluation
    is_correct: Optional[bool]
    score: Optional[float] = Field(None, ge=0, le=1)
    
    # Performance
    tokens_input: Optional[int]
    tokens_output: Optional[int]
    latency_ms: Optional[float]
    
    # Agent-specific
    agent_trace: Optional[AgentTrace] = None
    
    # RAG-specific
    retrieval_info: Optional[RetrievalInfo] = None
    faithfulness_score: Optional[float] = Field(None, ge=0, le=1)
    
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
