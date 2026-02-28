"""
Run SQLAlchemy Model

Stores individual LLM inference calls.
Each experiment has many runs (one per dataset example).

TODO (Iteration 1): Add basic logging
TODO (Iteration 2): Add token counting
TODO (Iteration 3): Add trace data for agents
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import ForeignKey, String, Text, Float, Integer, Boolean, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Run(Base):
    """
    Individual LLM inference run.
    
    Captures everything about a single LLM call:
    - Input/output text
    - Token counts
    - Timing information
    - Correctness evaluation
    - Agent trace (if applicable)
    
    Attributes:
        id: Unique identifier
        experiment_id: Parent experiment
        example_id: ID from dataset (for reproducibility)
        
        Input/Output:
        - input_text: Full prompt sent to model
        - output_text: Generated response
        - expected_output: Ground truth answer
        
        Evaluation:
        - is_correct: Whether answer was correct
        - score: Soft score (0-1) for partial credit
        
        Performance:
        - tokens_input: Input token count
        - tokens_output: Output token count
        - latency_ms: Generation time in milliseconds
        - gpu_memory_mb: Peak GPU memory usage
        
        Agent-specific:
        - agent_trace: Full thought/action/observation trace
        - tool_calls: Number of tool invocations
    """
    
    __tablename__ = "runs"
    
    # Unique constraint for idempotency: prevents duplicate runs per example
    __table_args__ = (
        UniqueConstraint("experiment_id", "example_id", name="uq_runs_experiment_example"),
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Dataset reference
    example_id: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # ----- Input/Output -----
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    output_text: Mapped[str] = mapped_column(Text, nullable=True)
    expected_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # ----- Evaluation -----
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_exact_match: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_substring_match: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    parsed_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    match_alias: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    semantic_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ----- Performance -----
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    gpu_memory_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ----- Agent-specific -----
    agent_trace: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    tool_calls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # ----- RAG-specific -----
    retrieved_chunks: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ----- Attempt tracking (non-destructive re-runs) -----
    attempt: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    
    # ----- Timestamps -----
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    
    def __repr__(self) -> str:
        return f"<Run {self.id} correct={self.is_correct}>"
