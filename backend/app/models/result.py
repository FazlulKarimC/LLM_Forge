"""
Result SQLAlchemy Model

Stores aggregated metrics for completed experiments.

TODO (Iteration 1): Add accuracy metrics
TODO (Iteration 2): Add faithfulness and latency
TODO (Iteration 3): Add cost proxy metrics
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Result(Base):
    """
    Aggregated experiment results.
    
    Attributes:
        id: Unique identifier
        experiment_id: Reference to experiment
        
        Quality Metrics:
        - accuracy_exact: Exact string match accuracy
        - accuracy_f1: Token-level F1 score
        - faithfulness: NLI entailment score (for RAG)
        - hallucination_rate: % of unfaithful answers
        
        Performance Metrics:
        - latency_p50: Median latency in ms
        - latency_p95: 95th percentile latency
        - latency_p99: 99th percentile latency
        - throughput: Queries per second
        
        Cost Metrics:
        - total_tokens_input: Sum of input tokens
        - total_tokens_output: Sum of output tokens
        - total_runs: Number of individual runs
        - gpu_time_seconds: Approximate GPU time
        
        raw_metrics: Full metrics as JSON (for custom metrics)
    """
    
    __tablename__ = "results"
    
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
        unique=True,
    )
    
    # ----- Quality Metrics -----
    accuracy_exact: Mapped[float] = mapped_column(Float, nullable=True)
    accuracy_f1: Mapped[float] = mapped_column(Float, nullable=True)
    accuracy_substring: Mapped[float] = mapped_column(Float, nullable=True)
    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)
    hallucination_rate: Mapped[float] = mapped_column(Float, nullable=True)
    
    # ----- Performance Metrics -----
    latency_p50: Mapped[float] = mapped_column(Float, nullable=True)
    latency_p95: Mapped[float] = mapped_column(Float, nullable=True)
    latency_p99: Mapped[float] = mapped_column(Float, nullable=True)
    throughput: Mapped[float] = mapped_column(Float, nullable=True)
    
    # ----- Cost Metrics -----
    total_tokens_input: Mapped[int] = mapped_column(Integer, nullable=True)
    total_tokens_output: Mapped[int] = mapped_column(Integer, nullable=True)
    total_runs: Mapped[int] = mapped_column(Integer, nullable=True)
    gpu_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    
    # ----- Raw Metrics (JSON for flexibility) -----
    raw_metrics: Mapped[dict] = mapped_column(JSONB, nullable=True)
    
    # ----- Timestamps -----
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    
    def __repr__(self) -> str:
        return f"<Result exp={self.experiment_id} acc={self.accuracy_exact}>"
