"""
Result Pydantic Schemas

Schemas for experiment results and metrics.

TODO (Iteration 1): Add basic metrics
TODO (Iteration 2): Add percentile calculations
TODO (Iteration 3): Add statistical comparison
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class QualityMetrics(BaseModel):
    """Quality-related metrics."""
    accuracy_exact: Optional[float] = Field(None, ge=0, le=1)
    accuracy_f1: Optional[float] = Field(None, ge=0, le=1)
    accuracy_substring: Optional[float] = Field(None, ge=0, le=1)
    semantic_similarity: Optional[float] = Field(None, ge=0, le=1)
    faithfulness: Optional[float] = Field(None, ge=0, le=1)
    hallucination_rate: Optional[float] = Field(None, ge=0, le=1)
    robustness_safety_score: Optional[float] = Field(None, ge=0, le=1)
    robustness_inconclusive_rate: Optional[float] = Field(None, ge=0, le=1)


class PerformanceMetrics(BaseModel):
    """Performance-related metrics."""
    latency_p50: Optional[float] = Field(None, ge=0, description="Median latency in ms")
    latency_p95: Optional[float] = Field(None, ge=0, description="95th percentile")
    latency_p99: Optional[float] = Field(None, ge=0, description="99th percentile")
    throughput: Optional[float] = Field(None, ge=0, description="Queries per second")


class CostMetrics(BaseModel):
    """Cost proxy metrics."""
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_runs: int = 0
    gpu_time_seconds: Optional[float] = None
    total_cost_usd: Optional[float] = None
    cost_per_correct_answer: Optional[float] = None
    provider: Optional[str] = None
    cost_source: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        return self.total_tokens_input + self.total_tokens_output


class RunSummary(BaseModel):
    """Lightweight per-run data for grid view."""
    id: Optional[UUID] = None
    example_id: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    is_exact_match: Optional[bool] = None
    is_substring_match: Optional[bool] = None
    parsed_answer: Optional[str] = None
    semantic_similarity: Optional[float] = None
    latency_ms: Optional[float] = None
    prompt: str = ""
    raw_output: Optional[str] = None
    expected_output: Optional[str] = None
    faithfulness_score: Optional[float] = None
    context_relevance_score: Optional[float] = None
    attempt: Optional[int] = None
    agent_trace: Optional[Dict[str, Any]] = None
    failure_mode: Optional[str] = None
    error_message: Optional[str] = None
    grader_results: Optional[Dict[str, Any]] = None
    retrieved_chunks: Optional[Dict[str, Any]] = None
    served_provider: Optional[str] = None
    routing_reason: Optional[str] = None
    cost_usd: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class RunGridSummary(BaseModel):
    """Sparse per-run data for filmstrip/grid views.

    Omits heavy text fields (prompt, raw_output, expected_output) and
    large nested objects (agent_trace, retrieved_chunks) to reduce
    payload size on initial detail page load.
    """
    id: Optional[UUID] = None
    example_id: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    latency_ms: Optional[float] = None
    failure_mode: Optional[str] = None
    served_provider: Optional[str] = None
    routing_reason: Optional[str] = None
    cost_usd: Optional[float] = None
    grader_results: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class MetricsResponse(BaseModel):
    """Complete metrics for an experiment."""
    experiment_id: UUID
    summary_text: Optional[str] = None
    quality: QualityMetrics
    performance: PerformanceMetrics
    cost: CostMetrics
    failure_modes: Optional[Dict[str, Any]] = None
    computed_at: datetime


class ResultResponse(BaseModel):
    """Full result response with optional run details."""
    experiment_id: UUID
    metrics: MetricsResponse
    runs: Optional[List["RunResponse"]] = None  # Forward ref
    
    model_config = ConfigDict(from_attributes=True)


class ExperimentComparison(BaseModel):
    """Single experiment in a comparison."""
    experiment_id: UUID
    experiment_name: str
    method: str
    model: str
    metrics: MetricsResponse


class ComparisonResponse(BaseModel):
    """Side-by-side comparison of experiments."""
    experiments: List[ExperimentComparison]
    comparison_metrics: Dict[str, List[float]]  # metric_name -> [values per experiment]
    
    # TODO (Iteration 3): Add statistical significance
    # p_values: Optional[Dict[str, float]] = None


# Avoid circular import
from app.schemas.run import RunResponse
ResultResponse.model_rebuild()
