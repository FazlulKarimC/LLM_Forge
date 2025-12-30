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

from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    """Quality-related metrics."""
    accuracy_exact: Optional[float] = Field(None, ge=0, le=1)
    accuracy_f1: Optional[float] = Field(None, ge=0, le=1)
    faithfulness: Optional[float] = Field(None, ge=0, le=1)
    hallucination_rate: Optional[float] = Field(None, ge=0, le=1)


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
    
    @property
    def total_tokens(self) -> int:
        return self.total_tokens_input + self.total_tokens_output


class MetricsResponse(BaseModel):
    """Complete metrics for an experiment."""
    experiment_id: UUID
    quality: QualityMetrics
    performance: PerformanceMetrics
    cost: CostMetrics
    computed_at: datetime


class ResultResponse(BaseModel):
    """Full result response with optional run details."""
    experiment_id: UUID
    metrics: MetricsResponse
    runs: Optional[List["RunResponse"]] = None  # Forward ref
    
    class Config:
        from_attributes = True


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
