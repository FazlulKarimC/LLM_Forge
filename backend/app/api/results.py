# Results API Routes (updated)
"""
Results API Routes

Endpoints for experiment results and metrics:
- Get results for an experiment
- Get aggregated metrics
- Compare experiments
- Export results
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.result import Result
from app.models.run import Run
from app.models.experiment import Experiment
from app.schemas.result import (
    ResultResponse,
    MetricsResponse,
    ComparisonResponse,
    ExperimentComparison,
    QualityMetrics,
    PerformanceMetrics,
    CostMetrics,
    RunSummary,
)
from app.services.metrics_service import MetricsService

logger = logging.getLogger(__name__)

router = APIRouter()


def _result_to_metrics_response(result: Result) -> MetricsResponse:
    """Convert a Result model instance to MetricsResponse schema."""
    return MetricsResponse(
        experiment_id=result.experiment_id,
        quality=QualityMetrics(
            accuracy_exact=result.accuracy_exact,
            accuracy_f1=result.accuracy_f1,
            accuracy_substring=result.accuracy_substring,
            semantic_similarity=result.semantic_similarity,
            faithfulness=result.faithfulness,
            hallucination_rate=result.hallucination_rate,
        ),
        performance=PerformanceMetrics(
            latency_p50=result.latency_p50,
            latency_p95=result.latency_p95,
            latency_p99=result.latency_p99,
            throughput=result.throughput,
        ),
        cost=CostMetrics(
            total_tokens_input=result.total_tokens_input or 0,
            total_tokens_output=result.total_tokens_output or 0,
            total_runs=result.total_runs or 0,
            gpu_time_seconds=result.gpu_time_seconds,
        ),
        computed_at=result.computed_at,
    )



# =========================================================================
# COMPARE routes — must be declared BEFORE /{experiment_id} to avoid
# FastAPI matching "compare" as a UUID path parameter.
# =========================================================================

@router.get("/compare", response_model=ComparisonResponse)
async def compare_experiments(
    experiment_ids: List[str] = Query(..., description="List of experiment IDs to compare"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare metrics across multiple experiments.
    
    Returns side-by-side metrics for all specified experiments.
    """
    if len(experiment_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 experiments required for comparison"
        )
    
    # Parse string IDs to UUIDs
    try:
        parsed_ids = [UUID(eid) for eid in experiment_ids]
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    
    comparisons = []
    accuracy_values = []
    f1_values = []
    latency_p50_values = []
    
    for exp_id in parsed_ids:
        # Get experiment
        exp_query = select(Experiment).where(Experiment.id == exp_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {exp_id} not found"
            )
        
        # Get result
        res_query = select(Result).where(Result.experiment_id == exp_id)
        res_result = await db.execute(res_query)
        db_result = res_result.scalar_one_or_none()
        
        if not db_result:
            raise HTTPException(
                status_code=404,
                detail=f"No results for experiment {exp_id}"
            )
        
        metrics = _result_to_metrics_response(db_result)
        
        config = experiment.config or {}
        comparisons.append(ExperimentComparison(
            experiment_id=exp_id,
            experiment_name=experiment.name,
            method=experiment.method or config.get("reasoning_method", "unknown"),
            model=experiment.model_name or config.get("model_name", "unknown"),
            metrics=metrics,
        ))
        
        accuracy_values.append(db_result.accuracy_exact or 0.0)
        f1_values.append(db_result.accuracy_f1 or 0.0)
        latency_p50_values.append(db_result.latency_p50 or 0.0)
    
    return ComparisonResponse(
        experiments=comparisons,
        comparison_metrics={
            "accuracy_exact": accuracy_values,
            "accuracy_f1": f1_values,
            "latency_p50": latency_p50_values,
        },
    )


@router.get("/compare/statistical")
async def statistical_comparison(
    experiment_a: str = Query(..., description="First experiment ID"),
    experiment_b: str = Query(..., description="Second experiment ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Statistical comparison between two experiments.
    
    Computes:
    - McNemar's test for paired accuracy comparison
    - Bootstrap confidence intervals for both experiments
    - Per-example agreement/disagreement breakdown
    
    Returns statistical significance results.
    """
    from app.services.statistical_service import StatisticalService
    
    # Parse string IDs to UUIDs
    try:
        parsed_a = UUID(experiment_a)
        parsed_b = UUID(experiment_b)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    
    stat_service = StatisticalService(db)
    
    try:
        result = await stat_service.compare_experiments(parsed_a, parsed_b)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =========================================================================
# Per-experiment routes — /{experiment_id} parameterized routes below
# =========================================================================

@router.get("/{experiment_id}", response_model=ResultResponse)
async def get_results(
    experiment_id: UUID,
    include_runs: bool = Query(False, description="Include individual run logs"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get results for a specific experiment.
    
    Args:
        experiment_id: UUID of the experiment
        include_runs: Whether to include per-example run logs
    
    Returns:
        Aggregated results with optional run details
    """
    # Fetch result
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment {experiment_id}. "
                   "The experiment may not have been executed yet."
        )
    
    metrics = _result_to_metrics_response(db_result)
    
    # Optionally include runs
    runs = None
    if include_runs:
        run_query = select(Run).where(Run.experiment_id == experiment_id)
        run_result = await db.execute(run_query)
        run_rows = run_result.scalars().all()
        
        from app.schemas.run import RunResponse
        runs = [RunResponse.model_validate(r) for r in run_rows]
    
    return ResultResponse(
        experiment_id=experiment_id,
        metrics=metrics,
        runs=runs,
    )


@router.get("/{experiment_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    experiment_id: UUID,
    recompute: bool = Query(False, description="Force recompute metrics"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get computed metrics for an experiment.
    
    Metrics include:
    - Accuracy (exact match, substring, F1)
    - Latency (p50, p95, p99)
    - Cost proxies (tokens, GPU time)
    
    Set recompute=true to force recalculation from run data.
    """
    if recompute:
        # Force recompute from runs
        metrics_svc = MetricsService(db)
        try:
            db_result = await metrics_svc.compute_and_save(experiment_id)
            await db.commit()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return _result_to_metrics_response(db_result)
    
    # Fetch existing result
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for experiment {experiment_id}. "
                   "Run the experiment first, or use recompute=true."
        )
    
    return _result_to_metrics_response(db_result)


@router.get("/{experiment_id}/runs", response_model=List[RunSummary])
async def get_run_summaries(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get lightweight run summaries for the correctness grid view.
    
    Returns all runs with: id, example_id, is_correct, score, latency_ms,
    input_text, output_text, expected_output.
    """
    query = select(Run).where(Run.experiment_id == experiment_id)
    result = await db.execute(query)
    runs = result.scalars().all()
    
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No runs found for experiment {experiment_id}"
        )
    
    return [RunSummary.model_validate(r) for r in runs]


@router.get("/{experiment_id}/profile")
async def get_profile(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get optimization profiling data for an experiment (Phase 8).
    
    Returns timing breakdown, cache stats, and batch stats
    from Result.raw_metrics["optimization"].
    """
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment {experiment_id}"
        )
    
    raw = db_result.raw_metrics or {}
    optimization = raw.get("optimization", {})
    
    if not optimization:
        return JSONResponse(
            content={
                "experiment_id": str(experiment_id),
                "message": "No optimization data. Run with profiling enabled.",
                "profiling_summary": {},
                "cache_stats": {},
                "batch_stats": {},
                "total_wall_time_ms": None,
            }
        )
    
    return JSONResponse(
        content={
            "experiment_id": str(experiment_id),
            "profiling_summary": optimization.get("profiling_summary", {}),
            "cache_stats": optimization.get("cache_stats", {}),
            "batch_stats": optimization.get("batch_stats", {}),
            "total_wall_time_ms": optimization.get("total_wall_time_ms"),
        }
    )


@router.get("/{experiment_id}/export")
async def export_results(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Export full results as JSON download.
    
    Includes experiment info, metrics, and all runs.
    """
    # Get experiment
    exp_query = select(Experiment).where(Experiment.id == experiment_id)
    exp_result = await db.execute(exp_query)
    experiment = exp_result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get result
    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    db_result = res_result.scalar_one_or_none()
    
    # Get runs
    run_query = select(Run).where(Run.experiment_id == experiment_id)
    run_result = await db.execute(run_query)
    runs = run_result.scalars().all()
    
    export_data = {
        "experiment": {
            "id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status.value if experiment.status else None,
            "config": experiment.config,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        },
        "metrics": None,
        "runs": [
            {
                "id": str(run.id),
                "example_id": run.example_id,
                "input_text": run.input_text,
                "output_text": run.output_text,
                "expected_output": run.expected_output,
                "is_correct": run.is_correct,
                "score": run.score,
                "latency_ms": run.latency_ms,
                "tokens_input": run.tokens_input,
                "tokens_output": run.tokens_output,
            }
            for run in runs
        ],
        "total_runs": len(runs),
    }
    
    if db_result:
        export_data["metrics"] = {
            "accuracy_exact": db_result.accuracy_exact,
            "accuracy_f1": db_result.accuracy_f1,
            "accuracy_substring": db_result.accuracy_substring,
            "semantic_similarity": db_result.semantic_similarity,
            "faithfulness": db_result.faithfulness,
            "hallucination_rate": db_result.hallucination_rate,
            "latency_p50": db_result.latency_p50,
            "latency_p95": db_result.latency_p95,
            "latency_p99": db_result.latency_p99,
            "throughput": db_result.throughput,
            "total_tokens_input": db_result.total_tokens_input,
            "total_tokens_output": db_result.total_tokens_output,
            "total_runs": db_result.total_runs,
            "gpu_time_seconds": db_result.gpu_time_seconds,
            "computed_at": db_result.computed_at.isoformat() if db_result.computed_at else None,
        }
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{experiment.name}_results.json"'
        },
    )


@router.post("/{experiment_id}/judge")
async def run_llm_judge(
    experiment_id: UUID,
    sample_size: int = Query(20, ge=1, le=50, description="Number of runs to sample"),
    db: AsyncSession = Depends(get_db),
):
    """
    Run LLM-as-judge evaluation on a sampled subset of runs (P2 #13).
    
    Evaluates coherence, helpfulness, and factuality using a free HF model.
    Budget-capped to prevent runaway costs.
    """
    from app.services.llm_judge_service import LLMJudgeService
    
    judge = LLMJudgeService(db, sample_size=sample_size)
    
    try:
        result = await judge.evaluate_experiment(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judge evaluation failed: {str(e)[:200]}")
    
    # Save judge results to Result.raw_metrics
    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    result_obj = res_result.scalar_one_or_none()
    if result_obj:
        from sqlalchemy.orm.attributes import flag_modified
        raw = dict(result_obj.raw_metrics or {})
        raw["llm_judge"] = result
        result_obj.raw_metrics = raw
        flag_modified(result_obj, "raw_metrics")
        await db.flush()
        await db.commit()
    
    return JSONResponse(content=result)


@router.post("/synthetic/generate")
async def generate_synthetic_dataset(
    pairs_per_chunk: int = Query(3, ge=1, le=5, description="QA pairs per chunk"),
    max_chunks: int = Query(10, ge=1, le=20, description="Max chunks to process"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
):
    """
    Generate synthetic QA pairs from knowledge base chunks (P2 #14).
    
    Uses a free HF instruct model to create evaluation datasets.
    """
    from app.services.synthetic_data_service import SyntheticDatasetService
    from app.services.rag_service import RAGPipeline
    
    # Load knowledge base chunks
    try:
        rag = RAGPipeline()
        rag.load_knowledge_base()
        chunks = [chunk.text for chunk in rag.chunks[:max_chunks * 2]]  # Load extra for selection
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load knowledge base: {str(e)[:200]}"
        )
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No knowledge base chunks found")
    
    synth = SyntheticDatasetService()
    result = await synth.generate_from_chunks(
        chunks=chunks,
        pairs_per_chunk=pairs_per_chunk,
        max_chunks=max_chunks,
        seed=seed,
    )
    
    return JSONResponse(content=result)

