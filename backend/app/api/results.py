# Results API Routes (updated)
"""
Results API Routes

Endpoints for experiment results and metrics:
- Get results for an experiment
- Get aggregated metrics
- Compare experiments
- Export results
"""

import json
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.background_jobs import (
    create_job,
    get_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
)
from app.core.database import async_session_maker, get_db
from app.models.result import Result
from app.models.run import Run
from app.models.experiment import Experiment
from app.schemas.experiment import regression_status_from_verdict
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
    raw_metrics = result.raw_metrics or {}
    cost_metrics = raw_metrics.get("cost", {})
    return MetricsResponse(
        experiment_id=result.experiment_id,
        summary_text=raw_metrics.get("summary_text"),
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
            total_cost_usd=cost_metrics.get("total_cost_usd"),
            cost_per_correct_answer=cost_metrics.get("cost_per_correct_answer"),
            provider=cost_metrics.get("provider"),
        ),
        failure_modes=raw_metrics.get("failure_modes"),
        computed_at=result.computed_at,
    )


async def _latest_runs_for_experiment(db: AsyncSession, experiment_id: UUID) -> List[Run]:
    """Return only the latest-attempt runs for an experiment."""
    attempt_query = (
        select(Run.attempt)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.attempt.desc())
        .limit(1)
    )
    attempt_result = await db.execute(attempt_query)
    latest_attempt = attempt_result.scalar_one_or_none()
    if latest_attempt is None:
        return []

    runs_query = select(Run).where(
        Run.experiment_id == experiment_id,
        Run.attempt == latest_attempt,
    )
    runs_result = await db.execute(runs_query)
    return runs_result.scalars().all()


async def _save_llm_judge_result(db: AsyncSession, experiment_id: UUID, result: dict) -> None:
    """Persist LLM judge output into Result.raw_metrics when aggregate results exist."""
    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    result_obj = res_result.scalar_one_or_none()
    if result_obj is None:
        return

    from sqlalchemy.orm.attributes import flag_modified

    raw = dict(result_obj.raw_metrics or {})
    raw["llm_judge"] = result
    result_obj.raw_metrics = raw
    flag_modified(result_obj, "raw_metrics")
    await db.flush()
    await db.commit()


def _load_knowledge_base_chunks(max_chunks: int) -> List[str]:
    """Load a bounded subset of knowledge-base chunks for synthetic generation."""
    from app.core.config import settings
    from app.services.rag_service import ChunkingService

    try:
        articles_path = settings.data_dir / "knowledge_base" / "articles.json"
        with articles_path.open("r", encoding="utf-8") as f:
            articles = json.load(f)
        chunk_objects = ChunkingService.chunk_articles(articles)
        chunks = [chunk.text for chunk in chunk_objects[: max_chunks * 2]]
    except Exception as exc:  # pragma: no cover - surfaced through API response
        raise RuntimeError(f"Failed to load knowledge base: {str(exc)[:200]}") from exc

    if not chunks:
        raise LookupError("No knowledge base chunks found")

    return chunks


async def _run_llm_judge_job(job_id: str, experiment_id: UUID, sample_size: int) -> None:
    """Execute judge evaluation in the background and persist the result."""
    from app.services.llm_judge_service import LLMJudgeService

    mark_job_running(job_id)

    try:
        async with async_session_maker() as session:
            judge = LLMJudgeService(session, sample_size=sample_size)
            result = await judge.evaluate_experiment(experiment_id)
            await _save_llm_judge_result(session, experiment_id, result)
        mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.exception("Judge background job failed for %s", experiment_id)
        mark_job_failed(job_id, f"Judge evaluation failed: {str(exc)[:200]}")


async def _run_synthetic_generation_job(
    job_id: str,
    pairs_per_chunk: int,
    max_chunks: int,
    seed: Optional[int],
) -> None:
    """Execute synthetic dataset generation in the background."""
    from app.services.synthetic_data_service import SyntheticDatasetService

    mark_job_running(job_id)

    try:
        chunks = _load_knowledge_base_chunks(max_chunks)
        synth = SyntheticDatasetService()
        result = await synth.generate_from_chunks(
            chunks=chunks,
            pairs_per_chunk=pairs_per_chunk,
            max_chunks=max_chunks,
            seed=seed,
        )
        mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.exception("Synthetic generation background job failed")
        mark_job_failed(job_id, str(exc)[:200])



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


@router.get("/jobs/{job_id}")
async def get_background_job(job_id: str):
    """Return the current status for an asynchronous background job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or has expired")
    return JSONResponse(content=job)


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
        run_rows = await _latest_runs_for_experiment(db, experiment_id)

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
    runs = await _latest_runs_for_experiment(db, experiment_id)
    
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
    
    # Get runs from the latest attempt only
    runs = await _latest_runs_for_experiment(db, experiment_id)
    
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
                "prompt": run.prompt,
                "raw_output": run.raw_output,
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
    background_tasks: BackgroundTasks,
    sample_size: int = Query(20, ge=1, le=50, description="Number of runs to sample"),
    db: AsyncSession = Depends(get_db),
):
    """
    Queue LLM-as-judge evaluation and return a pollable job id immediately.
    """
    experiment_query = select(Experiment).where(Experiment.id == experiment_id)
    experiment_result = await db.execute(experiment_query)
    if experiment_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    job = create_job(
        "llm_judge",
        {
            "experiment_id": str(experiment_id),
            "sample_size": sample_size,
        },
    )
    background_tasks.add_task(_run_llm_judge_job, job["job_id"], experiment_id, sample_size)
    return JSONResponse(status_code=202, content=job)


@router.post("/synthetic/generate")
async def generate_synthetic_dataset(
    background_tasks: BackgroundTasks,
    pairs_per_chunk: int = Query(3, ge=1, le=5, description="QA pairs per chunk"),
    max_chunks: int = Query(10, ge=1, le=20, description="Max chunks to process"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
):
    """
    Queue synthetic dataset generation and return a pollable job id immediately.
    """
    job = create_job(
        "synthetic_generation",
        {
            "pairs_per_chunk": pairs_per_chunk,
            "max_chunks": max_chunks,
            "seed": seed,
        },
    )
    background_tasks.add_task(
        _run_synthetic_generation_job,
        job["job_id"],
        pairs_per_chunk,
        max_chunks,
        seed,
    )
    return JSONResponse(status_code=202, content=job)


# =============================================================================
# REGRESSION ENDPOINTS
# =============================================================================

@router.get("/{experiment_id}/regression")
async def get_regression_report(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get regression report for an experiment.
    
    Returns stored regression verdict from raw_metrics['regression'],
    or 404 if no regression data exists.
    """
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(status_code=404, detail="No results found for experiment")
    
    raw = db_result.raw_metrics or {}
    regression = raw.get("regression")
    
    if regression is None:
        raise HTTPException(status_code=404, detail="No regression data for this experiment")
    
    return regression


@router.post("/{experiment_id}/regression/rerun")
async def rerun_regression(
    experiment_id: UUID,
    baseline_id: Optional[UUID] = Query(None, description="Explicit baseline to compare against"),
    db: AsyncSession = Depends(get_db),
):
    """
    Force a fresh regression comparison.
    
    Uses explicit baseline_id if provided, otherwise auto-detects.
    """
    from app.services.regression_service import RegressionService
    from app.models.experiment import Experiment
    from sqlalchemy.orm.attributes import flag_modified
    
    reg_svc = RegressionService(db)
    
    # Load candidate experiment
    exp_query = select(Experiment).where(
        Experiment.id == experiment_id,
        Experiment.deleted_at.is_(None),
    )
    exp_result = await db.execute(exp_query)
    experiment = exp_result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "completed":
        raise HTTPException(status_code=400, detail="Only completed experiments can be regression-checked")

    # Clear any previous regression badge/report before re-running.
    experiment.regression_status = "not_checked"
    experiment.regression_passed = None
    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    result_obj = res_result.scalar_one_or_none()
    if result_obj:
        existing_raw = dict(result_obj.raw_metrics or {})
        existing_raw.pop("regression", None)
        result_obj.raw_metrics = existing_raw
        flag_modified(result_obj, "raw_metrics")

    # Find baseline
    if baseline_id:
        bl_query = select(Experiment).where(
            Experiment.id == baseline_id,
            Experiment.deleted_at.is_(None),
        )
        bl_result = await db.execute(bl_query)
        baseline = bl_result.scalar_one_or_none()
    else:
        baseline = await reg_svc.find_baseline(experiment)
    
    if not baseline:
        await db.commit()
        raise HTTPException(status_code=404, detail="No baseline found for comparison")
    
    if baseline.id == experiment_id:
        raise HTTPException(status_code=400, detail="Cannot compare experiment against itself")
    
    # Run comparison
    verdict = await reg_svc.run_regression_check(experiment_id, baseline.id)
    
    # Persist verdict
    if result_obj:
        existing_raw = dict(result_obj.raw_metrics or {})
        existing_raw["regression"] = verdict.to_dict()
        result_obj.raw_metrics = existing_raw
        flag_modified(result_obj, "raw_metrics")
    
    experiment.regression_passed = verdict.passed
    experiment.regression_status = regression_status_from_verdict(verdict.passed).value

    await db.commit()
    
    return verdict.to_dict()


# =============================================================================
# ROUTING TELEMETRY ENDPOINTS
# =============================================================================

@router.get("/{experiment_id}/routing")
async def get_routing_telemetry(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get routing telemetry for an experiment.
    
    Returns provider stats from raw_metrics['routing'].
    """
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(status_code=404, detail="No results found for experiment")
    
    raw = db_result.raw_metrics or {}
    routing = raw.get("routing")
    
    if routing is None:
        raise HTTPException(status_code=404, detail="No routing telemetry for this experiment")
    
    return routing
