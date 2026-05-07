"""Read, metric, run, profile, and export routes for results."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.results_common import latest_runs_for_experiment, result_to_metrics_response
from app.core.database import get_db
from app.models.experiment import Experiment
from app.models.result import Result
from app.schemas.result import MetricsResponse, ResultResponse, RunGridSummary, RunSummary
from app.services.metrics_service import MetricsService

router = APIRouter()


@router.get("/{experiment_id}", response_model=ResultResponse)
async def get_results(
    experiment_id: UUID,
    include_runs: bool = Query(False, description="Include individual run logs"),
    db: AsyncSession = Depends(get_db),
):
    """Get aggregated results for a specific experiment."""
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()

    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment {experiment_id}. "
            "The experiment may not have been executed yet.",
        )

    metrics = result_to_metrics_response(db_result)

    runs = None
    if include_runs:
        run_rows = await latest_runs_for_experiment(db, experiment_id)

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
    """Get computed metrics for an experiment."""
    if recompute:
        metrics_svc = MetricsService(db)
        try:
            db_result = await metrics_svc.compute_and_save(experiment_id)
            await db.commit()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return result_to_metrics_response(db_result)

    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()

    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for experiment {experiment_id}. "
            "Run the experiment first, or use recompute=true.",
        )

    return result_to_metrics_response(db_result)


@router.get("/{experiment_id}/runs")
async def get_run_summaries(
    experiment_id: UUID,
    sparse: bool = Query(False, description="Return sparse grid summaries (no prompt/output text)"),
    db: AsyncSession = Depends(get_db),
):
    """Get run summaries for the correctness grid view."""
    runs = await latest_runs_for_experiment(db, experiment_id)

    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No runs found for experiment {experiment_id}",
        )

    schema = RunGridSummary if sparse else RunSummary
    return [schema.model_validate(r) for r in runs]


@router.get("/{experiment_id}/profile")
async def get_profile(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get optimization profiling data for an experiment."""
    query = select(Result).where(Result.experiment_id == experiment_id)
    result = await db.execute(query)
    db_result = result.scalar_one_or_none()

    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment {experiment_id}",
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
    """Export full results as JSON download."""
    exp_query = select(Experiment).where(Experiment.id == experiment_id)
    exp_result = await db.execute(exp_query)
    experiment = exp_result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    db_result = res_result.scalar_one_or_none()

    runs = await latest_runs_for_experiment(db, experiment_id)

    export_data = {
        "experiment": {
            "id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status.value if experiment.status else None,
            "config": experiment.config,
            "run_manifest": experiment.run_manifest,
            "dataset_hash": experiment.dataset_hash,
            "sample_ids": experiment.sample_ids,
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
                "served_provider": run.served_provider,
                "routing_reason": run.routing_reason,
                "cost_usd": run.cost_usd,
                "failure_mode": run.failure_mode.value if run.failure_mode else None,
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
