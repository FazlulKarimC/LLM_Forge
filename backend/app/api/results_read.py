"""Read, metric, run, profile, and export routes for results."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.results_common import latest_runs_for_experiment, result_to_metrics_response
from app.core.database import get_db
from app.models.experiment import Experiment
from app.models.result import Result
from app.models.run import Run
from app.models.experiment_attempt import ExperimentAttempt
from app.services.attempts import current_number, result_at, lock_history_mutation
from app.schemas.result import BenchmarkAttemptPage, RunPage, MetricsResponse, ResultResponse, RunGridSummary, RunSummary
from app.services.metrics_service import MetricsService

router = APIRouter()


@router.get("/{experiment_id}/attempts", response_model=BenchmarkAttemptPage)
async def get_attempts(experiment_id: UUID, skip: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=50),
                       db: AsyncSession = Depends(get_db)):
    number = await current_number(db, experiment_id)
    rows = (await db.scalars(select(ExperimentAttempt).where(ExperimentAttempt.experiment_id == experiment_id)
                            .order_by(ExperimentAttempt.number.desc()).offset(skip).limit(limit))).all()
    total = await db.scalar(select(func.count()).select_from(ExperimentAttempt).where(ExperimentAttempt.experiment_id == experiment_id))
    return {"current_attempt": number, "total": total, "skip": skip, "limit": limit,
            "attempts": [{"number": r.number, "status": r.status, "provenance": r.provenance_status,
                          "generation_calls": r.generation_calls,
                          "dataset_hash": r.dataset_hash, "selected": len(r.sample_ids) if r.sample_ids is not None else None,
                          "created_at": r.created_at.isoformat(), "run_manifest": r.run_manifest} for r in rows]}


@router.get("/{experiment_id}/runs/page", response_model=RunPage)
async def get_run_page(experiment_id: UUID, attempt: int | None = Query(None, ge=0),
                       skip: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=50), db: AsyncSession = Depends(get_db)):
    number = await current_number(db, experiment_id, attempt)
    rows = await get_run_summaries(experiment_id, True, skip, limit, number, db)
    total = await db.scalar(select(func.count()).select_from(Run).where(Run.experiment_id == experiment_id, Run.attempt == number))
    return {"attempt": number, "total": total, "skip": skip, "limit": limit, "runs": rows}


@router.get("/{experiment_id}/runs/{run_id}", response_model=RunSummary)
async def get_run_detail(experiment_id: UUID, run_id: UUID, attempt: int | None = Query(None, ge=1), db: AsyncSession = Depends(get_db)):
    number = await current_number(db, experiment_id, attempt)
    run = await db.scalar(select(Run).where(Run.id == run_id, Run.experiment_id == experiment_id, Run.attempt == number))
    if run is None:
        raise HTTPException(404, "Case not found in selected attempt")
    return RunSummary.model_validate(run)


@router.delete("/{experiment_id}/attempts/{attempt}")
async def purge_attempt(experiment_id: UUID, attempt: int, exported: bool = False, db: AsyncSession = Depends(get_db)):
    await current_number(db, experiment_id, attempt)
    await lock_history_mutation(db)
    experiment = await db.scalar(select(Experiment).where(Experiment.id == experiment_id).with_for_update())
    if not exported:
        raise HTTPException(409, "Export this attempt before confirming purge with exported=true")
    if attempt == experiment.pinned_attempt or (attempt == experiment.current_attempt and experiment.status in ("queued", "running")):
        raise HTTPException(409, "Active and pinned attempts cannot be purged")
    from app.models.benchmark_reservation import BenchmarkReservation
    from app.services.benchmark_execution import utcnow
    active = await db.scalar(select(BenchmarkReservation.id).where(
        BenchmarkReservation.resource_id == str(experiment_id), BenchmarkReservation.attempt == attempt,
        BenchmarkReservation.released_at.is_(None), BenchmarkReservation.expires_at > utcnow(),
    ).limit(1))
    if active:
        raise HTTPException(409, "An issued call still owns this attempt; wait for it to finish")
    referenced = await db.scalar(select(Experiment.id).where(Experiment.baseline_id == experiment_id).limit(1))
    report_reference = await db.scalar(select(Result.id).where(
        Result.raw_metrics["regression"]["baseline_id"].as_string() == str(experiment_id),
        Result.raw_metrics["regression"]["baseline_attempt"].as_integer() == attempt,
    ).limit(1))
    if referenced or report_reference:
        raise HTTPException(409, "An experiment or regression report references this baseline")
    if await db.get(ExperimentAttempt, (experiment_id, attempt)) is None:
        raise HTTPException(404, "Attempt not found")
    await db.execute(delete(Run).where(Run.experiment_id == experiment_id, Run.attempt == attempt))
    await db.execute(delete(Result).where(Result.experiment_id == experiment_id, Result.attempt == attempt))
    await db.execute(delete(ExperimentAttempt).where(ExperimentAttempt.experiment_id == experiment_id, ExperimentAttempt.number == attempt))
    if attempt == experiment.current_attempt:
        from app.schemas.experiment import ExperimentStatus
        experiment.status = ExperimentStatus.FAILED
        experiment.error_message = "Results were manually purged after export. Rerun to create a new attempt."
        experiment.run_manifest, experiment.dataset_hash, experiment.sample_ids = None, None, None
        experiment.regression_status, experiment.regression_passed = "not_checked", None
    await db.commit()
    return {"purged_attempt": attempt}


@router.get("/{experiment_id}", response_model=ResultResponse)
async def get_results(
    experiment_id: UUID,
    include_runs: bool = Query(False, description="Include individual run logs"),
    attempt: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get aggregated results for a specific experiment."""
    db_result = await result_at(db, experiment_id, attempt)

    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment {experiment_id}. "
            "The experiment may not have been executed yet.",
        )

    metrics = result_to_metrics_response(db_result)

    runs = None
    if include_runs:
        run_rows = await latest_runs_for_experiment(db, experiment_id, db_result.attempt)

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
    attempt: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get computed metrics for an experiment."""
    if recompute:
        number = await current_number(db, experiment_id, attempt, legacy_metrics=True)
        if number == 0:
            raise HTTPException(409, "Select a positive case attempt explicitly; an unbound historical aggregate cannot be recomputed")
        metrics_svc = MetricsService(db)
        try:
            db_result = await metrics_svc.compute_and_save(experiment_id, attempt=attempt)
            await db.commit()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return result_to_metrics_response(db_result)

    db_result = await result_at(db, experiment_id, attempt)

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
    skip: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=50),
    attempt: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get run summaries for the correctness grid view."""
    number = await current_number(db, experiment_id, attempt)
    schema = RunGridSummary if sparse else RunSummary
    query = select(*[getattr(Run, name) for name in schema.model_fields]) if sparse else select(Run)
    query = query.where(Run.experiment_id == experiment_id, Run.attempt == number).order_by(Run.example_id, Run.id).offset(skip).limit(limit)
    result = await db.execute(query)
    if sparse:
        return [schema.model_validate(dict(row)) for row in result.mappings()]
    return [schema.model_validate(row) for row in result.scalars()]



@router.get("/{experiment_id}/profile")
async def get_profile(
    experiment_id: UUID,
    attempt: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get optimization profiling data for an experiment."""
    db_result = await result_at(db, experiment_id, attempt)

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
    format: str = Query("json", pattern="^(json|jsonl)$"),
    attempt: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Export full results as JSON download."""
    exp_query = select(Experiment).where(Experiment.id == experiment_id)
    exp_result = await db.execute(exp_query)
    experiment = exp_result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    from app.api.results_export import export_attempt
    return await export_attempt(db, experiment, attempt, format)
