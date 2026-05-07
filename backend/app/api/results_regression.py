"""Regression report routes for experiment results."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.experiment import Experiment
from app.models.result import Result
from app.schemas.experiment import regression_status_from_verdict
from app.services.regression_service import RegressionService

router = APIRouter()


@router.get("/{experiment_id}/regression")
async def get_regression_report(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get regression report for an experiment."""
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
    """Force a fresh regression comparison."""
    from sqlalchemy.orm.attributes import flag_modified

    reg_svc = RegressionService(db)

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

    verdict = await reg_svc.run_regression_check(experiment_id, baseline.id)

    if result_obj:
        existing_raw = dict(result_obj.raw_metrics or {})
        existing_raw["regression"] = verdict.to_dict()
        result_obj.raw_metrics = existing_raw
        flag_modified(result_obj, "raw_metrics")

    experiment.regression_passed = verdict.passed
    experiment.regression_status = regression_status_from_verdict(verdict.passed).value

    await db.commit()

    return verdict.to_dict()
