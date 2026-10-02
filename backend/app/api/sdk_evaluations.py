"""SDK evaluations require an explicitly scoped project key."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.project_keys import get_evaluation_project
from app.models.evaluation import Dataset, DatasetRevision
from app.schemas.evaluation import (
    EvaluationCreate,
    EvaluationSubmission,
    dataset_response,
    revision_response,
)
from app.services import evaluation_runs
from app.schemas.evaluator import ScoreOnlyCreate

router = APIRouter(tags=["SDK evaluations"])


@router.get("/evaluators")
async def evaluator_catalog(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    from app.services.evaluator_service import list_evaluators

    return await list_evaluators(db, offset=offset, limit=limit)


@router.get("/datasets/{name}")
async def fetch_dataset(
    name: str,
    version: int | None = Query(None, ge=1),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    dataset = await db.scalar(
        select(Dataset).where(Dataset.name == name, Dataset.archived.is_(False))
    )
    if dataset is None:
        raise HTTPException(404, "Dataset not found")
    revision = await db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.dataset_id == dataset.id,
            DatasetRevision.version == (version or dataset.latest_version),
        )
    )
    if revision is None:
        raise HTTPException(404, "Dataset revision not found")
    return {
        "dataset": dataset_response(dataset),
        "revision": revision_response(revision),
    }


@router.post("/runs", status_code=202)
async def start_run(
    data: EvaluationCreate,
    tasks: BackgroundTasks,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.dispatch_evaluation(data, tasks, project_id, db)


@router.get("/runs/{run_id}")
async def read_run(
    run_id: UUID,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.get_run(run_id, db)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: UUID,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.cancel_run(run_id, db)


@router.post("/runs/{run_id}/score", status_code=202)
async def score_outputs(
    run_id: UUID,
    data: ScoreOnlyCreate,
    tasks: BackgroundTasks,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.dispatch_scoring(run_id, data, tasks, project_id, db)


@router.post("/submissions", status_code=201)
async def submit_results(
    data: EvaluationSubmission,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.submit_results(data, db)
