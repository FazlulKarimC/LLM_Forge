"""Clerk-authenticated evaluation routes."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.tenancy import ProjectContext, get_project_context
from app.schemas.evaluation import EvaluationCreate
from app.services import evaluation_runs

router = APIRouter(tags=["Evaluations"])


@router.post("", status_code=202)
async def create_evaluation(
    data: EvaluationCreate,
    tasks: BackgroundTasks,
    context: ProjectContext = Depends(get_project_context),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.dispatch_evaluation(
        data, tasks, context.project_id, db
    )


@router.get("")
async def list_evaluations(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.list_runs(db, offset, limit)


@router.get("/{run_id}")
async def get_evaluation(run_id: UUID, db: AsyncSession = Depends(get_db)):
    return await evaluation_runs.get_run(run_id, db)


@router.post("/{run_id}/cancel")
async def cancel_evaluation(run_id: UUID, db: AsyncSession = Depends(get_db)):
    return await evaluation_runs.cancel_run(run_id, db)
