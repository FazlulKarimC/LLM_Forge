"""Clerk-authenticated evaluation routes."""

from uuid import UUID
from app.services import evaluation_recovery
from app.schemas.evaluation_recovery import ResumeEvaluation

from fastapi import APIRouter, Request, Header, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.tenancy import ProjectContext, get_project_context
from app.schemas.evaluation import EvaluationCreate
from app.schemas.evaluator import ScoreOnlyCreate
from app.services import evaluation_runs

from app.schemas.contracts import (
    CasePageResponse,
    CaseResponse,
    EvaluationDetailResponse,
    EvaluationPage,
    EvaluationRunResponse,
    EvaluationSummaryResponse,
)

router = APIRouter(tags=["Evaluations"])


@router.post("", status_code=202, response_model=EvaluationRunResponse)
async def create_evaluation(
    data: EvaluationCreate,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    context: ProjectContext = Depends(get_project_context),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.dispatch_evaluation(
        data,
        None,
        context.project_id,
        db,
        (str(context.user_id), "create", idempotency_key)
        if idempotency_key is not None
        else None,
        request.state.request_id,
    )


@router.get("", response_model=EvaluationPage)
async def list_evaluations(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    dataset_revision_id: UUID | None = None,
    search: str = Query("", max_length=255),
    status: str | None = Query(
        None, pattern="^(queued|running|completed|failed|cancelled)$"
    ),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.list_runs(
        db, offset, limit, dataset_revision_id, search, status
    )


@router.get("/{run_id}", response_model=EvaluationDetailResponse)
async def get_evaluation(run_id: UUID, db: AsyncSession = Depends(get_db)):
    return await evaluation_runs.get_run(run_id, db)


@router.post("/{run_id}/cancel", response_model=EvaluationRunResponse)
async def cancel_evaluation(run_id: UUID, db: AsyncSession = Depends(get_db)):
    return await evaluation_runs.cancel_run(run_id, db)


@router.post("/{run_id}/score", status_code=202, response_model=EvaluationRunResponse)
async def score_outputs(
    run_id: UUID,
    data: ScoreOnlyCreate,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    context: ProjectContext = Depends(get_project_context),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.dispatch_scoring(
        run_id,
        data,
        None,
        context.project_id,
        db,
        (str(context.user_id), f"score:{run_id}", idempotency_key)
        if idempotency_key is not None
        else None,
        request.state.request_id,
    )


@router.get("/{run_id}/summary", response_model=EvaluationSummaryResponse)
async def run_summary(run_id: UUID, db: AsyncSession = Depends(get_db)):
    return await evaluation_recovery.summary(db, run_id)


@router.get("/{run_id}/cases", response_model=CasePageResponse)
async def run_cases(
    run_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    filter: str = Query(
        "all",
        pattern="^(all|passed|failed|errors|improved|regressed|unchanged|comparison_errors)$",
    ),
    reference_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_recovery.cases(
        db, run_id, offset, limit, filter, reference_id
    )


@router.get("/{run_id}/cases/{case_index}", response_model=CaseResponse)
async def read_case(run_id: UUID, case_index: int, db: AsyncSession = Depends(get_db)):
    return await evaluation_recovery.case_detail(db, run_id, case_index)


@router.post("/{run_id}/resume", status_code=202, response_model=EvaluationRunResponse)
async def resume_run(
    run_id: UUID,
    data: ResumeEvaluation,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    context: ProjectContext = Depends(get_project_context),
    db: AsyncSession = Depends(get_db),
):
    token = (
        (str(context.user_id), f"resume:{run_id}", idempotency_key)
        if idempotency_key is not None
        else None
    )
    return await evaluation_recovery.resume(
        run_id, data, context.project_id, db, token, request.state.request_id
    )
