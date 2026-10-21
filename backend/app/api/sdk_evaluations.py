"""SDK evaluations require an explicitly scoped project key."""

from uuid import UUID
from app.services import evaluation_recovery
from app.schemas.evaluation_recovery import ResumeEvaluation

from fastapi import APIRouter, Request, Header, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.request_limits import LocalReportRoute
from app.core.project_keys import get_evaluation_project, get_sdk_key
from app.models.prompt import ProjectAPIKey
from app.models.evaluation import Dataset, DatasetRevision
from app.schemas.evaluation import (
    EvaluationCreate,
    EvaluationSubmission,
    LocalEvaluationSubmission,
    dataset_response,
    revision_response,
)
from app.services import evaluation_runs
from app.schemas.evaluator import ScoreOnlyCreate

from app.schemas.contracts import (
    CasePageResponse,
    CaseResponse,
    DatasetDetail,
    EvaluationDetailResponse,
    EvaluationRunResponse,
    EvaluationSummaryResponse,
    EvaluatorPage,
)

router = APIRouter(tags=["SDK evaluations"])
local_reports_router = APIRouter(route_class=LocalReportRoute)


@router.get("/evaluators", response_model=EvaluatorPage)
async def evaluator_catalog(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    from app.services.evaluator_service import list_evaluators

    return await list_evaluators(db, offset=offset, limit=limit)


@router.get("/datasets/{name}", response_model=DatasetDetail)
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


@router.post("/runs", status_code=202, response_model=EvaluationRunResponse)
async def start_run(
    data: EvaluationCreate,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
    key: ProjectAPIKey = Depends(get_sdk_key),
):
    return await evaluation_runs.dispatch_evaluation(
        data,
        None,
        project_id,
        db,
        (str(key.id), "create", idempotency_key)
        if idempotency_key is not None
        else None,
        request.state.request_id,
    )


@router.get("/runs/{run_id}", response_model=EvaluationDetailResponse)
async def read_run(
    run_id: UUID,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.get_run(run_id, db)


@router.post("/runs/{run_id}/cancel", response_model=EvaluationRunResponse)
async def cancel_run(
    run_id: UUID,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_runs.cancel_run(run_id, db)


@router.post(
    "/runs/{run_id}/score", status_code=202, response_model=EvaluationRunResponse
)
async def score_outputs(
    run_id: UUID,
    data: ScoreOnlyCreate,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
    key: ProjectAPIKey = Depends(get_sdk_key),
):
    return await evaluation_runs.dispatch_scoring(
        run_id,
        data,
        None,
        project_id,
        db,
        (str(key.id), f"score:{run_id}", idempotency_key)
        if idempotency_key is not None
        else None,
        request.state.request_id,
    )


@router.post("/submissions", status_code=201, response_model=EvaluationRunResponse)
async def submit_results(
    data: EvaluationSubmission,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
    key: ProjectAPIKey = Depends(get_sdk_key),
):
    return await evaluation_runs.submit_results(
        data,
        db,
        (str(key.id), "submit", idempotency_key)
        if idempotency_key is not None
        else None,
    )


@router.get("/runs/{run_id}/summary", response_model=EvaluationSummaryResponse)
async def run_summary(
    run_id: UUID,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_recovery.summary(db, run_id)


@router.get("/runs/{run_id}/cases", response_model=CasePageResponse)
async def run_cases(
    run_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    filter: str = Query(
        "all",
        pattern="^(all|passed|failed|errors|improved|regressed|unchanged|comparison_errors)$",
    ),
    reference_id: UUID | None = None,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_recovery.cases(
        db, run_id, offset, limit, filter, reference_id
    )


@router.get("/runs/{run_id}/cases/{case_index}", response_model=CaseResponse)
async def read_case(
    run_id: UUID,
    case_index: int,
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
):
    return await evaluation_recovery.case_detail(db, run_id, case_index)


@router.post(
    "/runs/{run_id}/resume", status_code=202, response_model=EvaluationRunResponse
)
async def resume_run(
    run_id: UUID,
    data: ResumeEvaluation,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    project_id: UUID = Depends(get_evaluation_project),
    key: ProjectAPIKey = Depends(get_sdk_key),
    db: AsyncSession = Depends(get_db),
):
    token = (
        (str(key.id), f"resume:{run_id}", idempotency_key)
        if idempotency_key is not None
        else None
    )
    return await evaluation_recovery.resume(
        run_id, data, project_id, db, token, request.state.request_id
    )


@local_reports_router.post(
    "/local-reports", status_code=201, response_model=EvaluationRunResponse,
    responses={413: {"description": "Original request body exceeds 64,000 bytes"}},
)
async def submit_local_report(
    data: LocalEvaluationSubmission,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    project_id: UUID = Depends(get_evaluation_project),
    db: AsyncSession = Depends(get_db),
    key: ProjectAPIKey = Depends(get_sdk_key),
):
    from app.services.local_evaluations import submit_local
    return await submit_local(data, db, (str(key.id), "local-report", idempotency_key) if idempotency_key is not None else None)


router.include_router(local_reports_router)
