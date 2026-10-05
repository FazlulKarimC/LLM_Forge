"""Authenticated project evaluator library; execution preview does not save scores."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.evaluator import (
    EvaluatorCreate,
    EvaluatorVersionCreate,
    EvaluatorUpdate,
    EvaluatorTest,
)
from app.services import evaluator_service as service

from app.schemas.contracts import (
    EvaluatorPage,
    EvaluatorPreview,
    EvaluatorResponse,
    EvaluatorVersionResponse,
)

router = APIRouter(tags=["Evaluators"])


@router.get("", response_model=EvaluatorPage)
async def list_evaluators(
    search: str = Query("", max_length=120),
    archived: bool = False,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_evaluators(db, search, archived, offset, limit)


@router.post("", status_code=201, response_model=EvaluatorResponse)
async def create_evaluator(data: EvaluatorCreate, db: AsyncSession = Depends(get_db)):
    return await service.create_evaluator(db, data)


@router.get("/{identifier}", response_model=EvaluatorResponse)
async def get_evaluator(identifier: UUID, db: AsyncSession = Depends(get_db)):
    return await service.get_evaluator(db, identifier)


@router.patch("/{identifier}", response_model=EvaluatorResponse)
async def update_evaluator(
    identifier: UUID, data: EvaluatorUpdate, db: AsyncSession = Depends(get_db)
):
    return await service.update_evaluator(db, identifier, data)


@router.get("/{identifier}/versions", response_model=list[EvaluatorVersionResponse])
async def versions(
    identifier: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await service.versions(db, identifier, offset, limit)


@router.post(
    "/{identifier}/versions", status_code=201, response_model=EvaluatorVersionResponse
)
async def save_version(
    identifier: UUID, data: EvaluatorVersionCreate, db: AsyncSession = Depends(get_db)
):
    return await service.save_version(db, identifier, data)


@router.post("/versions/{version_id}/test", response_model=EvaluatorPreview)
async def test_evaluator(
    version_id: UUID, data: EvaluatorTest, db: AsyncSession = Depends(get_db)
):
    return await service.test_evaluator(db, version_id, data)
