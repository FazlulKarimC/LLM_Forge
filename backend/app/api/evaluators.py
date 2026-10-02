"""Authenticated project evaluator library; execution preview does not save scores."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.evaluator import Evaluator
from app.schemas.evaluator import (
    EvaluatorCreate,
    EvaluatorVersionCreate,
    EvaluatorUpdate,
    EvaluatorTest,
)
from app.schemas.evaluation import EvaluatorSelection
from app.services import evaluator_service as service
from app.services.prompt_service import flush_unique

router = APIRouter(tags=["Evaluators"])


@router.get("")
async def list_evaluators(
    search: str = Query("", max_length=120),
    archived: bool = False,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_evaluators(db, search, archived, offset, limit)


@router.post("", status_code=201)
async def create_evaluator(data: EvaluatorCreate, db: AsyncSession = Depends(get_db)):
    item = Evaluator(name=data.name, description=data.description)
    db.add(item)
    await flush_unique(
        db, "An evaluator with this name already exists, including archived evaluators"
    )
    await service.add_version(db, item, data)
    payload = await service.describe(db, item)
    await db.commit()
    return payload


@router.get("/{identifier}")
async def get_evaluator(identifier: UUID, db: AsyncSession = Depends(get_db)):
    return await service.describe(db, await service.find_evaluator(db, identifier))


@router.patch("/{identifier}")
async def update_evaluator(
    identifier: UUID, data: EvaluatorUpdate, db: AsyncSession = Depends(get_db)
):
    item = await service.find_evaluator(db, identifier, lock=True)
    item.archived = data.archived
    payload = await service.describe(db, item)
    await db.commit()
    return payload


@router.get("/{identifier}/versions")
async def versions(
    identifier: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    await service.find_evaluator(db, identifier)
    from app.models.evaluator import EvaluatorVersion

    return [
        service.version_response(item)
        for item in (
            await db.scalars(
                select(EvaluatorVersion)
                .where(EvaluatorVersion.evaluator_id == identifier)
                .order_by(EvaluatorVersion.version.desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()
    ]


@router.post("/{identifier}/versions", status_code=201)
async def save_version(
    identifier: UUID, data: EvaluatorVersionCreate, db: AsyncSession = Depends(get_db)
):
    payload = service.version_response(
        await service.add_version(
            db, await service.find_evaluator(db, identifier, lock=True), data
        )
    )
    await db.commit()
    return payload


@router.post("/versions/{version_id}/test")
async def test_evaluator(
    version_id: UUID, data: EvaluatorTest, db: AsyncSession = Depends(get_db)
):
    case = {"inputs": data.inputs, "expected_output": data.expected_output}
    snapshots, keys = await service.prepare_evaluators(
        db, [EvaluatorSelection(version_id=version_id, api_key=data.api_key)], [case]
    )
    await db.rollback()  # No database transaction spans a provider call.
    return {
        "checks": await service.execute_evaluator(
            snapshots[0], case, data.output, keys.get(str(version_id))
        ),
        "preview": True,
    }
