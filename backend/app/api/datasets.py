from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.evaluation import Dataset, DatasetRevision
from app.schemas.evaluation import (
    DatasetCreate,
    DatasetImport,
    DatasetUpdate,
    RevisionCreate,
    dataset_response,
    revision_response,
)
from app.services.evaluation_service import import_cases

router = APIRouter(tags=["Datasets"])


async def find_dataset(db, dataset_id, lock=False):
    statement = select(Dataset).where(Dataset.id == dataset_id)
    item = await db.scalar(statement.with_for_update() if lock else statement)
    if item is None:
        raise HTTPException(404, "Dataset not found")
    return item


@router.post("/import")
async def parse_import(data: DatasetImport):
    return {"cases": import_cases(data)}


@router.get("")
async def list_datasets(
    archived: bool = False,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    total = await db.scalar(
        select(func.count()).select_from(Dataset).where(Dataset.archived == archived)
    )
    items = (
        await db.scalars(
            select(Dataset)
            .where(Dataset.archived == archived)
            .order_by(Dataset.created_at.desc(), Dataset.id)
            .offset(offset)
            .limit(limit)
        )
    ).all()
    return {"items": [dataset_response(item) for item in items], "total": total}


@router.post("", status_code=201)
async def create_dataset(data: DatasetCreate, db: AsyncSession = Depends(get_db)):
    item = Dataset(name=data.name, description=data.description, latest_version=1)
    db.add(item)
    try:
        await db.flush()
        revision = DatasetRevision(
            dataset_id=item.id,
            version=1,
            cases=[case.model_dump() for case in data.cases],
        )
        db.add(revision)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, "A dataset with this name already exists") from exc
    return {"dataset": dataset_response(item), "revision": revision_response(revision)}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: UUID, db: AsyncSession = Depends(get_db)):
    item = await find_dataset(db, dataset_id)
    revision = await db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.dataset_id == item.id,
            DatasetRevision.version == item.latest_version,
        )
    )
    return {"dataset": dataset_response(item), "revision": revision_response(revision)}


@router.patch("/{dataset_id}")
async def update_dataset(
    dataset_id: UUID, data: DatasetUpdate, db: AsyncSession = Depends(get_db)
):
    item = await find_dataset(db, dataset_id, lock=True)
    for key, value in data.model_dump(exclude_unset=True).items():
        if value is None:
            raise HTTPException(422, f"{key} cannot be null")
        setattr(item, key, value)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, "A dataset with this name already exists") from exc
    return dataset_response(item)


@router.delete("/{dataset_id}", status_code=204)
async def archive_dataset(dataset_id: UUID, db: AsyncSession = Depends(get_db)):
    item = await find_dataset(db, dataset_id, lock=True)
    item.archived = True
    await db.commit()
    return Response(status_code=204)


@router.get("/{dataset_id}/revisions")
async def revisions(
    dataset_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    await find_dataset(db, dataset_id)
    items = (
        await db.scalars(
            select(DatasetRevision)
            .where(DatasetRevision.dataset_id == dataset_id)
            .order_by(DatasetRevision.version.desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()
    return [revision_response(item) for item in items]


@router.post("/{dataset_id}/revisions", status_code=201)
async def add_revision(
    dataset_id: UUID, data: RevisionCreate, db: AsyncSession = Depends(get_db)
):
    item = await find_dataset(db, dataset_id, lock=True)
    if item.archived:
        raise HTTPException(409, "Restore this dataset before adding a revision")
    if data.base_version != item.latest_version:
        raise HTTPException(
            409, "Dataset changed; reload the latest revision before saving"
        )
    cases = [case.model_dump() for case in data.cases]
    latest = await db.scalar(
        select(DatasetRevision).where(
            DatasetRevision.dataset_id == item.id,
            DatasetRevision.version == item.latest_version,
        )
    )
    if latest.cases == cases:
        raise HTTPException(409, "Cases match the latest revision")
    item.latest_version += 1
    revision = DatasetRevision(
        dataset_id=item.id, version=item.latest_version, cases=cases
    )
    db.add(revision)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, "Dataset changed; reload before saving") from exc
    return revision_response(revision)
