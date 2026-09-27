"""Legacy benchmark prompt snapshots, backed by the workspace prompt library."""
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.tenancy import get_project_context, ProjectContext
from app.models.prompt import Prompt
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import PromptCreate, VersionCreate, VersionResponse, version_response
from app.services.prompt_service import add_version, find_prompt, flush_unique

router = APIRouter(tags=["Prompts"])


class PromptVersionCreate(PromptCreate):
    template_format: str = Field(default="fstring", pattern="^(mustache|fstring)$")
    parent_id: Optional[UUID] = None


@router.post("", response_model=VersionResponse, status_code=201)
async def create_prompt_version(data: PromptVersionCreate, context: ProjectContext = Depends(get_project_context), db: AsyncSession = Depends(get_db)):
    base = data.base_version
    if data.parent_id:
        parent = (await db.execute(select(PromptVersion).where(PromptVersion.id == data.parent_id))).scalar_one_or_none()
        if parent is None:
            raise HTTPException(404, "Parent version not found")
        prompt = await find_prompt(db, parent.prompt_id, lock=True)
        if prompt.name != data.name:
            raise HTTPException(422, "Parent version must belong to the same named prompt")
        base = parent.version
    else:
        prompt = (await db.execute(select(Prompt).where(Prompt.name == data.name).with_for_update())).scalar_one_or_none()
        if prompt is None:
            prompt = Prompt(project_id=context.project_id, name=data.name, description=data.description)
            db.add(prompt)
            await flush_unique(db, "This prompt was created concurrently. Reload and retry.")
        elif prompt.archived:
            raise HTTPException(409, "Prompt is archived. Restore it in the prompt library.")
    version = await add_version(db, prompt, VersionCreate(template_text=data.template_text,
        template_format=data.template_format, description=data.description, base_version=base))
    await db.commit()
    return version_response(version)


@router.get("", response_model=list[VersionResponse])
async def list_prompt_versions(name: Optional[str] = None, skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    query = select(PromptVersion).order_by(PromptVersion.created_at.desc())
    if name:
        query = query.where(PromptVersion.name == name)
    versions = (await db.execute(query.offset(skip).limit(limit))).scalars().all()
    return [version_response(version) for version in versions]


@router.get("/{prompt_id}", response_model=VersionResponse)
async def get_prompt_version(prompt_id: UUID, db: AsyncSession = Depends(get_db)):
    version = (await db.execute(select(PromptVersion).where(PromptVersion.id == prompt_id))).scalar_one_or_none()
    if version is None:
        raise HTTPException(404, "Prompt version not found")
    return version_response(version)


@router.get("/{prompt_id}/history", response_model=list[VersionResponse])
async def get_prompt_history(prompt_id: UUID, db: AsyncSession = Depends(get_db)):
    history = []
    current = prompt_id
    for _ in range(100):
        version = (await db.execute(select(PromptVersion).where(PromptVersion.id == current))).scalar_one_or_none()
        if version is None:
            if not history:
                raise HTTPException(404, "Prompt version not found")
            break
        history.append(version_response(version))
        if not version.parent_id:
            break
        current = version.parent_id
    return history
