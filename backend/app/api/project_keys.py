import secrets
from datetime import datetime, timezone
from uuid import UUID
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import Field, field_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.project_keys import hash_key
from app.core.tenancy import get_project_context, ProjectContext
from app.models.prompt import ProjectAPIKey
from app.models.workspace import Project
from app.schemas.prompt import StrictModel

router = APIRouter(tags=["Project API keys"])


class KeyCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    scopes: list[Literal["prompts:read", "prompts:write", "evaluations:write"]] = Field(default_factory=lambda: ["prompts:read"], min_length=1, max_length=3)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip():
            raise ValueError("Key name cannot be blank")
        return value.strip()


def key_response(key):
    return {"id": key.id, "name": key.name, "prefix": key.prefix, "scope": "prompts:read",
        "scopes": key.scopes,
        "created_at": key.created_at, "revoked_at": key.revoked_at}


def require_owner(context):
    if context.role != "owner":
        raise HTTPException(403, "Only workspace owners can manage project API keys")


@router.get("")
async def list_keys(context: ProjectContext = Depends(get_project_context), db: AsyncSession = Depends(get_db)):
    require_owner(context)
    keys = (await db.execute(select(ProjectAPIKey).order_by(ProjectAPIKey.created_at.desc()))).scalars().all()
    return [key_response(key) for key in keys]


@router.post("", status_code=201)
async def create_key(data: KeyCreate, context: ProjectContext = Depends(get_project_context), db: AsyncSession = Depends(get_db)):
    require_owner(context)
    await db.execute(select(Project).where(Project.id == context.project_id).with_for_update())
    count = (await db.execute(select(func.count()).select_from(ProjectAPIKey).where(ProjectAPIKey.revoked_at.is_(None)))).scalar_one()
    if count >= 20:
        raise HTTPException(409, "This project already has 20 active keys. Revoke one before creating another.")
    secret = "lf_live_" + secrets.token_urlsafe(32)
    scopes = sorted(set(["prompts:read", *data.scopes]))
    key = ProjectAPIKey(project_id=context.project_id, name=data.name, prefix=secret[:16], secret_hash=hash_key(secret), scopes=scopes)
    db.add(key)
    await db.flush()
    payload = {**key_response(key), "secret": secret}
    await db.commit()
    return payload


@router.delete("/{key_id}", status_code=204)
async def revoke_key(key_id: UUID, context: ProjectContext = Depends(get_project_context), db: AsyncSession = Depends(get_db)):
    require_owner(context)
    key = (await db.execute(select(ProjectAPIKey).where(ProjectAPIKey.id == key_id))).scalar_one_or_none()
    if key is None:
        raise HTTPException(404, "Project API key not found")
    if key.revoked_at is None:
        key.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    return Response(status_code=204)
