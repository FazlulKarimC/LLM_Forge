"""
Prompts API

CRUD endpoints for prompt version management.
Supports creating, listing, and retrieving versioned prompt templates.
"""

import hashlib
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.prompt_version import PromptVersion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["Prompts"])


# ─── Schemas ────────────────────────────────────────────────────────────────

class PromptVersionCreate(BaseModel):
    """Schema for creating a new prompt version."""
    name: str = Field(..., min_length=1, max_length=255, description="Prompt template name")
    template_text: str = Field(..., min_length=1, description="Prompt template content")
    parent_id: Optional[UUID] = Field(None, description="Parent version ID (for version history)")
    description: Optional[str] = Field(None, description="What changed in this version")


class PromptVersionResponse(BaseModel):
    """Schema for prompt version response."""
    id: UUID
    name: str
    template_text: str
    version: int
    sha256_hash: str
    parent_id: Optional[UUID]
    description: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


# ─── Endpoints ──────────────────────────────────────────────────────────────

@router.post("", response_model=PromptVersionResponse, status_code=201)
async def create_prompt_version(
    data: PromptVersionCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new prompt version.

    If parent_id is provided, the version number auto-increments.
    The SHA-256 hash is computed from the template text.
    """
    # Compute hash
    sha256_hash = PromptVersion.compute_hash(data.template_text)

    # Check for duplicate (same name + same hash = no actual change)
    existing = await db.execute(
        select(PromptVersion).where(
            PromptVersion.name == data.name,
            PromptVersion.sha256_hash == sha256_hash,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="Identical prompt version already exists (same name + content)"
        )

    # Determine version number
    version = 1
    if data.parent_id:
        parent = await db.execute(
            select(PromptVersion).where(PromptVersion.id == data.parent_id)
        )
        parent_obj = parent.scalar_one_or_none()
        if not parent_obj:
            raise HTTPException(status_code=404, detail="Parent version not found")
        version = parent_obj.version + 1
    else:
        # Find latest version for this name
        max_version_q = await db.execute(
            select(func.coalesce(func.max(PromptVersion.version), 0)).where(
                PromptVersion.name == data.name
            )
        )
        version = (max_version_q.scalar() or 0) + 1

    # Create version
    prompt_version = PromptVersion(
        name=data.name,
        template_text=data.template_text,
        version=version,
        sha256_hash=sha256_hash,
        parent_id=data.parent_id,
        description=data.description,
    )
    db.add(prompt_version)
    await db.commit()
    await db.refresh(prompt_version)

    logger.info("Created prompt version: %s v%d (hash=%s)", data.name, version, sha256_hash[:8])

    return PromptVersionResponse(
        id=prompt_version.id,
        name=prompt_version.name,
        template_text=prompt_version.template_text,
        version=prompt_version.version,
        sha256_hash=prompt_version.sha256_hash,
        parent_id=prompt_version.parent_id,
        description=prompt_version.description,
        created_at=prompt_version.created_at.isoformat(),
    )


@router.get("", response_model=List[PromptVersionResponse])
async def list_prompt_versions(
    name: Optional[str] = Query(None, description="Filter by prompt name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List prompt versions, optionally filtered by name."""
    query = select(PromptVersion).order_by(PromptVersion.created_at.desc())

    if name:
        query = query.where(PromptVersion.name == name)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    versions = result.scalars().all()

    return [
        PromptVersionResponse(
            id=v.id,
            name=v.name,
            template_text=v.template_text,
            version=v.version,
            sha256_hash=v.sha256_hash,
            parent_id=v.parent_id,
            description=v.description,
            created_at=v.created_at.isoformat(),
        )
        for v in versions
    ]


@router.get("/{prompt_id}", response_model=PromptVersionResponse)
async def get_prompt_version(
    prompt_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific prompt version by ID."""
    result = await db.execute(
        select(PromptVersion).where(PromptVersion.id == prompt_id)
    )
    version = result.scalar_one_or_none()

    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    return PromptVersionResponse(
        id=version.id,
        name=version.name,
        template_text=version.template_text,
        version=version.version,
        sha256_hash=version.sha256_hash,
        parent_id=version.parent_id,
        description=version.description,
        created_at=version.created_at.isoformat(),
    )


@router.get("/{prompt_id}/history", response_model=List[PromptVersionResponse])
async def get_prompt_history(
    prompt_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the version history chain for a prompt.

    Walks up the parent_id chain to show all ancestor versions.
    """
    history = []
    current_id = prompt_id

    # Walk up the chain (max 50 to prevent infinite loops)
    for _ in range(50):
        result = await db.execute(
            select(PromptVersion).where(PromptVersion.id == current_id)
        )
        version = result.scalar_one_or_none()

        if not version:
            break

        history.append(PromptVersionResponse(
            id=version.id,
            name=version.name,
            template_text=version.template_text,
            version=version.version,
            sha256_hash=version.sha256_hash,
            parent_id=version.parent_id,
            description=version.description,
            created_at=version.created_at.isoformat(),
        ))

        if not version.parent_id:
            break
        current_id = version.parent_id

    return history
