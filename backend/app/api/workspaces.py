"""Workspace provisioning and organization/project management."""

import re
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import Identity, get_current_user
from app.core.database import get_db
from app.models.workspace import Organization, OrganizationMembership, Project, User

router = APIRouter(tags=["Workspaces"])


class NamedWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank")
        return value


def slug(name: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:100] or "workspace") + "-" + uuid4().hex[:8]


async def provision_user(db: AsyncSession, identity: Identity) -> User:
    user = (await db.execute(select(User).where(User.auth_subject == identity.subject))).scalar_one_or_none()
    if user:
        return user
    user_id = uuid5(NAMESPACE_URL, f"llmforge:user:{identity.subject}")
    org_id = uuid5(NAMESPACE_URL, f"llmforge:personal:{identity.subject}")
    project_id = uuid5(NAMESPACE_URL, f"llmforge:default:{identity.subject}")
    try:
        # Savepoint + uniqueness allow concurrent first requests to converge.
        async with db.begin_nested():
            user = User(id=user_id, auth_subject=identity.subject, display_name=identity.display_name)
            db.add(user)
            db.add(Organization(id=org_id, name="Personal workspace", slug=f"personal-{org_id.hex}"))
            await db.flush()
            db.add(OrganizationMembership(organization_id=org_id, user_id=user_id, role="owner"))
            db.add(Project(id=project_id, organization_id=org_id, name="Default project", slug="default"))
            await db.flush()
        await db.commit()
    except IntegrityError:
        await db.rollback()
        user = (await db.execute(select(User).where(User.auth_subject == identity.subject))).scalar_one_or_none()
        if user is None:
            raise
    return user


async def membership_for(db, organization_id, user_id):
    membership = (await db.execute(select(OrganizationMembership).where(
        OrganizationMembership.organization_id == organization_id,
        OrganizationMembership.user_id == user_id,
    ))).scalar_one_or_none()
    if not membership:
        raise HTTPException(404, "Organization not found")
    return membership


@router.get("")
async def list_workspaces(identity: Identity = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user = await provision_user(db, identity)
    rows = (await db.execute(select(Organization, OrganizationMembership)
        .join(OrganizationMembership, OrganizationMembership.organization_id == Organization.id)
        .where(OrganizationMembership.user_id == user.id).order_by(Organization.created_at, Organization.id))).all()
    projects = (await db.execute(select(Project).join(OrganizationMembership,
        OrganizationMembership.organization_id == Project.organization_id)
        .where(OrganizationMembership.user_id == user.id).order_by(Project.created_at, Project.id))).scalars().all()
    return {
        "user": {"id": user.id, "display_name": user.display_name},
        "organizations": [{"id": org.id, "name": org.name, "slug": org.slug, "role": membership.role,
            "projects": [{"id": p.id, "name": p.name, "slug": p.slug, "organization_id": p.organization_id}
                for p in projects if p.organization_id == org.id]} for org, membership in rows],
    }


@router.post("/organizations", status_code=201)
async def create_organization(data: NamedWorkspace, identity: Identity = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user = await provision_user(db, identity)
    org = Organization(name=data.name, slug=slug(data.name))
    db.add(org)
    await db.flush()
    db.add(OrganizationMembership(organization_id=org.id, user_id=user.id, role="owner"))
    project = Project(organization_id=org.id, name="Default project", slug="default")
    db.add(project)
    await db.flush()
    return {"id": org.id, "name": org.name, "project_id": project.id}


@router.post("/organizations/{organization_id}/projects", status_code=201)
async def create_project(organization_id: UUID, data: NamedWorkspace, identity: Identity = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user = await provision_user(db, identity)
    membership = await membership_for(db, organization_id, user.id)
    if membership.role != "owner":
        raise HTTPException(403, "Only workspace owners can create projects")
    project = Project(organization_id=organization_id, name=data.name, slug=slug(data.name))
    db.add(project)
    await db.flush()
    return {"id": project.id, "name": project.name, "slug": project.slug, "organization_id": organization_id}
