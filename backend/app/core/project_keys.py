"""Explicit SDK capabilities, isolated from Clerk dashboard authentication."""
import hashlib
import hmac
from uuid import UUID
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.prompt import ProjectAPIKey

sdk_bearer = HTTPBearer(auto_error=False)


def hash_key(secret: str) -> str:
    # A 256-bit random secret does not need a slow password hash or a salt.
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


async def get_sdk_key(credentials: HTTPAuthorizationCredentials | None = Depends(sdk_bearer), db: AsyncSession = Depends(get_db)) -> ProjectAPIKey:
    denied = HTTPException(401, "Invalid or revoked project API key", headers={"WWW-Authenticate": "Bearer"})
    if credentials is None or not credentials.credentials.startswith("lf_live_") or len(credentials.credentials) > 128:
        raise denied
    digest = hash_key(credentials.credentials)
    key = (await db.execute(select(ProjectAPIKey).where(ProjectAPIKey.secret_hash == digest,
        ProjectAPIKey.revoked_at.is_(None)))).scalar_one_or_none()
    if key is None or not hmac.compare_digest(key.secret_hash, digest):
        raise denied
    db.info["project_id"] = key.project_id
    return key


async def get_sdk_project(key: ProjectAPIKey = Depends(get_sdk_key)) -> UUID:
    if "prompts:read" not in key.scopes:
        raise HTTPException(403, "This key does not allow prompt reads")
    return key.project_id


async def get_prompt_write_project(key: ProjectAPIKey = Depends(get_sdk_key)) -> UUID:
    if "prompts:write" not in key.scopes:
        raise HTTPException(403, "This key does not allow prompt writes")
    return key.project_id


async def get_evaluation_project(key: ProjectAPIKey = Depends(get_sdk_key)) -> UUID:
    if "evaluations:write" not in key.scopes:
        raise HTTPException(403, "Create a project API key with evaluations:write in Settings to use this endpoint")
    return key.project_id
