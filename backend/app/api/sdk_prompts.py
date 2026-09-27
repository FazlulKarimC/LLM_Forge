import hashlib
from uuid import UUID
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.project_keys import get_sdk_project
from app.models.prompt import Prompt, PromptLabel
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import ReleaseLabel, version_response

router = APIRouter(tags=["SDK prompt reads"])


@router.get("/{name}")
async def fetch_prompt(name: str, label: ReleaseLabel | None = None, version: int | None = Query(None, ge=1),
                       if_none_match: str | None = Header(None), project_id: UUID = Depends(get_sdk_project),
                       db: AsyncSession = Depends(get_db)):
    if label is not None and version is not None:
        raise HTTPException(422, "Select a release label or an explicit version, not both")
    prompt = (await db.execute(select(Prompt).where(Prompt.name == name, Prompt.project_id == project_id,
        Prompt.archived.is_(False)))).scalar_one_or_none()
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    query = select(PromptVersion).where(PromptVersion.prompt_id == prompt.id)
    if version is not None:
        query = query.where(PromptVersion.version == version)
    else:
        query = query.join(PromptLabel, PromptLabel.version_id == PromptVersion.id).where(
            PromptLabel.prompt_id == prompt.id, PromptLabel.label == (label or "production"))
    snapshot = (await db.execute(query)).scalar_one_or_none()
    if snapshot is None:
        raise HTTPException(404, "Requested prompt version or release label not found. Promote a version first.")
    payload = version_response(snapshot).model_copy(update={"name": prompt.name})
    etag = '"' + hashlib.sha256(payload.model_dump_json().encode()).hexdigest() + '"'
    headers = {"ETag": etag, "Cache-Control": "private, no-store", "Vary": "Authorization"}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload.model_dump(mode="json"), headers=headers)
