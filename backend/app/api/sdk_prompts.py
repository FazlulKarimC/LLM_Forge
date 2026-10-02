import hashlib
from uuid import UUID
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.project_keys import get_sdk_project, get_prompt_write_project
from app.models.prompt import Prompt, PromptLabel
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import ReleaseLabel, PromptCreate, StrictModel, VersionResponse, version_response
from pydantic import Field
from app.services.prompt_service import add_version, assign_label, flush_unique, list_prompt_catalog

router = APIRouter(tags=["SDK prompts"])


class SDKSnapshot(VersionResponse):
    labels: list[str]
    tags: list[str]


async def snapshot_response(db, prompt, version):
    labels = list((await db.scalars(select(PromptLabel.label).where(PromptLabel.prompt_id == prompt.id, PromptLabel.version_id == version.id))).all())
    if version.version == prompt.latest_version:
        labels.append("latest")
    return SDKSnapshot(**{**version_response(version).model_dump(), "name": prompt.name}, labels=sorted(labels), tags=prompt.tags)


@router.post("", status_code=201)
async def create_prompt(data: PromptCreate, project_id: UUID = Depends(get_prompt_write_project), db: AsyncSession = Depends(get_db)):
    prompt = await db.scalar(select(Prompt).where(Prompt.name == data.name, Prompt.project_id == project_id).with_for_update())
    if prompt is None:
        prompt = Prompt(project_id=project_id, name=data.name, description=data.description, prompt_type=data.prompt_type, tags=data.tags)
        db.add(prompt)
        await flush_unique(db, "This prompt was created concurrently. Fetch it and retry with base_version.")
    elif prompt.archived:
        raise HTTPException(409, "Restore the prompt before creating another version")
    elif "tags" in data.model_fields_set:
        prompt.tags = data.tags
    version = await add_version(db, prompt, data, created_by="API")
    result = await snapshot_response(db, prompt, version)
    await db.commit()
    return result


class LabelAssignment(StrictModel):
    version: int = Field(ge=1)


@router.put("/{name:path}/labels/{label}")
async def update_label(name: str, label: ReleaseLabel, data: LabelAssignment, project_id: UUID = Depends(get_prompt_write_project), db: AsyncSession = Depends(get_db)):
    prompt = await db.scalar(select(Prompt).where(Prompt.name == name, Prompt.project_id == project_id, Prompt.archived.is_(False)).with_for_update())
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    version = await db.scalar(select(PromptVersion).where(PromptVersion.prompt_id == prompt.id, PromptVersion.version == data.version))
    if version is None:
        raise HTTPException(404, "Version not found in this prompt")
    await assign_label(db, prompt, label, version.id)
    result = await snapshot_response(db, prompt, version)
    await db.commit()
    return result


@router.get("")
async def list_sdk_prompts(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                           tag: str = Query("", max_length=64), label: ReleaseLabel | None = None,
                           project_id: UUID = Depends(get_sdk_project), db: AsyncSession = Depends(get_db)):
    return await list_prompt_catalog(db, offset=offset, limit=limit, tag=tag, label=label)


@router.get("/{name:path}")
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
    elif label == "latest":
        query = query.where(PromptVersion.version == prompt.latest_version)
    else:
        query = query.join(PromptLabel, PromptLabel.version_id == PromptVersion.id).where(
            PromptLabel.prompt_id == prompt.id, PromptLabel.label == (label or "production"))
    snapshot = (await db.execute(query)).scalar_one_or_none()
    if snapshot is None:
        raise HTTPException(404, "Requested prompt version or release label not found. Promote a version first.")
    payload = await snapshot_response(db, prompt, snapshot)
    etag = '"' + hashlib.sha256(payload.model_dump_json().encode()).hexdigest() + '"'
    headers = {"ETag": etag, "Cache-Control": "private, no-store", "Vary": "Authorization"}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload.model_dump(mode="json"), headers=headers)
