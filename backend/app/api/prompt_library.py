"""Workspace prompt hub. SDK fetching lives in a separate read-only router."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.tenancy import get_project_context, ProjectContext
from app.models.prompt import Prompt, PromptLabel
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import PromptCreate, PromptUpdate, VersionCreate, Promotion, CompileRequest, PlaygroundRequest, ReleaseLabel, PromptResponse, VersionResponse, version_response
from app.services.prompt_service import find_prompt, add_version, flush_unique, describe_prompt, describe_prompts
from app.services.prompt_templates import compile_template

router = APIRouter(tags=["Prompt library"])


@router.post("/compile")
async def compile_draft(data: CompileRequest):
    try:
        compiled = compile_template(data.template_text, data.variables, data.template_format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"compiled_prompt": compiled}


@router.post("/playground")
async def run_playground(data: PlaygroundRequest, db: AsyncSession = Depends(get_db)):
    from app.services.playground_service import generate_playground
    # Project membership was already verified by the router dependency. Release
    # its read transaction before waiting for a remote model provider.
    await db.rollback()
    return await generate_playground(data)


@router.get("")
async def list_prompts(search: str = Query("", max_length=255), archived: bool = False,
                       offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                       db: AsyncSession = Depends(get_db)):
    filters = [Prompt.archived == archived]
    if search.strip():
        # Treat %, _ and backslash as literal search characters.
        pattern = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Prompt.name.ilike(f"%{pattern}%", escape="\\"))
    total = (await db.execute(select(func.count()).select_from(Prompt).where(*filters))).scalar_one()
    prompts = (await db.execute(select(Prompt).where(*filters).order_by(Prompt.updated_at.desc(), Prompt.id)
        .offset(offset).limit(limit))).scalars().all()
    return {"items": await describe_prompts(db, prompts), "total": total}


@router.post("", status_code=201)
async def create_prompt(data: PromptCreate, context: ProjectContext = Depends(get_project_context), db: AsyncSession = Depends(get_db)):
    prompt = Prompt(project_id=context.project_id, name=data.name, description=data.description)
    db.add(prompt)
    await flush_unique(db, "A prompt with this name already exists in this project, including archived prompts.")
    version = await add_version(db, prompt, data)
    payload = {"prompt": await describe_prompt(db, prompt), "version": version_response(version)}
    await db.commit()
    return payload


@router.get("/{prompt_id}")
async def get_prompt(prompt_id: UUID, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, allow_archived=True)
    latest = (await db.execute(select(PromptVersion).where(PromptVersion.prompt_id == prompt.id,
        PromptVersion.version == prompt.latest_version))).scalar_one()
    return {"prompt": await describe_prompt(db, prompt), "version": version_response(latest)}


@router.patch("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: UUID, data: PromptUpdate, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, lock=True, allow_archived=True)
    for field, value in data.model_dump(exclude_unset=True).items():
        if value is None:
            raise HTTPException(422, f"{field} cannot be null")
        setattr(prompt, field, value)
    await flush_unique(db, "A prompt with this name already exists in this project.")
    payload = await describe_prompt(db, prompt)
    await db.commit()
    return payload


@router.delete("/{prompt_id}", status_code=204)
async def archive_prompt(prompt_id: UUID, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, lock=True, allow_archived=True)
    prompt.archived = True
    await db.commit()
    return Response(status_code=204)


@router.get("/{prompt_id}/versions", response_model=list[VersionResponse])
async def list_versions(prompt_id: UUID, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    await find_prompt(db, prompt_id, allow_archived=True)
    versions = (await db.execute(select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
        .order_by(PromptVersion.version.desc()).offset(offset).limit(limit))).scalars().all()
    return [version_response(version) for version in versions]


@router.post("/{prompt_id}/versions", response_model=VersionResponse, status_code=201)
async def save_version(prompt_id: UUID, data: VersionCreate, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, lock=True)
    payload = version_response(await add_version(db, prompt, data))
    await db.commit()
    return payload


@router.put("/{prompt_id}/labels/{label}", response_model=PromptResponse)
async def promote(prompt_id: UUID, label: ReleaseLabel, data: Promotion, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, lock=True)
    version = (await db.execute(select(PromptVersion).where(PromptVersion.id == data.version_id,
        PromptVersion.prompt_id == prompt.id))).scalar_one_or_none()
    if version is None:
        raise HTTPException(404, "Version not found in this prompt")
    release = (await db.execute(select(PromptLabel).where(PromptLabel.prompt_id == prompt.id, PromptLabel.label == label))).scalar_one_or_none()
    if release:
        release.version_id = version.id
    else:
        db.add(PromptLabel(project_id=prompt.project_id, prompt_id=prompt.id, label=label, version_id=version.id))
    await db.flush()
    payload = await describe_prompt(db, prompt)
    await db.commit()
    return payload


@router.delete("/{prompt_id}/labels/{label}", status_code=204)
async def remove_label(prompt_id: UUID, label: ReleaseLabel, db: AsyncSession = Depends(get_db)):
    prompt = await find_prompt(db, prompt_id, lock=True)
    release = (await db.execute(select(PromptLabel).where(PromptLabel.prompt_id == prompt.id, PromptLabel.label == label))).scalar_one_or_none()
    if release:
        await db.delete(release)
    await db.commit()
    return Response(status_code=204)
