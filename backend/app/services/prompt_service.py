"""Prompt lifecycle with serialized, immutable version allocation."""
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.models.prompt import Prompt, PromptLabel
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import PromptResponse, LabelResponse
from app.services.prompt_templates import template_variables


async def find_prompt(db, prompt_id: UUID, *, lock=False, allow_archived=False):
    query = select(Prompt).where(Prompt.id == prompt_id)
    if lock:
        query = query.with_for_update()
    prompt = (await db.execute(query)).scalar_one_or_none()
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    if prompt.archived and not allow_archived:
        raise HTTPException(409, "This prompt is archived. Restore it before making changes.")
    return prompt


async def flush_unique(db, message):
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, message) from exc


async def add_version(db, prompt, data):
    try:
        template_variables(data.template_text, data.template_format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if data.base_version is not None and data.base_version != prompt.latest_version:
        raise HTTPException(409, "A newer version was saved. Reload the prompt before saving your changes.")
    digest = PromptVersion.compute_hash(data.template_text)
    duplicate = (await db.execute(select(PromptVersion.id).where(PromptVersion.prompt_id == prompt.id,
        PromptVersion.sha256_hash == digest, PromptVersion.template_format == data.template_format).limit(1))).scalar_one_or_none()
    if duplicate:
        raise HTTPException(409, "An identical version already exists. Promote that version to roll back instead.")
    previous = (await db.execute(select(PromptVersion).where(
        PromptVersion.prompt_id == prompt.id, PromptVersion.version == prompt.latest_version))).scalar_one_or_none()
    version = PromptVersion(project_id=prompt.project_id, prompt_id=prompt.id, name=prompt.name,
        template_text=data.template_text, template_format=data.template_format,
        version=prompt.latest_version + 1, sha256_hash=digest,
        description=data.description, parent_id=previous.id if previous else None)
    prompt.latest_version += 1
    # Explicitly mark the identity as changed when a version is added.
    from app.models.prompt import now
    prompt.updated_at = now()
    db.add(version)
    await flush_unique(db, "An identical version already exists. Promote that version to roll back instead.")
    return version


async def describe_prompts(db, prompts):
    if not prompts:
        return []
    rows = (await db.execute(select(PromptLabel, PromptVersion.version)
        .join(PromptVersion, PromptVersion.id == PromptLabel.version_id)
        .where(PromptLabel.prompt_id.in_([prompt.id for prompt in prompts])))).all()
    labels = {}
    for release, number in rows:
        labels.setdefault(release.prompt_id, []).append(LabelResponse(label=release.label,
            version_id=release.version_id, version=number, updated_at=release.updated_at))
    return [PromptResponse.model_validate({**{field: getattr(prompt, field) for field in PromptResponse.model_fields if field != "labels"},
        "labels": sorted(labels.get(prompt.id, []), key=lambda value: value.label)}) for prompt in prompts]


async def describe_prompt(db, prompt):
    return (await describe_prompts(db, [prompt]))[0]
