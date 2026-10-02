"""Prompt lifecycle with serialized, immutable version allocation."""
import hashlib
import json
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy import select, func, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from app.models.prompt import Prompt, PromptLabel
from app.models.prompt_version import PromptVersion
from app.schemas.prompt import PromptResponse, LabelResponse
from app.services.prompt_templates import prompt_variables


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


async def add_version(db, prompt, data, *, created_by="UI"):
    if prompt.prompt_type != data.prompt_type:
        raise HTTPException(422, "A prompt's text/chat type cannot change between versions")
    try:
        prompt_variables(data.template_text, data.messages, data.prompt_type, data.template_format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if data.base_version is not None and data.base_version != prompt.latest_version:
        raise HTTPException(409, "A newer version was saved. Reload the prompt before saving your changes.")
    digest = PromptVersion.compute_hash(data.template_text)
    if data.prompt_type == "chat" or data.config:
        digest = hashlib.sha256(json.dumps({"type": data.prompt_type, "text": data.template_text,
            "messages": [message.model_dump() for message in data.messages], "config": data.config},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    duplicate = (await db.execute(select(PromptVersion.id).where(PromptVersion.prompt_id == prompt.id,
        PromptVersion.sha256_hash == digest, PromptVersion.template_format == data.template_format).limit(1))).scalar_one_or_none()
    if duplicate:
        raise HTTPException(409, "An identical version already exists. Promote that version to roll back instead.")
    previous = (await db.execute(select(PromptVersion).where(
        PromptVersion.prompt_id == prompt.id, PromptVersion.version == prompt.latest_version))).scalar_one_or_none()
    version = PromptVersion(project_id=prompt.project_id, prompt_id=prompt.id, name=prompt.name,
        template_text=data.template_text, template_format=data.template_format,
        prompt_type=data.prompt_type, messages=[message.model_dump() for message in data.messages],
        config=data.config, created_by=created_by,
        version=prompt.latest_version + 1, sha256_hash=digest,
        description=data.description, parent_id=previous.id if previous else None)
    prompt.latest_version += 1
    # Explicitly mark the identity as changed when a version is added.
    from app.models.prompt import now
    prompt.updated_at = now()
    db.add(version)
    await flush_unique(db, "An identical version already exists. Promote that version to roll back instead.")
    for label in data.labels:
        await assign_label(db, prompt, label, version.id)
    return version


async def assign_label(db, prompt, label, version_id):
    if label == "latest":
        raise HTTPException(422, "latest is managed automatically and cannot be changed or removed")
    release = await db.scalar(select(PromptLabel).where(PromptLabel.prompt_id == prompt.id, PromptLabel.label == label))
    if release:
        release.version_id = version_id
    else:
        db.add(PromptLabel(project_id=prompt.project_id, prompt_id=prompt.id, label=label, version_id=version_id))
    await db.flush()


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
    latest = (await db.execute(select(PromptVersion).join(Prompt, Prompt.id == PromptVersion.prompt_id)
        .where(Prompt.id.in_([prompt.id for prompt in prompts]), PromptVersion.version == Prompt.latest_version))).scalars().all()
    for version in latest:
        labels.setdefault(version.prompt_id, []).append(LabelResponse(label="latest", version_id=version.id,
            version=version.version, updated_at=version.created_at))
    return [PromptResponse.model_validate({**{field: getattr(prompt, field) for field in PromptResponse.model_fields if field != "labels"},
        "labels": sorted(labels.get(prompt.id, []), key=lambda value: value.label)}) for prompt in prompts]


async def describe_prompt(db, prompt):
    return (await describe_prompts(db, [prompt]))[0]


async def list_prompt_catalog(db, *, search="", archived=False, offset=0, limit=50, tag="", folder="", label=None):
    """Filter the authorized collection before counting or paginating it."""
    filters = [Prompt.archived == archived]
    if tag:
        if db.bind.dialect.name == "postgresql":
            filters.append(cast(Prompt.tags, JSONB).contains([tag]))
        else:
            tags = func.json_each(Prompt.tags).table_valued("value")
            filters.append(select(1).select_from(tags).where(tags.c.value == tag).exists())
    if folder.strip("/"):
        prefix = folder.strip("/").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Prompt.name.ilike(prefix + "/%", escape="\\"))
    if label and label != "latest":
        filters.append(select(PromptLabel.prompt_id).where(PromptLabel.prompt_id == Prompt.id, PromptLabel.label == label).exists())
    if search.strip():
        pattern = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Prompt.name.ilike(f"%{pattern}%", escape="\\"))
    total = (await db.execute(select(func.count()).select_from(Prompt).where(*filters))).scalar_one()
    prompts = (await db.execute(select(Prompt).where(*filters).order_by(Prompt.updated_at.desc(), Prompt.id).offset(offset).limit(limit))).scalars().all()
    return {"items": await describe_prompts(db, prompts), "total": total}
