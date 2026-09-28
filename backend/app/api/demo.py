"""Create a small provider-free example in the current project."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.tenancy import ProjectContext, get_project_context
from app.models.evaluation import Dataset, DatasetRevision
from app.models.prompt import Prompt
from app.models.prompt_version import PromptVersion
from app.models.workspace import Project
from app.schemas.prompt import VersionCreate
from app.services.prompt_service import add_version

router = APIRouter(tags=["Demo"])

PROMPT_NAME = "Echo demo"
DATASET_NAME = "Greetings"
TEMPLATE = "{{query}}"
CASES = [
    {"name": "Greeting", "inputs": {"query": "Hello"}, "expected_output": "Hello"},
    {"name": "Farewell", "inputs": {"query": "Goodbye"}, "expected_output": "Goodbye"},
]


class DemoExamples(BaseModel):
    prompt_id: UUID
    prompt_version_id: UUID
    dataset_id: UUID
    dataset_revision_id: UUID


@router.post("/setup", status_code=201, response_model=DemoExamples)
async def setup_demo(
    context: ProjectContext = Depends(get_project_context),
    db: AsyncSession = Depends(get_db),
):
    # Serialize repeated clicks in the same project. Never replace user data.
    await db.scalar(
        select(Project).where(Project.id == context.project_id).with_for_update()
    )
    prompt = await db.scalar(select(Prompt).where(Prompt.name == PROMPT_NAME))
    if prompt is None:
        prompt = Prompt(
            project_id=context.project_id,
            name=PROMPT_NAME,
            description="Provider-free evaluation example",
        )
        db.add(prompt)
        await db.flush()
        version = await add_version(
            db, prompt, VersionCreate(template_text=TEMPLATE, template_format="mustache")
        )
    else:
        version = await db.scalar(
            select(PromptVersion).where(
                PromptVersion.prompt_id == prompt.id, PromptVersion.version == 1
            )
        )
        if prompt.archived or version is None or (
            version.template_text != TEMPLATE or version.template_format != "mustache"
        ):
            raise HTTPException(
                409, "Echo demo already exists with different content; use another project"
            )

    dataset = await db.scalar(select(Dataset).where(Dataset.name == DATASET_NAME))
    if dataset is None:
        dataset = Dataset(
            project_id=context.project_id,
            name=DATASET_NAME,
            description="Two fixed cases for the provider-free demo",
            latest_version=1,
        )
        db.add(dataset)
        await db.flush()
        revision = DatasetRevision(
            project_id=context.project_id,
            dataset_id=dataset.id,
            version=1,
            cases=CASES,
        )
        db.add(revision)
    else:
        revision = await db.scalar(
            select(DatasetRevision).where(
                DatasetRevision.dataset_id == dataset.id, DatasetRevision.version == 1
            )
        )
        if dataset.archived or revision is None or revision.cases != CASES:
            raise HTTPException(
                409, "Greetings already exists with different cases; use another project"
            )

    await db.commit()
    return {
        "prompt_id": prompt.id,
        "prompt_version_id": version.id,
        "dataset_id": dataset.id,
        "dataset_revision_id": revision.id,
    }
