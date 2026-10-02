"""Request-scoped project authorization and ORM filtering for legacy services.

API routers require get_project_context. Internal workers use an independent
session and an already-authorized experiment ID; they never accept browser IDs.
"""

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, with_loader_criteria

from app.core.auth import Identity, get_current_user
from app.core.database import get_db
from app.models.workspace import OrganizationMembership, Project, User


@dataclass(frozen=True)
class ProjectContext:
    project_id: UUID
    organization_id: UUID
    user_id: UUID
    role: str


async def get_project_context(
    project_id: UUID | None = Header(None, alias="X-Project-ID"),
    identity: Identity = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProjectContext:
    if project_id is None:
        raise HTTPException(400, "Select a project (X-Project-ID is required)")
    row = (await db.execute(
        select(Project, OrganizationMembership, User)
        .join(OrganizationMembership, OrganizationMembership.organization_id == Project.organization_id)
        .join(User, User.id == OrganizationMembership.user_id)
        .where(Project.id == project_id, User.auth_subject == identity.subject)
    )).one_or_none()
    if row is None:
        raise HTTPException(404, "Project not found")
    project, membership, user = row
    db.info["project_id"] = project.id
    return ProjectContext(project.id, project.organization_id, user.id, membership.role)


@event.listens_for(Session, "do_orm_execute")
def scope_project_queries(state):
    project_id = state.session.info.get("project_id")
    if not isinstance(project_id, UUID):
        return
    from app.models.experiment import Experiment
    from app.models.prompt_version import PromptVersion
    from app.models.run import Run
    from app.models.result import Result
    from app.models.background_job import BackgroundJobRecord
    from app.models.prompt import Prompt, PromptLabel, ProjectAPIKey
    from app.models.evaluation import Dataset, DatasetRevision, EvaluationRun, EvaluationResult
    from app.models.evaluator import Evaluator, EvaluatorVersion, EvaluationScore

    # Covers legacy aggregates, exports, comparisons and ORM UPDATE/DELETE too.
    experiment_ids = select(Experiment.id).where(Experiment.project_id == project_id, Experiment.deleted_at.is_(None))
    state.statement = state.statement.options(
        with_loader_criteria(Dataset, Dataset.project_id == project_id, include_aliases=True),
        with_loader_criteria(DatasetRevision, DatasetRevision.project_id == project_id, include_aliases=True),
        with_loader_criteria(EvaluationRun, EvaluationRun.project_id == project_id, include_aliases=True),
        with_loader_criteria(EvaluationResult, EvaluationResult.project_id == project_id, include_aliases=True),
        with_loader_criteria(Evaluator, Evaluator.project_id == project_id, include_aliases=True),
        with_loader_criteria(EvaluatorVersion, EvaluatorVersion.project_id == project_id, include_aliases=True),
        with_loader_criteria(EvaluationScore, EvaluationScore.project_id == project_id, include_aliases=True),
        with_loader_criteria(Prompt, Prompt.project_id == project_id, include_aliases=True),
        with_loader_criteria(PromptLabel, PromptLabel.project_id == project_id, include_aliases=True),
        with_loader_criteria(ProjectAPIKey, ProjectAPIKey.project_id == project_id, include_aliases=True),
        with_loader_criteria(Experiment, Experiment.project_id == project_id, include_aliases=True),
        with_loader_criteria(PromptVersion, PromptVersion.project_id == project_id, include_aliases=True),
        with_loader_criteria(BackgroundJobRecord, BackgroundJobRecord.project_id == project_id, include_aliases=True),
        with_loader_criteria(Run, Run.experiment_id.in_(experiment_ids), include_aliases=True),
        with_loader_criteria(Result, Result.experiment_id.in_(experiment_ids), include_aliases=True),
    )


@event.listens_for(Session, "before_flush")
def scope_project_writes(session, _flush_context, _instances):
    project_id = session.info.get("project_id")
    if not isinstance(project_id, UUID):
        return
    from app.models.experiment import Experiment
    from app.models.prompt_version import PromptVersion
    from app.models.background_job import BackgroundJobRecord
    from app.models.prompt import Prompt, PromptLabel, ProjectAPIKey
    from app.models.evaluation import Dataset, DatasetRevision, EvaluationRun, EvaluationResult
    from app.models.evaluator import Evaluator, EvaluatorVersion, EvaluationScore
    for obj in session.new:
        if isinstance(obj, (Experiment, PromptVersion, BackgroundJobRecord, Prompt, PromptLabel, ProjectAPIKey, Dataset, DatasetRevision, EvaluationRun, EvaluationResult, Evaluator, EvaluatorVersion, EvaluationScore)):
            if obj.project_id is not None and obj.project_id != project_id:
                raise ValueError("Cannot write to a different project")
            obj.project_id = project_id
