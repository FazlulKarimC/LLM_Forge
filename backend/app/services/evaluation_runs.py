"""Persisted evaluation lifecycle shared by dashboard and project-key APIs."""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import (
    Dataset,
    DatasetRevision,
    EvaluationResult,
    EvaluationRun,
)
from app.models.prompt import Prompt, now
from app.models.prompt_version import PromptVersion
from app.models.workspace import Project
from app.schemas.evaluation import EvaluationSubmission, run_response
from app.services.evaluation_service import (
    ACTIVE,
    execute_evaluation,
    expire_run,
    validate_assertions,
)
from app.services.prompt_templates import compile_prompt


async def find_run(db, run_id):
    run = await db.scalar(
        select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
    )
    if run is None:
        raise HTTPException(404, "Evaluation not found")
    await expire_run(run)
    return run


async def load_inputs(db, prompt_version_id, dataset_revision_id, action):
    version = await db.scalar(
        select(PromptVersion).where(PromptVersion.id == prompt_version_id)
    )
    revision = await db.scalar(
        select(DatasetRevision).where(DatasetRevision.id == dataset_revision_id)
    )
    if version is None or revision is None:
        raise HTTPException(404, "Prompt version or dataset revision not found")
    prompt = await db.scalar(select(Prompt).where(Prompt.id == version.prompt_id))
    dataset = await db.scalar(select(Dataset).where(Dataset.id == revision.dataset_id))
    if prompt.archived or dataset.archived:
        raise HTTPException(409, f"Restore the prompt and dataset before {action}")
    return version, revision, prompt, dataset


async def dispatch_evaluation(data, tasks, project_id, db):
    # Serialize admission within one project; two live runs is ample for a demo.
    await db.scalar(select(Project).where(Project.id == project_id).with_for_update())
    active = (
        await db.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.status.in_(ACTIVE))
            .with_for_update()
        )
    ).all()
    for run in active:
        await expire_run(run)
    if sum(run.status in ACTIVE for run in active) >= 2:
        raise HTTPException(
            409, "Two evaluations are already active; wait or cancel one"
        )
    version, revision, prompt, dataset = await load_inputs(
        db, data.prompt_version_id, data.dataset_revision_id, "starting an evaluation"
    )
    maximum = 20 if data.judge else (100 if data.provider == "mock" else 50)
    if len(revision.cases) > maximum:
        raise HTTPException(
            422,
            f"This run is limited to {maximum} cases; create a smaller dataset revision",
        )
    validate_assertions(data.assertions, revision.cases)
    for index, case in enumerate(revision.cases):
        try:
            compile_prompt(version.template_text, version.messages, version.prompt_type, case["inputs"], version.template_format)
        except ValueError as exc:
            raise HTTPException(422, f"Case {index + 1}: {exc}") from exc
    config = data.model_dump(
        mode="json",
        exclude={"api_key", "judge", "prompt_version_id", "dataset_revision_id"},
    )
    if data.judge:
        config["judge"] = data.judge.model_dump(mode="json", exclude={"api_key"})
    config.update(
        prompt_name=prompt.name,
        prompt_version=version.version,
        dataset_name=dataset.name,
        dataset_version=revision.version,
        is_mock=data.provider == "mock",
    )
    run = EvaluationRun(
        config=config,
        prompt_version_id=version.id,
        dataset_revision_id=revision.id,
        total=len(revision.cases),
    )
    db.add(run)
    await db.commit()  # The worker must see a committed record.
    tasks.add_task(
        execute_evaluation,
        run.id,
        project_id,
        data,
        {
            "template_text": version.template_text,
            "template_format": version.template_format,
            "prompt_type": version.prompt_type,
            "messages": version.messages,
        },
        revision.cases,
    )
    return run_response(run)


async def list_runs(db: AsyncSession, offset: int = 0, limit: int = 50):
    # Reconcile abandoned runs only in this authorized project.
    active = (
        await db.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.status.in_(ACTIVE))
            .with_for_update()
        )
    ).all()
    for run in active:
        await expire_run(run)
    await db.flush()
    total = await db.scalar(select(func.count()).select_from(EvaluationRun))
    items = (
        await db.scalars(
            select(EvaluationRun)
            .order_by(EvaluationRun.created_at.desc(), EvaluationRun.id)
            .offset(offset)
            .limit(limit)
        )
    ).all()
    response = {"items": [run_response(run) for run in items], "total": total}
    await db.commit()
    return response


async def get_run(run_id: UUID, db: AsyncSession):
    run = await find_run(db, run_id)
    results = (
        await db.scalars(
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run.id)
            .order_by(EvaluationResult.case_index)
        )
    ).all()
    response = {
        "run": run_response(run),
        "results": [
            {"case_index": result.case_index, **result.data} for result in results
        ],
    }
    await db.commit()
    return response


async def cancel_run(run_id: UUID, db: AsyncSession):
    run = await find_run(db, run_id)
    if run.status in ACTIVE:
        run.status, run.updated_at = "cancelled", now()
    await db.commit()
    return run_response(run)


async def submit_results(data: EvaluationSubmission, db: AsyncSession):
    version, revision, prompt, dataset = await load_inputs(
        db, data.prompt_version_id, data.dataset_revision_id, "submitting results"
    )
    if sorted(result.case_index for result in data.results) != list(
        range(len(revision.cases))
    ):
        raise HTTPException(
            422, "Submit exactly one result for every case in the saved revision"
        )
    for result in data.results:
        if result.error is None and (result.output is None or not result.checks):
            raise HTTPException(
                422, "Successful cases require an output and at least one check"
            )
    config = {
        "source": "sdk_submission",
        "provider": "external",
        "model": data.model,
        "prompt_name": prompt.name,
        "prompt_version": version.version,
        "dataset_name": dataset.name,
        "dataset_version": revision.version,
        "is_mock": False,
        "assertions": [],
        "metrics": data.metrics,
    }
    run = EvaluationRun(
        config=config,
        prompt_version_id=version.id,
        dataset_revision_id=revision.id,
        status="completed",
        total=len(revision.cases),
        completed=len(revision.cases),
        passed=0,
        errors=0,
    )
    db.add(run)
    await db.flush()
    for result in data.results:
        case = revision.cases[result.case_index]
        passed = result.error is None and all(check.passed for check in result.checks)
        run.passed += int(passed)
        run.errors += int(result.error is not None)
        record = {
            **case,
            "output": result.output,
            "error": result.error,
            "passed": passed,
            "checks": [
                {
                    "kind": "external:" + check.name,
                    **check.model_dump(exclude={"name"}, exclude_none=True),
                }
                for check in result.checks
            ],
            "latency_ms": result.latency_ms,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
        }
        db.add(
            EvaluationResult(run_id=run.id, case_index=result.case_index, data=record)
        )
    await db.commit()
    return run_response(run)
