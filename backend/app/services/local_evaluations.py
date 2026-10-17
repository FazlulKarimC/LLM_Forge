"""Persist SDK-reported summaries without hosted datasets or case payloads."""
from fastapi import HTTPException
from app.models.prompt import Prompt
from app.models.prompt_version import PromptVersion
from app.models.evaluation import EvaluationRun
from app.schemas.evaluation import run_response
from app.services.evaluation_runs import lock_project
from app.services.evaluation_idempotency import replay, remember

async def submit_local(data, db, token=None):
    await lock_project(db, db.info["project_id"])
    prior = await replay(db, token, data)
    if prior:
        return run_response(prior)
    version = await db.get(PromptVersion, data.prompt_version_id)
    prompt = await db.get(Prompt, version.prompt_id) if version else None
    if not prompt or prompt.archived:
        raise HTTPException(404, "Prompt version not found")
    run = EvaluationRun(
        prompt_version_id=version.id, dataset_revision_id=None,
        status=data.status, total=data.total, completed=data.completed,
        passed=data.passed, errors=data.errors,
        config={
            "source": "sdk_local", "provider": "external", "model": data.model,
            "prompt_name": prompt.name, "prompt_version": version.version,
            "dataset_name": data.dataset_name, "dataset_version": None,
            "dataset_fingerprint": data.dataset_fingerprint,
            "scoring_fingerprint": data.scoring_fingerprint,
            "reporting": "summary", "is_mock": False, "assertions": [],
            "metrics": data.metrics,
        },
    )
    db.add(run)
    await db.flush()
    remember(db, token, data, run)
    await db.commit()
    return run_response(run)

def compatible(candidate, reference):
    if candidate.config.get("source") == "sdk_local" or reference.config.get("source") == "sdk_local":
        return (
            candidate.config.get("source") == reference.config.get("source") == "sdk_local"
            and candidate.config.get("dataset_fingerprint") == reference.config.get("dataset_fingerprint")
            and candidate.config.get("scoring_fingerprint") == reference.config.get("scoring_fingerprint")
        )
    return candidate.dataset_revision_id == reference.dataset_revision_id
