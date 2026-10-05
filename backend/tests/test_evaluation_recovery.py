"""Interruptions preserve stage commits, fences and actor-scoped dispatch."""

import asyncio
import json
from datetime import timedelta
from uuid import UUID
import pytest
from sqlalchemy import select
from app.models.evaluation import EvaluationRun, EvaluationResult
from app.models.evaluation_execution import EvaluationExecution, EvaluationCheckpoint
from app.models.evaluator import EvaluationScore
from app.models.prompt import now
from app.repositories import evaluation_execution as repository
from tests.factories import create_evaluation_scenario
from tests.support import bootstrap


@pytest.mark.asyncio
async def test_idempotency_summary_pagination_and_isolation(api):
    app, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    token = {**headers, "Idempotency-Key": "same-intent"}
    first = await client.post("/api/v1/evaluations", headers=token, json=data)
    assert first.status_code == 202, first.text
    second = await client.post("/api/v1/evaluations", headers=token, json=data)
    assert second.json()["id"] == first.json()["id"]
    assert (
        await client.post(
            "/api/v1/evaluations", headers=token, json={**data, "max_tokens": 128}
        )
    ).status_code == 409
    root = f"/api/v1/evaluations/{first.json()['id']}"
    summary = await client.get(root + "/summary", headers=headers)
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body["results"] == [] and body["run"]["completed"] == 2
    assert "Hello" not in json.dumps(body)
    assert (
        body["score_summary"][0]["count"] == 2
        and body["score_summary"][0]["mean"] == 0.5
    )
    rows = (await client.get(root + "/cases?limit=1", headers=headers)).json()
    assert rows["total"] == 2 and len(rows["items"]) == 1
    assert "inputs" not in rows["items"][0] and "checks" not in rows["items"][0]
    assert (await client.get(root + "/cases/0", headers=headers)).json()[
        "output"
    ] == "Hello"
    assert (
        await client.post(root + "/resume", headers=headers, json={})
    ).status_code == 409
    _, stranger = await bootstrap(app, client, "stranger")
    for suffix in ("/summary", "/cases", "/cases/0"):
        assert (await client.get(root + suffix, headers=stranger)).status_code == 404
    assert (
        await client.post(root + "/resume", headers=stranger, json={})
    ).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_stage", ["generation", "assertions"])
async def test_resume_preserves_committed_stages(api, monkeypatch, crash_stage):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    original = repository.checkpoint
    from app.services import evaluation_service

    generate = evaluation_service.generate_playground
    calls = []

    async def counted(request):
        calls.append(request.variables["query"])
        return await generate(request)

    async def crash(db, execution_id, owner, index, stage, payload, checks=None):
        value = await original(db, execution_id, owner, index, stage, payload, checks)
        if index == 0 and stage == crash_stage:
            raise asyncio.CancelledError()
        return value

    monkeypatch.setattr(evaluation_service, "generate_playground", counted)
    monkeypatch.setattr(repository, "checkpoint", crash)
    from app.services.evaluation_runner import runner

    enqueue = runner.enqueue

    async def queue_only(run_id, project_id, request, keys, **kwargs):
        runner.credentials[run_id] = (project_id, request, keys)
        runner.attempts[run_id] = kwargs.get("execution_id")

    monkeypatch.setattr(runner, "enqueue", queue_only)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    with pytest.raises(asyncio.CancelledError):
        await runner.execute(UUID(response.json()["id"]))
    monkeypatch.setattr(runner, "enqueue", enqueue)
    async with sessions() as db:
        run = await db.scalar(select(EvaluationRun))
        execution = await db.scalar(select(EvaluationExecution))
        old_id, old_owner, run_id = execution.id, execution.owner, run.id
        execution.lease_until = now() - timedelta(seconds=1)
        await db.commit()
    root = f"/api/v1/evaluations/{run_id}"
    summary = (await client.get(root + "/summary", headers=headers)).json()
    assert summary["recovery"]["available"]
    assert summary["recovery"]["failure_code"] == "runner_interrupted"
    monkeypatch.setattr(repository, "checkpoint", original)
    resumed = await client.post(
        root + "/resume", headers={**headers, "Idempotency-Key": "resume-once"}, json={}
    )
    assert resumed.status_code == 202, resumed.text
    final = (await client.get(root, headers=headers)).json()
    assert final["run"]["completed"] == 2 and final["run"]["passed"] == 1
    assert calls == ["Hello", "Goodbye"]
    assert (
        await client.post(
            root + "/resume",
            headers={**headers, "Idempotency-Key": "resume-once"},
            json={},
        )
    ).status_code == 202
    async with sessions() as db:
        assert len((await db.scalars(select(EvaluationResult))).all()) == 2
        assert len((await db.scalars(select(EvaluationScore))).all()) == 2
        assert await repository.fence(db, old_id, old_owner) is None
        assert len((await db.scalars(select(EvaluationExecution))).all()) == 2


@pytest.mark.asyncio
async def test_cancelled_owner_cannot_checkpoint(api, monkeypatch):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner

    async def queue_only(*args, **kwargs):
        pass

    monkeypatch.setattr(runner, "enqueue", queue_only)
    run = (await client.post("/api/v1/evaluations", headers=headers, json=data)).json()
    async with sessions() as db:
        execution_id = await repository.claim(db, UUID(run["id"]), "owner")
    await client.post(f"/api/v1/evaluations/{run['id']}/cancel", headers=headers)
    async with sessions() as db:
        assert not await repository.checkpoint(
            db, execution_id, "owner", 0, "generation", {"output": "late"}
        )
        assert not (await db.scalars(select(EvaluationCheckpoint))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_first", [False, True])
async def test_live_recovery_skips_saved_judge_and_requires_keys(
    api, monkeypatch, error_first
):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services import evaluation_service, evaluator_service
    from app.services.evaluation_runner import runner

    definitions = []
    for name in ("Quality", "Helpfulness"):
        response = await client.post(
            "/api/v1/evaluators",
            headers=headers,
            json={
                "name": name,
                "definition": {
                    "kind": "llm_judge",
                    "provider": "groq",
                    "model": "judge",
                    "rubric": "Judge correctness",
                    "outputs": [
                        {
                            "name": "quality",
                            "data_type": "numeric",
                            "minimum": 0,
                            "maximum": 1,
                            "threshold": 0.7,
                        }
                    ],
                },
            },
        )
        assert response.status_code == 201, response.text
        definitions.append(response.json()["latest"]["id"])
    calls = []

    async def generate(request):
        calls.append(("generation", request.variables["query"]))
        return {
            "output": request.variables["query"],
            "latency_ms": 1,
            "tokens_input": 3,
            "tokens_output": 2,
        }

    async def judge(snapshot, case, output, key):
        assert key.get_secret_value() in ("judge-secret", "new-judge-secret")
        calls.append((snapshot["version_id"], output))
        checks = [
            {
                "kind": "quality",
                "name": "quality",
                "assignment": snapshot["version_id"],
                "evaluator_version_id": snapshot["version_id"],
                "data_type": "numeric",
                "value": 0.9,
                "passed": True,
                "required": True,
                "source": "judge",
                "reason": "Correct",
            }
        ]

        if (
            error_first
            and snapshot["version_id"] == definitions[0]
            and output == "Hello"
        ):
            checks[0].update(
                passed=False,
                error="Provider rejected request",
                reason="Provider rejected request",
            )
        return checks

    original_checkpoint = repository.checkpoint

    async def crash(db, execution_id, owner, index, stage, payload, checks=None):
        result = await original_checkpoint(
            db, execution_id, owner, index, stage, payload, checks
        )
        if index == 0 and stage == definitions[0]:
            raise asyncio.CancelledError()
        return result

    monkeypatch.setattr(evaluation_service, "generate_playground", generate)
    monkeypatch.setattr(evaluator_service, "execute_evaluator", judge)
    monkeypatch.setattr(repository, "checkpoint", crash)
    enqueue = runner.enqueue

    async def queue_only(run_id, project_id, request, keys, **kwargs):
        runner.credentials[run_id] = (project_id, request, keys)
        runner.attempts[run_id] = kwargs.get("execution_id")

    monkeypatch.setattr(runner, "enqueue", queue_only)
    request = {
        **data,
        "provider": "groq",
        "model": "model",
        "api_key": "generation-secret",
        "evaluators": [
            {"version_id": version, "api_key": "judge-secret"}
            for version in definitions
        ],
    }
    response = await client.post("/api/v1/evaluations", headers=headers, json=request)
    assert response.status_code == 202, response.text
    run_id = UUID(response.json()["id"])
    with pytest.raises(asyncio.CancelledError):
        await runner.execute(run_id)
    async with sessions() as db:
        execution = await db.scalar(select(EvaluationExecution))
        execution.lease_until = now() - timedelta(seconds=1)
        await db.commit()
        await repository.reconcile(db)
    await runner.tick()
    assert not runner.tasks  # A live interruption never silently replays paid work.
    root = f"/api/v1/evaluations/{run_id}"
    assert (
        await client.post(root + "/resume", headers=headers, json={})
    ).status_code == 422
    monkeypatch.setattr(repository, "checkpoint", original_checkpoint)
    monkeypatch.setattr(runner, "enqueue", enqueue)
    reply = await client.post(
        root + "/resume",
        headers=headers,
        json={
            "api_key": "new-generation-secret",
            "evaluator_keys": {version: "new-judge-secret" for version in definitions},
        },
    )
    assert reply.status_code == 202, reply.text
    assert calls.count((definitions[0], "Hello")) == 1
    assert calls.count(("generation", "Hello")) == 1
    assert len(calls) == 6
    final = (await client.get(root + "/summary", headers=headers)).json()
    assert final["run"]["completed"] == 2
    assert final["run"]["errors"] == int(error_first)
    async with sessions() as db:
        run = await db.get(EvaluationRun, run_id)
        checkpoints = (await db.scalars(select(EvaluationCheckpoint))).all()
        serialized = json.dumps([run.config, *[item.data for item in checkpoints]])
        assert "secret" not in serialized and "api_key" not in serialized


@pytest.mark.asyncio
async def test_generic_retention_preserves_active_jobs(api):
    _, _, sessions = api
    from app.models.background_job import BackgroundJobRecord
    from app.core.background_jobs import _cleanup_expired_jobs

    _, _, _, _, _, _ = await create_evaluation_scenario(api)
    async with sessions() as db:
        from app.models.workspace import Project

        project = await db.scalar(select(Project))
        for status in ("queued", "running", "completed", "failed"):
            db.add(
                BackgroundJobRecord(
                    project_id=project.id,
                    job_id=status,
                    kind="test",
                    status=status,
                    updated_at=now() - timedelta(hours=7),
                )
            )
        await db.commit()
        await _cleanup_expired_jobs(db)
        await db.commit()
        assert set((await db.scalars(select(BackgroundJobRecord.status))).all()) == {
            "queued",
            "running",
        }


@pytest.mark.asyncio
async def test_old_attempt_preserves_new_resume_credentials(api, monkeypatch):
    _, client, _, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner

    original = repository.checkpoint
    replacement = None

    async def queued(run_id, project_id, request, keys, **kwargs):
        runner.credentials[run_id] = (project_id, request, keys)
        runner.attempts[run_id] = kwargs.get("execution_id")

    async def crash(db, execution_id, owner, index, stage, payload, checks=None):
        nonlocal replacement
        await original(db, execution_id, owner, index, stage, payload, checks)
        execution = await db.get(EvaluationExecution, execution_id)
        accepted = runner.credentials[execution.run_id]
        replacement = tuple(list(accepted))  # Same intent, distinct accepting handoff.
        runner.credentials[execution.run_id] = replacement
        raise asyncio.CancelledError()

    monkeypatch.setattr(runner, "enqueue", queued)
    monkeypatch.setattr(repository, "checkpoint", crash)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    run_id = UUID(response.json()["id"])
    with pytest.raises(asyncio.CancelledError):
        await runner.execute(run_id)
    assert replacement is not None and runner.credentials[run_id] is replacement


def test_fingerprints_preserve_user_metrics_and_exclude_only_credentials():
    from app.schemas.evaluation import EvaluationCreate, EvaluationSubmission
    from app.schemas.evaluation_recovery import ResumeEvaluation
    from app.services.evaluation_idempotency import fingerprint

    identity = {
        "prompt_version_id": "00000000-0000-0000-0000-000000000001",
        "dataset_revision_id": "00000000-0000-0000-0000-000000000002",
    }
    live = {
        **identity,
        "provider": "groq",
        "model": "model",
        "assertions": [{"kind": "exact_match"}],
    }
    assert fingerprint(EvaluationCreate(**live, api_key="first-secret")) == fingerprint(
        EvaluationCreate(**live, api_key="other-secret")
    )
    assert fingerprint(ResumeEvaluation(judge_key="first")) == fingerprint(
        ResumeEvaluation(judge_key="second")
    )
    submitted = {
        **identity,
        "model": "local",
        "results": [
            {
                "case_index": 0,
                "output": "saved",
                "checks": [{"name": "correct", "passed": True}],
            }
        ],
    }
    assert fingerprint(
        EvaluationSubmission(**submitted, metrics={"api_key": 0.1})
    ) != fingerprint(EvaluationSubmission(**submitted, metrics={"api_key": 0.9}))


@pytest.mark.asyncio
async def test_old_credentials_cannot_claim_a_superseding_attempt(api, monkeypatch):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner, new_execution

    original_enqueue = type(runner).enqueue.__get__(runner)

    async def queued(*args, **kwargs):
        await original_enqueue(*args, **kwargs)

    monkeypatch.setattr(runner, "enqueue", queued)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    run_id = UUID(response.json()["id"])
    old_identity = runner.attempts[run_id]
    original_claim = repository.claim
    replacement = None

    async def supersede(db, identity, owner, **kwargs):
        nonlocal replacement
        if replacement is None:
            async with sessions() as transition:
                old = await transition.get(EvaluationExecution, old_identity)
                old.status, old.failure_code = "interrupted", "runner_interrupted"
                run = await transition.get(EvaluationRun, run_id)
                recovered = new_execution(run, 2)
                transition.add(recovered)
                await transition.commit()
                replacement = recovered.id
            accepted = runner.credentials[run_id]
            await original_enqueue(run_id, *accepted, execution_id=replacement)
        return await original_claim(db, identity, owner, **kwargs)

    monkeypatch.setattr(repository, "claim", supersede)
    assert await runner.execute(run_id)
    async with sessions() as db:
        assert (await db.get(EvaluationExecution, replacement)).status == "queued"
        assert not (await db.scalars(select(EvaluationCheckpoint))).all()
    await runner.execute(run_id)
    async with sessions() as db:
        assert (await db.get(EvaluationRun, run_id)).status == "completed"
