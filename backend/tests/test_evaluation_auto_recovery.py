from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import select

from app.models.evaluation import EvaluationRun
from app.models.evaluation_execution import EvaluationExecution
from app.models.prompt import now
from app.repositories import evaluation_execution as repository
from app.services.evaluation_runner import request_from_config
from tests.factories import create_evaluation_scenario


@pytest.mark.asyncio
async def test_provider_free_recovery_obeys_project_admission(api, monkeypatch):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner

    async def queued(*args, **kwargs):
        pass

    monkeypatch.setattr(runner, "enqueue", queued)
    interrupted = (
        await client.post("/api/v1/evaluations", headers=headers, json=data)
    ).json()
    run_id = UUID(interrupted["id"])
    async with sessions() as db:
        execution_id = await repository.claim(db, run_id, "lost-process")
        execution = await db.get(EvaluationExecution, execution_id)
        execution.lease_until = now() - timedelta(seconds=1)
        await db.commit()
        await repository.reconcile(db)
    others = [
        (await client.post("/api/v1/evaluations", headers=headers, json=data)).json()
        for _ in range(2)
    ]
    async with sessions() as db:
        run = await db.get(EvaluationRun, run_id)
        execution = await db.get(EvaluationExecution, execution_id)
        assert not await runner.recover_provider_free(db, run, execution)
        assert (
            len(
                (
                    await db.scalars(
                        select(EvaluationExecution).where(
                            EvaluationExecution.run_id == run_id
                        )
                    )
                ).all()
            )
            == 1
        )
    await client.post(f"/api/v1/evaluations/{others[0]['id']}/cancel", headers=headers)
    async with sessions() as db:
        run = await db.get(EvaluationRun, run_id)
        execution = await db.get(EvaluationExecution, execution_id)
        assert await runner.recover_provider_free(db, run, execution)
        runner.credentials[run_id] = (run.project_id, request_from_config(run), {})
    await runner.execute(run_id)
    async with sessions() as db:
        run = await db.get(EvaluationRun, run_id)
        assert run.status == "completed" and run.completed == 2
