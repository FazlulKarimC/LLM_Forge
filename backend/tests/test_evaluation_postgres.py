"""Real database checks: SQLite does not implement row locks or SKIP LOCKED."""

import asyncio
from uuid import UUID
import pytest
from sqlalchemy import select
from app.models.evaluation import EvaluationResult, EvaluationRun
from app.models.evaluation_execution import EvaluationExecution
from app.repositories import evaluation_execution as repository
from tests.factories import create_evaluation_scenario

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


async def test_project_admission_lock_allows_checkpoint_foreign_keys(api, monkeypatch):
    _, sessions, headers, _, run_id = await queued(api, monkeypatch)
    from app.services.evaluation_runs import lock_project

    async with sessions() as db:
        execution_id = await repository.claim(db, run_id, "worker")

    async def write():
        async with sessions() as db:
            return await repository.checkpoint(
                db, execution_id, "worker", 0, "generation", {"output": "Hello"}
            )

    async with sessions() as admission:
        await lock_project(admission, UUID(headers["X-Project-ID"]))
        # Old FOR UPDATE blocks the checkpoint's FK check even though this
        # admission has not touched any run row. This deadline catches the cycle.
        assert await asyncio.wait_for(write(), timeout=3)


async def test_polling_does_not_lock_live_run_rows(api, monkeypatch):
    client, sessions, headers, _, run_id = await queued(api, monkeypatch)
    async with sessions() as db:
        await repository.claim(db, run_id, "worker")
    async with sessions() as writer:
        await writer.scalar(
            select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
        )
        responses = await asyncio.wait_for(
            asyncio.gather(
                *[
                    client.get(path, headers=headers)
                    for path in (
                        "/api/v1/evaluations",
                        f"/api/v1/evaluations/{run_id}",
                        f"/api/v1/evaluations/{run_id}/summary",
                    )
                ]
            ),
            timeout=3,
        )
        assert all(response.status_code == 200 for response in responses)


async def test_two_mock_runs_complete_while_history_and_summaries_poll(
    api, monkeypatch
):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner, EvaluationRunner

    # Use the real asynchronous enqueue path, not the fixture's synchronous run.
    monkeypatch.setattr(runner, "enqueue", EvaluationRunner.enqueue.__get__(runner))
    original_checkpoint = repository.checkpoint

    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.02)
        return await original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(repository, "checkpoint", delayed)
    responses = await asyncio.gather(
        *[
            client.post(
                "/api/v1/evaluations",
                headers={**headers, "Idempotency-Key": f"overlapping-{index}"},
                json=data,
            )
            for index in range(2)
        ]
    )
    assert all(response.status_code == 202 for response in responses)
    ids = [UUID(response.json()["id"]) for response in responses]
    workers = [asyncio.create_task(runner.execute(run_id)) for run_id in ids]
    try:
        while not all(worker.done() for worker in workers):
            replies = await asyncio.wait_for(
                asyncio.gather(
                    client.get("/api/v1/evaluations", headers=headers),
                    *[
                        client.get(
                            f"/api/v1/evaluations/{run_id}/summary", headers=headers
                        )
                        for run_id in ids
                    ],
                ),
                timeout=3,
            )
            assert all(reply.status_code == 200 for reply in replies)
            await asyncio.sleep(0.01)
        assert all(await asyncio.gather(*workers))
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    async with sessions() as db:
        runs = (await db.scalars(select(EvaluationRun))).all()
        assert len(runs) == 2
        assert all(
            run.status == "completed" and run.completed == run.total == 2
            for run in runs
        )


async def queued(api, monkeypatch):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    from app.services.evaluation_runner import runner

    async def enqueue(*args, **kwargs):
        pass

    monkeypatch.setattr(runner, "enqueue", enqueue)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    assert response.status_code == 202, response.text
    return client, sessions, headers, data, UUID(response.json()["id"])


async def test_one_owner_and_atomic_case_finalization(api, monkeypatch):
    _, sessions, _, _, run_id = await queued(api, monkeypatch)

    async def claim(owner):
        async with sessions() as db:
            return await repository.claim(db, run_id, owner)

    claims = await asyncio.gather(claim("first"), claim("second"))
    assert sum(value is not None for value in claims) == 1
    owner = "first" if claims[0] else "second"
    execution_id = claims[0] or claims[1]
    checks = [
        {
            "kind": name,
            "name": name,
            "assignment": "evaluator",
            "passed": True,
            "data_type": "numeric",
            "value": 0.9,
            "reason": "saved",
        }
        for name in ("quality", "accuracy")
    ]

    async def checkpoint():
        async with sessions() as db:
            return await repository.checkpoint(
                db, execution_id, owner, 0, "evaluator", {"checks": checks}, checks
            )

    assert all(await asyncio.gather(checkpoint(), checkpoint()))
    async with sessions() as db:
        from app.models.evaluator import EvaluationScore

        assert len((await db.scalars(select(EvaluationScore))).all()) == 2
    record = {"passed": True, "error": None, "output": "saved", "checks": []}

    async def finish():
        async with sessions() as db:
            return await repository.finalize(db, execution_id, owner, 0, record)

    assert all(await asyncio.gather(finish(), finish()))
    async with sessions() as db:
        assert len((await db.scalars(select(EvaluationResult))).all()) == 1
        from app.models.evaluation import EvaluationRun

        assert (await db.get(EvaluationRun, run_id)).completed == 1


async def test_global_capacity_and_duplicate_dispatch(api, monkeypatch):
    client, sessions, headers, data, run_id = await queued(api, monkeypatch)
    token = {**headers, "Idempotency-Key": "parallel-create"}
    replies = await asyncio.gather(
        *[
            client.post("/api/v1/evaluations", headers=token, json=data)
            for _ in range(2)
        ]
    )
    assert all(reply.status_code == 202 for reply in replies), [
        reply.text for reply in replies
    ]
    assert replies[0].json()["id"] == replies[1].json()["id"]
    second = UUID(replies[0].json()["id"])
    async with sessions() as db:
        assert await repository.claim(db, run_id, "first")
    async with sessions() as db:
        assert await repository.claim(db, second, "second")
    from tests.support import bootstrap

    app = api[0]
    _, other_headers = await bootstrap(app, client, "user_other_project")
    prompt = (
        await client.post(
            "/api/v1/prompt-library",
            headers=other_headers,
            json={"name": "Other echo", "template_text": "{{query}}"},
        )
    ).json()
    dataset = (
        await client.post(
            "/api/v1/datasets",
            headers=other_headers,
            json={
                "name": "Other cases",
                "cases": [{"inputs": {"query": "Hello"}, "expected_output": "Hello"}],
            },
        )
    ).json()
    reply = await client.post(
        "/api/v1/evaluations",
        headers=other_headers,
        json={
            "prompt_version_id": prompt["version"]["id"],
            "dataset_revision_id": dataset["revision"]["id"],
            "assertions": [{"kind": "exact_match"}],
        },
    )
    assert reply.status_code == 202, reply.text
    third_id = UUID(reply.json()["id"])
    async with sessions() as db:
        db.info["project_id"] = UUID(other_headers["X-Project-ID"])
        assert await repository.claim(db, third_id, "third") is None
    # Global count must remain two even when more work is waiting.
    async with sessions() as db:
        active = (
            await db.scalars(
                select(EvaluationExecution).where(
                    EvaluationExecution.status == "running"
                )
            )
        ).all()
        assert len(active) == 2


async def test_expired_owner_cannot_write_or_heartbeat(api, monkeypatch):
    _, sessions, _, _, run_id = await queued(api, monkeypatch)
    from datetime import timedelta
    from app.models.prompt import now

    async with sessions() as db:
        execution_id = await repository.claim(db, run_id, "old")
    async with sessions() as db:
        execution = await db.get(EvaluationExecution, execution_id)
        execution.lease_until = now() - timedelta(seconds=1)
        await db.commit()
    async with sessions() as db:
        assert not await repository.heartbeat(db, execution_id, "old")
        assert not await repository.checkpoint(
            db, execution_id, "old", 0, "generation", {"output": "late"}
        )
        assert not await repository.finalize(
            db, execution_id, "old", 0, {"passed": True, "error": None}
        )
        await repository.reconcile(db)
    async with sessions() as db:
        assert (await db.get(EvaluationExecution, execution_id)).status == "interrupted"
        assert not (await db.scalars(select(EvaluationResult))).all()

    # Credentials queued for an older attempt must never claim its replacement.
    from app.models.evaluation import EvaluationRun
    from app.services.evaluation_runner import new_execution

    async with sessions() as db:
        run = await db.get(EvaluationRun, run_id)
        run.status = "queued"
        replacement = new_execution(run, 2)
        db.add(replacement)
        await db.commit()
        replacement_id = replacement.id
    async with sessions() as db:
        assert (
            await repository.claim(
                db, run_id, "old", expected_execution_id=execution_id
            )
            is None
        )
    async with sessions() as db:
        assert (
            await repository.claim(
                db, run_id, "new", expected_execution_id=replacement_id
            )
            == replacement_id
        )
        assert not await repository.heartbeat(db, execution_id, "old")
