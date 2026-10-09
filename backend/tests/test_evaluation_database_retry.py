from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from sqlalchemy.exc import DBAPIError
from app.core.database_transactions import transaction_call
from app.repositories import evaluation_execution as repository
from tests.factories import create_evaluation_scenario


def aborted(state="40P01"):
    return DBAPIError("private SQL", {"key": "secret"}, SimpleNamespace(sqlstate=state))


@pytest.mark.asyncio
async def test_aborted_checkpoint_retries_without_repeating_generation(
    api, monkeypatch
):
    _, client, _, headers, _, data = await create_evaluation_scenario(api)
    from app.services import evaluation_service

    original = repository.checkpoint
    checkpoint_calls = []
    generation = AsyncMock(wraps=evaluation_service.generate_playground)

    async def checkpoint(*args, **kwargs):
        checkpoint_calls.append(1)
        if len(checkpoint_calls) == 1:
            raise aborted()
        return await original(*args, **kwargs)

    monkeypatch.setattr(repository, "checkpoint", checkpoint)
    monkeypatch.setattr(evaluation_service, "generate_playground", generation)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    assert response.status_code == 202
    run = (
        await client.get(
            f"/api/v1/evaluations/{response.json()['id']}", headers=headers
        )
    ).json()["run"]
    assert run["status"] == "completed" and run["completed"] == 2
    assert generation.await_count == 2
    assert len(checkpoint_calls) == 5


@pytest.mark.asyncio
async def test_exhausted_aborted_transaction_preserves_partial_run_for_resume(
    api, monkeypatch
):
    _, client, _, headers, _, data = await create_evaluation_scenario(api)
    from app.services import evaluation_service

    original = repository.checkpoint
    generation = AsyncMock(wraps=evaluation_service.generate_playground)
    failures = []

    async def checkpoint(db, execution_id, owner, index, stage, payload, checks=None):
        if index == 1 and stage == "generation":
            failures.append(1)
            raise aborted()
        return await original(db, execution_id, owner, index, stage, payload, checks)

    monkeypatch.setattr(repository, "checkpoint", checkpoint)
    monkeypatch.setattr(evaluation_service, "generate_playground", generation)
    response = await client.post("/api/v1/evaluations", headers=headers, json=data)
    root = f"/api/v1/evaluations/{response.json()['id']}"
    result = (await client.get(root + "/summary", headers=headers)).json()
    assert result["run"]["status"] == "failed" and result["run"]["completed"] == 1
    assert result["recovery"]["available"]
    assert result["recovery"]["failure_code"] == "database_interrupted"
    assert generation.await_count == 2 and len(failures) == 3
    monkeypatch.setattr(repository, "checkpoint", original)
    assert (
        await client.post(root + "/resume", headers=headers, json={})
    ).status_code == 202
    final = (await client.get(root + "/summary", headers=headers)).json()
    assert final["run"]["status"] == "completed" and final["run"]["completed"] == 2
    # Resume executes only the generation that never reached a durable commit.
    assert generation.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("state,attempts", [("40P01", 3), ("40001", 3), ("08006", 1)])
async def test_retry_is_bounded_and_ambiguous_failures_are_not_replayed(
    api, state, attempts
):
    sessions = api[2]
    callback = AsyncMock(side_effect=aborted(state))
    with pytest.raises(DBAPIError):
        await transaction_call(sessions, None, callback, stage="checkpoint")
    assert callback.await_count == attempts


@pytest.mark.asyncio
async def test_database_exception_response_is_handled_and_contains_request_id(api):
    app, client, _ = api

    @app.get("/database-failure-fixture")
    async def fail():
        raise aborted()

    # ASGITransport re-raises unhandled exceptions; this must be fully handled.
    response = await client.get("/database-failure-fixture")
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "secret" not in response.text and "private SQL" not in response.text
