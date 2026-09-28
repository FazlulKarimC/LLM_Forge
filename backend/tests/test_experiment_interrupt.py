"""Explicit legacy-run recovery keeps partial data and fences late workers."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from sqlalchemy import select

from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentStatus
from app.services.experiment_service import ExecutionInterrupted, ExperimentService
from tests.support import bootstrap, login


@pytest.mark.asyncio
async def test_interrupt_running_experiment_and_fence_worker(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "user_interrupt_owner")
    created = await client.post(
        "/api/v1/experiments",
        headers=headers,
        json={
            "name": "Interrupted demo",
            "config": {
                "model_name": "mock-model",
                "reasoning_method": "naive",
                "dataset_name": "sample",
                "num_samples": 1,
            },
        },
    )
    assert created.status_code == 201, created.text
    experiment_id = UUID(created.json()["id"])
    async with sessions() as db:
        row = await db.scalar(select(Experiment).where(Experiment.id == experiment_id))
        row.status = ExperimentStatus.RUNNING
        row.started_at = datetime.now(timezone.utc)
        await db.commit()

    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    service = ExperimentService(None)
    await service._ensure_active_attempt(experiment_id, 1)

    stopped = await client.post(f"/api/v1/experiments/{experiment_id}/interrupt", headers=headers)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] == "failed"
    assert "Interrupted by user" in stopped.json()["error_message"]
    assert (await client.post(f"/api/v1/experiments/{experiment_id}/interrupt", headers=headers)).status_code == 409
    with pytest.raises(ExecutionInterrupted):
        await service._ensure_active_attempt(experiment_id, 1)
    async with sessions() as db:
        late_worker = ExperimentService(db)
        with pytest.raises(ExecutionInterrupted):
            await late_worker._lock_active_attempt(experiment_id, 1)
        assert not await late_worker._fail_active_attempt(
            experiment_id, "Late worker error", attempt=1
        )
        await db.rollback()
    detail = await client.get(f"/api/v1/experiments/{experiment_id}", headers=headers)
    assert "Interrupted by user" in detail.json()["error_message"]

    login(app, "user_interrupt_other")
    _, other_headers = await bootstrap(app, client, "user_interrupt_other")
    assert (await client.post(f"/api/v1/experiments/{experiment_id}/interrupt", headers=other_headers)).status_code == 404


@pytest.mark.asyncio
async def test_interrupted_queued_experiment_cannot_start_late(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "user_interrupt_queued")
    created = await client.post(
        "/api/v1/experiments",
        headers=headers,
        json={
            "name": "Queued demo",
            "config": {
                "model_name": "mock-model",
                "reasoning_method": "naive",
                "dataset_name": "sample",
                "num_samples": 1,
            },
        },
    )
    experiment_id = UUID(created.json()["id"])
    async with sessions() as db:
        row = await db.scalar(select(Experiment).where(Experiment.id == experiment_id))
        row.status = ExperimentStatus.QUEUED
        await db.commit()
    stopped = await client.post(f"/api/v1/experiments/{experiment_id}/interrupt", headers=headers)
    assert stopped.status_code == 200
    async with sessions() as db:
        with pytest.raises(ExecutionInterrupted):
            await ExperimentService(db)._prepare_execution_attempt(
                experiment_id, require_queued=True
            )
