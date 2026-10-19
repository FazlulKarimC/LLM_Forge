"""Provider-free execution regressions; PostgreSQL variants verify real locks."""

import asyncio
from datetime import timedelta
from threading import Event
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.models.benchmark_reservation import BenchmarkReservation
from app.models.experiment import Experiment
from app.models.run import Run
from app.schemas.experiment import ExperimentConfig, ExperimentStatus
from app.services.benchmark_execution import (
    BenchmarkBudgetExceeded,
    BudgetedEngine,
    ExecutionInterrupted,
    run_blocking,
    utcnow,
)
from app.services.benchmark_preflight import validate_benchmark
from app.services.experiment_service import ExperimentService
from app.services.inference.base import GenerationConfig, GenerationResult
from tests.support import bootstrap, login


async def create_benchmark(client, headers, **updates):
    response = await client.post(
        "/api/v1/experiments",
        headers=headers,
        json={
            "name": "Bounded demo",
            "config": {
                "model_name": "mock-model",
                "dataset_name": "sample",
                "num_samples": 3,
                **updates,
            },
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["id"])


class FakeEngine:
    def __init__(self, callback=None):
        self.calls = 0
        self.callback = callback

    def generate(self, prompt, config):
        self.calls += 1
        if self.callback:
            self.callback(self.calls)
        return GenerationResult(
            text="Paris",
            tokens_input=5,
            tokens_output=1,
            latency_ms=1,
            finish_reason="stop",
        )

    def unload_model(self):
        pass


@pytest.mark.parametrize(
    "storage", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
@pytest.mark.asyncio
async def test_stop_keeps_committed_cases_and_holds_slot_until_call_exits(
    api, monkeypatch, storage
):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-stop")
    identity = await create_benchmark(client, headers)
    # A second project must see the same capacity limit.
    _, other_headers = await bootstrap(app, client, "benchmark-other-project")
    other = await create_benchmark(client, other_headers)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        await db.commit()

        def generation(number):
            assert not db.in_transaction(), "Provider I/O held a database transaction"
            if number == 2:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5), "Test did not release the fake provider"

        engine = FakeEngine(generation)
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (engine, "fake")
        )
        # Any accidental optional enrichment fails the test without using a provider.
        monkeypatch.setattr(
            service._runner_helper(),
            "score_semantic_similarity",
            lambda *a: pytest.fail("Hidden embedding call"),
        )
        worker = asyncio.create_task(
            service.execute(identity, require_queued=True, expected_attempt=attempt)
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            async with sessions() as inspection:
                assert (
                    await inspection.scalar(
                        select(func.count())
                        .select_from(Run)
                        .where(Run.experiment_id == identity)
                    )
                    == 1
                )
            login(app, "benchmark-stop")
            stopped = await asyncio.wait_for(
                client.post(
                    f"/api/v1/experiments/{identity}/interrupt", headers=headers
                ),
                2,
            )
            assert stopped.status_code == 200, stopped.text
            # Interrupt acknowledges immediately but cannot cancel issued HTTP work.
            login(app, "benchmark-other-project")
            busy = await client.post(
                f"/api/v1/experiments/{other}/run", headers=other_headers
            )
            assert busy.status_code == 429, busy.text
        finally:
            release.set()
            await asyncio.wait_for(worker, 5)
        assert engine.calls == 2
        async with sessions() as inspection:
            assert (
                await inspection.scalar(
                    select(func.count())
                    .select_from(Run)
                    .where(Run.experiment_id == identity)
                )
                == 1
            )
            row = await inspection.get(Experiment, identity)
            assert row.status == ExperimentStatus.FAILED
            assert "Interrupted by user" in row.error_message
            with pytest.raises(ExecutionInterrupted):
                await ExperimentService(inspection)._save_completed_cases(
                    identity, attempt, []
                )
        async with sessions() as next_session:
            # The slot is released only after the old call exits.
            await ExperimentService(next_session).queue_for_execution(other)
            await next_session.commit()


@pytest.mark.asyncio
async def test_failed_call_preserves_previous_case_and_releases_capacity(
    api, monkeypatch
):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-failure")
    identity = await create_benchmark(client, headers)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        await db.commit()

        def generation(number):
            assert not db.in_transaction()
            if number == 2:
                raise RuntimeError("Fake provider failed")

        engine = FakeEngine(generation)
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (engine, "fake")
        )
        with pytest.raises(RuntimeError, match="Fake provider failed"):
            await service.execute(
                identity, require_queued=True, expected_attempt=attempt
            )
    async with sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(Run)
                .where(Run.experiment_id == identity)
            )
            == 1
        )
        reservation = await inspection.scalar(
            select(BenchmarkReservation).where(
                BenchmarkReservation.resource_id == str(identity)
            )
        )
        assert reservation.released_at is not None


@pytest.mark.parametrize("method", ["naive", "react"])
@pytest.mark.asyncio
async def test_cancellation_drains_issued_thread_before_releasing(
    api, monkeypatch, method
):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-cancellation")
    identity = await create_benchmark(client, headers, reasoning_method=method)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    started, release = asyncio.Event(), Event()
    loop = asyncio.get_running_loop()
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        await db.commit()

        def generation(number):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)

        engine = FakeEngine(generation)
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (engine, "fake")
        )
        worker = asyncio.create_task(
            service.execute(identity, require_queued=True, expected_attempt=attempt)
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            worker.cancel()
            await asyncio.sleep(0)
            async with sessions() as inspection:
                row = await inspection.scalar(
                    select(BenchmarkReservation).where(
                        BenchmarkReservation.resource_id == str(identity)
                    )
                )
                assert row.released_at is None
            assert not worker.done()
            worker.cancel()
            await asyncio.sleep(0)
            assert not worker.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
    async with sessions() as inspection:
        row = await inspection.get(Experiment, identity)
        assert row.status == ExperimentStatus.FAILED
    assert engine.calls == 1


@pytest.mark.asyncio
async def test_saved_prompt_batch_completes_without_embedding_calls(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-saved-prompt")
    prompt = await client.post(
        "/api/v1/prompts",
        headers=headers,
        json={
            "name": "Benchmark text",
            "template_text": "Answer: {question}",
            "template_format": "fstring",
        },
    )
    assert prompt.status_code == 201, prompt.text
    # Use the stored immutable version rather than relying on a response shape.
    from app.models.prompt_version import PromptVersion

    async with sessions() as inspection:
        version = await inspection.scalar(
            select(PromptVersion).where(PromptVersion.name == "Benchmark text")
        )
        version_id = str(version.id)
    identity = await create_benchmark(
        client,
        headers,
        prompt_version_id=version_id,
        optimization={"enable_batching": True, "batch_size": 8},
    )
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        await db.commit()
        engine = FakeEngine()
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (engine, "fake")
        )
        await service.execute(identity, require_queued=True, expected_attempt=attempt)
    async with sessions() as inspection:
        row = await inspection.get(Experiment, identity)
        assert row.status == ExperimentStatus.COMPLETED
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(Run)
                .where(Run.experiment_id == identity)
            )
            == 3
        )
        assert (
            row.run_manifest["effective_execution"]["optimization"]["batch_size"] == 1
        )
        assert (
            row.run_manifest["effective_execution"]["execution_budget"][
                "max_provider_calls"
            ]
            == 20
        )
    assert engine.calls == 3


@pytest.mark.asyncio
async def test_call_budget_and_deadline_do_not_issue_extra_requests():
    engine = FakeEngine()
    budget = BudgetedEngine(
        engine, deadline_at=utcnow() + timedelta(seconds=5), max_calls=1
    )
    await run_blocking(budget.generate, "first", GenerationConfig())
    with pytest.raises(BenchmarkBudgetExceeded, match="call budget"):
        await run_blocking(budget.generate, "second", GenerationConfig())
    expired = BudgetedEngine(engine, deadline_at=utcnow() - timedelta(seconds=1))
    with pytest.raises(BenchmarkBudgetExceeded, match="time budget"):
        await run_blocking(expired.generate, "late", GenerationConfig())
    assert engine.calls == 1


@pytest.mark.asyncio
async def test_expired_reservation_does_not_restart_provider_work(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-expired")
    identity = await create_benchmark(client, headers)
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        reservation = await db.scalar(
            select(BenchmarkReservation).where(
                BenchmarkReservation.resource_id == str(identity)
            )
        )
        reservation.deadline_at = utcnow() - timedelta(seconds=1)
        await db.commit()
        engine = FakeEngine()
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (engine, "fake")
        )
        with pytest.raises(BenchmarkBudgetExceeded):
            await service.execute(
                identity, require_queued=True, expected_attempt=attempt
            )
        assert engine.calls == 0


@pytest.mark.asyncio
async def test_direct_attempt_does_not_reuse_an_empty_failed_attempt(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-empty-history")
    identity = await create_benchmark(client, headers)
    async with sessions() as db:
        row = await db.get(Experiment, identity)
        row.current_attempt = 7
        row.status = ExperimentStatus.FAILED
        await db.commit()
        attempt, _ = await ExperimentService(db)._prepare_execution_attempt(identity)
        assert attempt == 8


@pytest.mark.asyncio
async def test_invalid_setup_fails_before_dispatch(api, monkeypatch):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "benchmark-invalid")
    identity = await create_benchmark(
        client,
        headers,
        reasoning_method="react",
        agent={"max_iterations": 3, "tools": ["unknown_tool"]},
    )
    dispatch = AsyncMock()
    monkeypatch.setattr("app.core.task_dispatch.dispatch_experiment", dispatch)
    response = await client.post(f"/api/v1/experiments/{identity}/run", headers=headers)
    assert response.status_code == 422, response.text
    assert "tools" in response.text
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_request_releases_transaction_before_background_execution(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-request-connection")
    identity = await create_benchmark(client, headers)
    checked = []

    async def dispatch(background_tasks, experiment_id, db, **kwargs):
        from app.core.task_dispatch import DispatchResult

        async def worker():
            assert not db.in_transaction(), "Run request held a connection during background execution"
            async with sessions() as worker_db:
                experiment = await worker_db.get(Experiment, experiment_id)
                assert experiment.status == ExperimentStatus.QUEUED
            checked.append(experiment_id)

        background_tasks.add_task(worker)
        return DispatchResult(backend_used="inline")

    monkeypatch.setattr("app.core.task_dispatch.dispatch_experiment", dispatch)
    response = await client.post(f"/api/v1/experiments/{identity}/run", headers=headers)
    assert response.status_code == 200, response.text
    assert checked == [identity]


@pytest.mark.parametrize(
    "updates",
    [
        {"num_samples": 21},
        {"rag": {"retrieval_method": "naive"}},
        {"routing": {"strict_comparison": False}},
        {
            "reasoning_method": "react",
            "agent": {"max_iterations": 4, "tools": ["calculator"]},
        },
    ],
)
def test_preflight_rejects_unsupported_hosted_work(updates):
    config = ExperimentConfig(model_name="mock-model", dataset_name="sample", **updates)
    with pytest.raises(ValueError):
        validate_benchmark(config)


@pytest.mark.asyncio
async def test_judge_and_synthetic_share_benchmark_capacity(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-model-jobs")
    identity = await create_benchmark(client, headers)
    async with sessions() as db:
        await ExperimentService(db).queue_for_execution(identity)
        await db.commit()
    for path in (
        f"/api/v1/results/{identity}/judge",
        "/api/v1/results/synthetic/generate",
    ):
        response = await client.post(path, headers=headers)
        assert response.status_code == 429, response.text


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_global_admission_serializes_competing_projects(api):
    from fastapi import HTTPException

    app, client, sessions = api
    _, first_headers = await bootstrap(app, client, "benchmark-race-a")
    first = await create_benchmark(client, first_headers)
    _, second_headers = await bootstrap(app, client, "benchmark-race-b")
    second = await create_benchmark(client, second_headers)

    async def queue(identity):
        async with sessions() as db:
            try:
                attempt = await ExperimentService(db).queue_for_execution(identity)
                await db.commit()
                return attempt
            except HTTPException as exc:
                await db.rollback()
                assert exc.status_code == 429
                return "busy"

    outcomes = await asyncio.wait_for(asyncio.gather(queue(first), queue(second)), 5)
    assert outcomes.count(1) == 1
    assert outcomes.count("busy") == 1


def test_strict_router_never_replays_a_partially_failed_batch():
    from app.services.inference.provider_router import ProviderRouter

    class PartialFailure(FakeEngine):
        provider_id = "fake"
        is_loaded = True
        model_name = "fake-model"

        def generate_batch(self, prompts, config, max_workers):
            self.generate(prompts[0], config)
            raise RuntimeError("Batch failed after one response")

    engine = PartialFailure()
    router = ProviderRouter([engine], strict_comparison=True)
    with pytest.raises(RuntimeError, match="Batch failed"):
        router.generate_batch(["first", "second"], GenerationConfig())
    assert engine.calls == 1


def test_react_stop_prevents_a_tool_call_after_generation():
    from app.services.agent_service import CalculatorTool, ReActAgent

    class ActionEngine(FakeEngine):
        def generate(self, prompt, config):
            result = super().generate(prompt, config)
            result.text = "Thought: calculate\nAction: calculator\nAction Input: 2+2"
            return result

    engine = ActionEngine()
    tool = CalculatorTool()
    tool.execute = Mock()
    agent = ReActAgent(engine=engine, tools=[tool], max_iterations=3)
    checks = 0

    def stopped_between_generation_and_tool():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ExecutionInterrupted()

    agent.ensure_active = stopped_between_generation_and_tool
    with pytest.raises(ExecutionInterrupted):
        agent.run("What is 2+2?")
    tool.execute.assert_not_called()
    assert engine.calls == 1


def test_explicit_hf_provider_does_not_use_the_default_mock_engine(monkeypatch):
    from app.services.inference.engine_factory import create_inference_engine

    constructor = Mock(return_value=FakeEngine())
    monkeypatch.setattr("app.services.inference.hf_api_engine.HFAPIEngine", constructor)
    engine, label = create_inference_engine(
        model_name="real-model", provider_value="hf_api", default_engine="mock"
    )
    assert engine is constructor.return_value
    assert label == "hf_api"


@pytest.mark.asyncio
async def test_judge_job_uses_budget_and_releases_reservation(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "benchmark-judge-success")
    identity = await create_benchmark(client, headers)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    monkeypatch.setattr("app.api.results_common.async_session_maker", sessions)
    async with sessions() as db:
        service = ExperimentService(db)
        attempt = await service.queue_for_execution(identity)
        await db.commit()
        monkeypatch.setattr(
            service, "_initialize_engine", lambda *a, **k: (FakeEngine(), "fake")
        )
        await service.execute(identity, require_queued=True, expected_attempt=attempt)

    class JudgeEngine(FakeEngine):
        def generate(self, prompt, config):
            result = super().generate(prompt, config)
            result.text = '{"coherence":4,"helpfulness":4,"factuality":4}'
            return result

    judge = JudgeEngine()
    monkeypatch.setattr(
        "app.services.llm_judge_service.HFAPIEngine", lambda **kwargs: judge
    )
    response = await client.post(
        f"/api/v1/results/{identity}/judge?sample_size=2", headers=headers
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = await client.get(f"/api/v1/results/jobs/{job_id}", headers=headers)
    assert status.json()["status"] == "completed", status.text
    assert status.json()["result"]["evaluated"] == 2
    assert judge.calls == 2
    async with sessions() as db:
        reservation = await db.scalar(
            select(BenchmarkReservation).where(
                BenchmarkReservation.resource_id == job_id
            )
        )
        assert reservation.released_at is not None
