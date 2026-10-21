"""Provider-free integration checks for attempt identity, bounded reads and gates."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy import event, select

from app.models.experiment import Experiment
from app.models.experiment_attempt import ExperimentAttempt
from app.models.result import Result
from app.models.run import Run
from app.services.experiment_service import ExperimentService
from app.services.experiment_runner import ExperimentRunExecutor, case_metadata, cached_usage
from app.services.metrics_service import MetricsService
from app.services.regression_service import RegressionService
from app.services.statistical_service import StatisticalService
from app.services.benchmark_retention import bounded_record
from app.services.inference.base import GenerationResult
from tests.test_benchmark_execution import create_benchmark, FakeEngine
from tests.support import bootstrap, login


async def execute(sessions, identity, engine=None):
    async with sessions() as db:
        service = ExperimentService(db)
        service._initialize_engine = lambda *args, **kwargs: (engine or FakeEngine(), "mock")
        attempt = await service.queue_for_execution(identity)
        await db.commit()
        await service.execute(identity, require_queued=True, expected_attempt=attempt)
        return attempt


@pytest.mark.asyncio
async def test_failed_rerun_never_reuses_metrics_or_manifest(api, monkeypatch, caplog):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "attempt-history")
    identity = await create_benchmark(client, headers)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    monkeypatch.setattr("app.api.results_export.async_session_maker", sessions)
    assert await execute(sessions, identity) == 1
    async with sessions() as db:
        first = await db.get(ExperimentAttempt, (identity, 1))
        manifest = copy.deepcopy(first.run_manifest)
        await RegressionService(db).pin_baseline(identity)
        experiment = await db.get(Experiment, identity)
        updated = copy.deepcopy(experiment.config)
        updated["hyperparameters"]["max_tokens"] = 128
        experiment.config = updated
        await db.commit()
    def fail(_):
        raise RuntimeError("sentinel-provider-key-must-not-escape")
    with pytest.raises(RuntimeError):
        await execute(sessions, identity, FakeEngine(fail))
    latest = await client.get(f"/api/v1/results/{identity}/metrics", headers=headers)
    assert latest.status_code == 404
    page = (await client.get(f"/api/v1/results/{identity}/runs/page", headers=headers)).json()
    assert page["attempt"] == 2 and page["total"] == 0
    older = await client.get(f"/api/v1/results/{identity}/metrics?attempt=1", headers=headers)
    assert older.status_code == 200 and older.json()["attempt"] == 1
    exported = await client.get(f"/api/v1/results/{identity}/export?attempt=1", headers=headers)
    assert exported.status_code == 200, exported.text
    payload = exported.json()
    assert payload["experiment"]["run_manifest"] == manifest
    assert len(payload["runs"]) == 3
    assert {"run_metadata", "tokens_input", "tokens_output", "agent_trace", "grader_results"} <= payload["runs"][0].keys()
    assert payload["runs"][0]["run_metadata"]["scoring_version"] == "deterministic-v2"
    assert "sentinel-provider-key" not in exported.text
    assert "sentinel-provider-key" not in caplog.text
    async with sessions() as db:
        experiment = await db.get(Experiment, identity)
        assert experiment.pinned_attempt == 1
        assert (await db.get(ExperimentAttempt, (identity, 1))).run_manifest == manifest
        assert (await db.get(ExperimentAttempt, (identity, 2))).status == "failed"
    candidate = await create_benchmark(client, headers)
    await execute(sessions, candidate)
    async with sessions() as db:
        candidate_exp = await db.get(Experiment, candidate)
        baseline = await RegressionService(db).find_baseline(candidate_exp)
        assert baseline.id == identity  # Match frozen config, not the rerun's config.
        verdict = await RegressionService(db).run_regression_check(candidate, identity)
        assert verdict.baseline_attempt == 1 and verdict.passed is True
    unpinned = await client.delete(f"/api/v1/experiments/{identity}/set-baseline", headers=headers)
    assert unpinned.status_code == 200, unpinned.text
    assert unpinned.json()["status"] == "failed"
    assert unpinned.json()["is_baseline"] is False and unpinned.json()["pinned_attempt"] is None
    historical = await client.get(f"/api/v1/results/{identity}/metrics?attempt=1", headers=headers)
    assert historical.status_code == 200 and historical.json()["attempt"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["naive", "react"])
async def test_mock_latency_provenance_survives_json_and_jsonl_exports(api, monkeypatch, method):
    import json
    from app.services.inference.mock_engine import MockEngine

    app, client, sessions = api
    _, headers = await bootstrap(app, client, "mock-latency-export")
    options = {"agent": {"max_iterations": 1, "tools": ["calculator"]}} if method == "react" else {}
    identity = await create_benchmark(client, headers, reasoning_method=method, num_samples=1, **options)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    engine = MockEngine(simulate_latency_ms=0)
    engine.load_model("mock-export")
    await execute(sessions, identity, engine)
    exported = await client.get(f"/api/v1/results/{identity}/export?attempt=1", headers=headers)
    assert exported.status_code == 200, exported.text
    row = exported.json()["runs"][0]
    metadata = row["run_metadata"]
    if method == "naive":
        assert metadata["latency_kind"] == "synthetic_delay"
        assert metadata["usage_source"] == metadata["cost_source"] == "synthetic"
    else:
        assert metadata["latency_kind"] == "agent_wall_time"
        assert metadata["calls"][0]["latency_kind"] == "synthetic_delay"
    stream = await client.get(f"/api/v1/results/{identity}/export?attempt=1&format=jsonl", headers=headers)
    assert stream.status_code == 200, stream.text
    cases = [json.loads(line) for line in stream.text.splitlines() if json.loads(line)["type"] == "case"]
    assert len(cases) == 1 and cases[0]["run_metadata"] == metadata


@pytest.mark.asyncio
async def test_project_scope_case_pages_detail_export_and_manual_purge(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "attempt-owner")
    identity = await create_benchmark(client, headers)
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    monkeypatch.setattr("app.api.results_export.async_session_maker", sessions)
    await execute(sessions, identity)
    await execute(sessions, identity)
    statements = []
    async with sessions() as db:
        engine = db.bind.sync_engine
    def capture(conn, cursor, sql, params, context, many):
        statements.append(sql)
    event.listen(engine, "before_cursor_execute", capture)
    try:
        page = await client.get(f"/api/v1/results/{identity}/runs/page?attempt=1&limit=1", headers=headers)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert page.status_code == 200, page.text
    assert page.json()["total"] == 3 and len(page.json()["runs"]) == 1
    selects = [sql.lower() for sql in statements if "runs.id" in sql.lower()]
    assert selects and all("runs.raw_output" not in sql and "runs.agent_trace" not in sql for sql in selects)
    case_id = page.json()["runs"][0]["id"]
    detail = await client.get(f"/api/v1/results/{identity}/runs/{case_id}?attempt=1", headers=headers)
    assert detail.status_code == 200 and detail.json()["attempt"] == 1
    assert (await client.get(f"/api/v1/results/{identity}/runs/{case_id}?attempt=2", headers=headers)).status_code == 404
    assert (await client.get(f"/api/v1/results/{identity}/runs?limit=51", headers=headers)).status_code == 422
    assert (await client.delete(f"/api/v1/results/{identity}/attempts/1", headers=headers)).status_code == 409
    stream = await client.get(f"/api/v1/results/{identity}/export?attempt=1&format=jsonl", headers=headers)
    assert stream.status_code == 200 and len(stream.text.splitlines()) == 4
    _, outsider = await bootstrap(app, client, "attempt-outsider")
    for path in ("attempts", "runs/page", f"runs/{case_id}", "export"):
        assert (await client.get(f"/api/v1/results/{identity}/{path}", headers=outsider)).status_code == 404
    login(app, "attempt-owner")
    purged = await client.delete(f"/api/v1/results/{identity}/attempts/1?exported=true", headers=headers)
    assert purged.status_code == 200, purged.text
    async with sessions() as db:
        assert await db.get(ExperimentAttempt, (identity, 1)) is None
        assert await db.scalar(select(Result).where(Result.experiment_id == identity, Result.attempt == 2))
    assert (await client.delete(f"/api/v1/results/{identity}/attempts/2?exported=true", headers=headers)).status_code == 200
    # Purging terminal current results frees retention capacity without reusing an identity.
    assert await execute(sessions, identity) == 3


def rows(n=60):
    return [SimpleNamespace(example_id=str(i), is_correct=True, score=1., latency_ms=1., raw_output="Paris", expected_output="Paris",
            run_metadata=case_metadata({"id": str(i), "question": "Capital?", "answer": "Paris"})) for i in range(n)]


@pytest.mark.asyncio
async def test_legacy_aggregate_is_never_attached_to_a_guessed_case_attempt(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "unknown-history")
    identity = await create_benchmark(client, headers)
    async with sessions() as db:
        from app.schemas.experiment import ExperimentStatus
        experiment = await db.get(Experiment, identity)
        experiment.current_attempt, experiment.status = 2, ExperimentStatus.COMPLETED
        for number in (0, 1, 2):
            db.add(ExperimentAttempt(experiment_id=identity, project_id=experiment.project_id, number=number, status="legacy", provenance_status="unknown"))
        db.add(Run(experiment_id=identity, attempt=1, example_id="old", prompt="old question"))
        db.add(Result(experiment_id=identity, attempt=0, accuracy_exact=.7, raw_metrics={"attempt": 2}))
        await db.commit()
    response = await client.get(f"/api/v1/results/{identity}?include_runs=true", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["metrics"]["attempt"] == 0 and response.json()["runs"] == []
    assert (await client.get(f"/api/v1/results/{identity}/metrics?attempt=2", headers=headers)).status_code == 404
    assert (await client.get(f"/api/v1/results/{identity}/metrics?recompute=true", headers=headers)).status_code == 409
    exported = await client.get(f"/api/v1/results/{identity}/export", headers=headers)
    assert exported.status_code == 200, exported.text
    assert exported.json()["experiment"]["attempt"] == 0
    assert exported.json()["experiment"]["config"] is None and exported.json()["runs"] == []


def test_matching_ids_require_case_content_scorer_and_unique_identity():
    a, b = rows(), rows()
    assert StatisticalService.compare_run_sets(a, b)["comparable"]
    b[0].run_metadata["case_fingerprint"] = "different-reference"
    mismatch = StatisticalService.compare_run_sets(a, b)
    assert not mismatch["comparable"] and mismatch["mcnemar"]["p_value"] is None
    b = rows()
    b[0].run_metadata.pop("scoring_version")
    assert not StatisticalService.compare_run_sets(a, b)["comparable"]
    with pytest.raises(ValueError, match="unique"):
        StatisticalService.compare_run_sets(a + a[:1], b)


@pytest.mark.asyncio
async def test_regression_gate_counts_cases_beyond_display_limit():
    a, b = rows(75), rows(75)
    for item in b:
        item.is_correct, item.score = False, 0.
    config = {"model_name": "mock-model", "dataset_name": "sample", "regression": {"max_new_failures": 60}}
    candidate, baseline = SimpleNamespace(id=UUID(int=1), project_id=UUID(int=3), current_attempt=1, pinned_attempt=None, config=config), SimpleNamespace(id=UUID(int=2), project_id=UUID(int=3), current_attempt=1, pinned_attempt=1, config=config)
    snapshot = SimpleNamespace(config=config, status="completed", provenance_status="recorded", sample_ids=[str(i) for i in range(75)], run_manifest={})
    service = RegressionService(SimpleNamespace(get=AsyncMock(return_value=snapshot), scalar=AsyncMock(return_value=True)))
    service._load_experiment = AsyncMock(side_effect=[candidate, baseline])
    service._load_runs_at_attempt = AsyncMock(side_effect=[a, b])
    verdict = await service.run_regression_check(candidate.id, baseline.id)
    assert verdict.passed is False
    assert len(verdict.statistical_comparison["per_example_differences"]) == 50
    assert verdict.to_dict()["sample_regressions_count"] == 75
    assert any(v["rule"] == "max_new_failures" and v["count"] == 75 for v in verdict.violations)


def test_exact_numeric_and_substring_diagnostic_scoring():
    scorer, metrics = ExperimentRunExecutor(), MetricsService(None)
    def score(answer, output):
        return scorer.score_response(item={"answer": answer}, raw_output=output, parsed_answer=output, metrics_svc=metrics, use_robustness_scoring=False)
    diagnostic = score("Paris", "Paris or London")
    assert diagnostic["is_substring_match"] and diagnostic["is_correct"] is False
    assert score("1,000", "1000.00")["is_correct"] is True
    assert score("10", "210")["is_correct"] is False


def test_cached_usage_and_retention_do_not_invent_provider_work():
    result = GenerationResult("Paris", 100, 10, 350., "stop", cost_usd=.2)
    cached = cached_usage(result)
    assert cached.tokens_input == cached.tokens_output == cached.latency_ms == cached.cost_usd == 0
    assert result.tokens_input == 100
    assert cached.latency_kind == "cache_reuse" and result.latency_kind == "unknown"
    record, size = bounded_record({"prompt": "q", "raw_output": "x" * 50000, "agent_trace": {"steps": ["x" * 50000]}})
    assert len(record["raw_output"].encode()) == 8192
    assert size < 65536 and record["run_metadata"]["retention_truncated"]["raw_output"]["original_bytes"] == 50000
    assert record["agent_trace"]["retention_truncated"]
    usage = MetricsService(None)._compute_cost([
        SimpleNamespace(tokens_input=100, tokens_output=10, latency_ms=350, cost_usd=.2, is_correct=True, run_metadata={"cost_source": "pricing_table_estimate"}),
        SimpleNamespace(tokens_input=0, tokens_output=0, latency_ms=0, cost_usd=0, is_correct=True, run_metadata={"cache_hit": True}),
    ])
    assert usage["total_cost_usd"] == .2 and usage["total_tokens"] == 110
    assert usage["cache_hits"] == 1 and usage["provider_calls"] == 1
    assert usage["gpu_time_seconds"] is None


def test_unknown_primary_scores_are_not_counted_as_wrong_pairs():
    a, b = rows(2), rows(2)
    for row in b:
        row.is_correct, row.score = None, None
    stats = StatisticalService.compare_run_sets(a, b)
    assert stats["accuracy_b"] is None and stats["mcnemar"]["p_value"] is None
    assert stats["paired_scored_count"] == 0
    assert sum(stats["summary"].values()) == 0


@pytest.mark.asyncio
async def test_project_retention_cap_and_pinned_report_references(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "attempt-retention")
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    baseline = await create_benchmark(client, headers)
    await execute(sessions, baseline)
    async with sessions() as db:
        await RegressionService(db).pin_baseline(baseline)
        await db.commit()
    candidate = await create_benchmark(client, headers)
    await execute(sessions, candidate)
    assert (await client.delete(f"/api/v1/results/{baseline}/attempts/1?exported=true", headers=headers)).status_code == 409
    async with sessions() as db:
        await RegressionService(db).unpin_baseline(baseline)
        await db.commit()
    # The stored regression still references baseline attempt 1 after unpinning.
    assert (await client.delete(f"/api/v1/results/{baseline}/attempts/1?exported=true", headers=headers)).status_code == 409
    monkeypatch.setattr("app.services.benchmark_retention.MAX_PROJECT_ATTEMPTS", 2)
    dispatch = AsyncMock()
    monkeypatch.setattr("app.core.task_dispatch.dispatch_experiment", dispatch)
    draft = await create_benchmark(client, headers)
    blocked = await client.post(f"/api/v1/experiments/{draft}/run", headers=headers)
    assert blocked.status_code == 409 and "retention" in blocked.text
    dispatch.assert_not_awaited()
    assert (await client.delete(f"/api/v1/results/{candidate}/attempts/1?exported=true", headers=headers)).status_code == 200
    async with sessions() as db:
        service = ExperimentService(db)
        assert await service.queue_for_execution(draft) == 1
        await db.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_pin_blocks_purge_until_reference_is_committed(api, monkeypatch):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "pin-purge-race")
    monkeypatch.setattr("app.core.database.async_session_maker", sessions)
    identity = await create_benchmark(client, headers)
    await execute(sessions, identity)
    from app.services.attempts import lock_history_mutation
    async with sessions() as db:
        await lock_history_mutation(db)
        purge = asyncio.create_task(client.delete(f"/api/v1/results/{identity}/attempts/1?exported=true", headers=headers))
        try:
            await asyncio.sleep(.05)
            assert not purge.done()
            await RegressionService(db).pin_baseline(identity)
            await db.commit()
            response = await asyncio.wait_for(purge, 5)
            assert response.status_code == 409
        finally:
            if not purge.done():
                purge.cancel()
                await asyncio.gather(purge, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_runner_reconciles_once_then_waits_without_database_polling(monkeypatch):
    from app.services.evaluation_runner import EvaluationRunner
    reconcile = AsyncMock(return_value=[])
    monkeypatch.setattr("app.services.evaluation_runner.repository.reconcile", reconcile)
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr("app.services.evaluation_runner.async_session_maker", Session)
    runner = EvaluationRunner()
    runner.tick = AsyncMock()
    await runner.start()
    await asyncio.sleep(.02)
    assert runner.health() == "idle"
    reconcile.assert_awaited_once()
    runner.tick.assert_not_awaited()
    runner.wakeup.set()
    await asyncio.sleep(.02)
    runner.tick.assert_awaited_once()
    await asyncio.sleep(.02)
    runner.tick.assert_awaited_once()
    await runner.stop()


def test_database_urls_fail_closed_and_preserve_explicit_options():
    from app.core.database_url import async_database_options
    with pytest.raises(ValueError, match="required"):
        async_database_options("", production=True)
    with pytest.raises(ValueError, match="TLS"):
        async_database_options("postgresql://a:b@host/db?sslmode=disable", production=True)
    url, args = async_database_options("postgresql://a:b@host/db?sslmode=verify-full&application_name=forge&prepared_statement_cache_size=0", production=True)
    assert args["ssl"].check_hostname and args["server_settings"] == {"application_name": "forge"}
    assert url.query["prepared_statement_cache_size"] == "0"
    with pytest.raises(ValueError, match="channel_binding"):
        async_database_options("postgresql://a:b@host/db?channel_binding=require")
