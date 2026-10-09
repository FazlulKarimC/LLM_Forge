import json
import logging
from uuid import UUID

import pytest
from sqlalchemy import event, select

from app.core.diagnostics import PlatformFormatter, timed_stage
from app.models.evaluation_execution import EvaluationCheckpoint
from tests.factories import create_evaluation_scenario


def test_timing_spans_are_non_secret_on_success_and_interruption():
    from app.core.diagnostics import logger

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(json.loads(PlatformFormatter().format(record)))
    logger.addHandler(handler)
    try:
        with timed_stage("checkpoint", run_id="fixture", api_key="secret", output="private"):
            pass
        with pytest.raises(RuntimeError):
            with timed_stage("scoring", api_key="secret"):
                raise RuntimeError("private provider exception")
    finally:
        logger.removeHandler(handler)
    assert all(row["duration_ms"] >= 0 for row in records)
    assert [row["outcome"] for row in records] == ["completed", "interrupted"]
    assert "secret" not in json.dumps(records) and "private" not in json.dumps(records)


@pytest.mark.asyncio
async def test_saved_builtin_has_no_empty_inline_checkpoint_and_keeps_durable_evidence(api):
    _, client, sessions, headers, _, data = await create_evaluation_scenario(api)
    evaluator = (await client.post("/api/v1/evaluators", headers=headers, json={
        "name": "Saved exact match", "definition": {"kind": "builtin",
        "assertion": {"kind": "exact_match"}, "outputs": [{"name": "correct"}]},
    })).json()
    version = evaluator["latest"]["id"]
    commits = []
    engine = sessions.kw["bind"].sync_engine
    listener = lambda _: commits.append(1)
    event.listen(engine, "commit", listener)
    try:
        run = (await client.post("/api/v1/evaluations", headers=headers, json={
            **data, "assertions": [], "evaluators": [{"version_id": version}],
        })).json()
    finally:
        event.remove(engine, "commit", listener)
    async with sessions() as db:
        checkpoints = (await db.scalars(select(EvaluationCheckpoint).where(
            EvaluationCheckpoint.run_id == UUID(run["id"])
        ))).all()
    assert {row.stage for row in checkpoints} == {"generation", version}
    assert len(checkpoints) == run["total"] * 2
    # Admission/claim plus three durable writes per case; no empty policy writes.
    assert len(commits) == 2 + run["total"] * 3
    summary = (await client.get(f'/api/v1/evaluations/{run["id"]}/summary', headers=headers)).json()
    assert summary["run"]["status"] == "completed"
    # The shared fixture deliberately contains one matching and one wrong reference.
    assert summary["run"]["passed"] == 1
    assert summary["run"]["errors"] == 0
    assert summary["score_summary"][0]["count"] == run["total"]


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_saved_builtin_write_budget_on_postgres(api):
    await test_saved_builtin_has_no_empty_inline_checkpoint_and_keeps_durable_evidence(api)
