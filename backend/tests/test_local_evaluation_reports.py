"""Current local reporting requires no hosted cases and remains tenant scoped."""
import pytest
from sqlalchemy import select, func
from app.models.evaluation import Dataset, DatasetRevision, EvaluationResult
from tests.support import bootstrap

async def setup(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "user_local_reports")
    prompt = (await client.post("/api/v1/prompt-library", headers=headers, json={"name": "Echo", "template_text": "{{query}}"})).json()
    key = (await client.post("/api/v1/project-keys", headers=headers, json={"name": "CI", "scopes": ["evaluations:write"]})).json()
    payload = {"prompt_version_id": prompt["version"]["id"], "dataset_name": "Private local tests",
               "dataset_fingerprint": "sha256:" + "a"*64, "scoring_fingerprint": "sha256:" + "b"*64,
               "model": "application", "total": 1000, "completed": 1000, "passed": 990, "errors": 2}
    return client, sessions, headers, {"Authorization": "Bearer " + key["secret"]}, payload

@pytest.mark.asyncio
async def test_local_summary_without_cases_replay_and_history(api):
    client, sessions, headers, key, payload = await setup(api)
    token = {**key, "Idempotency-Key": "one-local-run"}
    first = await client.post("/api/v1/sdk/evaluations/local-reports", headers=token, json=payload)
    assert first.status_code == 201, first.text
    run = first.json()
    assert run["dataset_revision_id"] is None and run["passed"] == 990
    replay = await client.post("/api/v1/sdk/evaluations/local-reports", headers=token, json=payload)
    assert replay.json()["id"] == run["id"]
    changed = await client.post("/api/v1/sdk/evaluations/local-reports", headers=token, json={**payload, "passed": 1})
    assert changed.status_code == 409
    summary = (await client.get(f'/api/v1/evaluations/{run["id"]}/summary', headers=headers)).json()
    assert summary["recovery"]["available"] is False
    assert summary["recovery"]["remaining_calls"]["total"] == 0
    assert summary["results"] == [] and summary["score_summary"] == []
    cases = await client.get(f'/api/v1/evaluations/{run["id"]}/cases', headers=headers)
    assert cases.json()["items"] == []
    assert (await client.get(f'/api/v1/evaluations/{run["id"]}/cases?filter=failed', headers=headers)).status_code == 409
    assert (await client.get(f'/api/v1/evaluations/{run["id"]}/cases/0', headers=headers)).status_code == 404
    assert (await client.post(f'/api/v1/evaluations/{run["id"]}/resume', headers=headers, json={})).status_code == 409
    async with sessions() as db:
        for model in (Dataset, DatasetRevision, EvaluationResult):
            assert await db.scalar(select(func.count()).select_from(model)) == 0

@pytest.mark.asyncio
async def test_local_report_scopes_and_foreign_prompt(api):
    client, _, headers, key, payload = await setup(api)
    read_key = (await client.post("/api/v1/project-keys", headers=headers, json={"name": "Read"})).json()
    assert (await client.post("/api/v1/sdk/evaluations/local-reports", headers={"Authorization": "Bearer " + read_key["secret"]}, json=payload)).status_code == 403
    app = api[0]
    _, other = await bootstrap(app, client, "user_elsewhere")
    other_key = (await client.post("/api/v1/project-keys", headers=other, json={"name": "Other", "scopes": ["evaluations:write"]})).json()
    assert (await client.post("/api/v1/sdk/evaluations/local-reports", headers={"Authorization": "Bearer " + other_key["secret"]}, json=payload)).status_code == 404
    run = (await client.post("/api/v1/sdk/evaluations/local-reports", headers=key, json=payload)).json()
    assert (await client.get(f'/api/v1/evaluations/{run["id"]}/summary', headers=other)).status_code == 404

@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"total": 0}, {"total": True}, {"passed": 9999}, {"completed": 999},
    {"dataset_fingerprint": "bad"}, {"scoring_fingerprint": "bad"},
    {"metrics": {"bad": float("inf")}}, {"results": [{"inputs": {"query": "private"}}]},
    {"api_key": "provider-secret"},
])
async def test_invalid_local_reports_rejected(api, changes):
    client, _, _, key, payload = await setup(api)
    import json
    response = await client.post("/api/v1/sdk/evaluations/local-reports", headers={**key, "Content-Type": "application/json"}, content=json.dumps({**payload, **changes}))
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_reference_filter_requires_matching_data_and_scoring(api):
    client, _, headers, key, payload = await setup(api)
    runs = []
    for changes in ({}, {"dataset_fingerprint": "sha256:" + "c"*64}, {"scoring_fingerprint": "sha256:" + "d"*64}):
        runs.append((await client.post("/api/v1/sdk/evaluations/local-reports", headers=key, json={**payload, **changes})).json())
    matches = (await client.get("/api/v1/evaluations", headers=headers, params={"reference_run_id": runs[0]["id"], "status": "completed"})).json()
    assert [r["id"] for r in matches["items"]] == [runs[0]["id"]]

@pytest.mark.asyncio
async def test_dataset_authoring_disabled_by_default(api, monkeypatch):
    client, _, headers, _, _ = await setup(api)
    monkeypatch.setattr("app.core.config.settings.LEGACY_DATASET_WRITES", False)
    for path, body in [("datasets", {"name": "No", "cases": [{"inputs": {"query": "x"}}]}),
                       ("datasets/import", {"format": "json", "content": "[]"}), ("demo/setup", {})]:
        assert (await client.post("/api/v1/" + path, headers=headers, json=body)).status_code == 410
