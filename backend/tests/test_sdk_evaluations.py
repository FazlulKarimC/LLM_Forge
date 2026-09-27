import pytest
from app.core.auth import get_current_user

from tests.factories import CASES, create_evaluation_scenario
from tests.support import bootstrap, login


@pytest.mark.asyncio
async def test_explicit_key_scope_and_revoke(api, monkeypatch):
    app, client, _, headers, dataset, request = await create_evaluation_scenario(api)
    read_key = (
        await client.post(
            "/api/v1/project-keys", headers=headers, json={"name": "Read only"}
        )
    ).json()
    assert read_key["scopes"] == ["prompts:read"]
    read_headers = {"Authorization": "Bearer " + read_key["secret"]}
    assert (
        await client.post(
            "/api/v1/sdk/evaluations/runs", headers=read_headers, json=request
        )
    ).status_code == 403
    assert (
        await client.get(
            "/api/v1/sdk/evaluations/datasets/Greetings", headers=read_headers
        )
    ).status_code == 403
    key = (
        await client.post(
            "/api/v1/project-keys",
            headers=headers,
            json={"name": "CI", "scopes": ["prompts:read", "evaluations:write"]},
        )
    ).json()
    sdk_headers = {"Authorization": "Bearer " + key["secret"]}
    data = await client.get(
        "/api/v1/sdk/evaluations/datasets/Greetings?version=1", headers=sdk_headers
    )
    assert data.status_code == 200 and data.json()["revision"]["cases"] == CASES
    response = await client.post(
        "/api/v1/sdk/evaluations/runs", headers=sdk_headers, json=request
    )
    assert response.status_code == 202
    run_id = response.json()["id"]
    detail = await client.get(
        f"/api/v1/sdk/evaluations/runs/{run_id}", headers=sdk_headers
    )
    assert detail.json()["run"]["passed"] == 1
    # SDK keys cannot authorize dashboard edits even when evaluation-enabled.
    app.dependency_overrides.pop(get_current_user)
    assert (
        await client.post(
            "/api/v1/datasets",
            headers=sdk_headers,
            json={"name": "Forbidden", "cases": CASES},
        )
    ).status_code == 401
    login(app, "user_alice")
    await client.delete("/api/v1/project-keys/" + key["id"], headers=headers)
    assert (
        await client.get(f"/api/v1/sdk/evaluations/runs/{run_id}", headers=sdk_headers)
    ).status_code == 401


@pytest.mark.asyncio
async def test_external_submission_counts_and_tenant_isolation(api, monkeypatch):
    app, client, _, headers, dataset, request = await create_evaluation_scenario(api)
    key = (
        await client.post(
            "/api/v1/project-keys",
            headers=headers,
            json={"name": "CI", "scopes": ["evaluations:write"]},
        )
    ).json()
    sdk_headers = {"Authorization": "Bearer " + key["secret"]}
    body = {
        "prompt_version_id": request["prompt_version_id"],
        "dataset_revision_id": request["dataset_revision_id"],
        "model": "own-app",
        "metrics": {"accuracy": 0.5},
        "results": [
            {
                "case_index": 0,
                "output": "Hello",
                "checks": [{"name": "correct", "passed": True, "score": 1}],
            },
            {"case_index": 1, "error": "Provider unavailable"},
        ],
    }
    submitted = await client.post(
        "/api/v1/sdk/evaluations/submissions", headers=sdk_headers, json=body
    )
    assert submitted.status_code == 201
    assert submitted.json()["passed"] == 1 and submitted.json()["errors"] == 1
    run_id = submitted.json()["id"]
    detail = (await client.get(f"/api/v1/evaluations/{run_id}", headers=headers)).json()
    assert detail["run"]["config"]["source"] == "sdk_submission"
    assert detail["run"]["config"]["metrics"] == {"accuracy": 0.5}
    assert detail["results"][0]["checks"][0]["kind"] == "external:correct"
    assert (
        await client.post(
            "/api/v1/sdk/evaluations/submissions",
            headers=sdk_headers,
            json={**body, "results": body["results"][:1]},
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/sdk/evaluations/submissions",
            headers=sdk_headers,
            json={**body, "results": [body["results"][0]] * 2},
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/sdk/evaluations/submissions",
            headers=sdk_headers,
            json={**body, "metrics": {"": 1}},
        )
    ).status_code == 422
    _, other = await bootstrap(app, client, "user_bob")
    other_key = (
        await client.post(
            "/api/v1/project-keys",
            headers=other,
            json={"name": "Other CI", "scopes": ["evaluations:write"]},
        )
    ).json()
    other_sdk = {"Authorization": "Bearer " + other_key["secret"]}
    assert (
        await client.get(f"/api/v1/sdk/evaluations/runs/{run_id}", headers=other_sdk)
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/sdk/evaluations/runs/{run_id}/cancel", headers=other_sdk
        )
    ).status_code == 404
    assert (
        await client.post(
            "/api/v1/sdk/evaluations/submissions", headers=other_sdk, json=body
        )
    ).status_code == 404
