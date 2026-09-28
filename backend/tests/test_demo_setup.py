"""The guided demo creates stable examples without provider credentials."""

import pytest

from tests.support import bootstrap


@pytest.mark.asyncio
async def test_demo_setup_is_idempotent_and_project_scoped(api):
    app, client, _sessions = api
    _, alice = await bootstrap(app, client, "user_demo_alice")
    first = await client.post("/api/v1/demo/setup", headers=alice)
    again = await client.post("/api/v1/demo/setup", headers=alice)
    assert first.status_code == again.status_code == 201
    assert first.json() == again.json()

    run = await client.post(
        "/api/v1/evaluations",
        headers=alice,
        json={
            "prompt_version_id": first.json()["prompt_version_id"],
            "dataset_revision_id": first.json()["dataset_revision_id"],
            "provider": "mock",
            "assertions": [{"kind": "exact_match"}],
        },
    )
    assert run.status_code == 202
    detail = await client.get(
        "/api/v1/evaluations/" + run.json()["id"], headers=alice
    )
    assert detail.json()["run"]["passed"] == 2

    _, bob = await bootstrap(app, client, "user_demo_bob")
    other = await client.post("/api/v1/demo/setup", headers=bob)
    assert other.status_code == 201
    assert other.json()["prompt_id"] != first.json()["prompt_id"]


@pytest.mark.asyncio
async def test_demo_setup_does_not_replace_existing_prompt(api):
    app, client, _sessions = api
    _, headers = await bootstrap(app, client, "user_demo_conflict")
    existing = await client.post(
        "/api/v1/prompt-library",
        headers=headers,
        json={"name": "Echo demo", "template_text": "Different {{query}}"},
    )
    assert existing.status_code == 201
    response = await client.post("/api/v1/demo/setup", headers=headers)
    assert response.status_code == 409
    datasets = await client.get("/api/v1/datasets", headers=headers)
    assert datasets.json()["total"] == 0
