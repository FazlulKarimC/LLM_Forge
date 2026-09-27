"""Reusable, checked API resource factories for integration tests."""

from tests.support import bootstrap

CASES = [
    {"inputs": {"query": "Hello"}, "expected_output": "Hello", "name": "Greeting"},
    {
        "inputs": {"query": "Goodbye"},
        "expected_output": "different",
        "name": "Farewell",
    },
]


async def create_evaluation_scenario(api):
    app, client, sessions = api
    _, headers = await bootstrap(app, client, "user_alice")
    prompt = (
        await client.post(
            "/api/v1/prompt-library",
            headers=headers,
            json={"name": "Echo", "template_text": "{{query}}"},
        )
    ).json()
    dataset = (
        await client.post(
            "/api/v1/datasets",
            headers=headers,
            json={"name": "Greetings", "cases": CASES},
        )
    ).json()
    request = {
        "prompt_version_id": prompt["version"]["id"],
        "dataset_revision_id": dataset["revision"]["id"],
        "assertions": [{"kind": "exact_match"}],
    }
    return app, client, sessions, headers, dataset, request
