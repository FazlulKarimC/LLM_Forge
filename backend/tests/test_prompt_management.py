"""Prompt management parity: roles, immutable config, selectors and scoped writes."""
import pytest
from app.services.inference.base import GenerationResult
from tests.support import bootstrap


async def create(client, headers, **changes):
    response = await client.post("/api/v1/prompt-library", headers=headers, json={"name": "support/answer", "template_text": "Hello {{query}}", **changes})
    assert response.status_code == 201, response.text
    return response.json()


async def key(client, headers, scopes=None):
    response = await client.post("/api/v1/project-keys", headers=headers, json={"name": "test", **({"scopes": scopes} if scopes else {})})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["secret"]}


@pytest.mark.asyncio
async def test_config_only_versions_and_canonical_duplicate_detection(api):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_config")
    first = await create(client, headers, config={"temperature": 0, "model": "one"})
    path = f"/api/v1/prompt-library/{first['prompt']['id']}"
    second = await client.post(path + "/versions", headers=headers, json={"template_text": "Hello {{query}}", "config": {"model": "two", "temperature": 0}, "base_version": 1})
    assert second.status_code == 201, second.text
    assert second.json()["sha256_hash"] != first["version"]["sha256_hash"]
    duplicate = await client.post(path + "/versions", headers=headers, json={"template_text": "Hello {{query}}", "config": {"temperature": 0, "model": "two"}})
    assert duplicate.status_code == 409
    versions = (await client.get(path + "/versions", headers=headers)).json()
    assert versions[1]["config"] == {"temperature": 0, "model": "one"}
    assert versions[1]["template_text"] == "Hello {{query}}"


@pytest.mark.asyncio
async def test_chat_roles_compile_and_provider_request(api, monkeypatch):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_chat")
    messages = [{"role": "system", "content": "Respond in {{language}}"}, {"role": "user", "content": "{{query}}"}]
    draft = {"prompt_type": "chat", "template_text": "", "messages": messages}
    created = await create(client, headers, **draft)
    assert created["version"]["variables"] == ["language", "query"]
    assert created["prompt"]["prompt_type"] == "chat"
    variables = {"language": "English", "query": "{{literal}}"}
    compiled = await client.post("/api/v1/prompt-library/compile", headers=headers, json={**draft, "variables": variables})
    expected = [{"role": "system", "content": "Respond in English"}, {"role": "user", "content": "{{literal}}"}]
    assert compiled.json()["compiled_messages"] == expected
    assert (await client.post("/api/v1/prompt-library/compile", headers=headers, json={**draft, "variables": {"query": "hi"}})).status_code == 422

    async def generate(_self, prompt, _config):
        assert prompt == expected  # Never flatten system/user roles into a user string.
        return GenerationResult("hello", 5, 1, 0.5, "stop")
    monkeypatch.setattr("app.services.inference.openai_engine.OpenAIEngine.generate_async", generate)
    response = await client.post("/api/v1/prompt-library/playground", headers=headers, json={**draft, "variables": variables, "provider": "openai", "api_key": "fake-test-only"})
    assert response.status_code == 200, response.text
    assert response.json()["compiled_messages"] == expected
    changed_type = await client.post(f"/api/v1/prompt-library/{created['prompt']['id']}/versions", headers=headers, json={"template_text": "Text"})
    assert changed_type.status_code == 422


@pytest.mark.asyncio
async def test_chat_evaluation_keeps_roles_and_missing_variables_are_rejected(api):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_chat_eval")
    created = await create(client, headers, prompt_type="chat", template_text="", messages=[{"role": "system", "content": "Echo"}, {"role": "user", "content": "{{query}}"}])
    dataset = await client.post("/api/v1/datasets", headers=headers, json={"name": "Chat cases", "cases": [{"inputs": {"query": "hi"}, "expected_output": "[system]\nEcho\n\n[user]\nhi"}]})
    assert dataset.status_code == 201, dataset.text
    run = await client.post("/api/v1/evaluations", headers=headers, json={"prompt_version_id": created["version"]["id"], "dataset_revision_id": dataset.json()["revision"]["id"], "provider": "mock", "assertions": [{"kind": "exact_match"}]})
    assert run.status_code == 202, run.text
    result = (await client.get("/api/v1/evaluations/" + run.json()["id"], headers=headers)).json()
    assert result["run"]["passed"] == 1
    assert result["results"][0]["output"] == "[system]\nEcho\n\n[user]\nhi"


@pytest.mark.asyncio
async def test_sdk_writes_labels_tags_and_project_isolation(api):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_writer")
    reader = await key(client, headers)
    payload = {"name": "support/answer", "template_text": "v1 {{query}}", "tags": ["support"], "labels": ["production", "experiment-a"]}
    assert (await client.post("/api/v1/sdk/prompts", headers=reader, json=payload)).status_code == 403
    writer = await key(client, headers, ["prompts:read", "prompts:write"])
    first = await client.post("/api/v1/sdk/prompts", headers=writer, json=payload)
    assert first.status_code == 201, first.text
    assert first.json()["created_by"] == "API"
    second = await client.post("/api/v1/sdk/prompts", headers=writer, json={"name": payload["name"], "template_text": "v2 {{query}}", "base_version": 1})
    assert second.status_code == 201, second.text
    sdk_path = "/api/v1/sdk/prompts/support/answer"
    assert (await client.get(sdk_path, headers=reader)).json()["version"] == 1
    assert (await client.get(sdk_path, headers=reader, params={"label": "latest"})).json()["version"] == 2
    assert (await client.get(sdk_path, headers=reader, params={"label": "experiment-a"})).json()["version"] == 1
    assert (await client.get(sdk_path, headers=reader, params={"label": "typo"})).status_code == 404
    assert (await client.get(sdk_path, headers=reader, params={"label": "latest", "version": 1})).status_code == 422
    stale = await client.post("/api/v1/sdk/prompts", headers=writer, json={"name": payload["name"], "template_text": "stale", "base_version": 1})
    assert stale.status_code == 409
    promoted = await client.put(sdk_path + "/labels/experiment-a", headers=writer, json={"version": 2})
    assert promoted.status_code == 200, promoted.text
    assert (await client.put(sdk_path + "/labels/latest", headers=writer, json={"version": 1})).status_code == 422
    assert (await client.put(sdk_path + "/labels/production", headers=reader, json={"version": 2})).status_code == 403
    listed = await client.get("/api/v1/sdk/prompts", headers=reader, params={"tag": "support", "label": "experiment-a"})
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1 and listed.json()["items"][0]["tags"] == ["support"]
    # The SDK scope comes from the key, never a caller's project header.
    _, other = await bootstrap(app, client, "user_other")
    other_reader = await key(client, other)
    assert (await client.get(sdk_path, headers={**other_reader, **headers})).status_code == 404
    assert (await client.get("/api/v1/sdk/prompts", headers=other_reader)).json()["total"] == 0


@pytest.mark.asyncio
async def test_catalog_filters_before_pagination_and_latest_is_read_only(api):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_tags")
    first = await create(client, headers, name="support/answer", tags=["support", "support"])
    await create(client, headers, name="support/classify", tags=["support"])
    await create(client, headers, name="support_other/answer", tags=["other"])
    listed = (await client.get("/api/v1/prompt-library", headers=headers, params={"folder": "support", "tag": "support", "limit": 1})).json()
    assert listed["total"] == 2 and len(listed["items"]) == 1
    assert any(label["label"] == "latest" for label in listed["items"][0]["labels"])
    missing = (await client.get("/api/v1/prompt-library", headers=headers, params={"tag": "supp"})).json()
    assert missing["total"] == 0
    path = f"/api/v1/prompt-library/{first['prompt']['id']}/labels/latest"
    assert (await client.delete(path, headers=headers)).status_code == 422
    assert (await client.put(path, headers=headers, json={"version_id": first["version"]["id"]})).status_code == 422


@pytest.mark.parametrize("changes", [
    {"messages": [{"role": "user", "content": "mixed"}]},
    {"prompt_type": "chat", "template_text": "", "messages": []},
    {"prompt_type": "chat", "template_text": "", "messages": [{"role": "tool", "content": "no tool contract"}]},
    {"config": {"huge": "x" * 20_001}}, {"labels": ["latest"]}, {"labels": ["bad label"]},
    {"tags": [" "]}, {"name": "support/../answer"},
])
@pytest.mark.asyncio
async def test_prompt_contract_rejects_invalid_shapes(api, changes):
    app, client, _ = api
    _, headers = await bootstrap(app, client, "user_shapes")
    response = await client.post("/api/v1/prompt-library", headers=headers, json={"name": "support", "template_text": "Hello", **changes})
    assert response.status_code == 422, response.text
