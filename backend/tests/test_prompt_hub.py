"""Prompt releases, SDK credentials, tenant boundaries and safe compilation."""

from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError
from app.core.auth import get_current_user
from app.core.project_keys import hash_key
from app.models.prompt import ProjectAPIKey
from app.models.workspace import OrganizationMembership
from app.services.experiment_runtime import VersionedPromptTemplate
from app.services.inference.base import GenerationResult
from app.services.prompt_templates import compile_template, template_variables
from app.schemas.prompt import PromptCreate
from openai import AuthenticationError
from sqlalchemy import select

from tests.support import bootstrap, login


@pytest.mark.parametrize(
    "template,format,values,expected",
    [
        ("Hello {{name}} / {{ name }}", "mustache", {"name": "Ada"}, "Hello Ada / Ada"),
        (
            'Return JSON {"answer": "{{query}}"}',
            "mustache",
            {"query": "yes"},
            'Return JSON {"answer": "yes"}',
        ),
        ("{{query}}", "mustache", {"query": "{{other}}"}, "{{other}}"),
        ("{{name}}", "mustache", {"name": "\\n$(){{attack}}"}, "\\n$(){{attack}}"),
        ("{{literal}} {query}", "fstring", {"query": "question"}, "{literal} question"),
        ("{{{query}}}", "fstring", {"query": "question"}, "{question}"),
        ("A static prompt", "mustache", {}, "A static prompt"),
    ],
)
def test_compilation(template, format, values, expected):
    assert compile_template(template, values, format) == expected


@pytest.mark.parametrize(
    "template,format",
    [
        ("{{unclosed", "mustache"),
        ("stray }}", "mustache"),
        ("{{user.secret}}", "mustache"),
        ("{{#loop}}", "mustache"),
        ("{{}}", "mustache"),
        ("{user.secret}", "fstring"),
        ("{query!r}", "fstring"),
        ("{query:1000000}", "fstring"),
        ("{}", "fstring"),
    ],
)
def test_expressions_and_malformed_placeholders_rejected(template, format):
    with pytest.raises(ValueError):
        template_variables(template, format)


def test_missing_variables_and_compiled_size_limit():
    with pytest.raises(ValueError, match="Missing variables: query"):
        compile_template("{{query}}", {})
    with pytest.raises(ValueError, match="100,000"):
        compile_template("{{query}}" * 11, {"query": "x" * 10_000})
    adapter = VersionedPromptTemplate(
        "Question: {{question}}", lambda text: text, "mustache"
    )
    assert adapter.format("Is this compiled?") == "Question: Is this compiled?"


@pytest.mark.parametrize("name", [".", ".."])
def test_reserved_prompt_names_are_rejected(name):
    with pytest.raises(ValidationError):
        PromptCreate(name=name, template_text="{{query}}")


async def new_prompt(client, headers, name="support-answer", text="Answer {{query}}"):
    response = await client.post(
        "/api/v1/prompt-library",
        headers=headers,
        json={"name": name, "template_text": text},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def new_key(client, headers):
    response = await client.post(
        "/api/v1/project-keys", headers=headers, json={"name": "SDK test"}
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
async def test_version_promotion_rollback_and_sdk_cache(api):
    app, client, sessions = api
    _org, headers = await bootstrap(app, client, "user_alice")
    created = await new_prompt(client, headers)
    prompt, first = created["prompt"], created["version"]
    assert first["variables"] == ["query"]
    assert prompt["latest_version"] == 1
    assert (await new_prompt(client, headers, name="another", text="Static"))[
        "version"
    ]["version"] == 1
    duplicate = await client.post(
        "/api/v1/prompt-library",
        headers=headers,
        json={"name": prompt["name"], "template_text": "Different"},
    )
    assert duplicate.status_code == 409
    second = await client.post(
        f"/api/v1/prompt-library/{prompt['id']}/versions",
        headers=headers,
        json={"template_text": "Answer carefully: {{query}}", "base_version": 1},
    )
    assert second.status_code == 201, second.text
    second = second.json()
    assert second["version"] == 2 and second["parent_id"] == first["id"]
    for body in [
        {"template_text": "stale", "base_version": 1},
        {"template_text": first["template_text"]},
    ]:
        assert (
            await client.post(
                f"/api/v1/prompt-library/{prompt['id']}/versions",
                headers=headers,
                json=body,
            )
        ).status_code == 409
    history = (
        await client.get(
            f"/api/v1/prompt-library/{prompt['id']}/versions", headers=headers
        )
    ).json()
    assert [version["version"] for version in history] == [2, 1]
    assert history[1]["template_text"] == first["template_text"]
    key = await new_key(client, headers)
    sdk = {"Authorization": "Bearer " + key["secret"]}
    assert (
        await client.get("/api/v1/sdk/prompts/support-answer", headers=sdk)
    ).status_code == 404
    for label, version in [("staging", second), ("production", first)]:
        response = await client.put(
            f"/api/v1/prompt-library/{prompt['id']}/labels/{label}",
            headers=headers,
            json={"version_id": version["id"]},
        )
        assert response.status_code == 200, response.text
    fetched = await client.get("/api/v1/sdk/prompts/support-answer", headers=sdk)
    assert fetched.status_code == 200 and fetched.json()["version"] == 1
    etag = fetched.headers["etag"]
    assert (
        await client.get(
            "/api/v1/sdk/prompts/support-answer", headers={**sdk, "If-None-Match": etag}
        )
    ).status_code == 304
    assert (
        await client.get(
            "/api/v1/sdk/prompts/support-answer?label=staging", headers=sdk
        )
    ).json()["version"] == 2
    assert (
        await client.get("/api/v1/sdk/prompts/support-answer?version=2", headers=sdk)
    ).json()["version"] == 2
    assert (
        await client.get(
            "/api/v1/sdk/prompts/support-answer?label=staging&version=2", headers=sdk
        )
    ).status_code == 422
    await client.put(
        f"/api/v1/prompt-library/{prompt['id']}/labels/production",
        headers=headers,
        json={"version_id": second["id"]},
    )
    changed = await client.get(
        "/api/v1/sdk/prompts/support-answer", headers={**sdk, "If-None-Match": etag}
    )
    assert changed.status_code == 200 and changed.json()["version"] == 2
    await client.put(
        f"/api/v1/prompt-library/{prompt['id']}/labels/production",
        headers=headers,
        json={"version_id": first["id"]},
    )
    assert (await client.get("/api/v1/sdk/prompts/support-answer", headers=sdk)).json()[
        "version"
    ] == 1
    renamed = await client.patch(
        f"/api/v1/prompt-library/{prompt['id']}",
        headers=headers,
        json={"name": "support-v2", "description": "Purpose"},
    )
    assert renamed.status_code == 200
    assert (
        await client.get("/api/v1/sdk/prompts/support-answer", headers=sdk)
    ).status_code == 404
    assert (await client.get("/api/v1/sdk/prompts/support-v2", headers=sdk)).json()[
        "name"
    ] == "support-v2"
    assert (
        await client.delete(f"/api/v1/prompt-library/{prompt['id']}", headers=headers)
    ).status_code == 204
    assert (
        await client.get("/api/v1/sdk/prompts/support-v2", headers=sdk)
    ).status_code == 404
    assert (await client.get("/api/v1/prompt-library", headers=headers)).json()[
        "total"
    ] == 1
    assert (
        await client.get("/api/v1/prompt-library?archived=true", headers=headers)
    ).json()["total"] == 1
    assert (
        await client.post(
            f"/api/v1/prompt-library/{prompt['id']}/versions",
            headers=headers,
            json={"template_text": "blocked"},
        )
    ).status_code == 409
    assert (await client.get(f"/api/v1/prompts/{first['id']}", headers=headers)).json()[
        "template_text"
    ] == first["template_text"]
    await client.patch(
        f"/api/v1/prompt-library/{prompt['id']}",
        headers=headers,
        json={"archived": False},
    )
    assert (
        await client.get("/api/v1/sdk/prompts/support-v2", headers=sdk)
    ).status_code == 200
    listed = (await client.get("/api/v1/project-keys", headers=headers)).json()
    assert "secret" not in listed[0] and "secret_hash" not in listed[0]
    async with sessions() as db:
        stored = (await db.execute(select(ProjectAPIKey))).scalar_one()
        assert (
            stored.secret_hash == hash_key(key["secret"])
            and stored.secret_hash != key["secret"]
        )
    assert (
        await client.delete(f"/api/v1/project-keys/{key['id']}", headers=headers)
    ).status_code == 204
    assert (
        await client.get("/api/v1/sdk/prompts/support-v2", headers=sdk)
    ).status_code == 401


@pytest.mark.asyncio
async def test_prompt_and_key_tenant_boundaries_and_owner_permissions(api):
    app, client, sessions = api
    org, alice = await bootstrap(app, client, "user_alice")
    a = await new_prompt(client, alice)
    a_key = await new_key(client, alice)
    _org, bob = await bootstrap(app, client, "user_bob")
    b = await new_prompt(client, bob)
    assert (await client.get("/api/v1/prompt-library", headers=bob)).json()[
        "total"
    ] == 1
    assert (await client.get("/api/v1/project-keys", headers=bob)).json() == []
    paths = [f"/{a['prompt']['id']}", f"/{a['prompt']['id']}/versions"]
    for path in paths:
        assert (
            await client.get("/api/v1/prompt-library" + path, headers=bob)
        ).status_code == 404
    assert (
        await client.put(
            f"/api/v1/prompt-library/{b['prompt']['id']}/labels/production",
            headers=bob,
            json={"version_id": a["version"]["id"]},
        )
    ).status_code == 404
    assert (
        await client.delete(f"/api/v1/project-keys/{a_key['id']}", headers=bob)
    ).status_code == 404
    # A read-only project key cannot authenticate dashboard mutations.
    app.dependency_overrides.pop(get_current_user)
    assert (
        await client.post(
            "/api/v1/prompt-library",
            headers={**alice, "Authorization": "Bearer " + a_key["secret"]},
            json={"name": "attack", "template_text": "text"},
        )
    ).status_code == 401
    login(app, "user_alice")
    async with sessions() as db:
        member = (
            await db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == UUID(org["id"])
                )
            )
        ).scalar_one()
        member.role = "member"
        await db.commit()
    assert (
        await client.post(
            "/api/v1/project-keys", headers=alice, json={"name": "Forbidden"}
        )
    ).status_code == 403
    assert (await client.get("/api/v1/project-keys", headers=alice)).status_code == 403


@pytest.mark.asyncio
async def test_playground_compile_mock_and_sanitized_provider_failures(
    api, monkeypatch
):
    app, client, _sessions = api
    _org, headers = await bootstrap(app, client, "user_alice")
    draft = {"template_text": "Hello {{query}}", "variables": {"query": "world"}}
    preview = await client.post(
        "/api/v1/prompt-library/compile", headers=headers, json=draft
    )
    assert preview.json()["compiled_prompt"] == "Hello world"
    assert (
        await client.post(
            "/api/v1/prompt-library/compile",
            headers=headers,
            json={"template_text": "{{query}}"},
        )
    ).status_code == 422
    demo = await client.post(
        "/api/v1/prompt-library/playground", headers=headers, json=draft
    )
    assert demo.status_code == 200 and demo.json()["is_mock"] is True
    assert demo.json()["tokens_input"] is None
    assert (
        await client.post(
            "/api/v1/prompt-library/playground",
            headers=headers,
            json={**draft, "provider": "groq"},
        )
    ).status_code == 422

    async def reject(_self, _prompt, _config):
        response = httpx.Response(
            401, request=httpx.Request("POST", "https://provider.example")
        )
        raise AuthenticationError("private-key-value", response=response, body={})

    monkeypatch.setattr(
        "app.services.inference.openai_engine.OpenAIEngine.generate_async", reject
    )
    failed = await client.post(
        "/api/v1/prompt-library/playground",
        headers=headers,
        json={**draft, "provider": "groq", "api_key": "private-key-value"},
    )
    assert failed.status_code == 502 and "private-key-value" not in failed.text

    async def success(_self, prompt, config):
        assert prompt == "Hello world" and config.temperature == 0
        return GenerationResult("Real mocked transport result", 2, 4, 7.5, "stop")

    monkeypatch.setattr(
        "app.services.inference.openai_engine.OpenAIEngine.generate_async", success
    )
    good = await client.post(
        "/api/v1/prompt-library/playground",
        headers=headers,
        json={
            **draft,
            "provider": "openrouter",
            "model": "example-model",
            "api_key": "not-persisted",
            "temperature": 0,
        },
    )
    assert good.status_code == 200 and good.json()["is_mock"] is False
    assert good.json()["tokens_output"] == 4
    assert (await client.get("/api/v1/prompt-library", headers=headers)).json()[
        "total"
    ] == 0
