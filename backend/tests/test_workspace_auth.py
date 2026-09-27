"""Signature validation and API isolation tests; no live Clerk/LLM required."""

import time
from uuid import UUID, uuid4

import jwt
import pytest
from app.core.auth import verify_session_token
from app.core.config import settings
from app.models.background_job import BackgroundJobRecord
from app.models.experiment import Experiment
from app.models.result import Result
from app.models.run import Run
from app.models.workspace import OrganizationMembership
from app.schemas.experiment import ExperimentStatus
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import select

from tests.support import bootstrap, login


@pytest.fixture
def signed_token(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode()
    )
    monkeypatch.setattr(settings, "CLERK_ISSUER_URL", "https://test.clerk.accounts.dev")
    monkeypatch.setattr(settings, "CLERK_JWT_PUBLIC_KEY", public)
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "http://localhost:3000")
    monkeypatch.setattr(settings, "CLERK_AUDIENCE", "")

    def make(**overrides):
        now = int(time.time())
        claims = {
            "sub": "user_alice",
            "sid": "sess_123",
            "iss": settings.CLERK_ISSUER_URL,
            "azp": "http://localhost:3000",
            "iat": now,
            "nbf": now,
            "exp": now + 60,
        }
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256")

    return make


def test_valid_session(signed_token):
    assert verify_session_token(signed_token()).subject == "user_alice"


def test_project_key_is_rejected_without_clerk_configuration(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "CLERK_ISSUER_URL", "")
    monkeypatch.setattr(settings, "CLERK_JWT_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "")
    with pytest.raises(HTTPException) as caught:
        verify_session_token("lf_live_test_key")
    assert caught.value.status_code == 401
    assert caught.value.headers == {"WWW-Authenticate": "Bearer"}


def test_session_auth_reports_missing_clerk_configuration(signed_token, monkeypatch):
    from fastapi import HTTPException

    token = signed_token()
    monkeypatch.setattr(settings, "CLERK_ISSUER_URL", "")
    with pytest.raises(HTTPException) as caught:
        verify_session_token(token)
    assert caught.value.status_code == 503


@pytest.mark.parametrize(
    "claims",
    [
        {"exp": 1},
        {"iss": "https://other.clerk.accounts.dev"},
        {"azp": "https://evil.example"},
        {"sid": None},
        {"sub": "machine_123"},
        {"nbf": int(time.time()) + 3600},
    ],
)
def test_invalid_sessions_are_rejected(signed_token, claims):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        verify_session_token(signed_token(**claims))
    assert caught.value.status_code == 401


def test_forged_signature_is_rejected(signed_token):
    from fastapi import HTTPException

    valid = signed_token()
    claims = jwt.decode(valid, options={"verify_signature": False})
    forged = jwt.encode(
        claims,
        rsa.generate_private_key(public_exponent=65537, key_size=2048),
        algorithm="RS256",
    )
    with pytest.raises(HTTPException) as caught:
        verify_session_token(forged)
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_anonymous_api_is_protected_and_health_public(api):
    _app, client, _sessions = api
    for url in [
        "/api/v1/workspaces",
        "/api/v1/experiments",
        "/api/v1/prompts",
        "/api/v1/results/jobs/unknown",
    ]:
        assert (await client.get(url)).status_code == 401
    assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_provisioning_is_idempotent_and_project_header_required(api):
    app, client, _sessions = api
    org, headers = await bootstrap(app, client, "user_alice")
    again = (await client.get("/api/v1/workspaces")).json()
    assert len(again["organizations"]) == 1
    assert again["organizations"][0]["id"] == org["id"]
    assert (await client.get("/api/v1/experiments")).status_code == 400
    assert (await client.get("/api/v1/experiments", headers=headers)).json()[
        "total"
    ] == 0


@pytest.mark.asyncio
async def test_two_users_and_same_user_projects_are_isolated(api):
    app, client, sessions = api
    org_a, headers_a = await bootstrap(app, client, "user_alice")
    payload = {
        "name": "Alice run",
        "config": {
            "model_name": "mock-model",
            "provider": "custom",
            "reasoning_method": "naive",
            "dataset_name": "sample",
            "num_samples": 1,
        },
    }
    response = await client.post("/api/v1/experiments", headers=headers_a, json=payload)
    assert response.status_code == 201, response.text
    experiment_id = response.json()["id"]
    prompt = await client.post(
        "/api/v1/prompts",
        headers=headers_a,
        json={"name": "alice-prompt", "template_text": "Hello {question}"},
    )
    assert prompt.status_code == 201, prompt.text
    prompt_id = prompt.json()["id"]
    async with sessions() as db:
        experiment = (await db.execute(select(Experiment))).scalar_one()
        experiment.status = ExperimentStatus.COMPLETED
        db.add(Result(experiment_id=experiment.id, accuracy_exact=1, raw_metrics={}))
        db.add(
            Run(
                experiment_id=experiment.id,
                example_id="1",
                prompt="secret",
                raw_output="secret result",
                attempt=1,
            )
        )
        db.add(
            BackgroundJobRecord(
                job_id="a" * 32,
                project_id=experiment.project_id,
                kind="judge",
                status="completed",
                result={"secret": True},
            )
        )
        await db.commit()
    assert (await client.get("/api/v1/experiments", headers=headers_a)).json()[
        "total"
    ] == 1
    assert (
        await client.get(f"/api/v1/results/{experiment_id}/export", headers=headers_a)
    ).status_code == 200
    _org_b, headers_b = await bootstrap(app, client, "user_bob")
    assert (await client.get("/api/v1/experiments", headers=headers_b)).json()[
        "total"
    ] == 0
    assert (await client.get("/api/v1/prompts", headers=headers_b)).json() == []
    assert (await client.get("/api/v1/experiments/stats", headers=headers_b)).json()[
        "total"
    ] == 0
    assert (
        await client.get("/api/v1/experiments", headers=headers_a)
    ).status_code == 404
    for path in [
        f"/experiments/{experiment_id}",
        f"/prompts/{prompt_id}",
        f"/results/{experiment_id}/metrics",
        f"/results/{experiment_id}/runs",
        f"/results/{experiment_id}/export",
        "/results/jobs/" + "a" * 32,
    ]:
        assert (
            await client.get("/api/v1" + path, headers=headers_b)
        ).status_code == 404, path
    assert (
        await client.delete(f"/api/v1/experiments/{experiment_id}", headers=headers_b)
    ).status_code == 404
    assert (
        await client.post(f"/api/v1/experiments/{experiment_id}/run", headers=headers_b)
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/experiments/{experiment_id}/set-baseline", headers=headers_b
        )
    ).status_code in (400, 404)
    assert (
        await client.post(
            "/api/v1/prompts",
            headers=headers_b,
            json={"name": "stolen", "template_text": "changed", "parent_id": prompt_id},
        )
    ).status_code == 404
    payload["config"]["prompt_version_id"] = prompt_id
    assert (
        await client.post("/api/v1/experiments", headers=headers_b, json=payload)
    ).status_code == 400
    assert (
        await client.get(
            "/api/v1/results/compare",
            headers=headers_b,
            params=[
                ("experiment_ids", experiment_id),
                ("experiment_ids", str(uuid4())),
            ],
        )
    ).status_code == 404
    assert (
        await client.get(
            "/api/v1/results/compare/statistical",
            headers=headers_b,
            params={"experiment_a": experiment_id, "experiment_b": str(uuid4())},
        )
    ).status_code == 404
    login(app, "user_alice")
    project_response = await client.post(
        f"/api/v1/workspaces/organizations/{org_a['id']}/projects",
        json={"name": "Second project"},
    )
    assert project_response.status_code == 201
    second_headers = {"X-Project-ID": project_response.json()["id"]}
    assert (await client.get("/api/v1/experiments", headers=second_headers)).json()[
        "total"
    ] == 0
    assert (
        await client.get(f"/api/v1/experiments/{experiment_id}", headers=second_headers)
    ).status_code == 404
    assert (
        await client.get(f"/api/v1/experiments/{experiment_id}", headers=headers_a)
    ).status_code == 200


@pytest.mark.asyncio
async def test_organization_creation_and_owner_permissions(api):
    app, client, sessions = api
    org_a, headers_a = await bootstrap(app, client, "user_alice")
    assert (
        await client.post("/api/v1/workspaces/organizations", json={"name": "   "})
    ).status_code == 422
    response = await client.post(
        "/api/v1/workspaces/organizations", json={"name": "Demo team"}
    )
    assert response.status_code == 201
    snapshot = (await client.get("/api/v1/workspaces")).json()
    assert len(snapshot["organizations"]) == 2
    _org_b, _headers_b = await bootstrap(app, client, "user_bob")
    assert (
        await client.post(
            f"/api/v1/workspaces/organizations/{org_a['id']}/projects",
            json={"name": "Intrusion"},
        )
    ).status_code == 404
    async with sessions() as db:
        from app.models.workspace import User

        alice_id = (
            await db.execute(select(User.id).where(User.auth_subject == "user_alice"))
        ).scalar_one()
        membership = (
            await db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == alice_id,
                    OrganizationMembership.organization_id == UUID(org_a["id"]),
                )
            )
        ).scalar_one()
        membership.role = "member"
        await db.commit()
    login(app, "user_alice")
    assert (
        await client.get("/api/v1/experiments", headers=headers_a)
    ).status_code == 200
    assert (
        await client.post(
            f"/api/v1/workspaces/organizations/{org_a['id']}/projects",
            json={"name": "Forbidden"},
        )
    ).status_code == 403
