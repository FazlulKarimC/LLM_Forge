import json
from typing import get_args

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import create_application
from app.schemas.evaluation import EvaluationProvider, JudgeProvider


@pytest.mark.asyncio
async def test_capabilities_match_generation_and_judge_contracts_without_database():
    app = create_application()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/capabilities")
    assert response.status_code == 200
    body = response.json()
    schema = app.openapi()["components"]["schemas"]
    assert body["generation_providers"] == list(get_args(EvaluationProvider)) == schema["EvaluationCreate"]["properties"]["provider"]["enum"]
    assert body["judge_providers"] == list(get_args(JudgeProvider)) == schema["JudgeConfig"]["properties"]["provider"]["enum"]
    assert body["external_submissions"] is True
    assert "gemini" not in body["generation_providers"]


@pytest.mark.asyncio
async def test_version_uses_upstream_identity_and_ignores_extra_build_fields(monkeypatch, tmp_path):
    from app.core import build_info

    path = tmp_path / "build.json"
    path.write_text(json.dumps({"commit": "a" * 40, "build_id": "github-42-1", "api_key": "never-expose"}))
    monkeypatch.setattr(build_info, "BUILD_INFO_PATH", path)
    monkeypatch.setattr(settings, "BUILD_COMMIT", "")
    monkeypatch.setattr(settings, "BUILD_ID", "")
    async with AsyncClient(transport=ASGITransport(app=create_application()), base_url="http://test") as client:
        body = (await client.get("/api/v1/version")).json()
    assert body == {"service": "llmforge-backend", "api_version": settings.VERSION, "commit": "a" * 40, "build_id": "github-42-1"}
    assert "never-expose" not in json.dumps(body)


def test_unknown_or_malformed_identity_is_explicit(monkeypatch, tmp_path):
    from app.core import build_info

    monkeypatch.setattr(build_info, "BUILD_INFO_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(settings, "BUILD_COMMIT", "")
    monkeypatch.setattr(settings, "BUILD_ID", "")
    assert build_info.build_identity() == {"commit": None, "build_id": None}
    monkeypatch.setattr(settings, "BUILD_COMMIT", "https://secret.example/token")
    monkeypatch.setattr(settings, "BUILD_ID", "key=secret")
    assert build_info.build_identity() == {"commit": None, "build_id": None}
