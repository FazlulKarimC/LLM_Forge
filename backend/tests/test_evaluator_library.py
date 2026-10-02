"""Evaluator lifecycle, pinned provenance, typed scores and generation-free reruns."""

import json
from uuid import UUID

import pytest
from sqlalchemy import select
from fastapi import HTTPException

from app.models.evaluator import EvaluationScore
from app.models.evaluation import EvaluationRun
from app.schemas.evaluator import EvaluatorDefinition
from app.services.evaluator_service import execute_evaluator, call_budget
from tests.factories import create_evaluation_scenario
from tests.support import bootstrap


def builtin(kind="exact_match", value="", name="correctness"):
    return {
        "kind": "builtin",
        "assertion": {"kind": kind, "value": value},
        "outputs": [{"name": name, "data_type": "boolean"}],
    }


async def create(client, headers, name="Correctness", definition=None):
    response = await client.post(
        "/api/v1/evaluators",
        headers=headers,
        json={"name": name, "definition": definition or builtin()},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
async def test_versions_scoring_source_and_tenant_isolation(api, monkeypatch):
    app, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    evaluator = await create(client, headers)
    source = (
        await client.post("/api/v1/evaluations", headers=headers, json=request)
    ).json()
    original = (
        await client.get(f"/api/v1/evaluations/{source['id']}", headers=headers)
    ).json()

    # All generation paths fail if a score-only pass accidentally calls them.
    async def forbidden(*args, **kwargs):
        raise AssertionError("Score-only run generated an output")

    monkeypatch.setattr(
        "app.services.evaluation_service.generate_playground", forbidden
    )
    result = await client.post(
        f"/api/v1/evaluations/{source['id']}/score",
        headers=headers,
        json={"evaluators": [{"version_id": evaluator["latest"]["id"]}]},
    )
    assert result.status_code == 202, result.text
    scored = (
        await client.get(f"/api/v1/evaluations/{result.json()['id']}", headers=headers)
    ).json()
    assert scored["run"]["status"] == "completed" and scored["run"]["passed"] == 1
    assert scored["run"]["source_run_id"] == source["id"]
    assert scored["run"]["config"]["call_budget"]["total"] == 0
    assert [item["output"] for item in scored["results"]] == [
        item["output"] for item in original["results"]
    ]
    assert scored["score_summary"][0]["mean"] == 0.5
    # A changed definition is a new version; the old scoring pass stays pinned.
    updated = await client.post(
        f"/api/v1/evaluators/{evaluator['id']}/versions",
        headers=headers,
        json={"base_version": 1, "definition": builtin("contains", "Hello")},
    )
    assert updated.status_code == 201 and updated.json()["version"] == 2
    stale = await client.post(
        f"/api/v1/evaluators/{evaluator['id']}/versions",
        headers=headers,
        json={"base_version": 1, "definition": builtin("contains", "Goodbye")},
    )
    assert stale.status_code == 409
    unchanged = (
        await client.get(f"/api/v1/evaluations/{result.json()['id']}", headers=headers)
    ).json()
    assert unchanged["run"]["config"]["evaluators"][0]["version"] == 1
    assert (
        await client.get(f"/api/v1/evaluations/{source['id']}", headers=headers)
    ).json() == original
    async with sessions() as db:
        rows = (
            await db.scalars(
                select(EvaluationScore).where(
                    EvaluationScore.run_id == UUID(scored["run"]["id"])
                )
            )
        ).all()
        assert len(rows) == 2 and all(
            row.evaluator_version_id == UUID(evaluator["latest"]["id"]) for row in rows
        )
    _, other = await bootstrap(app, client, "other_evaluator_user")
    assert (await client.get("/api/v1/evaluators", headers=other)).json()["total"] == 0
    assert (
        await client.get(
            f"/api/v1/evaluators/{evaluator['id']}/versions", headers=other
        )
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/evaluations/{source['id']}/score",
            headers=other,
            json={"evaluators": [{"version_id": evaluator["latest"]["id"]}]},
        )
    ).status_code == 404
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=other,
            json={**request, "evaluators": [{"version_id": evaluator["latest"]["id"]}]},
        )
    ).status_code == 404


@pytest.mark.asyncio
async def test_required_informational_preview_archive_and_validation(api):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    required = await create(client, headers, "Valid text", builtin("contains", "o"))
    information = await create(
        client, headers, "Informational check", builtin("contains", "impossible")
    )
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "assertions": [],
            "evaluators": [
                {"version_id": required["latest"]["id"]},
                {"version_id": information["latest"]["id"], "required": False},
            ],
        },
    )
    detail = (
        await client.get(
            f"/api/v1/evaluations/{response.json()['id']}", headers=headers
        )
    ).json()
    assert detail["run"]["passed"] == 2 and detail["score_summary"][1]["mean"] == 0
    assert detail["results"][0]["checks"][1]["required"] is False
    preview = await client.post(
        f"/api/v1/evaluators/versions/{required['latest']['id']}/test",
        headers=headers,
        json={"output": "Hello", "inputs": {}, "expected_output": None},
    )
    assert preview.status_code == 200 and preview.json()["preview"] is True
    assert (await client.get("/api/v1/evaluations", headers=headers)).json()[
        "total"
    ] == 1
    archived = await client.patch(
        f"/api/v1/evaluators/{required['id']}", headers=headers, json={"archived": True}
    )
    assert archived.status_code == 200
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={**request, "evaluators": [{"version_id": required["latest"]["id"]}]},
        )
    ).status_code == 409
    invalid = await client.post(
        "/api/v1/evaluators",
        headers=headers,
        json={
            "name": "Secret",
            "definition": {**builtin(), "api_key": "must-not-save"},
        },
    )
    assert invalid.status_code == 422 and "must-not-save" not in invalid.text
    no_gate = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "assertions": [],
            "evaluators": [
                {"version_id": information["latest"]["id"], "required": False}
            ],
        },
    )
    assert no_gate.status_code == 422


@pytest.mark.asyncio
async def test_sdk_scopes_and_missing_source_outputs(api, monkeypatch):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    evaluator = await create(client, headers)
    read_key = (
        await client.post(
            "/api/v1/project-keys", headers=headers, json={"name": "Read only"}
        )
    ).json()
    read_headers = {"Authorization": "Bearer " + read_key["secret"]}
    assert (
        await client.get("/api/v1/sdk/evaluations/evaluators", headers=read_headers)
    ).status_code == 403
    key = (
        await client.post(
            "/api/v1/project-keys",
            headers=headers,
            json={"name": "Eval CI", "scopes": ["evaluations:write"]},
        )
    ).json()
    sdk_headers = {"Authorization": "Bearer " + key["secret"]}
    catalog = await client.get(
        "/api/v1/sdk/evaluations/evaluators", headers=sdk_headers
    )
    assert (
        catalog.status_code == 200
        and catalog.json()["items"][0]["latest"]["id"] == evaluator["latest"]["id"]
    )
    submitted = await client.post(
        "/api/v1/sdk/evaluations/submissions",
        headers=sdk_headers,
        json={
            "prompt_version_id": request["prompt_version_id"],
            "dataset_revision_id": request["dataset_revision_id"],
            "model": "own-app",
            "results": [
                {
                    "case_index": 0,
                    "output": "Hello",
                    "checks": [{"name": "correct", "passed": True}],
                },
                {"case_index": 1, "error": "Generation failed"},
            ],
        },
    )
    assert submitted.status_code == 201, submitted.text
    source = submitted.json()["id"]
    body = {"evaluators": [{"version_id": evaluator["latest"]["id"]}]}
    assert (
        await client.post(
            f"/api/v1/sdk/evaluations/runs/{source}/score",
            headers=read_headers,
            json=body,
        )
    ).status_code == 403

    async def forbidden(*args, **kwargs):
        raise AssertionError("Saved-output scoring generated text")

    monkeypatch.setattr(
        "app.services.evaluation_service.generate_playground", forbidden
    )
    scored = await client.post(
        f"/api/v1/sdk/evaluations/runs/{source}/score", headers=sdk_headers, json=body
    )
    assert scored.status_code == 202, scored.text
    detail = (
        await client.get(
            f"/api/v1/sdk/evaluations/runs/{scored.json()['id']}", headers=sdk_headers
        )
    ).json()
    assert detail["run"]["passed"] == 1 and detail["run"]["errors"] == 1
    metric = detail["score_summary"][0]
    assert metric["count"] == 1 and metric["missing"] == 1 and metric["mean"] == 1
    assert detail["results"][1]["output"] is None
    await client.delete("/api/v1/project-keys/" + key["id"], headers=headers)
    assert (
        await client.get("/api/v1/sdk/evaluations/evaluators", headers=sdk_headers)
    ).status_code == 401


@pytest.mark.asyncio
async def test_judge_failure_isolated_and_keys_excluded(api, monkeypatch):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    required = await create(client, headers, "Basic check", builtin("contains", "o"))
    judge = await create(
        client,
        headers,
        "Quality judge",
        {
            "kind": "llm_judge",
            "provider": "groq",
            "model": "judge-model",
            "rubric": "Measure quality",
            "outputs": [{"name": "quality", "data_type": "numeric"}],
        },
    )

    async def broken(*args, **kwargs):
        return {
            "output": '{"scores":[{"name":"quality","value":true,"reason":"wrong type"}]}',
            "latency_ms": 1,
            "tokens_input": 10,
            "tokens_output": 10,
        }

    monkeypatch.setattr("app.services.evaluator_service.generate_playground", broken)
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "assertions": [],
            "evaluators": [
                {"version_id": required["latest"]["id"]},
                {
                    "version_id": judge["latest"]["id"],
                    "required": False,
                    "api_key": "private-judge-token",
                },
            ],
        },
    )
    assert response.status_code == 202
    detail_response = await client.get(
        f"/api/v1/evaluations/{response.json()['id']}", headers=headers
    )
    detail = detail_response.json()
    assert "private-judge-token" not in detail_response.text
    assert detail["run"]["passed"] == 2 and detail["run"]["errors"] == 0
    metric = next(item for item in detail["score_summary"] if item["name"] == "quality")
    assert metric["errors"] == 2 and metric["count"] == 0 and metric["mean"] is None
    assert detail["run"]["config"]["call_budget"]["judging"] == 2
    no_key = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={**request, "evaluators": [{"version_id": judge["latest"]["id"]}]},
    )
    assert no_key.status_code == 422


@pytest.mark.asyncio
async def test_multi_output_judge_types_mapping_usage_and_invalid_shape(monkeypatch):
    definition = EvaluatorDefinition(
        kind="llm_judge",
        provider="groq",
        model="judge",
        rubric="Assess DATA only",
        outputs=[
            {"name": "correct", "data_type": "boolean"},
            {"name": "quality", "data_type": "numeric", "threshold": 0.8},
            {
                "name": "tone",
                "data_type": "categorical",
                "categories": ["good", "bad"],
                "passing_categories": ["good"],
            },
        ],
        mapping={"candidate": "output"},
    )
    from pydantic import SecretStr

    async def generated(request):
        assert "injection text" in request.template_text
        assert "expected_output" not in request.template_text
        return {
            "output": json.dumps(
                {
                    "scores": [
                        {"name": "correct", "value": True, "reason": "Yes"},
                        {"name": "quality", "value": 0.9, "reason": "Good"},
                        {"name": "tone", "value": "good", "reason": "Friendly"},
                    ]
                }
            ),
            "latency_ms": 3,
            "tokens_input": 4,
            "tokens_output": 5,
        }

    monkeypatch.setattr("app.services.evaluator_service.generate_playground", generated)
    snapshot = {
        "version_id": "version",
        "name": "Judge",
        "version": 1,
        "required": True,
        "definition": definition.model_dump(),
    }
    checks = await execute_evaluator(
        snapshot,
        {"inputs": {}, "expected_output": None},
        "injection text",
        SecretStr("token"),
    )
    assert [item["value"] for item in checks] == [True, 0.9, "good"]
    assert sum(item.get("tokens_input", 0) for item in checks) == 4

    async def malformed(*args):
        return {
            "output": '{"scores":[{"name":[],"value":1}]}',
            "latency_ms": 1,
            "tokens_input": 0,
            "tokens_output": 0,
        }

    monkeypatch.setattr("app.services.evaluator_service.generate_playground", malformed)
    with pytest.raises(HTTPException):
        await execute_evaluator(
            snapshot,
            {"inputs": {}, "expected_output": None},
            "text",
            SecretStr("token"),
        )


@pytest.mark.asyncio
async def test_inline_judge_errors_have_coverage_and_do_not_skip_saved_checks(
    api, monkeypatch
):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    evaluator = await create(client, headers)

    async def broken(*args):
        raise HTTPException(502, "Judge returned invalid JSON")

    monkeypatch.setattr("app.services.evaluation_service.judge_output", broken)
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "judge": {
                "provider": "groq",
                "model": "judge",
                "api_key": "request-only",
                "rubric": "Correctness",
            },
            "evaluators": [{"version_id": evaluator["latest"]["id"]}],
        },
    )
    assert response.status_code == 202
    detail = (
        await client.get(
            f"/api/v1/evaluations/{response.json()['id']}", headers=headers
        )
    ).json()
    judge = next(
        item for item in detail["score_summary"] if item["name"] == "llm_judge"
    )
    assert (
        judge["count"] == 0
        and judge["errors"] == 2
        and judge["missing"] == 0
        and judge["mean"] is None
    )
    assert detail["run"]["errors"] == 2
    assert (
        next(item for item in detail["score_summary"] if item["evaluator_version_id"])[
            "count"
        ]
        == 2
    )


def test_call_budget_judges_cases_and_variants():
    judge = {"definition": {"kind": "llm_judge"}}
    assert call_budget(20, "groq", [judge, judge])["total"] == 60
    assert call_budget(20, "groq", [judge, judge], score_only=True)["generation"] == 0
    for args in [(1, "mock", [judge] * 3), (21, "mock", [judge]), (51, "groq", [])]:
        with pytest.raises(HTTPException):
            call_budget(*args)


@pytest.mark.asyncio
async def test_cancellation_between_selected_judges_stops_later_calls(api, monkeypatch):
    _, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    definition = {
        "kind": "llm_judge",
        "provider": "groq",
        "model": "judge",
        "rubric": "Assess correctness",
        "outputs": [{"name": "quality", "data_type": "numeric"}],
    }
    first = await create(client, headers, "First judge", definition)
    second = await create(client, headers, "Second judge", definition)
    called = []

    async def cancel_after_first(snapshot, *args):
        called.append(snapshot["version_id"])
        async with sessions() as db:
            run = await db.scalar(
                select(EvaluationRun).where(EvaluationRun.status == "running")
            )
            run.status = "cancelled"
            await db.commit()
        return [{"kind": "quality", "passed": True, "value": 1, "reason": "Passed"}]

    monkeypatch.setattr(
        "app.services.evaluator_service.execute_evaluator", cancel_after_first
    )
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "evaluators": [
                {"version_id": first["latest"]["id"], "api_key": "first-key"},
                {"version_id": second["latest"]["id"], "api_key": "second-key"},
            ],
        },
    )
    assert response.status_code == 202
    detail = (
        await client.get(
            f"/api/v1/evaluations/{response.json()['id']}", headers=headers
        )
    ).json()
    assert called == [first["latest"]["id"]]
    assert detail["run"]["status"] == "cancelled" and detail["run"]["completed"] == 0
    assert detail["results"] == []


@pytest.mark.asyncio
async def test_catalog_filters_before_pagination_and_scope(api):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    source = await client.post("/api/v1/evaluations", headers=headers, json=request)
    rows = (
        await client.get(
            "/api/v1/evaluations",
            headers=headers,
            params={
                "dataset_revision_id": request["dataset_revision_id"],
                "search": "Echo",
                "status": "completed",
                "limit": 1,
            },
        )
    ).json()
    assert rows["total"] == 1 and rows["items"][0]["id"] == source.json()["id"]
    assert (
        await client.get("/api/v1/evaluations", headers=headers, params={"search": "%"})
    ).json()["total"] == 0
