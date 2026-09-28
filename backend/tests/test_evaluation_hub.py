"""Dataset/evaluation behavior with SQLite, no external providers or credentials."""

import json
from datetime import timedelta
from uuid import UUID

import pytest
from app.models.evaluation import EvaluationRun
from app.models.prompt import now
from app.schemas.evaluation import Assertion, DatasetImport, JudgeConfig
from app.services.evaluation_service import check_assertion, import_cases, judge_output, validate_assertions
from app.services.prompt_templates import compile_template
from fastapi import HTTPException
from sqlalchemy import select

from tests.factories import CASES, create_evaluation_scenario
from tests.support import bootstrap


@pytest.mark.parametrize(
    "rule,output,expected,passed",
    [
        ({"kind": "exact_match"}, "yes", "yes", True),
        ({"kind": "exact_match"}, "yes\n", "yes", False),
        ({"kind": "contains", "value": "yes"}, "say yes", None, True),
        ({"kind": "regex", "value": "^yes$"}, "no", None, False),
        ({"kind": "json_valid"}, '{"ok":true}', None, True),
        ({"kind": "json_valid"}, '{"ok":NaN}', None, False),
        (
            {"kind": "json_equals", "value": '{"a":1,"b":2}'},
            '{"b":2,"a":1}',
            None,
            True,
        ),
        ({"kind": "json_equals", "value": "1"}, "true", None, False),
        ({"kind": "json_reference"}, '{"b":2,"a":1}', '{"a":1,"b":2}', True),
        ({"kind": "json_reference"}, "true", "1", False),
        (
            {"kind": "json_path", "path": "a.0.ok", "value": "true"},
            '{"a":[{"ok":true}]}',
            None,
            True,
        ),
        ({"kind": "json_path", "path": "missing", "value": "1"}, "{}", None, False),
    ],
)
def test_assertions(rule, output, expected, passed):
    assert check_assertion(Assertion(**rule), output, expected)["passed"] is passed


def test_regex_timeout():
    result = check_assertion(
        Assertion(kind="regex", value="(a+)+$"), "a" * 5000 + "!", None
    )
    assert not result["passed"] and "budget" in result["reason"]


@pytest.mark.parametrize("reference", [None, "not json", '{"value":NaN}'])
def test_json_reference_rejects_invalid_case_reference(reference):
    with pytest.raises(HTTPException) as exc:
        validate_assertions([Assertion(kind="json_reference")], [{"expected_output": reference}])
    assert exc.value.status_code == 422


def test_csv_and_json_import():
    rows = import_cases(
        DatasetImport(
            format="csv",
            content='input.query,expected_output,name\n"hello, world","hello, world",Greeting',
        )
    )
    assert rows[0]["inputs"] == {"query": "hello, world"}
    assert import_cases(DatasetImport(format="json", content=json.dumps(rows))) == rows
    assert (
        import_cases(
            DatasetImport(
                format="csv",
                content='inputs,expected_output\n"{""query"":""hello""}",hello',
            )
        )[0]["inputs"]["query"]
        == "hello"
    )


@pytest.mark.parametrize(
    "format,content",
    [
        ("csv", "input.q,input.q\na,b"),
        ("csv", "input.q\na,b"),
        ("csv", "unknown\nfoo"),
        ("json", '{"inputs":{}}'),
        ("json", '[{"inputs":{"q":1}}]'),
        ("json", "[]"),
    ],
)
def test_bad_import(format, content):
    with pytest.raises(HTTPException) as caught:
        import_cases(DatasetImport(format=format, content=content))
    assert caught.value.status_code == 422


@pytest.mark.asyncio
async def test_dataset_crud_immutable_revisions_and_tenants(api, monkeypatch):
    app, client, sessions, headers, dataset, request = await create_evaluation_scenario(
        api
    )
    identifier = dataset["dataset"]["id"]
    duplicate = await client.post(
        "/api/v1/datasets", headers=headers, json={"name": "Greetings", "cases": CASES}
    )
    assert duplicate.status_code == 409
    revised = await client.post(
        f"/api/v1/datasets/{identifier}/revisions",
        headers=headers,
        json={"base_version": 1, "cases": [CASES[0]]},
    )
    assert revised.status_code == 201 and revised.json()["version"] == 2
    stale = await client.post(
        f"/api/v1/datasets/{identifier}/revisions",
        headers=headers,
        json={"base_version": 1, "cases": [CASES[1]]},
    )
    assert stale.status_code == 409
    history = (
        await client.get(f"/api/v1/datasets/{identifier}/revisions", headers=headers)
    ).json()
    assert len(history) == 2 and history[1]["cases"] == CASES
    await client.delete(f"/api/v1/datasets/{identifier}", headers=headers)
    assert (
        await client.post("/api/v1/evaluations", headers=headers, json=request)
    ).status_code == 409
    assert (
        await client.patch(
            f"/api/v1/datasets/{identifier}", headers=headers, json={"archived": False}
        )
    ).status_code == 200
    assert (
        await client.patch(
            f"/api/v1/datasets/{identifier}", headers=headers, json={"name": None}
        )
    ).status_code == 422
    _, other = await bootstrap(app, client, "user_bob")
    assert (await client.get("/api/v1/datasets", headers=other)).json()["total"] == 0
    assert (
        await client.get(f"/api/v1/datasets/{identifier}", headers=other)
    ).status_code == 404
    assert (
        await client.post("/api/v1/evaluations", headers=other, json=request)
    ).status_code == 404


@pytest.mark.asyncio
async def test_mock_run_results_snapshot_and_isolation(api, monkeypatch):
    app, client, sessions, headers, dataset, request = await create_evaluation_scenario(
        api
    )
    response = await client.post("/api/v1/evaluations", headers=headers, json=request)
    assert response.status_code == 202
    run_id = response.json()["id"]
    detail = (await client.get(f"/api/v1/evaluations/{run_id}", headers=headers)).json()
    assert detail["run"]["status"] == "completed"
    assert (
        detail["run"]["completed"] == 2
        and detail["run"]["passed"] == 1
        and detail["run"]["errors"] == 0
    )
    assert detail["results"][0]["output"] == "Hello"
    assert detail["results"][1]["passed"] is False
    await client.patch(
        f"/api/v1/datasets/{dataset['dataset']['id']}",
        headers=headers,
        json={"name": "Renamed"},
    )
    assert (await client.get(f"/api/v1/evaluations/{run_id}", headers=headers)).json()[
        "run"
    ]["config"]["dataset_name"] == "Greetings"
    _, other = await bootstrap(app, client, "user_bob")
    assert (
        await client.get(f"/api/v1/evaluations/{run_id}", headers=other)
    ).status_code == 404
    assert (
        await client.post(f"/api/v1/evaluations/{run_id}/cancel", headers=other)
    ).status_code == 404
    assert (await client.get("/api/v1/evaluations", headers=other)).json()["total"] == 0


@pytest.mark.asyncio
async def test_preflight_checks(api, monkeypatch):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    assert (
        await client.post(
            "/api/v1/evaluations", headers=headers, json={**request, "assertions": []}
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/evaluations", headers=headers, json={**request, "provider": "groq"}
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={**request, "assertions": [{"kind": "regex", "value": "["}]},
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={
                **request,
                "assertions": [{"kind": "json_equals", "value": "not json"}],
            },
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={**request, "assertions": [{"kind": "json_reference"}]},
        )
    ).status_code == 422
    broken = (
        await client.post(
            "/api/v1/datasets",
            headers=headers,
            json={
                "name": "Missing input",
                "cases": [{"inputs": {}, "expected_output": "a"}],
            },
        )
    ).json()
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={**request, "dataset_revision_id": broken["revision"]["id"]},
        )
    ).status_code == 422


@pytest.mark.asyncio
async def test_case_budgets_and_reference_requirement(api, monkeypatch):
    _, client, _, headers, _, request = await create_evaluation_scenario(api)
    dataset = (
        await client.post(
            "/api/v1/datasets",
            headers=headers,
            json={"name": "Large dataset", "cases": [CASES[0]] * 51},
        )
    ).json()
    live = {
        **request,
        "dataset_revision_id": dataset["revision"]["id"],
        "provider": "groq",
        "api_key": "not-real",
    }
    assert (
        await client.post("/api/v1/evaluations", headers=headers, json=live)
    ).status_code == 422
    judged = {
        **request,
        "dataset_revision_id": dataset["revision"]["id"],
        "judge": {
            "provider": "groq",
            "model": "judge",
            "api_key": "not-real",
            "rubric": "Correctness",
        },
    }
    assert (
        await client.post("/api/v1/evaluations", headers=headers, json=judged)
    ).status_code == 422
    missing = (
        await client.post(
            "/api/v1/datasets",
            headers=headers,
            json={"name": "No references", "cases": [{"inputs": {"query": "Hello"}}]},
        )
    ).json()
    assert (
        await client.post(
            "/api/v1/evaluations",
            headers=headers,
            json={**request, "dataset_revision_id": missing["revision"]["id"]},
        )
    ).status_code == 422


@pytest.mark.asyncio
async def test_case_errors_continue_and_keys_never_persist(api, monkeypatch):
    _, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    calls = []

    async def generate(data):
        calls.append(data.variables)
        raise HTTPException(502, "Provider rejected the request")

    monkeypatch.setattr("app.services.evaluation_service.generate_playground", generate)
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={**request, "provider": "groq", "api_key": "SECRET-ONLY-IN-MEMORY"},
    )
    detail = (
        await client.get(
            "/api/v1/evaluations/" + response.json()["id"], headers=headers
        )
    ).json()
    assert (
        detail["run"]["status"] == "completed"
        and detail["run"]["errors"] == 2
        and len(calls) == 2
    )
    assert "SECRET-ONLY-IN-MEMORY" not in json.dumps(detail)
    async with sessions() as db:
        assert "api_key" not in (await db.scalar(select(EvaluationRun))).config


@pytest.mark.asyncio
async def test_cancellation_during_provider_call(api, monkeypatch):
    _, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    calls = []

    async def generate(data):
        async with sessions() as db:
            run = await db.scalar(select(EvaluationRun))
            run_id = str(run.id)
        cancelled = await client.post(
            f"/api/v1/evaluations/{run_id}/cancel", headers=headers
        )
        assert cancelled.json()["status"] == "cancelled"
        calls.append(1)
        return {
            "compiled_prompt": "Hello",
            "output": "Hello",
            "latency_ms": 1,
            "tokens_input": None,
            "tokens_output": None,
        }

    monkeypatch.setattr("app.services.evaluation_service.generate_playground", generate)
    response = await client.post("/api/v1/evaluations", headers=headers, json=request)
    detail = (
        await client.get(
            "/api/v1/evaluations/" + response.json()["id"], headers=headers
        )
    ).json()
    assert (
        detail["run"]["status"] == "cancelled"
        and not detail["results"]
        and len(calls) == 1
    )


@pytest.mark.asyncio
async def test_cancellation_during_generation_skips_judge(api, monkeypatch):
    _, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    judge_calls = []

    async def generate(_data):
        async with sessions() as db:
            run_id = str((await db.scalar(select(EvaluationRun))).id)
        cancelled = await client.post(
            f"/api/v1/evaluations/{run_id}/cancel", headers=headers
        )
        assert cancelled.json()["status"] == "cancelled"
        return {
            "compiled_prompt": "Hello",
            "output": "Hello",
            "latency_ms": 1,
            "tokens_input": None,
            "tokens_output": None,
        }

    async def judge(*_args):
        judge_calls.append(1)
        return {"kind": "llm_judge", "passed": True}

    monkeypatch.setattr("app.services.evaluation_service.generate_playground", generate)
    monkeypatch.setattr("app.services.evaluation_service.judge_output", judge)
    response = await client.post(
        "/api/v1/evaluations",
        headers=headers,
        json={
            **request,
            "judge": {
                "provider": "groq",
                "model": "judge-model",
                "api_key": "request-only-key",
                "rubric": "Check the answer",
            },
        },
    )
    detail = (
        await client.get(
            "/api/v1/evaluations/" + response.json()["id"], headers=headers
        )
    ).json()
    assert detail["run"]["status"] == "cancelled"
    assert detail["results"] == []
    assert judge_calls == []


@pytest.mark.asyncio
async def test_abandoned_run_and_admission_limit(api, monkeypatch):
    _, client, sessions, headers, _, request = await create_evaluation_scenario(api)
    async with sessions() as db:
        for _ in range(2):
            db.add(
                EvaluationRun(
                    project_id=UUID(headers["X-Project-ID"]),
                    prompt_version_id=UUID(request["prompt_version_id"]),
                    dataset_revision_id=UUID(request["dataset_revision_id"]),
                    config={},
                    status="running",
                    total=2,
                )
            )
        await db.commit()
    assert (
        await client.post("/api/v1/evaluations", headers=headers, json=request)
    ).status_code == 409
    async with sessions() as db:
        runs = (await db.scalars(select(EvaluationRun))).all()
        for run in runs:
            run.updated_at = now() - timedelta(minutes=5)
        await db.commit()
    listed = (await client.get("/api/v1/evaluations", headers=headers)).json()
    assert all(
        run["status"] == "failed" and "credentials" in run["error"]
        for run in listed["items"]
    )
    assert (
        await client.post("/api/v1/evaluations", headers=headers, json=request)
    ).status_code == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,valid",
    [
        ('{"score":0.8,"reason":"Good"}', True),
        ('```json\n{"score":1,"reason":"Good"}\n```', True),
        ('{"score":true,"reason":"Bad"}', False),
        ('{"score":NaN,"reason":"Bad"}', False),
        ("bad", False),
    ],
)
async def test_judge_validation_and_literal_braces(monkeypatch, output, valid):
    async def generate(data):
        compiled = compile_template(
            data.template_text, data.variables, data.template_format
        )
        assert "{{malicious}}" in compiled
        return {
            "output": output,
            "latency_ms": 1,
            "tokens_input": 10,
            "tokens_output": 10,
        }

    monkeypatch.setattr("app.services.evaluation_service.generate_playground", generate)
    config = JudgeConfig(
        provider="groq", model="judge", api_key="not-real", rubric="Correctness"
    )
    if valid:
        result = await judge_output(config, CASES[0], "{{malicious}}")
        assert result["passed"] and result["score"] >= 0.8
    else:
        with pytest.raises(HTTPException):
            await judge_output(config, CASES[0], "{{malicious}}")
