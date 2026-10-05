"""The ergonomic SDK consumes responses conforming to backend-owned OpenAPI."""
import json
from pathlib import Path
import httpx
from jsonschema import Draft202012Validator
from llmforge import LLMForgeError
import pytest


def test_summary_cases_resume_and_idempotency(sdk_client):
    calls = []
    schema = json.loads((Path(__file__).resolve().parents[3] / "contracts/openapi.json").read_text())
    run = {"id": "00000000-0000-0000-0000-000000000001", "prompt_version_id": "00000000-0000-0000-0000-000000000002",
        "dataset_revision_id": "00000000-0000-0000-0000-000000000003", "source_run_id": None,
        "config": {}, "status": "failed", "total": 1, "completed": 0, "passed": 0, "errors": 0,
        "error": "Interrupted", "created_at": "2026-10-05T00:00:00Z", "updated_at": "2026-10-05T00:00:00Z"}
    summary = {"run": run, "results": [], "score_summary": [], "recovery": {"available": True,
        "failure_code": "runner_interrupted", "attempts": [], "remaining_calls": {"generation": 1, "judging": 0, "total": 1}},
        "usage": {"generation": {"tokens_input": 0, "tokens_output": 0}, "judging": {"tokens_input": 0, "tokens_output": 0}}}
    Draft202012Validator({"components": schema["components"], "$ref": "#/components/schemas/EvaluationSummaryResponse"}).validate(summary)
    page = {"items": [], "total": 0, "offset": 0, "limit": 25, "comparison_totals": None}
    Draft202012Validator({"components": schema["components"], "$ref": "#/components/schemas/CasePageResponse"}).validate(page)
    def handler(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=run)
        return httpx.Response(200, json=page if request.url.path.endswith("cases") else summary)
    with sdk_client(handler) as client:
        assert client.get_evaluation_summary(run["id"])["recovery"]["available"]
        assert client.list_evaluation_cases(run["id"])["items"] == []
        assert client.resume_evaluation(run["id"], api_key="request-only-key", idempotency_key="resume-once") == run["id"]
        assert calls[-1].headers["Idempotency-Key"] == "resume-once"
        client.start_evaluation("prompt", "dataset", idempotency_key="start-once")
        assert calls[-1].headers["Idempotency-Key"] == "start-once"


def test_resume_credentials_redacted_without_retry(sdk_client):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"detail": "bad judge-secret evaluator-secret"})
    with sdk_client(handler) as client:
        with pytest.raises(LLMForgeError) as error:
            client.resume_evaluation("run", judge_key="judge-secret", evaluator_keys={"judge": "evaluator-secret"})
        assert "judge-secret" not in str(error.value) and "evaluator-secret" not in str(error.value)
        assert len(calls) == 1
