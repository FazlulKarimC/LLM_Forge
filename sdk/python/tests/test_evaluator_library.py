import json
import httpx
import pytest
from llmforge import LLMForgeError


def test_library_selection_scoring_and_summary(completed_run, sdk_client):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("evaluators"):
            return httpx.Response(200, json={"items": [], "total": 0})
        if request.method == "POST":
            return httpx.Response(202, json={"id": "scored"})
        return httpx.Response(
            200,
            json={
                "run": completed_run,
                "results": [],
                "score_summary": [{"name": "quality", "mean": 0.9}],
            },
        )

    selections = [
        {"version_id": "evaluator-v1", "required": True, "api_key": "judge-key"}
    ]
    with sdk_client(handler) as client:
        assert client.list_evaluators()["total"] == 0
        client.start_evaluation(
            "prompt", "dataset", assertions=[], evaluators=selections
        )
        assert json.loads(calls[-1].content)["evaluators"] == selections
        assert client.score_evaluation("source", selections) == "scored"
        assert calls[-1].url.path.endswith("/source/score")
        report = client.get_evaluation("scored")
        assert report.to_dict()["score_summary"][0]["mean"] == 0.9


def test_nested_judge_key_redacted_and_not_retried(sdk_client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"detail": "Rejected private-evaluator-key"})

    with sdk_client(handler) as client:
        with pytest.raises(LLMForgeError) as error:
            client.score_evaluation(
                "run", [{"version_id": "evaluator", "api_key": "private-evaluator-key"}]
            )
    assert "private-evaluator-key" not in str(error.value)
    assert len(calls) == 1
