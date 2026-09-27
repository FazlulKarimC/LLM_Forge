import json

import httpx
import pytest
from llmforge import EvaluationTimeout, LLMForgeError


def test_fetch_conditional_cache_and_revocation(snapshot, sdk_client):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer lf_live_SECRET"
        if len(requests) == 1:
            return httpx.Response(200, json=snapshot, headers={"etag": '"one"'})
        if len(requests) == 2:
            assert request.headers["if-none-match"] == '"one"'
            return httpx.Response(304)
        return httpx.Response(401, json={"message": "Revoked"})

    with sdk_client(handler) as forge:
        snapshot = forge.get_prompt("Echo demo")
        assert forge.get_prompt("Echo demo") is snapshot
        assert snapshot.compile(query="{{literal}}") == "{{literal}}"
        assert requests[0].url.params["label"] == "production"
        with pytest.raises(LLMForgeError) as caught:
            forge.get_prompt("Echo demo")
        assert caught.value.status_code == 401 and (not forge._cache)


def test_promotion_refreshes_snapshot(snapshot, sdk_client):
    responses = iter(
        [snapshot, {**snapshot, "version": 2, "template_text": "Hi {{query}}"}]
    )
    with sdk_client(
        lambda request: httpx.Response(
            200, json=next(responses), headers={"etag": '"two"'}
        )
    ) as forge:
        assert forge.get_prompt("Echo demo").version == 1
        assert forge.get_prompt("Echo demo").compile(query="Sam") == "Hi Sam"


def test_network_errors_never_use_cached_prompt(snapshot, sdk_client):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=snapshot, headers={"etag": "one"})
        raise httpx.ConnectError("lf_live_SECRET", request=request)

    with sdk_client(handler) as forge:
        forge.get_prompt("Echo demo")
        with pytest.raises(LLMForgeError) as caught:
            forge.get_prompt("Echo demo")
        assert "SECRET" not in str(caught.value) and len(calls) == 2


def test_wait_and_submission_request_shape(completed_run, sdk_client):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"id": "run-id"})
        return httpx.Response(200, json={"run": completed_run, "results": []})

    with sdk_client(handler) as forge:
        identifier = forge.start_evaluation("version-id", "revision-id")
        result = forge.wait_for_evaluation(identifier)
        assert result.meets_threshold() and result.pass_rate == 1
        assert json.loads(requests[0].content)["assertions"] == [
            {"kind": "exact_match"}
        ]
        forge.submit_evaluation(
            "version-id",
            "revision-id",
            model="own-model",
            results=[
                {
                    "case_index": 0,
                    "output": "a",
                    "checks": [{"name": "contract", "passed": True}],
                }
            ],
            metrics={"latency": 10},
        )
        assert requests[-1].url.path.endswith("/submissions")
        assert json.loads(requests[-1].content)["metrics"] == {"latency": 10}


def test_timeout_attempts_cancel(monkeypatch, completed_run, sdk_client):
    ticks = iter([0, 0.01, 2, 3])
    monkeypatch.setattr("llmforge.client.time.monotonic", lambda: next(ticks))
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(
            200, json={"run": {**completed_run, "status": "running"}, "results": []}
        )

    with sdk_client(handler) as forge:
        with pytest.raises(EvaluationTimeout):
            forge.wait_for_evaluation("run-id", timeout=1, cancel_on_timeout=True)
    assert methods == ["GET", "POST"]


def test_secrets_are_redacted_from_error(sdk_client):
    with sdk_client(
        lambda request: httpx.Response(
            422, json={"detail": "lf_live_SECRET PROVIDER_SECRET JUDGE_SECRET"}
        )
    ) as forge:
        with pytest.raises(LLMForgeError) as caught:
            forge.start_evaluation(
                "v", "d", api_key="PROVIDER_SECRET", judge={"api_key": "JUDGE_SECRET"}
            )
        assert "SECRET" not in str(caught.value)


def test_sdk_selector_validation(sdk_client):
    with sdk_client(
        lambda request: pytest.fail("Invalid selections must not call server")
    ) as forge:
        with pytest.raises(ValueError):
            forge.get_prompt("Echo", label="production", version=1)
        with pytest.raises(ValueError):
            forge.get_prompt("Echo", version=0)
        with pytest.raises(ValueError):
            forge.get_dataset("Cases", version=False)
