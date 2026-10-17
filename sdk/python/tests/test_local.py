import json
import httpx
import pytest
from llmforge import LLMForgeError, Prompt
from llmforge.local import digest

CASES = [
    {"id": "a", "inputs": {"query": "Hello"}, "expected_output": "Hello", "metadata": {"category": "greeting"}},
    {"id": "b", "inputs": {"query": "Bye"}, "expected_output": "Bye"},
]
PROMPT = Prompt(id="version", prompt_id="prompt", name="Echo", version=1, template_text="{{query}}", template_format="mustache", variables=("query",))

def test_local_callbacks_keep_payloads_private_and_hash_order_independent(sdk_client, tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path.endswith("/local-reports")
        assert body["total"] == body["passed"] == 2
        assert not {"results", "inputs", "output", "api_key"} & body.keys()
        assert "Hello" not in request.content.decode()
        return httpx.Response(201, json={"id": "hosted"})
    with sdk_client(handler) as forge:
        def run(cases, **kwargs):
            return forge.evaluate_local(PROMPT, cases, task=lambda x: x,
                evaluators={"exact": lambda output, case: output == case["expected_output"]},
                scoring_version="v1", report_path=tmp_path/"report.json", **kwargs)
        first = run(CASES)
        assert first.id == "hosted" and first.meets_threshold()
        local = json.loads((tmp_path/"report.json").read_text())
        assert local["results"][0]["output"] == "Hello"
        assert local["results"][0]["metadata"] == {"category": "greeting"}
        assert requests[0].headers["Idempotency-Key"] == local["submission_key"]
        reordered = run(list(reversed(CASES)), upload=False)
        assert first.run["dataset_fingerprint"] == reordered.run["dataset_fingerprint"]
        assert len(requests) == 1
        changed = run([{**CASES[0], "expected_output": "different"}, CASES[1]], upload=False)
        assert changed.run["dataset_fingerprint"] != first.run["dataset_fingerprint"]
        assert not changed.meets_threshold()

def test_local_errors_do_not_pass_and_failed_upload_keeps_artifact(sdk_client, tmp_path):
    def handler(request): return httpx.Response(503, json={"message": "Unavailable"})
    with sdk_client(handler) as forge:
        with pytest.raises(LLMForgeError):
            forge.evaluate_local(PROMPT, CASES, task=lambda x: x,
                evaluators={"exact": lambda output, case: True}, scoring_version="v1",
                report_path=tmp_path/"report.json")
        assert json.loads((tmp_path/"report.json").read_text())["run"]["passed"] == 2
        result = forge.evaluate_local(PROMPT, CASES, task=lambda x: 1/0,
            evaluators={"exact": lambda output, case: True}, scoring_version="v1",
            upload=False, report_path=tmp_path/"errors.json")
        assert result.run["errors"] == 2 and not result.meets_threshold(0)

@pytest.mark.parametrize("cases", [[], [CASES[0], CASES[0]], [{"inputs": {"query": "x"}}], [{"id": "a", "inputs": {"x": 1}}]])
def test_local_validation_before_any_call(sdk_client, cases):
    with sdk_client(lambda request: pytest.fail("Unexpected HTTP")) as forge:
        with pytest.raises(ValueError):
            forge.evaluate_local(PROMPT, cases, task=lambda x: pytest.fail("Unexpected task"),
                evaluators={"exact": lambda output, case: True}, scoring_version="v1", report_path=None)

def test_scoring_version_changes_fingerprint(sdk_client):
    with sdk_client(lambda request: pytest.fail("Unexpected HTTP")) as forge:
        def run(version):
            return forge.evaluate_local(PROMPT, CASES, task=lambda x: x,
                evaluators={"exact": lambda output, case: True}, scoring_version=version,
                upload=False, report_path=None)
        assert run("v1").run["scoring_fingerprint"] != run("v2").run["scoring_fingerprint"]


@pytest.mark.parametrize("upload", [False, True])
def test_local_cli_uses_local_file_and_explicit_summary_upload(sdk_client, snapshot, monkeypatch, tmp_path, upload):
    from llmforge.cli import main
    requests = []
    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=snapshot)
        body = json.loads(request.content)
        assert request.url.path.endswith("/local-reports")
        assert body["passed"] == 2
        assert "Hello" not in request.content.decode()
        return httpx.Response(201, json={"id": "hosted-cli"})
    monkeypatch.setattr("llmforge.cli.LLMForge", lambda: sdk_client(handler))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(CASES), encoding="utf-8")
    report = tmp_path / "report.json"
    args = ["evaluate-local", "--prompt", "Echo", "--version", "1", "--dataset", str(cases), "--output", str(report)]
    assert main(args + (["--upload"] if upload else [])) == 0
    assert len([request for request in requests if request.method == "POST"]) == int(upload)
    artifact = json.loads(report.read_text(encoding="utf-8"))
    assert artifact["results"][0]["output"] == "Hello"
    assert main(["check", str(report)]) == 0
