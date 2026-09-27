import json

import pytest
from llmforge import Evaluation, Prompt
from llmforge.cli import gate, main


@pytest.mark.parametrize(
    "changes,minimum,code",
    [
        ({}, 1, 0),
        ({"passed": 1}, 1, 1),
        ({"passed": 1}, 0.5, 0),
        ({"errors": 1, "passed": 1}, 0.5, 1),
        ({"status": "cancelled"}, 0, 2),
    ],
)
def test_gate(changes, minimum, code, completed_run):
    assert gate(Evaluation({**completed_run, **changes}, []), minimum) == code


def test_offline_cli_report_and_bad_counts(tmp_path, completed_run):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"run": completed_run, "results": []}))
    assert main(["check", str(report)]) == 0
    report.write_text(
        json.dumps({"run": {**completed_run, "passed": 100}, "results": []})
    )
    assert main(["check", str(report)]) == 2
    assert main(["check", str(report), "--min-pass-rate", "NaN"]) == 2


def test_cli_evaluate_writes_artifact(monkeypatch, tmp_path, completed_run):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get_prompt(self, *_args, **_kwargs):
            return Prompt("v", "p", "Echo", 1, "{{q}}", "mustache", ("q",))

        def get_dataset(self, *_args, **_kwargs):
            return {"revision": {"id": "d", "version": 1}}

        def start_evaluation(self, *_args, **_kwargs):
            return "run-id"

        def wait_for_evaluation(self, *_args, **kwargs):
            assert kwargs["cancel_on_timeout"]
            return Evaluation(completed_run, [])

    monkeypatch.setattr("llmforge.cli.LLMForge", FakeClient)
    output = tmp_path / "nested" / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--prompt",
                "Echo",
                "--dataset",
                "Cases",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["run"]["passed"] == 2
