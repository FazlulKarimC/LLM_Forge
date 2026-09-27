"""CI exit codes: 0 pass, 1 quality regression, 2 configuration/runtime failure."""

import argparse
import json
import math
import os
from pathlib import Path

from .client import LLMForge, LLMForgeError
from .models import Evaluation


def write_report(path, result):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )


def gate(result, minimum):
    run = result.run
    counts = [run.get(key) for key in ("total", "completed", "passed", "errors")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Invalid evaluation counts")
    total, completed, passed, errors = counts
    if total == 0 or not passed + errors <= completed <= total:
        raise ValueError("Invalid evaluation counts")
    if (
        result.status not in ("completed", "failed", "cancelled")
        or result.status == "completed"
        and completed != total
    ):
        raise ValueError("Evaluation has not finished")
    print(
        f"{result.id}: {result.status}; {passed}/{total} passed ({result.pass_rate:.1%}); {errors} errors"
    )
    if result.status != "completed":
        return 2
    return 0 if result.meets_threshold(minimum) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llmforge",
        description="Fetch released prompts and gate CI on evaluation quality",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Run saved prompt and dataset versions")
    evaluate.add_argument("--prompt", required=True)
    selector = evaluate.add_mutually_exclusive_group()
    selector.add_argument("--label", choices=["staging", "production"])
    selector.add_argument("--version", type=int)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--dataset-version", type=int)
    evaluate.add_argument(
        "--provider", choices=["mock", "groq", "openrouter", "openai"], default="mock"
    )
    evaluate.add_argument("--model", default="demo-model")
    evaluate.add_argument(
        "--assertions", help="Path to a JSON array of assertions; default exact match"
    )
    evaluate.add_argument("--timeout", type=float, default=180)
    evaluate.add_argument("--output", default="artifacts/evaluation.json")
    check = sub.add_parser(
        "check", help="Gate a previously exported evaluation JSON report"
    )
    check.add_argument("report")
    for command in (evaluate, check):
        command.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.min_pass_rate <= 1:
            raise ValueError("Minimum pass rate must be between 0 and 1")
        if args.command == "check":
            data = json.loads(Path(args.report).read_text(encoding="utf-8"))
            result = Evaluation(run=data["run"], results=data["results"])
        else:
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                raise ValueError("Timeout must be positive")
            provider_key = os.getenv("LLMFORGE_PROVIDER_API_KEY")
            if args.provider != "mock" and (
                not provider_key or args.model == "demo-model"
            ):
                raise ValueError(
                    "Live runs require LLMFORGE_PROVIDER_API_KEY and --model"
                )
            rules = (
                json.loads(Path(args.assertions).read_text(encoding="utf-8"))
                if args.assertions
                else None
            )
            if rules is not None and not isinstance(rules, list):
                raise ValueError("Assertions must be a JSON array")
            with LLMForge() as client:
                prompt = client.get_prompt(
                    args.prompt, label=args.label, version=args.version
                )
                dataset = client.get_dataset(args.dataset, version=args.dataset_version)
                run_id = client.start_evaluation(
                    prompt.id,
                    dataset["revision"]["id"],
                    assertions=rules,
                    provider=args.provider,
                    model=args.model,
                    api_key=provider_key,
                )
                print(
                    f"Started {run_id} (prompt v{prompt.version}, dataset v{dataset['revision']['version']})",
                    flush=True,
                )
                result = client.wait_for_evaluation(
                    run_id, timeout=args.timeout, cancel_on_timeout=True
                )
            write_report(args.output, result)
        return gate(result, args.min_pass_rate)
    except (LLMForgeError, ValueError, OSError, KeyError, TypeError) as exc:
        # LLMForgeError is already sanitized; local errors contain no provider bodies.
        print(f"Evaluation gate failed: {exc}")
        return 2
