"""Local evaluation: content stays in the local artifact, only summaries upload."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from .models import Evaluation

def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

def evaluate_local(client, prompt, cases, *, task, evaluators, scoring_version,
                   dataset_name="Local cases", model="application", upload=True,
                   report_path="artifacts/evaluation.json"):
    """Callbacks run on this machine. Evaluators map names to (output, case)->bool.
    scoring_version identifies evaluator code/settings; change it when policy changes.
    report_path is written before upload so a failed submission preserves local work.
    """
    if isinstance(cases, (str, Path)):
        path = Path(cases)
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1_000_000:
        raise ValueError("Supply 1–1,000,000 local cases")
    if not isinstance(scoring_version, str) or not scoring_version.strip() or len(scoring_version) > 200:
        raise ValueError("Supply a bounded scoring_version for the evaluator code/settings")
    if not evaluators or len(evaluators) > 10 or any(not isinstance(k, str) or not k.strip() or len(k) > 120 or not callable(v) for k, v in evaluators.items()):
        raise ValueError("Supply 1–10 named evaluator callbacks")
    if not isinstance(dataset_name, str) or not dataset_name.strip() or len(dataset_name) > 200 or not isinstance(model, str) or not model.strip() or len(model) > 255:
        raise ValueError("Supply bounded dataset/model names")
    identities = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"id", "inputs", "expected_output", "metadata"}:
            raise ValueError("Local cases accept id, inputs, expected_output and metadata")
        identity = case.get("id")
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 200 or identity in seen:
            raise ValueError("Every local case needs a unique stable id")
        if not isinstance(case.get("inputs"), dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in case["inputs"].items()):
            raise ValueError("Case inputs must map variable names to strings")
        if case.get("expected_output") is not None and not isinstance(case["expected_output"], str):
            raise ValueError("Expected outputs must be strings or null")
        seen.add(identity)
        identities.append({"id": identity, "inputs": case["inputs"], "expected_output": case.get("expected_output"), "metadata": case.get("metadata")})
    dataset_fingerprint = digest({"format": "llmforge-local-v1", "cases": sorted(identities, key=lambda c: c["id"])})
    policy = digest({"format": "llmforge-policy-v1", "version": scoring_version, "checks": sorted(evaluators)})
    results = []
    for index, case in enumerate(cases):
        record = {"case_index": index, "case_id": case["id"], "inputs": case["inputs"], "expected_output": case.get("expected_output"), "metadata": case.get("metadata"), "output": None, "checks": [], "error": None, "passed": False}
        try:
            output = task(prompt.compile(case["inputs"]))
            if not isinstance(output, str):
                raise ValueError("Tasks must return a string")
            record["output"] = output
            for name, evaluator in evaluators.items():
                passed = evaluator(output, case)
                if type(passed) is not bool:
                    raise ValueError("Evaluators must return bool")
                record["checks"].append({"name": name, "passed": passed})
            record["passed"] = all(check["passed"] for check in record["checks"])
        except Exception as exc:
            # Avoid retaining arbitrary provider exception messages even locally.
            record["error"] = type(exc).__name__
        results.append(record)
    run = {
        "id": str(uuid4()), "prompt_version_id": prompt.id, "status": "completed",
        "total": len(cases), "completed": len(cases),
        "passed": sum(r["passed"] and r["error"] is None for r in results),
        "errors": sum(r["error"] is not None for r in results),
        "dataset_name": dataset_name, "dataset_fingerprint": dataset_fingerprint,
        "scoring_fingerprint": policy, "model": model,
    }
    submission_key = str(uuid4())
    def save():
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"run": run, "results": results, "submission_key": submission_key}, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    save()
    if upload:
        hosted = client.submit_local_evaluation(
            prompt.id, dataset_name=dataset_name, dataset_fingerprint=dataset_fingerprint,
            scoring_fingerprint=policy, model=model, total=run["total"],
            completed=run["completed"], passed=run["passed"], errors=run["errors"],
            idempotency_key=submission_key,
        )
        run["id"] = hosted["id"]
        save()
    return Evaluation(run=run, results=results)
