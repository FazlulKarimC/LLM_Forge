# LLMForge Python SDK

A small synchronous client for prompt releases and evaluation gates. Requires Python 3.10+. Install from this repository (the package is not published to PyPI):

```bash
python -m pip install -e ./sdk/python
```

Configure `LLMFORGE_URL` with the API root, for example `http://localhost:8000/api/v1`, and `LLMFORGE_API_KEY` with a project key created in Settings. Use HTTPS for a hosted backend. Keep keys in environment variables or your CI secret store.

For a local model-free walkthrough, sign in to the web app, select a project, and click **Create demo examples** on Overview. Create a project key with **Allow evaluations for CI** enabled in Settings, then use `Echo demo` and `Greetings` in the examples below. The backend must have its migrations applied; a project key grants access only to its project.

```python
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("support-classifier", label="production")
    text = prompt.compile(query="I was charged twice")
    print(prompt.version, text)
```

`get_prompt(name)` selects production. Select a custom label, `label="latest"`, or `version=2` explicitly; labels and versions are mutually exclusive. Unassigned labels and archived prompts return an error; pinned/latest reads do not need production to be assigned. There is no fallback to another release. Names may use folder paths such as `support/answer`. Compile accepts keyword inputs or a string mapping. Mustache and restricted Python-brace templates match the server's single-pass substitution rules. No attributes, expressions, loops, conversions or format specifications are evaluated.

Text prompts compile to a string. Chat prompts compile to an ordered list of `role`/`content` messages (system, user, assistant). `prompt.config` contains the saved version's JSON configuration; `prompt.tags`, `prompt.labels`, `prompt.description`, and `prompt.sha256_hash` expose metadata. Applications decide how to apply config. Runtime message placeholders and tool calls are not supported.

## Create and release prompts

Enable **Allow prompt creation and label changes (prompts:write)** when an owner creates a new key in Settings. Existing/default keys remain read-only. This capability can move production; use a separate read-only key in applications that only fetch prompts.

```python
with LLMForge() as forge:
    prompt = forge.create_prompt(
        "support/answer",
        [
            {"role": "system", "content": "Reply in {{language}}."},
            {"role": "user", "content": "{{query}}"},
        ],
        config={"temperature": 0, "max_tokens": 256},
        tags=["support"],
        commit_message="Initial chat prompt",
    )
    messages = prompt.compile(language="English", query="How do I reset my password?")
    forge.set_prompt_label("support/answer", prompt.version, "staging")
    catalog = forge.list_prompts(tag="support", label="staging", limit=50)
```

Creating the same named prompt again saves its next version. Type is fixed; omitted tags inherit, while config describes the new snapshot and defaults to `{}`. Supply `base_version` for stale-write protection. Optional `labels=["production"]` explicitly assigns labels in the creation transaction; otherwise production remains unchanged. `latest` follows the newest version automatically and cannot be assigned. To roll back, move a label to an earlier version. Exact duplicate content/format/config is rejected. Creation and label changes are never retried automatically.

An in-memory cache holds up to 128 prompt snapshots. Every fetch contacts the server with an ETag, so promotions/revocations take effect on the next fetch. Network/auth failures raise an error; there is no offline fallback. Compilation of a snapshot you already fetched remains local.

## Run an evaluation

Create a key with **Allow evaluations for CI** enabled. Existing/default keys only read prompts. Evaluation keys read project datasets, run/cancel evaluations, and submit results; they do not edit prompts/datasets or manage keys.

```python
with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo", version=1)
    dataset = forge.get_dataset("Greetings", version=1)
    run_id = forge.start_evaluation(
        prompt.id, dataset["revision"]["id"],
        assertions=[{"kind": "exact_match"}],
    )
    result = forge.wait_for_evaluation(run_id, timeout=180, cancel_on_timeout=True)
    assert result.meets_threshold(1.0)
```

Mock mode echoes the compiled prompt and does not measure model quality. For live generation pass `provider`, `model`, and the provider's `api_key`. `judge` accepts a dict with `provider`, `model`, `api_key`, `rubric`, and `threshold`. These request-only keys aren't persisted. Server limits and per-case error behavior are documented in [Phase 3](../../docs/PHASE_3_EVALUATIONS.md).

`wait_for_evaluation` returns completed, failed or cancelled runs. Its timeout raises `EvaluationTimeout` with `run_id`. Default timeout behavior leaves the run running; `cancel_on_timeout=True` attempts cancellation. An in-flight provider call may still finish. HTTP requests use a 10-second timeout by default (`LLMForge(timeout=...)`); there are no automatic retries, avoiding duplicate runs/submissions. A wait can exceed its deadline by an in-flight HTTP request timeout.

## Reuse evaluators and score saved outputs

Create evaluators in the dashboard, then use an evaluation-enabled project key to discover and pin their version IDs. The catalog is paginated (default 50). SDK keys select definitions; they do not author or archive evaluators.

```python
with LLMForge() as forge:
    catalog = forge.list_evaluators()
    evaluator_version_id = catalog["items"][0]["latest"]["id"]
    selected = [{"version_id": evaluator_version_id, "required": True}]
    run_id = forge.start_evaluation(
        prompt_version_id, dataset_revision_id,
        assertions=[], evaluators=selected,
    )
    result = forge.wait_for_evaluation(run_id)
    scoring_id = forge.score_evaluation(result.id, selected)
    rescored = forge.wait_for_evaluation(scoring_id)
    print(rescored.score_summary)
```

Supply at least one required evaluator when omitting inline assertions/judge. `assertions=[]` removes the SDK's default exact-match assertion. Judges require `api_key` in their selection; keys are request-only and excluded from saved definitions, reports and error text. Informational selections use `required=False` and do not gate pass/fail. A version change creates a new ID; pin it explicitly for repeatable CI.

`score_evaluation` creates a separate run linked to a completed source, reuses its outputs and makes no generation calls. Missing source outputs stay errors. Judges still consume provider calls. Limits are ten saved selections, two judges including the inline judge, 20 cases with judges, and 60 model calls per run. Typed summaries keep missing/error coverage separate from measured means; `Evaluation.to_dict()` includes available summaries.

## Submit your application's results

```python
with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo", version=1)
    dataset = forge.get_dataset("Greetings", version=1)
    results = []
    for index, case in enumerate(dataset["revision"]["cases"]):
        output = prompt.compile(case["inputs"])  # replace with your application's model call
        results.append({
            "case_index": index, "output": output,
            "checks": [{"name": "exact_match", "passed": output == case["expected_output"]}],
        })
    run_id = forge.submit_evaluation(
        prompt.id, dataset["revision"]["id"], model="local-echo-demo",
        results=results, metrics={"cases_checked": float(len(results))},
    )
```

Submit exactly one result per dataset case, with either an output plus at least one check or an error. Check scores (optional) are 0–1; metrics are named finite numbers. The backend derives pass/error counts from checks and labels the run **External evaluation**. Submitted checks/metrics are self-reported, not independently verified; no provider is called. Results appear in the normal grid and can be compared on the same dataset revision.

## CI command

```bash
python -m llmforge evaluate --prompt "Echo demo" --version 1 --dataset Greetings --dataset-version 1 --min-pass-rate 1 --output artifacts/evaluation.json
python -m llmforge check artifacts/evaluation.json --min-pass-rate 1
```

Exit codes: **0** passed, **1** quality regression or case errors, **2** configuration/transport/timeout/incomplete-run failure. Every case error fails the gate regardless of the pass-rate threshold. The evaluate command attempts to cancel a run on timeout. Pin prompt/dataset versions for reproducibility; use `--label staging` to gate the current candidate release.

Live CLI runs require `LLMFORGE_PROVIDER_API_KEY`, `--provider`, and `--model`. `--assertions rules.json` loads an array of Phase 3 assertions. CLI judging is configured through the Python API rather than a key-bearing command-line argument. The offline `check` command needs no credentials. Reports contain inputs/outputs; upload them as CI artifacts only where intended.

`check` validates the report shape and evaluation counts before applying the gate. It cannot independently verify self-reported results in a JSON file; use `evaluate` against a saved prompt and dataset for a server-produced run. The CLI is available as either `llmforge` or `python -m llmforge` after installation.

Examples: [prompt fetch](examples/fetch_prompt.py), [external results](examples/submit_results.py), [GitHub Actions](../../docs/examples/evaluation-gate.yml).

```bash
python -m pip install -e './sdk/python[test]'
python -m pytest sdk/python/tests
python -m pip wheel --no-deps ./sdk/python -w dist
```
