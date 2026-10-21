# LLMForge Python SDK

A synchronous Python 3.10+ client for prompt releases and local evaluation gates. Install from this repository; the package is not published to PyPI.

```bash
python -m pip install -e ./sdk/python
```

Configure a full API root and a project key from Settings. The SDK can use the hosted API without running a local backend:

```powershell
$env:LLMFORGE_URL = 'https://faz-ai-llm-forge.hf.space/api/v1'
$env:LLMFORGE_API_KEY = '<project key from Settings>'
```

For your own migrated backend, use `http://localhost:8000/api/v1`. Keep project keys in environment variables or CI secrets. Default keys read prompts; uploading reports requires **Allow evaluations for CI**. Existing keys do not gain scopes automatically.

## Fetch and compile prompts

```python
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("support/answer", label="production")
    print(prompt.version, prompt.compile(query="I was charged twice"))
```

`get_prompt(name)` selects production. Use `label="staging"`, a custom label, `label="latest"`, or `version=2` explicitly; label and version are mutually exclusive. Missing labels and archived prompts return errors. There is no automatic fallback. Names may include folder paths.

Text prompts compile to strings; chat prompts compile to ordered role/content messages. Supply keyword inputs or a string mapping. Mustache and restricted Python-brace templates match server substitution rules, without expressions, loops, attributes or formatting specifications. Applications decide how to apply `prompt.config`. Snapshots also expose tags, labels, description and content hash.

An in-memory cache holds up to 128 snapshots. Every fetch checks the server with an ETag; promotions and revocations apply on the next fetch. Network/auth failures raise errors. Compiling an already fetched snapshot remains local.

## Create and release prompts

Owner-created keys can opt into `prompts:write`. Use separate read-only keys in applications that only consume releases.

```python
with LLMForge() as forge:
    prompt = forge.create_prompt(
        "support/answer", "Reply in {{language}}: {{query}}",
        config={"temperature": 0}, tags=["support"],
        commit_message="Initial prompt",
    )
    forge.set_prompt_label("support/answer", prompt.version, "staging")
    catalog = forge.list_prompts(tag="support", label="staging", limit=50)
```

Creating an existing name saves its next immutable version. Type stays fixed; omitted tags inherit and config describes the new snapshot. Supply `base_version` for stale-write protection. Optional `labels=["production"]` explicitly assigns a release; otherwise production stays unchanged. `latest` follows the newest version automatically. Roll back by assigning an earlier version. Mutations are not retried automatically.

## Evaluate local cases

Save an `Echo` text prompt with template `{{query}}`, then run:

```python
with LLMForge() as forge:
    prompt = forge.get_prompt("Echo", version=1)
    report = forge.evaluate_local(
        prompt,
        [{"id": "hello", "inputs": {"query": "Hello"},
          "expected_output": "Hello"}],
        task=lambda compiled: compiled,
        evaluators={"exact_match": lambda output, case:
                    output == case["expected_output"]},
        scoring_version="exact-match-v1",
        dataset_name="Local greetings", model="local-echo-demo",
        upload=True, report_path="artifacts/evaluation.json",
    )
    assert report.meets_threshold(1.0)
```

The echo task makes no model calls. Replace it with your application/provider callback; it receives the compiled text or chat messages and returns a string. Each named evaluator receives `(output, case)` and returns a boolean. Up to ten checks are supported. Errors are counted separately and always fail a gate.

Cases may be a list or a local JSON/JSONL path. Each needs a unique stable string `id` and string-valued `inputs`; `expected_output` and JSON `metadata` are optional. The SDK accepts 1–1,000,000 cases, processing them sequentially in memory. This is a count bound, not a streaming interface or resource guarantee.

The dataset fingerprint hashes the selected cases, including IDs, inputs, references and metadata, sorted by ID. Row order does not change identity; case/subset changes do. The scoring fingerprint includes evaluator names and explicit `scoring_version`. Change that version when evaluator code/settings change. Pin the prompt version and record application/provider versions yourself.

Full inputs, references, metadata, outputs and checks are written to the local artifact before upload. Provider keys are never submitted. The server receives only prompt version, display names, fingerprints, counts and optional metrics; it creates no hosted dataset or case rows. Use `upload=False` to skip submission. The default is `upload=True`; uploads need an evaluation-enabled key.

The returned `Evaluation` exposes `id`, `status`, `pass_rate`, `meets_threshold(minimum)` and `to_dict()`. Uploaded summaries appear in **Evaluations → Run history**. Compatible references match both data and scoring fingerprints. Scores and identities are self-reported; the server cannot independently verify private execution or show per-case regressions.

For an existing local runner, call `submit_local_evaluation(prompt_version_id, dataset_name=..., dataset_fingerprint=..., scoring_fingerprint=..., model=..., total=..., completed=..., passed=..., errors=..., status="completed", metrics={...}, idempotency_key=...)`. It returns the hosted run dictionary. The endpoint accepts at most **64,000 original request bytes**, 20 finite named metrics and consistent terminal counts. The byte limit includes JSON whitespace/escapes and is enforced before parsing, even without `Content-Length`; oversized bodies return HTTP 413 without creating a report. Normalized summaries are also bounded to 64,000 bytes. The endpoint rejects case payloads and unknown fields.

After a failed upload, the artifact retains the full report and `submission_key`. Retry the unchanged summary with that key; do not execute the model again just to retry ingestion. Replays return the existing run for 24 hours; changed intent returns 409. Keys are scoped by project, SDK key and operation. No mutation retries automatically.

## CLI and CI

The CLI's local command uses an **echo task and exact-match check**, for a provider-free example. Live integrations use the Python callback API above.

```bash
python -m llmforge evaluate-local --prompt Echo --version 1 --dataset ./cases.json --upload --min-pass-rate 1 --output artifacts/evaluation.json
python -m llmforge check artifacts/evaluation.json --min-pass-rate 1
```

`--upload` is explicit on the CLI; omit it for local-only reports. Prompt fetching still requires API access. Exit codes are **0** passed, **1** failed checks or case errors, **2** configuration/transport/incomplete-run failure. Any case error fails the gate regardless of the threshold. Offline `check` validates report shape/counts and applies the gate; it cannot independently verify supplied scores. Full reports may contain private data, so choose their artifact destination deliberately.

Examples: [fetch prompts](examples/fetch_prompt.py), [local evaluation](examples/submit_results.py), [GitHub Actions](../../docs/examples/evaluation-gate.yml).

## Legacy compatibility

Hosted dataset authoring has been removed from the product. Dataset writes and old demo creation return 410 by default. Existing hosted datasets/results remain preserved, and older integrations may explicitly enable `LEGACY_DATASET_WRITES=true` on their backend.

Legacy `get_dataset`, `start_evaluation`, `submit_evaluation`, `score_evaluation` and CLI `evaluate` operate on existing hosted revisions. They are not needed for the current local workflow. The evaluator catalog and saved-output scoring remain available for historical hosted runs; local summaries cannot be resumed or re-scored without their local outputs.

`get_evaluation`, `get_evaluation_summary` and `wait_for_evaluation` can inspect saved runs. `list_evaluation_cases` and `get_evaluation_case` return content only for legacy hosted runs. `resume_evaluation` requires interrupted work with durable checkpoints and fresh request-only provider credentials. CI never resumes paid work automatically. See [architecture and compatibility contracts](../../docs/ARCHITECTURE.md#local-evaluations-and-legacy-compatibility).

HTTP requests use a ten-second default timeout (`LLMForge(timeout=...)`). A legacy wait may exceed its deadline by an in-flight request timeout; inspect the same run ID after a timeout before creating another run.

## SDK development

```bash
python -m pip install -e './sdk/python[test]'
python -m pytest sdk/python/tests
python -m pip wheel --no-deps ./sdk/python -w dist
```

Use the [development guide](../../docs/DEVELOPMENT.md) for contracts, PostgreSQL migrations and the SDK/CLI HTTP smoke check. Record SDK/source versions separately from backend/frontend build identities.
