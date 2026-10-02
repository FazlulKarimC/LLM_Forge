# Phase 3: datasets and evaluations

## Try a reliable demo

1. Sign in and select your project.
2. On Overview, click **Create demo examples**. This creates or reuses `Echo demo` v1 (`{{query}}`) and the two-case `Greetings` dataset in the current project, then opens **Evaluations** with both selected. Existing content with these names is never overwritten; a conflict is shown instead.
3. Keep **Demo** and **Exact match**. Start the run.
5. Both cases pass: Demo echoes the compiled prompt. It does not call a model or claim to measure real model quality.
6. Save prompt v2 as `Reply: {{query}}`. Run it against the same dataset revision. Both cases fail exact match.
7. Open the candidate from **Run history** and choose the passing run as its **Reference run**. Inspect the regressed cases and their outputs. Copy the page URL to revisit the selected runs, or export the results as JSON.

This demonstrates a reproducible regression test: a prompt change breaks the expected output contract. For a practical model demo, use a small support-ticket classification dataset, a prompt that returns a JSON label, JSON-path assertions, and an optional correctness rubric.

## Dataset format and editing

Cases are a JSON array. Inputs map prompt variable names to strings. `expected_output` is a string (or null when no reference exists), and `name` is optional.

```json
[
  {"name": "Greeting", "inputs": {"query": "Hello"}, "expected_output": "Hello"},
  {"name": "Farewell", "inputs": {"query": "Goodbye"}, "expected_output": "Goodbye"}
]
```

CSV uses `input.variable` columns, `expected_output`, and optional `name`:

```csv
input.query,expected_output,name
Hello,Hello,Greeting
Goodbye,Goodbye,Farewell
```

Alternatively, use an `inputs` column containing a CSV-quoted JSON object. Don't mix both input styles. Unknown/duplicate headers, malformed rows, non-string inputs, empty datasets, and oversized revisions are rejected. A missing CSV reference column means null; an empty reference cell means the empty string.

Use the case table for add/edit/duplicate/remove, or switch to Advanced JSON. Upload a file or paste content, validate and review its preview, explicitly replace the draft cases, then save. Import validation does not create a dataset until you save. Dataset metadata can be edited, and datasets can be archived/restored. Archiving preserves revisions and existing results but prevents new runs. Dataset names are unique within a project.

Saved revisions are immutable. Editing cases creates a new revision with a stale-edit check. Selecting an older revision loads a draft; saving it creates a new revision. **Reload latest / discard draft** explicitly discards unsaved work. Dataset history and lists are paginated.

## Evaluation checks

All inline checks and required saved evaluators must pass for a case to pass. Informational evaluators record scores without determining that outcome; their execution errors remain visible in score coverage.

| Check | Behavior |
| --- | --- |
| Exact match | Output equals that case's reference, including whitespace/case. Every case needs a reference. |
| Contains | Case-sensitive substring match. |
| Regex | Pattern search with a 50 ms execution limit; use anchors for a whole-output match. |
| Valid JSON | Output parses as JSON. Markdown fences fail this deterministic check. |
| JSON equals | Compare parsed JSON, ignoring object key order. Booleans and numbers remain distinct. |
| JSON matches each reference | Compare parsed output with each case's `expected_output`; every reference must be valid JSON. Object key order is ignored. |
| JSON path equals | Traverse dot-separated object keys / numeric array indexes, then compare to a JSON literal. Example: path `answer.label`, value `"billing"`. Keys containing dots aren't supported. |
| LLM judge | Separate provider/model, rubric, and passing threshold. Requires a JSON score 0–1 and reason. Malformed responses produce case errors. |

Judge scores are model opinions, may vary, and remain susceptible to prompt injection from candidate content. The judge receives a rubric and a delimited JSON data object; deterministic checks remain the best choice for strict output contracts. Judge usage and latency are stored with its check; generation usage and latency appear in the grid.

## Reusable evaluators and saved-output scoring

Open **Evaluations → Evaluators** to create a named built-in check or LLM judge. Each save creates an immutable version with notes; concurrent stale edits return a conflict and preserve the draft. Archive prevents new selections while preserving history. New runs explicitly select saved versions, which are pinned with the full non-secret definition and passing policy at dispatch.

Built-ins emit one boolean score. Judges can emit up to five named boolean, bounded numeric or categorical scores in one provider call per case. Numeric scores use a declared range and passing threshold; categorical scores use declared categories and passing categories. Every declared output must be present with the correct type. Mapping is limited to case inputs, candidate output and expected output; arbitrary code/expressions are not executed. Boolean scores retain their type rather than accepting numbers as booleans.

**Test evaluator** previews the exact saved version against one pasted sample, showing validated values, reasons and judge usage. Unsaved definitions must be saved first. Preview creates no evaluation run or persisted score. Judge previews make one explicit provider call and need a request-only key.

On a completed run, expand **Score saved outputs again**, select evaluators and start a new scoring run. It references the original run, retains its outputs and historical scores, and makes zero generation calls. Cases without an output remain errors; the system does not generate replacements. This also works for application-submitted outputs. At least one evaluator must be required. A judge can still make paid calls while scoring saved outputs.

Results show per-score coverage, execution errors, missing scores, numeric/boolean means and categorical counts. Means exclude missing/error cases. The pass-rate denominator is all dataset cases, including errors and pending cases, matching SDK/CLI behavior; incomplete runs label it provisional. Judge usage appears with its scores separately from generation usage. Scoring-only runs reuse the source generation metadata, which is historical usage rather than a new generation call.

Compare runs on the same dataset revision using the server-filtered, paginated reference lookup. A changed assertion, judge rubric/provider/model/threshold, evaluator version or required policy produces a warning and disables numeric deltas. Externally supplied scores have no pinned evaluator definition and do not get numeric deltas. Generation settings can be explicitly loaded with **Use saved prompt settings**; only valid temperature/max-token values are applied, and the effective values remain visible and overridable.

## Live inference and limits

Choose Groq, OpenRouter, or OpenAI and supply your provider key. You may configure a separate judge key. Keys remain only in the form/request/worker memory; they are excluded from run configuration, database rows, exports, and query caches. The form clears keys after a successful start, provider changes, and workspace switching. No provider keys are taken implicitly from the server environment for evaluations.

Limits: 100 cases / 1 MB per dataset revision; 50 cases per live generation run; 20 cases when judging; 10 inline assertions and 10 saved evaluator selections; at most two judges including the inline judge; at most 60 model calls (generation plus judging); two active runs per project; output limit 2,048 tokens and 32,000 characters. Built-in checks do not consume model calls. Setup shows expected call counts, and the backend enforces limits before dispatch. Each generation/judge call has a 30-second deadline and no automatic retries. Missing prompt inputs and invalid definitions are rejected before calling a provider. A case-level provider/judge error is recorded and subsequent cases continue. A completed run can contain errors; inspect case errors and per-score errors.

The runner uses FastAPI background tasks, sequential calls, and short database transactions. Polling shows durable progress. Cancellation prevents subsequent calls and won't be overwritten by an in-flight call, which may still consume provider credits. Restarting the backend loses in-memory credentials; after 120 seconds without progress, polling marks that run failed. Runs are never automatically resumed or retried. This fits a small personal project with a persistent backend process; serverless request lifecycles aren't appropriate for this runner.

## API and models

All endpoints require Clerk Bearer authentication and `X-Project-ID`. Read-only SDK keys from Phase 2 cannot create datasets or run evaluations.

- `GET/POST /api/v1/datasets`
- `POST /api/v1/datasets/import` — validated cases from `{format: "csv"|"json", content: "..."}`
- `GET/PATCH/DELETE /api/v1/datasets/{id}` — soft archive on DELETE
- `GET/POST /api/v1/datasets/{id}/revisions` — POST includes `cases` and `base_version`
- `GET/POST /api/v1/evaluations` — POST returns 202
- `GET /api/v1/evaluations/{id}` — run plus ordered per-case results
- `POST /api/v1/evaluations/{id}/cancel`
- `POST /api/v1/evaluations/{id}/score` — new run using saved outputs and pinned evaluator selections
- `GET/POST /api/v1/evaluators` — project-scoped catalog and creation
- `GET/PATCH /api/v1/evaluators/{id}` — detail and archive/restore
- `GET/POST /api/v1/evaluators/{id}/versions` — immutable versions; updates require `base_version`
- `POST /api/v1/evaluators/versions/{version_id}/test` — one-sample preview
- `POST /api/v1/demo/setup` — idempotent project-scoped provider-free example setup

Run creation selects `prompt_version_id`, `dataset_revision_id`, provider/model/generation settings, `assertions`, and optional `judge`. Run history snapshots prompt/dataset names, version numbers, and non-secret settings. Comparisons align case indexes only when both runs use the same dataset revision. Existing benchmark experiments remain a separate workflow.

Tables: `evaluation_datasets`, `dataset_revisions` (bounded cases JSON), `evaluation_runs`, `evaluation_case_results`, `evaluators`, `evaluator_versions`, `evaluation_scores`. Base migration: `k4l5m6n7o8p9`; evaluator migration: `n7o8p9q0r1s2`. All tables are project-scoped. New scores are committed with case results; legacy JSON results are adapted for summaries without a historical backfill or invented evaluator identity. No separate queue service is required. Apply migrations to the intended database before deploying this code.

```powershell
cd backend
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m alembic upgrade head
```

Automated coverage includes import validation, revision conflicts, assertions, regex timeout, tenant isolation, provider failures, secret exclusion, cancellation races, abandoned runs, judge parsing, and UI save/run/comparison flows. Phase 4 remains the packaged Python SDK, CI developer workflow, and documentation/landing page polish.

For a database-backed smoke test in a temporary workspace (removed afterwards), run `python -m scripts.smoke_evaluations` from `backend`. Add `--live-groq` to make one generation and one judge call with the configured `GROQ_API_KEY`. This explicit smoke-test flag is the only evaluation test that reads the server's provider key.
