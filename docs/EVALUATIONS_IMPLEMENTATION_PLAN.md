# Evaluations: Langfuse review and implementation plan

Reviewed October 2, 2026. The comparison below records the starting point; the first implementation is described here and in the evaluation guide.

## First release implemented

Phase 0 consistency fixes and the reusable-evaluator / saved-output portions of Phases 1–2 are implemented. The library provides immutable versions, built-in checks, multi-output LLM judges, preview tests, required/informational selections, typed score records and separate scoring runs linked to saved outputs. Results show score coverage, errors, means/category counts and compatible numeric deltas. The SDK can discover evaluators, select pinned versions and score saved outputs; the CLI accepts custom/latest prompt labels.

The release retains PostgreSQL and the sequential FastAPI runner. It supports one generation variant, at most ten evaluator selections, two judges (including the inline judge) and 60 model calls per run. Judges use the existing 20-case limit. No additional hosted service is required. Existing result JSON remains readable; historical checks are adapted without inventing evaluator provenance. Development migration `n7o8p9q0r1s2` adds the evaluator/version/score tables and source-run link.

Named experiment groups, variant matrices, stable dataset-item identities, manual reviews, SDK callbacks, per-metric CI gates, idempotent dispatch, durable restart recovery and online scheduling remain future milestones. This is the bounded learning-project scope agreed after the infrastructure review, rather than completion of every phase below.

## Basis and recommendation

This review compares the user-supplied Langfuse evaluation implementation notes with LLMForge's current models, schemas, evaluation services, APIs, frontend, Python SDK, CLI and existing tests. Official Langfuse documentation was used to verify its product concepts. The checkout-specific queue internals in the supplied notes were not independently inspected.

LLMForge has a working offline evaluation product. Improve the usefulness and repeatability of that workflow before adding a production observability pipeline. Adopt Langfuse's separation of evaluator definition, target selection, execution and scores. Keep PostgreSQL as the primary store and retain immutable prompt/dataset snapshots, explicit provider credentials and the separate Benchmarks workspace.

## Current capability comparison

| Area | LLMForge today | Improvement |
| --- | --- | --- |
| Reproducible inputs | Saved prompt versions, immutable dataset revisions, saved non-secret run settings; text/chat supported | Add evaluator versions and scoring-policy snapshots to complete the reproduction contract |
| Deterministic evaluation | Seven assertion kinds, preflight validation, bounded regex execution | Reusable named definitions, declared score schemas and evaluator-specific summaries |
| LLM judging | One optional judge per run, one numeric score 0–1 plus a reason and threshold | Multiple named judges; structured numeric/boolean/categorical outputs, configurable mappings and an editor test action |
| Definition reuse | Assertions and judge configuration are inline per run | Project-scoped evaluator library with immutable versions |
| Targeting | Entire selected dataset revision | Explicit case selection/metadata filters first; online rules only after application-event ingestion exists |
| Execution | Sequential FastAPI background task; durable case progress, bounded calls and cancellation checks | Separate generation and scoring stages; later durable execution jobs and explicit restart recovery |
| Scores | Check objects embedded in result JSON; all selected checks must pass | Typed score records with evaluator/source provenance; required gates separate from informational metrics |
| Historical scoring | Generating a new run is coupled to scoring | Evaluate saved outputs again using a new evaluator version without another generation call |
| Comparison | Two runs, same dataset revision, pass/fail transitions, case detail and URL state | Full scoring-configuration compatibility, per-metric deltas, stable item identity and better reference lookup |
| Analytics | Pass rate, processed/failed/error counts; per-case generation latency/tokens; judge usage stored in check JSON | Coverage/error denominators, per-evaluator quality and generation/judge usage shown separately |
| External evaluation | Python SDK starts runs or submits application-computed outputs/checks/metrics; CLI pass-rate gate | Task/evaluator callbacks, declared external evaluator provenance, metric-specific baseline gates |
| Human evaluation | No persisted manual-review workflow in the evaluation hub | Case-level reviews, independent of automated results, then a small review queue |
| Online evaluation | No generic trace/observation pipeline | Optional later event ingestion, rules, deterministic sampling and recursion guards |

## Concrete issues to address first

1. `frontend/src/components/evaluations/evaluation-results.tsx` compares only `config.assertions` for its changed-check warning. Changing judge provider, model, rubric or threshold can change pass/fail without that warning. Compare the entire non-secret scoring definition; distinguish quality regressions from incomparable scoring policies.
2. Evaluation setup starts with temperature 0 and max tokens 256, without loading supported values from the selected prompt version's saved config. Add an explicit **Use saved prompt settings** action, show applied values/overrides, validate bounds, and snapshot the effective generation settings. Do not silently apply arbitrary config keys or provider credentials.
3. `sdk/python/src/llmforge/cli.py` still restricts `--label` to staging/production even though prompt management now supports custom labels and latest. Accept the shared label contract and retain mutual exclusion with `--version`.
4. The frontend pass rate is over processed cases; the SDK's pass rate is over total cases. Label provisional versus final values clearly and provide processed coverage, scored coverage, check failures and execution errors separately. Never convert missing scores into a zero score.
5. The reference selector shows the currently loaded run-history page. Add a server-filtered, paginated reference search scoped to the dataset revision, with useful labels and compatibility information. Preserve URL-restored references that are outside the current page.

## Proposed architecture

```text
Immutable prompt + dataset revision -> generation -> saved case outputs
                                                   |
Pinned evaluator versions + input mappings ---------+
                                                   v
                                      evaluation executions -> typed scores
                                                   |
                                summaries / comparisons / review / CI gates
```

Generation should be usable with several scoring passes. Existing runs remain valid; introduce a distinct scoring-pass identity attached to a source run rather than replacing its outputs or historical scores.

Add project-scoped `Evaluator` and immutable `EvaluatorVersion` models. Start with built-in assertions and LLM judges, using a common input context: inputs, output, expected output and case metadata. Store the declared score names/types/ranges/categories and default mapping with the version; mapping validation must precede paid calls.

Add scoring-pass evaluator assignments, per-case/per-evaluator execution records and typed score records. A record should identify source run/result, evaluator version, assignment/mapping, score definition, value, reason, source (built-in/judge/external/human), timestamps and usage where applicable. Keep arbitrary settings in bounded JSON; use relational identities for querying and provenance. Store pass policies separately from raw score values.

Resolve and pin evaluator versions when a scoring pass is created. The supplied Langfuse checkout resolves the latest evaluator at worker pickup; LLMForge should favor explicit reproducibility instead. Editing an evaluator after dispatch must not alter queued work.

Use one executor/result contract for built-ins, judges and externally computed scores. An execution can emit several scores. Quality failure is a valid result, while provider/schema failure is an execution error. Informational scores do not automatically fail a case. Preserve the existing all-checks-required policy for legacy runs.

Write terminal execution state and score rows in a short PostgreSQL transaction. User-visible completion should mean the committed scores can be queried. LLMForge does not need Langfuse's separate blob/ClickHouse ingestion path for this contract.

## Delivery sequence

### Infrastructure fit and learning-project scope

The repository's deployment configuration uses a Vercel frontend, a single FastAPI/Uvicorn container on Hugging Face Spaces, and remote Neon/PostgreSQL storage. This is a configuration review, not a measurement of current account quotas or deployed capacity.

Phases 0–3 can use these existing services: evaluator versions, typed scores, reviews and scoring passes are ordinary database records; comparison and configuration changes are API/frontend work. Built-in evaluation runs on the backend CPU; custom task/code callbacks run on the SDK user's machine or in CI. No Redis, ClickHouse or additional hosted worker is required for the initial release.

Start with one generation variant, sequential evaluation and at most two selected judges. Enforce a total call budget before enabling multiple judges: the existing 20-case judge limit does not by itself bound expanded evaluator/variant counts. Show generation and judge call counts before starting; preserve existing project admission, timeout and dataset limits. A 20-case run with one generation and two judges can make 60 model calls. Score-only passes avoid generation calls, but LLM judging still consumes provider quota.

Phase 4 can persist/claim work in PostgreSQL and execute inside the existing container. It can improve interruption handling and restart recovery, but cannot guarantee processing while the container is stopped or provider credentials have been lost. Keep explicit recovery and re-entry of credentials; do not promise unattended paid-call resumption. Reliable continuous online evaluation in Phase 5 is a separate availability requirement and should remain deferred.

The initial implementation should omit the multi-variant matrix and automatic background scheduling. Add those only after the single-variant offline workflow is useful and its storage/call volume has been measured. Apply additive migrations to the intended deployment database before deploying code that requires them.

### Phase 0 — Fix consistency and comparison contracts

Address the five concrete issues above. Keep existing APIs and saved runs compatible. Add comparison compatibility reasons, explicit effective settings and searchable references.

Acceptance: a rubric/threshold change is visibly identified; saved config can be applied and overridden; custom/latest CLI labels work; incomplete/error cases do not masquerade as measured quality; references are reachable across history pages.

### Phase 1 — Reusable evaluator library and typed scores

Add immutable evaluator versions, declared score definitions, a normalized score adapter for legacy checks and an executor registry. Provide create/edit/archive/select flows and a **Test evaluator** preview using one saved output or a pasted sample. Preview must show compiled inputs, validated scores, reasons, usage and errors without becoming a normal experiment score.

First score types: boolean, bounded numeric and declared categorical. Use comments for explanation; free-form qualitative score types can wait. Preserve historical result JSON for compatibility; expose old check data through an adapter rather than inventing evaluator-version provenance for it. Extend tenancy filters and write scoping for every new model.

Acceptance: one evaluator is reused across datasets; edits create a new version; historical runs keep their original definition; one judge can return several validated metrics; invalid/missing outputs remain explicit errors.

### Phase 2 — Score saved outputs and improve experiment analysis

Introduce scoring passes that reference immutable saved outputs, multiple evaluator selections and required/informational score policies. Add a named experiment group for runs on the same dataset revision, while keeping a single variant as the default. A small model/prompt variant matrix can then create separately identifiable child runs.

Show evaluator means or category distributions, coverage, errors, generation/judge token usage and latency separately. Estimated monetary costs require known prices and visible assumptions; unknown prices remain unknown. Compare raw per-metric deltas as well as pass transitions, with a compact configuration diff and explicit candidate/reference labels. Add filters for evaluator, failed metric, errors and regressions.

Acceptance: a new rubric can score existing outputs with zero generation calls; scoring passes preserve prior scores; incompatible evaluator versions are identified; summaries agree with case-level results; small multi-variant runs share reproducible inputs.

### Phase 3 — Dataset identity, feedback and SDK/CI workflows

Add stable dataset-item IDs and bounded metadata (for example language, category and difficulty). Retain immutable revision snapshots. Cross-revision comparison requires both stable identity and matching case input/reference fingerprints; changed cases are shown separately rather than assumed equivalent.

Add **Add to dataset draft** from a failed/reviewed case with provenance and an explicit save/new revision. Persist manual scores/comments separately from automated scores, with author/time history. A simple unreviewed/reviewed queue is sufficient initially.

Extend the Python SDK with `run_experiment(task, evaluators)` around the existing external-submission workflow so application pipelines can be evaluated, rather than only a prompt call. Add request idempotency for submissions and named score provenance. Extend CLI gates with explicit per-metric thresholds, maximum errors and baseline deltas; comparisons require compatible datasets/scoring definitions. Preserve existing exit codes and pass-rate flags.

Acceptance: dataset reordering does not lose identity; changed cases are flagged; human review never overwrites automated history; custom RAG/agent tasks can submit outputs/scores; duplicate requests do not create duplicate submissions; CI identifies the metric causing a failure.

### Phase 4 — Execution reliability

Separate orchestration, provider execution and persistence through injectable interfaces. Add durable per-stage jobs, atomic claims, leases/heartbeats, fencing and deterministic identities for each scoring pass/case/evaluator. Reuse proven patterns in the existing experiment job lifecycle where appropriate; an additional Redis service is not initially necessary.

Use bounded concurrency, project admission/call budgets and cancellation checks before every generation/judge call. Scope deterministic identities to a pass so an intentional reevaluation creates new work. Distinguish retryable database work, pre-send failures and ambiguous provider outcomes; do not claim exactly-once paid calls.

Keep credentials request-only. On restart, deterministic scoring can recover from saved outputs; work requiring lost provider credentials remains interrupted and offers an explicit resume/new-pass flow with supplied credentials. Do not quietly introduce stored provider keys or automatic paid replays. Carry forward completed results and make any potential repeated call visible.

Acceptance: duplicate dispatch is harmless; stale workers cannot finalize a newer attempt; cancellation persists through in-flight completion; restart retains outputs/scores and clearly identifies credential-dependent interrupted stages. Include out-of-order completion, partial judge failure and duplicate-score tests.

### Phase 5 — Optional online evaluation

Only after the offline workflow is complete, introduce a small authenticated application-event ingestion API with a stable event ID and immutable input/output/metadata snapshot. Then add rules with filters, enabled state and deterministic sampling, and assignments to the same pinned evaluator versions used offline. Manual historical selection uses the same scheduling/execution path.

Exclude evaluator-generated events to prevent recursive evaluation. Make rule/assignment removal and permission changes explicit cancellation conditions. Keep source events, evaluation jobs and score visibility distinct in the UI. This is a new product capability, not a hidden extension to existing dataset runs.

Acceptance: duplicate events schedule once per rule/assignment; sampling is repeatable; stopped rules do not start fresh work; manual runs remain individually identifiable; evaluator outputs cannot trigger an endless evaluation loop.

## Frontend placement

Keep **Evaluations** as the main workspace entry, with **Runs** and **Evaluators** tabs. A global Scores page can wait until scores are used outside experiments.

- Runs: compact toolbar with server filters above history; searchable dataset/prompt/evaluator identity, status and comparison actions.
- New run: Dataset/revision, Prompt/model variants, Evaluators; sticky summary with effective settings, case count and generation/judge call counts. Keys remain explicit and transient.
- Evaluator detail: identity/version header; definition and variable mappings on the left, sample input and test results on the right; usage/history in secondary tabs.
- Run detail: summary and Reference/Candidate controls above the case table; score columns and evaluator filters; selected-case detail shows inputs, expected output, variant outputs, score reasons and usage. Stack detail panels on narrow screens.
- Dataset detail: item metadata and revision history; reviewed failures enter a draft, never mutate a revision already used by a run.

Follow `DESIGN_SYSTEM.md`: shared primitives/tokens, compact controls, explicit versions, table/detail layouts, URL-preserved selection, accessible labels and visible error/loading/empty states.

## Scope and verification

The recommended first release is Phases 0–2. Dataset feedback/SDK improvements and execution reliability follow; online rules remain an optional advanced milestone. Do not add ClickHouse, blob-storage ingestion, decision-model providers, arbitrary uploaded server-side code execution or a full annotation-team product now. Custom code belongs in SDK task/evaluator callbacks first.

Implementation should include additive migrations, project isolation for every new record, old-run compatibility, schema/score validation, version pinning, judge errors/injection test cases, request-only secret exclusion, score-only call counts and meaningful browser checks. Later phases add job recovery/idempotency tests. Existing backend, frontend and SDK regression suites remain required; browser checks cover saved evaluator previews, immutable version edits and scoring existing outputs.

## Local verification — October 2, 2026

- Backend: 575 tests passed; new coverage includes immutable versions, stale edits, tenant/key scoping, typed judge responses, secret exclusion, missing source outputs and cancellation between judges.
- Frontend: 54 tests passed; lint, type checks and optimized production build passed. Tests cover preview/version behavior, a slow preview within the provider deadline, required gates, full scoring-policy warnings, saved settings, selection request payloads and selected evaluator state across form remounts.
- Python SDK: 29 tests passed, including custom/latest CLI labels, evaluator selection, saved-output scoring and nested judge-key redaction. The wheel was built and its new client methods imported from a separate local install.
- The additive migration was applied to the development database. Signed-in browser checks created evaluator v1/v2, previewed a sample and scored the existing Echo demo outputs with zero model calls; the original 100% run remained intact while the new passing policy produced 50%. Desktop/mobile layouts and the incompatible-policy warning were reviewed.
- Live judge execution was verified with mocked provider responses rather than paid calls. Deployment, remote CI and production-database migration are not implied by these local checks.

## Sources and code anchors

- User-supplied Langfuse evaluation notes: evaluator/rule/assignment, snapshot scheduling, latest-at-pickup behavior, queues, retry semantics and score ingestion.
- [Langfuse core concepts](https://langfuse.com/docs/evaluation/core-concepts): reusable evaluators, targeting rules, offline/online workflows.
- [Langfuse scores](https://langfuse.com/docs/evaluation/scores/overview): typed evaluation results and sources.
- [Langfuse LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge): reusable judge configuration and testing.
- [Langfuse code evaluators](https://langfuse.com/docs/evaluation/evaluation-methods/code-evaluators): common evaluator inputs and multiple score outputs.
- LLMForge: `backend/app/models/evaluation.py`, `backend/app/schemas/evaluation.py`, `backend/app/services/evaluation_runs.py`, `backend/app/services/evaluation_service.py`, `backend/app/api/sdk_evaluations.py`, `frontend/src/components/evaluations-workbench.tsx`, `frontend/src/components/evaluations/evaluation-results.tsx`, `sdk/python/src/llmforge/client.py`, `sdk/python/src/llmforge/models.py`, `sdk/python/src/llmforge/cli.py`, `docs/PHASE_3_EVALUATIONS.md`.
