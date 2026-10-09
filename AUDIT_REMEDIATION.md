# Hosted consumer audit remediation

Date: October 9, 2026. Scope: approved implementation-plan items 1–5 in the
LLMForge repository. Optional L-002 scoring extension remains deferred.

## Evidence and version boundaries

Inputs were the consumer handoff, test matrix, findings, and Practical
improvements report. The handoff also repeats the test matrix; duplicate rows
are not counted again. Original IDs are retained below. Unnumbered observations
keep their report descriptions rather than receiving invented audit IDs.
Referenced raw JSON artifacts/screenshots were not attached, so detailed
authenticated hosted results remain attributed to the consumer reports.

Audited SDK source and pre-remediation repository HEAD both equal
`64980bd071d93a1a68d58826c46c4c4c495c267f`. SDK package version and deployed API
version are `0.1.0`; this does not establish source parity. Hosted backend and
frontend source commits were unknown. Remediation was developed from that
checkout; it does not change the separate audit app. Audit-time deployed
identities must not be inferred from the remediation source.

Read-only hosted checks reproduced localhost-only onboarding, the four server
providers, Assertion fields kind/value/path, and ready/degraded status with a
healthy evaluation runner and missing RQ worker. No hosted authenticated
bootstrap, resource mutation, restart, access-control change or paid call was
performed. Workspace GET can provision a user, so it was not used as an anonymous
claim of read-only authenticated verification.

## Finding map

| Original ID / report item | Classification and diagnosis | Implementation | Verification and limitations |
| --- | --- | --- | --- |
| **U-001** | Confirmed current recovery usability gap; historical rejection cause unverified. October 8 failure preceded successful October 9 login; correction timing is not recorded. | Typed missing-token errors and 401 handling; Refresh session bypasses cached token; Sign in again ends rejected session and preserves same-origin path/query/fragment. Other errors retain retry. Uses existing panels/tokens. | Component tests cover missing tokens, 401, repeated rejection, sign-out destination, 503 and gated/private state. Browser fixture imports the actual WorkspaceProvider with synthetic auth/transport responses: refresh, sign-out/sign-in return and narrow layout passed. A real hosted expired Clerk session was not retested. |
| **U-001: authorized-party configuration** | Deployment problem supplied by owner; local signed JWT reproduces 401 for trailing slash and success for exact origin. It plausibly explains historical rejection but causation is unproven. | Exact-origin setup/example documentation and signed-token regression. Auth validation remains exact; no relaxation or production configuration edit. | Hosted origin is `https://llmforge.vercel.app`, without slash/path. Wrong origin remains rejected. Owner should retain the corrected deployed value. |
| **U-002** | Confirmed docs/usability mismatch, reproduced in repository and hosted public guide. | Hosted HTTPS `/api/v1` example and separate local-backend target; clarified that installing SDK locally does not require running FastAPI/PostgreSQL. | Rendered guide test and actual local production-build browser check. SDK/API behavior unchanged. Existing hosted guide remains old until frontend deployment. |
| **L-001** | Advertised capability limitation, not prompt-fetch/submission failure. Gemini is absent from the server generation contract. | Public provider-free `/api/v1/capabilities` derived from generation/judge Literal types; displayed in new evaluation setup and public guide; documented local Gemini/LangGraph submission route and credential distinction. | Endpoint tests match OpenAPI provider enums without DB access. UI covers successful discovery and older/unavailable endpoint. Existing SDK external submissions and unknown-provider rejection coverage pass. Native Gemini adapter remains deferred. |
| **L-002** | Missing reference-field comparator. Runtime tests/probes confirm whole-reference JSON versus path-to-fixed-literal semantics. | Clarified existing behavior and external deterministic checks in public/SDK/Phase 3/4 docs. No new assertion implemented. | Existing comparator regression coverage passes. Field-to-reference extension was optional and is deferred; external numeric deltas remain disabled. |
| **P-001** | Measured hosted performance concern; no proven infrastructure cause or SLA defect. Evaluation runner is separate from optional RQ benchmark dispatch. | Separate allowlisted admission/dispatch/claim/queue/fence/generation/scoring/checkpoint/finalization timing; local PostgreSQL SDK/HTTP/runner benchmark; omit empty inline-assertion stages for saved-evaluator-only runs. Keep generation/evaluator checkpoints, fencing and atomic finalization. | Before/after measurements below. Deterministic write budget verified on SQLite and PostgreSQL, including pass/fail evidence and score coverage. Existing cancellation, interruption/resume, scope/idempotency tests pass. Hosted latency root cause remains unresolved; no production speed claim. |
| **Release provenance (unnumbered)** | Missing diagnostic capability; API/package versions do not identify deployed source. | Backend `/api/v1/version`, HF staging embeds upstream GitHub SHA/build identity; frontend `/api/version` captures its own build identity; public guide shows both independently. | Tests cover populated, absent and malformed metadata, and exclusion of unrelated fields. Local browser shows explicit unknown identities. Existing deployments gain identity only after a rebuild; source is not guessed from SDK or HF staging commit. |
| **SDK authoring discoverability (unnumbered)** | Usability/docs gap; authoring/import/run discovery already worked. | Documented dataset import → preview → apply draft → save revision; saved evaluator creation/preview/version IDs; SDK consumption/scopes and run history; external comparison warning even with matching check names. | Public guide regression and browser review, SDK/evaluator suites. No new authoring methods, key scopes or direct-DB workflows. |
| **Readiness / missing worker (unnumbered)** | Deployment observation, not a proven evaluation bug. Inline benchmark dispatch is an existing fallback. | Clarified evaluation runner versus benchmark RQ dispatch in docs and this record. | Read-only hosted ready/degraded observed; PostgreSQL benchmark uses actual evaluation runner. No worker, queue or deployment change. |
| **Crash/recovery / local repeat (unnumbered)** | Consumer hosted crash test deferred; source recovery behavior already implemented. | Ran existing local interruption/resume, automatic provider-free recovery, fencing and real PostgreSQL locking coverage; real local SDK/HTTP/runner repeated fixtures added. | Injected interruption tests and real DB checks pass. Real hosted restart and full OS-process crash/restart walkthrough remain unverified; no production restart was authorized. |
| **SDK installation restriction (unnumbered)** | Consumer environment workaround, not platform defect. | No SDK install behavior change. | SDK suite passes in the repository environment. |
| **Timeout assertion correction (unnumbered)** | Consumer harness mistake: completed run with 1 ns deadline. Corrected nonterminal probe passed. | No timeout semantic change or monkey-patch. | Existing SDK deadline/cancellation tests pass; no unconditional timeout guarantee added. |
| **Baseline urgency/model errors (unnumbered)** | Ordinary schema-valid model/prompt outcomes, not platform defects. | No changes to consumer prompts, graph or model calls. | Consumer reported baseline 9/12 and candidate 12/12 on one fixed sample. No broader quality claim and no additional Gemini calls. |

## Provider-free measurements

Disposable local PostgreSQL 17 container, four immutable echo cases, mock provider,
same exact-match policy, actual synchronous SDK over HTTP, lifecycle runner.
Each fixture has one first-run sample and ten warm samples. Before used timing
instrumentation with the original empty-inline-stage behavior; after changed only
the empty-stage behavior. These measurements do not reproduce remote Neon/network
conditions or the hosted browser run's different configuration.

| Fixture | Before SDK p50 / p95 | After SDK p50 / p95 | Average observed SQL statements before → after | Average observed commits before → after |
| --- | --- | --- | --- | --- |
| Inline exact-match | 400.13 / 642.01 ms | 477.71 / 622.77 ms | 183.8 → 185.0 | 24.3 → 24.5 |
| Saved builtin only | 474.19 / 497.37 ms | 404.77 / 437.69 ms | 222.0 → 184.0 | 29.0 → 24.0 |

Statement/commit counts include SDK polling and runner activity. Saved-only work
removes four empty checkpoint transactions for four cases; the observed total
commit reduction also includes polling differences. Inline timings vary and do
not show a consistent improvement. No performance guarantee is inferred.

`backend/artifacts/audit-benchmark-before.json` and `audit-benchmark-after.json`
retain every sample, first-run versus warm timing, lifecycle timing and stage
aggregates. The script refuses hosted DB targets and requires explicit
LLMFORGE_TEST_POSTGRES_URL on localhost; it removes its generated schema and HTTP
server. Reports contain no keys, inputs, outputs or token values. Optional
`--max-p95-ms` supports an environment-specific regression budget. A provisional
local 1,000 ms budget is separate from an agreed hosted SLA, which does not exist.

## Verification

- Backend default suite: **593 passed, 4 skipped**; PostgreSQL-only tests are
  skipped without explicit dedicated DB configuration.
- PostgreSQL ownership/concurrency/fencing: **3 passed**. Timing/write-budget and
  recovery selection, including the real PostgreSQL variant: **14 passed**.
- SDK complete suite: **31 passed**; no public SDK method or payload behavior changed.
- Frontend complete suite: **72 passed**, lint/types and production build passed.
- Full migration chain passed inside a disposable PostgreSQL schema/transaction,
  then rolled back. No schema migration was added for these changes.
- Repeated OpenAPI/type regeneration retained identical SHA-256 hashes;
  final `git diff --check` passed.
- Browser: actual local built guide shows hosted URL, provider choices and unknown
  build identities; real recovery component in an explicitly synthetic fixture
  passed refresh, persistent rejection/sign-in return, and 503 retry distinction.
  Recovery at 375 px and 1,280 px remains usable. Screenshots under `frontend/artifacts/` show
  only public guidance or fixture data; no live auth/session tokens were used.
- Local threshold verification: ten warm runs per fixture passed the provisional
  1,000 ms p95 budget; measured inline p95 **495.45 ms**, saved-builtin p95
  **397.13 ms** (`backend/artifacts/audit-benchmark-threshold.json`).
- All generated test schemas were removed (zero remained before cleanup).
  Temporary preview processes and the dedicated PostgreSQL container were removed;
  no listeners remained on their ports. Verification artifacts remain ignored locally.

Frontend Node execution required escaping the filesystem sandbox after installed
Vitest dependency resolution returned EPERM. Docker also required local runtime
access outside the sandbox. Neither was diagnosed as an application failure.

## Deployment/configuration actions

1. Retain exact `CLERK_AUTHORIZED_PARTIES=https://llmforge.vercel.app`; verify the
   same Clerk issuer/application and existing frontend CORS origin. No key scopes
   or tenant/access-control changes are required.
2. Deploy backend discovery/timing changes before the frontend consumes discovery.
   An older backend remains usable; frontend shows discovery unavailable.
3. HF sync stamps the GitHub source SHA/build identity during staging. Alternate
   builds can supply non-secret BUILD_COMMIT/BUILD_ID. Vercel identity is captured
   from its build metadata; alternate frontend builds use explicit
   LLMFORGE_FRONTEND_COMMIT/LLMFORGE_FRONTEND_BUILD_ID. Missing identity stays null.
4. Rebuild/deploy the frontend for recovery and guide updates. This work adds no
   migration; existing durable-evaluation migrations must already be applied.
5. After deployment, inspect public version/capability/readiness endpoints and use
   authorized deployment logs to attribute hosted latency. Hosted mutation or
   paid provider benchmarks require separate authorization.

Implementation verification used local/disposable resources and read-only hosted
checks. On October 9, after those checks passed, the owner authorized committing,
merging and pushing to main, including the resulting automatic deployments.
Remote CI and deployment status must be checked after publication. The separate
consumer audit application and its existing hosted resources were left unchanged.

## Deferred

Optional L-002 field-to-reference assertion; native Gemini server adapter; a new
queue, hosted tracing, vector/database replacements and generalized callbacks.
Current evidence supports neither broad architecture changes nor production
configuration mutations as routine debugging.
