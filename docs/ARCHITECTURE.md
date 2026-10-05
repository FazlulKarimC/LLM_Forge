# Architecture foundation

LLMForge keeps Next.js, FastAPI and PostgreSQL. Configuration, execution and analytical reads have separate boundaries without introducing a hosted queue. This adopts Langfuse's architectural principles at learning-project scale; generic application tracing and online evaluation rules are later milestones.

## Boundaries and contracts

Dashboard and SDK routes enforce current authentication, project isolation and scopes, then call services. Evaluation execution claims/checkpoints and analytical SQL live in `backend/app/repositories`. Evaluator lifecycle and execution orchestration live in services; simple CRUD does not need a repository wrapper.

Frontend prompts, datasets and evaluations own their components, adapters and hooks in `frontend/src/features`. Shared primitives stay in `components/ui`. React Query owns server records with project-aware keys; URL parameters identify shareable selections. Drafts and credentials stay in editor state. Existing account/project cache clearing, query cancellation and unsaved-change guards remain required. Compatibility exports at old component/API paths keep consumers stable during the transition.

Pydantic response models define the wire contract. Export deterministically without starting lifespan or connecting to dependencies:

```sh
cd backend
python -m scripts.export_openapi
cd ../frontend
npm run contracts:generate
```

Commit both `contracts/openapi.json` and `frontend/src/lib/generated/api-schema.ts`. Frontend view models may refine generated wire types for editor defaults; transport and authentication remain unchanged. The Python SDK retains ergonomic methods/models and validates sample responses against the exported schema. CI rejects contract drift before deployment sync.

## Execution and interruption

Every dispatched run receives a persisted execution attempt referencing immutable prompt, dataset, evaluator and policy snapshots. Provider keys exist only in the accepting process's memory, bound to the exact accepted execution attempt. A superseded task cannot claim a replacement attempt or discard its fresh credentials. A global capacity row serializes claims across processes; short transactions lock the run and claim the execution using `SKIP LOCKED`. At most two evaluation attempts execute globally; project admission remains two active runs. Execution is sequential within each run. Benchmarks retain their separate dispatch path.

Ownership heartbeats every 15 seconds and expires after 90 seconds. Reconciliation runs every 30 seconds and dispatch wakes the runner immediately. Every provider stage checks ownership before starting; checkpoints and case finalization check it again under lock. Cancellation and expired attempts cannot commit late results or start subsequent calls. A request already sent to a provider can still finish.

Generation is committed before judging. Each evaluator's checkpoint and all its typed scores commit together. Case results and counters finalize once. Recovery skips committed stages, including stages with ordinary provider/evaluator errors. Those errors count toward the run's existing result policy; they are not retried by recovery.

Lost ownership produces public status `failed`, internal attempt status `interrupted`, safe failure code `runner_interrupted` and recovery metadata. Provider-free work can recover automatically. Live recovery requires **Resume unfinished work** and fresh keys, preserves the same run and saved results, and appends an attempt. Remaining call estimates are upper bounds. A provider response lost before checkpointing may have been charged; recovery can repeat that call. Paid work is never automatically resumed. Completed/cancelled/ordinary terminal failures cannot use this action. Legacy runs without execution records are readable but cannot resume.

The backend cannot process work while its container is stopped. Multiple processes can claim safely, but live work remains tied to the process that received its credentials. Readiness checks database migration availability and runner health. Generic background-job retention only removes terminal jobs older than six hours; active jobs survive.

## Dispatch identity and reads

Creation, saved-output scoring, submissions and resume accept optional `Idempotency-Key` (1–128 visible ASCII characters). Records live for 24 hours, scoped by project, authenticated actor and operation. The non-secret intent fingerprint and resulting run commit in the same transaction. Matching replay returns the existing resource; different intent returns 409. Credentials are excluded from fingerprints and persistence. Authentication and authorization are evaluated on every replay. Without a key, behavior remains unchanged. Frontend actions retain a key after an uncertain response; mutations still have no automatic HTTP retries.

Dashboard evaluation paths add `/{run_id}/summary`, `/{run_id}/cases`, `/{run_id}/cases/{case_index}` and `/{run_id}/resume`; SDK paths add the same operations under `/sdk/evaluations/runs`. Existing full-detail endpoints remain compatible. Summaries exclude case payloads, show aggregates/usage/attempt history, and poll only active runs. Case pages contain bounded previews, server filters and comparison totals; opening a case fetches full content. Comparisons require the same dataset revision and retain scoring-policy warnings. Missing scores stay missing. New aggregates use typed score records; historical runs fall back to JSON checks. Generation usage is zero for saved-output scoring.

## Diagnostics and rollout

Platform JSON logs use an allowlist of request, project, run, attempt, stage, duration, outcome, status and exception-type fields. They exclude credentials, prompt content, outputs and arbitrary exception context. Safe failure codes and attempt history are exposed separately from content. Optional Sentry events are reduced to operational identifiers/types; this is not customer tracing.

Apply additive migration `o8p9q0r1s2t3` before starting the new backend. Existing completed runs remain readable. Downgrade refuses to discard execution records. CI runs the complete migration chain in a disposable PostgreSQL schema and real PostgreSQL locking tests before deployment sync. Start the new runner only after those checks pass. Rollback should preserve new records; do not downgrade after new dispatches. No Redis, ClickHouse, gateway or object storage is required.

Local validation:

```sh
cd backend
python -m pytest tests -q
python -m scripts.check_postgres
python -m scripts.check_migrations
python -m pytest ../sdk/python/tests -q
cd ../frontend
npm run lint
npx tsc --noEmit
npm test
npm run build
```

PostgreSQL checks use `LLMFORGE_TEST_POSTGRES_URL`, or the configured development database when explicitly invoking the scripts. They create uniquely named temporary schemas, qualify ORM tables explicitly (important with pooled connections), and remove only their own schemas. Migration validation uses a single transaction and rolls it back. SQLite tests exercise semantics but cannot establish locking guarantees.
