# Retest remediation — 9 October 2026

Changes are based on the retest of commit `f18e0ba`. Remediation and validation
were completed locally without hosted writes, provider calls or a production
restart. Publication to `main` was subsequently authorized; hosted validation
remains separate from the local results below.

## Findings and changes

| Finding | Evidence and remedy | Verification / remaining limit |
| --- | --- | --- |
| RETEST-N001: failed browser creation | Supplied logs confirm `DBAPIError` / PostgreSQL `DeadlockDetectedError` during admission at 15:13:47 UTC. Admission held the project row with `FOR UPDATE` and then locked active runs; progress held a run while inserting children with project foreign keys. Shared admission now uses `FOR NO KEY UPDATE`, allowing FK key-share checks while still serializing admissions. Counts and history reads no longer lock every live run. | Real PostgreSQL tests cover project-lock/checkpoint compatibility, overlapping admissions, duplicate intent and the two-run capacity limit. |
| RETEST-N002: partial mock run failed | Logs place the generation checkpoint interruption for `cbefdbfc-95c1-48e6-9252-d8d13018ab2c` alongside a history deadlock. Its runner event omitted the exception type, so the exact historical exception remains unproven. The same lock conflict is eliminated; aborted database transactions retry at most three times in fresh sessions. | Tests verify checkpoint retry without repeating generation, partial-run preservation, and explicit resume after exhausted retries. Only SQLSTATE `40P01` and `40001` retry. Ambiguous connection failures and model calls never retry automatically. Exhausted aborted writes become `database_interrupted`; unrelated failures remain ordinary failures. Existing hosted failed rows are not rewritten. |
| P001: database overhead | Successful hosted traces showed multi-second fences/checkpoints/finalization and negligible mock generation/scoring. Provider-boundary ownership checks now use a single unlocked read; durable writes retain atomic fencing. Read-side expiry locks only expired candidates and rechecks the current lease/status after obtaining locks. Multi-run reconciliation uses stable run order. | Local measurements below; cancellation, stale owners, recovery, checkpoint uniqueness and cross-project isolation all pass. Hosted latency remains to be measured after deployment. |
| Missing diagnostic IDs | Workbench create, saved-output scoring and resume failures now show request IDs and explain checking history before retrying an uncertain dispatch. Existing unchanged dispatch identities are retained. Timing/runner failures include safe exception types; retry events include SQLSTATE and retry number. DBAPI errors receive the safe existing HTTP response inside handled middleware, avoiding an unhandled SQLAlchemy traceback with bound parameters. | UI test covers an uncertain create followed by a retry with the same identity. Backend test verifies handled HTTP 500, correlation header/body, and exclusion of exception SQL/parameters. |
| SDK saved evaluators / waiting | Public guide and SDK README include pinned prompt/dataset IDs, `evaluators=[{"version_id": ...}]`, `assertions=[]`, discovery, and a bounded 300-second wait. Timeout guidance preserves the original run ID and warns against creating another run without checking history. | Docs render assertions and all 31 SDK tests pass. SDK runtime/default timeout unchanged. |
| Optional discovery failures | Capabilities remain informational. | The actual workbench test starts successfully with unavailable capabilities, then verifies uncertain-dispatch handling. Existing deployment-info failure tests pass. |
| U001: hosted expired-session checks | Existing recovery implementation retained; its seven local WorkspaceProvider tests pass, including stale session/project gating. | Live expired/revoked Clerk-session checks remain unverified: this batch did not invalidate hosted credentials. Prior browser fixtures are supporting local evidence, not hosted proof. |

The project lock choice follows PostgreSQL's documented compatibility between
`FOR NO KEY UPDATE` and `FOR KEY SHARE`; see the
[row-lock documentation](https://www.postgresql.org/docs/current/explicit-locking.html#LOCKING-ROWS).
In SQLAlchemy this mode is `with_for_update(key_share=True)` with `read=False`.

## Local comparison

Both runs used the same disposable local PostgreSQL 16 container, four echo cases,
actual SDK over HTTP and the lifecycle runner. Before uses the archived `f18e0ba`
source; after uses this working tree. Each fixture includes one first sample and
ten warm samples. Counts include polling and runner activity. Generation/evaluator
checkpoints and case-finalization commits are still independent.

The live-row polling regression also fails at its three-second deadline against
the archived pre-fix source and passes against the working tree. The expected
baseline failure is recorded in `backend/artifacts/retest-baseline-polling.log`.

| Fixture | Warm SDK p50 / p95 before | After | Average SQL statements before → after |
| --- | --- | --- | --- |
| Inline exact match | 534.82 / 685.68 ms | 357.93 / 441.75 ms | 187.4 → 153.4 |
| Saved builtin only | 452.69 / 520.76 ms | 321.11 / 336.43 ms | 186.4 → 153.0 |

First samples: inline 681.91 → 480.34 ms; saved builtin 524.17 → 423.90 ms.
These results do not reproduce remote Neon/network conditions or establish a
hosted latency guarantee. Full samples are retained in ignored local artifacts:
`backend/artifacts/retest-benchmark-before.json` and `retest-benchmark-after.json`.

## Cleanup and validation

Removed 27 unused imports and the unused `init_db` / `close_db` wrappers. Alembic
and the real lifecycle/session management remain in place. Runtime dependency
review found uses for the declared backend/frontend libraries, including dynamic
RAG/statistics imports, so no runtime package was removed without evidence.

- Backend: **606 passed**, including real PostgreSQL concurrency checks.
- SDK: **31 passed**.
- Frontend: **74 passed** in the complete suite, then the updated workbench/docs
  tests **15 passed**, adding one workbench regression (**75 total tests**).
- ESLint, TypeScript, production build, changed-file Python import checks and
  deterministic OpenAPI export pass.

Native Gemini generation and a new field-aware comparator remain deferred.
Hosted expired-session/revoked-key checks, isolated crash/restart tests and the
deployment retest require their original prerequisites; no local pass is labeled
as hosted validation. The next deployment should re-run overlapping mock runs,
browser admission/history and safe session recovery checks.
