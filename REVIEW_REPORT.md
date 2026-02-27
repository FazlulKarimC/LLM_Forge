# REVIEW_REPORT

This report captures a prioritized improvement plan for `LlmForge` after a deep review of frontend, backend, and free-tier deployment paths (Vercel Hobby + HF Spaces Free).

## Priority Levels

- **P0 (Critical):** fix before next public deploy
- **P1 (High):** stability and correctness improvements
- **P2 (Medium):** free-tier optimization and product polish
- **P3 (Low):** long-term quality and learning improvements

---

## P0 - Critical Fixes (Before Next Deploy)

- [ ] **Fix `Create & Run` race condition** so clicking `Create & Run` always starts execution
  - Files: `frontend/src/app/experiments/new/page.tsx`
- [ ] **Prevent experiments getting stuck in `queued`** when enqueue fails
  - Make status transition atomic (or rollback/reset on enqueue error)
  - Files: `backend/app/api/experiments.py`
- [ ] **Add minimal auth protection for mutating APIs** (`create`, `run`, `delete`)
  - Files: `backend/app/api/experiments.py`, `backend/app/main.py`
- [ ] **Harden custom base URL behavior** (avoid unsafe URLs; ensure production-safe defaults)
  - Files: `backend/app/core/config.py`, `backend/app/api/experiments.py`, `backend/app/services/inference/openai_engine.py`
- [ ] **Fix frontend lint errors** (`no-explicit-any`, unescaped entities, unused vars)
  - Files: `frontend/src/lib/api.ts`, `frontend/src/app/error.tsx`, `frontend/src/app/experiments/page.tsx`, `frontend/src/app/not-found.tsx`, `frontend/src/app/experiments/[id]/page.tsx`
- [ ] **Fix backend pytest collection issue** (exclude non-test text artifacts)
  - Files: `backend/pytest.ini`

---

## P1 - High Impact Stability/Correctness

- [ ] **Add backend dashboard stats endpoint** (avoid client-side counting from only first 100 rows)
  - Files: `backend/app/api/experiments.py` (or new stats route), `frontend/src/lib/api.ts`
- [ ] **Standardize API error response shape** across all routes
  - Files: `backend/app/core/exception_handlers.py`, `backend/app/api/results.py`, `backend/app/api/experiments.py`
- [ ] **Sync docs with implementation** for custom run execution headers
  - Files: `README.md`, `frontend/README.md`
- [ ] **Add request timeout/retry strategy in frontend API client** for cold starts and transient failures
  - Files: `frontend/src/lib/api.ts`
- [ ] **Migrate Pydantic class-based config to `ConfigDict`** to remove deprecation warnings
  - Files: `backend/app/schemas/run.py`, `backend/app/schemas/result.py`
- [ ] **Add startup preflight checks** for required env vars by mode (DB, tokens, queue)
  - Files: `backend/app/main.py`, `backend/app/core/config.py`

---

## P2 - Free-Tier Friendly Feature Improvements

- [ ] **Add readiness panel in UI** (DB/queue/inference mode/health summary)
  - Files: `frontend/src/app/dashboard/page.tsx`, `backend/app/api/health.py`
- [ ] **Add pre-run estimator** (samples, expected API calls, rough latency/token budget)
  - Files: `frontend/src/app/experiments/new/page.tsx`
- [ ] **Add retry actions** (`Retry failed`, `Recompute metrics only`)
  - Files: `frontend/src/app/experiments/[id]/page.tsx`, `backend/app/api/results.py`, `backend/app/api/experiments.py`
- [ ] **Add optional data-retention cleanup** for old runs/results (free DB storage safety)
  - Files: backend service/API + optional script
- [ ] **Improve queue mode clarity** for single-container free-tier deployments
  - Support explicit inline mode fallback when worker infra is absent
  - Files: `backend/app/api/experiments.py`, deployment docs

---

## P3 - Code Quality and Learning Enhancements

- [ ] **Remove remaining loose typing (`any`) and tighten TypeScript models**
  - Files: `frontend/src/lib/api.ts`, `frontend/src/app/error.tsx`, `frontend/src/app/experiments/[id]/page.tsx`
- [ ] **Improve keyboard/accessibility on clickable rows and custom modal interactions**
  - Files: `frontend/src/app/dashboard/page.tsx`, `frontend/src/app/experiments/page.tsx`
- [ ] **Fix invalid/fragile utility classes** (e.g., wrapping behavior)
  - Files: `frontend/src/app/experiments/compare/page.tsx`
- [ ] **Batch DB writes for run creation** to improve throughput on free-tier DBs
  - Files: `backend/app/services/run_service.py`, `backend/app/services/experiment_service.py`
- [ ] **Add targeted regression tests** for critical flows
  - `Create & Run` behavior
  - enqueue failure status handling
  - dashboard stats correctness
  - Files: `backend/tests/*`, frontend test setup (if added)

---

## Suggested Execution Order

- [ ] **Milestone 1 (P0):** critical correctness + security + lint gate
- [ ] **Milestone 2 (P1):** API reliability and observability hardening
- [ ] **Milestone 3 (P2):** free-tier UX and productivity features
- [ ] **Milestone 4 (P3):** quality and performance refinement
