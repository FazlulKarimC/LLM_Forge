# LlmForge frontend

Next.js 16 and React 19 interface for a personal prompt-development and reasoning-benchmark project. Clerk handles sign-in; the selected organization and project scope API requests. The app uses Tailwind CSS v4, TanStack Query, Lucide icons, and optional Sentry reporting. Components and styling follow the repository's [design system](../DESIGN_SYSTEM.md).

## Main workflows

- `/dashboard` shows project activity and backend readiness. New evaluations use local SDK cases and summary uploads.
- `/prompts` and `/prompts/[id]` manage immutable versions, playground calls, release labels, and project API keys.
- `/evaluations` provides local SDK instructions, summary history and compatible reference reports. Existing hosted results retain case inspection and recovery. `/datasets` redirects to the SDK guide.
- `/experiments`, `/experiments/new`, `/experiments/[id]`, and `/experiments/compare` cover bounded reasoning benchmarks, historical attempt inspection, exports and statistical comparison. **Stop run** preserves committed cases; reruns create independent attempts and keep pinned baselines attached to their original attempt.
- `/settings` manages organizations and projects. `/docs` is a public walkthrough.

The benchmark builder reads limits and bundled case counts from `/api/v1/capabilities`: the default hosted profile allows 20 cases/calls, or five ReAct cases with three iterations. New hosted RAG execution is paused. Attempt selectors refresh as execution status changes, including the first run whose attempt number does not change. Case pages load 25 summaries at a time and fetch one case's full content on selection. Historical comparison accepts explicit attempt numbers; unknown provenance and incomplete coverage do not become passing regression evidence.

**Unpin baseline** stays available for an existing baseline after failed/interrupted reruns and while a new attempt is queued/running. **Pin as baseline** is offered only for completed non-baselines. Unpinning preserves historical results.

**Export complete JSONL** downloads all retained cases for the selected terminal attempt. **Export then purge** downloads before asking to remove an attempt; the API protects active, pinned and referenced history. Active-run polling backs off from 3 to 10 to 30 seconds and pauses in hidden tabs. Custom-provider credentials stay in the current form/request, are cleared on accepted runs/provider changes, and are not saved in browser storage; custom execution remains development-only.

## Local setup

Use Node.js 20.9+ and start the [backend](../backend/README.md) with a migrated PostgreSQL database. Copy `.env.example` to `.env.local` and set:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_REPLACE_ME
CLERK_SECRET_KEY=sk_test_REPLACE_ME
```

The frontend and backend must use the same Clerk application. Set the backend's `CLERK_ISSUER_URL` and `CLERK_AUTHORIZED_PARTIES` as described in [workspace setup](../docs/DEVELOPMENT.md#authentication). The API URL must include `/api/v1`; without it, a production build's browser requests would fall back to localhost. Configure `NEXT_PUBLIC_SENTRY_DSN` only if you want Sentry reporting.

The existing deployment uses Vercel with `NEXT_PUBLIC_API_URL=https://faz-ai-llm-forge.hf.space/api/v1`. Public environment values are embedded at build time. Frontend `/api/version` and backend `/api/v1/version` report separate source/build identities; see the [deployment guide](../docs/DEVELOPMENT.md#existing-free-hosting-profile) before comparing releases.

```bash
npm ci
npm run dev
```

Open <http://localhost:3000>, sign in, and select a project. The app shows an auth-setup screen if Clerk keys are absent. The [local SDK walkthrough](../docs/DEVELOPMENT.md#local-sdk-evaluations) provides a model-free path to exercise the product.

## Checks and production build

```bash
npm test
npx tsc --noEmit
npm run lint
npm run build
npm run start
```

Stop `next dev` before building because both use `.next`. CI runs lint, typecheck, Vitest, and the production build with a fresh `npm ci`. Component tests do not replace a browser walkthrough of authentication, project selection, and live API flows.

## Code map

`src/app/(app)` contains authenticated pages; `src/components` contains domain components and UI primitives. `src/lib/api-client.ts` handles request context, timeouts, safe read retries, and API errors. Mutations are not retried automatically. `src/instrumentation.ts`, `src/instrumentation-client.ts`, and the Sentry server/edge files configure optional error reporting.
