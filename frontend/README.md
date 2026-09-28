# LlmForge frontend

Next.js 16 and React 19 interface for a personal prompt-development and reasoning-benchmark project. Clerk handles sign-in; the selected organization and project scope API requests. The app uses Tailwind CSS v4, TanStack Query, Lucide icons, and optional Sentry reporting. Components and styling follow the repository's [design system](../DESIGN_SYSTEM.md).

## Main workflows

- `/dashboard` shows project activity and backend readiness. **Create demo examples** creates a provider-free prompt and dataset without replacing existing content.
- `/prompts` and `/prompts/[id]` manage immutable versions, playground calls, release labels, and project API keys.
- `/datasets` edits versioned cases; `/evaluations` runs and compares assertions or optional rubric-judged results.
- `/experiments`, `/experiments/new`, `/experiments/[id]`, and `/experiments/compare` cover the legacy benchmark, result inspection, export, and statistical comparison. The detail page has **Stop run** for a stranded queued/running attempt; partial results remain available.
- `/settings` manages organizations and projects. `/docs` is a public walkthrough.

## Local setup

Use Node.js 20.9+ and start the [backend](../backend/README.md) with a migrated PostgreSQL database. Copy `.env.example` to `.env.local` and set:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_REPLACE_ME
CLERK_SECRET_KEY=sk_test_REPLACE_ME
```

The frontend and backend must use the same Clerk application. Set the backend's `CLERK_ISSUER_URL` and `CLERK_AUTHORIZED_PARTIES` as described in [workspace setup](../docs/PHASE_1_SETUP.md). The API URL must include `/api/v1`; without it, a production build's browser requests would fall back to localhost. Configure `NEXT_PUBLIC_SENTRY_DSN` only if you want Sentry reporting.

```bash
npm ci
npm run dev
```

Open <http://localhost:3000>, sign in, and select a project. The app shows an auth-setup screen if Clerk keys are absent. The dashboard's demo setup and the [evaluation walkthrough](../docs/PHASE_3_EVALUATIONS.md) provide a model-free path to exercise the product.

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
