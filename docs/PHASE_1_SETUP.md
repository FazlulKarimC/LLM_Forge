# Phase 1: authenticated workspaces

Clerk handles sign-in. LLMForge stores users, organizations, owner/member
memberships and projects in PostgreSQL. All organization members can work in
its projects; only owners can create projects. Invitations are deferred.

## 1. Configure Clerk

1. Create an application at https://dashboard.clerk.com.
2. Enable email sign-in or your preferred social provider.
3. Open API Keys and copy the publishable key and secret key to
   `frontend/.env.local`. Use `frontend/.env.example` as the field reference.
4. Copy the application's Frontend API URL to `CLERK_ISSUER_URL` in
   `backend/.env`. It must include `https://` and belong to that same Clerk app.
5. Set `CLERK_AUTHORIZED_PARTIES=http://localhost:3000`. For a deployed frontend,
   add its exact origin, separated by commas, and also update `CORS_ORIGINS`.
6. Configure allowed frontend URLs/domains in Clerk for the deployed app.

Use these values locally (replace placeholders with values from the same Clerk app):

`frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_YOUR_KEY
CLERK_SECRET_KEY=sk_test_YOUR_KEY
```

`backend/.env` (keep your existing database/provider settings):

```dotenv
CLERK_ISSUER_URL=https://YOUR_APP.clerk.accounts.dev
CLERK_AUTHORIZED_PARTIES=http://localhost:3000
CORS_ORIGINS=http://localhost:3000
```

Copy the actual Frontend API URL from Clerk; do not guess its hostname. No JWT
template or Clerk Organizations configuration is needed: organizations and
projects belong to LLMForge's database. Keep secret keys out of chat and Git.
See [Clerk's Next.js quickstart](https://clerk.com/docs/nextjs/getting-started/quickstart)
and [JWT verification guide](https://clerk.com/docs/guides/sessions/manual-jwt-verification).

The backend uses the issuer's JWKS to verify RS256 signatures, issuer, expiry,
not-before, user/session identity and authorized party. It does not require the
Clerk secret key. An optional `CLERK_JWT_PUBLIC_KEY` enables verification without
a JWKS request. Set `CLERK_AUDIENCE` only if the session token has that audience.

Missing keys display a setup screen. Missing/invalid backend credentials deny
protected API access; there is no development authentication bypass.

## 2. Install and initialize

From `backend`, activate the virtual environment and run:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Migration `i2j3k4l5m6n7` intentionally truncates experiments, prompt versions,
background jobs and dependent benchmark results, then requires project ownership.
This follows the personal-project reset decision. Downgrading cannot restore
those discarded records. It does not recreate the database itself.

From `frontend`:

```powershell
npm install
npm run dev
```

Restart both processes after environment changes. Open http://localhost:3000,
click Launch App and sign in. The first authenticated workspace request creates
a personal organization and default project atomically. Repeat requests reuse
them. Open Settings to create another organization or project.

## 3. Verify the demonstration

1. Sign in as user A; create a benchmark in the default project.
2. Create a second project in Settings. The benchmark list should be empty there.
3. Switch back; the first benchmark should still be visible.
4. Sign in as user B in another browser profile. User A's projects and records
   must not be available, including when an old detail URL is pasted.
5. Sign out; protected pages redirect to sign-in and API calls return 401.

Prompt versioning, datasets, evaluations, and SDK workflows are now implemented.
See the Phase 2–4 guides for the current demo sequence.

## API contract

- `GET /api/v1/workspaces`: provision current user and list memberships/projects.
- `POST /api/v1/workspaces/organizations`: create organization + default project.
- `POST /api/v1/workspaces/organizations/{id}/projects`: owner-only creation.
- Existing experiment/result/prompt APIs require `Authorization: Bearer <session>`
  and `X-Project-ID: <uuid>`.
- Non-member projects and inaccessible records return 404.
- Session tokens are obtained fresh from Clerk and never stored in browser storage.
- Only the selected project ID is persisted, under a user-scoped storage key.
- Query caches are recreated on project/account changes.
- Internal background work uses trusted, already-authorized experiment/job IDs.
- API startup no longer globally resets running jobs belonging to other processes.

## Tests

```powershell
# backend
python -m pytest -q
# frontend
npm test
npm run lint
npx tsc --noEmit
npm run build
```

`test_workspace_auth.py` verifies signatures and uses an isolated database to test
real route authorization, tenant filtering, exports, comparisons and job polling.
Existing mocked route tests have an explicit test-only identity override.
Live sign-in still requires your Clerk application and browser verification.
