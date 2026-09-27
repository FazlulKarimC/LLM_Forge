# Running the regression checks

From the repository root, install the backend development dependencies and SDK test extra:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\backend\venv\Scripts\python.exe -m pip install -e './sdk/python[test]'
```

Run the backend suite in a shell configured for mock inference and a non-production database URL:

```powershell
cd backend
$env:INFERENCE_ENGINE = 'mock'
$env:DATABASE_URL = 'postgresql+asyncpg://test:test@127.0.0.1:1/test'
.\venv\Scripts\python.exe -m pytest tests -q
```

The workspace, prompt and evaluation integration tests use the shared `api` fixture in `backend/tests/conftest.py`, with an isolated SQLite database and fresh application per test. Both request sessions and background evaluation sessions use that database. `support.py` handles test sign-in/provisioning; `factories.py` creates checked resource scenarios. Test modules do not import each other's fixtures or setup functions.

The old mocked benchmark-route tests explicitly request `legacy_route_identity` using `pytestmark`. This override is limited to those modules, restored afterwards, and does not apply to the workspace/security integration tests. The application has no test authentication bypass.

Run the SDK suite separately from the root:

```powershell
.\backend\venv\Scripts\python.exe -m pytest sdk/python/tests -q
```

SDK tests are grouped into HTTP client/cache behavior, template compilation, and CLI quality gates. Their fixtures provide fresh response payloads and mock HTTP clients. No Clerk or provider credentials are required.

Frontend checks from `frontend`:

```powershell
npm test
npx tsc --noEmit
npm run lint
npm run build
```

Stop `next dev` before building because both use `.next`. Production builds and component tests verify compilation and behavior; they do not constitute browser visual acceptance testing.

For a database-backed smoke test, use a separate shell with the normal backend environment, apply migrations, and run the API locally. Then run `python -m scripts.smoke_sdk` from `backend`. It creates a disposable project, exercises the actual SDK and CLI over HTTP, checks passing/failing gates and revocation, and removes its own workspace. It doesn't call a model. Do not use the deliberately invalid test database URL for this smoke test.
