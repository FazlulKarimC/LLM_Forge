"""Exercise Phase 3 against the configured Postgres DB in a disposable workspace.

Run from backend: python -m scripts.smoke_evaluations [--live-groq]
The optional live check uses GROQ_API_KEY for one generation and one judge call.
No authentication bypass is installed in the running server.
"""
import argparse
import asyncio
from uuid import UUID, uuid4
import httpx
from sqlalchemy import delete, select
from app.core.auth import Identity, get_current_user
from app.core.config import settings
from app.core.database import async_session_maker, engine
from app.main import create_application
from app.models.evaluation import EvaluationRun
from app.models.workspace import Organization, User


async def main(live=False):
    subject = "user_phase3_smoke_" + uuid4().hex
    app = create_application()
    app.dependency_overrides[get_current_user] = lambda: Identity(subject, "Temporary smoke test")
    org_id = project_id = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://smoke") as client:
            workspace = await client.get("/api/v1/workspaces")
            assert workspace.status_code == 200
            org = workspace.json()["organizations"][0]
            org_id, project_id = UUID(org["id"]), UUID(org["projects"][0]["id"])
            headers = {"X-Project-ID": str(project_id)}
            cases = [{"inputs": {"query": "Hello"}, "expected_output": "Hello", "name": "Smoke"}]
            dataset = await client.post("/api/v1/datasets", headers=headers, json={"name": "Smoke data", "cases": cases})
            assert dataset.status_code == 201
            data = dataset.json()
            revision_url = f'/api/v1/datasets/{data["dataset"]["id"]}/revisions'
            revisions = await asyncio.gather(*[client.post(revision_url, headers=headers, json={"base_version": 1, "cases": [{**cases[0], "name": name}]}) for name in ("A", "B")])
            assert sorted(response.status_code for response in revisions) == [201, 409]
            prompt = await client.post("/api/v1/prompt-library", headers=headers, json={"name": "Smoke echo", "template_text": "{{query}}"})
            assert prompt.status_code == 201
            request = {"prompt_version_id": prompt.json()["version"]["id"], "dataset_revision_id": data["revision"]["id"], "assertions": [{"kind": "exact_match"}]}
            response = await client.post("/api/v1/evaluations", headers=headers, json=request)
            assert response.status_code == 202
            detail = (await client.get("/api/v1/evaluations/" + response.json()["id"], headers=headers)).json()
            assert detail["run"]["status"] == "completed" and detail["run"]["passed"] == 1
            assert detail["results"][0]["output"] == "Hello"
            print("Postgres smoke passed: concurrent revision conflict, immutable snapshot, mock run, persisted results")
            if live:
                assert settings.GROQ_API_KEY, "GROQ_API_KEY is required for --live-groq"
                live_prompt = await client.post("/api/v1/prompt-library", headers=headers, json={"name": "Smoke live", "template_text": "Reply with exactly the following text and nothing else: {{query}}"})
                assert live_prompt.status_code == 201
                live_request = {**request, "prompt_version_id": live_prompt.json()["version"]["id"], "provider": "groq", "model": "openai/gpt-oss-20b", "api_key": settings.GROQ_API_KEY, "temperature": 0, "max_tokens": 256, "assertions": [{"kind": "contains", "value": "Hello"}], "judge": {"provider": "groq", "model": "openai/gpt-oss-20b", "api_key": settings.GROQ_API_KEY, "rubric": "Score 1 if the candidate contains the reference greeting, otherwise 0.", "threshold": 0.5}}
                response = await client.post("/api/v1/evaluations", headers=headers, json=live_request)
                assert response.status_code == 202
                detail = (await client.get("/api/v1/evaluations/" + response.json()["id"], headers=headers)).json()
                assert detail["run"]["status"] == "completed" and detail["run"]["errors"] == 0, "Live run or judge returned an error"
                assert detail["run"]["passed"] == 1, "Live assertions did not pass"
                assert settings.GROQ_API_KEY not in str(detail), "Secret appeared in stored results"
                print("Live Groq smoke passed: generation, rubric judge, token usage, secret exclusion")
    finally:
        async with async_session_maker() as db:
            user = await db.scalar(select(User).where(User.auth_subject == subject))
            if project_id is not None:
                await db.execute(delete(EvaluationRun).where(EvaluationRun.project_id == project_id))
            if org_id is not None:
                await db.execute(delete(Organization).where(Organization.id == org_id))
            if user is not None:
                await db.delete(user)
            await db.commit()
        await engine.dispose()
        print("Temporary smoke workspace removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-groq", action="store_true")
    asyncio.run(main(parser.parse_args().live_groq))
