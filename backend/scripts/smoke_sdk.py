"""Actual SDK/CLI against localhost in a disposable workspace; no model calls.
Install sdk/python, start port 8000, then run python -m scripts.smoke_sdk.
"""
import asyncio
import os
import sys
import tempfile
from uuid import UUID, uuid4
import httpx
from sqlalchemy import delete, select
from llmforge import LLMForge, LLMForgeError
from app.core.auth import Identity, get_current_user
from app.core.database import async_session_maker, engine
from app.main import create_application
from app.models.evaluation import EvaluationRun
from app.models.workspace import Organization, User


def verify_client(read_key, evaluation_key):
    with LLMForge(api_key=read_key) as forge:
        prompt = forge.get_prompt("SDK smoke")
        assert prompt.compile(query="Hello") == "Hello"
        assert forge.get_prompt("SDK smoke") == prompt
        try:
            forge.get_dataset("SDK cases")
        except LLMForgeError as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("Read-only key allowed dataset reads")
    with LLMForge(api_key=evaluation_key) as forge:
        prompt = forge.get_prompt("SDK smoke", version=1)
        cases = [{"id": "hello", "inputs": {"query": "Hello"}, "expected_output": "Hello"}]
        result = forge.evaluate_local(prompt, cases, task=lambda compiled: compiled,
            evaluators={"correctness": lambda output, case: output == case["expected_output"]},
            scoring_version="exact-v1", model="sdk-local-echo", report_path=None)
        detail = forge.get_evaluation(result.id)
        assert detail.meets_threshold() and detail.run["config"]["source"] == "sdk_local"
        assert detail.results == []



async def main():
    subject = "user_phase4_smoke_" + uuid4().hex
    app = create_application()
    app.dependency_overrides[get_current_user] = lambda: Identity(subject, "Temporary SDK smoke")
    org_id = project_id = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://setup") as setup:
            workspace = await setup.get("/api/v1/workspaces")
            assert workspace.status_code == 200
            org = workspace.json()["organizations"][0]
            org_id, project_id = UUID(org["id"]), UUID(org["projects"][0]["id"])
            headers = {"X-Project-ID": str(project_id)}
            async def post(path, body):
                response = await setup.post("/api/v1/" + path, headers=headers, json=body)
                assert response.status_code == 201, "Smoke setup failed"
                return response.json()
            prompt = await post("prompt-library", {"name": "SDK smoke", "template_text": "{{query}}"})
            await post(f'prompt-library/{prompt["prompt"]["id"]}/versions', {"template_text": "Reply: {{query}}", "base_version": 1})
            response = await setup.put(f'/api/v1/prompt-library/{prompt["prompt"]["id"]}/labels/production', headers=headers, json={"version_id": prompt["version"]["id"]})
            assert response.status_code == 200
            read_key = await post("project-keys", {"name": "Read only"})
            ci_key = await post("project-keys", {"name": "Smoke CI", "scopes": ["evaluations:write"]})
            await asyncio.to_thread(verify_client, read_key["secret"], ci_key["secret"])
            with tempfile.TemporaryDirectory(prefix="llmforge-sdk-smoke-") as reports:
                import json
                cases_path = os.path.join(reports, "cases.json")
                with open(cases_path, "w", encoding="utf-8") as handle:
                    json.dump([{"id": "hello", "inputs": {"query": "Hello"}, "expected_output": "Hello"}], handle)
                env = {**os.environ, "LLMFORGE_URL": "http://localhost:8000/api/v1", "LLMFORGE_API_KEY": ci_key["secret"]}
                for version, expected in ((1, 0), (2, 1)):
                    report = os.path.join(reports, f"v{version}.json")
                    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "llmforge", "evaluate-local", "--prompt", "SDK smoke", "--version", str(version), "--dataset", cases_path, "--upload", "--output", report, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await process.communicate()
                    assert process.returncode == expected, f"Expected gate code {expected} for prompt v{version}"
                    checked = await asyncio.create_subprocess_exec(sys.executable, "-m", "llmforge", "check", report, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await checked.communicate()
                    assert checked.returncode == expected
            await setup.delete("/api/v1/project-keys/" + ci_key["id"], headers=headers)
            def verify_revocation():
                with LLMForge(api_key=ci_key["secret"]) as forge:
                    try:
                        forge.get_prompt("SDK smoke")
                    except LLMForgeError as exc:
                        assert exc.status_code == 401
                    else:
                        raise AssertionError("Revoked key was accepted")
            await asyncio.to_thread(verify_revocation)
            print("SDK/CLI smoke passed: fetch/compile/cache, scopes, external metrics, passing/failing gates, revocation")
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
        print("Temporary SDK smoke workspace removed")


if __name__ == "__main__":
    asyncio.run(main())
