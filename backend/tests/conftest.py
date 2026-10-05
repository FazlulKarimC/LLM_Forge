"""Explicit auth override for pre-auth mocked route contract tests only.

Workspace/security tests exercise the real project dependency and real database.
There is no authentication bypass in application code or environment settings.
"""

from uuid import UUID, uuid4
import os
from sqlalchemy import text

import httpx
import pytest
import pytest_asyncio
from app.core.database import Base, get_db
from app.main import create_application
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool


@pytest.fixture
def legacy_route_identity():
    from app.core.tenancy import ProjectContext, get_project_context
    from app.main import app

    test_id = UUID("00000000-0000-0000-0000-000000000001")
    previous = app.dependency_overrides.get(get_project_context)
    app.dependency_overrides[get_project_context] = lambda: ProjectContext(
        test_id, test_id, test_id, "owner"
    )
    try:
        yield
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_project_context, None)
        else:
            app.dependency_overrides[get_project_context] = previous


@compiles(JSONB, "sqlite")
def sqlite_jsonb(_type, _compiler, **_kwargs):
    return "JSON"


@pytest_asyncio.fixture
async def api(monkeypatch, request):
    schema = None
    if request.node.get_closest_marker("postgres"):
        url = os.environ.get("LLMFORGE_TEST_POSTGRES_URL")
        if not url:
            pytest.skip(
                "Set LLMFORGE_TEST_POSTGRES_URL for real PostgreSQL concurrency tests"
            )
        schema = "llmforge_test_" + uuid4().hex
        admin_engine = create_async_engine(
            url.split("?")[0].replace("postgresql:", "postgresql+asyncpg:")
        )
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(
            url.split("?")[0].replace("postgresql:", "postgresql+asyncpg:"),
            execution_options={"schema_translate_map": {None: schema}},
        )
    else:
        engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)

    app = None
    try:

        @event.listens_for(engine.sync_engine, "connect")
        def foreign_keys(connection, _record):
            if schema is None:
                connection.execute("PRAGMA foreign_keys=ON")

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        from app.models.evaluation_execution import EvaluationCapacity

        async with sessions() as session:
            session.add(EvaluationCapacity(id=1))
            await session.commit()
        from app.services.evaluation_runner import EvaluationRunner

        runner = EvaluationRunner()
        original_enqueue = runner.enqueue

        async def enqueue(run_id, project_id, request, keys, **kwargs):
            await original_enqueue(run_id, project_id, request, keys, **kwargs)
            await runner.execute(run_id)

        monkeypatch.setattr(runner, "enqueue", enqueue)
        monkeypatch.setattr("app.services.evaluation_runner.runner", runner)
        monkeypatch.setattr("app.services.evaluation_recovery.runner", runner)
        monkeypatch.setattr(
            "app.services.evaluation_runner.async_session_maker", sessions
        )

        async def database():
            async with sessions() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app = create_application()
        app.dependency_overrides[get_db] = database
        monkeypatch.setattr("app.core.background_jobs.async_session_maker", sessions)
        monkeypatch.setattr(
            "app.services.evaluation_service.async_session_maker", sessions
        )
        monkeypatch.setattr(
            "app.core.rate_limit.check_create_rate_limit", _no_rate_limit
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield app, client, sessions
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        await engine.dispose()
        if schema:
            assert schema.startswith("llmforge_test_") and len(schema) == 46
            async with admin_engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin_engine.dispose()


async def _no_rate_limit(_request):
    pass
