"""Upgrade disposable PostgreSQL schema in one transaction, then roll it back."""

import asyncio
import os
from uuid import uuid4
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.database_url import async_database_options


async def main():
    url = os.environ.get("LLMFORGE_TEST_POSTGRES_URL", "")
    if not url.startswith("postgres") or make_url(url).host not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("Set LLMFORGE_TEST_POSTGRES_URL to disposable localhost PostgreSQL; hosted targets are refused")
    schema = "llmforge_test_" + uuid4().hex
    engine = create_async_engine(
        async_database_options(url)[0], connect_args=async_database_options(url)[1]
    )
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
                # Same transaction keeps pooled connections and legacy raw SQL isolated.
                config = Config("alembic.ini")

                def upgrade(sync_connection):
                    config.attributes["connection"] = sync_connection
                    command.upgrade(config, "q0r1s2t3u4v5")
                    # Seed pre-migration reruns and an aggregate whose attempt is
                    # unknowable. Latest config must never be assigned to old runs.
                    from app.models.workspace import Organization, Project
                    from app.models.experiment import Experiment
                    from app.models.run import Run
                    from app.schemas.experiment import ExperimentStatus
                    organization, project, experiment = uuid4(), uuid4(), uuid4()
                    sync_connection.execute(Organization.__table__.insert().values(id=organization, name="Migration test", slug="migration-test"))
                    sync_connection.execute(Project.__table__.insert().values(id=project, organization_id=organization, name="Migration test", slug="migration-test"))
                    sync_connection.execute(Experiment.__table__.insert().values(
                        id=experiment, project_id=project, name="Historical baseline", config={"model_name": "mock-model", "dataset_name": "sample"},
                        method="naive", model_name="mock-model", dataset_name="sample", status=ExperimentStatus.COMPLETED,
                        current_attempt=2, is_baseline=True, pinned_attempt=1, run_manifest={"latest": "cannot-guess-old-provenance"},
                    ))
                    for attempt in (1, 2):
                        sync_connection.execute(Run.__table__.insert().values(experiment_id=experiment, example_id="same-case", attempt=attempt, prompt="Original question", raw_output="Paris", expected_output="Paris"))
                    sync_connection.execute(text("INSERT INTO results (id, experiment_id, accuracy_exact, total_runs, computed_at) VALUES (:id, :experiment, 0.5, 1, now())"), {"id": uuid4(), "experiment": experiment})
                    command.upgrade(config, "head")
                    attempts = sync_connection.execute(text("SELECT number, config, run_manifest, provenance_status, storage_bytes FROM experiment_attempts WHERE experiment_id=:id ORDER BY number"), {"id": experiment}).all()
                    assert [row.number for row in attempts] == [0, 1, 2]
                    assert all(row.config is None and row.run_manifest is None and row.provenance_status == "unknown" for row in attempts)
                    assert attempts[1].storage_bytes > 0
                    assert sync_connection.scalar(text("SELECT attempt FROM results WHERE experiment_id=:id"), {"id": experiment}) == 0
                    assert sync_connection.scalar(text("SELECT pinned_attempt FROM experiments WHERE id=:id"), {"id": experiment}) == 1
                    assert sync_connection.scalar(text("SELECT count(*) FROM runs WHERE experiment_id=:id"), {"id": experiment}) == 2

                await connection.run_sync(upgrade)
                assert (
                    await connection.scalar(
                        text(f'SELECT version_num FROM "{schema}".alembic_version')
                    )
                    == "r1s2t3u4v5w6"
                )
                assert (
                    await connection.scalar(
                        text(f'SELECT count(*) FROM "{schema}".evaluation_capacity')
                    )
                    == 1
                )
                assert await connection.scalar(
                    text(f'SELECT count(*) FROM "{schema}".benchmark_reservations')
                ) == 0
                print(
                    "Migration chain passed in a disposable schema; transaction rolled back."
                )
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
