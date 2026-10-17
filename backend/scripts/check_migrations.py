"""Upgrade disposable PostgreSQL schema in one transaction, then roll it back."""

import asyncio
import os
from uuid import uuid4
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings


async def main():
    url = os.environ.get("LLMFORGE_TEST_POSTGRES_URL") or settings.DATABASE_URL
    if not url.startswith("postgres"):
        raise RuntimeError("A PostgreSQL connection is required")
    schema = "llmforge_test_" + uuid4().hex
    engine = create_async_engine(
        url.split("?")[0].replace("postgresql:", "postgresql+asyncpg:")
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
                    command.upgrade(config, "head")

                await connection.run_sync(upgrade)
                assert (
                    await connection.scalar(
                        text(f'SELECT version_num FROM "{schema}".alembic_version')
                    )
                    == "p9q0r1s2t3u4"
                )
                assert (
                    await connection.scalar(
                        text(f'SELECT count(*) FROM "{schema}".evaluation_capacity')
                    )
                    == 1
                )
                print(
                    "Migration chain passed in a disposable schema; transaction rolled back."
                )
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
