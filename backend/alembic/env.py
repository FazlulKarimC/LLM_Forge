"""
Alembic Migration Environment

Configured for async SQLAlchemy with NeonDB (PostgreSQL).
Loads DATABASE_URL from environment via dotenv.
"""

import os
import re
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Import app models
from app.core.database import Base
from app.models.experiment import Experiment
from app.models.result import Result
from app.models.run import Run

# Alembic Config object
config = context.config

# Get database URL from environment
database_url = os.getenv("DATABASE_URL", "")

# Strip all query parameters for asyncpg (it handles SSL automatically for neon.tech)
# NeonDB URLs come with sslmode, channel_binding etc. that asyncpg doesn't understand
if database_url and "?" in database_url:
    database_url = database_url.split("?")[0]

# Setup loggers
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    Generates SQL script without connecting to the database.
    """
    # Convert to async URL for offline mode
    url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', database_url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.
    """
    # Convert postgresql:// to postgresql+asyncpg://
    async_url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', database_url)
    
    engine = create_async_engine(async_url, echo=True, poolclass=pool.NullPool)

    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
