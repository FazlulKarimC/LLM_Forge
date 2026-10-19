"""
Database Connection and Session Management

Uses SQLAlchemy async with PostgreSQL (NeonDB).
Provides session dependency for FastAPI routes.
"""

import re
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


from app.core.database_url import async_database_options


def _get_async_database_url():
    return async_database_options(settings.DATABASE_URL, production=settings.ENVIRONMENT != "development")[0]


_database_url, _connect_args = async_database_options(settings.DATABASE_URL, production=settings.ENVIRONMENT != "development")
if str(_database_url).startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    engine = create_async_engine(_database_url, echo=False, poolclass=StaticPool)
else:
    engine = create_async_engine(
        _database_url, echo=False, pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW, pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True, connect_args=_connect_args,
    )

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions.
    
    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    
    Yields:
        AsyncSession: Database session that auto-commits on success,
                      rolls back on exception.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
