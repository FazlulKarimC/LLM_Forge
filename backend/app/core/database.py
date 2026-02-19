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


def _get_async_database_url() -> str:
    """
    Convert DATABASE_URL to async-compatible format for asyncpg.
    
    NeonDB URLs contain query params (sslmode, channel_binding) that
    asyncpg doesn't understand, so we strip them.
    """
    url = settings.DATABASE_URL
    
    if not url:
        # Fallback for local development
        return "sqlite+aiosqlite:///./test.db"
    
    # Strip query parameters (asyncpg handles SSL automatically for neon.tech)
    if "?" in url:
        url = url.split("?")[0]
    
    # Convert to async driver
    url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', url)
    
    return url


# Create async engine
_database_url = _get_async_database_url()

# SQLite doesn't support pool_size/max_overflow — use StaticPool instead
if _database_url.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    engine = create_async_engine(
        _database_url,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_async_engine(
        _database_url,
        echo=settings.DEBUG,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
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


async def init_db() -> None:
    """
    Initialize database tables.
    
    Called on application startup.
    Note: We use Alembic migrations, so this is mainly for testing.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """
    Close database connections.
    
    Called on application shutdown.
    Disposes of the connection pool.
    """
    await engine.dispose()
