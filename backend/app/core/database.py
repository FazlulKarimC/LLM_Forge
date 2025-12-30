"""
Database Connection and Session Management

Uses SQLAlchemy async with PostgreSQL (NeonDB).
Provides session dependency for FastAPI routes.

TODO (Iteration 1): Implement connection pooling
TODO (Iteration 2): Add query logging for debugging
TODO (Iteration 3): Add read replica support for scaling
"""

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


# TODO (Iteration 1): Initialize engine with proper connection string
# For now, we use a placeholder that will fail gracefully
_database_url = settings.DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://"
) if settings.DATABASE_URL else "sqlite+aiosqlite:///./test.db"

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
    Creates all tables defined in models.
    
    TODO (Iteration 1): Add migration support with Alembic
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
