"""Retry only database transactions PostgreSQL explicitly aborted."""

import asyncio
from sqlalchemy.exc import DBAPIError
from app.core.diagnostics import logger, timed_stage


def sqlstate(exc):
    return getattr(getattr(exc, "orig", None), "sqlstate", None)


async def transaction_call(sessions, project_id, callback, *, stage):
    for attempt in range(1, 4):
        try:
            with timed_stage(stage, project_id=str(project_id)):
                async with sessions() as db:
                    db.info["project_id"] = project_id
                    return await callback(db)
        except DBAPIError as exc:
            # These states guarantee rollback. Never retry ambiguous connection
            # failures or provider calls; callbacks contain DB operations only.
            if sqlstate(exc) not in ("40P01", "40001") or attempt == 3:
                raise
            logger.warning(
                "evaluation_transaction_retry",
                extra={
                    "stage": stage,
                    "exception_type": type(exc).__name__,
                    "sqlstate": sqlstate(exc),
                    "retry_attempt": attempt,
                },
            )
            await asyncio.sleep(0.05 * attempt)
