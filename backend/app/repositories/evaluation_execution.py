"""Short transactions for fenced evaluation claims, progress and recovery."""

from datetime import timedelta
from sqlalchemy import func, select
from app.models.evaluation import EvaluationRun, EvaluationResult
from app.models.evaluation_execution import (
    EvaluationCapacity,
    EvaluationCheckpoint,
    EvaluationExecution,
)
from app.models.prompt import now

LEASE_SECONDS = 90
HEARTBEAT_SECONDS = 15


def expired(stamp):
    if stamp is None:
        return True
    return (
        stamp.replace(tzinfo=now().tzinfo) <= now()
        if stamp.tzinfo is None
        else stamp <= now()
    )


async def claim(db, run_id, owner, *, expected_execution_id=None):
    # Migration seeds this row. Tests seed it in their isolated databases.
    capacity = await db.scalar(
        select(EvaluationCapacity).where(EvaluationCapacity.id == 1).with_for_update()
    )
    if capacity is None:
        raise RuntimeError("Evaluation capacity migration is missing")
    queued = select(EvaluationExecution).where(
        EvaluationExecution.run_id == run_id, EvaluationExecution.status == "queued"
    )
    if expected_execution_id is not None:
        queued = queued.where(EvaluationExecution.id == expected_execution_id)
    execution = await db.scalar(queued)
    if execution is None:
        return None
    # Capacity is global even when the caller supplied a project-scoped session.
    executions = EvaluationExecution.__table__
    active = await db.scalar(
        select(func.count())
        .select_from(executions)
        .where(
            executions.c.status == "running",
            executions.c.lease_until > now(),
        )
    )
    if active >= 2:
        return None
    run = await db.scalar(
        select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
    )
    execution = await db.scalar(
        select(EvaluationExecution)
        .where(EvaluationExecution.id == execution.id)
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    )
    if execution is None or execution.status != "queued" or run.status != "queued":
        return None
    execution.status, execution.owner = "running", owner
    execution.heartbeat_at = execution.updated_at = now()
    execution.lease_until = now() + timedelta(seconds=LEASE_SECONDS)
    run.status, run.updated_at = "running", now()
    await db.commit()
    return execution.id


async def fence(db, execution_id, owner):
    execution = await db.scalar(
        select(EvaluationExecution).where(EvaluationExecution.id == execution_id)
    )
    if (
        execution is None
        or execution.status != "running"
        or execution.owner != owner
        or expired(execution.lease_until)
    ):
        return None
    run = await db.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.id == execution.run_id)
        .with_for_update()
    )
    execution = await db.scalar(
        select(EvaluationExecution)
        .where(EvaluationExecution.id == execution_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return (
        execution
        if run is not None
        and execution is not None
        and run.status == "running"
        and execution.status == "running"
        and execution.owner == owner
        and not expired(execution.lease_until)
        else None
    )


async def authorized(db, execution_id, owner):
    # Provider calls cannot hold a transaction open. Writes still lock and
    # recheck ownership atomically, while this boundary needs only one read.
    return await db.scalar(
        select(EvaluationExecution.id)
        .join(EvaluationRun, EvaluationRun.id == EvaluationExecution.run_id)
        .where(
            EvaluationExecution.id == execution_id,
            EvaluationExecution.status == "running",
            EvaluationExecution.owner == owner,
            EvaluationExecution.lease_until > now(),
            EvaluationRun.status == "running",
        )
    )


async def heartbeat(db, execution_id, owner):
    execution = await fence(db, execution_id, owner)
    if execution is None:
        return False
    execution.heartbeat_at = execution.updated_at = now()
    execution.lease_until = now() + timedelta(seconds=LEASE_SECONDS)
    await db.commit()
    return True


async def checkpoint(db, execution_id, owner, case_index, stage, data, checks=None):
    execution = await fence(db, execution_id, owner)
    if execution is None:
        return False
    existing = await db.scalar(
        select(EvaluationCheckpoint).where(
            EvaluationCheckpoint.run_id == execution.run_id,
            EvaluationCheckpoint.case_index == case_index,
            EvaluationCheckpoint.stage == stage,
        )
    )
    if existing is None:
        db.add(
            EvaluationCheckpoint(
                project_id=execution.project_id,
                run_id=execution.run_id,
                case_index=case_index,
                stage=stage,
                data=data,
            )
        )
        if checks:
            from app.services.evaluator_service import persist_scores

            persist_scores(
                db,
                execution.run_id,
                case_index,
                checks,
                project_id=execution.project_id,
            )
    await db.commit()
    return True


async def checkpoints(db, run_id, case_index):
    rows = (
        await db.scalars(
            select(EvaluationCheckpoint).where(
                EvaluationCheckpoint.run_id == run_id,
                EvaluationCheckpoint.case_index == case_index,
            )
        )
    ).all()
    return {row.stage: row.data for row in rows}


async def finalize(db, execution_id, owner, case_index, data):
    execution = await fence(db, execution_id, owner)
    if execution is None:
        return False
    run = await db.get(EvaluationRun, execution.run_id)
    existing = await db.scalar(
        select(EvaluationResult.id).where(
            EvaluationResult.run_id == run.id, EvaluationResult.case_index == case_index
        )
    )
    if existing is None:
        db.add(
            EvaluationResult(
                project_id=execution.project_id,
                run_id=run.id,
                case_index=case_index,
                data=data,
            )
        )
        run.completed += 1
        run.passed += int(data["passed"])
        run.errors += int(data["error"] is not None)
    run.updated_at = now()
    if run.completed == run.total:
        run.status = execution.status = "completed"
        execution.updated_at = now()
    await db.commit()
    return True


async def reconcile(db):
    rows = (
        await db.scalars(
            select(EvaluationExecution)
            .where(
                EvaluationExecution.status.in_(("queued", "running")),
                EvaluationExecution.lease_until <= now(),
            )
            .order_by(EvaluationExecution.run_id)
        )
    ).all()
    for execution in rows:
        run = await db.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.id == execution.run_id)
            .with_for_update()
        )
        execution = await db.scalar(
            select(EvaluationExecution)
            .where(EvaluationExecution.id == execution.id)
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
        if (
            execution is None
            or execution.status not in ("queued", "running")
            or not expired(execution.lease_until)
        ):
            continue
        execution.status, execution.failure_code, execution.updated_at = (
            "interrupted",
            "runner_interrupted",
            now(),
        )
        if run and run.status in ("queued", "running"):
            run.status, run.updated_at = "failed", now()
            run.error = "Execution interrupted. Saved progress is preserved; resume unfinished work."
    await db.commit()
    return rows
