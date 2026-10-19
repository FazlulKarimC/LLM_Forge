"""Explicit attempt selection shared by benchmark reads and execution only."""

import copy
from sqlalchemy import select, func

from app.models.experiment import Experiment
from app.models.experiment_attempt import ExperimentAttempt
from app.models.result import Result


async def lock_history_mutation(db):
    """Use admission's lock order for pin/report/purge reference integrity."""
    from app.models.evaluation_execution import EvaluationCapacity
    from fastapi import HTTPException
    if await db.scalar(select(EvaluationCapacity).where(EvaluationCapacity.id == 1).with_for_update()) is None:
        raise HTTPException(503, "Execution capacity is missing; apply database migrations")


async def current_number(db, experiment_id, requested=None, *, legacy_metrics=False):
    number = await db.scalar(select(Experiment.current_attempt).where(Experiment.id == experiment_id, Experiment.deleted_at.is_(None)))
    if number is None:
        from app.core.custom_exceptions import ResourceNotFoundException
        raise ResourceNotFoundException("Experiment", experiment_id)
    if requested is not None:
        return requested
    if legacy_metrics:
        record = await db.get(ExperimentAttempt, (experiment_id, number))
        if record is None or record.provenance_status == "unknown":
            legacy = await db.scalar(select(Result.id).where(Result.experiment_id == experiment_id, Result.attempt == 0))
            if legacy is not None:
                return 0
    return number


async def result_at(db, experiment_id, requested=None):
    number = await current_number(db, experiment_id, requested, legacy_metrics=True)
    return await db.scalar(select(Result).where(Result.experiment_id == experiment_id, Result.attempt == number))


async def create_attempt(db, experiment, number, status):
    record = await db.get(ExperimentAttempt, (experiment.id, number))
    if record is None:
        from app.services.benchmark_retention import MAX_PROJECT_ATTEMPTS, MAX_PROJECT_BYTES
        count, size = (await db.execute(select(func.count(), func.coalesce(func.sum(ExperimentAttempt.storage_bytes), 0)).where(ExperimentAttempt.project_id == experiment.project_id))).one()
        if count >= MAX_PROJECT_ATTEMPTS or size >= MAX_PROJECT_BYTES:
            from fastapi import HTTPException
            raise HTTPException(409, "Project retention limit reached; export and purge older unreferenced attempts")
        record = ExperimentAttempt(
            experiment_id=experiment.id, number=number, project_id=experiment.project_id,
            status=status, config=copy.deepcopy(experiment.config), provenance_status="recorded",
            generation_calls=0,
        )
        db.add(record)
    return record


async def freeze_manifest(db, experiment, number):
    record = await db.get(ExperimentAttempt, (experiment.id, number))
    if record is None or record.status not in ("queued", "running"):
        raise ValueError("Attempt provenance can only be recorded during execution")
    if record.run_manifest is not None:
        raise ValueError("Attempt manifest is already frozen")
    record.run_manifest = copy.deepcopy(experiment.run_manifest)
    record.dataset_hash = experiment.dataset_hash
    record.sample_ids = copy.deepcopy(experiment.sample_ids)
