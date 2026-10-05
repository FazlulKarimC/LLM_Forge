"""Actor-scoped dispatch tokens. All fingerprints exclude provider credentials."""

import hashlib
import json
from datetime import timedelta
from fastapi import HTTPException
from sqlalchemy import delete, select
from app.models.evaluation_execution import EvaluationIdempotency
from app.models.evaluation import EvaluationRun
from app.models.prompt import now


def fingerprint(data):
    # Exclude credential fields by schema position, not arbitrary user metric names.
    public = data.model_dump(
        mode="json",
        exclude={
            "api_key": True,
            "judge_key": True,
            "evaluator_keys": True,
            "judge": {"api_key"},
            "evaluators": {"__all__": {"api_key"}},
        },
    )
    return hashlib.sha256(
        json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def replay(db, token, data):
    if token is None:
        return None
    actor, operation, key = token
    if (
        not key
        or len(key) > 128
        or not key.isascii()
        or any(ord(c) < 33 or ord(c) > 126 for c in key)
    ):
        raise HTTPException(
            422, "Idempotency-Key must contain 1–128 visible ASCII characters"
        )
    await db.execute(
        delete(EvaluationIdempotency).where(EvaluationIdempotency.expires_at <= now())
    )
    row = await db.scalar(
        select(EvaluationIdempotency).where(
            EvaluationIdempotency.actor == actor,
            EvaluationIdempotency.operation == operation,
            EvaluationIdempotency.key == key,
        )
    )
    if row:
        if row.fingerprint != fingerprint(data):
            raise HTTPException(
                409, "Idempotency-Key was already used for a different request"
            )
        return await db.get(EvaluationRun, row.run_id)
    return None


def remember(db, token, data, run):
    if token:
        actor, operation, key = token
        db.add(
            EvaluationIdempotency(
                project_id=run.project_id,
                actor=actor,
                operation=operation,
                key=key,
                fingerprint=fingerprint(data),
                run_id=run.id,
                expires_at=now() + timedelta(hours=24),
            )
        )
