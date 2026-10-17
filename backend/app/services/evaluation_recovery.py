"""Recovery and compact analytical responses share project-scoped persistence."""

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from app.models.evaluation import EvaluationRun, EvaluationResult
from app.models.evaluator import EvaluationScore
from app.models.evaluation_execution import EvaluationExecution, EvaluationCheckpoint
from app.schemas.evaluation import run_response
from app.models.prompt import now
from app.services.evaluation_runs import find_run, lock_project
from app.services.evaluator_service import score_summary
from app.services.evaluation_runner import runner, new_execution, request_from_config


async def recovery(db, run):
    if run.config.get("source") == "sdk_local":
        return {"available": False, "remaining_calls": {"generation": 0, "judging": 0, "total": 0}, "failure_code": None, "attempts": []}
    attempts = (
        await db.scalars(
            select(EvaluationExecution)
            .where(EvaluationExecution.run_id == run.id)
            .order_by(EvaluationExecution.attempt)
        )
    ).all()
    available = bool(
        attempts and attempts[-1].status == "interrupted" and run.status == "failed"
    )
    stages = (
        await db.execute(
            select(EvaluationCheckpoint.case_index, EvaluationCheckpoint.stage).where(
                EvaluationCheckpoint.run_id == run.id
            )
        )
    ).all()
    done = set(stages)
    generation = (
        0
        if run.source_run_id or run.config.get("provider") == "mock"
        else sum((i, "generation") not in done for i in range(run.total))
    )
    judges = []
    if run.config.get("judge"):
        judges.append("inline_judge")
    judges.extend(
        item["version_id"]
        for item in run.config.get("evaluators", [])
        if item["definition"]["kind"] == "llm_judge"
    )
    judging = sum((i, stage) not in done for i in range(run.total) for stage in judges)
    return {
        "available": available,
        "remaining_calls": {
            "generation": generation,
            "judging": judging,
            "total": generation + judging,
        },
        "failure_code": attempts[-1].failure_code if attempts else None,
        "attempts": [
            {
                "attempt": item.attempt,
                "status": item.status,
                "failure_code": item.failure_code,
                "request_id": item.request_id,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in attempts
        ],
    }


async def resume(run_id, data, project_id, db, token=None, request_id=None):
    from app.services.evaluation_idempotency import replay, remember

    await lock_project(db, project_id)
    prior = await replay(db, token, data)
    if prior:
        return run_response(prior)
    run = await find_run(db, run_id, lock=True)
    info = await recovery(db, run)
    if not info["available"]:
        raise HTTPException(
            409, "Only interrupted runs with saved checkpoints can be resumed"
        )
    active = (
        await db.scalars(
            select(EvaluationRun).where(EvaluationRun.status.in_(("queued", "running")))
        )
    ).all()
    if len(active) >= 2:
        raise HTTPException(
            409, "Two evaluations are already active; wait or cancel one"
        )
    try:
        request = request_from_config(
            run, data.api_key, data.judge_key, data.evaluator_keys
        )
    except ValidationError as exc:
        raise HTTPException(
            422, "Re-enter the generation and judge keys required by this pinned run"
        ) from exc
    required = {
        item["version_id"]
        for item in run.config.get("evaluators", [])
        if item["definition"]["kind"] == "llm_judge"
    }
    if any(
        not data.evaluator_keys.get(key)
        or not data.evaluator_keys[key].get_secret_value().strip()
        for key in required
    ):
        raise HTTPException(422, "Re-enter a key for each pinned LLM evaluator")
    keys = {key: value for key, value in data.evaluator_keys.items() if key in required}
    run.status, run.error, run.updated_at = "queued", None, now()
    execution = new_execution(run, len(info["attempts"]) + 1, request_id)
    db.add(execution)
    remember(db, token, data, run)
    await db.commit()
    await runner.enqueue(run.id, project_id, request, keys, execution_id=execution.id)
    return run_response(run)


async def summary(db, run_id):
    run = await find_run(db, run_id)
    # Typed scores supply successful values; checkpoints supply execution errors.
    scores = (
        await db.scalars(
            select(EvaluationScore).where(EvaluationScore.run_id == run.id)
        )
    ).all()
    checks = [
        {
            "kind": score.name,
            "name": score.name,
            "assignment": score.assignment,
            "evaluator_version_id": str(score.evaluator_version_id)
            if score.evaluator_version_id
            else None,
            "data_type": score.data_type,
            "value": score.value,
            "passed": score.passed,
            "required": score.required,
        }
        for score in scores
    ]
    checkpoint_checks = (
        await db.scalars(
            select(EvaluationCheckpoint.data["checks"]).where(
                EvaluationCheckpoint.run_id == run.id
            )
        )
    ).all()
    checks.extend(
        check
        for items in checkpoint_checks
        if items
        for check in items
        if check.get("error")
    )
    if not scores and not checkpoint_checks:
        historical = (
            await db.scalars(
                select(EvaluationResult.data["checks"]).where(
                    EvaluationResult.run_id == run.id
                )
            )
        ).all()
        checks = [check for items in historical if items for check in items]
    result = {
        "run": run_response(run),
        "results": [],
        "score_summary": score_summary([{"checks": checks}], run.total, run.config),
        "recovery": await recovery(db, run),
    }
    # Only usage columns, never generation content.
    usage = (
        await db.execute(
            select(
                EvaluationCheckpoint.data["latency_ms"].as_float(),
                EvaluationCheckpoint.data["tokens_input"].as_integer(),
                EvaluationCheckpoint.data["tokens_output"].as_integer(),
            ).where(
                EvaluationCheckpoint.run_id == run.id,
                EvaluationCheckpoint.stage == "generation",
            )
        )
    ).all()
    if not usage:
        # Legacy/submitted runs have usage metadata but no durable checkpoints.
        usage = (
            await db.execute(
                select(
                    EvaluationResult.data["latency_ms"].as_float(),
                    EvaluationResult.data["tokens_input"].as_integer(),
                    EvaluationResult.data["tokens_output"].as_integer(),
                ).where(EvaluationResult.run_id == run.id)
            )
        ).all()
    result["usage"] = {
        "generation": {
            "tokens_input": sum(r[1] or 0 for r in usage),
            "tokens_output": sum(r[2] or 0 for r in usage),
        },
        "judging": {
            "tokens_input": sum(
                check.get("tokens_input") or 0
                for items in checkpoint_checks
                if items
                for check in items
            ),
            "tokens_output": sum(
                check.get("tokens_output") or 0
                for items in checkpoint_checks
                if items
                for check in items
            ),
        },
    }
    if run.source_run_id:
        result["usage"]["generation"] = {"tokens_input": 0, "tokens_output": 0}
    await db.commit()
    return result


async def cases(db, run_id, offset, limit, filter, reference_id=None):
    from app.repositories.evaluation_reads import case_rows

    run = await find_run(db, run_id)
    if run.config.get("source") == "sdk_local":
        if reference_id or filter != "all":
            raise HTTPException(409, "Per-case comparisons require uploaded case results; this is a summary-only report")
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    if reference_id:
        reference = await find_run(db, reference_id)
        from app.services.local_evaluations import compatible
        if not compatible(run, reference):
            raise HTTPException(422, "Compare runs using the same dataset revision")
    elif filter in ("improved", "regressed", "unchanged", "comparison_errors"):
        raise HTTPException(422, "Select a reference run for comparison filters")
    payload = await case_rows(db, run.id, offset, limit, filter, reference_id)
    if reference_id:
        payload["comparison_totals"] = {
            name: (await case_rows(db, run.id, 0, 1, name, reference_id))["total"]
            for name in ("improved", "regressed", "unchanged", "comparison_errors")
        }
    await db.commit()
    return payload


async def case_detail(db, run_id, case_index):
    run = await find_run(db, run_id)
    if run.config.get("source") == "sdk_local":
        raise HTTPException(404, "Case content is retained in the local SDK report")
    result = await db.scalar(
        select(EvaluationResult).where(
            EvaluationResult.run_id == run_id, EvaluationResult.case_index == case_index
        )
    )
    if result is None:
        raise HTTPException(404, "Case result not found")
    await db.commit()
    return {"case_index": result.case_index, **result.data}
