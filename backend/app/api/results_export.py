"""Complete attempt exports, with a bounded JSON and streaming JSONL option."""

import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.database import async_session_maker
from app.models.experiment_attempt import ExperimentAttempt
from app.api.results_common import latest_runs_for_experiment, result_to_metrics_response
from app.schemas.result import RunSummary
from app.services.attempts import current_number, result_at


async def export_attempt(db, experiment, attempt, format):
    number = await current_number(db, experiment.id, attempt, legacy_metrics=True)
    snapshot = await db.get(ExperimentAttempt, (experiment.id, number))
    if snapshot and snapshot.status in ("queued", "running"):
        raise HTTPException(409, "Stop the attempt or wait for completion before exporting a consistent snapshot")
    from sqlalchemy import select, func
    from app.models.run import Run
    from app.models.benchmark_reservation import BenchmarkReservation
    from app.services.benchmark_execution import utcnow
    if await db.scalar(select(BenchmarkReservation.id).where(
        BenchmarkReservation.resource_id == str(experiment.id), BenchmarkReservation.attempt == number,
        BenchmarkReservation.released_at.is_(None), BenchmarkReservation.expires_at > utcnow(),
    ).limit(1)):
        raise HTTPException(409, "An issued call is finishing; export after its reservation is released")
    result = await result_at(db, experiment.id, number)
    expected_count = await db.scalar(select(func.count()).select_from(Run).where(Run.experiment_id == experiment.id, Run.attempt == number)) if number > 0 else 0
    header = {"experiment": {"id": str(experiment.id), "name": experiment.name, "attempt": number,
              "status": snapshot.status if snapshot else "unknown", "provenance": snapshot.provenance_status if snapshot else "unknown",
              "generation_calls": snapshot.generation_calls if snapshot else None,
              "usage_coverage": "Committed responses only; interrupted/failed calls can have unknown token use or charges",
              "config": snapshot.config if snapshot else None, "run_manifest": snapshot.run_manifest if snapshot else None,
              "dataset_hash": snapshot.dataset_hash if snapshot else None, "sample_ids": snapshot.sample_ids if snapshot else None},
              "metrics": result_to_metrics_response(result).model_dump(mode="json") if result else None,
              "raw_metrics": result.raw_metrics if result else None}
    header["total_runs"] = expected_count
    project_id, identity = db.info.get("project_id"), experiment.id
    await db.commit()

    async def pages():
        skip = 0
        while number > 0:
            async with async_session_maker() as session:
                if project_id:
                    session.info["project_id"] = project_id
                rows = await latest_runs_for_experiment(session, identity, number, skip=skip, limit=25)
                page = [RunSummary.model_validate(row).model_dump(mode="json") for row in rows]
            if not page:
                break
            yield page
            skip += len(page)
        if skip != expected_count:
            raise RuntimeError("Attempt history changed during export; retry the download before purging")

    filename = f"{identity}_attempt_{number}"
    if format == "jsonl":
        async def lines():
            yield json.dumps({"type": "manifest", **header}, ensure_ascii=False) + "\n"
            async for page in pages():
                for row in page:
                    yield json.dumps({"type": "case", **row}, ensure_ascii=False) + "\n"
        return StreamingResponse(lines(), media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="{filename}.jsonl"'})
    rows, size = [], len(json.dumps(header).encode())
    async for page in pages():
        size += len(json.dumps(page, ensure_ascii=False).encode())
        if size > 8 * 1024 * 1024 or len(rows) + len(page) > 500:
            raise HTTPException(413, "Use format=jsonl for a complete bounded-memory historical export")
        rows.extend(page)
    return JSONResponse({**header, "runs": rows, "total_runs": len(rows)},
                        headers={"Content-Disposition": f'attachment; filename="{filename}.json"'})
