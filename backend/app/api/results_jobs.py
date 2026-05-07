"""Background job routes associated with result evaluation."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.results_common import run_llm_judge_job, run_synthetic_generation_job
from app.core.background_jobs import create_job, get_job
from app.core.database import get_db
from app.models.experiment import Experiment

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_background_job(job_id: str):
    """Return the current status for an asynchronous background job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or has expired")
    return JSONResponse(content=job)


@router.post("/synthetic/generate")
async def generate_synthetic_dataset(
    background_tasks: BackgroundTasks,
    pairs_per_chunk: int = Query(3, ge=1, le=5, description="QA pairs per chunk"),
    max_chunks: int = Query(10, ge=1, le=20, description="Max chunks to process"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
):
    """Create a pollable job record and schedule best-effort in-process generation."""
    job = await create_job(
        "synthetic_generation",
        {
            "pairs_per_chunk": pairs_per_chunk,
            "max_chunks": max_chunks,
            "seed": seed,
        },
    )
    background_tasks.add_task(
        run_synthetic_generation_job,
        job["job_id"],
        pairs_per_chunk,
        max_chunks,
        seed,
    )
    return JSONResponse(status_code=202, content=job)


@router.post("/{experiment_id}/judge")
async def run_llm_judge(
    experiment_id: UUID,
    background_tasks: BackgroundTasks,
    sample_size: int = Query(20, ge=1, le=50, description="Number of runs to sample"),
    db: AsyncSession = Depends(get_db),
):
    """Create a pollable job record and schedule best-effort in-process judge work."""
    experiment_query = select(Experiment).where(Experiment.id == experiment_id)
    experiment_result = await db.execute(experiment_query)
    if experiment_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    job = await create_job(
        "llm_judge",
        {
            "experiment_id": str(experiment_id),
            "sample_size": sample_size,
        },
    )
    background_tasks.add_task(run_llm_judge_job, job["job_id"], experiment_id, sample_size)
    return JSONResponse(status_code=202, content=job)
