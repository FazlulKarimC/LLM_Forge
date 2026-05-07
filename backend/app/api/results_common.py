"""Shared helpers for results API routers."""

import json
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.background_jobs import mark_job_completed, mark_job_failed, mark_job_running
from app.core.database import async_session_maker
from app.models.result import Result
from app.models.run import Run
from app.schemas.result import (
    CostMetrics,
    MetricsResponse,
    PerformanceMetrics,
    QualityMetrics,
)


def result_to_metrics_response(result: Result) -> MetricsResponse:
    """Convert a Result model instance to MetricsResponse schema."""
    raw_metrics = result.raw_metrics or {}
    cost_metrics = raw_metrics.get("cost", {})
    robustness_metrics = raw_metrics.get("robustness") or {}
    robustness_breakdown = robustness_metrics.get("breakdown") or {}
    return MetricsResponse(
        experiment_id=result.experiment_id,
        summary_text=raw_metrics.get("summary_text"),
        quality=QualityMetrics(
            accuracy_exact=result.accuracy_exact,
            accuracy_f1=result.accuracy_f1,
            accuracy_substring=result.accuracy_substring,
            semantic_similarity=result.semantic_similarity,
            faithfulness=result.faithfulness,
            hallucination_rate=result.hallucination_rate,
            robustness_safety_score=robustness_metrics.get("safety_score"),
            robustness_inconclusive_rate=(
                robustness_breakdown.get("inconclusive_pct") / 100
                if robustness_breakdown.get("inconclusive_pct") is not None
                else None
            ),
        ),
        performance=PerformanceMetrics(
            latency_p50=result.latency_p50,
            latency_p95=result.latency_p95,
            latency_p99=result.latency_p99,
            throughput=result.throughput,
        ),
        cost=CostMetrics(
            total_tokens_input=result.total_tokens_input or 0,
            total_tokens_output=result.total_tokens_output or 0,
            total_runs=result.total_runs or 0,
            gpu_time_seconds=result.gpu_time_seconds,
            total_cost_usd=cost_metrics.get("total_cost_usd"),
            cost_per_correct_answer=cost_metrics.get("cost_per_correct_answer"),
            provider=cost_metrics.get("provider"),
            cost_source=cost_metrics.get("cost_source"),
        ),
        failure_modes=raw_metrics.get("failure_modes"),
        computed_at=result.computed_at,
    )


async def latest_runs_for_experiment(db: AsyncSession, experiment_id: UUID) -> List[Run]:
    """Return only the latest-attempt runs for an experiment."""
    attempt_query = (
        select(Run.attempt)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.attempt.desc())
        .limit(1)
    )
    attempt_result = await db.execute(attempt_query)
    latest_attempt = attempt_result.scalar_one_or_none()
    if latest_attempt is None:
        return []

    runs_query = select(Run).where(
        Run.experiment_id == experiment_id,
        Run.attempt == latest_attempt,
    )
    runs_result = await db.execute(runs_query)
    return runs_result.scalars().all()


async def save_llm_judge_result(db: AsyncSession, experiment_id: UUID, result: dict) -> None:
    """Persist LLM judge output into Result.raw_metrics when aggregate results exist."""
    res_query = select(Result).where(Result.experiment_id == experiment_id)
    res_result = await db.execute(res_query)
    result_obj = res_result.scalar_one_or_none()
    if result_obj is None:
        return

    from sqlalchemy.orm.attributes import flag_modified

    raw = dict(result_obj.raw_metrics or {})
    raw["llm_judge"] = result
    result_obj.raw_metrics = raw
    flag_modified(result_obj, "raw_metrics")
    await db.flush()
    await db.commit()


def load_knowledge_base_chunks(max_chunks: int) -> List[str]:
    """Load a bounded subset of knowledge-base chunks for synthetic generation."""
    from app.core.config import settings
    from app.services.rag_service import ChunkingService

    try:
        articles_path = settings.data_dir / "knowledge_base" / "articles.json"
        with articles_path.open("r", encoding="utf-8") as f:
            articles = json.load(f)
        chunk_objects = ChunkingService.chunk_articles(articles)
        chunks = [chunk.text for chunk in chunk_objects[: max_chunks * 2]]
    except Exception as exc:  # pragma: no cover - surfaced through API response
        raise RuntimeError(f"Failed to load knowledge base: {str(exc)[:200]}") from exc

    if not chunks:
        raise LookupError("No knowledge base chunks found")

    return chunks


async def run_llm_judge_job(job_id: str, experiment_id: UUID, sample_size: int) -> None:
    """Execute judge evaluation as best-effort in-process background work."""
    import logging

    from app.services.llm_judge_service import LLMJudgeService

    logger = logging.getLogger(__name__)
    await mark_job_running(job_id)

    try:
        async with async_session_maker() as session:
            judge = LLMJudgeService(session, sample_size=sample_size)
            result = await judge.evaluate_experiment(experiment_id)
            await save_llm_judge_result(session, experiment_id, result)
        await mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.exception("Judge background job failed for %s", experiment_id)
        await mark_job_failed(job_id, f"Judge evaluation failed: {str(exc)[:200]}")


async def run_synthetic_generation_job(
    job_id: str,
    pairs_per_chunk: int,
    max_chunks: int,
    seed: int | None,
) -> None:
    """Execute synthetic dataset generation as best-effort in-process background work."""
    import logging

    from app.services.synthetic_data_service import SyntheticDatasetService

    logger = logging.getLogger(__name__)
    await mark_job_running(job_id)

    try:
        chunks = load_knowledge_base_chunks(max_chunks)
        synth = SyntheticDatasetService()
        result = await synth.generate_from_chunks(
            chunks=chunks,
            pairs_per_chunk=pairs_per_chunk,
            max_chunks=max_chunks,
            seed=seed,
        )
        await mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.exception("Synthetic generation background job failed")
        await mark_job_failed(job_id, str(exc)[:200])
