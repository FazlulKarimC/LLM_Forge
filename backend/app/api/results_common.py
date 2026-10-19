"""Shared helpers for results API routers."""

import json
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.background_jobs import mark_job_completed, mark_job_failed, mark_job_running
from app.core.database import async_session_maker
from app.services.attempts import result_at, current_number
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
    accuracy_metrics = raw_metrics.get("accuracy", {})
    retrieval_quality = raw_metrics.get("retrieval_quality")
    completion_quality_label = raw_metrics.get("completion_quality")

    # Build structured completion_quality from the label string
    completion_quality_obj = None
    if completion_quality_label:
        failure_modes = raw_metrics.get("failure_modes", {})
        total_runs = accuracy_metrics.get("total_evaluated", 0)
        total_failures = failure_modes.get("total_failures", 0)
        completion_quality_obj = {
            "label": completion_quality_label,
            "failure_rate": total_failures / total_runs if total_runs > 0 else 0,
            "total_failures": total_failures,
            "total_runs": total_runs,
        }

    return MetricsResponse(
        experiment_id=result.experiment_id,
        attempt=result.attempt or 0,
        coverage=raw_metrics.get("coverage", {}),
        scoring_versions=raw_metrics.get("scoring_versions", ["unknown"]),
        summary_text=raw_metrics.get("summary_text"),
        quality=QualityMetrics(
            primary_accuracy=accuracy_metrics.get("primary_accuracy"),
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
            accuracy_excluding_failures=accuracy_metrics.get("accuracy_excluding_failures"),
            total_excluding_failures=accuracy_metrics.get("total_excluding_failures"),
            completion_quality=completion_quality_obj,
            retrieval_quality=retrieval_quality if retrieval_quality else None,
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
            provider_time_seconds=cost_metrics.get("provider_time_seconds"),
            cache_hits=cost_metrics.get("cache_hits", 0),
            provider_calls=cost_metrics.get("provider_calls", 0),
            usage_sources=cost_metrics.get("usage_sources", ["unknown"]),
            total_cost_usd=cost_metrics.get("total_cost_usd"),
            cost_per_correct_answer=cost_metrics.get("cost_per_correct_answer"),
            provider=cost_metrics.get("provider"),
            cost_source=cost_metrics.get("cost_source"),
            cost_per_sample_usd=cost_metrics.get("cost_per_sample_usd"),
            accuracy_per_dollar=cost_metrics.get("accuracy_per_dollar"),
        ),
        failure_modes=raw_metrics.get("failure_modes"),
        computed_at=result.computed_at,
    )


async def latest_runs_for_experiment(db: AsyncSession, experiment_id: UUID, attempt=None, *, skip=0, limit=50) -> List[Run]:
    """Filter the requested/current attempt in SQL before loading bounded rows."""
    number = await current_number(db, experiment_id, attempt)
    return list((await db.scalars(select(Run).where(
        Run.experiment_id == experiment_id, Run.attempt == number
    ).order_by(Run.example_id, Run.id).offset(skip).limit(limit))).all())


async def save_llm_judge_result(db: AsyncSession, experiment_id: UUID, result: dict, attempt=None) -> None:
    """Persist LLM judge output into Result.raw_metrics when aggregate results exist."""
    result_obj = await result_at(db, experiment_id, attempt)
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


async def run_llm_judge_job(job_id: str, experiment_id: UUID, sample_size: int, attempt=None) -> None:
    """Execute judge evaluation as best-effort in-process background work."""
    import logging

    from app.services.llm_judge_service import LLMJudgeService

    logger = logging.getLogger(__name__)
    await mark_job_running(job_id)

    try:
        async with async_session_maker() as session:
            from app.services.benchmark_execution import BudgetedEngine, execution_deadline
            deadline = await execution_deadline(session, kind="llm_judge", resource_id=job_id)
            await session.commit()
            judge = LLMJudgeService(session, sample_size=sample_size)
            result = await judge.evaluate_experiment(
                experiment_id, attempt=attempt, budget=lambda engine: BudgetedEngine(engine, deadline_at=deadline),
            )
            await save_llm_judge_result(session, experiment_id, result, attempt)
        await mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.error("Judge background job failed: experiment=%s exception_type=%s", experiment_id, type(exc).__name__)
        await mark_job_failed(job_id, "Judge evaluation failed; completed benchmark cases were kept")
    finally:
        from app.services.benchmark_execution import release_execution
        async with async_session_maker() as session:
            await release_execution(session, kind="llm_judge", resource_id=job_id)
            await session.commit()


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
        from app.services.benchmark_execution import BudgetedEngine, execution_deadline
        async with async_session_maker() as session:
            deadline = await execution_deadline(session, kind="synthetic_generation", resource_id=job_id)
        chunks = load_knowledge_base_chunks(max_chunks)
        synth = SyntheticDatasetService()
        result = await synth.generate_from_chunks(
            chunks=chunks,
            pairs_per_chunk=pairs_per_chunk,
            max_chunks=max_chunks,
            seed=seed,
            budget=lambda engine: BudgetedEngine(engine, deadline_at=deadline),
        )
        await mark_job_completed(job_id, result)
    except Exception as exc:  # pragma: no cover - best-effort background execution
        logger.error("Synthetic generation failed: exception_type=%s", type(exc).__name__)
        await mark_job_failed(job_id, "Synthetic generation failed")
    finally:
        from app.services.benchmark_execution import release_execution
        async with async_session_maker() as session:
            await release_execution(session, kind="synthetic_generation", resource_id=job_id)
            await session.commit()
