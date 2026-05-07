"""Post-run persistence, grader, routing, and regression hooks."""

import logging
import time as _time
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.result import Result
from app.schemas.experiment import ExperimentResponse, regression_status_from_verdict

logger = logging.getLogger(__name__)


class ExperimentPostProcessor:
    """Handle work that happens after run rows have been written."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def apply_graders(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        current_attempt: int,
    ) -> None:
        """Apply configured run graders to the latest attempt."""
        graders_config = experiment_response.config.graders
        if not graders_config or not graders_config.rules:
            return

        from app.models.run import Run as _GraderRun
        from app.services.grader_service import GraderEngine

        grader_engine = GraderEngine()
        reasoning = experiment_response.config.reasoning_method.value
        has_rag = bool(
            experiment_response.config.rag
            and experiment_response.config.rag.retrieval_method != "none"
        )

        grader_query = select(_GraderRun).where(
            _GraderRun.experiment_id == experiment_id,
            _GraderRun.attempt == current_attempt,
        )
        grader_result = await self.db.execute(grader_query)
        grader_runs = grader_result.scalars().all()

        for run in grader_runs:
            verdicts = [
                grader_engine.grade_run(run, rule, reasoning, has_rag)
                for rule in graders_config.rules
            ]
            existing_results = dict(run.grader_results or {})
            existing_results.update({verdict.grader_name: verdict.to_dict() for verdict in verdicts})
            run.grader_results = existing_results

        await self.db.flush()
        logger.info(
            "[EXECUTE] Applied %d graders to %d runs",
            len(graders_config.rules),
            len(grader_runs),
        )

    async def save_optimization_report(
        self,
        *,
        experiment_id: UUID,
        engine,
        opt_report,
        batch_stats: dict[str, int],
        cache,
        profiler,
        wall_start: float,
    ) -> None:
        """Persist optimization and routing telemetry into the result raw_metrics."""
        from app.services.inference.provider_router import ProviderRouter as _PR
        from sqlalchemy.orm.attributes import flag_modified

        wall_end = _time.perf_counter()
        opt_report.cache_stats = cache.stats() if cache else {}
        opt_report.profiling_summary = profiler.summary()
        opt_report.batch_stats = dict(batch_stats)
        opt_report.total_wall_time_ms = (wall_end - wall_start) * 1000

        res_query = select(Result).where(Result.experiment_id == experiment_id)
        res_result = await self.db.execute(res_query)
        result_obj = res_result.scalar_one_or_none()
        if not result_obj:
            return

        existing_raw = dict(result_obj.raw_metrics or {})
        existing_raw["optimization"] = opt_report.to_dict()

        if isinstance(engine, _PR):
            existing_raw["routing"] = engine.stats_tracker.summary()
            logger.info("[EXECUTE] Routing telemetry saved to raw_metrics")

        result_obj.raw_metrics = existing_raw
        flag_modified(result_obj, "raw_metrics")
        await self.db.flush()
        await self.db.commit()
        logger.info("[EXECUTE] Optimization report saved to raw_metrics")

    async def run_auto_regression_check(self, experiment_id: UUID) -> None:
        """Run the post-execution regression check when a baseline exists."""
        try:
            from app.models.experiment import Experiment as _RegExp
            from app.services.regression_service import RegressionService
            from sqlalchemy.orm.attributes import flag_modified as _flag_modified

            reg_svc = RegressionService(self.db)
            reg_query = select(_RegExp).where(_RegExp.id == experiment_id)
            reg_result = await self.db.execute(reg_query)
            reg_exp = reg_result.scalar_one_or_none()
            if not reg_exp:
                return

            baseline = await reg_svc.find_baseline(reg_exp)
            if not baseline or baseline.id == experiment_id:
                return

            verdict = await reg_svc.run_regression_check(experiment_id, baseline.id)

            res_query = select(Result).where(Result.experiment_id == experiment_id)
            res_result = await self.db.execute(res_query)
            result_obj = res_result.scalar_one_or_none()
            if result_obj:
                existing_raw: dict[str, Any] = dict(result_obj.raw_metrics or {})
                existing_raw["regression"] = verdict.to_dict()
                result_obj.raw_metrics = existing_raw
                _flag_modified(result_obj, "raw_metrics")

            reg_exp.regression_status = regression_status_from_verdict(verdict.passed).value
            reg_exp.regression_passed = verdict.passed
            await self.db.flush()
            await self.db.commit()

            status = "PASS" if verdict.passed else ("FAIL" if verdict.passed is False else "INCONCLUSIVE")
            logger.info(
                "[EXECUTE] Regression check: %s (overlap=%.2f, violations=%d)",
                status,
                verdict.overlap_ratio,
                len(verdict.violations),
            )
        except Exception as reg_err:
            logger.warning("[EXECUTE] Regression check failed (non-fatal): %s", reg_err)
