"""
Metrics Service

Computes evaluation metrics from experiment runs:
- Accuracy: exact match, substring containment, F1 token overlap
- Semantic: embedding cosine similarity (P1 #9)
- Latency: p50, p95, p99 percentiles, throughput (wall-clock based)
- Cost: total tokens, estimated GPU time
- Faithfulness: aggregated from per-run NLI scores (P0 #4)
"""

import collections
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.result import Result
from app.models.run import Run

logger = logging.getLogger(__name__)


# =============================================================================
# Shared normalization (P0 #2) — consistent preprocessing across all metrics
# =============================================================================

def _normalize(text: str) -> str:
    """
    Normalize text for consistent evaluation.

    Applied identically across exact match, substring, and F1 computations.
    """
    # Lowercase
    s = text.lower()
    # Strip leading/trailing whitespace
    s = s.strip()
    # Remove trailing punctuation (., ,, !, ?, ;, :)
    s = re.sub(r'[.,!?;:]+$', '', s)
    # Collapse internal whitespace
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


class MetricsService:
    """
    Service for computing and storing experiment metrics.

    Computes metrics from individual Run rows and saves
    aggregated results to the Result table.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def compute_and_save(
        self,
        experiment_id: UUID,
        wall_clock_ms: Optional[float] = None,
    ) -> Result:
        """
        Compute all metrics for an experiment and save to Result table.

        Args:
            experiment_id: UUID of the completed experiment
            wall_clock_ms: Total wall-clock execution time in ms (for accurate throughput)

        Returns:
            Created or updated Result instance
        """
        logger.info(f"Computing metrics for experiment {experiment_id}")

        # Fetch all runs (latest attempt only)
        query = select(Run).where(Run.experiment_id == experiment_id)
        result = await self.db.execute(query)
        all_runs = result.scalars().all()

        if not all_runs:
            raise ValueError(f"No runs found for experiment {experiment_id}")

        # Use only the latest attempt
        max_attempt = max(r.attempt for r in all_runs)
        runs = [r for r in all_runs if r.attempt == max_attempt]

        logger.info(f"Found {len(runs)} runs (attempt {max_attempt}), computing metrics...")

        # Compute metrics
        accuracy = self._compute_accuracy(runs)
        latency = self._compute_latency(runs, wall_clock_ms)
        cost = self._compute_cost(runs)
        faithfulness_metrics = self._compute_faithfulness(runs)
        similarity_metrics = self._compute_semantic_similarity(runs)

        # Build raw metrics dict
        raw_metrics = {
            "accuracy": accuracy,
            "latency": latency,
            "cost": cost,
            "faithfulness": faithfulness_metrics,
            "semantic_similarity": similarity_metrics,
            "attempt": max_attempt,
            "per_run": [
                {
                    "example_id": run.example_id,
                    "is_correct": run.is_correct,
                    "is_exact_match": run.is_exact_match,
                    "is_substring_match": run.is_substring_match,
                    "score": run.score,
                    "semantic_similarity": run.semantic_similarity,
                    "faithfulness_score": run.faithfulness_score,
                    "latency_ms": run.latency_ms,
                    "tokens_input": run.tokens_input,
                    "tokens_output": run.tokens_output,
                }
                for run in runs
            ],
        }

        # Check if result already exists (upsert)
        existing_query = select(Result).where(Result.experiment_id == experiment_id)
        existing_result = await self.db.execute(existing_query)
        db_result = existing_result.scalar_one_or_none()

        # Common field values
        fields = dict(
            accuracy_exact=accuracy["exact_match"],
            accuracy_f1=accuracy["f1_mean"],
            accuracy_substring=accuracy["substring"],
            semantic_similarity=similarity_metrics.get("mean"),
            faithfulness=faithfulness_metrics.get("mean"),
            hallucination_rate=faithfulness_metrics.get("hallucination_rate"),
            latency_p50=latency["p50"],
            latency_p95=latency["p95"],
            latency_p99=latency["p99"],
            throughput=latency["throughput"],
            total_tokens_input=cost["total_tokens_input"],
            total_tokens_output=cost["total_tokens_output"],
            total_runs=cost["total_runs"],
            gpu_time_seconds=cost["gpu_time_seconds"],
            raw_metrics=raw_metrics,
            computed_at=datetime.now(timezone.utc),
        )

        if db_result:
            for k, v in fields.items():
                setattr(db_result, k, v)
        else:
            db_result = Result(experiment_id=experiment_id, **fields)
            self.db.add(db_result)

        await self.db.flush()
        await self.db.refresh(db_result)

        logger.info(
            f"Metrics saved: accuracy_exact={accuracy['exact_match']:.3f}, "
            f"f1={accuracy['f1_mean']:.3f}, p50={latency['p50']:.1f}ms"
        )

        return db_result

    async def clear_results(self, experiment_id: UUID) -> None:
        """
        Delete aggregated results for an experiment.

        Useful when re-running an experiment to clear old data.

        Args:
            experiment_id: UUID of the experiment
        """
        await self.db.execute(
            delete(Result).where(Result.experiment_id == experiment_id)
        )
        await self.db.flush()

    # =========================================================================
    # P0 #1: Accuracy from stored booleans (not reconstructed from score)
    # =========================================================================

    def _compute_accuracy(self, runs: List[Run]) -> dict:
        """
        Compute accuracy metrics directly from the stored boolean flags.

        Uses is_exact_match / is_substring_match when available (new runs),
        falls back to is_correct / score heuristic for legacy runs.
        """
        exact_matches = 0
        substring_matches = 0
        f1_scores = []

        for run in runs:
            if run.score is not None:
                f1_scores.append(run.score)

            # Prefer stored boolean flags (P0 fix)
            if run.is_exact_match is not None:
                if run.is_exact_match:
                    exact_matches += 1
                elif run.is_substring_match:
                    substring_matches += 1
            elif run.is_correct:
                # Legacy fallback: approximate from score
                if run.score == 1.0:
                    exact_matches += 1
                else:
                    substring_matches += 1

        total = len(runs)

        return {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "substring": substring_matches / total if total > 0 else 0.0,
            "accuracy_any": (exact_matches + substring_matches) / total if total > 0 else 0.0,
            "f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "f1_median": float(np.median(f1_scores)) if f1_scores else 0.0,
            "total_evaluated": total,
        }

    # =========================================================================
    # P0 #3: Throughput from wall-clock time (not sum of per-run latency)
    # =========================================================================

    def _compute_latency(self, runs: List[Run], wall_clock_ms: Optional[float] = None) -> dict:
        """
        Compute latency metrics from runs.

        Throughput uses wall-clock experiment duration when available,
        falling back to sum-of-latencies for legacy data.
        """
        latencies = [run.latency_ms for run in runs if run.latency_ms is not None]

        if not latencies:
            return {
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "mean": 0.0, "min": 0.0, "max": 0.0,
                "throughput": 0.0,
                "throughput_source": "none",
            }

        arr = np.array(latencies)

        # P0 #3: Prefer wall-clock time for throughput
        if wall_clock_ms and wall_clock_ms > 0:
            throughput = len(latencies) / (wall_clock_ms / 1000.0)
            throughput_source = "wall_clock"
        else:
            total_time_seconds = float(np.sum(arr)) / 1000.0
            throughput = len(latencies) / total_time_seconds if total_time_seconds > 0 else 0.0
            throughput_source = "sum_latency_fallback"

        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "throughput": throughput,
            "throughput_source": throughput_source,
        }

    def _compute_cost(self, runs: List[Run]) -> dict:
        """
        Compute cost proxy metrics from runs.

        Returns total tokens, runs count, estimated GPU time.
        """
        total_input = sum(run.tokens_input or 0 for run in runs)
        total_output = sum(run.tokens_output or 0 for run in runs)
        total_latency_ms = sum(run.latency_ms or 0 for run in runs)

        return {
            "total_tokens_input": total_input,
            "total_tokens_output": total_output,
            "total_tokens": total_input + total_output,
            "total_runs": len(runs),
            "gpu_time_seconds": total_latency_ms / 1000.0,
        }

    # =========================================================================
    # P0 #4: Aggregate faithfulness from per-run scores
    # =========================================================================

    def _compute_faithfulness(self, runs: List[Run]) -> dict:
        """
        Aggregate faithfulness scores from per-run NLI evaluations.

        Returns mean faithfulness and hallucination rate (fraction < 0.5).
        """
        scores = [
            run.faithfulness_score
            for run in runs
            if run.faithfulness_score is not None
        ]

        if not scores:
            return {"mean": None, "hallucination_rate": None, "count": 0}

        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "hallucination_rate": float(np.mean(arr < 0.5)),
            "count": len(scores),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # =========================================================================
    # P1 #9: Aggregate semantic similarity
    # =========================================================================

    def _compute_semantic_similarity(self, runs: List[Run]) -> dict:
        """
        Aggregate semantic similarity scores from per-run embeddings.
        """
        scores = [
            run.semantic_similarity
            for run in runs
            if run.semantic_similarity is not None
        ]

        if not scores:
            return {"mean": None, "count": 0}

        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(scores),
        }

    # =========================================================================
    # Text comparison methods (P0 #2: all use shared _normalize)
    # =========================================================================

    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score between prediction and ground truth.

        Uses shared normalization for consistency with exact match.
        """
        pred_tokens = _normalize(prediction).split()
        truth_tokens = _normalize(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 0.0

        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def check_exact_match(prediction: str, ground_truth: str) -> bool:
        """
        Case-insensitive exact string match with shared normalization.
        """
        return _normalize(prediction) == _normalize(ground_truth)

    @staticmethod
    def check_substring(prediction: str, ground_truth: str) -> bool:
        """
        Check if ground truth is contained in prediction (case-insensitive).

        Uses word boundaries to prevent 'paris' from matching 'comparison'.
        """
        pred = _normalize(prediction)
        truth = _normalize(ground_truth)
        pattern = r'\b' + re.escape(truth) + r'\b'
        return bool(re.search(pattern, pred))

    @staticmethod
    def check_any_alias_match(
        prediction: str,
        aliases: List[str],
    ) -> Tuple[bool, bool, float, str]:
        """
        Check prediction against multiple answer aliases.

        Returns:
            (exact_match, substring_match, max_f1_score, matched_alias)
        """
        exact = False
        substring = False
        max_f1 = 0.0
        matched_alias = ""

        for alias in aliases:
            if MetricsService.check_exact_match(prediction, alias):
                exact = True
                matched_alias = alias
            if MetricsService.check_substring(prediction, alias):
                substring = True
                if not matched_alias:
                    matched_alias = alias
            f1 = MetricsService.compute_f1(prediction, alias)
            if f1 > max_f1:
                max_f1 = f1
                if not matched_alias:
                    matched_alias = alias

        return exact, substring, max_f1, matched_alias
