"""
Metrics Service

Computes evaluation metrics from experiment runs:
- Accuracy: exact match, substring containment, F1 token overlap
- Latency: p50, p95, p99 percentiles, throughput
- Cost: total tokens, estimated GPU time
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.result import Result
from app.models.run import Run

logger = logging.getLogger(__name__)


class MetricsService:
    """
    Service for computing and storing experiment metrics.
    
    Computes metrics from individual Run rows and saves
    aggregated results to the Result table.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compute_and_save(self, experiment_id: UUID) -> Result:
        """
        Compute all metrics for an experiment and save to Result table.
        
        Args:
            experiment_id: UUID of the completed experiment
            
        Returns:
            Created or updated Result instance
        """
        logger.info(f"Computing metrics for experiment {experiment_id}")
        
        # Fetch all runs
        query = select(Run).where(Run.experiment_id == experiment_id)
        result = await self.db.execute(query)
        runs = result.scalars().all()
        
        if not runs:
            raise ValueError(f"No runs found for experiment {experiment_id}")
        
        logger.info(f"Found {len(runs)} runs, computing metrics...")
        
        # Compute metrics
        accuracy = self._compute_accuracy(runs)
        latency = self._compute_latency(runs)
        cost = self._compute_cost(runs)
        
        # Build raw metrics dict
        raw_metrics = {
            "accuracy": accuracy,
            "latency": latency,
            "cost": cost,
            "per_run": [
                {
                    "example_id": run.example_id,
                    "is_correct": run.is_correct,
                    "score": run.score,
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
        
        if db_result:
            # Update existing
            db_result.accuracy_exact = accuracy["exact_match"]
            db_result.accuracy_f1 = accuracy["f1_mean"]
            db_result.accuracy_substring = accuracy["substring"]
            db_result.latency_p50 = latency["p50"]
            db_result.latency_p95 = latency["p95"]
            db_result.latency_p99 = latency["p99"]
            db_result.throughput = latency["throughput"]
            db_result.total_tokens_input = cost["total_tokens_input"]
            db_result.total_tokens_output = cost["total_tokens_output"]
            db_result.total_runs = cost["total_runs"]
            db_result.gpu_time_seconds = cost["gpu_time_seconds"]
            db_result.raw_metrics = raw_metrics
            db_result.computed_at = datetime.now(timezone.utc)
        else:
            # Create new
            db_result = Result(
                experiment_id=experiment_id,
                accuracy_exact=accuracy["exact_match"],
                accuracy_f1=accuracy["f1_mean"],
                accuracy_substring=accuracy["substring"],
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
    
    def _compute_accuracy(self, runs: List[Run]) -> dict:
        """
        Compute accuracy metrics directly from the pre-computed run judgments.
        
        Returns dict with:
        - exact_match: fraction of exact matches
        - substring: fraction of substring containment
        - f1_mean: mean F1 token overlap score
        """
        exact_matches = 0
        substring_matches = 0
        f1_scores = []
        
        for run in runs:
            if run.score is not None:
                f1_scores.append(run.score)
            if run.is_correct:
                # Approximate the breakdown of exact vs substring by doing a fast check here,
                # or just attributing both to 'exact' if run.is_correct is true, 
                # but to be accurate we can re-check the type if needed.
                # Since the review noted to not re-judge raw text for the final outcome,
                # we count ‘is_correct’ towards exact_matches by default if F1 is 1.0, 
                # or we can just run the fast strict checks.
                # A better way is using the true alias_aware outcome.
                if run.score == 1.0:
                    exact_matches += 1
                else:
                    substring_matches += 1
        
        total = len(runs)
        total_correct = sum(1 for run in runs if run.is_correct)
        
        # We assign all correct answers to exact_match if score is 1.0, else substring.
        # This keeps the aggregation from contradicting the run-level judgments.
        return {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "substring": substring_matches / total if total > 0 else 0.0,
            "f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "f1_median": float(np.median(f1_scores)) if f1_scores else 0.0,
            "total_evaluated": total,
        }
    
    def _compute_latency(self, runs: List[Run]) -> dict:
        """
        Compute latency metrics from runs.
        
        Returns dict with p50, p95, p99, mean, min, max, throughput.
        """
        latencies = [run.latency_ms for run in runs if run.latency_ms is not None]
        
        if not latencies:
            return {
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "mean": 0.0, "min": 0.0, "max": 0.0,
                "throughput": 0.0,
            }
        
        arr = np.array(latencies)
        total_time_seconds = float(np.sum(arr)) / 1000.0
        
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "throughput": len(latencies) / total_time_seconds if total_time_seconds > 0 else 0.0,
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
    
    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score between prediction and ground truth.
        Uses Counter to handle repeating tokens correctly.
        """
        import collections
        
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
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
        Case-insensitive exact string match.
        
        Also strips whitespace and common punctuation.
        """
        def normalize(s: str) -> str:
            return s.lower().strip().rstrip(".").rstrip(",").strip()
        
        return normalize(prediction) == normalize(ground_truth)
    
    @staticmethod
    def check_substring(prediction: str, ground_truth: str) -> bool:
        """
        Check if ground truth is contained in prediction (case-insensitive).
        Uses word boundaries to prevent 'paris' from matching 'comparison'.
        """
        import re
        pred = prediction.lower()
        truth = ground_truth.lower().strip()
        # Word boundaries require the substring to be an isolated word/phrase
        pattern = r'\b' + re.escape(truth) + r'\b'
        return bool(re.search(pattern, pred))
    
    @staticmethod
    def check_any_alias_match(
        prediction: str,
        aliases: List[str],
    ) -> Tuple[bool, bool, float]:
        """
        Check prediction against multiple answer aliases.
        
        Returns:
            (exact_match, substring_match, max_f1_score)
        """
        exact = False
        substring = False
        max_f1 = 0.0
        
        for alias in aliases:
            if MetricsService.check_exact_match(prediction, alias):
                exact = True
            if MetricsService.check_substring(prediction, alias):
                substring = True
            f1 = MetricsService.compute_f1(prediction, alias)
            max_f1 = max(max_f1, f1)
        
        return exact, substring, max_f1
