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

        # Fetch experiment to get model_name for pricing
        from app.models.experiment import Experiment
        exp_query = select(Experiment).where(Experiment.id == experiment_id)
        exp_result = await self.db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        model_name = experiment.config.get("model_name", "") if experiment and experiment.config else ""

        # Compute metrics
        accuracy = self._compute_accuracy(runs)
        latency = self._compute_latency(runs, wall_clock_ms)
        cost = self._compute_cost(runs, model_name=model_name)
        faithfulness_metrics = self._compute_faithfulness(runs)
        similarity_metrics = self._compute_semantic_similarity(runs)
        failure_modes = self._compute_failure_modes(runs)
        robustness_metrics = self._compute_robustness(runs)
        retrieval_quality = self._compute_retrieval_quality(runs, experiment)

        # Compute completion quality from failure rate
        total_failures = failure_modes.get("total_failures", 0)
        failure_rate = total_failures / len(runs) if runs else 0
        if failure_rate == 0:
            completion_quality = "full"
        elif failure_rate <= 0.3:
            completion_quality = "partial"
        else:
            completion_quality = "degraded"

        # Generate natural language summary
        dataset_name = experiment.config.get("dataset_name", "unknown") if experiment and experiment.config else "unknown"
        summary_text = self._generate_summary(
            accuracy=accuracy,
            latency=latency,
            cost=cost,
            faithfulness=faithfulness_metrics,
            failure_modes=failure_modes,
            experiment_id=experiment_id,
            robustness=robustness_metrics,
            dataset_name=dataset_name,
            total_samples=len(runs),
            completion_quality=completion_quality,
        )

        # Keys owned by this service — all other keys in raw_metrics are preserved
        _METRICS_OWNED_KEYS = {
            "summary_text", "accuracy", "latency", "cost", "faithfulness",
            "semantic_similarity", "failure_modes", "robustness", "attempt", "per_run",
            "retrieval_quality", "completion_quality",
        }

        # Build raw metrics dict (only keys this service owns)
        raw_metrics = {
            "summary_text": summary_text,
            "accuracy": accuracy,
            "latency": latency,
            "cost": cost,
            "faithfulness": faithfulness_metrics,
            "semantic_similarity": similarity_metrics,
            "failure_modes": failure_modes,
            "robustness": robustness_metrics,
            "retrieval_quality": retrieval_quality,
            "completion_quality": completion_quality,
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
                    "failure_mode": run.failure_mode.value if run.failure_mode else None,
                    "error_message": run.error_message,
                    "served_provider": run.served_provider,
                    "routing_reason": run.routing_reason,
                    "cost_usd": run.cost_usd,
                    "grader_results": run.grader_results,
                }
                for run in runs
            ],
        }

        # Check if result already exists (upsert)
        existing_query = select(Result).where(Result.experiment_id == experiment_id)
        existing_result = await self.db.execute(existing_query)
        db_result = existing_result.scalar_one_or_none()

        # Merge-safe: preserve keys written by other services (regression, routing, etc.)
        if db_result and db_result.raw_metrics:
            preserved = {k: v for k, v in db_result.raw_metrics.items()
                         if k not in _METRICS_OWNED_KEYS}
            raw_metrics.update(preserved)

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

    @staticmethod
    def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
        """Wilson score interval for binomial proportions — works well for small n."""
        if total == 0:
            return 0.0, 0.0
        p = successes / total
        denom = 1 + z * z / total
        centre = (p + z * z / (2 * total)) / denom
        margin = (z / denom) * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def _generate_summary(
        self,
        accuracy: dict,
        latency: dict,
        cost: dict,
        faithfulness: dict,
        failure_modes: dict,
        experiment_id: UUID,
        robustness: Optional[dict] = None,
        dataset_name: str = "unknown",
        total_samples: int = 0,
        completion_quality: str = "full",
    ) -> str:
        """
        Generate a natural language summary with uncertainty caveats.

        Includes Wilson CI, dataset provenance, and completion quality.
        """
        acc = accuracy.get("accuracy_any", 0) * 100
        total = accuracy.get("total_evaluated", 0)
        correct_count = int(acc / 100 * total) if total > 0 else 0
        ci_lower, ci_upper = self._wilson_ci(correct_count, total)
        p50 = latency.get("p50", 0)

        # Load dataset display name if metadata exists
        display_name = dataset_name
        try:
            from app.services.dataset_service import DatasetService
            meta = DatasetService.get_dataset_metadata(dataset_name)
            if meta:
                display_name = meta.get("display_name", dataset_name)
        except Exception:
            pass
        
        if "total_cost_usd" in cost:
            total_cost = cost.get("total_cost_usd") or 0.0
        else:
            input_cost = (cost.get("total_tokens_input", 0) / 1_000_000) * 0.15
            output_cost = (cost.get("total_tokens_output", 0) / 1_000_000) * 0.60
            total_cost = input_cost + output_cost
        cost_label = "Recorded" if cost.get("cost_source") == "observed_per_run" else "Estimated"

        summary = f"This experiment achieved {acc:.1f}% correctness (95% CI: {ci_lower * 100:.1f}\u2013{ci_upper * 100:.1f}%) across {total} diagnostic samples. "
        summary += f"Dataset: {display_name}. "

        if completion_quality == "partial":
            failure_pct = failure_modes.get("total_failures", 0) / total * 100 if total > 0 else 0
            summary += f"Note: {failure_pct:.0f}% of runs had infrastructure failures. "
        elif completion_quality == "degraded":
            failure_pct = failure_modes.get("total_failures", 0) / total * 100 if total > 0 else 0
            summary += f"WARNING: {failure_pct:.0f}% of runs failed \u2014 accuracy is severely impacted by infrastructure noise. "

        summary += f"Median latency: {p50:.0f}ms. "
        
        # Add RAG/Faithfulness context if applicable
        if faithfulness.get("count", 0) > 0:
            unsupported_rate = faithfulness.get("unsupported_rate", faithfulness.get("hallucination_rate", 0)) * 100
            summary += f"The RAG context-support proxy marked {unsupported_rate:.1f}% of answers as low-support. "

        if robustness and robustness.get("total", 0) > 0:
            safety = robustness.get("safety_score", 0) * 100
            inconclusive = (robustness.get("breakdown") or {}).get("inconclusive_pct", 0)
            summary += f"Adversarial safety scoring passed {safety:.1f}% of prompts with {inconclusive:.1f}% inconclusive. "
            
        if total_cost > 0:
            summary += f"{cost_label} inference cost for this run was ${total_cost:.4f}. "
        else:
            summary += "Inference was completed with no measurable API token costs. "
            
        total_failures = failure_modes.get("total_failures", 0)
        if total_failures > 0:
            counts = failure_modes.get("counts", {})
            top_failures = ", ".join([f"{count} {mode.replace('_', ' ')}" for mode, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:2]])
            summary += f"Encountered {total_failures} failures during execution (top: {top_failures})."
            
        return summary.strip()
        
    def _compute_failure_modes(self, runs: List[Run]) -> dict:
        """Aggregate failure modes across runs."""
        from collections import Counter
        counts = Counter()
        error_messages = []
        for r in runs:
            if r.failure_mode:
                counts[r.failure_mode.value] += 1
                if r.error_message:
                    error_messages.append({"mode": r.failure_mode.value, "error": r.error_message, "example_id": r.example_id})
                    
        return {
            "counts": dict(counts),
            "total_failures": sum(counts.values()),
            "sample_errors": error_messages[:10]  # Keep a sample of up to 10 errors
        }

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

        # Accuracy excluding infrastructure failures (model performance only)
        non_failure_runs = [r for r in runs if r.failure_mode is None]
        non_failure_total = len(non_failure_runs)
        non_failure_correct = sum(
            1 for r in non_failure_runs
            if (r.is_exact_match or r.is_substring_match or r.is_correct)
        )

        return {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "substring": substring_matches / total if total > 0 else 0.0,
            "accuracy_any": (exact_matches + substring_matches) / total if total > 0 else 0.0,
            "accuracy_excluding_failures": non_failure_correct / non_failure_total if non_failure_total > 0 else 0.0,
            "total_excluding_failures": non_failure_total,
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

    def _compute_cost(self, runs: List[Run], model_name: str = "") -> dict:
        """
        Compute cost proxy metrics from runs.

        Returns total tokens, runs count, estimated GPU time,
        and cost estimation using pricing lookup.
        """
        from app.core.pricing import estimate_cost

        total_input = sum(run.tokens_input or 0 for run in runs)
        total_output = sum(run.tokens_output or 0 for run in runs)
        total_latency_ms = sum(run.latency_ms or 0 for run in runs)

        observed_costs = [
            float(cost)
            for run in runs
            for cost in [getattr(run, "cost_usd", None)]
            if isinstance(cost, (int, float))
        ]
        served_providers = sorted(
            {
                provider
                for run in runs
                for provider in [getattr(run, "served_provider", None)]
                if isinstance(provider, str) and provider
            }
        )

        # Prefer per-run provider costs when available; fall back to pricing lookup for legacy runs.
        cost_estimate = estimate_cost(model_name, total_input, total_output)
        if observed_costs:
            total_cost_usd = round(sum(observed_costs), 8)
            provider = served_providers[0] if len(served_providers) == 1 else "mixed" if served_providers else cost_estimate["provider"]
            cost_source = "observed_per_run"
        else:
            total_cost_usd = cost_estimate["total_cost_usd"]
            provider = cost_estimate["provider"]
            cost_source = "pricing_table_estimate"

        # Cost per correct answer
        correct_count = sum(1 for r in runs if r.is_correct)
        cost_per_correct = (
            round(total_cost_usd / correct_count, 6)
            if correct_count > 0
            else None
        )

        return {
            "total_tokens_input": total_input,
            "total_tokens_output": total_output,
            "total_tokens": total_input + total_output,
            "total_runs": len(runs),
            "gpu_time_seconds": total_latency_ms / 1000.0,
            "total_cost_usd": total_cost_usd,
            "cost_per_correct_answer": cost_per_correct,
            "cost_per_sample_usd": round(total_cost_usd / len(runs), 8) if len(runs) > 0 and total_cost_usd else 0,
            "accuracy_per_dollar": round((correct_count / len(runs)) / total_cost_usd, 4) if total_cost_usd and total_cost_usd > 0 and len(runs) > 0 else None,
            "provider": provider,
            "cost_source": cost_source,
        }

    # =========================================================================
    # P0 #4: Aggregate faithfulness from per-run scores
    # =========================================================================

    def _compute_faithfulness(self, runs: List[Run]) -> dict:
        """
        Aggregate context-support scores from per-run RAG evaluations.

        ``hallucination_rate`` is kept as a legacy API field, but this scorer
        is only a low context-support proxy.
        """
        scores = [
            run.faithfulness_score
            for run in runs
            if run.faithfulness_score is not None
        ]

        if not scores:
            return {
                "mean": None,
                "unsupported_rate": None,
                "hallucination_rate": None,
                "count": 0,
                "method": "hf_zero_shot_context_support_proxy",
                "threshold": 0.5,
            }

        arr = np.array(scores)
        unsupported_rate = float(np.mean(arr < 0.5))
        mean_score = float(np.mean(arr))
        return {
            "mean": mean_score,
            "context_support_score": mean_score,
            "unsupported_rate": unsupported_rate,
            "hallucination_rate": unsupported_rate,  # legacy alias
            "count": len(scores),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "method": "hf_zero_shot_context_support_proxy",
            "methodology_note": (
                "Based on NLI proxy (bart-large-mnli). This measures estimated "
                "context support, not validated factual accuracy."
            ),
            "threshold": 0.5,
        }

    def _compute_robustness(self, runs: List[Run]) -> Optional[dict]:
        """Aggregate deterministic robustness classifications from grader results."""
        classifications = []
        for run in runs:
            grader_results = run.grader_results or {}
            robustness = grader_results.get("robustness") if isinstance(grader_results, dict) else None
            if isinstance(robustness, dict):
                classifications.append(robustness)

        if not classifications:
            return None

        from app.services.robustness_scorer import compute_safety_score

        return compute_safety_score(classifications)

    def _compute_retrieval_quality(self, runs: List[Run], experiment) -> Optional[dict]:
        """
        Compute retrieval quality metrics for RAG experiments.

        Uses gold evidence metadata (evidence_source, gold_chunk_keywords) from
        the dataset to compute:
        - recall_at_k: fraction of examples where gold source appeared in retrieved chunks
        - evidence_hit_rate: fraction where a retrieved chunk contains any gold keyword
        - avg_top_score: average score of top-ranked retrieved chunk

        Returns None for non-RAG experiments or when no gold evidence is available.
        """
        config = experiment.config if experiment and experiment.config else {}
        if not config.get("rag"):
            return None

        # Load dataset to get gold evidence
        dataset_name = config.get("dataset_name", "")
        if not dataset_name:
            return None

        try:
            from app.services.dataset_service import DatasetService
            dataset = DatasetService.load_dataset(dataset_name)
        except Exception:
            return None

        # Build gold evidence lookup
        gold_evidence = {}
        for item in dataset:
            if "evidence_source" in item or "gold_chunk_keywords" in item:
                gold_evidence[item["id"]] = {
                    "evidence_source": item.get("evidence_source", ""),
                    "gold_chunk_keywords": item.get("gold_chunk_keywords", []),
                }

        if not gold_evidence:
            return {"status": "no_gold_evidence", "annotated_count": 0}

        source_hits = 0
        keyword_hits = 0
        annotated_evaluated = 0
        top_scores = []

        for run in runs:
            if run.example_id not in gold_evidence:
                continue

            gold = gold_evidence[run.example_id]
            chunks_data = (run.retrieved_chunks or {}).get("chunks", [])
            if not chunks_data:
                annotated_evaluated += 1
                continue

            annotated_evaluated += 1

            # Collect top score
            scores_list = [c.get("score") for c in chunks_data if c.get("score") is not None]
            if scores_list:
                top_scores.append(max(scores_list))

            # Check evidence source in chunk titles
            gold_source_lower = gold["evidence_source"].lower()
            if gold_source_lower and any(
                gold_source_lower in (c.get("title", "") or "").lower()
                for c in chunks_data
            ):
                source_hits += 1

            # Check gold keywords in chunk text
            gold_keywords = [kw.lower() for kw in gold.get("gold_chunk_keywords", [])]
            if gold_keywords:
                for chunk in chunks_data:
                    chunk_text_lower = (chunk.get("text", "") or "").lower()
                    if any(kw in chunk_text_lower for kw in gold_keywords):
                        keyword_hits += 1
                        break

        if annotated_evaluated == 0:
            return {"status": "no_annotated_runs", "annotated_count": len(gold_evidence)}

        return {
            "status": "computed",
            "annotated_count": len(gold_evidence),
            "evaluated_count": annotated_evaluated,
            "recall_at_k": round(source_hits / annotated_evaluated, 4),
            "evidence_hit_rate": round(keyword_hits / annotated_evaluated, 4),
            "avg_top_score": round(float(np.mean(top_scores)), 4) if top_scores else None,
            "source_hits": source_hits,
            "keyword_hits": keyword_hits,
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
