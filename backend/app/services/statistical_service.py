"""
Statistical Service

Provides statistical analysis for comparing experiments:
- Bootstrap confidence intervals for accuracy metrics
- McNemar's test for paired accuracy comparison
- Per-example agreement/disagreement analysis
"""

import logging
from typing import List, Dict, Tuple, Optional
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.run import Run

logger = logging.getLogger(__name__)


class StatisticalService:
    """
    Service for statistical comparison of experiments.
    
    Used in Phase 4 to validate whether CoT improves over
    Naive prompting with statistical significance.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @staticmethod
    def bootstrap_confidence_interval(
        values: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            values: List of per-example metric values (e.g., per-run scores)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (0.95 = 95% CI)
            seed: Random seed for reproducibility
            
        Returns:
            Dict with mean, lower, upper, and std
        """
        if not values:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}
        
        rng = np.random.default_rng(seed)
        arr = np.array(values)
        n = len(arr)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=n, replace=True)
            bootstrap_means.append(float(np.mean(sample)))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        
        return {
            "mean": float(np.mean(arr)),
            "lower": lower,
            "upper": upper,
            "std": float(np.std(bootstrap_means)),
        }
    
    @staticmethod
    def mcnemar_test(
        correct_a: List[bool],
        correct_b: List[bool],
    ) -> Dict[str, float]:
        """
        Compute McNemar's test for paired accuracy comparison.
        
        Compares two methods on the same set of examples.
        The test checks whether the disagreements between methods
        are symmetric (null hypothesis) or not (significant difference).
        
        Args:
            correct_a: Per-example correctness for method A
            correct_b: Per-example correctness for method B
            
        Returns:
            Dict with statistic, p_value, is_significant, and contingency table counts
        """
        if len(correct_a) != len(correct_b):
            raise ValueError("Both methods must have the same number of results")
        
        if len(correct_a) == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "b": 0,  # A correct, B wrong
                "c": 0,  # A wrong, B correct
                "n": 0,
            }
        
        # Count disagreements
        # b = number where A is correct but B is wrong
        # c = number where A is wrong but B is correct
        b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
        c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
        
        n = len(correct_a)
        
        # McNemar's test with continuity correction
        # If b + c is too small, the test is unreliable
        if b + c == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "b": b,
                "c": c,
                "n": n,
            }
        
        try:
            from statsmodels.stats.contingency_tables import mcnemar
            
            # Build 2x2 contingency table
            # [[both correct, A correct B wrong], [A wrong B correct, both wrong]]
            both_correct = sum(1 for a, bb in zip(correct_a, correct_b) if a and bb)
            both_wrong = sum(1 for a, bb in zip(correct_a, correct_b) if not a and not bb)
            
            table = [[both_correct, b], [c, both_wrong]]
            
            # Use exact test for small samples, chi-square for larger
            use_exact = (b + c) < 25
            result = mcnemar(table, exact=use_exact)
            
            return {
                "statistic": float(result.statistic) if hasattr(result, 'statistic') else 0.0,
                "p_value": float(result.pvalue),
                "is_significant": float(result.pvalue) < 0.05,
                "b": b,
                "c": c,
                "n": n,
            }
        except ImportError:
            logger.warning("statsmodels not installed, using manual McNemar's test")
            # Manual chi-square approximation
            statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
            # Approximate p-value using chi-square with 1 df
            from scipy.stats import chi2
            p_value = 1.0 - chi2.cdf(statistic, df=1)
            
            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": float(p_value) < 0.05,
                "b": b,
                "c": c,
                "n": n,
            }
    
    async def compare_experiments(
        self,
        experiment_a_id: UUID,
        experiment_b_id: UUID,
    ) -> Dict:
        """
        Full statistical comparison between two experiments.
        
        Matches runs by example_id and computes:
        - McNemar's test for accuracy comparison
        - Bootstrap CIs for both experiments' scores
        - Per-example agreement/disagreement breakdown
        
        Args:
            experiment_a_id: UUID of first experiment
            experiment_b_id: UUID of second experiment
            
        Returns:
            Dict with statistical comparison results
        """
        # Fetch runs for both experiments
        query_a = select(Run).where(Run.experiment_id == experiment_a_id)
        result_a = await self.db.execute(query_a)
        runs_a = result_a.scalars().all()
        
        query_b = select(Run).where(Run.experiment_id == experiment_b_id)
        result_b = await self.db.execute(query_b)
        runs_b = result_b.scalars().all()
        
        if not runs_a or not runs_b:
            raise ValueError("Both experiments must have runs to compare")
        
        # Index runs by example_id for matching
        runs_a_by_example = {r.example_id: r for r in runs_a}
        runs_b_by_example = {r.example_id: r for r in runs_b}
        
        # Find common examples
        common_examples = set(runs_a_by_example.keys()) & set(runs_b_by_example.keys())
        
        if not common_examples:
            raise ValueError("No common examples found between experiments")
        
        # Build paired lists
        correct_a = []
        correct_b = []
        scores_a = []
        scores_b = []
        per_example = []
        
        for example_id in sorted(common_examples):
            run_a = runs_a_by_example[example_id]
            run_b = runs_b_by_example[example_id]
            
            ca = bool(run_a.is_correct)
            cb = bool(run_b.is_correct)
            correct_a.append(ca)
            correct_b.append(cb)
            
            sa = float(run_a.score or 0.0)
            sb = float(run_b.score or 0.0)
            scores_a.append(sa)
            scores_b.append(sb)
            
            # Track per-example differences
            if ca != cb:
                per_example.append({
                    "example_id": example_id,
                    "a_correct": ca,
                    "b_correct": cb,
                    "a_output": run_a.output_text,
                    "b_output": run_b.output_text,
                    "expected": run_a.expected_output,
                    "a_score": sa,
                    "b_score": sb,
                })
        
        # Compute McNemar's test
        mcnemar_result = self.mcnemar_test(correct_a, correct_b)
        
        # Compute bootstrap CIs
        ci_a = self.bootstrap_confidence_interval(scores_a)
        ci_b = self.bootstrap_confidence_interval(scores_b)
        
        # Accuracy summary
        acc_a = sum(correct_a) / len(correct_a) if correct_a else 0.0
        acc_b = sum(correct_b) / len(correct_b) if correct_b else 0.0
        
        return {
            "experiment_a_id": str(experiment_a_id),
            "experiment_b_id": str(experiment_b_id),
            "num_common_examples": len(common_examples),
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "accuracy_diff": acc_b - acc_a,
            "mcnemar": mcnemar_result,
            "bootstrap_ci_a": ci_a,
            "bootstrap_ci_b": ci_b,
            "per_example_differences": per_example[:50],  # Limit to 50
            "summary": {
                "both_correct": sum(1 for a, b in zip(correct_a, correct_b) if a and b),
                "both_wrong": sum(1 for a, b in zip(correct_a, correct_b) if not a and not b),
                "a_only_correct": mcnemar_result["b"],
                "b_only_correct": mcnemar_result["c"],
            },
        }
