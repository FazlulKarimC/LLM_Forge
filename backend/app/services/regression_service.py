"""
Regression Service

Compares candidate experiments against pinned baselines to detect regressions.

Key design decisions:
- Baselines are immutable: pinned_attempt freezes the attempt at pin time
- Runs are loaded at specific attempts, never "latest"
- Candidate's grader config is applied to BOTH sides (baselines may predate graders)
- Overlap ratio below threshold → inconclusive (None) verdict
- One baseline per (dataset_name, model_name) lineage
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment
from app.models.run import Run
from app.schemas.experiment import ExperimentConfig, RegressionConfig, GradersConfig
from app.services.grader_service import GraderEngine, VerdictStatus
from app.services.statistical_service import StatisticalService

logger = logging.getLogger(__name__)


@dataclass
class RegressionVerdict:
    """Result of a regression comparison between candidate and baseline."""
    passed: Optional[bool]               # None = inconclusive (low overlap)
    baseline_experiment_id: UUID
    baseline_attempt: int                # Frozen: from experiment.pinned_attempt
    candidate_experiment_id: UUID
    candidate_attempt: int               # From experiment.current_attempt
    overlap_ratio: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    sample_regressions: List[Dict] = field(default_factory=list)
    sample_improvements: List[Dict] = field(default_factory=list)
    grader_summary: Dict[str, Any] = field(default_factory=dict)
    statistical_comparison: Dict[str, Any] = field(default_factory=dict)
    config_diff: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for raw_metrics storage."""
        return {
            "passed": self.passed,
            "baseline_id": str(self.baseline_experiment_id),
            "baseline_attempt": self.baseline_attempt,
            "candidate_attempt": self.candidate_attempt,
            "overlap_ratio": self.overlap_ratio,
            "violations": self.violations,
            "sample_regressions_count": len(self.sample_regressions),
            "sample_improvements_count": len(self.sample_improvements),
            "sample_regressions": self.sample_regressions[:20],
            "sample_improvements": self.sample_improvements[:20],
            "grader_summary": self.grader_summary,
            "statistical": self.statistical_comparison,
            "config_diff": self.config_diff,
        }


class RegressionService:
    """
    Orchestrates regression checks between experiments.
    
    Usage:
        svc = RegressionService(db)
        baseline = await svc.find_baseline(experiment)
        if baseline:
            verdict = await svc.run_regression_check(candidate_id, baseline.id)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def find_baseline(self, experiment: Experiment) -> Optional[Experiment]:
        """
        Find the appropriate baseline for comparison.
        
        Resolution order:
        1. Explicit baseline_id on the experiment → use it
        2. Most recent is_baseline=True with same dataset_name AND dataset_hash
        3. None if no match
        """
        # 1. Explicit baseline
        if experiment.baseline_id:
            query = select(Experiment).where(
                Experiment.id == experiment.baseline_id,
                Experiment.deleted_at.is_(None),
            )
            result = await self.db.execute(query)
            baseline = result.scalar_one_or_none()
            if baseline:
                return baseline
        
        # 2. Auto-detect by dataset_name + dataset_hash
        if not experiment.dataset_hash:
            logger.warning(
                "Experiment %s has no dataset_hash — cannot auto-detect baseline",
                experiment.id,
            )
            return None
        
        query = (
            select(Experiment)
            .where(
                Experiment.is_baseline == True,
                Experiment.deleted_at.is_(None),
                Experiment.dataset_name == experiment.dataset_name,
                Experiment.dataset_hash == experiment.dataset_hash,
                Experiment.id != experiment.id,
            )
            .order_by(Experiment.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def run_regression_check(
        self,
        candidate_id: UUID,
        baseline_id: UUID,
    ) -> RegressionVerdict:
        """
        Full regression comparison between candidate and pinned baseline.
        
        Loads runs at specific attempts (pinned_attempt for baseline,
        current_attempt for candidate). Grades both sides with candidate's
        grader config.
        """
        # Load experiments
        candidate_exp = await self._load_experiment(candidate_id)
        baseline_exp = await self._load_experiment(baseline_id)
        
        if not candidate_exp or not baseline_exp:
            raise ValueError("Both experiments must exist for regression check")
        
        # Determine attempts
        baseline_attempt = baseline_exp.pinned_attempt or baseline_exp.current_attempt
        candidate_attempt = candidate_exp.current_attempt
        
        # Load runs at specific attempts
        baseline_runs = await self._load_runs_at_attempt(baseline_id, baseline_attempt)
        candidate_runs = await self._load_runs_at_attempt(candidate_id, candidate_attempt)
        
        if not baseline_runs or not candidate_runs:
            raise ValueError("Both experiments must have runs at their respective attempts")
        
        # Parse configs
        candidate_config = ExperimentConfig(**candidate_exp.config)
        regression_config = candidate_config.regression or RegressionConfig()
        
        # Grade both sides with candidate's grader config
        grader_summary = {}
        if candidate_config.graders and candidate_config.graders.rules:
            grader_summary = self._grade_both_sides(
                baseline_runs, candidate_runs,
                candidate_config.graders,
                candidate_config.reasoning_method.value,
                bool(candidate_config.rag and candidate_config.rag.retrieval_method != "none"),
            )
        
        # Statistical comparison on pre-filtered runs
        try:
            stats = StatisticalService.compare_run_sets(baseline_runs, candidate_runs)
        except ValueError:
            # No common examples
            return RegressionVerdict(
                passed=None,
                baseline_experiment_id=baseline_id,
                baseline_attempt=baseline_attempt,
                candidate_experiment_id=candidate_id,
                candidate_attempt=candidate_attempt,
                overlap_ratio=0.0,
            )
        
        overlap_ratio = stats.get("overlap_ratio", 0.0)
        
        # Check overlap threshold
        if overlap_ratio < regression_config.min_overlap_ratio:
            logger.warning(
                "Overlap ratio %.2f below threshold %.2f — verdict inconclusive",
                overlap_ratio, regression_config.min_overlap_ratio,
            )
            return RegressionVerdict(
                passed=None,
                baseline_experiment_id=baseline_id,
                baseline_attempt=baseline_attempt,
                candidate_experiment_id=candidate_id,
                candidate_attempt=candidate_attempt,
                overlap_ratio=overlap_ratio,
                statistical_comparison=stats,
                grader_summary=grader_summary,
            )
        
        # Check aggregate thresholds
        violations = self._check_aggregate_thresholds(stats, regression_config)
        
        # Detect per-sample regressions
        regressions, improvements = self._detect_sample_regressions(stats)
        
        # Check sample-level constraints
        if regression_config.no_sample_regressions and len(regressions) > 0:
            violations.append({
                "rule": "no_sample_regressions",
                "message": f"{len(regressions)} previously-correct samples now fail",
                "count": len(regressions),
            })
        
        if regression_config.max_new_failures is not None:
            if len(regressions) > regression_config.max_new_failures:
                violations.append({
                    "rule": "max_new_failures",
                    "message": f"{len(regressions)} new failures exceed max {regression_config.max_new_failures}",
                    "count": len(regressions),
                    "max": regression_config.max_new_failures,
                })
        
        # Config diff
        config_diff = self._compute_config_diff(
            baseline_exp.run_manifest,
            candidate_exp.run_manifest,
        )
        
        passed = len(violations) == 0
        
        return RegressionVerdict(
            passed=passed,
            baseline_experiment_id=baseline_id,
            baseline_attempt=baseline_attempt,
            candidate_experiment_id=candidate_id,
            candidate_attempt=candidate_attempt,
            overlap_ratio=overlap_ratio,
            violations=violations,
            sample_regressions=regressions,
            sample_improvements=improvements,
            grader_summary=grader_summary,
            statistical_comparison=stats,
            config_diff=config_diff,
        )

    async def pin_baseline(self, experiment_id: UUID) -> Experiment:
        """
        Pin an experiment as the baseline for its lineage.
        
        Rules:
        1. Only COMPLETED experiments can be pinned
        2. pinned_attempt = current_attempt at pin time
        3. One baseline per (dataset_name, model_name) — unpins old
        4. Cannot pin an experiment that has baseline_id set
        """
        experiment = await self._load_experiment(experiment_id)
        if not experiment:
            raise ValueError("Experiment not found")
        
        if experiment.status != "completed":
            raise ValueError("Only completed experiments can be pinned as baselines")
        
        if experiment.baseline_id is not None:
            raise ValueError("Cannot pin an experiment that references another baseline")
        
        # Unpin existing baselines for the same lineage
        existing_query = select(Experiment).where(
            Experiment.is_baseline == True,
            Experiment.dataset_name == experiment.dataset_name,
            Experiment.model_name == experiment.model_name,
            Experiment.deleted_at.is_(None),
            Experiment.id != experiment_id,
        )
        result = await self.db.execute(existing_query)
        for old_baseline in result.scalars().all():
            old_baseline.is_baseline = False
            old_baseline.pinned_attempt = None
            logger.info("Unpinned old baseline %s", old_baseline.id)
        
        # Pin this experiment
        experiment.is_baseline = True
        experiment.pinned_attempt = experiment.current_attempt
        
        await self.db.flush()
        logger.info(
            "Pinned experiment %s as baseline (attempt=%d)",
            experiment_id, experiment.pinned_attempt,
        )
        return experiment

    async def unpin_baseline(self, experiment_id: UUID) -> Experiment:
        """Remove baseline status. Does NOT invalidate previous regression verdicts."""
        experiment = await self._load_experiment(experiment_id)
        if not experiment:
            raise ValueError("Experiment not found")
        
        experiment.is_baseline = False
        experiment.pinned_attempt = None
        await self.db.flush()
        return experiment

    # ─── Private helpers ─────────────────────────────────────────────────

    async def _load_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        query = select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.deleted_at.is_(None),
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _load_runs_at_attempt(
        self, experiment_id: UUID, attempt: int
    ) -> List[Run]:
        query = select(Run).where(
            Run.experiment_id == experiment_id,
            Run.attempt == attempt,
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    def _grade_both_sides(
        self,
        baseline_runs: List[Run],
        candidate_runs: List[Run],
        graders_config: GradersConfig,
        reasoning_method: str,
        has_rag: bool,
    ) -> Dict[str, Any]:
        """Grade both sides with candidate's grader config and summarize."""
        engine = GraderEngine()
        
        baseline_results = engine.grade_all_runs(
            baseline_runs, graders_config, reasoning_method, has_rag,
        )
        candidate_results = engine.grade_all_runs(
            candidate_runs, graders_config, reasoning_method, has_rag,
        )
        
        # Summarize: per-grader pass/fail/skip counts for both sides
        summary: Dict[str, Any] = {"baseline": {}, "candidate": {}}
        
        for side_name, side_results in [("baseline", baseline_results), ("candidate", candidate_results)]:
            for run_id, verdicts in side_results.items():
                for v in verdicts:
                    if v.grader_name not in summary[side_name]:
                        summary[side_name][v.grader_name] = {"pass": 0, "fail": 0, "skip": 0}
                    summary[side_name][v.grader_name][v.status.value] += 1
        
        return summary

    @staticmethod
    def _check_aggregate_thresholds(
        stats: Dict[str, Any],
        config: RegressionConfig,
    ) -> List[Dict[str, Any]]:
        """Check aggregate metrics against regression thresholds."""
        violations = []
        
        accuracy_diff = stats.get("accuracy_diff", 0.0)
        
        # accuracy_diff = candidate - baseline (negative = candidate is worse)
        if accuracy_diff < config.accuracy_min_delta:
            violations.append({
                "rule": "accuracy_min_delta",
                "message": f"Accuracy dropped {accuracy_diff:.4f} (threshold: {config.accuracy_min_delta})",
                "actual": accuracy_diff,
                "threshold": config.accuracy_min_delta,
            })
        
        # F1 check via CIs
        f1_ci_a = stats.get("f1_ci_a", {})
        f1_ci_b = stats.get("f1_ci_b", {})
        if f1_ci_a and f1_ci_b:
            f1_diff = f1_ci_b.get("mean", 0) - f1_ci_a.get("mean", 0)
            if f1_diff < config.f1_min_delta:
                violations.append({
                    "rule": "f1_min_delta",
                    "message": f"F1 dropped {f1_diff:.4f} (threshold: {config.f1_min_delta})",
                    "actual": f1_diff,
                    "threshold": config.f1_min_delta,
                })
        
        return violations

    @staticmethod
    def _detect_sample_regressions(
        stats: Dict[str, Any],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract per-sample regressions and improvements from statistical comparison."""
        regressions = []
        improvements = []
        
        for diff in stats.get("per_example_differences", []):
            if diff["a_correct"] and not diff["b_correct"]:
                # Baseline correct, candidate wrong → regression
                regressions.append({
                    "example_id": diff["example_id"],
                    "baseline_score": diff["a_score"],
                    "candidate_score": diff["b_score"],
                })
            elif not diff["a_correct"] and diff["b_correct"]:
                # Baseline wrong, candidate correct → improvement
                improvements.append({
                    "example_id": diff["example_id"],
                    "baseline_score": diff["a_score"],
                    "candidate_score": diff["b_score"],
                })
        
        return regressions, improvements

    @staticmethod
    def _compute_config_diff(
        baseline_manifest: Optional[Dict],
        candidate_manifest: Optional[Dict],
    ) -> Dict[str, Any]:
        """Diff two run manifests, ignoring manifest_hash."""
        if not baseline_manifest or not candidate_manifest:
            return {"error": "missing_manifest"}
        
        changes = {}
        all_keys = set(baseline_manifest.keys()) | set(candidate_manifest.keys())
        all_keys.discard("manifest_hash")
        
        for key in sorted(all_keys):
            baseline_val = baseline_manifest.get(key)
            candidate_val = candidate_manifest.get(key)
            if baseline_val != candidate_val:
                changes[key] = {
                    "baseline": baseline_val,
                    "candidate": candidate_val,
                }
        
        return changes
