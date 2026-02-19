#!/usr/bin/env python3
"""
Phase 8 Benchmark Script — run_benchmark.py

Generates a benchmark matrix comparing optimization strategies across
reasoning methods. Uses the mock engine for consistent timing.

Usage:
    python scripts/run_benchmark.py [--samples N] [--output FILE]

The script creates experiments programmatically, runs them, and
collects optimization profiling data into a summary JSON.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.database import async_session_factory
from app.services.experiment_service import ExperimentService
from app.models.experiment import Experiment
from sqlalchemy import select


# =============================================================================
# Benchmark Configuration
# =============================================================================

METHODS = ["naive", "cot"]

OPTIMIZATION_CONFIGS = {
    "baseline": {
        "enable_batching": False,
        "enable_caching": False,
        "enable_profiling": True,
    },
    "batched_4": {
        "enable_batching": True,
        "batch_size": 4,
        "enable_caching": False,
        "enable_profiling": True,
    },
    "batched_8": {
        "enable_batching": True,
        "batch_size": 8,
        "enable_caching": False,
        "enable_profiling": True,
    },
    "cached": {
        "enable_batching": False,
        "enable_caching": True,
        "cache_max_size": 256,
        "enable_profiling": True,
    },
    "batched_8_cached": {
        "enable_batching": True,
        "batch_size": 8,
        "enable_caching": True,
        "cache_max_size": 256,
        "enable_profiling": True,
    },
}


async def run_benchmark(num_samples: int = 10, output_file: str = "benchmark_results.json"):
    """Execute the benchmark matrix."""
    results = []
    total_runs = len(METHODS) * len(OPTIMIZATION_CONFIGS)
    run_idx = 0

    print(f"\n{'='*60}")
    print(f"  Phase 8 Benchmark Matrix")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"  Optimizations: {', '.join(OPTIMIZATION_CONFIGS.keys())}")
    print(f"  Samples per run: {num_samples}")
    print(f"  Total experiments: {total_runs}")
    print(f"{'='*60}\n")

    for method in METHODS:
        for opt_name, opt_config in OPTIMIZATION_CONFIGS.items():
            run_idx += 1
            exp_name = f"bench_{method}_{opt_name}"
            print(f"[{run_idx}/{total_runs}] {exp_name}...", end=" ", flush=True)

            config = {
                "model_name": "mock-model",
                "reasoning_method": method,
                "dataset_name": "sample",
                "hyperparameters": {
                    "temperature": 0.1,
                    "max_tokens": 150,
                },
                "num_samples": num_samples,
                "optimization": opt_config,
            }

            # Create experiment via service
            async with async_session_factory() as db:
                service = ExperimentService(db)
                experiment = await service.create(
                    name=exp_name,
                    description=f"Benchmark: {method} + {opt_name}",
                    config=config,
                )
                exp_id = experiment.id
                await db.commit()

            # Run experiment
            start = time.time()
            async with async_session_factory() as db:
                service = ExperimentService(db)
                try:
                    await service.execute(exp_id)
                    await db.commit()
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
            elapsed = time.time() - start

            # Collect results
            async with async_session_factory() as db:
                from app.models.result import Result
                query = select(Result).where(Result.experiment_id == exp_id)
                res = await db.execute(query)
                db_result = res.scalar_one_or_none()

            opt_data = {}
            if db_result and db_result.raw_metrics:
                opt_data = db_result.raw_metrics.get("optimization", {})

            entry = {
                "experiment_id": str(exp_id),
                "name": exp_name,
                "method": method,
                "optimization": opt_name,
                "config": opt_config,
                "wall_time_s": round(elapsed, 3),
                "profiling": opt_data.get("profiling_summary", {}),
                "cache_stats": opt_data.get("cache_stats", {}),
                "batch_stats": opt_data.get("batch_stats", {}),
                "throughput": db_result.throughput if db_result else None,
                "accuracy_exact": db_result.accuracy_exact if db_result else None,
            }
            results.append(entry)
            print(f"done ({elapsed:.2f}s, accuracy={entry['accuracy_exact']})")

    # Write output
    output = {
        "benchmark": "Phase 8 Optimization Matrix",
        "timestamp": datetime.now().isoformat(),
        "num_samples": num_samples,
        "results": results,
    }

    output_path = Path(output_file)
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n✅ Results written to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<8} {'Optimization':<18} {'Wall Time':>10} {'Throughput':>12} {'Accuracy':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['method']:<8} {r['optimization']:<18} "
            f"{r['wall_time_s']:>9.2f}s "
            f"{(r['throughput'] or 0):>11.1f}/s "
            f"{((r['accuracy_exact'] or 0) * 100):>9.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Benchmark Matrix")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per experiment")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    asyncio.run(run_benchmark(num_samples=args.samples, output_file=args.output))


if __name__ == "__main__":
    main()
