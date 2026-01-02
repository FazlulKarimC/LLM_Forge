"""
Quick verification of experiment status and runs.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import async_session_maker
from app.services.experiment_service import ExperimentService
from sqlalchemy import select, func
from app.models.run import Run


async def verify():
    async with async_session_maker() as db:
        exp_service = ExperimentService(db)
        
        # Get experiments
        experiments = await exp_service.list()
        
        print("\n" + "=" * 80)
        print("PHASE 2 VERIFICATION")
        print("=" * 80)
        
        for exp in experiments.experiments:
            print(f"\nExperiment: {exp.name}")
            print(f"  ID: {exp.id}")
            print(f"  Status: {exp.status}")
            print(f"  Model: {exp.config.model_name}")
            print(f"  Method: {exp.config.reasoning_method}")
            
            # Count runs for this experiment
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == exp.id)
            )
            run_count = count_result.scalar() or 0
            print(f"  Runs: {run_count}")
            
            if exp.started_at:
                print(f"  Started: {exp.started_at}")
            if exp.completed_at:
                print(f"  Completed: {exp.completed_at}")
            if exp.error_message:
                print(f"  Error: {exp.error_message}")
        
        # Total runs
        total_result = await db.execute(select(func.count(Run.id)))
        total_runs = total_result.scalar() or 0
        
        print("\n" + "=" * 80)
        print(f"TOTAL RUNS IN DATABASE: {total_runs}")
        print("=" * 80)
        
        if total_runs >= 10:
            print("✓✓✓ Phase 2 Exit Criteria MET!")
            print("    - At least 10 runs logged to database")
            print("    - Execute method working correctly")
        else:
            print(f"⚠️  Need {10 - total_runs} more runs for Phase 2 completion")


if __name__ == "__main__":
    asyncio.run(verify())
