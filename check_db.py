import asyncio
import os
import sys

# Add backend to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from app.core.database import async_session_maker
from app.models.experiment import Experiment
from app.models.run import Run
from sqlalchemy import select

async def main():
    async with async_session_maker() as session:
        # Get latest experiments
        result = await session.execute(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(3)
        )
        experiments = result.scalars().all()
        
        for exp in experiments:
            print(f"\nExperiment: {exp.id} | Name: {exp.name} | Status: {exp.status} | Model: {exp.model_name}")
            print(f"Created: {exp.created_at} | Started: {exp.started_at} | Completed: {exp.completed_at}")
            if exp.error_message:
                print(f"Error: {exp.error_message}")
                
            # Count runs
            run_result = await session.execute(
                select(Run).where(Run.experiment_id == exp.id)
            )
            runs = run_result.scalars().all()
            print(f"Total runs logged: {len(runs)}")
            
            if runs:
                print(f"Sample run latency: {runs[0].latency_ms}ms")

if __name__ == "__main__":
    asyncio.run(main())
