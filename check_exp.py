import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from app.core.database import async_session_maker
from app.models.experiment import Experiment
from app.services.experiment_service import ExperimentService
from sqlalchemy import select

async def main():
    async with async_session_maker() as session:
        # Get the latest experiment
        result = await session.execute(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(1)
        )
        exp = result.scalar_one_or_none()
        
        if exp:
            print(f"Executing latest experiment: {exp.id} | Name: {exp.name}")
            svc = ExperimentService(session)
            
            try:
                await svc.execute(exp.id)
                print("Execution successful!")
            except Exception as e:
                import traceback
                print("\nException caught during execution:")
                traceback.print_exc()
        else:
            print("No experiments found.")

if __name__ == "__main__":
    asyncio.run(main())
