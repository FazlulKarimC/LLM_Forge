"""Quick database check - ASCII only output"""
import asyncio
import sys
sys.path.insert(0, ".")

async def check():
    from app.core.database import async_session_maker
    from app.models.run import Run
    from app.models.experiment import Experiment
    from sqlalchemy import select, func, desc
    
    async with async_session_maker() as db:
        # Count runs
        result = await db.execute(select(func.count()).select_from(Run))
        run_count = result.scalar()
        print(f"Total runs in database: {run_count}")
        
        # Get latest experiment
        result = await db.execute(
            select(Experiment).order_by(desc(Experiment.created_at)).limit(1)
        )
        exp = result.scalar_one_or_none()
        if exp:
            print(f"\nLatest experiment:")
            print(f"  ID: {exp.id}")
            print(f"  Name: {exp.name}")
            print(f"  Status: {exp.status}")
            
            # Get runs for this experiment
            result = await db.execute(
                select(Run).where(Run.experiment_id == exp.id)
            )
            runs = result.scalars().all()
            print(f"  Runs for this experiment: {len(runs)}")
            
            if runs:
                print("\nFirst 3 runs:")
                for i, run in enumerate(runs[:3], 1):
                    print(f"  {i}. {run.example_id}: {run.output_text[:40]}...")

if __name__ == "__main__":
    asyncio.run(check())
