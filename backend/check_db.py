"""Quick database check script"""
import asyncio
from sqlalchemy import select, desc
import sys
sys.path.insert(0, ".")

async def check_db():
    from app.core.database import SessionLocal
    from app.models.experiment import Experiment
    from app.models.run import Run
    
    async with SessionLocal() as db:
        # Get last experiment
        result = await db.execute(
            select(Experiment).order_by(desc(Experiment.created_at)).limit(1)
        )
        exp = result.scalar_one_or_none()
        
        if not exp:
            print("No experiments found")
            return
        
        print(f"Last Experiment:")
        print(f"  ID: {exp.id}")
        print(f"  Name: {exp.name}")
        print(f"  Status: {exp.status}")
        print(f"  Method: {exp.config.get('reasoning_method')}")
        
        # Get runs
        result2 = await db.execute(
            select(Run).where(Run.experiment_id == exp.id)
        )
        runs = result2.scalars().all()
        print(f"  Runs logged: {len(runs)}")
        
        if runs:
            for i, run in enumerate(runs[:3], 1):
                print(f"\n  Run {i}:")
                print(f"    Example: {run.example_id}")
                print(f"    Output: {run.output_text[:40]}...")
                print(f"    Tokens: {run.tokens_input} → {run.tokens_output}")
                print(f"    Latency: {run.latency_ms:.0f}ms")

if __name__ == "__main__":
    asyncio.run(check_db())
