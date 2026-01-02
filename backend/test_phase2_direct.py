"""
Phase 2 - Direct Execution Test
Bypasses background tasks to test core functionality
"""
import asyncio
import httpx
from uuid import UUID

async def main():
    print("=" * 70)
    print("PHASE 2: DIRECT EXECUTION TEST")
    print("=" * 70)
    print("\nThis test bypasses FastAPI background tasks")
    print("and executes the experiment directly.\n")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Create experiment
    print("1. Creating experiment...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        experiment_data = {
            "name": "Phase 2 Direct Test",
            "description": "Direct execution for Phase 2 verification",
            "config": {
                "model_name": "microsoft/phi-2",
                "reasoning_method": "naive",
                "dataset_name": "triviaqa",
                "generation_params": {
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "seed": 42
            }
        }
        
        response = await client.post(f"{base_url}/experiments/", json=experiment_data)
        if response.status_code != 201:
            print(f"Failed: {response.status_code}")
            return
        
        experiment = response.json()
        experiment_id = UUID(experiment["id"])
        print(f"Created: {experiment_id}\n")
    
    # Execute directly
    print("2. Executing experiment directly...")
    from app.core.database import async_session_maker
    from app.services.experiment_service import ExperimentService
    
    async with async_session_maker() as db:
        service = ExperimentService(db)
        try:
            await service.execute(experiment_id)
            await db.commit()
            print("Execution completed!\n")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Verify results
    print("3. Verifying results...")
    from app.models.run import Run
    from sqlalchemy import select, func
    
    async with async_session_maker() as db:
        # Count runs for this experiment
        result = await db.execute(
            select(func.count()).select_from(Run).where(Run.experiment_id == experiment_id)
        )
        run_count = result.scalar()
        print(f"Runs logged: {run_count}")
        
        if run_count == 0:
            print("FAILED: No runs logged!")
            return
        
        # Get runs
        result = await db.execute(
            select(Run).where(Run.experiment_id == experiment_id).limit(3)
        )
        runs = result.scalars().all()
        
        print("\nSample runs:")
        for i, run in enumerate(runs, 1):
            print(f"  {i}. Question ID: {run.example_id}")
            print(f"     Answer: {run.output_text[:50]}...")
            print(f"     Tokens: {run.tokens_input} -> {run.tokens_output}")
            print(f"     Latency: {run.latency_ms:.0f}ms")
            print(f"     Correct: {run.is_correct}")
    
    # Check exit criteria
    print("\n" + "=" * 70)
    print("PHASE 2 EXIT CRITERIA CHECK")
    print("=" * 70)
    
    async with async_session_maker() as db:
        result = await db.execute(
            select(Run).where(Run.experiment_id == experiment_id)
        )
        runs = result.scalars().all()
        
        criteria_passed = 0
        criteria_total = 4
        
        # 1. Run 10 consecutive inferences
        if len(runs) == 10:
            print("PASS [1/4] 10 consecutive inferences without crashes")
            criteria_passed += 1
        else:
            print(f"FAIL [1/4] Expected 10 runs, got {len(runs)}")
        
        # 2. Token counts populated
        token_counts_valid = all(r.tokens_input > 0 and r.tokens_output > 0 for r in runs)
        if token_counts_valid:
            print("PASS [2/4] Token counts populated (input + output)")
            criteria_passed += 1
        else:
            print("FAIL [2/4] Some token counts are 0")
        
        # 3. Latency reasonable
        avg_latency = sum(r.latency_ms for r in runs) / len(runs) if runs else 0
        if avg_latency < 5000:
            print(f"PASS [3/4] Latency reasonable ({avg_latency:.0f}ms avg < 5000ms)")
            criteria_passed += 1
        else:
            print(f"FAIL [3/4] Latency too high ({avg_latency:.0f}ms)")
        
        # 4. All required fields populated
        all_fields_valid = all(
            r.input_text and r.output_text and
            r.tokens_input is not None and r.tokens_output is not None and
            r.latency_ms is not None
            for r in runs
        )
        if all_fields_valid:
            print("PASS [4/4] All non-null required fields populated")
            criteria_passed += 1
        else:
            print("FAIL [4/4] Some required fields are null")
        
        print("\n" + "=" * 70)
        print(f"RESULT: {criteria_passed}/{criteria_total} criteria passed")
        print("=" * 70)
        
        if criteria_passed == criteria_total:
            print("\nSUCCESS! Phase 2 core functionality works!")
            print("Note: Background task execution needs investigation,")
            print("but the core execute() method works perfectly.")
        else:
            print(f"\n{criteria_total - criteria_passed} criteria failed")

if __name__ == "__main__":
    asyncio.run(main())
