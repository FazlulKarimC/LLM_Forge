"""
Comprehensive Phase 2 Diagnostic Script
Tests each component systematically
"""
import asyncio
import sys
sys.path.insert(0, ".")

print("=" * 70)
print("PHASE 2 COMPREHENSIVE DIAGNOSTIC")
print("=" * 70)

async def test_1_check_database():
    """Check if any runs exist in database"""
    print("\n" + "=" * 70)
    print("TEST 1: Database Check")
    print("=" * 70)
    
    from app.core.database import async_session_maker
    from app.models.run import Run
    from app.models.experiment import Experiment
    from sqlalchemy import select, func
    
    async with async_session_maker() as db:
        # Count total experiments
        result = await db.execute(select(func.count()).select_from(Experiment))
        exp_count = result.scalar()
        print(f"✓ Total experiments in database: {exp_count}")
        
        # Count total runs
        result = await db.execute(select(func.count()).select_from(Run))
        run_count = result.scalar()
        print(f"✓ Total runs in database: {run_count}")
        
        if run_count > 0:
            print(f"\n🎉 RUNS EXIST! The background task DID execute!")
            result = await db.execute(select(Run).limit(5))
            runs = result.scalars().all()
            for i, run in enumerate(runs, 1):
                print(f"\n  Run {i}:")
                print(f"    Example: {run.example_id}")
                print(f"    Output: {run.output_text[:60]}...")
                print(f"    Tokens: {run.tokens_input} → {run.tokens_output}")
            return True
        else:
            print("❌ No runs found - background task not executing properly")
            return False

async def test_2_mock_engine():
    """Test MockEngine standalone"""
    print("\n" + "=" * 70)
    print("TEST 2: MockEngine Standalone Test")
    print("=" * 70)
    
    from app.services.inference.mock_engine import MockEngine
    from app.services.inference.base import GenerationConfig
    
    try:
        engine = MockEngine()
        engine.load_model("microsoft/phi-2")
        print("✓ MockEngine loaded")
        
        config = GenerationConfig(max_tokens=256, temperature=0.7, top_p=0.9)
        result = engine.generate("Question: What is 2+2?\nAnswer:", config)
        
        print(f"✓ Generation successful!")
        print(f"  Output: {result.text[:100]}")
        print(f"  Tokens: {result.tokens_input} → {result.tokens_output}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
        
        engine.unload_model()
        print("✓ MockEngine works perfectly!")
        return True
    except Exception as e:
        print(f"❌ MockEngine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_3_sample_questions():
    """Test loading sample questions"""
    print("\n" + "=" * 70)
    print("TEST 3: Sample Questions Loading")
    print("=" * 70)
    
    import json
    from pathlib import Path
    
    # Try different path calculations
    paths_to_try = [
        Path("configs/sample_questions.json"),
        Path("../configs/sample_questions.json"),
        Path(__file__).parent.parent / "configs" / "sample_questions.json",
    ]
    
    for i, path in enumerate(paths_to_try, 1):
        try:
            abs_path = path.resolve()
            print(f"\n  Trying path {i}: {abs_path}")
            if abs_path.exists():
                with open(abs_path, "r") as f:
                    questions = json.load(f)
                print(f"  ✓ SUCCESS! Loaded {len(questions)} questions")
                print(f"  ✓ Correct path: {abs_path}")
                return True
            else:
                print(f"  ✗ File not found")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("❌ Could not load sample questions from any path!")
    return False

async def test_4_direct_execute():
    """Test execute method directly"""
    print("\n" + "=" * 70)
    print("TEST 4: Direct Execute Method Test")
    print("=" * 70)
    
    import httpx
    from uuid import UUID
    from app.core.database import async_session_maker
    from app.services.experiment_service import ExperimentService
    
    # Create experiment via API
    async with httpx.AsyncClient(timeout=30.0) as client:
        experiment_data = {
            "name": "Direct Execute Test",
            "description": "Testing execute directly",
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
        
        print("  Creating experiment via API...")
        response = await client.post("http://localhost:8000/api/v1/experiments/", json=experiment_data)
        if response.status_code != 201:
            print(f"  ❌ Failed to create experiment: {response.status_code}")
            return False
        
        experiment = response.json()
        experiment_id = UUID(experiment["id"])
        print(f"  ✓ Created experiment: {experiment_id}")
    
    # Execute directly
    print("\n  Executing directly (not via background task)...")
    async with async_session_maker() as db:
        service = ExperimentService(db)
        try:
            await service.execute(experiment_id)
            await db.commit()
            print("  ✓ Execute method completed successfully!")
            return True
        except Exception as e:
            print(f"  ❌ Execute failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    results = {}
    
    # Test 1: Database
    results['database'] = await test_1_check_database()
    
    # Test 2: MockEngine
    results['mock_engine'] = await test_2_mock_engine()
    
    # Test 3: Sample questions
    results['sample_questions'] = await test_3_sample_questions()
    
    # Test 4: Direct execute (only if others pass)
    if results['mock_engine'] and results['sample_questions']:
        results['direct_execute'] = await test_4_direct_execute()
    else:
        print("\n⏭️  Skipping direct execute test (dependencies failed)")
        results['direct_execute'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")
    
    print("\n" + "=" * 70)
    if all(results.values()):
        print("🎉 ALL TESTS PASSED! Phase 2 implementation works!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"⚠️  Failed tests: {', '.join(failed)}")
        print("Need to investigate failures above.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
