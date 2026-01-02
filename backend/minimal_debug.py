"""
Minimal debug script - avoids Unicode issues and tests core execution.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import async_session_maker
from sqlalchemy import select, text

async def check_database_state():
    """Check current state of experiments and runs."""
    print("\n" + "="*60)
    print("DATABASE STATE CHECK")
    print("="*60)
    
    async with async_session_maker() as db:
        # Check experiments
        result = await db.execute(text('SELECT id, name, status, error_message FROM experiments ORDER BY created_at DESC LIMIT 5'))
        rows = result.fetchall()
        print("\n--- EXPERIMENTS (Latest 5) ---")
        for row in rows:
            print(f"  ID: {row[0]}")
            print(f"     Name: {row[1]}")
            print(f"     Status: {row[2]}")
            if row[3]:
                print(f"     Error: {row[3]}")
            print()
        
        # Check runs count
        result = await db.execute(text('SELECT COUNT(*) FROM runs'))
        count = result.scalar()
        print(f"--- TOTAL RUNS: {count} ---")
        
        return rows


async def test_execution_steps():
    """Test each step of execution manually."""
    print("\n" + "="*60)
    print("TESTING EXECUTION STEPS")
    print("="*60)
    
    from app.services.experiment_service import ExperimentService
    from app.services.inference.mock_engine import MockEngine
    from app.services.inference.base import GenerationConfig
    from app.services.inference.prompting import NaivePromptTemplate
    from app.services.run_service import RunService
    from app.models.run import Run
    from sqlalchemy import func
    
    async with async_session_maker() as db:
        try:
            # Step 1: Get experiment
            print("\n[STEP 1] Getting experiment from database...")
            exp_service = ExperimentService(db)
            experiments = await exp_service.list()
            
            if not experiments.experiments:
                print("  ERROR: No experiments found!")
                return
            
            experiment = experiments.experiments[0]
            print(f"  OK - Found: {experiment.name} (ID: {experiment.id})")
            print(f"       Status: {experiment.status}")
            
            # Step 2: Initialize MockEngine
            print("\n[STEP 2] Initializing MockEngine...")
            engine = MockEngine()
            engine.load_model(experiment.config.model_name)
            print(f"  OK - Model loaded: {experiment.config.model_name}")
            
            # Step 3: Load sample questions
            print("\n[STEP 3] Loading sample questions...")
            questions_path = Path(__file__).parent.parent / "configs" / "sample_questions.json"
            print(f"  Path: {questions_path}")
            print(f"  Exists: {questions_path.exists()}")
            
            if not questions_path.exists():
                print(f"  ERROR: File not found!")
                return
            
            with open(questions_path, "r") as f:
                questions = json.load(f)
            print(f"  OK - Loaded {len(questions)} questions")
            
            # Step 4: Create generation config
            print("\n[STEP 4] Creating GenerationConfig...")
            gen_config = GenerationConfig(
                max_tokens=experiment.config.hyperparameters.max_tokens,
                temperature=experiment.config.hyperparameters.temperature,
                top_p=experiment.config.hyperparameters.top_p,
            )
            print(f"  OK - max_tokens={gen_config.max_tokens}, temp={gen_config.temperature}")
            
            # Step 5: Test single generation
            print("\n[STEP 5] Testing single generation...")
            test_question = questions[0]
            prompt = NaivePromptTemplate.format(test_question["question"])
            print(f"  Prompt length: {len(prompt)} chars")
            
            result = engine.generate(prompt, gen_config)
            print(f"  OK - Generated response")
            print(f"       Output: {result.text[:100]}...")
            print(f"       Tokens: {result.tokens_input} in, {result.tokens_output} out")
            
            # Step 6: Parse response
            print("\n[STEP 6] Parsing response...")
            parsed_answer = NaivePromptTemplate.parse_response(result.text)
            print(f"  OK - Parsed: {parsed_answer[:50]}...")
            
            # Step 7: Initialize RunService
            print("\n[STEP 7] Initializing RunService...")
            run_service = RunService(db)
            print("  OK - RunService ready")
            
            # Step 8: Count existing runs
            print("\n[STEP 8] Counting existing runs...")
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment.id)
            )
            existing_runs = count_result.scalar() or 0
            print(f"  Existing runs for this experiment: {existing_runs}")
            
            # Step 9: Create a test run
            print("\n[STEP 9] Creating test run...")
            is_correct = parsed_answer.lower().strip() == test_question["answer"].lower().strip()
            
            run = await run_service.create_run(
                experiment_id=experiment.id,
                example_id=test_question["id"],
                input_text=prompt,
                output_text=parsed_answer,
                expected_output=test_question["answer"],
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                tokens_input=result.tokens_input,
                tokens_output=result.tokens_output,
                latency_ms=result.latency_ms,
                gpu_memory_mb=result.gpu_memory_mb,
            )
            print(f"  OK - Run created (ID: {run.id})")
            
            # Step 10: Commit
            print("\n[STEP 10] Committing to database...")
            await db.commit()
            print("  OK - Committed")
            
            # Step 11: Verify
            print("\n[STEP 11] Verifying run was saved...")
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment.id)
            )
            new_count = count_result.scalar() or 0
            print(f"  Total runs now: {new_count}")
            
            if new_count > existing_runs:
                print("  SUCCESS - Run was saved!")
            else:
                print("  FAILED - Run was NOT saved!")
            
            # Step 12: Cleanup
            print("\n[STEP 12] Cleanup...")
            engine.unload_model()
            print("  OK - Model unloaded")
            
            print("\n" + "="*60)
            print("ALL STEPS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"\n  EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    print("Starting minimal debug test...")
    
    async def main():
        await check_database_state()
        await test_execution_steps()
    
    asyncio.run(main())

