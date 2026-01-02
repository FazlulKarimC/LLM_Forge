"""
Debug script to test execute() method step by step.
Identifies exact failure point with comprehensive logging.
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import async_session_maker
from app.services.experiment_service import ExperimentService
from app.services.inference.mock_engine import MockEngine
from app.services.inference.base import GenerationConfig
from app.services.inference.prompting import NaivePromptTemplate
from app.services.run_service import RunService
from sqlalchemy import select, func
from app.models.run import Run

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_step_by_step():
    """Test each step of execute() independently."""
    
    logger.info("=" * 80)
    logger.info("PHASE 2 DEBUG - EXECUTE METHOD STEP-BY-STEP TEST")
    logger.info("=" * 80)
    
    async with async_session_maker() as db:
        try:
            # Step 1: Get experiment service
            logger.info("\n[STEP 1] Initializing ExperimentService...")
            exp_service = ExperimentService(db)
            logger.info("✓ ExperimentService initialized")
            
            # Step 2: List experiments
            logger.info("\n[STEP 2] Listing experiments...")
            experiments = await exp_service.list()
            if not experiments.experiments:
                logger.error("✗ No experiments found in database!")
                return
            
            experiment = experiments.experiments[0]
            logger.info(f"✓ Found experiment: {experiment.name} (ID: {experiment.id})")
            logger.info(f"  Status: {experiment.status}")
            logger.info(f"  Config: {experiment.config}")
            
            # Step 3: Initialize MockEngine
            logger.info("\n[STEP 3] Initializing MockEngine...")
            engine = MockEngine()
            logger.info("✓ MockEngine created")
            
            # Step 4: Load model
            logger.info("\n[STEP 4] Loading model...")
            model_name = experiment.config.model_name
            logger.info(f"  Model name: {model_name}")
            engine.load_model(model_name)
            logger.info(f"✓ Model loaded (is_loaded: {engine.is_loaded})")
            
            # Step 5: Load sample questions
            logger.info("\n[STEP 5] Loading sample_questions.json...")
            questions_path = Path(__file__).parent.parent / "configs" / "sample_questions.json"
            logger.info(f"  Path: {questions_path}")
            logger.info(f"  Exists: {questions_path.exists()}")
            
            if not questions_path.exists():
                logger.error(f"✗ File not found: {questions_path}")
                return
            
            with open(questions_path, "r") as f:
                questions = json.load(f)
            logger.info(f"✓ Loaded {len(questions)} questions")
            logger.info(f"  First question: {questions[0]['question'][:50]}...")
            
            # Step 6: Create generation config
            logger.info("\n[STEP 6] Creating GenerationConfig...")
            gen_config = GenerationConfig(
                max_tokens=experiment.config.hyperparameters.max_tokens,
                temperature=experiment.config.hyperparameters.temperature,
                top_p=experiment.config.hyperparameters.top_p,
            )
            logger.info(f"✓ Config: max_tokens={gen_config.max_tokens}, "
                       f"temp={gen_config.temperature}, top_p={gen_config.top_p}")
            
            # Step 7: Test single generation
            logger.info("\n[STEP 7] Testing single generation...")
            test_question = questions[0]
            prompt = NaivePromptTemplate.format(test_question["question"])
            logger.info(f"  Prompt length: {len(prompt)} chars")
            
            result = engine.generate(prompt, gen_config)
            logger.info(f"✓ Generation successful!")
            logger.info(f"  Output: {result.text[:100]}...")
            logger.info(f"  Tokens: {result.tokens_input} in, {result.tokens_output} out")
            logger.info(f"  Latency: {result.latency_ms:.2f}ms")
            
            # Step 8: Test response parsing
            logger.info("\n[STEP 8] Testing response parsing...")
            parsed_answer = NaivePromptTemplate.parse_response(result.text)
            logger.info(f"✓ Parsed answer: {parsed_answer[:50]}...")
            
            # Step 9: Initialize RunService
            logger.info("\n[STEP 9] Initializing RunService...")
            run_service = RunService(db)
            logger.info("✓ RunService initialized")
            
            # Step 10: Count existing runs
            logger.info("\n[STEP 10] Checking existing runs...")
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment.id)
            )
            existing_runs = count_result.scalar() or 0
            logger.info(f"  Existing runs for this experiment: {existing_runs}")
            
            # Step 11: Create a single test run
            logger.info("\n[STEP 11] Creating test run in database...")
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
            logger.info(f"✓ Run created (ID: {run.id})")
            logger.info(f"  Is correct: {is_correct}")
            logger.info(f"  Score: {run.score}")
            
            # Step 12: Commit to database
            logger.info("\n[STEP 12] Committing to database...")
            await db.commit()
            logger.info("✓ Changes committed")
            
            # Step 13: Verify run was saved
            logger.info("\n[STEP 13] Verifying run was saved...")
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment.id)
            )
            new_count = count_result.scalar() or 0
            logger.info(f"  Total runs now: {new_count}")
            
            if new_count > existing_runs:
                logger.info("✓✓✓ SUCCESS! Run was saved to database!")
            else:
                logger.error("✗✗✗ FAILED! Run was NOT saved to database!")
            
            # Step 14: Cleanup
            logger.info("\n[STEP 14] Cleanup...")
            engine.unload_model()
            logger.info("✓ Model unloaded")
            
            logger.info("\n" + "=" * 80)
            logger.info("ALL STEPS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n✗✗✗ EXCEPTION CAUGHT! ✗✗✗")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.exception("Full traceback:")
            raise


if __name__ == "__main__":
    asyncio.run(test_step_by_step())
