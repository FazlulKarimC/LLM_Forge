"""
Test full execution of experiment with all 10 questions.
"""

import asyncio
import sys
import logging
from pathlib import Path
from uuid import UUID

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import async_session_maker
from app.services.experiment_service import ExperimentService
from sqlalchemy import select, func
from app.models.run import Run
from app.models.experiment import Experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_full_execution():
    """Test complete execute() method with all questions."""
    
    logger.info("=" * 80)
    logger.info("TESTING FULL EXECUTE METHOD - ALL 10 QUESTIONS")
    logger.info("=" * 80)
    
    async with async_session_maker() as db:
        try:
            # Get experiment service
            exp_service = ExperimentService(db)
            
            # List experiments
            experiments = await exp_service.list()
            if not experiments.experiments:
                logger.error("No experiments found!")
                return
            
            experiment = experiments.experiments[0]
            experiment_id = experiment.id
            logger.info(f"\nUsing experiment: {experiment.name}")
            logger.info(f"  ID: {experiment_id}")
            logger.info(f"  Current status: {experiment.status}")
            
            # Count runs before
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment_id)
            )
            runs_before = count_result.scalar() or 0
            logger.info(f"  Runs before: {runs_before}")
            
            # Execute experiment (this should process all 10 questions)
            logger.info("\n🚀 Starting experiment execution...")
            await exp_service.execute(experiment_id)
            logger.info("✓ Execute method completed!")
            
            # Wait a moment for database commit
            await asyncio.sleep(0.5)
            
            # Check status after
            await db.rollback()  # Refresh session
            experiment_after = await exp_service.get(experiment_id)
            logger.info(f"\n📊 Results:")
            logger.info(f"  Final status: {experiment_after.status}")
            if experiment_after.error_message:
                logger.error(f"  Error: {experiment_after.error_message}")
            
            # Count runs after
            count_result = await db.execute(
                select(func.count(Run.id)).where(Run.experiment_id == experiment_id)
            )
            runs_after = count_result.scalar() or 0
            runs_created = runs_after - runs_before
            
            logger.info(f"  Runs after: {runs_after}")
            logger.info(f"  New runs created: {runs_created}")
            
            # Verify success
            logger.info("\n" + "=" * 80)
            if runs_created == 10 and experiment_after.status == "completed":
                logger.info("✓✓✓ SUCCESS! All 10 runs logged and experiment completed!")
                logger.info("Phase 2 exit criteria MET!")
            elif runs_created > 0:
                logger.warning(f"⚠️  Partial success: {runs_created}/10 runs created")
            else:
                logger.error("✗✗✗ FAILED! No runs were created")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n✗✗✗ EXCEPTION! ✗✗✗")
            logger.error(f"Error: {str(e)}")
            logger.exception("Full traceback:")
            raise


if __name__ == "__main__":
    asyncio.run(test_full_execution())
