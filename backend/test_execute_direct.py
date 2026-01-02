"""
Direct test of execute method to find the error
"""
import asyncio
from uuid import UUID

async def test_execute_directly():
    print("=" * 60)
    print("DIRECT EXECUTE METHOD TEST")
    print("=" * 60)
    
    # First create an experiment via API
    import httpx
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create experiment
        experiment_data = {
            "name": "Direct Test",
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
        
        response = await client.post(f"{base_url}/experiments/", json=experiment_data)
        if response.status_code != 201:
            print(f"❌ Failed to create experiment")
            return
        
        experiment = response.json()
        experiment_id = UUID(experiment["id"])
        print(f"✅ Created experiment: {experiment_id}\n")
    
    # Now call execute directly
    print("Calling execute method directly...")
    print("-" * 60)
    
    from app.core.database import SessionLocal
    from app.services.experiment_service import ExperimentService
    
    async with SessionLocal() as db:
        service = ExperimentService(db)
        try:
            await service.execute(experiment_id)
            print("\n✅ Execute completed successfully!")
        except Exception as e:
            print(f"\n❌ Execute failed with error:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_execute_directly())
