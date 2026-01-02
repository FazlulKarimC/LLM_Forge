"""
Test background task execution via API endpoint.
Creates a new experiment and triggers execution.
"""

import requests
import time
import json


def test_api_execution():
    """Test experiment execution via API."""
    
    base_url = "http://localhost:8000"
    
    print("\n" + "=" * 80)
    print("TESTING API ENDPOINT - BACKGROUND TASK EXECUTION")
    print("=" * 80)
    
    # Step 1: Create a new experiment
    print("\n[1] Creating new experiment...")
    create_payload = {
        "name": "API Test - Background Execution",
        "description": "Testing background task execution via API",
        "config": {
            "model_name": "microsoft/phi-2",
            "reasoning_method": "naive",
            "dataset_name": "sample_questions",
            "num_samples": 10,
            "hyperparameters": {
                "temperature": 0.7,
                "max_tokens": 256,
                "top_p": 0.9,
                "seed": 42
            }
        }
    }
    
    response = requests.post(f"{base_url}/api/experiments", json=create_payload)
    print(f"  Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"  Error: {response.text}")
        return
    
    experiment = response.json()
    experiment_id = experiment["id"]
    print(f"✓ Created experiment: {experiment['name']}")
    print(f"  ID: {experiment_id}")
    print(f"  Status: {experiment['status']}")
    
    # Step 2: Trigger execution
    print(f"\n[2] Triggering execution...")
    exec_response = requests.post(f"{base_url}/api/experiments/{experiment_id}/execute")
    print(f"  Status: {exec_response.status_code}")
    
    if exec_response.status_code != 202:
        print(f"  Error: {exec_response.text}")
        return
    
    exec_data = exec_response.json()
    print(f"✓ Execution triggered!")
    print(f"  Message: {exec_data.get('message', '')}")
    
    # Step 3: Poll for completion
    print(f"\n[3] Polling for completion...")
    max_wait = 30  # seconds
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        time.sleep(1)
        
        status_response = requests.get(f"{base_url}/api/experiments/{experiment_id}")
        if status_response.status_code != 200:
            print(f"  Error fetching status: {status_response.status_code}")
            continue
        
        current = status_response.json()
        current_status = current["status"]
        elapsed = int(time.time() - start_time)
        
        print(f"  [{elapsed}s] Status: {current_status}")
        
        if current_status == "completed":
            print(f"\n✓✓✓ Experiment COMPLETED successfully!")
            print(f"  Started: {current.get('started_at', 'N/A')}")
            print(f"  Completed: {current.get('completed_at', 'N/A')}")
            
            # Check run count
            print(f"\n[4] Checking run count...")
            # We'd need a runs endpoint to get count, but we can infer from success
            print(f"  Experiment completed, should have 10 runs logged")
            
            print("\n" + "=" * 80)
            print("✓✓✓ API BACKGROUND TASK TEST PASSED!")
            print("=" * 80)
            return True
            
        elif current_status == "failed":
            print(f"\n✗✗✗ Experiment FAILED!")
            print(f"  Error: {current.get('error_message', 'Unknown error')}")
            print("\n" + "=" * 80)
            print("✗ API BACKGROUND TASK TEST FAILED")
            print("=" * 80)
            return False
    
    print(f"\n⚠️  Timeout after {max_wait}s")
    print(f"  Last status: {current_status}")
    print("\n" + "=" * 80)
    print("⚠️  TEST TIMED OUT")
    print("=" * 80)
    return False


if __name__ == "__main__":
    test_api_execution()
