import asyncio
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.inference.hf_api_engine import HFAPIEngine
from app.services.inference.base import GenerationConfig

async def test_models():
    models_to_test = [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
        "microsoft/Phi-3-mini-4k-instruct",
        "deepseek-ai/DeepSeek-V2.5"
    ]
    
    try:
        engine = HFAPIEngine()
        config = GenerationConfig(max_tokens=20)
        
        for model in models_to_test:
            print(f"Testing {model}...")
            try:
                engine.load_model(model)
                res = engine.generate("Hello, say hi!", config)
                print(f"SUCCESS {model}: {res.text.strip()}")
            except Exception as e:
                print(f"FAILED {model}: {e}")
    except Exception as e:
        print(f"Engine init failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_models())
