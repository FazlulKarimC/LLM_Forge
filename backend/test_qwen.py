import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.inference.hf_api_engine import HFAPIEngine
from app.services.inference.base import GenerationConfig

async def test_models():
    models_to_test = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ]
    
    try:
        engine = HFAPIEngine()
        config = GenerationConfig(max_tokens=20)
        
        for model in models_to_test:
            print(f"Testing {model}...")
            try:
                engine.load_model(model)
                res = engine.generate("Hello, write a python hello world format", config)
                print(f"SUCCESS {model}: {res.text.strip()}")
            except Exception as e:
                print(f"FAILED {model}: {e}")
    except Exception as e:
        print(f"Engine init failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_models())
