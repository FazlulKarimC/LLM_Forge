import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings
from huggingface_hub import InferenceClient

async def test_models():
    models_to_test = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
        "deepseek-ai/DeepSeek-V2.5"
    ]
    
    api_key = settings.HF_TOKEN or os.getenv("HF_TOKEN")
    # No provider specified, should use default HF routing
    client = InferenceClient(api_key=api_key, timeout=20)
    
    for model in models_to_test:
        print(f"Testing default routing for {model}...")
        try:
            res = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Say hi!"}],
                max_tokens=10
            )
            print(f"SUCCESS {model}: {res.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"FAILED {model}: {e}")

if __name__ == "__main__":
    asyncio.run(test_models())
