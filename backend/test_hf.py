import asyncio
from app.services.inference.hf_api_engine import HFAPIEngine
from app.services.inference.base import GenerationConfig
from dotenv import load_dotenv

load_dotenv("c:/Users/FAZLUL/Desktop/MainProject/LlmForge/backend/.env")

engine = HFAPIEngine(model_name="Qwen/Qwen2.5-3B-Instruct")
config = GenerationConfig(max_tokens=150, temperature=0.1, top_p=0.9)

try:
    print("Generating...")
    result = engine.generate("Hello world", config)
    print("Success:", result.text)
except Exception as e:
    print("Failed with:", type(e))
    # We want to print the deep exception
    import traceback
    traceback.print_exc()
