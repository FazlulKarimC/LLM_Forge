"""Run after publishing Echo demo to production and setting SDK environment."""
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo")
    print(f"Using prompt v{prompt.version}")
    print(prompt.compile(query="Hello"))
