"""Evaluate local cases and upload a content-free summary; no model call."""
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo", version=1)
    report = forge.evaluate_local(
        prompt, [{"id": "hello", "inputs": {"query": "Hello"}, "expected_output": "Hello"}],
        task=lambda compiled: compiled,
        evaluators={"exact_match": lambda output, case: output == case["expected_output"]},
        scoring_version="exact-v1", dataset_name="Local greetings", model="local-echo-demo",
    )
    print(f"Submitted {report.id}: {report.pass_rate:.0%} passed")
