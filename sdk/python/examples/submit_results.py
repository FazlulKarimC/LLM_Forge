"""Submit a local echo contract test with an evaluation-scoped project key."""
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo", version=1)
    dataset = forge.get_dataset("Greetings", version=1)
    results = []
    for index, case in enumerate(dataset["revision"]["cases"]):
        output = prompt.compile(case["inputs"])
        results.append({"case_index": index, "output": output,
            "checks": [{"name": "exact_match", "passed": output == case["expected_output"]}]})
    run_id = forge.submit_evaluation(prompt.id, dataset["revision"]["id"], model="local-echo-demo", results=results)
    result = forge.get_evaluation(run_id)
    print(f"Submitted {result.id}: {result.pass_rate:.0%} passed")
