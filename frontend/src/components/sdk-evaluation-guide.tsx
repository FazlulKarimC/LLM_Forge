import Link from "next/link";

export const localExample = `from llmforge import LLMForge

cases = [
    {"id": "greeting", "inputs": {"query": "Hello"},
     "expected_output": "Hello"},
]

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo", version=1)
    report = forge.evaluate_local(
        prompt, cases,
        task=lambda compiled: compiled,  # Echo demo; replace with your model
        evaluators={
            "exact_match": lambda output, case:
                output == case["expected_output"]
        },
        scoring_version="exact-match-v1",
        dataset_name="Local greetings",
        model="local-echo-demo",
        upload=True,
        report_path="artifacts/evaluation.json",
    )
    print(report.id, report.pass_rate)
`;

export function SdkEvaluationGuide() {
  return <section className="panel p-5 space-y-4" aria-label="Local SDK evaluation">
    <div className="space-y-2">
      <h2 className="text-lg font-semibold">Evaluate locally, track results here</h2>
      <p className="text-sm text-(--muted-foreground)">Keep test cases, reference answers, model calls and full reports in your application or CI. LLMForge stores the pinned prompt version, dataset and scoring fingerprints, counts and optional aggregate metrics.</p>
    </div>
    <ol className="list-decimal pl-5 space-y-2 text-sm">
      <li>Save a prompt such as Echo with the template <code>{"{{query}}"}</code>. Pin its saved version.</li>
      <li>Install the Python SDK from this repository and configure LLMFORGE_URL and an evaluation-enabled project key from Settings.</li>
      <li>Run local task and evaluator callbacks. Summary upload creates a run here without creating a hosted dataset.</li>
    </ol>
    <pre className="code-panel overflow-auto text-xs" tabIndex={0} aria-label="Local SDK example"><code>{localExample}</code></pre>
    <p className="field-help">This example echoes the prompt without a model call. Replace the task with your application or provider client for live evaluation. Use upload=False to keep the report entirely local. Change scoring_version when evaluator code or settings change. Provider keys stay in your process.</p>
    <div className="flex flex-wrap gap-2">
      <Link className="btn-secondary" href="/prompts/new">Create prompt</Link>
      <Link className="btn-secondary" href="/settings">Project API keys</Link>
    </div>
  </section>;
}
