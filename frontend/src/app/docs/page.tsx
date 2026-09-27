import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "SDK & demo guide", description: "Get started with LLMForge prompt releases, dataset evaluations and Python CI gates." };

const install = `# From the LLMForge repository root (Python 3.10+)
python -m pip install -e ./sdk/python

# Set environment variables in your shell / CI secret store:
# LLMFORGE_URL=http://localhost:8000/api/v1
# LLMFORGE_API_KEY=<project key from Settings>`;
const fetchExample = `from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo", label="production")
    print(prompt.compile(query="Hello"))
    # Fetch a pinned snapshot with version=1 instead of label=...`;
const gateExample = `python -m llmforge evaluate \\
  --prompt "Echo demo" --version 1 \\
  --dataset Greetings --dataset-version 1 \\
  --min-pass-rate 1 --output artifacts/evaluation.json

python -m llmforge check artifacts/evaluation.json`;

function Code({ children }: { children: string }) { return <pre className="mt-4 overflow-x-auto rounded-xl border border-(--border) bg-(--surface-2) p-4 text-xs leading-6"><code>{children}</code></pre>; }

export default function DocsPage() {
  return <div className="page-width max-w-5xl px-4 py-10 sm:px-6">
    <nav className="flex flex-wrap justify-between gap-3"><Link href="/" className="font-semibold">LLMForge</Link><div className="flex gap-4"><Link href="/prompts" className="underline">Workspace</Link><a href="https://github.com/FazlulKarimC/LLM_Forge" className="underline">Repository</a></div></nav>
    <header className="my-10"><p className="page-eyebrow">Developer guide</p><h1 className="mt-4 text-4xl font-semibold tracking-tight">A tested prompt, from draft to application.</h1><p className="mt-4 max-w-3xl text-(--muted-foreground) leading-7">Keep prompt releases and test cases in one project. Inspect regressions in the workspace, fetch a released prompt in Python, and use evaluation results to gate CI.</p></header>
    <nav aria-label="Guide sections" className="flex flex-wrap gap-4 text-sm underline"><a href="#demo">Five-minute demo</a><a href="#sdk">Install & fetch</a><a href="#ci">CI gate</a><a href="#external">External results</a><a href="#limits">Limits & troubleshooting</a></nav>
    <section id="demo" className="mt-10 panel p-6 scroll-mt-6"><h2 className="text-2xl font-semibold">Five-minute demo without model credits</h2><ol className="mt-4 list-decimal pl-5 space-y-3 text-sm leading-7">
      <li>Sign in, select a project, and create a prompt named <strong>Echo demo</strong> with Mustache text <code>{"{{query}}"}</code>. Save v1.</li>
      <li>Create a dataset named <strong>Greetings</strong>. Keep the two sample cases and save revision 1.</li>
      <li>Evaluate prompt v1 on dataset revision 1 using <strong>Demo</strong> and <strong>Exact match</strong>. Both cases pass.</li>
      <li>Save prompt v2 with text <code>{"Reply: {{query}}"}</code>. Run it on the same revision; both checks fail. Compare the two runs in the results grid.</li>
      <li>Promote v1 to production. Fetch it with the SDK below. Enable evaluation access on a separate CI key to try the gate.</li>
    </ol><p className="mt-4 text-sm text-(--muted-foreground)">Demo mode echoes the compiled prompt; it does not call a generation model or measure model quality. For a live use case, test a support-ticket classifier using JSON labels, references, and JSON-path checks.</p></section>
    <section id="sdk" className="mt-8 panel p-6 scroll-mt-6"><h2 className="text-2xl font-semibold">Install and fetch a release</h2><p className="mt-4 text-sm leading-7">The Python package is included in this repository and is not published to PyPI. Create a project API key in Settings. Default keys read prompts; evaluation access is optional and must be enabled when creating the key.</p><Code>{install}</Code><Code>{fetchExample}</Code><p className="mt-4 text-sm text-(--muted-foreground)">Default fetch selects production. Staging and explicit versions are supported. The SDK revalidates its memory cache with the server on every fetch. Missing releases, revoked keys, and network failures raise errors without silently selecting another version.</p></section>
    <section id="ci" className="mt-8 panel p-6 scroll-mt-6"><h2 className="text-2xl font-semibold">Gate changes in CI</h2><p className="mt-4 text-sm leading-7">Use a project key with evaluations:write. Pin a prompt version and dataset revision for reproducible results, or select staging to evaluate the current candidate. CI needs a reachable backend with a persistent process.</p><Code>{gateExample}</Code><p className="mt-4 text-sm leading-7"><strong>Exit 0:</strong> passed. <strong>Exit 1:</strong> quality regression or case errors. <strong>Exit 2:</strong> configuration, transport, timeout, or incomplete-run failure. Reports can be uploaded as CI artifacts; they include dataset inputs and outputs.</p><p className="mt-4 text-sm">For live generation set LLMFORGE_PROVIDER_API_KEY and select --provider / --model. Live calls use your provider credits. The CLI attempts cancellation on timeout.</p><p className="mt-4 text-sm">Copy docs/examples/evaluation-gate.yml from the repository for a complete GitHub Actions workflow. Set the backend URL as a repository variable and the evaluation key as a secret.</p></section>
    <section id="external" className="mt-8 panel p-6 scroll-mt-6"><h2 className="text-2xl font-semibold">Evaluate in your own application</h2><p className="mt-4 text-sm leading-7">Use get_dataset to load fixed cases, run your application or model locally, and submit one output and check set per case with submit_evaluation. Include named numeric metrics if useful. External results appear in the same grid, explicitly marked as application-supplied checks. They are self-reported and do not trigger provider calls.</p><Code>{`run_id = forge.submit_evaluation(
    prompt.id, dataset["revision"]["id"],
    model="your-application", results=results,
    metrics={"accuracy": 0.95},
)
result = forge.get_evaluation(run_id)
assert result.meets_threshold(0.9)`}</Code><p className="mt-4 text-sm text-(--muted-foreground)">Each result includes case_index, output, and checks such as {"{\"name\": \"correctness\", \"passed\": true}"}, or an error. See the complete example in sdk/python/examples/submit_results.py.</p></section>
    <section id="limits" className="mt-8 panel p-6 scroll-mt-6"><h2 className="text-2xl font-semibold">Limits and troubleshooting</h2><ul className="mt-4 space-y-3 list-disc pl-5 text-sm leading-7">
      <li><strong>401:</strong> check your project key and whether it was revoked. Clerk browser tokens and project API keys serve different endpoints.</li>
      <li><strong>403:</strong> create a new key with evaluation access enabled. Existing read-only keys remain read-only.</li>
      <li><strong>404:</strong> verify the project, exact name, release label and version. Promote a prompt before fetching production.</li>
      <li><strong>422:</strong> check variable names, references, assertions, and case limits. Exact match requires a reference for every case.</li>
      <li>Dataset revisions allow 100 cases / 1 MB. Live runs allow 50 cases, or 20 with judging. Two runs can be active per project.</li>
      <li>Provider/judge calls have a 30-second limit and no automatic retry. Restarting the backend loses in-memory credentials; abandoned runs fail after 120 seconds without progress.</li>
      <li>LLM judge scores are model opinions. Prefer deterministic checks for strict JSON or output contracts, and inspect judge reasons.</li>
    </ul><p className="mt-4 text-sm text-(--muted-foreground)">The packaged SDK and guide cover this personal project&apos;s current workflow. Legacy reasoning benchmarks remain available under Experiments.</p></section>
  </div>;
}
