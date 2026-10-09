import type { Metadata } from "next";
import { PublicHeader } from "@/components/public-header";
import { PublicFooter } from "@/components/public-footer";
import { ServerCapabilities } from "@/components/server-capabilities";
import { DeploymentInfo } from "@/components/deployment-info";

export const metadata: Metadata = {
  title: "SDK & demo guide",
  description:
    "Get started with LLMForge prompt releases, dataset evaluations and Python CI gates.",
};

const install = `# From the LLMForge repository root (Python 3.10+)
python -m pip install -e ./sdk/python

# Set environment variables in your shell / CI secret store:
# Hosted LLMForge (installing the SDK locally does not require a local server):
# LLMFORGE_URL=https://faz-ai-llm-forge.hf.space/api/v1
# Local backend, only if you are running it yourself:
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
const savedEvaluatorExample = `from uuid import uuid4
from llmforge import LLMForge

with LLMForge() as forge:
    prompt = forge.get_prompt("Echo demo", version=1)
    dataset = forge.get_dataset("Greetings", version=1)
    catalog = forge.list_evaluators()
    # Inspect catalog["items"], then copy a saved builtin version ID.
    evaluator_version_id = "<saved-builtin-version-UUID>"
    run_id = forge.start_evaluation(
        prompt.id, dataset["revision"]["id"],
        provider="mock", assertions=[],
        evaluators=[{"version_id": evaluator_version_id}],
        idempotency_key=str(uuid4()),
    )
    result = forge.wait_for_evaluation(run_id, timeout=300, poll_interval=3)
    print(result.status)
    # A wait timeout does not cancel the run; inspect this run_id before retrying.`;

function Code({ children }: { children: string }) {
  return (
    <pre className="mt-4 overflow-x-auto rounded-xl border border-(--border) bg-(--surface-2) p-4 text-xs leading-6">
      <code>{children}</code>
    </pre>
  );
}

export default function DocsPage() {
  return (
    <div className="public-shell">
      <PublicHeader />
      <main className="page-width max-w-5xl px-4 sm:px-6 docs-content">
      <header className="my-10">
        <p className="page-eyebrow">Developer guide</p>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight">
          A tested prompt, from draft to application.
        </h1>
        <p className="mt-4 max-w-3xl text-(--muted-foreground) leading-7">
          Keep prompt releases and test cases in one project. Inspect
          regressions in the workspace, fetch a released prompt in Python, and
          use evaluation results to gate CI.
        </p>
      </header>
      <nav
        aria-label="Guide sections"
        className="flex flex-wrap gap-4 text-sm underline"
      >
        <a href="#demo">Five-minute demo</a>
        <a href="#sdk">Install & fetch</a>
        <a href="#ci">CI gate</a>
        <a href="#external">External results</a>
        <a href="#limits">Limits & troubleshooting</a>
      </nav>
      <section id="demo" className="mt-10 panel p-6 scroll-mt-6">
        <h2 className="text-2xl font-semibold">
          Five-minute demo without model credits
        </h2>
        <ol className="mt-4 list-decimal pl-5 space-y-3 text-sm leading-7">
          <li>
            Sign in, select a project, and choose{" "}
            <strong>Create demo examples</strong> on Overview. This safely
            creates or reuses <strong>Echo demo</strong> v1 with{" "}
            <code>{"{{query}}"}</code> and a two-case <strong>Greetings</strong>{" "}
            dataset in this project.
          </li>
          <li>
            Open the preselected evaluation setup. The prompt version and
            dataset revision are fixed saved snapshots.
          </li>
          <li>
            Evaluate prompt v1 on dataset revision 1 using <strong>Demo</strong>{" "}
            and <strong>Exact match</strong>. Both cases pass.
          </li>
          <li>
            Save prompt v2 with text <code>{"Reply: {{query}}"}</code>. Run it
            on the same revision; both checks fail. Open the candidate from Run
            history, choose v1 as the reference run, and filter regressed cases.
          </li>
          <li>
            Promote v1 to production. Fetch it with the SDK below. Enable
            evaluation access on a separate CI key to try the gate.
          </li>
        </ol>
        <p className="mt-4 text-sm text-(--muted-foreground)">
          Demo mode echoes the compiled prompt; it does not call a generation
          model or measure model quality. For a live use case, test a
          support-ticket classifier using JSON labels and references. JSON-path
          checks compare a field to a fixed literal; JSON-reference checks compare
          the entire object. To compare selected fields to each case&apos;s reference,
          compute checks in your application and submit external results.
        </p>
      </section>
      <section id="sdk" className="mt-8 panel p-6 scroll-mt-6">
        <h2 className="text-2xl font-semibold">Install and fetch a release</h2>
        <p className="mt-4 text-sm leading-7">
          The Python package is included in this repository and is not published
          to PyPI. Create a project API key in Settings. Default keys read
          prompts; evaluation access is optional and must be enabled when
          creating the key.
        </p>
        <Code>{install}</Code>
        <p className="mt-4 text-sm leading-7">Set the API root including <code>/api/v1</code>. Hosted consumption needs the SDK and a scoped project key; it does not require running FastAPI or PostgreSQL locally.</p>
        <Code>{fetchExample}</Code>
        <p className="mt-4 text-sm text-(--muted-foreground)">
          Default fetch selects production. Staging and explicit versions are
          supported. The SDK revalidates its memory cache with the server on
          every fetch. Missing releases, revoked keys, and network failures
          raise errors without silently selecting another version.
        </p>
      </section>
      <section id="ci" className="mt-8 panel p-6 scroll-mt-6">
        <h2 className="text-2xl font-semibold">Gate changes in CI</h2>
        <p className="mt-4 text-sm leading-7">
          Use a project key with evaluations:write. Pin a prompt version and
          dataset revision for reproducible results, or select staging to
          evaluate the current candidate. CI needs a reachable backend with a
          persistent process.
        </p>
        <Code>{gateExample}</Code>
        <p className="mt-4 text-sm leading-7">
          <strong>Exit 0:</strong> passed. <strong>Exit 1:</strong> quality
          regression or case errors. <strong>Exit 2:</strong> configuration,
          transport, timeout, or incomplete-run failure. Reports can be uploaded
          as CI artifacts; they include dataset inputs and outputs.
        </p>
        <p className="mt-4 text-sm">
          For live generation set LLMFORGE_PROVIDER_API_KEY and select
          --provider / --model. Live calls use your provider credits. The CLI
          attempts cancellation on timeout.
        </p>
        <p className="mt-4 text-sm">
          Copy docs/examples/evaluation-gate.yml from the repository for a
          complete GitHub Actions workflow. Set the backend URL as a repository
          variable and the evaluation key as a secret.
        </p>
      </section>
      <section id="external" className="mt-8 panel p-6 scroll-mt-6">
        <h2 className="text-2xl font-semibold">
          Evaluate in your own application
        </h2>
        <p className="mt-4 text-sm leading-7">
          Use get_dataset to load fixed cases, run your application or model
          locally, and submit one output and check set per case with
          submit_evaluation. Include named numeric metrics if useful. External
          results appear in the same grid, explicitly marked as
          application-supplied checks. They are self-reported and do not trigger
          provider calls.
        </p>
        <div className="mt-4"><ServerCapabilities /></div>
        <p className="mt-4 text-sm leading-7">For Gemini or LangGraph, fetch and pin the prompt once per batch, compile it locally, run your own provider or graph, then submit exactly one result per case. Direct Gemini server generation is currently unavailable; this does not prevent prompt fetching or external submissions.</p>
        <Code>{`run_id = forge.submit_evaluation(
    prompt.id, dataset["revision"]["id"],
    model="your-application", results=results,
    metrics={"accuracy": 0.95},
)
result = forge.get_evaluation(run_id)
assert result.meets_threshold(0.9)`}</Code>
        <p className="mt-4 text-sm text-(--muted-foreground)">
          Each result includes case_index, output, and checks such as{" "}
          {'{"name": "correctness", "passed": true}'}, or an error. See the
          complete example in sdk/python/examples/submit_results.py.
        </p>
        <p className="mt-4 text-sm leading-7">Open Evaluations → Run history to find SDK-created runs. External comparisons show case changes, but numeric deltas are disabled even when submitted check names match: the checks have no independently pinned evaluator definition.</p>
        <h3 className="mt-6 text-lg font-semibold">Author in the dashboard, pin in the SDK</h3>
        <ol className="mt-4 list-decimal pl-5 space-y-3 text-sm leading-7">
          <li>Select the intended project. In Datasets, create or open a dataset, import CSV/JSON, preview it, apply the cases to the draft, then explicitly save a revision. Applying an import does not save it.</li>
          <li>In Evaluations → Evaluators, create a definition, save its immutable version, and test a passing and failing sample. Save changed definitions as new versions.</li>
          <li>Use an evaluation-enabled key with <code>get_dataset(name, version=...)</code> and <code>list_evaluators()</code>. Record the dataset revision ID and chosen evaluator version ID. SDK keys consume/select these resources; they do not author datasets or evaluators.</li>
          <li>Pass saved evaluator selections to <code>start_evaluation</code> with <code>assertions=[]</code> to omit the SDK&apos;s default exact-match check. Saved-output scoring reuses generation outputs; judge evaluators can still incur model calls.</li>
        </ol>
        <Code>{savedEvaluatorExample}</Code>
      </section>
      <section id="limits" className="mt-8 panel p-6 scroll-mt-6">
        <h2 className="text-2xl font-semibold">Limits and troubleshooting</h2>
        <ul className="mt-4 space-y-3 list-disc pl-5 text-sm leading-7">
          <li>
            <strong>401:</strong> check your project key and whether it was
            revoked. Clerk browser tokens and project API keys serve different
            endpoints.
          </li>
          <li>
            <strong>403:</strong> create a new key with evaluation access
            enabled. Existing read-only keys remain read-only.
          </li>
          <li>
            <strong>404:</strong> verify the project, exact name, release label
            and version. Promote a prompt before fetching production.
          </li>
          <li>
            <strong>422:</strong> check variable names, references, assertions,
            and case limits. Exact match requires a reference for every case.
          </li>
          <li>
            Dataset revisions allow 100 cases / 1 MB. Live runs allow 50 cases,
            or 20 with judging. Two runs can be active per project.
          </li>
          <li>
            Provider/judge calls have a 30-second limit and no automatic retry.
            Restarting the backend loses in-memory credentials. Interrupted
            evaluations preserve committed checkpoints and offer explicit resume
            with fresh keys; a provider response lost before saving may be charged again.
          </li>
          <li>
            LLM judge scores are model opinions. Prefer deterministic checks for
            strict JSON or output contracts, and inspect judge reasons.
          </li>
        </ul>
        <p className="mt-4 text-sm text-(--muted-foreground)">
          The packaged SDK and guide cover this personal project&apos;s current
          workflow. Reasoning benchmarks remain available in the separate
          Benchmarks area.
        </p>
        <div className="mt-4"><DeploymentInfo /></div>
      </section>
      </main>
      <PublicFooter />
    </div>
  );
}
