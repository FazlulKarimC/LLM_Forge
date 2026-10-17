import type { Metadata } from "next";
import { PublicHeader } from "@/components/public-header";
import { PublicFooter } from "@/components/public-footer";
import { DeploymentInfo } from "@/components/deployment-info";
import { SdkEvaluationGuide } from "@/components/sdk-evaluation-guide";
export const metadata: Metadata = { title: "Guide | LLMForge", description: "Manage prompt releases and track local SDK evaluations." };
export default function DocsPage() {
  return <div className="public-shell"><PublicHeader /><main className="page-width px-4 py-10 space-y-6">
    <h1 className="text-3xl font-semibold">Prompts and local evaluations</h1>
    <p>Save immutable prompt versions, test them in your application, compare reported results and explicitly assign staging or production releases. Benchmarks remain a separate reasoning workflow.</p>
    <section className="panel p-5 space-y-3"><h2 className="font-semibold">Configure the SDK</h2>
      <p>Install from sdk/python in this repository; the SDK is not published to PyPI. Hosted consumption requires no local backend or database.</p>
      <pre className="code-panel overflow-auto"><code>{'python -m pip install -e ./sdk/python\nLLMFORGE_URL=https://faz-ai-llm-forge.hf.space/api/v1\nLLMFORGE_API_KEY=<evaluation-enabled project key>\n\n# For your own backend:\nLLMFORGE_URL=http://localhost:8000/api/v1'}</code></pre>
      <p>Use the same Clerk application for frontend and backend when hosting your own instance. Project keys determine SDK project scope; provider credentials belong only in your local application.</p>
    </section>
    <SdkEvaluationGuide />
    <section className="panel p-5 space-y-3"><h2 className="font-semibold">CI and comparison</h2>
      <p>Keep stable case IDs and pin the prompt version. Use a new scoring_version when evaluator code/settings change. The SDK computes fingerprints of the actual selected cases and scoring identity. The dashboard only offers references with matching fingerprints.</p>
      <p>Full reports stay local. The CLI check command gates a local artifact; incomplete runs or execution errors cannot pass a quality gate. Summary upload needs an evaluation-enabled project key. Copy docs/examples/evaluation-gate.yml for a provider-free CI example.</p>
      <p>Hosted dataset authoring is retired. Existing hosted evaluation history remains readable. External scores are self-reported; numeric per-score deltas and individual case regressions cannot be inferred from summary-only reports.</p>
    </section>
    <DeploymentInfo />
  </main><PublicFooter /></div>;
}
