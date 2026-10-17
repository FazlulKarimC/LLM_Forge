export const productPreviews = [
  {
    id: "benchmarks", label: "Benchmarks", mobileWidth: 369, mobileHeight: 395, title: "Compare quality, latency and usage together.",
    description: "Inspect quality, latency and token usage side by side on the same test set. Keep the configuration and per-case outputs behind every result.",
    caption: "Seeded examples: A is Compact setup; B is Deliberate setup. These illustrative measurements in the real workspace are not claims about model performance.",
    alt: "LLMForge benchmark comparison showing quality and latency for two seeded example configurations",
    benefit: "Find the model trade-off that fits.",
    detail: "Compare benchmark runs on fixed examples. Inspect accuracy, latency, token usage and disagreements before choosing a setup.",
  },
  {
    id: "prompts", label: "Prompts", mobileWidth: 384, mobileHeight: 560, title: "Make a prompt change you can trace.",
    description: "Edit a candidate, keep immutable versions, and move release labels when it is ready. Your application can fetch a released prompt through the Python SDK.",
    caption: "The Echo demo prompt in the actual prompt editor. Demo text is deliberately simple; live playground calls require your provider key.",
    alt: "LLMForge Echo demo prompt editor with a saved template and format controls",
    benefit: "Turn a prompt draft into a tested release.",
    detail: "Develop text or chat prompts, test candidates in the playground, and retain the version history behind each release.",
  },
  {
    id: "evaluations", label: "Evaluations", mobileWidth: 350, mobileHeight: 391, title: "Track evaluations from your application.",
    description: "Run local cases and evaluator callbacks with the SDK, then compare compatible summary reports. Full inputs, outputs and checks stay in your application.",
    caption: "Legacy hosted Echo history shown here remains readable. New evaluations upload summaries from the SDK; this illustration does not measure model quality.",
    alt: "Legacy LLMForge evaluation history with saved checks and candidate output",
    benefit: "Catch failures before the next release.",
    detail: "Keep local cases and full reports in your application, track summary counts here, and apply quality thresholds through SDK and CI gates.",
  },
] as const;
export type PreviewId = typeof productPreviews[number]["id"];
