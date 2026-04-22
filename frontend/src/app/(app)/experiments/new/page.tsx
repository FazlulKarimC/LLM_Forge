"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { AlertTriangle, ArrowRight, LoaderCircle, WandSparkles } from "lucide-react";
import { toast } from "sonner";

import {
  createExperiment,
  CreateExperimentRequest,
  ExperimentConfig,
  getAvailableModels,
  persistCustomProviderCredentials,
  resolveRunExperimentCredentials,
  runExperiment,
} from "@/lib/api";
import {
  ExperimentAdvancedPanel,
  ExperimentAgentPanel,
  ExperimentBasicsPanel,
  ExperimentBuilderSidebar,
  type ExperimentBuilderDatasetGroup,
  type ExperimentBuilderOption,
  type ExperimentBuilderPreset,
  type ExperimentBuilderPresetConfig,
  ExperimentConfigurationPanel,
  ExperimentEvaluationPanel,
  ExperimentGenerationPanel,
  ExperimentOptimizationPanel,
  ExperimentRegressionPanel,
  type ExperimentBuilderFormData,
} from "@/components/experiments/new-experiment-sections";
import { PageHeader } from "@/components/ui/primitives";

const datasetGroups: ExperimentBuilderDatasetGroup[] = [
  {
    label: "General",
    options: [
      { value: "trivia_qa", label: "TriviaQA", description: "Small diagnostic set for single-hop factual recall; useful for smoke comparisons, not broad generalization claims" },
      { value: "commonsense_qa", label: "Commonsense QA", description: "Small diagnostic set for everyday reasoning; compare trends, then inspect failures" },
      { value: "sample", label: "Sample Questions", description: "Built-in smoke test only; do not treat as evidence of model quality" },
    ],
  },
  {
    label: "RAG",
    options: [{ value: "knowledge_base", label: "Knowledge Base QA", description: "Grounded diagnostic questions answerable from indexed articles; best interpreted with retrieved-context inspection" }],
  },
  {
    label: "Reasoning",
    options: [
      { value: "multi_hop", label: "Multi-Hop QA", description: "Small multi-hop diagnostic set; useful for method comparisons with shared samples" },
      { value: "math_reasoning", label: "Math Reasoning", description: "Word-problem style reasoning; check exact outputs because parsing can dominate scores" },
    ],
  },
  {
    label: "Agent",
    options: [{ value: "react_bench", label: "ReAct Bench", description: "Small tool-use diagnostic set requiring search and calculation; inspect traces for termination quality" }],
  },
  {
    label: "Safety",
    options: [
      { value: "prompt_injection", label: "Prompt Injection", description: "Diagnostic prompt-injection probes scored with deterministic robustness rules" },
      { value: "jailbreak", label: "Jailbreak Attempts", description: "Diagnostic jailbreak probes scored with deterministic refusal/compliance rules" },
      { value: "edge_cases", label: "Edge Cases", description: "Diagnostic malformed-input probes; inconclusive responses require manual review" },
    ],
  },
];

const reasoningOptions: readonly ExperimentBuilderOption[] = [
  { value: "naive", label: "Naive prompting", description: "Baseline completion without explicit reasoning scaffolding" },
  { value: "cot", label: "Chain of thought", description: "Encourage intermediate reasoning before the final answer" },
  { value: "react", label: "ReAct agent", description: "Multi-step thought, tool use, and observation loops" },
] as const;

const providerOptions: readonly ExperimentBuilderOption[] = [
  { value: "auto", label: "Auto router", description: "Pick the best available provider and fall back automatically" },
  { value: "hf_api", label: "Hugging Face API", description: "Serverless and free, but often slower" },
  { value: "openrouter", label: "OpenRouter", description: "Use free-tier models with provider-side routing" },
  { value: "groq", label: "Groq", description: "Very fast inference with stricter limits" },
  { value: "custom", label: "Custom endpoint (dev only)", description: "Your own OpenAI-compatible endpoint for local or development-only testing" },
] as const;

const routingPolicyOptions: readonly ExperimentBuilderOption[] = [
  { value: "fallback_chain", label: "Fallback chain", description: "Try the primary provider first and fall back only on transient failures" },
  { value: "cheapest_first", label: "Cheapest first", description: "Prefer the lowest-cost provider based on per-request cost tracking" },
  { value: "fastest_first", label: "Fastest first", description: "Prefer the lowest-latency provider based on accumulated telemetry" },
  { value: "adaptive", label: "Adaptive", description: "Explore early, then exploit the best composite score for latency, cost, and error rate" },
] as const;

const retrievalOptions: readonly ExperimentBuilderOption[] = [
  { value: "none", label: "No retrieval", description: "Run the model on the prompt only" },
  { value: "naive", label: "Naive RAG", description: "Dense retrieval over indexed chunks" },
  { value: "hybrid", label: "Hybrid RAG", description: "Dense retrieval plus BM25" },
  { value: "reranked", label: "Reranked RAG", description: "Hybrid retrieval followed by a reranker" },
] as const;

const presetConfigs: ExperimentBuilderPresetConfig[] = [
  {
    title: "Baseline vs reasoning",
    description: "Start with a smaller QA dataset and compare naive prompting against CoT.",
    apply: {
      reasoning_method: "cot" as const,
      dataset_name: "multi_hop",
      num_samples: 20,
      retrieval_method: "none" as const,
    },
  },
  {
    title: "RAG grounding check",
    description: "Use the knowledge base dataset with hybrid retrieval to inspect grounded answers.",
    apply: {
      reasoning_method: "naive" as const,
      dataset_name: "knowledge_base",
      num_samples: 20,
      retrieval_method: "hybrid" as const,
    },
  },
  {
    title: "Agent stress test",
    description: "Run ReAct over the tool-use benchmark with explicit tool access.",
    apply: {
      reasoning_method: "react" as const,
      dataset_name: "react_bench",
      num_samples: 15,
      retrieval_method: "none" as const,
    },
  },
];

export default function NewExperimentPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const modelsQuery = useQuery({
    queryKey: ["available-models"],
    queryFn: ({ signal }) => getAvailableModels({ signal }),
    staleTime: Infinity,
  });

  const availableModels = modelsQuery.data?.models ?? [
    { value: "meta-llama/Llama-3.2-1B-Instruct", label: "Llama 3.2 (1B)", description: "Fast, efficient default" },
  ];

  const [formData, setFormData] = useState<ExperimentBuilderFormData>({
    name: "",
    description: "",
    model_name: "meta-llama/Llama-3.2-1B-Instruct",
    reasoning_method: "naive" as "naive" | "cot" | "react",
    dataset_name: "trivia_qa",
    provider: "auto" as "auto" | "hf_api" | "openrouter" | "groq" | "custom",
    temperature: 0.1,
    max_tokens: 150,
    num_samples: 10,
    retrieval_method: "none" as "none" | "naive" | "hybrid" | "reranked",
    rag_top_k: 5,
    agent_max_iterations: 5,
    agent_tools: ["wikipedia_search", "calculator"],
    enable_batching: false,
    batch_size: 8,
    enable_caching: false,
    cache_max_size: 256,
    routing_policy: "fallback_chain" as "fallback_chain" | "cheapest_first" | "fastest_first" | "adaptive",
    routing_epsilon: 0.15,
    routing_exploration_window: 10,
    enable_regression: false,
    regression_accuracy_min_delta: -0.05,
    regression_f1_min_delta: -0.05,
    regression_latency_p95_max_ms: "" as number | "",
    regression_no_sample_regressions: false,
    regression_max_new_failures: "" as number | "",
    regression_min_overlap_ratio: 0.8,
    prompt_version_id: "",
    graders_json: "",
    seed: "" as number | "",
  });

  const [validationError, setValidationError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [customBaseUrl, setCustomBaseUrl] = useState(() => {
    if (typeof window !== "undefined") return sessionStorage.getItem("customBaseUrl") || "http://localhost:8000/v1";
    return "http://localhost:8000/v1";
  });
  const [customApiKey, setCustomApiKey] = useState(() => {
    if (typeof window !== "undefined") return sessionStorage.getItem("customApiKey") || "";
    return "";
  });
  const [customModelId, setCustomModelId] = useState(() => {
    if (typeof window !== "undefined") return sessionStorage.getItem("customModelId") || "";
    return "";
  });

  const createMutation = useMutation({
    mutationFn: ({ request }: { request: CreateExperimentRequest; shouldRun: boolean }) => createExperiment(request),
    onSuccess: async (experiment, variables) => {
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });

      if (variables.shouldRun) {
        try {
          const credentials = resolveRunExperimentCredentials(variables.request.config);
          await runExperiment(experiment.id, credentials.customBaseUrl, credentials.customApiKey);
        } catch (error) {
          const message = error instanceof Error
            ? `Experiment created but failed to start: ${error.message}`
            : "Experiment created but failed to start. Retry from the detail page.";
          setRunError(message);
          toast.error(message);
        }
      }

      router.push(`/experiments/${experiment.id}`);
    },
  });

  const selectedModel = availableModels.find((model) => model.value === formData.model_name);
  const selectedDataset = datasetGroups.flatMap((group) => group.options).find((dataset) => dataset.value === formData.dataset_name);
  const complexityScore = Math.min(100, Math.round((formData.num_samples / 60) * 45 + (formData.max_tokens / 800) * 25 + (formData.retrieval_method !== "none" ? 15 : 0) + (formData.reasoning_method === "react" ? 15 : formData.reasoning_method === "cot" ? 8 : 0)));

  function updateField<K extends keyof typeof formData>(key: K, value: (typeof formData)[K]) {
    setFormData((current) => ({ ...current, [key]: value }));
  }

  function applyPreset(preset: ExperimentBuilderPreset) {
    setFormData((current) => ({
      ...current,
      reasoning_method: preset.reasoning_method,
      dataset_name: preset.dataset_name,
      num_samples: preset.num_samples,
      retrieval_method: preset.retrieval_method,
    }));
  }

  function handleSubmit(event: React.FormEvent, shouldRun = false) {
    event.preventDefault();

    if (!formData.name.trim()) {
      setValidationError("Experiment name is required.");
      return;
    }
    if (formData.num_samples < 1 || formData.num_samples > 500) {
      setValidationError("Samples must be between 1 and 500.");
      return;
    }
    if (formData.enable_batching && (formData.batch_size < 1 || formData.batch_size > 32)) {
      setValidationError("Batch size must be between 1 and 32.");
      return;
    }
    if (formData.enable_caching && (formData.cache_max_size < 16 || formData.cache_max_size > 2048)) {
      setValidationError("Cache size must be between 16 and 2,048.");
      return;
    }
    if (formData.routing_epsilon < 0 || formData.routing_epsilon > 1) {
      setValidationError("Adaptive router epsilon must be between 0 and 1.");
      return;
    }
    if (formData.routing_exploration_window < 1 || formData.routing_exploration_window > 50) {
      setValidationError("Adaptive router exploration window must be between 1 and 50.");
      return;
    }
    if (formData.regression_min_overlap_ratio < 0 || formData.regression_min_overlap_ratio > 1) {
      setValidationError("Minimum overlap ratio must be between 0 and 1.");
      return;
    }
    if (formData.model_name === "custom_hosted" && (!customBaseUrl.trim() || !customModelId.trim())) {
      setValidationError("Custom hosted models require a base URL and a model id.");
      return;
    }

    setValidationError(null);
    setRunError(null);

    if (formData.model_name === "custom_hosted") {
      persistCustomProviderCredentials(customModelId, customBaseUrl, customApiKey);
    }

    const config: ExperimentConfig = {
      model_name: formData.model_name === "custom_hosted" ? customModelId : formData.model_name,
      reasoning_method: formData.reasoning_method,
      dataset_name: formData.dataset_name,
      provider: formData.provider,
      hyperparameters: {
        temperature: formData.temperature,
        max_tokens: formData.max_tokens,
        ...(formData.seed !== "" ? { seed: formData.seed as number } : {}),
      },
      num_samples: formData.num_samples,
    };

    if (formData.retrieval_method !== "none") {
      config.rag = {
        retrieval_method: formData.retrieval_method,
        top_k: formData.rag_top_k,
      };
    }

    if (formData.reasoning_method === "react") {
      config.agent = {
        max_iterations: formData.agent_max_iterations,
        tools: formData.agent_tools,
      };
    }

    if (formData.enable_batching || formData.enable_caching) {
      config.optimization = {
        enable_batching: formData.enable_batching,
        batch_size: formData.batch_size,
        enable_caching: formData.enable_caching,
        cache_max_size: formData.cache_max_size,
        enable_profiling: true,
      };
    }

    if (formData.provider === "auto") {
      config.routing = {
        policy: formData.routing_policy,
        epsilon: formData.routing_epsilon,
        exploration_window: formData.routing_exploration_window,
      };
    }

    if (formData.enable_regression) {
      config.regression = {
        accuracy_min_delta: formData.regression_accuracy_min_delta,
        f1_min_delta: formData.regression_f1_min_delta,
        no_sample_regressions: formData.regression_no_sample_regressions,
        min_overlap_ratio: formData.regression_min_overlap_ratio,
        ...(formData.regression_latency_p95_max_ms !== "" ? { latency_p95_max_ms: formData.regression_latency_p95_max_ms } : {}),
        ...(formData.regression_max_new_failures !== "" ? { max_new_failures: formData.regression_max_new_failures } : {}),
      };
    }

    if (formData.prompt_version_id.trim()) {
      config.prompt_version_id = formData.prompt_version_id.trim();
    }

    if (formData.graders_json.trim()) {
      try {
        config.graders = JSON.parse(formData.graders_json);
      } catch {
        setValidationError("Deterministic graders JSON must be valid JSON.");
        return;
      }
    }

    const request: CreateExperimentRequest = {
      name: formData.name,
      description: formData.description || undefined,
      config,
    };

    createMutation.mutate({ request, shouldRun });
  }

  return (
    <div className="page-stack">
      <PageHeader
        backHref="/experiments"
        backLabel="Back to experiments"
        eyebrow={<><WandSparkles className="size-3.5" /> Experiment builder</>}
        title="Configure and launch a new experiment"
        description="Set up the model, dataset, reasoning method, and parameters for your next evaluation run."
      />

      {validationError ? (
        <div className="alert alert-danger">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p className="text-sm leading-7">{validationError}</p>
        </div>
      ) : null}
      {runError ? (
        <div className="alert alert-warning">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p className="text-sm leading-7">{runError}</p>
        </div>
      ) : null}
      {createMutation.error ? (
        <div className="alert alert-danger">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p className="text-sm leading-7">{createMutation.error instanceof Error ? createMutation.error.message : "Failed to create experiment"}</p>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <form
          onSubmit={(event) => handleSubmit(event, false)}
          className="space-y-6 route-enter"
        >
          <ExperimentBasicsPanel formData={formData} updateField={updateField} />

          <ExperimentConfigurationPanel
            availableModels={availableModels}
            customApiKey={customApiKey}
            customBaseUrl={customBaseUrl}
            customModelId={customModelId}
            formData={formData}
            providerOptions={providerOptions}
            reasoningOptions={reasoningOptions}
            routingPolicyOptions={routingPolicyOptions}
            selectedModelDescription={selectedModel?.description}
            updateField={updateField}
            onCustomApiKeyChange={setCustomApiKey}
            onCustomBaseUrlChange={setCustomBaseUrl}
            onCustomModelIdChange={setCustomModelId}
          />

          <ExperimentEvaluationPanel
            datasetGroups={datasetGroups}
            formData={formData}
            retrievalOptions={retrievalOptions}
            selectedDatasetDescription={selectedDataset?.description}
            updateField={updateField}
          />

          <ExperimentGenerationPanel formData={formData} updateField={updateField} />

          {formData.reasoning_method === "react" ? <ExperimentAgentPanel formData={formData} updateField={updateField} /> : null}

          <ExperimentOptimizationPanel formData={formData} updateField={updateField} />
          <ExperimentRegressionPanel formData={formData} updateField={updateField} />
          <ExperimentAdvancedPanel formData={formData} updateField={updateField} />

          <div className="flex flex-wrap justify-end gap-3">
            <Link href="/experiments" className="btn-secondary">Cancel</Link>
            <button type="submit" className="btn-secondary" disabled={createMutation.isPending}>
              {createMutation.isPending && !createMutation.variables?.shouldRun ? <LoaderCircle className="size-4 animate-spin" /> : null}
              Create draft
            </button>
            <button type="button" className="btn-primary" disabled={createMutation.isPending} onClick={(event) => handleSubmit(event, true)}>
              {createMutation.isPending && createMutation.variables?.shouldRun ? <LoaderCircle className="size-4 animate-spin" /> : null}
              Create and run
              <ArrowRight className="size-4" />
            </button>
          </div>
        </form>

        <ExperimentBuilderSidebar
          complexityScore={complexityScore}
          customModelId={customModelId}
          formData={formData}
          onApplyPreset={applyPreset}
          presetConfigs={presetConfigs}
          selectedModelLabel={selectedModel?.label}
        />
      </div>
    </div>
  );
}
