"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { AlertTriangle, ArrowRight, Database, Layers3, LoaderCircle, PlugZap, WandSparkles } from "lucide-react";
import { toast } from "sonner";

import { createExperiment, CreateExperimentRequest, ExperimentConfig, getAvailableModels, runExperiment } from "@/lib/api";
import { MetricBar, PageHeader, Panel, PanelHeader } from "@/components/ui/primitives";

const datasetGroups = [
  {
    label: "General",
    options: [
      { value: "trivia_qa", label: "TriviaQA", description: "Single-hop factual recall questions" },
      { value: "commonsense_qa", label: "Commonsense QA", description: "Everyday knowledge and reasoning" },
      { value: "sample", label: "Sample Questions", description: "Built-in smoke test questions" },
    ],
  },
  {
    label: "RAG",
    options: [{ value: "knowledge_base", label: "Knowledge Base QA", description: "Grounded questions answerable from indexed articles" }],
  },
  {
    label: "Reasoning",
    options: [
      { value: "multi_hop", label: "Multi-Hop QA", description: "Requires combining multiple facts" },
      { value: "math_reasoning", label: "Math Reasoning", description: "Word-problem style reasoning" },
    ],
  },
  {
    label: "Agent",
    options: [{ value: "react_bench", label: "ReAct Bench", description: "Tool-use questions requiring search and calculation" }],
  },
  {
    label: "Safety",
    options: [
      { value: "prompt_injection", label: "Prompt Injection", description: "Tests instruction override resistance" },
      { value: "jailbreak", label: "Jailbreak Attempts", description: "Tests DAN-style jailbreak resistance" },
      { value: "edge_cases", label: "Edge Cases", description: "Tests unusual or malformed inputs" },
    ],
  },
];

const reasoningOptions = [
  { value: "naive", label: "Naive prompting", description: "Baseline completion without explicit reasoning scaffolding" },
  { value: "cot", label: "Chain of thought", description: "Encourage intermediate reasoning before the final answer" },
  { value: "react", label: "ReAct agent", description: "Multi-step thought, tool use, and observation loops" },
] as const;

const providerOptions = [
  { value: "auto", label: "Auto router", description: "Pick the best available provider and fall back automatically" },
  { value: "hf_api", label: "Hugging Face API", description: "Serverless and free, but often slower" },
  { value: "openrouter", label: "OpenRouter", description: "Use free-tier models with provider-side routing" },
  { value: "groq", label: "Groq", description: "Very fast inference with stricter limits" },
  { value: "custom", label: "Custom endpoint", description: "Your own OpenAI-compatible endpoint" },
] as const;

const retrievalOptions = [
  { value: "none", label: "No retrieval", description: "Run the model on the prompt only" },
  { value: "naive", label: "Naive RAG", description: "Dense retrieval over indexed chunks" },
  { value: "hybrid", label: "Hybrid RAG", description: "Dense retrieval plus BM25" },
  { value: "reranked", label: "Reranked RAG", description: "Hybrid retrieval followed by a reranker" },
] as const;

const presetConfigs = [
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
    staleTime: 1000 * 60 * 5,
  });

  const availableModels = modelsQuery.data?.models ?? [
    { value: "meta-llama/Llama-3.2-1B-Instruct", label: "Llama 3.2 (1B)", description: "Fast, efficient default" },
  ];

  const [formData, setFormData] = useState({
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
    seed: "" as number | "",
  });

  const [validationError, setValidationError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [customBaseUrl, setCustomBaseUrl] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("customBaseUrl") || "http://localhost:8000/v1";
    return "http://localhost:8000/v1";
  });
  const [customApiKey, setCustomApiKey] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("customApiKey") || "";
    return "";
  });
  const [customModelId, setCustomModelId] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("customModelId") || "";
    return "";
  });

  const createMutation = useMutation({
    mutationFn: ({ request }: { request: CreateExperimentRequest; shouldRun: boolean }) => createExperiment(request),
    onSuccess: async (experiment, variables) => {
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });

      if (variables.shouldRun) {
        try {
          await runExperiment(
            experiment.id,
            formData.model_name === "custom_hosted" ? customBaseUrl : undefined,
            formData.model_name === "custom_hosted" ? customApiKey : undefined
          );
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

  function applyPreset(preset: (typeof presetConfigs)[number]["apply"]) {
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
    if (formData.enable_caching && (formData.cache_max_size < 8 || formData.cache_max_size > 4096)) {
      setValidationError("Cache size must be between 8 and 4096.");
      return;
    }
    if (formData.model_name === "custom_hosted" && (!customBaseUrl.trim() || !customModelId.trim())) {
      setValidationError("Custom hosted models require a base URL and a model id.");
      return;
    }

    setValidationError(null);
    setRunError(null);

    if (formData.model_name === "custom_hosted" && typeof window !== "undefined") {
      try {
        const settings = JSON.parse(localStorage.getItem("customLLMSettings") || "{}");
        settings[customModelId] = { baseUrl: customBaseUrl, apiKey: customApiKey };
        localStorage.setItem("customLLMSettings", JSON.stringify(settings));
        localStorage.setItem("customBaseUrl", customBaseUrl);
        localStorage.setItem("customApiKey", customApiKey);
        localStorage.setItem("customModelId", customModelId);
      } catch (error) {
        console.error("Failed to store custom model settings", error);
      }
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
        eyebrow={<><WandSparkles className="size-3.5" /> Experiment builder</>}
        title="Configure and launch a new experiment"
        description="Set up the model, dataset, reasoning method, and parameters for your next evaluation run."
        actions={<Link href="/experiments" className="btn-secondary">Back to experiments</Link>}
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
        <motion.form
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] as const }}
          onSubmit={(event) => handleSubmit(event, false)}
          className="space-y-4"
        >
          <Panel>
            <PanelHeader label="Basics" title="Name the experiment" description="Choose a clear title that reads well in tables, exports, and comparisons." />
            <div className="panel-body grid grid-cols-1 gap-4 lg:grid-cols-[1fr_1.1fr]">
              <div>
                <label className="field-label" htmlFor="experiment-name">Experiment name</label>
                <input
                  id="experiment-name"
                  type="text"
                  required
                  value={formData.name}
                  onChange={(event) => updateField("name", event.target.value)}
                  placeholder="e.g. cot_vs_naive_multihop"
                  className="input-shell"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="experiment-description">Description</label>
                <textarea
                  id="experiment-description"
                  value={formData.description}
                  onChange={(event) => updateField("description", event.target.value)}
                  placeholder="Optional context for why this run exists"
                  className="textarea-shell"
                  rows={3}
                />
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Configuration" title="Model and reasoning" description="Select the model, inference provider, and reasoning strategy." />
            <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <label className="field-label" htmlFor="model-name">Model</label>
                <select id="model-name" value={formData.model_name} onChange={(event) => {
                  const value = event.target.value;
                  updateField("model_name", value);
                  if (value === "custom_hosted" && formData.provider !== "custom") {
                    updateField("provider", "custom");
                  } else if (value !== "custom_hosted" && formData.provider === "custom") {
                    updateField("provider", "auto");
                  }
                }} className="select-shell">
                  <optgroup label="Hugging Face / serverless">
                    {availableModels.map((model) => (
                      <option key={model.value} value={model.value}>{model.label}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Custom">
                    <option value="custom_hosted">Custom hosted model</option>
                  </optgroup>
                </select>
                <p className="field-help mt-2">{formData.model_name === "custom_hosted" ? "Use this when pointing at a local or hosted OpenAI-compatible endpoint." : selectedModel?.description}</p>
              </div>
              <div>
                <label className="field-label" htmlFor="reasoning-method">Reasoning method</label>
                <select id="reasoning-method" value={formData.reasoning_method} onChange={(event) => updateField("reasoning_method", event.target.value as "naive" | "cot" | "react")} className="select-shell">
                  {reasoningOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <p className="field-help mt-2">{reasoningOptions.find((option) => option.value === formData.reasoning_method)?.description}</p>
              </div>
              <div>
                <label className="field-label" htmlFor="provider">Provider</label>
                <select id="provider" value={formData.provider} onChange={(event) => {
                  const value = event.target.value as typeof formData.provider;
                  updateField("provider", value);
                  if (value === "custom" && formData.model_name !== "custom_hosted") {
                    updateField("model_name", "custom_hosted");
                  } else if (value !== "custom" && formData.model_name === "custom_hosted") {
                    updateField("model_name", availableModels[0]?.value ?? "meta-llama/Llama-3.2-1B-Instruct");
                  }
                }} className="select-shell">
                  {providerOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <p className="field-help mt-2">{providerOptions.find((option) => option.value === formData.provider)?.description}</p>
              </div>
            </div>

            {formData.model_name === "custom_hosted" ? (
              <div className="panel-body pt-0">
                <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div>
                      <label className="field-label" htmlFor="custom-base-url">Base URL</label>
                      <input id="custom-base-url" type="url" value={customBaseUrl} onChange={(event) => setCustomBaseUrl(event.target.value)} className="input-shell font-mono text-sm" placeholder="http://localhost:8000/v1" />
                    </div>
                    <div>
                      <label className="field-label" htmlFor="custom-api-key">API key</label>
                      <input id="custom-api-key" type="password" value={customApiKey} onChange={(event) => setCustomApiKey(event.target.value)} className="input-shell font-mono text-sm" placeholder="sk-..." />
                    </div>
                    <div>
                      <label className="field-label" htmlFor="custom-model-id">Model id</label>
                      <input id="custom-model-id" type="text" value={customModelId} onChange={(event) => setCustomModelId(event.target.value)} className="input-shell font-mono text-sm" placeholder="e.g. llama-3-8b" />
                    </div>
                  </div>
                </div>
              </div>
            ) : null}
          </Panel>

          <Panel>
            <PanelHeader label="Evaluation setup" title="Dataset and runtime" description="Choose the evaluation dataset, retrieval strategy, and runtime parameters." />
            <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-2">
              <div>
                <label className="field-label" htmlFor="dataset">Dataset</label>
                <select id="dataset" value={formData.dataset_name} onChange={(event) => updateField("dataset_name", event.target.value)} className="select-shell">
                  {datasetGroups.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.options.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <p className="field-help mt-2">{selectedDataset?.description}</p>
              </div>
              <div>
                <label className="field-label" htmlFor="samples">Samples</label>
                <input id="samples" type="number" min="1" max="500" value={formData.num_samples} onChange={(event) => updateField("num_samples", parseInt(event.target.value, 10) || 1)} className="input-shell" />
                <p className="field-help mt-2">Max 500 samples on the current API.</p>
              </div>
              <div>
                <label className="field-label" htmlFor="retrieval-method">Retrieval method</label>
                <select id="retrieval-method" value={formData.retrieval_method} onChange={(event) => updateField("retrieval_method", event.target.value as typeof formData.retrieval_method)} className="select-shell">
                  {retrievalOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <p className="field-help mt-2">{retrievalOptions.find((option) => option.value === formData.retrieval_method)?.description}</p>
              </div>
              {formData.retrieval_method !== "none" ? (
                <div>
                  <label className="field-label" htmlFor="rag-top-k">Top-k chunks</label>
                  <input id="rag-top-k" type="number" min="1" max="20" value={formData.rag_top_k} onChange={(event) => updateField("rag_top_k", parseInt(event.target.value, 10) || 1)} className="input-shell" />
                  <p className="field-help mt-2">Controls how much retrieved context is injected per question.</p>
                </div>
              ) : null}
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Generation" title="Hyperparameters" description="Control randomness, output length, and reproducibility." />
            <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <label className="field-label" htmlFor="temperature">Temperature</label>
                <input id="temperature" type="number" step="0.1" min="0" max="2" value={formData.temperature} onChange={(event) => updateField("temperature", parseFloat(event.target.value) || 0)} className="input-shell" />
              </div>
              <div>
                <label className="field-label" htmlFor="max-tokens">Max tokens</label>
                <input id="max-tokens" type="number" min="1" max="4096" value={formData.max_tokens} onChange={(event) => updateField("max_tokens", parseInt(event.target.value, 10) || 1)} className="input-shell" />
              </div>
              <div>
                <label className="field-label" htmlFor="seed">Seed</label>
                <input id="seed" type="number" min="0" value={formData.seed} onChange={(event) => updateField("seed", event.target.value === "" ? "" : parseInt(event.target.value, 10))} className="input-shell" placeholder="Optional" />
              </div>
            </div>
          </Panel>

          {formData.reasoning_method === "react" ? (
            <Panel>
              <PanelHeader label="Agent settings" title="Tool loop controls" description="Configure tool access and iteration limits for ReAct agent runs." />
              <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <label className="field-label" htmlFor="agent-max-iterations">Max iterations</label>
                  <input id="agent-max-iterations" type="number" min="1" max="20" value={formData.agent_max_iterations} onChange={(event) => updateField("agent_max_iterations", parseInt(event.target.value, 10) || 1)} className="input-shell" />
                  <p className="field-help mt-2">Maximum Thought -&gt; Action -&gt; Observation loops.</p>
                </div>
                <div>
                  <div className="field-label">Enabled tools</div>
                  <div className="space-y-2 rounded-[18px] border border-(--border) bg-(--surface-2) p-4 text-sm text-(--muted-foreground)">
                    {[
                      { value: "wikipedia_search", label: "Wikipedia search" },
                      { value: "calculator", label: "Calculator" },
                      { value: "retrieval", label: "Retrieval" },
                    ].map((tool) => (
                      <label key={tool.value} className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          checked={formData.agent_tools.includes(tool.value)}
                          onChange={(event) => {
                            const next = event.target.checked
                              ? [...formData.agent_tools, tool.value]
                              : formData.agent_tools.filter((item) => item !== tool.value);
                            updateField("agent_tools", next);
                          }}
                        />
                        {tool.label}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </Panel>
          ) : null}

          <Panel>
            <PanelHeader label="Optimization" title="Execution options" description="Enable batching and caching to optimize throughput and reduce costs." />
            <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-2">
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <label className="flex items-center gap-3 font-medium">
                  <input type="checkbox" checked={formData.enable_batching} onChange={(event) => updateField("enable_batching", event.target.checked)} />
                  Enable batching
                </label>
                <p className="field-help mt-3">Parallelize eligible prompts to reduce wall-clock time.</p>
                {formData.enable_batching ? (
                  <div className="mt-4">
                    <label className="field-label" htmlFor="batch-size">Batch size</label>
                    <input id="batch-size" type="number" min="1" max="32" value={formData.batch_size} onChange={(event) => updateField("batch_size", parseInt(event.target.value, 10) || 1)} className="input-shell" />
                  </div>
                ) : null}
              </div>
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <label className="flex items-center gap-3 font-medium">
                  <input type="checkbox" checked={formData.enable_caching} onChange={(event) => updateField("enable_caching", event.target.checked)} />
                  Enable caching
                </label>
                <p className="field-help mt-3">Reuse identical prompts where deterministic settings allow it.</p>
                {formData.enable_caching ? (
                  <div className="mt-4">
                    <label className="field-label" htmlFor="cache-size">Cache size</label>
                    <input id="cache-size" type="number" min="8" max="4096" value={formData.cache_max_size} onChange={(event) => updateField("cache_max_size", parseInt(event.target.value, 10) || 8)} className="input-shell" />
                  </div>
                ) : null}
              </div>
            </div>
          </Panel>

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
        </motion.form>

        <div className="space-y-4 xl:sticky xl:top-24 xl:self-start">
          <Panel>
            <PanelHeader label="Summary" title={formData.name || "Untitled experiment"} description="Overview of the experiment configuration." />
            <div className="panel-body space-y-4">
              <div className="flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
                <span className="chip">{formData.reasoning_method.toUpperCase()}</span>
                <span className="chip">{formData.dataset_name}</span>
                <span className="chip">{formData.provider}</span>
              </div>
              <div className="space-y-3 text-sm text-(--muted-foreground)">
                <div className="flex items-center justify-between gap-3"><span>Model</span><span className="font-medium text-foreground">{formData.model_name === "custom_hosted" ? customModelId || "Custom hosted" : selectedModel?.label}</span></div>
                <div className="flex items-center justify-between gap-3"><span>Samples</span><span className="metric-value text-foreground">{formData.num_samples}</span></div>
                <div className="flex items-center justify-between gap-3"><span>Max tokens</span><span className="metric-value text-foreground">{formData.max_tokens}</span></div>
                <div className="flex items-center justify-between gap-3"><span>Retrieval</span><span className="font-medium text-foreground">{formData.retrieval_method}</span></div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-(--muted-foreground)">Complexity</span>
                  <span className="metric-value">{complexityScore}%</span>
                </div>
                <MetricBar value={complexityScore} />
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Quick starts" title="Preset configurations" description="Quick-start templates for common evaluation scenarios." />
            <div className="panel-body space-y-3">
              {presetConfigs.map((preset) => (
                <button key={preset.title} type="button" className="w-full rounded-[18px] border border-(--border) bg-(--surface-2) p-4 text-left transition-all hover:border-(--border-strong) hover:bg-(--surface-3)" onClick={() => applyPreset(preset.apply)}>
                  <div className="font-semibold tracking-[-0.03em]">{preset.title}</div>
                  <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">{preset.description}</p>
                </button>
              ))}
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Field notes" title="Execution trade-offs" description="Tips for understanding how settings affect run time and cost." />
            <div className="panel-body space-y-3 text-sm leading-7 text-(--muted-foreground)">
              <div className="flex gap-3 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <Database className="mt-1 size-4 shrink-0 text-(--accent)" />
                Retrieval adds context-fetch overhead, especially with hybrid or reranked modes.
              </div>
              <div className="flex gap-3 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <Layers3 className="mt-1 size-4 shrink-0 text-(--primary)" />
                More samples and larger max tokens increase result payload size and evaluation time.
              </div>
              <div className="flex gap-3 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <PlugZap className="mt-1 size-4 shrink-0 text-(--success)" />
                Batching and caching are safe to enable because the backend already profiles them in result exports.
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}



