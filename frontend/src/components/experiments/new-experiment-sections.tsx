"use client";

import { Database, Layers3, PlugZap } from "lucide-react";

import { MetricBar, Panel, PanelHeader } from "@/components/ui/primitives";
import type {
  ExperimentBuilderDatasetGroup,
  ExperimentBuilderFormData,
  ExperimentBuilderOption,
  ExperimentBuilderPreset,
  ExperimentBuilderPresetConfig,
  UpdateExperimentBuilderField,
} from "@/components/experiments/new-experiment-types";

export type {
  ExperimentBuilderDatasetGroup,
  ExperimentBuilderFormData,
  ExperimentBuilderOption,
  ExperimentBuilderPreset,
  ExperimentBuilderPresetConfig,
} from "@/components/experiments/new-experiment-types";

type UpdateField = UpdateExperimentBuilderField;

export function ExperimentBasicsPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
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
  );
}

export function ExperimentConfigurationPanel({
  availableModels,
  customApiKey,
  customBaseUrl,
  customModelId,
  formData,
  providerOptions,
  reasoningOptions,
  routingPolicyOptions,
  selectedModelDescription,
  updateField,
  onCustomApiKeyChange,
  onCustomBaseUrlChange,
  onCustomModelIdChange,
}: {
  availableModels: ExperimentBuilderOption[];
  customApiKey: string;
  customBaseUrl: string;
  customModelId: string;
  formData: ExperimentBuilderFormData;
  providerOptions: readonly ExperimentBuilderOption[];
  reasoningOptions: readonly ExperimentBuilderOption[];
  routingPolicyOptions: readonly ExperimentBuilderOption[];
  selectedModelDescription?: string;
  updateField: UpdateField;
  onCustomApiKeyChange: (value: string) => void;
  onCustomBaseUrlChange: (value: string) => void;
  onCustomModelIdChange: (value: string) => void;
}) {
  return (
    <Panel>
      <PanelHeader label="Configuration" title="Model and reasoning" description="Select the model, inference provider, and reasoning strategy." />
      <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="field-label" htmlFor="model-name">Model</label>
          <select
            id="model-name"
            value={formData.model_name}
            onChange={(event) => {
              const value = event.target.value;
              updateField("model_name", value);
              if (value === "custom_hosted" && formData.provider !== "custom") {
                updateField("provider", "custom");
              } else if (value !== "custom_hosted" && formData.provider === "custom") {
                updateField("provider", "auto");
              }
            }}
            className="select-shell"
          >
            <optgroup label="Hugging Face / serverless">
              {availableModels.map((model) => (
                <option key={model.value} value={model.value}>{model.label}</option>
              ))}
            </optgroup>
            <optgroup label="Custom">
              <option value="custom_hosted">Custom hosted model</option>
            </optgroup>
          </select>
          <p className="field-help mt-2">{formData.model_name === "custom_hosted" ? "Use this when pointing at a local or hosted OpenAI-compatible endpoint." : selectedModelDescription}</p>
        </div>
        <div>
          <label className="field-label" htmlFor="reasoning-method">Reasoning method</label>
          <select id="reasoning-method" value={formData.reasoning_method} onChange={(event) => updateField("reasoning_method", event.target.value as ExperimentBuilderFormData["reasoning_method"])} className="select-shell">
            {reasoningOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="field-help mt-2">{reasoningOptions.find((option) => option.value === formData.reasoning_method)?.description}</p>
        </div>
        <div>
          <label className="field-label" htmlFor="provider">Provider</label>
          <select
            id="provider"
            value={formData.provider}
            onChange={(event) => {
              const value = event.target.value as ExperimentBuilderFormData["provider"];
              updateField("provider", value);
              if (value === "custom" && formData.model_name !== "custom_hosted") {
                updateField("model_name", "custom_hosted");
              } else if (value !== "custom" && formData.model_name === "custom_hosted") {
                updateField("model_name", availableModels[0]?.value ?? "meta-llama/Llama-3.2-1B-Instruct");
              }
            }}
            className="select-shell"
          >
            {providerOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="field-help mt-2">{providerOptions.find((option) => option.value === formData.provider)?.description}</p>
        </div>
      </div>

      {formData.provider === "auto" ? (
        <div className="panel-body pt-0">
          <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <label className="field-label" htmlFor="routing-policy">Routing policy</label>
                <select id="routing-policy" value={formData.routing_policy} onChange={(event) => updateField("routing_policy", event.target.value as ExperimentBuilderFormData["routing_policy"])} className="select-shell">
                  {routingPolicyOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <p className="field-help mt-2">{routingPolicyOptions.find((option) => option.value === formData.routing_policy)?.description}</p>
              </div>
              <div>
                <label className="field-label" htmlFor="routing-epsilon">Adaptive epsilon</label>
                <input id="routing-epsilon" type="number" step="0.05" min="0" max="1" value={formData.routing_epsilon} onChange={(event) => updateField("routing_epsilon", parseFloat(event.target.value) || 0)} className="input-shell" />
                <p className="field-help mt-2">Only used by the adaptive policy.</p>
              </div>
              <div>
                <label className="field-label" htmlFor="routing-exploration-window">Exploration window</label>
                <input id="routing-exploration-window" type="number" min="1" max="50" value={formData.routing_exploration_window} onChange={(event) => updateField("routing_exploration_window", parseInt(event.target.value, 10) || 1)} className="input-shell" />
                <p className="field-help mt-2">Initial requests to round-robin before exploiting telemetry.</p>
              </div>
            </div>
            <label className="mt-4 flex items-start gap-3 rounded-[14px] border border-(--border) bg-(--surface-1) p-3 text-sm">
              <input
                type="checkbox"
                checked={formData.routing_strict_comparison}
                onChange={(event) => updateField("routing_strict_comparison", event.target.checked)}
                className="mt-1"
              />
              <span>
                <span className="block font-semibold">Strict comparison mode</span>
                <span className="mt-1 block text-(--muted-foreground)">
                  Pin the first available provider and disable cross-provider fallback for scored runs.
                </span>
              </span>
            </label>
          </div>
        </div>
      ) : null}

      {formData.model_name === "custom_hosted" ? (
        <div className="panel-body pt-0">
          <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <label className="field-label" htmlFor="custom-base-url">Base URL</label>
                <input id="custom-base-url" type="url" value={customBaseUrl} onChange={(event) => onCustomBaseUrlChange(event.target.value)} className="input-shell font-mono text-sm" placeholder="http://localhost:8000/v1" />
              </div>
              <div>
                <label className="field-label" htmlFor="custom-api-key">API key</label>
                <input id="custom-api-key" type="password" value={customApiKey} onChange={(event) => onCustomApiKeyChange(event.target.value)} className="input-shell font-mono text-sm" placeholder="sk-..." />
              </div>
              <div>
                <label className="field-label" htmlFor="custom-model-id">Model id</label>
                <input id="custom-model-id" type="text" value={customModelId} onChange={(event) => onCustomModelIdChange(event.target.value)} className="input-shell font-mono text-sm" placeholder="e.g. llama-3-8b" />
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

export function ExperimentEvaluationPanel({
  datasetGroups,
  formData,
  retrievalOptions,
  selectedDatasetDescription,
  updateField,
}: {
  datasetGroups: ExperimentBuilderDatasetGroup[];
  formData: ExperimentBuilderFormData;
  retrievalOptions: readonly ExperimentBuilderOption[];
  selectedDatasetDescription?: string;
  updateField: UpdateField;
}) {
  return (
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
          <p className="field-help mt-2">{selectedDatasetDescription}</p>
        </div>
        <div>
          <label className="field-label" htmlFor="samples">Samples</label>
          <input id="samples" type="number" min="1" max="500" value={formData.num_samples} onChange={(event) => updateField("num_samples", parseInt(event.target.value, 10) || 1)} className="input-shell" />
          <p className="field-help mt-2">Max 500 samples on the current API.</p>
        </div>
        <div>
          <label className="field-label" htmlFor="retrieval-method">Retrieval method</label>
          <select id="retrieval-method" value={formData.retrieval_method} onChange={(event) => updateField("retrieval_method", event.target.value as ExperimentBuilderFormData["retrieval_method"])} className="select-shell">
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
  );
}

export function ExperimentAgentPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
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
  );
}

export function ExperimentGenerationPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
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
  );
}

export function ExperimentOptimizationPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
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
              <input id="cache-size" type="number" min="16" max="2048" value={formData.cache_max_size} onChange={(event) => updateField("cache_max_size", parseInt(event.target.value, 10) || 16)} className="input-shell" />
            </div>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

export function ExperimentRegressionPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
    <Panel>
      <PanelHeader label="Regression" title="Regression gates" description="Attach thresholds so new attempts can be checked against a pinned baseline automatically." />
      <div className="panel-body space-y-4">
        <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
          <label className="flex items-center gap-3 font-medium">
            <input type="checkbox" checked={formData.enable_regression} onChange={(event) => updateField("enable_regression", event.target.checked)} />
            Enable regression gates
          </label>
          <p className="field-help mt-3">Use these thresholds when comparing this experiment against a pinned baseline.</p>
        </div>

        {formData.enable_regression ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <label className="field-label" htmlFor="reg-accuracy-delta">Accuracy minimum delta</label>
              <input id="reg-accuracy-delta" type="number" step="0.01" value={formData.regression_accuracy_min_delta} onChange={(event) => updateField("regression_accuracy_min_delta", parseFloat(event.target.value) || 0)} className="input-shell" />
            </div>
            <div>
              <label className="field-label" htmlFor="reg-f1-delta">F1 minimum delta</label>
              <input id="reg-f1-delta" type="number" step="0.01" value={formData.regression_f1_min_delta} onChange={(event) => updateField("regression_f1_min_delta", parseFloat(event.target.value) || 0)} className="input-shell" />
            </div>
            <div>
              <label className="field-label" htmlFor="reg-latency-p95">Latency p95 max (ms)</label>
              <input id="reg-latency-p95" type="number" min="0" value={formData.regression_latency_p95_max_ms} onChange={(event) => updateField("regression_latency_p95_max_ms", event.target.value === "" ? "" : parseFloat(event.target.value))} className="input-shell" placeholder="Optional" />
            </div>
            <div>
              <label className="field-label" htmlFor="reg-new-failures">Max new failures</label>
              <input id="reg-new-failures" type="number" min="0" value={formData.regression_max_new_failures} onChange={(event) => updateField("regression_max_new_failures", event.target.value === "" ? "" : parseInt(event.target.value, 10))} className="input-shell" placeholder="Optional" />
            </div>
            <div>
              <label className="field-label" htmlFor="reg-overlap">Minimum overlap ratio</label>
              <input id="reg-overlap" type="number" step="0.05" min="0" max="1" value={formData.regression_min_overlap_ratio} onChange={(event) => updateField("regression_min_overlap_ratio", parseFloat(event.target.value) || 0)} className="input-shell" />
            </div>
            <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
              <label className="flex items-center gap-3 font-medium">
                <input type="checkbox" checked={formData.regression_no_sample_regressions} onChange={(event) => updateField("regression_no_sample_regressions", event.target.checked)} />
                Fail on any sample regression
              </label>
              <p className="field-help mt-3">Treat any previously-correct sample becoming incorrect as a hard violation.</p>
            </div>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export function ExperimentAdvancedPanel({
  formData,
  updateField,
}: {
  formData: ExperimentBuilderFormData;
  updateField: UpdateField;
}) {
  return (
    <Panel>
      <PanelHeader label="Advanced" title="Prompt lineage and graders" description="Optional metadata and deterministic graders for power users." />
      <div className="panel-body grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="field-label" htmlFor="prompt-version-id">Prompt version ID</label>
          <input id="prompt-version-id" type="text" value={formData.prompt_version_id} onChange={(event) => updateField("prompt_version_id", event.target.value)} className="input-shell font-mono text-sm" placeholder="Optional UUID" />
          <p className="field-help mt-2">Attach this run to a saved prompt lineage if you are tracking prompt versions separately.</p>
        </div>
        <div>
          <label className="field-label" htmlFor="graders-json">Deterministic graders JSON</label>
          <textarea id="graders-json" value={formData.graders_json} onChange={(event) => updateField("graders_json", event.target.value)} className="textarea-shell font-mono text-sm" rows={8} placeholder={`{"rules":[{"name":"latency_budget","type":"latency_budget_ms","params":{"max":800}}]}`} />
          <p className="field-help mt-2">Paste a backend-compatible graders config when you want deterministic checks without leaving the UI.</p>
        </div>
      </div>
    </Panel>
  );
}

export function ExperimentBuilderSidebar({
  complexityScore,
  customModelId,
  formData,
  onApplyPreset,
  presetConfigs,
  selectedModelLabel,
}: {
  complexityScore: number;
  customModelId: string;
  formData: ExperimentBuilderFormData;
  onApplyPreset: (preset: ExperimentBuilderPreset) => void;
  presetConfigs: ExperimentBuilderPresetConfig[];
  selectedModelLabel?: string;
}) {
  return (
    <div className="space-y-4 xl:sticky xl:top-24 xl:self-start">
      <Panel>
        <PanelHeader label="Quick starts" title="Preset configurations" description="Quick-start templates for common evaluation scenarios." />
        <div className="panel-body space-y-3">
          {presetConfigs.map((preset) => (
            <button key={preset.title} type="button" className="w-full rounded-[18px] border border-(--border) bg-(--surface-2) p-4 text-left transition-all hover:border-(--border-strong) hover:bg-(--surface-3)" onClick={() => onApplyPreset(preset.apply)}>
              <div className="font-semibold tracking-[-0.03em]">{preset.title}</div>
              <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">{preset.description}</p>
            </button>
          ))}
        </div>
      </Panel>

      <Panel>
        <PanelHeader label="Summary" title={formData.name || "Untitled experiment"} description="Overview of the experiment configuration." />
        <div className="panel-body space-y-4">
          <div className="flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
            <span className="chip">{formData.reasoning_method.toUpperCase()}</span>
            <span className="chip">{formData.dataset_name}</span>
            <span className="chip">{formData.provider}</span>
          </div>
          <div className="space-y-3 text-sm text-(--muted-foreground)">
            <div className="flex items-center justify-between gap-3"><span>Model</span><span className="font-medium text-foreground">{formData.model_name === "custom_hosted" ? customModelId || "Custom hosted" : selectedModelLabel}</span></div>
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
  );
}
