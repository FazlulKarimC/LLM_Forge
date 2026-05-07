export type ExperimentBuilderFormData = {
  name: string;
  description: string;
  model_name: string;
  reasoning_method: "naive" | "cot" | "react";
  dataset_name: string;
  provider: "auto" | "hf_api" | "openrouter" | "groq" | "custom";
  temperature: number;
  max_tokens: number;
  num_samples: number;
  retrieval_method: "none" | "naive" | "hybrid" | "reranked";
  rag_top_k: number;
  agent_max_iterations: number;
  agent_tools: string[];
  enable_batching: boolean;
  batch_size: number;
  enable_caching: boolean;
  cache_max_size: number;
  routing_policy: "fallback_chain" | "cheapest_first" | "fastest_first" | "adaptive";
  routing_epsilon: number;
  routing_exploration_window: number;
  routing_strict_comparison: boolean;
  enable_regression: boolean;
  regression_accuracy_min_delta: number;
  regression_f1_min_delta: number;
  regression_latency_p95_max_ms: number | "";
  regression_no_sample_regressions: boolean;
  regression_max_new_failures: number | "";
  regression_min_overlap_ratio: number;
  prompt_version_id: string;
  graders_json: string;
  seed: number | "";
};

export type UpdateExperimentBuilderField = <K extends keyof ExperimentBuilderFormData>(
  key: K,
  value: ExperimentBuilderFormData[K],
) => void;

export type ExperimentBuilderPreset = {
  reasoning_method: ExperimentBuilderFormData["reasoning_method"];
  dataset_name: string;
  num_samples: number;
  retrieval_method: ExperimentBuilderFormData["retrieval_method"];
};

export type ExperimentBuilderPresetConfig = {
  title: string;
  description: string;
  apply: ExperimentBuilderPreset;
};

export type ExperimentBuilderOption = {
  value: string;
  label: string;
  description: string;
};

export type ExperimentBuilderDatasetGroup = {
  label: string;
  options: ExperimentBuilderOption[];
};
