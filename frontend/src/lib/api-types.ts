export interface HyperParameters {
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    top_k?: number;
    seed?: number;
}

export interface RAGConfig {
    retrieval_method: 'none' | 'naive' | 'hybrid' | 'reranked';
    top_k?: number;
    chunk_size?: number;
}

export interface AgentConfig {
    max_iterations?: number;
    tools?: string[];
}

export interface OptimizationConfig {
    enable_batching?: boolean;
    batch_size?: number;
    enable_caching?: boolean;
    cache_max_size?: number;
    enable_profiling?: boolean;
}

export interface GraderRule {
    name: string;
    type: 'max_turns' | 'required_tools' | 'forbidden_failure_modes' | 'must_use_retrieval_when_rag' | 'latency_budget_ms' | 'token_budget' | 'min_f1_score' | 'expected_tools';
    params?: Record<string, unknown>;
}

export interface GradersConfig {
    rules?: GraderRule[];
    llm_judge_on_failures?: boolean;
}

export interface RegressionConfig {
    accuracy_min_delta?: number;
    f1_min_delta?: number;
    latency_p95_max_ms?: number;
    no_sample_regressions?: boolean;
    max_new_failures?: number;
    min_overlap_ratio?: number;
}

export interface RoutingConfig {
    policy?: 'fallback_chain' | 'cheapest_first' | 'fastest_first' | 'adaptive';
    epsilon?: number;
    exploration_window?: number;
    strict_comparison?: boolean;
}

export type RegressionStatus = 'not_checked' | 'pass' | 'fail' | 'inconclusive';

export interface ExperimentConfig {
    model_name: string;
    reasoning_method: 'naive' | 'cot' | 'react';
    dataset_name: string;
    provider?: 'auto' | 'hf_api' | 'openrouter' | 'groq' | 'custom';
    hyperparameters?: HyperParameters;
    rag?: RAGConfig;
    agent?: AgentConfig;
    optimization?: OptimizationConfig;
    graders?: GradersConfig;
    regression?: RegressionConfig;
    routing?: RoutingConfig;
    prompt_version_id?: string;
    num_samples?: number;
}

export interface Experiment {
    id: string;
    name: string;
    description?: string;
    config: ExperimentConfig;
    status: 'pending' | 'queued' | 'running' | 'completed' | 'failed';
    created_at: string;
    started_at?: string;
    completed_at?: string;
    error_message?: string;
    tags?: string[];
    run_manifest?: Record<string, unknown>;
    is_baseline?: boolean;
    regression_status?: RegressionStatus;
    regression_passed?: boolean | null;
}

export interface ExperimentList {
    total: number;
    experiments: Experiment[];
    skip: number;
    limit: number;
}

export interface ExperimentListItem {
    id: string;
    name: string;
    description?: string;
    status: 'pending' | 'queued' | 'running' | 'completed' | 'failed';
    created_at: string;
    completed_at?: string;
    is_baseline?: boolean;
    regression_status?: RegressionStatus;
    provider?: 'auto' | 'hf_api' | 'openrouter' | 'groq' | 'custom';
    reasoning_method: string;
    model_name: string;
    dataset_name: string;
    num_samples?: number;
}

export interface ExperimentSlimList {
    total: number;
    experiments: ExperimentListItem[];
    skip: number;
    limit: number;
}

export interface CreateExperimentRequest {
    name: string;
    description?: string;
    config: ExperimentConfig;
}

export interface ListExperimentsParams {
    status?: string;
    method?: string;
    model?: string;
    skip?: number;
    limit?: number;
}

export interface Metrics {
    experiment_id: string;
    summary_text?: string;
    quality: {
        accuracy_exact?: number;
        accuracy_f1?: number;
        accuracy_substring?: number;
        semantic_similarity?: number;
        faithfulness?: number;
        hallucination_rate?: number;
        robustness_safety_score?: number;
        robustness_inconclusive_rate?: number;
        safety_score?: number;
        pass_at_k?: number;
        completion_quality?: { label: string; failure_rate: number; total_failures: number; total_runs: number };
        retrieval_quality?: { recall_at_k?: number; evidence_hit_rate?: number; total_evaluated?: number; k?: number };
        accuracy_excluding_failures?: number;
        total_excluding_failures?: number;
    };
    performance: {
        latency_p50?: number;
        latency_p95?: number;
        latency_p99?: number;
        throughput?: number;
    };
    cost: {
        total_tokens_input: number;
        total_tokens_output: number;
        total_runs: number;
        gpu_time_seconds?: number;
        total_cost_usd?: number;
        cost_per_correct_answer?: number;
        provider?: string;
        cost_source?: string;
        cost_per_sample_usd?: number;
        accuracy_per_dollar?: number;
    };
    failure_modes?: {
        counts: Record<string, number>;
        total_failures: number;
        sample_errors: Array<{ mode: string; error: string; example_id: string }>;
    };
    computed_at: string;
}

export interface AgentStep {
    thought: string;
    action?: string;
    action_input?: string;
    observation?: string;
}

export interface AgentTrace {
    steps: AgentStep[];
    total_tool_calls: number;
    successful_tool_calls: number;
    failed_tool_calls: number;
    success?: boolean;
    termination_reason?: string;
    total_iterations?: number;
}

export interface RunSummary {
    id: string;
    example_id?: string;
    is_correct?: boolean;
    score?: number;
    is_exact_match?: boolean;
    is_substring_match?: boolean;
    parsed_answer?: string;
    semantic_similarity?: number;
    latency_ms?: number;
    prompt: string;
    raw_output?: string;
    expected_output?: string;
    faithfulness_score?: number;
    context_relevance_score?: number;
    attempt?: number;
    retrieved_chunks?: { chunks: { text?: string; page_content?: string; score?: number }[] };
    failure_mode?: string;
    error_message?: string;
    agent_trace?: AgentTrace;
    served_provider?: string;
    routing_reason?: string;
    cost_usd?: number;
    grader_results?: Record<string, Record<string, unknown>>;
}

export interface RunGridSummary {
    id: string;
    example_id?: string;
    is_correct?: boolean;
    score?: number;
    latency_ms?: number;
    failure_mode?: string;
    served_provider?: string;
    routing_reason?: string;
    cost_usd?: number;
    grader_results?: Record<string, Record<string, unknown>>;
}

export interface ModelOption {
    value: string;
    label: string;
    description: string;
}

export interface DashboardStats {
    totalExperiments: number;
    completedExperiments: number;
    runningExperiments: number;
    pendingExperiments: number;
}

export interface ProfilingSectionStats {
    count: number;
    total_ms: number;
    mean_ms: number;
    p50_ms: number;
    p95_ms: number;
}

export interface ProfileData {
    experiment_id: string;
    message?: string;
    profiling_summary: Record<string, ProfilingSectionStats>;
    cache_stats: {
        hits?: number;
        misses?: number;
        hit_rate?: number;
        size?: number;
        max_size?: number;
        total_latency_saved_ms?: number;
    };
    batch_stats: {
        batches_processed?: number;
        total_prompts_batched?: number;
    };
    total_wall_time_ms?: number;
}

export interface ReadinessStatus {
    status: string;
    checks: Record<string, string>;
}

export interface ExperimentComparison {
    experiment_id: string;
    experiment_name: string;
    method: string;
    model: string;
    metrics: Metrics;
}

export interface ComparisonResponse {
    experiments: ExperimentComparison[];
    comparison_metrics: {
        accuracy_exact: number[];
        accuracy_f1: number[];
        latency_p50: number[];
    };
}

export interface BootstrapCI {
    mean: number;
    lower: number;
    upper: number;
    std: number;
}

export interface McNemarResult {
    statistic: number;
    p_value: number;
    is_significant: boolean;
    b: number;
    c: number;
    n: number;
    test_type?: string;
}

export interface PerExampleDiff {
    example_id: string;
    a_correct: boolean;
    b_correct: boolean;
    a_output: string;
    b_output: string;
    expected: string;
    a_score: number;
    b_score: number;
}

export interface StatisticalComparison {
    experiment_a_id: string;
    experiment_b_id: string;
    num_common_examples: number;
    overlap_ratio: number;
    total_examples_a?: number;
    total_examples_b?: number;
    accuracy_a: number;
    accuracy_b: number;
    accuracy_diff: number;
    mcnemar: McNemarResult;
    bootstrap_ci_a: BootstrapCI;
    bootstrap_ci_b: BootstrapCI;
    accuracy_ci_a?: BootstrapCI;
    accuracy_ci_b?: BootstrapCI;
    f1_ci_a?: BootstrapCI;
    f1_ci_b?: BootstrapCI;
    warnings?: string[];
    methodology_notes?: string[];
    routing?: {
        providers_a: string[];
        providers_b: string[];
        routing_reasons_a: string[];
        routing_reasons_b: string[];
    };
    per_example_differences: PerExampleDiff[];
    summary: {
        both_correct: number;
        both_wrong: number;
        a_only_correct: number;
        b_only_correct: number;
    };
}

export interface LLMJudgeResult {
    model_judge: string;
    sample_size: number;
    evaluated: number;
    attempt: number;
    scores: Record<string, {
        mean: number;
        median: number;
        min: number;
        max: number;
        std: number;
        count: number;
    }>;
    method: string;
    budget_cap: number;
    error?: string;
}

export interface SyntheticDataResult {
    pairs: Array<{
        id: string;
        question: string;
        answer: string;
        source_chunk_index: number;
        source_text: string;
    }>;
    total_generated: number;
    chunks_processed: number;
    errors: number;
    model: string;
    method: string;
}

export interface RegressionReport {
    passed: boolean | null;
    baseline_id: string;
    baseline_attempt: number;
    candidate_attempt: number;
    overlap_ratio: number;
    violations: Array<{ rule: string; message: string; actual?: unknown; threshold?: unknown }>;
    sample_regressions_count: number;
    sample_improvements_count: number;
    grader_summary: {
        baseline?: Record<string, { pass: number; fail: number; skip: number }>;
        candidate?: Record<string, { pass: number; fail: number; skip: number }>;
    };
    statistical: {
        per_example_differences?: PerExampleDiff[];
        [key: string]: unknown;
    };
    config_diff: Record<string, unknown>;
}

export interface ProviderStats {
    total_requests: number;
    total_errors: number;
    error_rate: number;
    mean_latency_ms: number | null;
    p95_latency_ms: number | null;
    total_tokens: number;
    total_cost_usd: number;
    cost_per_request: number;
}
