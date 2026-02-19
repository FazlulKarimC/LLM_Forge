/**
 * API Client for LLM Research Platform Backend
 * 
 * Centralized HTTP client for all API calls.
 * Handles error handling and response parsing.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

/**
 * Base fetch wrapper with error handling.
 */
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;

    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `API Error: ${response.status}`);
    }

    // Handle 204 No Content
    if (response.status === 204) {
        return undefined as T;
    }

    return response.json();
}

// =============================================================================
// TYPES
// =============================================================================

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

export interface ExperimentConfig {
    model_name: string;
    reasoning_method: 'naive' | 'cot' | 'react';
    dataset_name: string;
    hyperparameters?: HyperParameters;
    rag?: RAGConfig;
    agent?: AgentConfig;
    optimization?: OptimizationConfig;
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
}

export interface ExperimentList {
    total: number;
    experiments: Experiment[];
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
    quality: {
        accuracy_exact?: number;
        accuracy_f1?: number;
        accuracy_substring?: number;
        faithfulness?: number;
        hallucination_rate?: number;
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
    };
    computed_at: string;
}

export interface RunSummary {
    id: string;
    example_id?: string;
    is_correct?: boolean;
    score?: number;
    latency_ms?: number;
    input_text: string;
    output_text?: string;
    expected_output?: string;
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

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Create a new experiment.
 */
export async function createExperiment(data: CreateExperimentRequest): Promise<Experiment> {
    return fetchAPI<Experiment>('/experiments', {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

/**
 * List experiments with optional filters.
 */
export async function listExperiments(params?: ListExperimentsParams): Promise<ExperimentList> {
    const searchParams = new URLSearchParams();

    if (params?.status) searchParams.set('status', params.status);
    if (params?.method) searchParams.set('method', params.method);
    if (params?.model) searchParams.set('model', params.model);
    if (params?.skip !== undefined) searchParams.set('skip', String(params.skip));
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));

    const query = searchParams.toString();
    return fetchAPI<ExperimentList>(`/experiments${query ? `?${query}` : ''}`);
}

/**
 * Get experiment by ID.
 */
export async function getExperiment(id: string): Promise<Experiment> {
    return fetchAPI<Experiment>(`/experiments/${id}`);
}

/**
 * Run an experiment (trigger execution).
 */
export async function runExperiment(id: string): Promise<Experiment> {
    return fetchAPI<Experiment>(`/experiments/${id}/run`, { method: 'POST' });
}

/**
 * Delete an experiment (soft delete).
 */
export async function deleteExperiment(id: string): Promise<void> {
    return fetchAPI<void>(`/experiments/${id}`, { method: 'DELETE' });
}

/**
 * Get metrics for an experiment.
 */
export async function getMetrics(experimentId: string): Promise<Metrics> {
    return fetchAPI<Metrics>(`/results/${experimentId}/metrics`);
}

/**
 * Get run summaries for an experiment (for correctness grid).
 */
export async function getRunSummaries(experimentId: string): Promise<RunSummary[]> {
    return fetchAPI<RunSummary[]>(`/results/${experimentId}/runs`);
}

/**
 * Get optimization profiling data for an experiment.
 */
export async function getProfile(experimentId: string): Promise<ProfileData> {
    return fetchAPI<ProfileData>(`/results/${experimentId}/profile`);
}

/**
 * Export results as JSON download.
 */
export async function exportResults(experimentId: string): Promise<void> {
    const url = `${API_BASE_URL}/results/${experimentId}/export`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Export failed');

    const data = await response.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `experiment_results.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(downloadUrl);
}

/**
 * Get dashboard statistics.
 */
export async function getDashboardStats(): Promise<DashboardStats> {
    // Fetch all experiments and compute stats client-side
    // TODO: Add dedicated backend endpoint for efficiency
    const result = await listExperiments({ limit: 100 });

    const stats: DashboardStats = {
        totalExperiments: result.total,
        completedExperiments: 0,
        runningExperiments: 0,
        pendingExperiments: 0,
    };

    for (const exp of result.experiments) {
        if (exp.status === 'completed') stats.completedExperiments++;
        else if (exp.status === 'running') stats.runningExperiments++;
        else if (exp.status === 'pending') stats.pendingExperiments++;
    }

    return stats;
}

/**
 * Get available models for experiment creation.
 */
export async function getAvailableModels(): Promise<{ models: ModelOption[] }> {
    return fetchAPI<{ models: ModelOption[] }>('/experiments/models');
}

/**
 * Health check.
 */
export async function healthCheck(): Promise<{ status: string }> {
    // Health endpoint is at root, not under /api/v1
    const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';
    const response = await fetch(`${baseUrl}/health`);
    return response.json();
}

// =============================================================================
// COMPARISON TYPES
// =============================================================================

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
    accuracy_a: number;
    accuracy_b: number;
    accuracy_diff: number;
    mcnemar: McNemarResult;
    bootstrap_ci_a: BootstrapCI;
    bootstrap_ci_b: BootstrapCI;
    per_example_differences: PerExampleDiff[];
    summary: {
        both_correct: number;
        both_wrong: number;
        a_only_correct: number;
        b_only_correct: number;
    };
}

// =============================================================================
// COMPARISON API FUNCTIONS
// =============================================================================

/**
 * Compare metrics across multiple experiments.
 */
export async function compareExperiments(ids: string[]): Promise<ComparisonResponse> {
    const params = ids.map(id => `experiment_ids=${id}`).join('&');
    return fetchAPI<ComparisonResponse>(`/results/compare?${params}`);
}

/**
 * Get statistical comparison between two experiments.
 */
export async function getStatisticalComparison(
    experimentA: string,
    experimentB: string,
): Promise<StatisticalComparison> {
    return fetchAPI<StatisticalComparison>(
        `/results/compare/statistical?experiment_a=${experimentA}&experiment_b=${experimentB}`
    );
}

