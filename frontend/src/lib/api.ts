/**
 * API Client for LLM Research Platform Backend
 * 
 * Centralized HTTP client for all API calls.
 * Handles error handling and response parsing.
 */

export class ApiError extends Error {
    public statusCode: number;
    public requestId?: string;
    public details?: Record<string, unknown>[];
    public retryAfter?: number;

    constructor(message: string, statusCode: number, requestId?: string, details?: Record<string, unknown>[], retryAfter?: number) {
        super(message);
        this.name = 'ApiError';
        this.statusCode = statusCode;
        this.requestId = requestId;
        this.details = details;
        this.retryAfter = retryAfter;
    }
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// Warn loudly in production if the env var is not configured.
// This prevents silent failures where all API calls fallback to localhost.
if (process.env.NODE_ENV === 'production' && !process.env.NEXT_PUBLIC_API_URL) {
    console.error(
        '[LlmForge] NEXT_PUBLIC_API_URL is not set! ' +
        'All API calls will fall back to http://localhost:8000 which will fail in production. ' +
        'Set NEXT_PUBLIC_API_URL in your Vercel environment variables.'
    );
}

/**
 * Base fetch wrapper with error handling, timeout, and retry.
 */
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const TIMEOUT_MS = 15000;
    const MAX_RETRIES = 1; // Single retry for 5xx/network errors

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
            });

            clearTimeout(timeout);

            if (!response.ok) {
                let errorMessage = `API Error: ${response.status}`;
                let requestId: string | undefined;
                let details: Record<string, unknown>[] | undefined;
                let retryAfter: number | undefined;

                try {
                    const errorData = await response.json();
                    errorMessage = errorData.message || errorData.detail || errorMessage;
                    requestId = errorData.request_id || response.headers.get('X-Request-ID') || undefined;
                    details = errorData.details;
                    if (response.status === 429 && errorData.retry_after) {
                        retryAfter = errorData.retry_after;
                    }
                } catch {
                    requestId = response.headers.get('X-Request-ID') || undefined;
                }

                // Retry on 5xx (but not 429 or 4xx)
                if (response.status >= 500 && attempt < MAX_RETRIES) {
                    await new Promise(r => setTimeout(r, 2000)); // Wait 2s before retry
                    continue;
                }

                throw new ApiError(errorMessage, response.status, requestId, details, retryAfter);
            }

            // Handle 204 No Content
            if (response.status === 204) {
                return undefined as unknown as T;
            }

            return response.json();
        } catch (err) {
            clearTimeout(timeout);

            // If it's already an ApiError, rethrow
            if (err instanceof ApiError) throw err;

            // Retry on network/timeout errors
            if (attempt < MAX_RETRIES) {
                await new Promise(r => setTimeout(r, 2000));
                continue;
            }

            // AbortController timeout
            if (err instanceof DOMException && err.name === 'AbortError') {
                throw new ApiError('Request timed out. The server may be starting up — please try again.', 408);
            }

            // Network error
            throw new ApiError(
                err instanceof Error ? err.message : 'Network error',
                0
            );
        }
    }

    // Should never reach here, but TypeScript needs it
    throw new ApiError('Unexpected retry exhaustion', 500);
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
        semantic_similarity?: number;
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
    is_exact_match?: boolean;
    is_substring_match?: boolean;
    parsed_answer?: string;
    semantic_similarity?: number;
    latency_ms?: number;
    input_text: string;
    output_text?: string;
    expected_output?: string;
    faithfulness_score?: number;
    context_relevance_score?: number;
    attempt?: number;
    retrieved_chunks?: { chunks: { text?: string; page_content?: string; score?: number }[] };
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
export async function runExperiment(
    id: string,
    customBaseUrl?: string,
    customApiKey?: string
): Promise<Experiment> {
    const headers: Record<string, string> = {};
    if (customBaseUrl) headers['X-Custom-LLM-Base'] = customBaseUrl;
    if (customApiKey) headers['X-Custom-LLM-Key'] = customApiKey;

    return fetchAPI<Experiment>(`/experiments/${id}/run`, {
        method: 'POST',
        headers: Object.keys(headers).length > 0 ? headers : undefined
    });
}

/**
 * Delete an experiment (soft delete).
 */
export async function deleteExperiment(id: string): Promise<void> {
    await fetchAPI<undefined>(`/experiments/${id}`, { method: 'DELETE' });
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
export async function exportResults(experimentId: string, experimentName?: string): Promise<void> {
    const url = `${API_BASE_URL}/results/${experimentId}/export`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Export failed');

    const data = await response.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    // Use the experiment name in the filename if provided, fallback to ID
    const safeName = (experimentName || experimentId).replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
    a.download = `experiment_${safeName}_results.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(downloadUrl);
}

/**
 * Get dashboard statistics.
 */
export async function getDashboardStats(): Promise<DashboardStats> {
    const stats = await fetchAPI<{
        total: number;
        completed: number;
        running: number;
        pending: number;
        queued: number;
        failed: number;
    }>('/experiments/stats');

    return {
        totalExperiments: stats.total,
        completedExperiments: stats.completed,
        runningExperiments: stats.running + stats.queued,
        pendingExperiments: stats.pending,
    };
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
    // Use a separate base URL env var to avoid fragile string manipulation.
    // NEXT_PUBLIC_API_BASE_URL should be the root (e.g. https://myapp.hf.space)
    // NEXT_PUBLIC_API_URL should be the full API path (e.g. https://myapp.hf.space/api/v1)
    const baseUrl =
        process.env.NEXT_PUBLIC_API_BASE_URL ||
        API_BASE_URL.replace(/\/api\/v1\/?$/, '') ||
        'http://localhost:8000';
    const response = await fetch(`${baseUrl}/health`);
    return response.json();
}

export interface ReadinessStatus {
    status: string;
    checks: {
        database: string;
        vector_db: string;
        models: string;
    };
}

/**
 * Get system readiness status.
 */
export async function getReadinessStatus(): Promise<ReadinessStatus> {
    const baseUrl =
        process.env.NEXT_PUBLIC_API_BASE_URL ||
        API_BASE_URL.replace(/\/api\/v1\/?$/, '') ||
        'http://localhost:8000';
    const response = await fetch(`${baseUrl}/ready`);
    if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(
            `Readiness check failed (${response.status}): ${errorText}`
        );
    }
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
    accuracy_ci_a?: BootstrapCI;
    accuracy_ci_b?: BootstrapCI;
    f1_ci_a?: BootstrapCI;
    f1_ci_b?: BootstrapCI;
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

// =============================================================================
// LLM JUDGE & SYNTHETIC DATA (P2)
// =============================================================================

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

/**
 * Run LLM-as-judge evaluation on sampled runs (P2 #13).
 * Uses extended timeout since this calls external LLM APIs.
 */
export async function runLLMJudge(
    experimentId: string,
    sampleSize: number = 20,
): Promise<LLMJudgeResult> {
    const url = `${API_BASE_URL}/results/${experimentId}/judge?sample_size=${sampleSize}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000); // 2 min timeout

    try {
        const response = await fetch(url, {
            method: 'POST',
            signal: controller.signal,
            headers: { 'Content-Type': 'application/json' },
        });
        clearTimeout(timeout);
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new ApiError(err.detail || 'Judge evaluation failed', response.status);
        }
        return response.json();
    } catch (err) {
        clearTimeout(timeout);
        if (err instanceof ApiError) throw err;
        if (err instanceof DOMException && err.name === 'AbortError') {
            throw new ApiError('LLM Judge evaluation timed out (2 min limit)', 408);
        }
        throw new ApiError(err instanceof Error ? err.message : 'Network error', 0);
    }
}

/**
 * Generate synthetic QA pairs from knowledge base (P2 #14).
 * Uses extended timeout since this calls external LLM APIs.
 */
export async function generateSyntheticData(
    pairsPerChunk: number = 3,
    maxChunks: number = 10,
    seed?: number,
): Promise<SyntheticDataResult> {
    const params = new URLSearchParams({
        pairs_per_chunk: String(pairsPerChunk),
        max_chunks: String(maxChunks),
    });
    if (seed !== undefined) params.set('seed', String(seed));

    const url = `${API_BASE_URL}/results/synthetic/generate?${params}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000); // 2 min timeout

    try {
        const response = await fetch(url, {
            method: 'POST',
            signal: controller.signal,
            headers: { 'Content-Type': 'application/json' },
        });
        clearTimeout(timeout);
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new ApiError(err.detail || 'Synthetic data generation failed', response.status);
        }
        return response.json();
    } catch (err) {
        clearTimeout(timeout);
        if (err instanceof ApiError) throw err;
        if (err instanceof DOMException && err.name === 'AbortError') {
            throw new ApiError('Synthetic data generation timed out (2 min limit)', 408);
        }
        throw new ApiError(err instanceof Error ? err.message : 'Network error', 0);
    }
}

