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
 * Shared fetch wrapper with error handling, timeout, and retry.
 */
type FetchWithHandlingOptions = {
    timeoutMs?: number;
    maxRetries?: number;
};

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function buildApiError(response: Response): Promise<ApiError> {
    let errorMessage = `API Error: ${response.status}`;
    let requestId = response.headers.get('X-Request-ID') || undefined;
    let details: Record<string, unknown>[] | undefined;
    let retryAfter: number | undefined;

    const retryAfterHeader = response.headers.get('Retry-After');
    if (retryAfterHeader) {
        const parsedRetryAfter = Number(retryAfterHeader);
        if (!Number.isNaN(parsedRetryAfter)) {
            retryAfter = parsedRetryAfter;
        }
    }

    const contentType = response.headers.get('Content-Type') || '';
    try {
        if (contentType.includes('application/json')) {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.detail || errorMessage;
            requestId = errorData.request_id || requestId;
            details = errorData.details;
            if (response.status === 429 && errorData.retry_after) {
                retryAfter = errorData.retry_after;
            }
        } else {
            const errorText = (await response.text()).trim();
            if (errorText) {
                errorMessage = errorText;
            }
        }
    } catch {
        // Fall back to status code only.
    }

    return new ApiError(errorMessage, response.status, requestId, details, retryAfter);
}

async function fetchWithHandling(
    url: string,
    options: RequestInit = {},
    config: FetchWithHandlingOptions = {}
): Promise<Response> {
    const { timeoutMs = 15000, maxRetries = 1 } = config;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const headers = new Headers(options.headers);
            const hasBody = options.body !== undefined && options.body !== null;
            if (hasBody && !headers.has('Content-Type') && !(options.body instanceof FormData)) {
                headers.set('Content-Type', 'application/json');
            }

            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers,
            });

            clearTimeout(timeout);

            if (!response.ok) {
                const apiError = await buildApiError(response);
                if (response.status >= 500 && attempt < maxRetries) {
                    await sleep(2000);
                    continue;
                }
                throw apiError;
            }

            return response;
        } catch (err) {
            clearTimeout(timeout);

            if (err instanceof ApiError) {
                throw err;
            }

            if (attempt < maxRetries) {
                await sleep(2000);
                continue;
            }

            if (err instanceof DOMException && err.name === 'AbortError') {
                throw new ApiError('Request timed out. The backend may still be waking up on Hugging Face Spaces.', 408);
            }

            throw new ApiError(err instanceof Error ? err.message : 'Network error', 0);
        }
    }

    throw new ApiError('Unexpected retry exhaustion', 500);
}

/**
 * Base fetch wrapper with JSON parsing.
 */
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const response = await fetchWithHandling(`${API_BASE_URL}${endpoint}`, options);

    if (response.status === 204) {
        return undefined as unknown as T;
    }

    return response.json();
}

function getApiRootUrl(): string {
    return (
        process.env.NEXT_PUBLIC_API_BASE_URL ||
        API_BASE_URL.replace(/\/api\/v1\/?$/, '') ||
        'http://localhost:8000'
    );
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
    provider?: 'auto' | 'hf_api' | 'openrouter' | 'groq' | 'custom';
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
    summary_text?: string;
    quality: {
        accuracy_exact?: number;
        accuracy_f1?: number;
        accuracy_substring?: number;
        semantic_similarity?: number;
        faithfulness?: number;
        hallucination_rate?: number;
        safety_score?: number;
        pass_at_k?: number;
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
    const response = await fetchWithHandling(url, {}, { timeoutMs: 20000, maxRetries: 0 });
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
 * Export experiment results as a formatted Markdown report.
 */
export async function exportMarkdownReport(
    experimentId: string,
    experimentName?: string,
    metrics?: Metrics,
    runs?: RunSummary[],
): Promise<void> {
    const url = `${API_BASE_URL}/results/${experimentId}/export`;
    const response = await fetchWithHandling(url, {}, { timeoutMs: 20000, maxRetries: 0 });
    const data = await response.json();

    const name = experimentName || data.experiment?.name || experimentId;
    const config = data.experiment?.config || {};

    let md = `# Experiment Report: ${name}\n\n`;
    md += `**Generated**: ${new Date().toISOString()}\n\n`;
    md += `## Configuration\n\n`;
    md += `| Parameter | Value |\n|---|---|\n`;
    md += `| Model | ${config.model_name || 'N/A'} |\n`;
    md += `| Reasoning | ${config.reasoning_method || 'N/A'} |\n`;
    md += `| Dataset | ${config.dataset_name || 'N/A'} |\n`;
    md += `| Samples | ${config.num_samples || 'N/A'} |\n`;
    md += `| Temperature | ${config.hyperparameters?.temperature ?? 'N/A'} |\n`;
    md += `| Max Tokens | ${config.hyperparameters?.max_tokens ?? 'N/A'} |\n\n`;

    if (metrics) {
        md += `## Results\n\n`;
        if (metrics.summary_text) {
            md += `> ${metrics.summary_text}\n\n`;
        }
        md += `| Metric | Value |\n|---|---|\n`;
        md += `| Accuracy (Substring) | ${((metrics.quality.accuracy_substring ?? 0) * 100).toFixed(1)}% |\n`;
        md += `| F1 Score (Mean) | ${((metrics.quality.accuracy_f1 ?? 0) * 100).toFixed(1)}% |\n`;
        md += `| Latency p50 | ${(metrics.performance.latency_p50 ?? 0).toFixed(0)} ms |\n`;
        md += `| Latency p95 | ${(metrics.performance.latency_p95 ?? 0).toFixed(0)} ms |\n`;
        md += `| Throughput | ${(metrics.performance.throughput ?? 0).toFixed(1)} prompts/s |\n`;
        if (metrics.cost.total_cost_usd != null) {
            md += `| Total Cost | $${metrics.cost.total_cost_usd.toFixed(4)} |\n`;
        }
        md += `| Total Tokens | ${(metrics.cost.total_tokens_input + metrics.cost.total_tokens_output).toLocaleString()} |\n\n`;
    }

    if (runs && runs.length > 0) {
        const correct = runs.filter(r => r.is_correct).length;
        md += `## Per-Run Results (${correct}/${runs.length} correct)\n\n`;
        md += `| # | Example ID | Correct | F1 | Latency |\n|---|---|---|---|---|\n`;
        runs.slice(0, 50).forEach((r, i) => {
            md += `| ${i + 1} | ${r.example_id} | ${r.is_correct ? '✓' : '✗'} | ${(r.score ?? 0).toFixed(3)} | ${(r.latency_ms ?? 0).toFixed(0)}ms |\n`;
        });
        if (runs.length > 50) md += `\n*...and ${runs.length - 50} more runs*\n`;
    }

    const blob = new Blob([md], { type: 'text/markdown' });
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    const safeName = (experimentName || experimentId).replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
    a.download = `experiment_${safeName}_report.md`;
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
    const response = await fetchWithHandling(`${getApiRootUrl()}/health`, {}, {
        timeoutMs: 20000,
        maxRetries: 0,
    });
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
    const response = await fetchWithHandling(`${getApiRootUrl()}/ready`, {}, {
        timeoutMs: 20000,
        maxRetries: 0,
    });
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

