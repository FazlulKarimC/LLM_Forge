/**
 * API Client for LlmForge backend.
 *
 * Centralized HTTP client for all API calls.
 * Handles error handling and response parsing.
 */

import * as Sentry from "@sentry/nextjs";

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

const SAFE_METHOD_TIMEOUT_MS = 10000;
const UNSAFE_METHOD_TIMEOUT_MS = 15000;
const SAFE_METHOD_MAX_RETRIES = 1;
const UNSAFE_METHOD_MAX_RETRIES = 0;
const RETRY_BACKOFF_BASE_MS = 2000;
const BACKGROUND_JOB_POLL_INTERVAL_MS = 2500;
const BACKGROUND_JOB_TIMEOUT_MS = 5 * 60_000;
const CUSTOM_LLM_SETTINGS_KEY = "customLLMSettings";
const CUSTOM_BASE_URL_KEY = "customBaseUrl";
const CUSTOM_API_KEY_KEY = "customApiKey";
const CUSTOM_MODEL_ID_KEY = "customModelId";

function isSafeRetryMethod(method?: string): boolean {
    const normalizedMethod = (method || 'GET').toUpperCase();
    return normalizedMethod === 'GET' || normalizedMethod === 'HEAD';
}

function getAbortError(): DOMException {
    return new DOMException('The operation was aborted.', 'AbortError');
}

function getCustomProviderStorage(): Storage | null {
    if (typeof window === "undefined") {
        return null;
    }
    return window.sessionStorage;
}

function sleep(ms: number, signal?: AbortSignal | null): Promise<void> {
    return new Promise((resolve, reject) => {
        const abortSignal = signal ?? undefined;

        if (abortSignal?.aborted) {
            reject(abortSignal.reason ?? getAbortError());
            return;
        }

        const onAbort = () => {
            clearTimeout(timeout);
            abortSignal?.removeEventListener('abort', onAbort);
            reject(abortSignal?.reason ?? getAbortError());
        };

        const timeout = setTimeout(() => {
            abortSignal?.removeEventListener('abort', onAbort);
            resolve();
        }, ms);

        abortSignal?.addEventListener('abort', onAbort, { once: true });
    });
}

function combineAbortSignals(timeoutSignal: AbortSignal, externalSignal?: AbortSignal | null): AbortSignal {
    if (!externalSignal) {
        return timeoutSignal;
    }

    const abortSignalWithAny = AbortSignal as typeof AbortSignal & {
        any?: (signals: AbortSignal[]) => AbortSignal;
    };

    if (typeof abortSignalWithAny.any === 'function') {
        return abortSignalWithAny.any([timeoutSignal, externalSignal]);
    }

    const controller = new AbortController();
    const abortFrom = (signal: AbortSignal) => {
        if (!controller.signal.aborted) {
            controller.abort(signal.reason);
        }
    };

    if (timeoutSignal.aborted) abortFrom(timeoutSignal);
    if (externalSignal.aborted) abortFrom(externalSignal);

    timeoutSignal.addEventListener('abort', () => abortFrom(timeoutSignal), { once: true });
    externalSignal.addEventListener('abort', () => abortFrom(externalSignal), { once: true });

    return controller.signal;
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
    const safeMethod = isSafeRetryMethod(options.method);
    const {
        timeoutMs = safeMethod ? SAFE_METHOD_TIMEOUT_MS : UNSAFE_METHOD_TIMEOUT_MS,
        maxRetries = safeMethod ? SAFE_METHOD_MAX_RETRIES : UNSAFE_METHOD_MAX_RETRIES,
    } = config;
    const externalSignal = options.signal;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const start = Date.now();
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const combinedSignal = combineAbortSignals(controller.signal, externalSignal);

        try {
            if (externalSignal?.aborted) {
                throw externalSignal.reason ?? getAbortError();
            }

            const headers = new Headers(options.headers);
            const hasBody = options.body !== undefined && options.body !== null;
            if (hasBody && !headers.has('Content-Type') && !(options.body instanceof FormData)) {
                headers.set('Content-Type', 'application/json');
            }

            const response = await fetch(url, {
                ...options,
                signal: combinedSignal,
                headers,
            });

            clearTimeout(timeout);

            if (!response.ok) {
                const apiError = await buildApiError(response);
                if (safeMethod && response.status >= 500 && attempt < maxRetries) {
                    await sleep(RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt), externalSignal);
                    continue;
                }
                throw apiError;
            }

            return response;
        } catch (err) {
            clearTimeout(timeout);

            if (err instanceof ApiError) {
                Sentry.addBreadcrumb({
                    category: 'api',
                    message: `${options.method ?? 'GET'} ${url}`,
                    level: 'error',
                    data: {
                        status: err.statusCode,
                        duration_ms: Date.now() - start,
                        request_id: err.requestId,
                    },
                });
                throw err;
            }

            if (externalSignal?.aborted && !controller.signal.aborted) {
                throw err;
            }

            if (safeMethod && attempt < maxRetries) {
                await sleep(RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt), externalSignal);
                continue;
            }

            if (err instanceof DOMException && err.name === 'AbortError') {
                if (controller.signal.aborted && !externalSignal?.aborted) {
                    throw new ApiError('Request timed out. The backend may still be waking up on Hugging Face Spaces.', 408);
                }
                throw err;
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
    options: RequestInit = {},
    config: FetchWithHandlingOptions = {}
): Promise<T> {
    const response = await fetchWithHandling(`${API_BASE_URL}${endpoint}`, options, config);

    if (response.status === 204) {
        return undefined as unknown as T;
    }

    return response.json();
}

function getApiRootUrl(): string {
    return API_BASE_URL.replace(/\/api\/v1\/?$/, "") || "http://localhost:8000";
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

export interface GraderRule {
    name: string;
    type: 'max_turns' | 'required_tools' | 'forbidden_failure_modes' | 'must_use_retrieval_when_rag' | 'latency_budget_ms' | 'token_budget' | 'min_f1_score';
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

/** Slim experiment item for list/catalog views (no full config or run_manifest). */
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
    served_provider?: string;
    grader_results?: Record<string, {status: 'pass' | 'fail' | 'skip'; value?: unknown; threshold?: unknown; reason?: string}>;
}

/** Sparse per-run data for filmstrip/grid views (no prompt/output text). */
export interface RunGridSummary {
    id: string;
    example_id?: string;
    is_correct?: boolean;
    score?: number;
    latency_ms?: number;
    failure_mode?: string;
    served_provider?: string;
    grader_results?: Record<string, {status: 'pass' | 'fail' | 'skip'; value?: unknown; threshold?: unknown; reason?: string}>;
}

export function getStoredCustomProviderCredentials(modelName?: string): {
    customBaseUrl?: string;
    customApiKey?: string;
} {
    const storage = getCustomProviderStorage();
    if (!storage) {
        return {};
    }

    try {
        const settings = JSON.parse(storage.getItem(CUSTOM_LLM_SETTINGS_KEY) || '{}') as Record<
            string,
            { baseUrl?: string; apiKey?: string }
        >;

        if (modelName && settings?.[modelName]?.baseUrl) {
            return {
                customBaseUrl: settings[modelName].baseUrl,
                customApiKey: settings[modelName].apiKey,
            };
        }

        if (modelName && storage.getItem(CUSTOM_MODEL_ID_KEY) === modelName) {
            return {
                customBaseUrl: storage.getItem(CUSTOM_BASE_URL_KEY) || undefined,
                customApiKey: storage.getItem(CUSTOM_API_KEY_KEY) || undefined,
            };
        }
    } catch (error) {
        console.error('Failed to load custom model settings', error);
    }

    return {};
}

export function persistCustomProviderCredentials(
    modelName: string,
    customBaseUrl: string,
    customApiKey: string,
): void {
    const storage = getCustomProviderStorage();
    if (!storage || !modelName.trim()) {
        return;
    }

    try {
        const settings = JSON.parse(storage.getItem(CUSTOM_LLM_SETTINGS_KEY) || "{}") as Record<
            string,
            { baseUrl?: string; apiKey?: string }
        >;
        settings[modelName] = { baseUrl: customBaseUrl, apiKey: customApiKey };
        storage.setItem(CUSTOM_LLM_SETTINGS_KEY, JSON.stringify(settings));
        storage.setItem(CUSTOM_BASE_URL_KEY, customBaseUrl);
        storage.setItem(CUSTOM_API_KEY_KEY, customApiKey);
        storage.setItem(CUSTOM_MODEL_ID_KEY, modelName);
    } catch (error) {
        console.error("Failed to persist custom model settings", error);
    }
}

export function resolveRunExperimentCredentials(config?: Pick<ExperimentConfig, 'provider' | 'model_name'>): {
    customBaseUrl?: string;
    customApiKey?: string;
} {
    const credentials = getStoredCustomProviderCredentials(config?.model_name);

    if (config?.provider === 'custom' && !credentials.customBaseUrl) {
        throw new Error(
            `No saved custom endpoint settings were found for model "${config.model_name}". Update the custom provider settings and try again.`
        );
    }

    return credentials;
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

type BackgroundJobStatus = 'queued' | 'running' | 'completed' | 'failed';

interface BackgroundJob<T> {
    job_id: string;
    kind: string;
    status: BackgroundJobStatus;
    created_at: string;
    updated_at: string;
    metadata?: Record<string, unknown>;
    result?: T;
    error?: string;
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
export async function listExperiments(params?: ListExperimentsParams, options: RequestInit = {}): Promise<ExperimentList> {
    const searchParams = new URLSearchParams();

    if (params?.status) searchParams.set('status', params.status);
    if (params?.method) searchParams.set('method', params.method);
    if (params?.model) searchParams.set('model', params.model);
    if (params?.skip !== undefined) searchParams.set('skip', String(params.skip));
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));

    const query = searchParams.toString();
    return fetchAPI<ExperimentList>(`/experiments${query ? `?${query}` : ''}`, options);
}

/**
 * List experiments with slim payloads (no full config or run_manifest).
 */
export async function listExperimentsSlim(params?: ListExperimentsParams, options: RequestInit = {}): Promise<ExperimentSlimList> {
    const searchParams = new URLSearchParams();
    searchParams.set('slim', 'true');

    if (params?.status) searchParams.set('status', params.status);
    if (params?.method) searchParams.set('method', params.method);
    if (params?.model) searchParams.set('model', params.model);
    if (params?.skip !== undefined) searchParams.set('skip', String(params.skip));
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));

    const query = searchParams.toString();
    return fetchAPI<ExperimentSlimList>(`/experiments?${query}`, options);
}

/**
 * Get experiment by ID.
 */
export async function getExperiment(id: string, options: RequestInit = {}): Promise<Experiment> {
    return fetchAPI<Experiment>(`/experiments/${id}`, options);
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
export async function getMetrics(experimentId: string, options: RequestInit = {}): Promise<Metrics> {
    return fetchAPI<Metrics>(`/results/${experimentId}/metrics`, options);
}

/**
 * Get run summaries for an experiment (for correctness grid).
 */
export async function getRunSummaries(experimentId: string, options: RequestInit = {}): Promise<RunSummary[]> {
    return fetchAPI<RunSummary[]>(`/results/${experimentId}/runs`, options);
}

/**
 * Get sparse run grid summaries (no prompt/output text — for filmstrip).
 */
export async function getRunGridSummaries(experimentId: string, options: RequestInit = {}): Promise<RunGridSummary[]> {
    return fetchAPI<RunGridSummary[]>(`/results/${experimentId}/runs?sparse=true`, options);
}

/**
 * Get optimization profiling data for an experiment.
 */
export async function getProfile(experimentId: string, options: RequestInit = {}): Promise<ProfileData> {
    return fetchAPI<ProfileData>(`/results/${experimentId}/profile`, options);
}

/**
 * Export results as JSON download.
 */
export async function exportResults(experimentId: string, experimentName?: string): Promise<void> {
    const url = `${API_BASE_URL}/results/${experimentId}/export`;
    const response = await fetchWithHandling(url, {}, { timeoutMs: SAFE_METHOD_TIMEOUT_MS, maxRetries: SAFE_METHOD_MAX_RETRIES });
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
    runs?: RunGridSummary[],
): Promise<void> {
    const url = `${API_BASE_URL}/results/${experimentId}/export`;
    const response = await fetchWithHandling(url, {}, { timeoutMs: SAFE_METHOD_TIMEOUT_MS, maxRetries: SAFE_METHOD_MAX_RETRIES });
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
export async function getDashboardStats(options: RequestInit = {}): Promise<DashboardStats> {
    const stats = await fetchAPI<{
        total: number;
        completed: number;
        running: number;
        pending: number;
        queued: number;
        failed: number;
    }>('/experiments/stats', options);

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
export async function getAvailableModels(options: RequestInit = {}): Promise<{ models: ModelOption[] }> {
    return fetchAPI<{ models: ModelOption[] }>('/experiments/models', options);
}

/**
 * Health check.
 */
export async function healthCheck(options: RequestInit = {}): Promise<{ status: string }> {
    const response = await fetchWithHandling(`${getApiRootUrl()}/health`, options, {
        timeoutMs: SAFE_METHOD_TIMEOUT_MS,
        maxRetries: SAFE_METHOD_MAX_RETRIES,
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
export async function getReadinessStatus(options: RequestInit = {}): Promise<ReadinessStatus> {
    const response = await fetchWithHandling(`${getApiRootUrl()}/ready`, options, {
        timeoutMs: SAFE_METHOD_TIMEOUT_MS,
        maxRetries: SAFE_METHOD_MAX_RETRIES,
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
export async function compareExperiments(ids: string[], options: RequestInit = {}): Promise<ComparisonResponse> {
    const params = ids.map(id => `experiment_ids=${id}`).join('&');
    return fetchAPI<ComparisonResponse>(`/results/compare?${params}`, options);
}

/**
 * Get statistical comparison between two experiments.
 */
export async function getStatisticalComparison(
    experimentA: string,
    experimentB: string,
    options: RequestInit = {},
): Promise<StatisticalComparison> {
    return fetchAPI<StatisticalComparison>(
        `/results/compare/statistical?experiment_a=${experimentA}&experiment_b=${experimentB}`,
        options,
    );
}

async function waitForBackgroundJob<T>(jobId: string, signal?: AbortSignal | null): Promise<T> {
    const deadline = Date.now() + BACKGROUND_JOB_TIMEOUT_MS;

    while (Date.now() < deadline) {
        const response = await fetchWithHandling(
            `${API_BASE_URL}/results/jobs/${jobId}`,
            { signal },
            { timeoutMs: SAFE_METHOD_TIMEOUT_MS, maxRetries: SAFE_METHOD_MAX_RETRIES },
        );
        const job = await response.json() as BackgroundJob<T>;

        if (job.status === 'completed' && job.result !== undefined) {
            return job.result;
        }

        if (job.status === 'failed') {
            throw new ApiError(job.error || 'Background job failed', 500);
        }

        await sleep(BACKGROUND_JOB_POLL_INTERVAL_MS, signal);
    }

    throw new ApiError('Background job timed out while polling for completion.', 408);
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
    const response = await fetchWithHandling(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }, {
        timeoutMs: UNSAFE_METHOD_TIMEOUT_MS,
        maxRetries: 0,
    });
    const job = await response.json() as BackgroundJob<LLMJudgeResult>;
    return waitForBackgroundJob<LLMJudgeResult>(job.job_id);
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
    const response = await fetchWithHandling(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }, {
        timeoutMs: UNSAFE_METHOD_TIMEOUT_MS,
        maxRetries: 0,
    });
    const job = await response.json() as BackgroundJob<SyntheticDataResult>;
    return waitForBackgroundJob<SyntheticDataResult>(job.job_id);
}

// =============================================================================
// REGRESSION & BASELINE API FUNCTIONS
// =============================================================================

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

/**
 * Pin an experiment as the baseline for its lineage.
 */
export async function setBaseline(experimentId: string): Promise<Experiment> {
    return fetchAPI<Experiment>(`/experiments/${experimentId}/set-baseline`, {
        method: 'POST',
    });
}

/**
 * Remove baseline status from an experiment.
 */
export async function unsetBaseline(experimentId: string): Promise<Experiment> {
    return fetchAPI<Experiment>(`/experiments/${experimentId}/set-baseline`, {
        method: 'DELETE',
    });
}

/**
 * Get regression report for an experiment.
 */
export async function getRegressionReport(experimentId: string, options: RequestInit = {}): Promise<RegressionReport> {
    return fetchAPI<RegressionReport>(`/results/${experimentId}/regression`, options);
}

/**
 * Rerun regression check against a specific or auto-detected baseline.
 */
export async function rerunRegression(experimentId: string, baselineId?: string): Promise<RegressionReport> {
    const params = baselineId ? `?baseline_id=${baselineId}` : '';
    return fetchAPI<RegressionReport>(`/results/${experimentId}/regression/rerun${params}`, {
        method: 'POST',
    });
}

// =============================================================================
// ROUTING TELEMETRY API
// =============================================================================

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

/**
 * Get routing telemetry (per-provider stats) for an experiment.
 */
export async function getRoutingTelemetry(experimentId: string, options: RequestInit = {}): Promise<Record<string, ProviderStats>> {
    return fetchAPI<Record<string, ProviderStats>>(`/results/${experimentId}/routing`, options);
}
