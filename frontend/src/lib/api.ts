/**
 * API Client for LlmForge backend.
 *
 * Centralized HTTP client for all API calls.
 * Handles error handling and response parsing.
 */

import {
    ApiError,
    fetchAPI,
    fetchWithHandling,
    getApiBaseUrl,
    getApiRootUrl,
    SAFE_METHOD_MAX_RETRIES,
    SAFE_METHOD_TIMEOUT_MS,
    UNSAFE_METHOD_TIMEOUT_MS,
    waitForBackgroundJob,
} from "@/lib/api-client";
import {
    getStoredCustomProviderCredentials,
    persistCustomProviderCredentials,
    resolveRunExperimentCredentials,
} from "@/lib/custom-provider";
import type {
    ComparisonResponse,
    CreateExperimentRequest,
    DashboardStats,
    Experiment,
    ExperimentList,
    ExperimentSlimList,
    LLMJudgeResult,
    ListExperimentsParams,
    Metrics,
    ModelOption,
    ProfileData,
    ProviderStats,
    ReadinessStatus,
    RegressionReport,
    RunGridSummary,
    RunSummary,
    StatisticalComparison,
    SyntheticDataResult,
} from "@/lib/api-types";

export { ApiError };
export {
    getStoredCustomProviderCredentials,
    persistCustomProviderCredentials,
    resolveRunExperimentCredentials,
};
export * from "@/lib/api-types";

const API_BASE_URL = getApiBaseUrl();

interface QueuedBackgroundJob {
    job_id: string;
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
 * Get sparse run grid summaries (no prompt/output text - for filmstrip).
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
    const job = await response.json() as QueuedBackgroundJob;
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
    const job = await response.json() as QueuedBackgroundJob;
    return waitForBackgroundJob<SyntheticDataResult>(job.job_id);
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

/**
 * Get routing telemetry (per-provider stats) for an experiment.
 */
export async function getRoutingTelemetry(experimentId: string, options: RequestInit = {}): Promise<Record<string, ProviderStats>> {
    return fetchAPI<Record<string, ProviderStats>>(`/results/${experimentId}/routing`, options);
}
