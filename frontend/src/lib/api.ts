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

export interface ExperimentConfig {
    model_name: string;
    reasoning_method: 'naive' | 'cot' | 'react';
    dataset_name: string;
    hyperparameters?: HyperParameters;
    rag?: RAGConfig;
    num_samples?: number;
}

export interface Experiment {
    id: string;
    name: string;
    description?: string;
    config: ExperimentConfig;
    status: 'pending' | 'running' | 'completed' | 'failed';
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
    };
}

export interface DashboardStats {
    totalExperiments: number;
    completedExperiments: number;
    runningExperiments: number;
    pendingExperiments: number;
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
 * 
 * TODO (Phase 3): Implement when metrics endpoints are ready
 */
export async function getMetrics(experimentId: string): Promise<Metrics> {
    return fetchAPI<Metrics>(`/results/${experimentId}/metrics`);
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
 * Health check.
 */
export async function healthCheck(): Promise<{ status: string }> {
    // Health endpoint is at root, not under /api/v1
    const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';
    const response = await fetch(`${baseUrl}/health`);
    return response.json();
}
