/**
 * API Client for LLM Research Platform Backend
 * 
 * Centralized HTTP client for all API calls.
 * Handles authentication, error handling, and response parsing.
 * 
 * TODO (Iteration 1): Implement experiment CRUD
 * TODO (Iteration 2): Add results and metrics endpoints
 * TODO (Iteration 3): Add real-time updates via WebSocket
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

    return response.json();
}

// =============================================================================
// TYPES
// =============================================================================

export interface ExperimentConfig {
    model_name: string;
    reasoning_method: 'naive' | 'cot' | 'react';
    dataset_name: string;
    hyperparameters?: {
        temperature?: number;
        max_tokens?: number;
        top_p?: number;
        seed?: number;
    };
    rag?: {
        retrieval_method: 'none' | 'naive' | 'hybrid' | 'reranked';
        top_k?: number;
    };
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

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Create a new experiment.
 * 
 * TODO (Iteration 1): Connect to backend
 */
export async function createExperiment(data: {
    name: string;
    description?: string;
    config: ExperimentConfig;
}): Promise<Experiment> {
    // TODO: Implement
    // return fetchAPI<Experiment>('/experiments', {
    //   method: 'POST',
    //   body: JSON.stringify(data),
    // });
    throw new Error('Not implemented: Iteration 1');
}

/**
 * List experiments with optional filters.
 * 
 * TODO (Iteration 1): Connect to backend
 */
export async function listExperiments(params?: {
    status?: string;
    method?: string;
    skip?: number;
    limit?: number;
}): Promise<ExperimentList> {
    // TODO: Implement
    // const query = new URLSearchParams(params as any).toString();
    // return fetchAPI<ExperimentList>(`/experiments?${query}`);
    throw new Error('Not implemented: Iteration 1');
}

/**
 * Get experiment by ID.
 * 
 * TODO (Iteration 1): Connect to backend
 */
export async function getExperiment(id: string): Promise<Experiment> {
    // TODO: Implement
    // return fetchAPI<Experiment>(`/experiments/${id}`);
    throw new Error('Not implemented: Iteration 1');
}

/**
 * Run an experiment.
 * 
 * TODO (Iteration 1): Connect to backend
 */
export async function runExperiment(id: string): Promise<Experiment> {
    // TODO: Implement
    // return fetchAPI<Experiment>(`/experiments/${id}/run`, { method: 'POST' });
    throw new Error('Not implemented: Iteration 1');
}

/**
 * Get metrics for an experiment.
 * 
 * TODO (Iteration 2): Connect to backend
 */
export async function getMetrics(experimentId: string): Promise<Metrics> {
    // TODO: Implement
    // return fetchAPI<Metrics>(`/results/${experimentId}/metrics`);
    throw new Error('Not implemented: Iteration 2');
}

/**
 * Health check.
 */
export async function healthCheck(): Promise<{ status: string }> {
    return fetchAPI<{ status: string }>('/health');
}
