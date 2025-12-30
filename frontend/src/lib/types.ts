/**
 * Type definitions for LLM Research Platform
 * 
 * Shared types used across the frontend application.
 */

// Re-export API types
export type {
    Experiment,
    ExperimentConfig,
    ExperimentList,
    Metrics,
} from './api';

// =============================================================================
// UI TYPES
// =============================================================================

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed';

export type ReasoningMethod = 'naive' | 'cot' | 'react';

export type RetrievalMethod = 'none' | 'naive' | 'hybrid' | 'reranked';

export interface StatusBadge {
    label: string;
    color: 'gray' | 'blue' | 'green' | 'red';
}

export const STATUS_BADGES: Record<ExperimentStatus, StatusBadge> = {
    pending: { label: 'Pending', color: 'gray' },
    running: { label: 'Running', color: 'blue' },
    completed: { label: 'Completed', color: 'green' },
    failed: { label: 'Failed', color: 'red' },
};

export const METHOD_LABELS: Record<ReasoningMethod, string> = {
    naive: 'Naive',
    cot: 'Chain of Thought',
    react: 'ReAct Agent',
};

export const RETRIEVAL_LABELS: Record<RetrievalMethod, string> = {
    none: 'No RAG',
    naive: 'Naive RAG',
    hybrid: 'Hybrid RAG',
    reranked: 'Reranked RAG',
};
