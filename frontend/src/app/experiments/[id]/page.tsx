"use client";

/**
 * Experiment Detail Page
 * 
 * Displays experiment configuration and status.
 * Uses TanStack Query for data fetching.
 * Styled with DESIGN_SYSTEM.md.
 */

import { useQuery } from "@tanstack/react-query";
import { use } from "react";
import Link from "next/link";
import { getExperiment, Experiment } from "@/lib/api";

interface Props {
    params: Promise<{ id: string }>;
}

export default function ExperimentDetailPage({ params }: Props) {
    const { id } = use(params);

    const { data: experiment, isLoading, error } = useQuery({
        queryKey: ["experiment", id],
        queryFn: () => getExperiment(id),
    });

    if (isLoading) {
        return (
            <div className="min-h-screen bg-(--bg-page) flex items-center justify-center">
                <div className="animate-pulse text-(--text-muted)">Loading experiment...</div>
            </div>
        );
    }

    if (error || !experiment) {
        return (
            <div className="min-h-screen bg-(--bg-page)">
                <header className="bg-(--bg-card) shadow-sm border-b border-border">
                    <div className="max-w-7xl mx-auto px-4 py-6">
                        <Link href="/experiments" className="text-primary hover:underline text-sm">
                            ← Back to Experiments
                        </Link>
                        <h1 className="text-2xl font-serif text-(--text-heading) mt-1">Experiment Not Found</h1>
                    </div>
                </header>
                <main className="max-w-7xl mx-auto px-4 py-8">
                    <div className="card p-6">
                        <p className="text-(--error)">
                            {error instanceof Error ? error.message : "Experiment not found"}
                        </p>
                    </div>
                </main>
            </div>
        );
    }

    const statusClasses: Record<string, string> = {
        pending: "badge-pending",
        running: "badge-running",
        completed: "badge-completed",
        failed: "badge-failed",
    };

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-primary hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <div className="flex items-center gap-3 mt-1">
                        <h1 className="text-2xl font-serif text-(--text-heading)">
                            {experiment.name}
                        </h1>
                        <span className={`text-xs px-2 py-1 rounded-full ${statusClasses[experiment.status]}`}>
                            {experiment.status}
                        </span>
                    </div>
                    {experiment.description && (
                        <p className="mt-1 text-(--text-muted)">{experiment.description}</p>
                    )}
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Configuration */}
                    <div className="card p-6">
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Configuration</h2>
                        <pre className="bg-(--bg-page) p-4 rounded-lg text-sm overflow-auto font-mono text-(--text-body)">
                            {JSON.stringify(experiment.config, null, 2)}
                        </pre>
                    </div>

                    {/* Details */}
                    <div className="card p-6">
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Details</h2>
                        <dl className="space-y-3">
                            <div>
                                <dt className="text-sm text-(--text-muted)">Model</dt>
                                <dd className="font-mono text-(--text-body)">{experiment.config.model_name}</dd>
                            </div>
                            <div>
                                <dt className="text-sm text-(--text-muted)">Reasoning Method</dt>
                                <dd className="text-(--text-body) capitalize">{experiment.config.reasoning_method}</dd>
                            </div>
                            <div>
                                <dt className="text-sm text-(--text-muted)">Dataset</dt>
                                <dd className="text-(--text-body)">{experiment.config.dataset_name}</dd>
                            </div>
                            <div>
                                <dt className="text-sm text-(--text-muted)">Samples</dt>
                                <dd className="text-(--text-body)">{experiment.config.num_samples ?? 100}</dd>
                            </div>
                            <div>
                                <dt className="text-sm text-(--text-muted)">Created</dt>
                                <dd className="text-(--text-body)">
                                    {new Date(experiment.created_at).toLocaleString()}
                                </dd>
                            </div>
                            {experiment.started_at && (
                                <div>
                                    <dt className="text-sm text-(--text-muted)">Started</dt>
                                    <dd className="text-(--text-body)">
                                        {new Date(experiment.started_at).toLocaleString()}
                                    </dd>
                                </div>
                            )}
                            {experiment.completed_at && (
                                <div>
                                    <dt className="text-sm text-(--text-muted)">Completed</dt>
                                    <dd className="text-(--text-body)">
                                        {new Date(experiment.completed_at).toLocaleString()}
                                    </dd>
                                </div>
                            )}
                        </dl>
                    </div>
                </div>

                {/* Error display */}
                {experiment.error_message && (
                    <div className="card p-6 mt-6 border-l-4 border-l-(--error)">
                        <h2 className="text-lg font-serif text-(--error) mb-2">Error</h2>
                        <pre className="text-sm text-(--text-body) whitespace-pre-wrap">
                            {experiment.error_message}
                        </pre>
                    </div>
                )}

                {/* Results placeholder */}
                <div className="card p-6 mt-6">
                    <h2 className="text-lg font-serif text-(--text-heading) mb-4">Results</h2>
                    {experiment.status === "completed" ? (
                        <p className="text-(--text-muted) text-center py-4">
                            Results visualization coming in Phase 2.
                        </p>
                    ) : (
                        <p className="text-(--text-muted) text-center py-4">
                            {experiment.status === "running"
                                ? "Experiment is running..."
                                : "Run the experiment to see results."}
                        </p>
                    )}
                </div>
            </main>
        </div>
    );
}
