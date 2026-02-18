"use client";

/**
 * Experiments List Page
 * 
 * Displays all experiments with filtering by status and method.
 * Uses TanStack Query for data fetching.
 * Styled with DESIGN_SYSTEM.md.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { listExperiments, deleteExperiment, runExperiment, Experiment, ListExperimentsParams } from "@/lib/api";

export default function ExperimentsPage() {
    const queryClient = useQueryClient();
    const [statusFilter, setStatusFilter] = useState("");
    const [methodFilter, setMethodFilter] = useState("");

    const params: ListExperimentsParams = { limit: 50 };
    if (statusFilter) params.status = statusFilter;
    if (methodFilter) params.method = methodFilter;

    const { data, isLoading, error } = useQuery({
        queryKey: ["experiments", params],
        queryFn: () => listExperiments(params),
    });

    const deleteMutation = useMutation({
        mutationFn: deleteExperiment,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
        },
    });

    const runMutation = useMutation({
        mutationFn: runExperiment,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
        },
    });

    const experiments = data?.experiments ?? [];
    const total = data?.total ?? 0;

    function handleDelete(id: string, name: string) {
        if (!confirm(`Delete experiment "${name}"?`)) return;
        deleteMutation.mutate(id);
    }

    function handleRun(id: string) {
        runMutation.mutate(id);
    }

    const methodLabels: Record<string, string> = {
        naive: "Naive",
        cot: "Chain of Thought",
        react: "ReAct Agent",
    };

    function formatDate(dateStr: string) {
        return new Date(dateStr).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    }

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
                    <div>
                        <Link href="/" className="text-primary hover:underline text-sm">
                            ← Dashboard
                        </Link>
                        <h1 className="text-2xl font-serif text-(--text-heading) mt-1">
                            Experiments ({total})
                        </h1>
                    </div>
                    <div className="flex gap-3">
                        <Link href="/experiments/compare" className="border border-border rounded-full px-4 py-2 text-sm font-medium text-(--text-body) hover:bg-(--bg-page) transition-colors">
                            Compare
                        </Link>
                        <Link href="/experiments/new" className="btn-primary">
                            New Experiment
                        </Link>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {error && (
                    <div className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4">
                        <p className="text-red-700">{error instanceof Error ? error.message : "Failed to load"}</p>
                    </div>
                )}

                {/* Filters */}
                <div className="card p-4 mb-6">
                    <div className="flex gap-4">
                        <select
                            className="border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                        >
                            <option value="">All Statuses</option>
                            <option value="pending">Pending</option>
                            <option value="queued">Queued</option>
                            <option value="running">Running</option>
                            <option value="completed">Completed</option>
                            <option value="failed">Failed</option>
                        </select>
                        <select
                            className="border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                            value={methodFilter}
                            onChange={(e) => setMethodFilter(e.target.value)}
                        >
                            <option value="">All Methods</option>
                            <option value="naive">Naive</option>
                            <option value="cot">Chain of Thought</option>
                            <option value="react">ReAct Agent</option>
                        </select>
                    </div>
                </div>

                {/* Experiments Table */}
                <div className="card overflow-hidden">
                    <table className="min-w-full divide-y divide-border">
                        <thead className="bg-(--bg-page)">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Name</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Method</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Model</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Status</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Created</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-(--text-muted) uppercase">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="bg-(--bg-card) divide-y divide-border">
                            {isLoading ? (
                                <tr>
                                    <td colSpan={6} className="px-6 py-8 text-center">
                                        <div className="animate-pulse text-(--text-muted)">Loading experiments...</div>
                                    </td>
                                </tr>
                            ) : experiments.length === 0 ? (
                                <tr>
                                    <td colSpan={6} className="px-6 py-8 text-center text-(--text-muted)">
                                        No experiments found. Create your first experiment.
                                    </td>
                                </tr>
                            ) : (
                                experiments.map((exp) => (
                                    <tr key={exp.id} className="hover:bg-(--bg-page)">
                                        <td className="px-6 py-4">
                                            <Link href={`/experiments/${exp.id}`} className="text-primary hover:underline font-medium">
                                                {exp.name}
                                            </Link>
                                            {exp.description && (
                                                <p className="text-sm text-(--text-muted) truncate max-w-xs">{exp.description}</p>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-sm text-(--text-body)">
                                            {methodLabels[exp.config.reasoning_method] || exp.config.reasoning_method}
                                        </td>
                                        <td className="px-6 py-4 text-sm text-(--text-body) font-mono">{exp.config.model_name}</td>
                                        <td className="px-6 py-4">
                                            <StatusBadge status={exp.status} />
                                        </td>
                                        <td className="px-6 py-4 text-sm text-(--text-muted)">{formatDate(exp.created_at)}</td>
                                        <td className="px-6 py-4">
                                            <div className="flex gap-2 items-center">
                                                <Link href={`/experiments/${exp.id}`} className="text-primary hover:underline text-sm">
                                                    View
                                                </Link>
                                                {(exp.status === "pending" || exp.status === "failed") && (
                                                    <button
                                                        onClick={() => handleRun(exp.id)}
                                                        disabled={runMutation.isPending}
                                                        className="text-sm text-green-700 hover:underline disabled:opacity-50 cursor-pointer"
                                                    >
                                                        ▶ Run
                                                    </button>
                                                )}
                                                <button
                                                    onClick={() => handleDelete(exp.id, exp.name)}
                                                    className="text-(--error) hover:underline text-sm cursor-pointer"
                                                    disabled={deleteMutation.isPending}
                                                >
                                                    Delete
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </main>
        </div>
    );
}

function StatusBadge({ status }: { status: string }) {
    const classes: Record<string, string> = {
        pending: "badge-pending",
        queued: "badge-queued",
        running: "badge-running",
        completed: "badge-completed",
        failed: "badge-failed",
    };
    return <span className={`text-xs px-2 py-1 rounded-full ${classes[status] || "badge-pending"}`}>{status}</span>;
}
