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
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { listExperiments, deleteExperiment, runExperiment, Experiment, ListExperimentsParams } from "@/lib/api";
import { Eye, Play, Trash2, Loader2, AlertTriangle, X } from "lucide-react";
import { toast } from "sonner";

export default function ExperimentsPage() {
    const queryClient = useQueryClient();
    const router = useRouter();
    const [statusFilter, setStatusFilter] = useState("");
    const [methodFilter, setMethodFilter] = useState("");
    // Track in-flight action IDs so only the clicked row's button is disabled
    const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
    const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
    const [experimentToDelete, setExperimentToDelete] = useState<{ id: string, name: string } | null>(null);

    const params: ListExperimentsParams = { limit: 50 };
    if (statusFilter) params.status = statusFilter;
    if (methodFilter) params.method = methodFilter;

    const { data, isLoading, error } = useQuery({
        queryKey: ["experiments", params],
        queryFn: () => listExperiments(params),
    });

    const deleteMutation = useMutation({
        mutationFn: (id: string) => {
            setDeletingIds(prev => new Set(prev).add(id));
            return deleteExperiment(id);
        },
        onSuccess: (_data, id) => {
            setDeletingIds(prev => { const s = new Set(prev); s.delete(id); return s; });
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
            toast.success("Experiment deleted");
            setExperimentToDelete(null);
        },
        onError: (err, id) => {
            setDeletingIds(prev => { const s = new Set(prev); s.delete(id); return s; });
            toast.error(`Failed to delete: ${err.message}`);
            setExperimentToDelete(null);
        },
    });

    const runMutation = useMutation({
        mutationFn: (id: string) => {
            setRunningIds(prev => new Set(prev).add(id));
            return runExperiment(id);
        },
        onSuccess: (_data, id) => {
            setRunningIds(prev => { const s = new Set(prev); s.delete(id); return s; });
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
            toast.success("Experiment queued for execution");
        },
        onError: (err: Error, id) => {
            setRunningIds(prev => { const s = new Set(prev); s.delete(id); return s; });
            toast.error(`Failed to run experiment: ${err.message}`);
        },
    });

    const experiments = data?.experiments ?? [];
    const total = data?.total ?? 0;

    function confirmDelete() {
        if (!experimentToDelete) return;
        deleteMutation.mutate(experimentToDelete.id);
    }

    function handleDeleteClick(id: string, name: string) {
        setExperimentToDelete({ id, name });
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
                        <Link href="/dashboard" className="text-primary hover:underline text-sm">
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

            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-7xl mx-auto px-4 py-8"
            >
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
                                    <tr
                                        key={exp.id}
                                        className="hover:bg-(--bg-page) cursor-pointer"
                                        onClick={() => router.push(`/experiments/${exp.id}`)}
                                        tabIndex={0}
                                        role="button"
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter' || e.key === ' ') {
                                                e.preventDefault();
                                                router.push(`/experiments/${exp.id}`);
                                            }
                                        }}
                                    >
                                        <td className="px-6 py-4">
                                            <span className="text-primary font-medium">
                                                {exp.name}
                                            </span>
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
                                            <div className="flex gap-2 items-center" onClick={(e) => e.stopPropagation()}>
                                                <Link
                                                    href={`/experiments/${exp.id}`}
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium border border-border bg-(--bg-card) text-(--text-body) rounded-md hover:bg-(--bg-page) transition-colors shadow-xs"
                                                >
                                                    <Eye className="size-4" /> View
                                                </Link>

                                                <button
                                                    onClick={(e) => { e.stopPropagation(); handleRun(exp.id); }}
                                                    disabled={runningIds.has(exp.id) || exp.status === "running" || exp.status === "queued"}
                                                    className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium border rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shadow-xs ${exp.status === "completed"
                                                        ? "border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100"
                                                        : "border-green-200 bg-green-50 text-green-700 hover:bg-green-100"
                                                        }`}
                                                >
                                                    {runningIds.has(exp.id) ? (
                                                        <><Loader2 className="size-4 animate-spin" /> Starting...</>
                                                    ) : (
                                                        <><Play className="size-4" /> Run</>
                                                    )}
                                                </button>

                                                <button
                                                    onClick={(e) => { e.stopPropagation(); handleDeleteClick(exp.id, exp.name); }}
                                                    className="inline-flex items-center justify-center p-1.5 text-sm font-medium border border-red-200 bg-red-50 text-red-700 rounded-md hover:bg-red-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shadow-xs"
                                                    disabled={deletingIds.has(exp.id)}
                                                    title="Delete Experiment"
                                                >
                                                    {deletingIds.has(exp.id) ? (
                                                        <Loader2 className="size-4 animate-spin" />
                                                    ) : (
                                                        <Trash2 className="size-4" />
                                                    )}
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </motion.main>

            {/* Custom Delete Confirmation Modal */}
            {
                experimentToDelete && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#37322F]/40 backdrop-blur-sm animate-in fade-in">
                        <div className="bg-(--bg-card) p-6 rounded-xl shadow-lg border border-border max-w-md w-full mx-4 animate-in zoom-in-95 relative" role="dialog" aria-modal="true">

                            <button
                                onClick={() => setExperimentToDelete(null)}
                                className="absolute top-4 right-4 text-(--text-muted) hover:text-(--text-heading) transition-colors cursor-pointer"
                                aria-label="Close"
                            >
                                <X className="size-5" />
                            </button>

                            <div className="flex gap-4 items-start mb-6">
                                <div className="bg-red-100 p-2.5 rounded-full shrink-0 mt-1">
                                    <AlertTriangle className="size-6 text-red-600" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-serif text-(--text-heading) mb-1">Delete Experiment</h3>
                                    <p className="text-sm text-(--text-body)">
                                        Are you sure you want to delete <span className="font-medium text-(--text-heading)">"{experimentToDelete.name}"</span>?
                                        This action cannot be undone. All results and metrics will be permanently lost.
                                    </p>
                                </div>
                            </div>

                            <div className="flex justify-end gap-3">
                                <button
                                    onClick={() => setExperimentToDelete(null)}
                                    disabled={deleteMutation.isPending}
                                    className="px-5 py-2 text-sm font-medium text-(--text-body) border border-border rounded-full hover:bg-(--bg-page) transition-colors cursor-pointer"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmDelete}
                                    disabled={deleteMutation.isPending}
                                    className="inline-flex items-center justify-center gap-2 px-5 py-2 text-sm font-medium text-white bg-red-600 border border-red-700 rounded-full hover:bg-red-700 transition-colors disabled:opacity-50 cursor-pointer"
                                >
                                    {deleteMutation.isPending ? (
                                        <><Loader2 className="size-4 animate-spin" /> Deleting...</>
                                    ) : (
                                        "Delete"
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                )
            }
        </div >
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
