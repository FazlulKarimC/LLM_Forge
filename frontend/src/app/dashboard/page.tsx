"use client";

/**
 * LLM Research Platform - Dashboard
 * 
 * Main dashboard showing stats, recent experiments, and quick actions.
 * Uses TanStack Query for data fetching.
 * Styled with DESIGN_SYSTEM.md (4-color palette, Instrument Serif headings).
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { getDashboardStats, listExperiments, deleteExperiment, runExperiment, Experiment } from "@/lib/api";
import { Play, Trash2, Eye, Loader2, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

export default function DashboardPage() {
  const queryClient = useQueryClient();
  const router = useRouter();

  // Track in-flight action IDs
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
  const [experimentToDelete, setExperimentToDelete] = useState<{ id: string, name: string } | null>(null);

  const statsQuery = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: getDashboardStats,
  });

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "recent"],
    queryFn: () => listExperiments({ limit: 5 }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => {
      setDeletingIds(prev => new Set(prev).add(id));
      return deleteExperiment(id);
    },
    onSuccess: (_data, id) => {
      setDeletingIds(prev => { const s = new Set(prev); s.delete(id); return s; });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
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
      toast.success("Experiment started");
    },
    onError: (err, id) => {
      setRunningIds(prev => { const s = new Set(prev); s.delete(id); return s; });
      toast.error(`Failed to start: ${err.message}`);
    },
  });

  const handleRun = async (id: string) => {
    runMutation.mutate(id);
  };

  const handleDeleteClick = (id: string, name: string) => {
    setExperimentToDelete({ id, name });
  };

  const confirmDelete = () => {
    if (experimentToDelete) {
      deleteMutation.mutate(experimentToDelete.id);
    }
  };

  const stats = statsQuery.data;
  const recentExperiments = experimentsQuery.data?.experiments ?? [];
  const loading = statsQuery.isLoading || experimentsQuery.isLoading;
  const error = statsQuery.error || experimentsQuery.error;

  return (
    <div className="min-h-screen bg-(--bg-page)">
      {/* Header */}
      <header className="bg-(--bg-card) shadow-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-serif text-(--text-heading)">
            LLM Research Platform
          </h1>
          <p className="mt-1 text-(--text-body)">
            Config-driven experimentation for reasoning, retrieval, and alignment
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Error State */}
        {error && (
          <div className="mb-8 bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700">
              <strong>Error:</strong> {error instanceof Error ? error.message : 'Failed to load'}
            </p>
            <p className="text-sm text-red-600 mt-1">
              Backend URL: <code className="font-mono bg-red-100 px-1 rounded">{process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1 (fallback — NEXT_PUBLIC_API_URL not set)'}</code>
            </p>
          </div>
        )}


        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
        >
          <StatCard
            title="Total Experiments"
            value={loading ? "..." : stats?.totalExperiments ?? 0}
            loading={loading}
          />
          <StatCard
            title="Completed"
            value={loading ? "..." : stats?.completedExperiments ?? 0}
            loading={loading}
          />
          <StatCard
            title="Running"
            value={loading ? "..." : stats?.runningExperiments ?? 0}
            loading={loading}
          />
          <StatCard
            title="Pending"
            value={loading ? "..." : stats?.pendingExperiments ?? 0}
            loading={loading}
          />
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="card p-6 mb-8"
        >
          <h2 className="text-xl font-serif text-(--text-heading) mb-4">Quick Actions</h2>
          <div className="flex gap-4">
            <Link
              href="/experiments/new"
              className="btn-primary"
            >
              New Experiment
            </Link>
            <Link
              href="/experiments"
              className="px-6 py-2 rounded-full border border-border text-(--text-body) hover:bg-(--bg-page) transition-colors"
            >
              View All Experiments
            </Link>
          </div>
        </motion.div>

        {/* Recent Experiments */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-serif text-(--text-heading)">Recent Experiments</h2>
            <Link href="/experiments" className="text-primary hover:underline text-sm">
              View all →
            </Link>
          </div>

          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse h-16 bg-(--bg-page) rounded-lg" />
              ))}
            </div>
          ) : recentExperiments.length === 0 ? (
            <p className="text-(--text-muted) text-center py-8">
              No experiments yet. Create your first experiment to get started.
            </p>
          ) : (
            <div className="space-y-3">
              {recentExperiments.map((exp, i) => (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.05, duration: 0.3 }}
                  key={exp.id}
                  className="block border border-border rounded-lg p-4 hover:bg-(--bg-page) transition-colors cursor-pointer shadow-xs"
                  onClick={() => router.push(`/experiments/${exp.id}`)}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-medium text-(--text-heading)">{exp.name}</h3>
                      <p className="text-sm text-(--text-muted)">
                        {exp.config.model_name} • {exp.config.reasoning_method}
                      </p>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className={`text-xs px-2 py-1 rounded-full badge-${exp.status}`}>
                        {exp.status}
                      </span>

                      <div className="flex gap-2 items-center" onClick={(e) => e.stopPropagation()}>
                        <Link
                          href={`/experiments/${exp.id}`}
                          className="inline-flex items-center justify-center p-1.5 text-sm font-medium border border-border bg-(--bg-card) text-(--text-body) rounded-md hover:bg-(--bg-page) transition-colors shadow-xs"
                          title="View Detail"
                        >
                          <Eye className="size-4" />
                        </Link>

                        <button
                          onClick={(e) => { e.stopPropagation(); handleRun(exp.id); }}
                          disabled={runningIds.has(exp.id) || exp.status === "running" || exp.status === "queued"}
                          className={`inline-flex items-center justify-center p-1.5 text-sm font-medium border rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shadow-xs ${exp.status === "completed"
                            ? "border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100"
                            : "border-green-200 bg-green-50 text-green-700 hover:bg-green-100"
                            }`}
                          title="Run Experiment"
                        >
                          {runningIds.has(exp.id) ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            <Play className="size-4" />
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
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </main>

      {/* Custom Delete Confirmation Modal */}
      {experimentToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div
            className="bg-(--bg-card) border border-border rounded-xl shadow-xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-red-50 rounded-full shrink-0">
                  <AlertTriangle className="size-6 text-red-600" />
                </div>
                <div>
                  <h3 className="text-lg font-serif text-(--text-heading)">Delete Experiment</h3>
                  <p className="mt-2 text-sm text-(--text-muted) leading-relaxed">
                    Are you sure you want to delete <span className="font-medium text-(--text-heading)">{experimentToDelete.name}</span>?
                    This action cannot be undone.
                  </p>
                </div>
              </div>
            </div>

            <div className="px-6 py-4 bg-(--bg-page) border-t border-border flex items-center justify-end gap-3">
              <button
                onClick={() => setExperimentToDelete(null)}
                disabled={deleteMutation.isPending}
                className="px-4 py-2 text-sm font-medium text-(--text-body) bg-white border border-border rounded-lg hover:bg-gray-50 hover:text-gray-900 transition-colors disabled:opacity-50 cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={deleteMutation.isPending}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-700 transition-colors focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-xs cursor-pointer"
              >
                {deleteMutation.isPending ? (
                  <><Loader2 className="size-4 animate-spin" /> Deleting...</>
                ) : (
                  <><Trash2 className="size-4" /> Delete</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  title,
  value,
  loading,
}: {
  title: string;
  value: string | number;
  loading?: boolean;
}) {
  return (
    <div className="card p-6">
      <dt className="text-sm font-medium text-(--text-muted)">{title}</dt>
      <dd className={`mt-1 text-3xl font-serif text-(--text-heading) ${loading ? "animate-pulse" : ""}`}>
        {value}
      </dd>
    </div>
  );
}


