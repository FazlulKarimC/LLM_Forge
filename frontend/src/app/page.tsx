"use client";

/**
 * LLM Research Platform - Dashboard
 * 
 * Main dashboard showing stats, recent experiments, and quick actions.
 * Uses TanStack Query for data fetching.
 * Styled with DESIGN_SYSTEM.md (4-color palette, Instrument Serif headings).
 */

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { getDashboardStats, listExperiments, Experiment, DashboardStats } from "@/lib/api";

export default function DashboardPage() {
  const statsQuery = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: getDashboardStats,
  });

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "recent"],
    queryFn: () => listExperiments({ limit: 5 }),
  });

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
              Make sure the backend is running at http://localhost:8000
            </p>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
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
        </div>

        {/* Quick Actions */}
        <div className="card p-6 mb-8">
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
        </div>

        {/* Recent Experiments */}
        <div className="card p-6">
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
              {recentExperiments.map((exp) => (
                <ExperimentRow key={exp.id} experiment={exp} />
              ))}
            </div>
          )}
        </div>
      </main>
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

function ExperimentRow({ experiment }: { experiment: Experiment }) {
  const statusClasses: Record<string, string> = {
    pending: "badge-pending",
    queued: "badge-queued",
    running: "badge-running",
    completed: "badge-completed",
    failed: "badge-failed",
  };

  return (
    <Link
      href={`/experiments/${experiment.id}`}
      className="block border border-border rounded-lg p-4 hover:bg-(--bg-page) transition-colors"
    >
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-medium text-(--text-heading)">{experiment.name}</h3>
          <p className="text-sm text-(--text-muted)">
            {experiment.config.model_name} • {experiment.config.reasoning_method}
          </p>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${statusClasses[experiment.status]}`}>
          {experiment.status}
        </span>
      </div>
    </Link>
  );
}
