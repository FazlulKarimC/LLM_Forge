/**
 * LLM Research Platform - Dashboard
 * 
 * Main dashboard showing:
 * - Summary statistics
 * - Recent experiments
 * - Quick actions
 * 
 * TODO (Iteration 1): Fetch real data from API
 * TODO (Iteration 2): Add charts and visualizations
 * TODO (Iteration 3): Add real-time updates
 */

import Link from 'next/link';

// Placeholder data - will be replaced with API calls
const MOCK_STATS = {
  totalExperiments: 0,
  completedExperiments: 0,
  averageAccuracy: 0,
  totalRuns: 0,
};

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            LLM Research Platform
          </h1>
          <p className="mt-1 text-gray-600">
            Config-driven experimentation for reasoning, retrieval, and alignment
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Total Experiments"
            value={MOCK_STATS.totalExperiments}
          />
          <StatCard
            title="Completed"
            value={MOCK_STATS.completedExperiments}
          />
          <StatCard
            title="Avg Accuracy"
            value={`${(MOCK_STATS.averageAccuracy * 100).toFixed(1)}%`}
          />
          <StatCard
            title="Total Runs"
            value={MOCK_STATS.totalRuns}
          />
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
          <div className="flex gap-4">
            <Link
              href="/experiments/new"
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              New Experiment
            </Link>
            <Link
              href="/experiments"
              className="bg-gray-100 text-gray-700 px-4 py-2 rounded hover:bg-gray-200"
            >
              View All Experiments
            </Link>
          </div>
        </div>

        {/* Recent Experiments */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Experiments</h2>
          {/* TODO (Iteration 1): Fetch and display real experiments */}
          <p className="text-gray-500 text-center py-8">
            No experiments yet. Create your first experiment to get started.
          </p>
        </div>
      </main>
    </div>
  );
}

/**
 * Stat card component for dashboard metrics.
 */
function StatCard({ title, value }: { title: string; value: string | number }) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <dt className="text-sm font-medium text-gray-500">{title}</dt>
      <dd className="mt-1 text-3xl font-semibold text-gray-900">{value}</dd>
    </div>
  );
}
