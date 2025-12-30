/**
 * Experiments List Page
 * 
 * Displays all experiments with filtering and search.
 * 
 * TODO (Iteration 1): Fetch real data from API
 * TODO (Iteration 2): Add filtering by status, method
 * TODO (Iteration 3): Add search and sorting
 */

import Link from 'next/link';

export default function ExperimentsPage() {
    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow">
                <div className="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
                    <div>
                        <Link href="/" className="text-blue-600 hover:underline text-sm">
                            ← Dashboard
                        </Link>
                        <h1 className="text-2xl font-bold text-gray-900 mt-1">
                            Experiments
                        </h1>
                    </div>
                    <Link
                        href="/experiments/new"
                        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                    >
                        New Experiment
                    </Link>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* Filters - TODO (Iteration 2) */}
                <div className="bg-white rounded-lg shadow p-4 mb-6">
                    <div className="flex gap-4">
                        <select className="border rounded px-3 py-2">
                            <option value="">All Statuses</option>
                            <option value="pending">Pending</option>
                            <option value="running">Running</option>
                            <option value="completed">Completed</option>
                            <option value="failed">Failed</option>
                        </select>
                        <select className="border rounded px-3 py-2">
                            <option value="">All Methods</option>
                            <option value="naive">Naive</option>
                            <option value="cot">Chain of Thought</option>
                            <option value="react">ReAct Agent</option>
                        </select>
                    </div>
                </div>

                {/* Experiments Table */}
                <div className="bg-white rounded-lg shadow overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Name
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Method
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Model
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Status
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Created
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                    Actions
                                </th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {/* TODO (Iteration 1): Map over real experiments */}
                            <tr>
                                <td colSpan={6} className="px-6 py-8 text-center text-gray-500">
                                    No experiments found. Create your first experiment.
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </main>
        </div>
    );
}
