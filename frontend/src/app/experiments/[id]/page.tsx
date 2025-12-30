/**
 * Experiment Detail Page
 * 
 * Displays experiment configuration and results.
 * 
 * TODO (Iteration 1): Fetch experiment by ID
 * TODO (Iteration 2): Display results and metrics
 * TODO (Iteration 3): Add per-example drilldown
 */

import Link from 'next/link';

interface Props {
    params: Promise<{ id: string }>;
}

export default async function ExperimentDetailPage({ params }: Props) {
    const { id } = await params;

    // TODO (Iteration 1): Fetch experiment from API
    const experiment = null;

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-blue-600 hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <h1 className="text-2xl font-bold text-gray-900 mt-1">
                        Experiment: {id}
                    </h1>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* TODO: Replace with real data */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Configuration */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold mb-4">Configuration</h2>
                        <pre className="bg-gray-50 p-4 rounded text-sm overflow-auto">
                            {JSON.stringify({
                                model: 'microsoft/phi-2',
                                method: 'naive',
                                dataset: 'trivia_qa',
                                hyperparameters: {
                                    temperature: 0.7,
                                    max_tokens: 256,
                                    seed: 42,
                                },
                            }, null, 2)}
                        </pre>
                    </div>

                    {/* Status & Actions */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold mb-4">Status</h2>
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                                    Pending
                                </span>
                            </div>
                            <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                                Run Experiment
                            </button>
                        </div>
                    </div>
                </div>

                {/* Results - TODO (Iteration 2) */}
                <div className="bg-white rounded-lg shadow p-6 mt-6">
                    <h2 className="text-lg font-semibold mb-4">Results</h2>
                    <p className="text-gray-500 text-center py-8">
                        Run the experiment to see results.
                    </p>
                </div>
            </main>
        </div>
    );
}
