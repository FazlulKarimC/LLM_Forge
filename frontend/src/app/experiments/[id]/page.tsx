"use client";

/**
 * Experiment Detail Page
 * 
 * Displays experiment configuration, status, and full results dashboard.
 * Phase 3: Metrics cards, latency chart, correctness grid, export.
 */

import { useQuery } from "@tanstack/react-query";
import { use, useState } from "react";
import Link from "next/link";
import {
    getExperiment,
    getMetrics,
    getRunSummaries,
    exportResults,
    Experiment,
    Metrics,
    RunSummary,
} from "@/lib/api";

interface Props {
    params: Promise<{ id: string }>;
}

// =============================================================================
// Metric Card Component
// =============================================================================
function MetricCard({
    title,
    value,
    subtitle,
    color = "text-(--text-heading)",
}: {
    title: string;
    value: string;
    subtitle?: string;
    color?: string;
}) {
    return (
        <div className="card p-5">
            <p className="text-xs font-medium text-(--text-muted) uppercase tracking-wider">{title}</p>
            <p className={`text-2xl font-serif mt-1 ${color}`}>{value}</p>
            {subtitle && <p className="text-xs text-(--text-muted) mt-1">{subtitle}</p>}
        </div>
    );
}

// =============================================================================
// Latency Bar Chart (pure CSS, no Recharts dependency)
// =============================================================================
function LatencyChart({ runs }: { runs: RunSummary[] }) {
    const latencies = runs
        .filter((r) => r.latency_ms != null)
        .map((r) => r.latency_ms!);

    if (latencies.length === 0) return null;

    // Create histogram buckets
    const maxLatency = Math.max(...latencies);
    const minLatency = Math.min(...latencies);
    const bucketCount = Math.min(12, latencies.length);
    const bucketSize = (maxLatency - minLatency) / bucketCount || 1;

    const buckets: { label: string; count: number }[] = [];
    for (let i = 0; i < bucketCount; i++) {
        const low = minLatency + i * bucketSize;
        const high = low + bucketSize;
        const count = latencies.filter((l) => l >= low && (i === bucketCount - 1 ? l <= high : l < high)).length;
        buckets.push({
            label: `${Math.round(low)}`,
            count,
        });
    }

    const maxCount = Math.max(...buckets.map((b) => b.count));

    return (
        <div className="card p-6">
            <h3 className="text-lg font-serif text-(--text-heading) mb-4">Latency Distribution</h3>
            <div className="flex items-end gap-1 h-32">
                {buckets.map((bucket, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                        <div
                            className="w-full bg-[#37322F] rounded-t transition-all duration-300"
                            style={{
                                height: `${maxCount > 0 ? (bucket.count / maxCount) * 100 : 0}%`,
                                minHeight: bucket.count > 0 ? "4px" : "0px",
                            }}
                            title={`${bucket.count} runs`}
                        />
                    </div>
                ))}
            </div>
            <div className="flex gap-1 mt-1">
                {buckets.map((bucket, i) => (
                    <div key={i} className="flex-1 text-center">
                        <span className="text-[10px] text-(--text-muted)">{bucket.label}</span>
                    </div>
                ))}
            </div>
            <p className="text-xs text-(--text-muted) text-center mt-2">Latency (ms)</p>
        </div>
    );
}

// =============================================================================
// Correctness Grid
// =============================================================================
function CorrectnessGrid({ runs }: { runs: RunSummary[] }) {
    const [selectedRun, setSelectedRun] = useState<RunSummary | null>(null);

    return (
        <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-serif text-(--text-heading)">Response Correctness</h3>
                <div className="flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-3 h-3 rounded-sm bg-green-500" />
                        Correct
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-3 h-3 rounded-sm bg-red-400" />
                        Incorrect
                    </span>
                </div>
            </div>

            {/* Grid of colored squares */}
            <div className="flex flex-wrap gap-1.5 mb-4">
                {runs.map((run) => (
                    <button
                        key={run.id}
                        onClick={() => setSelectedRun(selectedRun?.id === run.id ? null : run)}
                        className={`w-7 h-7 rounded-sm transition-all duration-200 cursor-pointer border-2 ${run.is_correct
                            ? "bg-green-100 border-green-400 hover:bg-green-200"
                            : "bg-red-100 border-red-300 hover:bg-red-200"
                            } ${selectedRun?.id === run.id ? "ring-2 ring-[#37322F] ring-offset-1 scale-110" : ""}`}
                        title={`${run.example_id}: ${run.is_correct ? "✓" : "✗"} (F1: ${(run.score ?? 0).toFixed(2)})`}
                    >
                        <span className="text-[10px] font-mono">
                            {run.is_correct ? "✓" : "✗"}
                        </span>
                    </button>
                ))}
            </div>

            {/* Detail panel for selected run */}
            {selectedRun && (
                <div className="bg-(--bg-page) rounded-lg p-4 mt-2 border border-border animate-in fade-in">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-(--text-heading)">
                            {selectedRun.example_id}
                        </span>
                        <span
                            className={`text-xs px-2 py-0.5 rounded-full ${selectedRun.is_correct
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
                                }`}
                        >
                            {selectedRun.is_correct ? "Correct" : "Incorrect"}
                        </span>
                    </div>
                    <dl className="space-y-2 text-sm">
                        <div>
                            <dt className="text-(--text-muted) text-xs">Question</dt>
                            <dd className="text-(--text-body)">{selectedRun.input_text?.split("Question:").pop()?.split("Answer:")[0]?.trim() || selectedRun.input_text || "N/A"}</dd>
                        </div>
                        <div>
                            <dt className="text-(--text-muted) text-xs">Model Output</dt>
                            <dd className="font-mono text-(--text-body)">{selectedRun.output_text || "N/A"}</dd>
                        </div>
                        <div>
                            <dt className="text-(--text-muted) text-xs">Expected</dt>
                            <dd className="font-mono text-(--text-body)">{selectedRun.expected_output || "N/A"}</dd>
                        </div>
                        <div className="flex gap-4">
                            <div>
                                <dt className="text-(--text-muted) text-xs">F1 Score</dt>
                                <dd className="font-mono">{(selectedRun.score ?? 0).toFixed(3)}</dd>
                            </div>
                            <div>
                                <dt className="text-(--text-muted) text-xs">Latency</dt>
                                <dd className="font-mono">{(selectedRun.latency_ms ?? 0).toFixed(0)} ms</dd>
                            </div>
                        </div>
                    </dl>
                </div>
            )}
        </div>
    );
}

// =============================================================================
// Top/Bottom Runs Table
// =============================================================================
function ExtremeRunsTable({ runs }: { runs: RunSummary[] }) {
    const sorted = [...runs]
        .filter((r) => r.latency_ms != null)
        .sort((a, b) => (a.latency_ms ?? 0) - (b.latency_ms ?? 0));

    const fastest = sorted.slice(0, 5);
    const slowest = sorted.slice(-5).reverse();

    if (sorted.length === 0) return null;

    const renderTable = (title: string, items: RunSummary[]) => (
        <div>
            <h4 className="text-sm font-medium text-(--text-heading) mb-2">{title}</h4>
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-border">
                        <th className="text-left py-1.5 text-(--text-muted) font-medium">ID</th>
                        <th className="text-right py-1.5 text-(--text-muted) font-medium">Latency</th>
                        <th className="text-right py-1.5 text-(--text-muted) font-medium">F1</th>
                        <th className="text-center py-1.5 text-(--text-muted) font-medium">Correct</th>
                    </tr>
                </thead>
                <tbody>
                    {items.map((run) => (
                        <tr key={run.id} className="border-b border-border/50">
                            <td className="py-1.5 font-mono text-xs text-(--text-body)">{run.example_id}</td>
                            <td className="py-1.5 text-right font-mono text-xs">{(run.latency_ms ?? 0).toFixed(0)} ms</td>
                            <td className="py-1.5 text-right font-mono text-xs">{(run.score ?? 0).toFixed(2)}</td>
                            <td className="py-1.5 text-center">{run.is_correct ? "✓" : "✗"}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );

    return (
        <div className="card p-6">
            <h3 className="text-lg font-serif text-(--text-heading) mb-4">Performance Extremes</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {renderTable("⚡ Fastest 5", fastest)}
                {renderTable("🐢 Slowest 5", slowest)}
            </div>
        </div>
    );
}

// =============================================================================
// Results Dashboard
// =============================================================================
function ResultsDashboard({ experimentId }: { experimentId: string }) {
    const {
        data: metrics,
        isLoading: metricsLoading,
        error: metricsError,
    } = useQuery({
        queryKey: ["metrics", experimentId],
        queryFn: () => getMetrics(experimentId),
    });

    const {
        data: runs,
        isLoading: runsLoading,
    } = useQuery({
        queryKey: ["runs", experimentId],
        queryFn: () => getRunSummaries(experimentId),
    });

    const [exporting, setExporting] = useState(false);

    const handleExport = async () => {
        setExporting(true);
        try {
            await exportResults(experimentId);
        } catch (e) {
            console.error("Export failed:", e);
        } finally {
            setExporting(false);
        }
    };

    if (metricsLoading || runsLoading) {
        return (
            <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="card p-5 animate-pulse">
                            <div className="h-3 bg-(--bg-page) rounded w-20 mb-2" />
                            <div className="h-7 bg-(--bg-page) rounded w-24" />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (metricsError || !metrics) {
        return (
            <div className="card p-6">
                <p className="text-(--text-muted) text-center py-4">
                    No results available yet. Run the experiment to see metrics.
                </p>
            </div>
        );
    }

    const correctCount = runs?.filter((r) => r.is_correct).length ?? 0;
    const totalCount = runs?.length ?? 0;

    return (
        <div className="space-y-6">
            {/* Export Button */}
            <div className="flex justify-end">
                <button
                    onClick={handleExport}
                    disabled={exporting}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-[#37322F] text-white rounded-full hover:bg-[#2A2520] transition-colors disabled:opacity-50 cursor-pointer"
                >
                    {exporting ? (
                        <>
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Exporting...
                        </>
                    ) : (
                        <>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7,10 12,15 17,10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            Export JSON
                        </>
                    )}
                </button>
            </div>

            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    title="Accuracy (Exact Match)"
                    value={`${((metrics.quality.accuracy_exact ?? 0) * 100).toFixed(1)}%`}
                    subtitle={`${correctCount}/${totalCount} correct`}
                    color={
                        (metrics.quality.accuracy_exact ?? 0) >= 0.7
                            ? "text-green-600"
                            : (metrics.quality.accuracy_exact ?? 0) >= 0.4
                                ? "text-yellow-600"
                                : "text-red-600"
                    }
                />
                <MetricCard
                    title="Accuracy (Substring)"
                    value={`${((metrics.quality.accuracy_substring ?? 0) * 100).toFixed(1)}%`}
                    subtitle="Contains correct answer"
                />
                <MetricCard
                    title="F1 Score (Mean)"
                    value={`${((metrics.quality.accuracy_f1 ?? 0) * 100).toFixed(1)}%`}
                    subtitle="Token-level overlap"
                />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    title="Latency (p50)"
                    value={`${(metrics.performance.latency_p50 ?? 0).toFixed(0)} ms`}
                    subtitle="Median response time"
                />
                <MetricCard
                    title="Latency (p95)"
                    value={`${(metrics.performance.latency_p95 ?? 0).toFixed(0)} ms`}
                    subtitle="95th percentile"
                />
                <MetricCard
                    title="Total Tokens"
                    value={`${(metrics.cost.total_tokens_input + metrics.cost.total_tokens_output).toLocaleString()}`}
                    subtitle={`${metrics.cost.total_runs} runs · ${(metrics.cost.gpu_time_seconds ?? 0).toFixed(1)}s GPU`}
                />
            </div>

            {/* Latency Histogram */}
            {runs && <LatencyChart runs={runs} />}

            {/* Correctness Grid */}
            {runs && <CorrectnessGrid runs={runs} />}

            {/* Extreme Runs */}
            {runs && <ExtremeRunsTable runs={runs} />}
        </div>
    );
}

// =============================================================================
// Main Page
// =============================================================================
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

                {/* Results Dashboard */}
                <div className="mt-6">
                    <h2 className="text-xl font-serif text-(--text-heading) mb-4">Results</h2>
                    {experiment.status === "completed" ? (
                        <ResultsDashboard experimentId={id} />
                    ) : (
                        <div className="card p-6">
                            <p className="text-(--text-muted) text-center py-4">
                                {experiment.status === "running"
                                    ? "Experiment is running..."
                                    : "Run the experiment to see results."}
                            </p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
