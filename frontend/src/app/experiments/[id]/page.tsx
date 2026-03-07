"use client";

/**
 * Experiment Detail Page
 * 
 * Displays experiment configuration, status, and full results dashboard.
 * Phase 3: Metrics cards, latency chart, correctness grid, export.
 * Includes run button and auto-polling for running experiments.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { use, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
    getExperiment,
    getMetrics,
    getRunSummaries,
    getProfile,
    exportResults,
    exportMarkdownReport,
    runExperiment,
    Metrics,
    RunSummary,
    ProfileData,
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
const GRID_PAGE_SIZE = 50;

function CorrectnessGrid({ runs }: { runs: RunSummary[] }) {
    const [selectedRun, setSelectedRun] = useState<RunSummary | null>(null);
    const [page, setPage] = useState(0);

    const totalPages = Math.ceil(runs.length / GRID_PAGE_SIZE);
    const pageRuns = runs.slice(page * GRID_PAGE_SIZE, (page + 1) * GRID_PAGE_SIZE);
    const startIdx = page * GRID_PAGE_SIZE + 1;
    const endIdx = Math.min((page + 1) * GRID_PAGE_SIZE, runs.length);

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

            <div className="flex flex-wrap gap-1.5 mb-4">
                {pageRuns.map((run) => (

                    <button
                        key={run.id}
                        onClick={() => setSelectedRun(selectedRun?.id === run.id ? null : run)}
                        className={`w-7 h-7 rounded-sm transition-all duration-200 cursor-pointer border-2 ${run.is_correct
                            ? "bg-green-100 border-green-400 hover:bg-green-200"
                            : "bg-red-100 border-red-300 hover:bg-red-200"
                            } ${selectedRun?.id === run.id ? "ring-2 ring-[#37322F] ring-offset-1 scale-110" : ""}`}
                        title={`${run.example_id}: ${run.failure_mode ? "⚠️" : run.is_correct ? "✓" : "✗"} (F1: ${(run.score ?? 0).toFixed(2)})`}
                    >
                        <span className="text-[10px] font-mono">
                            {run.failure_mode ? "⚠️" : run.is_correct ? "✓" : "✗"}
                        </span>
                    </button>
                ))}
            </div>

            {selectedRun && (
                <div className="bg-(--bg-page) rounded-lg p-4 mt-2 border border-border animate-in fade-in">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-(--text-heading)">
                            {selectedRun.example_id}
                        </span>
                        <span
                            className={`text-xs px-2 py-0.5 rounded-full ${selectedRun.failure_mode
                                ? "bg-red-100 text-red-700"
                                : selectedRun.is_correct
                                    ? "bg-green-100 text-green-700"
                                    : "bg-red-100 text-red-700"
                                }`}
                        >
                            {selectedRun.failure_mode ? `Failed: ${selectedRun.failure_mode}` : selectedRun.is_correct ? "Correct" : "Incorrect"}
                        </span>
                    </div>
                    <dl className="space-y-2 text-sm">
                        <div>
                            <dt className="text-(--text-muted) text-xs">Question</dt>
                            <dd className="text-(--text-body)">{selectedRun.prompt?.split("Question:").pop()?.split("Answer:")[0]?.trim() || selectedRun.prompt || "N/A"}</dd>
                        </div>
                        {selectedRun.failure_mode && selectedRun.error_message && (
                            <div className="bg-red-50 p-2 rounded border border-red-100 mt-2 mb-2">
                                <dt className="text-red-700 text-xs font-semibold">Error Message</dt>
                                <dd className="font-mono text-red-600 text-[10px] whitespace-pre-wrap">{selectedRun.error_message}</dd>
                            </div>
                        )}
                        <div>
                            <dt className="text-(--text-muted) text-xs">Model Output</dt>
                            <dd className="font-mono text-(--text-body)">{selectedRun.raw_output || "N/A"}</dd>
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
                            {selectedRun.faithfulness_score !== undefined && selectedRun.faithfulness_score !== null && (
                                <div>
                                    <dt className="text-(--text-muted) text-xs">Faithfulness (heuristic)</dt>
                                    <dd className="font-mono">{selectedRun.faithfulness_score.toFixed(3)}</dd>
                                </div>
                            )}
                        </div>
                        {selectedRun.retrieved_chunks && selectedRun.retrieved_chunks.chunks && selectedRun.retrieved_chunks.chunks.length > 0 && (
                            <div className="pt-2 border-t border-border mt-2">
                                <dt className="text-(--text-muted) text-xs mb-1">Retrieved Context ({selectedRun.retrieved_chunks.chunks.length} chunks)</dt>
                                <dd className="bg-(--bg-card) p-3 rounded text-xs font-mono text-(--text-body) max-h-40 overflow-y-auto whitespace-pre-wrap">
                                    {selectedRun.retrieved_chunks.chunks.map((c: { text?: string; page_content?: string; score?: number }, i: number) => (
                                        <div key={i} className="mb-2 pb-2 border-b border-border/50 last:border-0 last:mb-0 last:pb-0">
                                            <span className="text-primary/80 font-bold">[{i + 1}]</span> {c.text || c.page_content || JSON.stringify(c)}
                                            {c.score != null && <span className="ml-2 text-(--text-muted)">(score: {c.score.toFixed(3)})</span>}
                                        </div>
                                    ))}
                                </dd>
                            </div>
                        )}
                        {selectedRun.agent_trace && (
                            <div className="pt-2 border-t border-border mt-2">
                                <dt className="text-(--text-muted) text-xs mb-1">Agent Trace ({selectedRun.agent_trace.total_tool_calls} tool calls)</dt>
                                <dd className="bg-[--bg-card] p-3 rounded text-xs font-mono text-(--text-body) max-h-60 overflow-y-auto whitespace-pre-wrap">
                                    <div className="mb-2 flex gap-4 text-[10px] text-[--text-muted]">
                                        <span className="text-green-600">✓ {selectedRun.agent_trace.successful_tool_calls} success</span>
                                        <span className={selectedRun.agent_trace.failed_tool_calls > 0 ? "text-red-500" : ""}>
                                            ✗ {selectedRun.agent_trace.failed_tool_calls} failed
                                        </span>
                                    </div>
                                    <div className="space-y-4">
                                        {selectedRun.agent_trace.steps.map((step, i) => (
                                            <div key={i} className="pl-2 border-l-2 border-[--border] space-y-1">
                                                <div className="text-blue-600 dark:text-blue-400 font-semibold">[Thought]</div>
                                                <div className="pl-2 pb-1">{step.thought}</div>

                                                {step.action && (
                                                    <>
                                                        <div className="text-amber-600 dark:text-amber-500 font-semibold mt-2">[Action: {step.action}]</div>
                                                        <div className="pl-2 text-[--text-muted]">{step.action_input}</div>
                                                    </>
                                                )}

                                                {step.observation && (
                                                    <>
                                                        <div className="text-green-600 dark:text-green-500 font-semibold mt-2">[Observation]</div>
                                                        <div className="pl-2 italic">{step.observation}</div>
                                                    </>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </dd>
                            </div>
                        )}
                    </dl>
                </div>
            )}

            {/* Pagination Controls */}
            {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                    <span className="text-xs text-(--text-muted)">
                        Showing {startIdx}–{endIdx} of {runs.length} runs
                    </span>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => { setPage(p => p - 1); setSelectedRun(null); }}
                            disabled={page === 0}
                            className="px-3 py-1 text-xs rounded-lg border border-border text-(--text-body) hover:bg-(--bg-page) disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors"
                        >
                            ← Prev
                        </button>
                        <span className="text-xs text-(--text-muted)">{page + 1} / {totalPages}</span>
                        <button
                            onClick={() => { setPage(p => p + 1); setSelectedRun(null); }}
                            disabled={page >= totalPages - 1}
                            className="px-3 py-1 text-xs rounded-lg border border-border text-(--text-body) hover:bg-(--bg-page) disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors"
                        >
                            Next →
                        </button>
                    </div>
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
            {/* Export Buttons */}
            <div className="flex justify-end gap-2">
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
                <button
                    onClick={async () => {
                        setExporting(true);
                        try {
                            await exportMarkdownReport(experimentId, undefined, metrics ?? undefined, runs ?? undefined);
                        } catch (e) {
                            console.error("Markdown export failed:", e);
                        } finally {
                            setExporting(false);
                        }
                    }}
                    disabled={exporting}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium border border-[#37322F] text-[#37322F] rounded-full hover:bg-[#37322F]/5 transition-colors disabled:opacity-50 cursor-pointer"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14,2 14,8 20,8" />
                    </svg>
                    Export Report
                </button>
            </div>

            {/* AI Summary */}
            {metrics.summary_text && (
                <div className="card p-6 bg-blue-50/50 border-blue-100">
                    <div className="flex items-start gap-3">
                        <div className="mt-1">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-500">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                            </svg>
                        </div>
                        <div>
                            <h3 className="text-sm font-semibold text-blue-900 mb-1">Experiment Summary</h3>
                            <p className="text-sm text-blue-800 leading-relaxed">
                                {metrics.summary_text}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Metrics Cards — Quality */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    title="Accuracy (Exact)"
                    value={`${((metrics.quality.accuracy_exact ?? 0) * 100).toFixed(1)}%`}
                    subtitle={`${correctCount}/${totalCount} perfect match`}
                    color={
                        (metrics.quality.accuracy_exact ?? 0) >= 0.7
                            ? "text-green-600"
                            : (metrics.quality.accuracy_exact ?? 0) >= 0.4
                                ? "text-yellow-600"
                                : "text-red-600"
                    }
                />
                <MetricCard
                    title="F1 Score (Mean)"
                    value={`${((metrics.quality.accuracy_f1 ?? 0) * 100).toFixed(1)}%`}
                    subtitle="Token-level overlap"
                />

                {/* Conditional Metrics based on experiment type */}
                {metrics.quality.safety_score !== undefined && (
                    <MetricCard
                        title="Safety Score"
                        value={`${(metrics.quality.safety_score * 100).toFixed(1)}%`}
                        subtitle="Refusal rate on adversarial prompts"
                        color={
                            metrics.quality.safety_score >= 0.9
                                ? "text-green-600"
                                : metrics.quality.safety_score >= 0.7
                                    ? "text-yellow-600"
                                    : "text-red-600"
                        }
                    />
                )}

                {metrics.quality.pass_at_k !== undefined && (
                    <MetricCard
                        title="Pass@5"
                        value={`${(metrics.quality.pass_at_k * 100).toFixed(1)}%`}
                        subtitle="Probability of correct answer in 5 tries"
                        color="text-purple-600"
                    />
                )}

                {metrics.quality.safety_score === undefined && metrics.quality.pass_at_k === undefined && (
                    <MetricCard
                        title="Throughput"
                        value={`${(metrics.performance.throughput ?? 0).toFixed(1)}/s`}
                        subtitle="Prompts per second"
                    />
                )}
            </div>

            {/* Metrics Cards — Performance & Cost */}
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
                    title="Cost Efficiency"
                    value={
                        metrics.cost.total_cost_usd != null && metrics.cost.total_cost_usd > 0
                            ? `$${metrics.cost.total_cost_usd.toFixed(4)}`
                            : "Free Tier"
                    }
                    subtitle={
                        metrics.cost.cost_per_correct_answer != null && metrics.cost.cost_per_correct_answer > 0
                            ? `$${metrics.cost.cost_per_correct_answer.toFixed(4)}/correct · ${(metrics.cost.total_tokens_input + metrics.cost.total_tokens_output).toLocaleString()} tokens`
                            : `${(metrics.cost.total_tokens_input + metrics.cost.total_tokens_output).toLocaleString()} tokens · ${metrics.cost.total_runs} runs`
                    }
                />
            </div>

            {/* Failure Modes Banner */}
            {metrics.failure_modes && metrics.failure_modes.total_failures > 0 && (
                <div className="card p-6 bg-red-50/50 border-red-100">
                    <h3 className="text-sm font-semibold text-red-900 mb-3 flex items-center gap-2">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                            <line x1="12" y1="9" x2="12" y2="13"></line>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                        Failure Analysis ({metrics.failure_modes.total_failures} Total)
                    </h3>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                        {Object.entries(metrics.failure_modes.counts).map(([mode, count]) => (
                            <div key={mode} className="bg-white p-3 rounded border border-red-100">
                                <span className="text-2xl font-bold text-red-700 block">{count}</span>
                                <span className="text-xs text-red-900 font-mono capitalize">{mode.replace(/_/g, " ")}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

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
// Optimization Profile Dashboard (Phase 8)
// =============================================================================
function ProfileDashboard({ experimentId }: { experimentId: string }) {
    const { data: profile, isLoading } = useQuery({
        queryKey: ["profile", experimentId],
        queryFn: () => getProfile(experimentId),
    });

    if (isLoading) {
        return (
            <div className="card p-6 animate-pulse">
                <div className="h-5 bg-(--bg-page) rounded w-48 mb-4" />
                <div className="h-20 bg-(--bg-page) rounded" />
            </div>
        );
    }

    if (!profile || profile.message) {
        return null; // No optimization data
    }

    const sections = Object.entries(profile.profiling_summary || {});
    const cache = profile.cache_stats || {};
    const batch = profile.batch_stats || {};
    const hasCache = cache.hits !== undefined || cache.misses !== undefined;
    const hasBatch = batch.batches_processed !== undefined;

    return (
        <div className="space-y-4">
            <h3 className="text-lg font-serif text-(--text-heading)">⚡ Optimization Profile</h3>

            {/* Wall Time + Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {profile.total_wall_time_ms != null && (
                    <MetricCard
                        title="Total Wall Time"
                        value={`${(profile.total_wall_time_ms / 1000).toFixed(2)}s`}
                        subtitle="End-to-end execution"
                    />
                )}
                {hasCache && (
                    <MetricCard
                        title="Cache Hit Rate"
                        value={`${((cache.hit_rate ?? 0) * 100).toFixed(1)}%`}
                        subtitle={`${cache.hits ?? 0} hits / ${cache.misses ?? 0} misses`}
                        color={(cache.hit_rate ?? 0) > 0.5 ? "text-green-600" : "text-(--text-heading)"}
                    />
                )}
                {hasBatch && (
                    <MetricCard
                        title="Batches Processed"
                        value={`${batch.batches_processed ?? 0}`}
                        subtitle={`${batch.total_prompts_batched ?? 0} prompts batched`}
                    />
                )}
            </div>

            {/* Timing Breakdown Table */}
            {sections.length > 0 && (
                <div className="card p-6">
                    <h4 className="text-sm font-medium text-(--text-heading) mb-3">Timing Breakdown</h4>
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-border">
                                <th className="text-left py-1.5 text-(--text-muted) font-medium">Phase</th>
                                <th className="text-right py-1.5 text-(--text-muted) font-medium">Count</th>
                                <th className="text-right py-1.5 text-(--text-muted) font-medium">Total</th>
                                <th className="text-right py-1.5 text-(--text-muted) font-medium">Mean</th>
                                <th className="text-right py-1.5 text-(--text-muted) font-medium">p50</th>
                                <th className="text-right py-1.5 text-(--text-muted) font-medium">p95</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sections.map(([name, stats]) => (
                                <tr key={name} className="border-b border-border/50">
                                    <td className="py-1.5 font-mono text-xs text-(--text-body) capitalize">{name.replace(/_/g, " ")}</td>
                                    <td className="py-1.5 text-right font-mono text-xs">{stats.count}</td>
                                    <td className="py-1.5 text-right font-mono text-xs">{stats.total_ms.toFixed(0)} ms</td>
                                    <td className="py-1.5 text-right font-mono text-xs">{stats.mean_ms.toFixed(1)} ms</td>
                                    <td className="py-1.5 text-right font-mono text-xs">{stats.p50_ms.toFixed(1)} ms</td>
                                    <td className="py-1.5 text-right font-mono text-xs">{stats.p95_ms.toFixed(1)} ms</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Cache Latency Saved */}
            {hasCache && (cache.total_latency_saved_ms ?? 0) > 0 && (
                <div className="card p-4">
                    <p className="text-sm text-(--text-body)">
                        💡 Cache saved <strong className="font-mono">{((cache.total_latency_saved_ms ?? 0) / 1000).toFixed(2)}s</strong> of API call time
                        ({cache.size ?? 0}/{cache.max_size ?? 0} entries used).
                    </p>
                </div>
            )}
        </div>
    );
}

// =============================================================================
// Main Page
// =============================================================================
export default function ExperimentDetailPage({ params }: Props) {
    const { id } = use(params);
    const queryClient = useQueryClient();

    const { data: experiment, isLoading, error } = useQuery({
        queryKey: ["experiment", id],
        queryFn: () => getExperiment(id),
        // Auto-refetch every 3s while experiment is running or queued
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return (status === "running" || status === "queued") ? 3000 : false;
        },
    });

    const runMutation = useMutation({
        mutationFn: () => {
            let customBaseUrl: string | undefined = undefined;
            let customApiKey: string | undefined = undefined;

            // Check if there are saved settings for the current experiment's model
            if (experiment?.config.model_name && typeof window !== "undefined") {
                try {
                    const settings = JSON.parse(localStorage.getItem("customLLMSettings") || "{}");
                    const modelSettings = settings[experiment.config.model_name];

                    if (modelSettings) {
                        customBaseUrl = modelSettings.baseUrl;
                        customApiKey = modelSettings.apiKey;
                    } else if (localStorage.getItem("customModelId") === experiment.config.model_name) {
                        // Fallback to legacy single-model storage
                        customBaseUrl = localStorage.getItem("customBaseUrl") || undefined;
                        customApiKey = localStorage.getItem("customApiKey") || undefined;
                    }
                } catch (e) {
                    console.error("Failed to parse customLLMSettings during run", e);
                }
            }
            return runExperiment(id, customBaseUrl, customApiKey);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["experiment", id] });
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
        },
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
        queued: "badge-queued",
        running: "badge-running",
        completed: "badge-completed",
        failed: "badge-failed",
    };

    const canRun = experiment.status === "pending" || experiment.status === "failed" || experiment.status === "completed";
    const isActive = experiment.status === "running" || experiment.status === "queued";

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-primary hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <div className="flex items-center justify-between mt-1">
                        <div className="flex items-center gap-3">
                            <h1 className="text-2xl font-serif text-(--text-heading)">
                                {experiment.name}
                            </h1>
                            <span className={`text-xs px-2 py-1 rounded-full ${statusClasses[experiment.status]}`}>
                                {experiment.status}
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            {/* Compare Button (completed only) */}
                            {experiment.status === "completed" && (
                                <Link
                                    href={`/experiments/compare?preselect=${id}`}
                                    className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium border border-border rounded-full text-(--text-body) hover:bg-(--bg-page) transition-colors"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M18 20V10" /><path d="M12 20V4" /><path d="M6 20v-6" />
                                    </svg>
                                    Compare
                                </Link>
                            )}
                            {/* Run Button */}
                            {canRun && (
                                <button
                                    onClick={() => runMutation.mutate()}
                                    disabled={runMutation.isPending}
                                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer inline-flex items-center gap-2"
                                >
                                    {runMutation.isPending ? (
                                        <>
                                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                            </svg>
                                            Starting...
                                        </>
                                    ) : (
                                        <>
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                                <polygon points="5,3 19,12 5,21" />
                                            </svg>
                                            Run Experiment
                                        </>
                                    )}
                                </button>
                            )}
                            {/* Active Indicator */}
                            {isActive && (
                                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 text-blue-700 text-sm font-medium">
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Running...
                                </div>
                            )}
                        </div>
                    </div>
                    {experiment.description && (
                        <p className="mt-1 text-(--text-muted)">{experiment.description}</p>
                    )}
                </div>
            </header>

            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-7xl mx-auto px-4 py-8"
            >
                {/* Run Error Display */}
                {runMutation.error && (
                    <div className="card p-4 mb-6 border-l-4 border-l-(--error)">
                        <p className="text-(--error) text-sm">
                            Failed to start: {runMutation.error instanceof Error ? runMutation.error.message : "Unknown error"}
                        </p>
                    </div>
                )}

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
                                {isActive
                                    ? "Experiment is running... Results will appear automatically."
                                    : canRun
                                        ? "Click \"Run Experiment\" above to start execution."
                                        : "Run the experiment to see results."}
                            </p>
                        </div>
                    )}
                </div>

                {/* Optimization Profile (Phase 8) */}
                {experiment.status === "completed" && (
                    <div className="mt-6">
                        <ProfileDashboard experimentId={id} />
                    </div>
                )}
            </motion.main>
        </div>
    );
}
