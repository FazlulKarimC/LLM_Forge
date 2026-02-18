"use client";

/**
 * Experiment Comparison Page
 *
 * Side-by-side comparison of two experiments.
 * Shows metrics differences, statistical significance, and per-example analysis.
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import {
    listExperiments,
    compareExperiments,
    getStatisticalComparison,
    type Experiment,
    type ComparisonResponse,
    type StatisticalComparison,
} from "@/lib/api";

// =============================================================================
// Metric Comparison Card
// =============================================================================

function MetricCompareCard({
    label,
    valueA,
    valueB,
    format = "percent",
    higherIsBetter = true,
}: {
    label: string;
    valueA?: number;
    valueB?: number;
    format?: "percent" | "ms" | "number";
    higherIsBetter?: boolean;
}) {
    const formatValue = (v?: number) => {
        if (v == null) return "—";
        if (format === "percent") return `${(v * 100).toFixed(1)}%`;
        if (format === "ms") return `${v.toFixed(0)}ms`;
        return v.toLocaleString();
    };

    const diff = valueA != null && valueB != null ? valueB - valueA : null;
    const isImprovement = diff != null && (higherIsBetter ? diff > 0 : diff < 0);
    const isRegression = diff != null && (higherIsBetter ? diff < 0 : diff > 0);

    const diffText = diff != null
        ? format === "percent"
            ? `${diff > 0 ? "+" : ""}${(diff * 100).toFixed(1)}%`
            : format === "ms"
                ? `${diff > 0 ? "+" : ""}${diff.toFixed(0)}ms`
                : `${diff > 0 ? "+" : ""}${diff.toFixed(2)}`
        : "";

    return (
        <div className="bg-(--bg-card) border border-border rounded-xl p-5 hover:shadow-md transition-shadow">
            <p className="text-xs font-medium text-(--text-muted) uppercase tracking-wider mb-3">{label}</p>
            <div className="flex items-baseline gap-2">
                <div className="flex items-baseline gap-1">
                    <span className="text-xs font-medium text-(--text-muted)">A</span>
                    <span className="text-xl font-semibold text-(--text-heading) font-mono">{formatValue(valueA)}</span>
                </div>
                <span className="text-(--text-muted) mx-1">vs</span>
                <div className="flex items-baseline gap-1">
                    <span className="text-xs font-medium text-(--text-muted)">B</span>
                    <span className="text-xl font-semibold text-(--text-heading) font-mono">{formatValue(valueB)}</span>
                </div>
            </div>
            {diffText && (
                <div className="mt-2 pt-2 border-t border-border">
                    <span className={`text-sm font-semibold font-mono ${isImprovement ? "text-green-600" : isRegression ? "text-red-500" : "text-(--text-muted)"
                        }`}>
                        {isImprovement ? "▲ " : isRegression ? "▼ " : ""}{diffText}
                    </span>
                    <span className="text-xs text-(--text-muted) ml-1">
                        {isImprovement ? "(better)" : isRegression ? "(worse)" : ""}
                    </span>
                </div>
            )}
        </div>
    );
}

// =============================================================================
// Agreement Summary
// =============================================================================

function AgreementSummary({ summary }: { summary: StatisticalComparison["summary"] }) {
    const total = summary.both_correct + summary.both_wrong + summary.a_only_correct + summary.b_only_correct;
    if (total === 0) return null;

    const pct = (v: number) => `${((v / total) * 100).toFixed(1)}%`;

    const items = [
        { label: "Both Correct", value: summary.both_correct, pct: pct(summary.both_correct), bg: "#dcfce7", color: "#166534", icon: "✓✓" },
        { label: "Both Wrong", value: summary.both_wrong, pct: pct(summary.both_wrong), bg: "#fee2e2", color: "#991b1b", icon: "✗✗" },
        { label: "Only A Correct", value: summary.a_only_correct, pct: pct(summary.a_only_correct), bg: "#fef3c7", color: "#92400e", icon: "✓✗" },
        { label: "Only B Correct", value: summary.b_only_correct, pct: pct(summary.b_only_correct), bg: "#dbeafe", color: "#1e40af", icon: "✗✓" },
    ];

    return (
        <div className="bg-(--bg-card) border border-border rounded-xl p-6">
            <h3 className="text-base font-serif text-(--text-heading) mb-4 flex items-center gap-2">
                <span className="text-lg">📊</span> Agreement Analysis
            </h3>
            <div className="grid grid-cols-2 gap-3">
                {items.map(item => (
                    <div key={item.label} className="text-center p-3 rounded-lg" style={{ background: item.bg }}>
                        <p className="text-xs font-mono mb-1" style={{ color: item.color, opacity: 0.7 }}>{item.icon}</p>
                        <p className="text-2xl font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
                        <p className="text-xs font-medium mt-0.5" style={{ color: item.color }}>{item.label}</p>
                        <p className="text-xs mt-0.5" style={{ color: item.color, opacity: 0.7 }}>{item.pct}</p>
                    </div>
                ))}
            </div>
            {/* Simple visual bar */}
            <div className="mt-4 flex rounded-full overflow-hidden h-2">
                {total > 0 && (
                    <>
                        <div style={{ width: pct(summary.both_correct), background: "#22c55e" }} />
                        <div style={{ width: pct(summary.a_only_correct), background: "#f59e0b" }} />
                        <div style={{ width: pct(summary.b_only_correct), background: "#3b82f6" }} />
                        <div style={{ width: pct(summary.both_wrong), background: "#ef4444" }} />
                    </>
                )}
            </div>
        </div>
    );
}

// =============================================================================
// Statistical Significance Card
// =============================================================================

function SignificanceCard({ stats }: { stats: StatisticalComparison }) {
    const significant = stats.mcnemar.is_significant;

    return (
        <div className="bg-(--bg-card) border border-border rounded-xl p-6">
            <h3 className="text-base font-serif text-(--text-heading) mb-4 flex items-center gap-2">
                <span className="text-lg">🔬</span> Statistical Significance
            </h3>

            {/* Verdict */}
            <div className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-4 ${significant ? "bg-green-50 border border-green-200" : "bg-yellow-50 border border-yellow-200"
                }`}>
                <span className={`inline-block w-2.5 h-2.5 rounded-full ${significant ? "bg-green-500" : "bg-yellow-500"}`} />
                <span className={`text-sm font-medium ${significant ? "text-green-800" : "text-yellow-800"}`}>
                    {significant
                        ? "Statistically significant difference (p < 0.05)"
                        : "Not statistically significant (p ≥ 0.05)"}
                </span>
            </div>

            {/* Metrics grid */}
            <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                <div>
                    <p className="text-(--text-muted) text-xs mb-0.5">McNemar&apos;s p-value</p>
                    <p className="font-mono font-semibold text-(--text-heading) text-base">
                        {stats.mcnemar.p_value < 0.0001 ? "< 0.0001" : stats.mcnemar.p_value.toFixed(4)}
                    </p>
                </div>
                <div>
                    <p className="text-(--text-muted) text-xs mb-0.5">Common Examples</p>
                    <p className="font-mono font-semibold text-(--text-heading) text-base">
                        {stats.num_common_examples}
                    </p>
                </div>
                <div>
                    <p className="text-(--text-muted) text-xs mb-0.5">Accuracy A (95% CI)</p>
                    <p className="font-mono text-(--text-heading)">
                        {(stats.accuracy_a * 100).toFixed(1)}%
                        <span className="text-(--text-muted) text-xs ml-1">
                            [{(stats.bootstrap_ci_a.lower * 100).toFixed(1)}, {(stats.bootstrap_ci_a.upper * 100).toFixed(1)}]
                        </span>
                    </p>
                </div>
                <div>
                    <p className="text-(--text-muted) text-xs mb-0.5">Accuracy B (95% CI)</p>
                    <p className="font-mono text-(--text-heading)">
                        {(stats.accuracy_b * 100).toFixed(1)}%
                        <span className="text-(--text-muted) text-xs ml-1">
                            [{(stats.bootstrap_ci_b.lower * 100).toFixed(1)}, {(stats.bootstrap_ci_b.upper * 100).toFixed(1)}]
                        </span>
                    </p>
                </div>
            </div>
        </div>
    );
}

// =============================================================================
// Per-Example Differences Table
// =============================================================================

function DifferencesTable({ diffs, nameA, nameB }: {
    diffs: StatisticalComparison["per_example_differences"];
    nameA: string;
    nameB: string;
}) {
    const [expanded, setExpanded] = useState<string | null>(null);

    if (!diffs.length) {
        return (
            <div className="bg-(--bg-card) border border-border rounded-xl p-8 text-center">
                <p className="text-(--text-muted) text-lg mb-1">🎯 Perfect Agreement</p>
                <p className="text-sm text-(--text-muted)">Both experiments agree on all examples.</p>
            </div>
        );
    }

    return (
        <div className="bg-(--bg-card) border border-border rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
                <h3 className="text-base font-serif text-(--text-heading) flex items-center gap-2">
                    <span className="text-lg">🔍</span> Per-Example Differences
                    <span className="text-xs font-normal text-(--text-muted) bg-(--bg-page) px-2 py-0.5 rounded-full ml-1">
                        {diffs.length} disagreements
                    </span>
                </h3>
                <p className="text-xs text-(--text-muted) mt-1">Click a row to expand model outputs</p>
            </div>
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border">
                    <thead className="bg-(--bg-page)">
                        <tr>
                            <th className="px-4 py-2.5 text-left text-xs font-semibold text-(--text-muted) uppercase tracking-wider">Example</th>
                            <th className="px-4 py-2.5 text-center text-xs font-semibold text-(--text-muted) uppercase tracking-wider">{nameA.length > 20 ? nameA.slice(0, 20) + "…" : nameA}</th>
                            <th className="px-4 py-2.5 text-center text-xs font-semibold text-(--text-muted) uppercase tracking-wider">{nameB.length > 20 ? nameB.slice(0, 20) + "…" : nameB}</th>
                            <th className="px-4 py-2.5 text-left text-xs font-semibold text-(--text-muted) uppercase tracking-wider">Expected</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {diffs.map((d) => (
                            <tr
                                key={d.example_id}
                                className="hover:bg-(--bg-page) cursor-pointer transition-colors"
                                onClick={() => setExpanded(expanded === d.example_id ? null : d.example_id)}
                            >
                                <td className="px-4 py-3 text-sm font-mono text-(--text-body)">{d.example_id}</td>
                                <td className="px-4 py-3 text-center">
                                    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${d.a_correct
                                        ? "bg-green-100 text-green-700"
                                        : "bg-red-100 text-red-700"
                                        }`}>
                                        {d.a_correct ? "✓" : "✗"}
                                    </span>
                                    {expanded === d.example_id && d.a_output && (
                                        <p className="mt-2 text-xs text-(--text-body) text-left max-w-xs break-words bg-(--bg-page) rounded p-2">{d.a_output}</p>
                                    )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${d.b_correct
                                        ? "bg-green-100 text-green-700"
                                        : "bg-red-100 text-red-700"
                                        }`}>
                                        {d.b_correct ? "✓" : "✗"}
                                    </span>
                                    {expanded === d.example_id && d.b_output && (
                                        <p className="mt-2 text-xs text-(--text-body) text-left max-w-xs break-words bg-(--bg-page) rounded p-2">{d.b_output}</p>
                                    )}
                                </td>
                                <td className="px-4 py-3 text-sm text-(--text-muted) max-w-xs truncate">{d.expected}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

// =============================================================================
// Main Page
// =============================================================================

export default function ComparePage() {
    const [selectedIds, setSelectedIds] = useState<string[]>([]);

    // Fetch all completed experiments for selection
    const { data: expData, isLoading: listLoading } = useQuery({
        queryKey: ["experiments", "completed"],
        queryFn: () => listExperiments({ status: "completed", limit: 50 }),
    });

    const experiments = expData?.experiments ?? [];

    // Fetch comparison when 2 experiments selected
    const { data: comparison, isLoading: compLoading, error: compError } = useQuery({
        queryKey: ["comparison", selectedIds],
        queryFn: () => compareExperiments(selectedIds),
        enabled: selectedIds.length >= 2,
    });

    // Fetch statistical comparison when exactly 2 selected
    const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
        queryKey: ["statistical", selectedIds],
        queryFn: () => getStatisticalComparison(selectedIds[0], selectedIds[1]),
        enabled: selectedIds.length === 2,
    });

    const toggleSelect = (id: string) => {
        setSelectedIds((prev) =>
            prev.includes(id)
                ? prev.filter((i) => i !== id)
                : prev.length < 2
                    ? [...prev, id]
                    : [prev[1], id] // Replace oldest selection
        );
    };

    const clearSelection = () => setSelectedIds([]);

    const expA = comparison?.experiments?.[0];
    const expB = comparison?.experiments?.[1];

    // Get full experiment objects for the selector labels
    const selectedExpA = experiments.find(e => e.id === selectedIds[0]);
    const selectedExpB = experiments.find(e => e.id === selectedIds[1]);

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-primary hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <div className="flex items-center justify-between mt-1">
                        <div>
                            <h1 className="text-2xl font-serif text-(--text-heading)">
                                Compare Experiments
                            </h1>
                            <p className="text-sm text-(--text-muted) mt-1">
                                Select 2 completed experiments to compare side-by-side
                            </p>
                        </div>
                        {selectedIds.length > 0 && (
                            <button
                                onClick={clearSelection}
                                className="text-sm text-(--text-muted) hover:text-(--text-body) border border-border rounded-lg px-3 py-1.5 hover:bg-(--bg-page) transition-colors cursor-pointer"
                            >
                                Clear Selection
                            </button>
                        )}
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
                {/* Experiment Selector */}
                <div className="bg-(--bg-card) border border-border rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-base font-serif text-(--text-heading)">Select Experiments</h2>
                        <span className="text-xs text-(--text-muted)">
                            {selectedIds.length}/2 selected
                        </span>
                    </div>

                    {listLoading ? (
                        <div className="text-center py-8">
                            <div className="animate-pulse text-(--text-muted)">Loading experiments...</div>
                        </div>
                    ) : experiments.length === 0 ? (
                        <div className="text-center py-8">
                            <p className="text-(--text-muted) text-lg mb-2">No completed experiments</p>
                            <p className="text-sm text-(--text-muted)">
                                Run some experiments first, then return here to compare results.
                            </p>
                            <Link
                                href="/experiments/new"
                                className="inline-block mt-4 bg-primary text-(--text-on-dark) rounded-full px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity"
                            >
                                Create Experiment
                            </Link>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {experiments.map((exp: Experiment) => {
                                const isSelected = selectedIds.includes(exp.id);
                                const index = selectedIds.indexOf(exp.id);
                                const label = index === 0 ? "A" : index === 1 ? "B" : null;
                                const borderColor = index === 0 ? "border-primary" : index === 1 ? "border-blue-500" : "";

                                return (
                                    <button
                                        key={exp.id}
                                        onClick={() => toggleSelect(exp.id)}
                                        className={`p-4 rounded-xl border-2 text-left transition-all cursor-pointer group ${isSelected
                                            ? `${borderColor} bg-opacity-5 shadow-sm`
                                            : "border-border hover:border-primary/40 hover:shadow-sm"
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-1.5">
                                            <span className="font-medium text-(--text-heading) text-sm truncate mr-2">
                                                {exp.name}
                                            </span>
                                            {label ? (
                                                <span className={`shrink-0 text-xs font-bold px-2.5 py-0.5 rounded-full ${index === 0
                                                    ? "bg-primary text-(--text-on-dark)"
                                                    : "bg-blue-500 text-white"
                                                    }`}>
                                                    {label}
                                                </span>
                                            ) : (
                                                <span className="shrink-0 text-xs text-(--text-muted) opacity-0 group-hover:opacity-100 transition-opacity">
                                                    Click to select
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xs text-(--text-muted)">
                                            <span className="inline-block bg-(--bg-page) rounded px-1.5 py-0.5 font-medium mr-1.5">
                                                {exp.config.reasoning_method.toUpperCase()}
                                            </span>
                                            {exp.config.model_name.split("/").pop()}
                                        </p>
                                        {exp.config.dataset_name && (
                                            <p className="text-xs text-(--text-muted) mt-1 opacity-70">
                                                📋 {exp.config.dataset_name}
                                            </p>
                                        )}
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Results Area */}
                {selectedIds.length < 2 ? (
                    <div className="bg-(--bg-card) border-2 border-dashed border-border rounded-xl p-12 text-center">
                        <p className="text-4xl mb-3 opacity-40">⚖️</p>
                        <p className="text-(--text-muted) text-lg font-medium">
                            {selectedIds.length === 0
                                ? "Select 2 experiments to compare"
                                : "Select 1 more experiment"
                            }
                        </p>
                        <p className="text-sm text-(--text-muted) mt-1">
                            Click the experiment cards above to select them
                        </p>
                    </div>
                ) : compLoading || statsLoading ? (
                    <div className="bg-(--bg-card) border border-border rounded-xl p-12 text-center">
                        <div className="inline-block animate-spin text-3xl mb-3">⚙️</div>
                        <p className="text-(--text-muted) font-medium">Loading comparison data...</p>
                    </div>
                ) : (compError || statsError) ? (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
                        <p className="text-red-700 font-medium mb-1">Failed to load comparison</p>
                        <p className="text-red-600 text-sm">
                            {compError instanceof Error ? compError.message
                                : statsError instanceof Error ? statsError.message
                                    : String(compError || statsError || "Unknown error")}
                        </p>
                        <p className="text-xs text-red-500 mt-2">
                            Make sure both experiments have completed with results.
                        </p>
                    </div>
                ) : comparison && expA && expB ? (
                    <div className="space-y-8">
                        {/* Experiment Labels */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-(--bg-card) border border-border rounded-xl p-5 border-l-4 border-l-primary">
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-bold bg-primary text-(--text-on-dark) px-2 py-0.5 rounded-full">A</span>
                                    <p className="text-xs font-medium text-(--text-muted) uppercase tracking-wide">Experiment A</p>
                                </div>
                                <p className="font-semibold text-(--text-heading) text-lg">{expA.experiment_name}</p>
                                <p className="text-sm text-(--text-muted) mt-1">
                                    <span className="inline-block bg-(--bg-page) rounded px-1.5 py-0.5 font-medium text-xs mr-1.5">
                                        {expA.method.toUpperCase()}
                                    </span>
                                    {expA.model.split("/").pop()}
                                </p>
                                {selectedExpA?.config.dataset_name && (
                                    <p className="text-xs text-(--text-muted) mt-1">📋 {selectedExpA.config.dataset_name}</p>
                                )}
                            </div>
                            <div className="bg-(--bg-card) border border-border rounded-xl p-5 border-l-4 border-l-blue-500">
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-bold bg-blue-500 text-white px-2 py-0.5 rounded-full">B</span>
                                    <p className="text-xs font-medium text-(--text-muted) uppercase tracking-wide">Experiment B</p>
                                </div>
                                <p className="font-semibold text-(--text-heading) text-lg">{expB.experiment_name}</p>
                                <p className="text-sm text-(--text-muted) mt-1">
                                    <span className="inline-block bg-(--bg-page) rounded px-1.5 py-0.5 font-medium text-xs mr-1.5">
                                        {expB.method.toUpperCase()}
                                    </span>
                                    {expB.model.split("/").pop()}
                                </p>
                                {selectedExpB?.config.dataset_name && (
                                    <p className="text-xs text-(--text-muted) mt-1">📋 {selectedExpB.config.dataset_name}</p>
                                )}
                            </div>
                        </div>

                        {/* Metrics Comparison Grid */}
                        <div>
                            <h2 className="text-xl font-serif text-(--text-heading) mb-4 flex items-center gap-2">
                                <span>📈</span> Metrics Comparison
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                <MetricCompareCard
                                    label="Exact Match Accuracy"
                                    valueA={expA.metrics.quality.accuracy_exact}
                                    valueB={expB.metrics.quality.accuracy_exact}
                                />
                                <MetricCompareCard
                                    label="F1 Score"
                                    valueA={expA.metrics.quality.accuracy_f1}
                                    valueB={expB.metrics.quality.accuracy_f1}
                                />
                                <MetricCompareCard
                                    label="Substring Accuracy"
                                    valueA={expA.metrics.quality.accuracy_substring}
                                    valueB={expB.metrics.quality.accuracy_substring}
                                />
                                <MetricCompareCard
                                    label="Latency p50"
                                    valueA={expA.metrics.performance.latency_p50}
                                    valueB={expB.metrics.performance.latency_p50}
                                    format="ms"
                                    higherIsBetter={false}
                                />
                                <MetricCompareCard
                                    label="Latency p95"
                                    valueA={expA.metrics.performance.latency_p95}
                                    valueB={expB.metrics.performance.latency_p95}
                                    format="ms"
                                    higherIsBetter={false}
                                />
                                <MetricCompareCard
                                    label="Total Tokens"
                                    valueA={(expA.metrics.cost.total_tokens_input || 0) + (expA.metrics.cost.total_tokens_output || 0)}
                                    valueB={(expB.metrics.cost.total_tokens_input || 0) + (expB.metrics.cost.total_tokens_output || 0)}
                                    format="number"
                                    higherIsBetter={false}
                                />
                            </div>
                        </div>

                        {/* Statistical Analysis */}
                        {stats && (
                            <>
                                <div>
                                    <h2 className="text-xl font-serif text-(--text-heading) mb-4 flex items-center gap-2">
                                        <span>📊</span> Statistical Analysis
                                    </h2>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <SignificanceCard stats={stats} />
                                        <AgreementSummary summary={stats.summary} />
                                    </div>
                                </div>

                                <DifferencesTable
                                    diffs={stats.per_example_differences}
                                    nameA={expA.experiment_name}
                                    nameB={expB.experiment_name}
                                />
                            </>
                        )}
                    </div>
                ) : null}
            </main>
        </div>
    );
}
