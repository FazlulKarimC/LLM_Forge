"use client";

/**
 * Experiment Comparison Page
 * 
 * Side-by-side comparison of two or more experiments.
 * Shows metrics differences, statistical significance, and per-example analysis.
 * Phase 4: Chain-of-Thought comparison view.
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
        <div className="card p-4">
            <p className="text-xs text-(--text-muted) uppercase tracking-wide mb-1">{label}</p>
            <div className="flex items-end gap-3">
                <span className="text-lg font-semibold text-(--text-heading)">{formatValue(valueA)}</span>
                <span className="text-(--text-muted)">→</span>
                <span className="text-lg font-semibold text-(--text-heading)">{formatValue(valueB)}</span>
                {diffText && (
                    <span className={`text-sm font-medium ml-2 ${isImprovement ? "text-green-600" : isRegression ? "text-red-600" : "text-(--text-muted)"
                        }`}>
                        {diffText}
                    </span>
                )}
            </div>
        </div>
    );
}

// =============================================================================
// Agreement Donut
// =============================================================================

function AgreementSummary({ summary }: { summary: StatisticalComparison["summary"] }) {
    const total = summary.both_correct + summary.both_wrong + summary.a_only_correct + summary.b_only_correct;
    if (total === 0) return null;

    const pct = (v: number) => `${((v / total) * 100).toFixed(1)}%`;

    return (
        <div className="card p-6">
            <h3 className="text-lg font-serif text-(--text-heading) mb-4">Agreement Analysis</h3>
            <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 rounded-lg" style={{ background: "#dcfce7" }}>
                    <p className="text-2xl font-semibold" style={{ color: "#166534" }}>
                        {summary.both_correct}
                    </p>
                    <p className="text-xs" style={{ color: "#166534" }}>Both Correct ({pct(summary.both_correct)})</p>
                </div>
                <div className="text-center p-3 rounded-lg" style={{ background: "#fee2e2" }}>
                    <p className="text-2xl font-semibold" style={{ color: "#991b1b" }}>
                        {summary.both_wrong}
                    </p>
                    <p className="text-xs" style={{ color: "#991b1b" }}>Both Wrong ({pct(summary.both_wrong)})</p>
                </div>
                <div className="text-center p-3 rounded-lg" style={{ background: "#fef3c7" }}>
                    <p className="text-2xl font-semibold" style={{ color: "#92400e" }}>
                        {summary.a_only_correct}
                    </p>
                    <p className="text-xs" style={{ color: "#92400e" }}>Only A Correct ({pct(summary.a_only_correct)})</p>
                </div>
                <div className="text-center p-3 rounded-lg" style={{ background: "#dbeafe" }}>
                    <p className="text-2xl font-semibold" style={{ color: "#1e40af" }}>
                        {summary.b_only_correct}
                    </p>
                    <p className="text-xs" style={{ color: "#1e40af" }}>Only B Correct ({pct(summary.b_only_correct)})</p>
                </div>
            </div>
        </div>
    );
}

// =============================================================================
// Statistical Significance Card
// =============================================================================

function SignificanceCard({ stats }: { stats: StatisticalComparison }) {
    return (
        <div className="card p-6">
            <h3 className="text-lg font-serif text-(--text-heading) mb-4">Statistical Significance</h3>
            <div className="space-y-4">
                <div className="flex items-center gap-3">
                    <span className={`inline-block w-3 h-3 rounded-full ${stats.mcnemar.is_significant ? "bg-green-500" : "bg-yellow-500"
                        }`} />
                    <span className="text-(--text-body)">
                        {stats.mcnemar.is_significant
                            ? "Statistically significant difference (p < 0.05)"
                            : "Not statistically significant (p ≥ 0.05)"}
                    </span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <p className="text-(--text-muted)">McNemar&apos;s p-value</p>
                        <p className="font-mono font-semibold text-(--text-heading)">
                            {stats.mcnemar.p_value.toFixed(4)}
                        </p>
                    </div>
                    <div>
                        <p className="text-(--text-muted)">Common Examples</p>
                        <p className="font-mono font-semibold text-(--text-heading)">
                            {stats.num_common_examples}
                        </p>
                    </div>
                    <div>
                        <p className="text-(--text-muted)">Accuracy A (95% CI)</p>
                        <p className="font-mono text-(--text-heading)">
                            {(stats.accuracy_a * 100).toFixed(1)}%
                            <span className="text-(--text-muted) text-xs ml-1">
                                [{(stats.bootstrap_ci_a.lower * 100).toFixed(1)}%, {(stats.bootstrap_ci_a.upper * 100).toFixed(1)}%]
                            </span>
                        </p>
                    </div>
                    <div>
                        <p className="text-(--text-muted)">Accuracy B (95% CI)</p>
                        <p className="font-mono text-(--text-heading)">
                            {(stats.accuracy_b * 100).toFixed(1)}%
                            <span className="text-(--text-muted) text-xs ml-1">
                                [{(stats.bootstrap_ci_b.lower * 100).toFixed(1)}%, {(stats.bootstrap_ci_b.upper * 100).toFixed(1)}%]
                            </span>
                        </p>
                    </div>
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
            <div className="card p-6 text-center text-(--text-muted)">
                Both experiments agree on all examples.
            </div>
        );
    }

    return (
        <div className="card overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
                <h3 className="text-lg font-serif text-(--text-heading)">
                    Per-Example Differences ({diffs.length})
                </h3>
                <p className="text-sm text-(--text-muted)">Examples where the two methods disagree</p>
            </div>
            <table className="min-w-full divide-y divide-border">
                <thead className="bg-(--bg-page)">
                    <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-(--text-muted) uppercase">Example</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-(--text-muted) uppercase">{nameA}</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-(--text-muted) uppercase">{nameB}</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-(--text-muted) uppercase">Expected</th>
                    </tr>
                </thead>
                <tbody className="bg-(--bg-card) divide-y divide-border">
                    {diffs.map((d) => (
                        <tr
                            key={d.example_id}
                            className="hover:bg-(--bg-page) cursor-pointer"
                            onClick={() => setExpanded(expanded === d.example_id ? null : d.example_id)}
                        >
                            <td className="px-4 py-3 text-sm font-mono text-(--text-body)">{d.example_id}</td>
                            <td className="px-4 py-3">
                                <span className={`text-xs px-2 py-0.5 rounded-full ${d.a_correct ? "badge-completed" : "badge-failed"
                                    }`}>
                                    {d.a_correct ? "✓" : "✗"}
                                </span>
                                {expanded === d.example_id && (
                                    <p className="mt-2 text-xs text-(--text-body) max-w-xs break-words">{d.a_output}</p>
                                )}
                            </td>
                            <td className="px-4 py-3">
                                <span className={`text-xs px-2 py-0.5 rounded-full ${d.b_correct ? "badge-completed" : "badge-failed"
                                    }`}>
                                    {d.b_correct ? "✓" : "✗"}
                                </span>
                                {expanded === d.example_id && (
                                    <p className="mt-2 text-xs text-(--text-body) max-w-xs break-words">{d.b_output}</p>
                                )}
                            </td>
                            <td className="px-4 py-3 text-sm text-(--text-muted) max-w-xs truncate">{d.expected}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// =============================================================================
// Main Page
// =============================================================================

export default function ComparePage() {
    const [selectedIds, setSelectedIds] = useState<string[]>([]);

    // Fetch all completed experiments for selection
    const { data: expData } = useQuery({
        queryKey: ["experiments", "completed"],
        queryFn: () => listExperiments({ status: "completed", limit: 50 }),
    });

    const experiments = expData?.experiments ?? [];

    // Fetch comparison when 2 experiments selected
    const { data: comparison, isLoading: compLoading } = useQuery({
        queryKey: ["comparison", selectedIds],
        queryFn: () => compareExperiments(selectedIds),
        enabled: selectedIds.length >= 2,
    });

    // Fetch statistical comparison when exactly 2 selected
    const { data: stats, isLoading: statsLoading } = useQuery({
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
                    : [prev[1], id]
        );
    };

    const expA = comparison?.experiments[0];
    const expB = comparison?.experiments[1];

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
                    <div>
                        <Link href="/experiments" className="text-primary hover:underline text-sm">
                            ← Back to Experiments
                        </Link>
                        <h1 className="text-2xl font-serif text-(--text-heading) mt-1">
                            Compare Experiments
                        </h1>
                        <p className="text-sm text-(--text-muted) mt-1">
                            Select 2 completed experiments to compare side-by-side
                        </p>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* Experiment Selector */}
                <div className="card p-6 mb-8">
                    <h2 className="text-lg font-serif text-(--text-heading) mb-4">Select Experiments</h2>
                    {experiments.length === 0 ? (
                        <p className="text-(--text-muted)">No completed experiments found. Run some experiments first.</p>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {experiments.map((exp: Experiment) => {
                                const isSelected = selectedIds.includes(exp.id);
                                const index = selectedIds.indexOf(exp.id);
                                return (
                                    <button
                                        key={exp.id}
                                        onClick={() => toggleSelect(exp.id)}
                                        className={`p-4 rounded-xl border-2 text-left transition-all cursor-pointer ${isSelected
                                                ? "border-primary bg-(--bg-primary) bg-opacity-5"
                                                : "border-border hover:border-primary/40"
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="font-medium text-(--text-heading) text-sm">{exp.name}</span>
                                            {isSelected && (
                                                <span className="text-xs bg-primary text-(--text-on-dark) px-2 py-0.5 rounded-full">
                                                    {index === 0 ? "A" : "B"}
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xs text-(--text-muted)">
                                            {exp.config.reasoning_method.toUpperCase()} • {exp.config.model_name}
                                        </p>
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Results Area */}
                {selectedIds.length < 2 ? (
                    <div className="card p-12 text-center text-(--text-muted)">
                        Select 2 experiments above to see the comparison
                    </div>
                ) : compLoading || statsLoading ? (
                    <div className="card p-12 text-center">
                        <div className="animate-pulse text-(--text-muted)">Loading comparison data...</div>
                    </div>
                ) : comparison && expA && expB ? (
                    <div className="space-y-8">
                        {/* Experiment Labels */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className="card p-4 border-l-4 border-l-primary">
                                <p className="text-xs text-(--text-muted) uppercase">Experiment A</p>
                                <p className="font-semibold text-(--text-heading)">{expA.experiment_name}</p>
                                <p className="text-sm text-(--text-muted)">{expA.method.toUpperCase()} • {expA.model}</p>
                            </div>
                            <div className="card p-4 border-l-4 border-l-blue-500">
                                <p className="text-xs text-(--text-muted) uppercase">Experiment B</p>
                                <p className="font-semibold text-(--text-heading)">{expB.experiment_name}</p>
                                <p className="text-sm text-(--text-muted)">{expB.method.toUpperCase()} • {expB.model}</p>
                            </div>
                        </div>

                        {/* Metrics Comparison Grid */}
                        <div>
                            <h2 className="text-xl font-serif text-(--text-heading) mb-4">Metrics Comparison</h2>
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
                                    valueA={expA.metrics.cost.total_tokens_input + expA.metrics.cost.total_tokens_output}
                                    valueB={expB.metrics.cost.total_tokens_input + expB.metrics.cost.total_tokens_output}
                                    format="number"
                                    higherIsBetter={false}
                                />
                            </div>
                        </div>

                        {/* Statistical Analysis */}
                        {stats && (
                            <>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <SignificanceCard stats={stats} />
                                    <AgreementSummary summary={stats.summary} />
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
