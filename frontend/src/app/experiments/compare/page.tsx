"use client";

import { useQuery } from "@tanstack/react-query";
import { Fragment, Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  FlaskConical,
  GitCompareArrows,
  ScanSearch,
  Sigma,
} from "lucide-react";

import {
  compareExperiments,
  getStatisticalComparison,
  listExperiments,
  type Experiment,
  type StatisticalComparison,
} from "@/lib/api";
import { EmptyState, MetricBar, PageHeader, Panel, PanelHeader, SkeletonBlock, StatusPill } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

function formatMetric(value?: number, format: "percent" | "ms" | "number" = "percent") {
  if (value == null) return "--";
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "ms") return `${value.toFixed(0)} ms`;
  return value.toLocaleString();
}

function getValidSelection(selectedIds: string[], experimentIds: Set<string>) {
  return selectedIds.filter((id) => experimentIds.has(id)).slice(0, 2);
}

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
  const diff = valueA != null && valueB != null ? valueB - valueA : null;
  const isPositive = diff != null && (higherIsBetter ? diff > 0 : diff < 0);
  const tone = diff == null ? "" : isPositive ? "text-[var(--success)]" : "text-[color:color-mix(in_oklab,var(--destructive)_84%,white_12%)]";
  const percentage = diff == null ? 0 : Math.min(100, Math.max(0, Math.abs(diff) * (format === "percent" ? 100 : 10)));

  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-[14px] border border-[var(--border)] bg-[var(--surface-2)] p-3">
          <div className="section-label">A</div>
          <div className="metric-value mt-2 text-2xl">{formatMetric(valueA, format)}</div>
        </div>
        <div className="rounded-[14px] border border-[var(--border)] bg-[var(--surface-2)] p-3">
          <div className="section-label">B</div>
          <div className="metric-value mt-2 text-2xl">{formatMetric(valueB, format)}</div>
        </div>
      </div>
      {diff != null ? (
        <div className="space-y-2">
          <div className={cn("metric-value text-sm", tone)}>
            {format === "percent"
              ? `${diff > 0 ? "+" : ""}${(diff * 100).toFixed(1)}%`
              : format === "ms"
                ? `${diff > 0 ? "+" : ""}${diff.toFixed(0)} ms`
                : `${diff > 0 ? "+" : ""}${diff.toFixed(2)}`}
          </div>
          <MetricBar value={percentage} />
        </div>
      ) : null}
    </div>
  );
}

function AgreementSummary({ summary }: { summary: StatisticalComparison["summary"] }) {
  const total = summary.both_correct + summary.both_wrong + summary.a_only_correct + summary.b_only_correct;
  if (!total) return null;

  const segments = [
    { label: "Both correct", value: summary.both_correct, color: "var(--success)" },
    { label: "A only", value: summary.a_only_correct, color: "var(--primary)" },
    { label: "B only", value: summary.b_only_correct, color: "var(--accent)" },
    { label: "Both wrong", value: summary.both_wrong, color: "color-mix(in oklab, var(--destructive) 84%, white 12%)" },
  ];

  return (
    <Panel>
      <PanelHeader label="Agreement" title="Outcome distribution" description="How often the experiments agree, disagree, or both fail on the same examples." />
      <div className="panel-body space-y-4">
        <div className="flex h-3 overflow-hidden rounded-full bg-[var(--muted)]">
          {segments.map((segment) => (
            <div key={segment.label} style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }} />
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {segments.map((segment) => (
            <div key={segment.label} className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
              <div className="section-label">{segment.label}</div>
              <div className="metric-value mt-2 text-2xl">{segment.value}</div>
              <div className="metric-caption mt-1">{((segment.value / total) * 100).toFixed(1)}% of common examples</div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function SignificancePanel({ stats }: { stats: StatisticalComparison }) {
  return (
    <Panel>
      <PanelHeader
        label="Significance"
        title="Statistical view"
        description="McNemar and bootstrap outputs stay visible so the winner is defensible in an interview, not just visually obvious."
      />
      <div className="panel-body space-y-4">
        <div className={cn("alert", stats.mcnemar.is_significant ? "alert-info" : "alert-warning")}>
          <Sigma className="mt-0.5 size-4 shrink-0" />
          <div className="space-y-1">
            <div className="font-semibold">
              {stats.mcnemar.is_significant ? "Difference is statistically significant" : "Difference is not statistically significant"}
            </div>
            <p className="text-sm text-[var(--muted-foreground)]">McNemar p-value {stats.mcnemar.p_value < 0.0001 ? "< 0.0001" : stats.mcnemar.p_value.toFixed(4)}</p>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="metric-card">
            <div className="metric-label">Accuracy A</div>
            <div className="metric-value text-3xl">{(stats.accuracy_a * 100).toFixed(1)}%</div>
            <div className="metric-caption">95% CI {(stats.accuracy_ci_a ?? stats.bootstrap_ci_a).lower.toFixed(2)} to {(stats.accuracy_ci_a ?? stats.bootstrap_ci_a).upper.toFixed(2)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Accuracy B</div>
            <div className="metric-value text-3xl">{(stats.accuracy_b * 100).toFixed(1)}%</div>
            <div className="metric-caption">95% CI {(stats.accuracy_ci_b ?? stats.bootstrap_ci_b).lower.toFixed(2)} to {(stats.accuracy_ci_b ?? stats.bootstrap_ci_b).upper.toFixed(2)}</div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function DifferencesTable({
  diffs,
  nameA,
  nameB,
}: {
  diffs: StatisticalComparison["per_example_differences"];
  nameA: string;
  nameB: string;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!diffs.length) {
    return (
      <EmptyState
        icon={<ScanSearch className="size-5" />}
        title="No disagreements found"
        description="Both experiments produced identical correctness outcomes across the common example set."
      />
    );
  }

  return (
    <Panel>
      <PanelHeader
        label="Disagreements"
        title={`Per-example review (${diffs.length})`}
        description="Expand a row to compare both outputs against the expected answer."
      />
      <div className="panel-body overflow-x-auto">
        <table className="data-table min-w-[780px]">
          <thead>
            <tr>
              <th>Example</th>
              <th>{nameA}</th>
              <th>{nameB}</th>
              <th>Expected answer</th>
            </tr>
          </thead>
          <tbody>
            {diffs.map((diff) => {
              const isExpanded = expanded === diff.example_id;
              return (
                <Fragment key={diff.example_id}>
                  <tr className="data-row cursor-pointer" onClick={() => setExpanded(isExpanded ? null : diff.example_id)}>
                    <td className="font-mono text-xs">{diff.example_id}</td>
                    <td>
                      <span className={cn("status-pill", diff.a_correct ? "status-completed" : "status-failed")}>{diff.a_correct ? "Correct" : "Incorrect"}</span>
                    </td>
                    <td>
                      <span className={cn("status-pill", diff.b_correct ? "status-completed" : "status-failed")}>{diff.b_correct ? "Correct" : "Incorrect"}</span>
                    </td>
                    <td className="text-sm text-[var(--muted-foreground)]">{diff.expected}</td>
                  </tr>
                  {isExpanded ? (
                    <tr>
                      <td colSpan={4} className="!p-0">
                        <div className="grid gap-3 border-b border-[var(--border)] bg-[var(--surface-2)] p-4 lg:grid-cols-3">
                          <div className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-1)] p-4">
                            <div className="section-label">{nameA}</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{diff.a_output || "No output"}</p>
                          </div>
                          <div className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-1)] p-4">
                            <div className="section-label">{nameB}</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{diff.b_output || "No output"}</p>
                          </div>
                          <div className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-1)] p-4">
                            <div className="section-label">Expected</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{diff.expected}</p>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function ComparePageInner() {
  const searchParams = useSearchParams();
  const preselect = searchParams.get("preselect");
  const [selectedIds, setSelectedIds] = useState<string[]>(() => (preselect ? [preselect] : []));

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "completed"],
    queryFn: ({ signal }) => listExperiments({ status: "completed", limit: 50 }, { signal }),
  });

  const experiments = useMemo(() => experimentsQuery.data?.experiments ?? [], [experimentsQuery.data?.experiments]);
  const experimentIds = useMemo(() => new Set(experiments.map((experiment) => experiment.id)), [experiments]);
  const effectiveSelectedIds = useMemo(() => getValidSelection(selectedIds, experimentIds), [selectedIds, experimentIds]);

  const comparisonQuery = useQuery({
    queryKey: ["comparison", effectiveSelectedIds],
    queryFn: ({ signal }) => compareExperiments(effectiveSelectedIds, { signal }),
    enabled: effectiveSelectedIds.length === 2,
  });

  const statsQuery = useQuery({
    queryKey: ["statistical", effectiveSelectedIds],
    queryFn: ({ signal }) => getStatisticalComparison(effectiveSelectedIds[0], effectiveSelectedIds[1], { signal }),
    enabled: effectiveSelectedIds.length === 2,
  });

  const selectedA = experiments.find((experiment) => experiment.id === effectiveSelectedIds[0]);
  const selectedB = experiments.find((experiment) => experiment.id === effectiveSelectedIds[1]);
  const comparison = comparisonQuery.data;
  const expA = comparison?.experiments[0];
  const expB = comparison?.experiments[1];

  function toggleSelection(id: string) {
    setSelectedIds((current) => {
      const base = getValidSelection(current, experimentIds);

      if (base.includes(id)) {
        return base.filter((item) => item !== id);
      }

      if (base.length < 2) {
        return [...base, id];
      }

      return [base[1], id];
    });
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={<><GitCompareArrows className="size-3.5" /> Comparison workspace</>}
        title="Judge experiments side by side."
        description="Use the same shell to inspect metrics, significance, and disagreement examples without inventing new backend requirements."
        actions={<Link href="/experiments" className="btn-secondary">Back to experiments</Link>}
      />

      <section className="grid gap-4 xl:grid-cols-[0.82fr_1.18fr]">
        <Panel>
          <PanelHeader
            label="Selection"
            title="Choose two completed runs"
            description="The oldest selected item is replaced when you choose a third candidate."
            actions={effectiveSelectedIds.length ? <button type="button" className="btn-secondary" onClick={() => setSelectedIds([])}>Clear</button> : null}
          />
          <div className="panel-body">
            {experimentsQuery.isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, index) => <SkeletonBlock key={index} className="h-[96px]" />)}
              </div>
            ) : experiments.length === 0 ? (
              <EmptyState
                icon={<FlaskConical className="size-5" />}
                title="No completed experiments"
                description="Finish at least two runs before opening the comparison workspace."
                action={<Link href="/experiments/new" className="btn-primary">Create experiment</Link>}
              />
            ) : (
              <div className="space-y-3">
                {experiments.map((experiment: Experiment) => {
                  const selectedIndex = effectiveSelectedIds.indexOf(experiment.id);
                  const active = selectedIndex !== -1;
                  const slot = selectedIndex === 0 ? "A" : selectedIndex === 1 ? "B" : null;

                  return (
                    <button
                      key={experiment.id}
                      type="button"
                      onClick={() => toggleSelection(experiment.id)}
                      className={cn(
                        "w-full rounded-[20px] border p-4 text-left transition-all",
                        active
                          ? "border-[color:color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color:color-mix(in_oklab,var(--accent)_10%,transparent)]"
                          : "border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--border-strong)] hover:bg-[var(--surface-3)]"
                      )}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-2">
                          <div className="text-base font-semibold tracking-[-0.03em]">{experiment.name}</div>
                          <div className="flex flex-wrap gap-2 text-xs text-[var(--muted-foreground)]">
                            <span className="chip">{experiment.config.reasoning_method.toUpperCase()}</span>
                            <span className="chip">{experiment.config.model_name.split("/").pop()}</span>
                            <span className="chip">{experiment.config.dataset_name}</span>
                          </div>
                        </div>
                        {slot ? <span className="status-pill status-running">Slot {slot}</span> : <StatusPill status={experiment.status} />}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </Panel>

        {effectiveSelectedIds.length < 2 ? (
          <EmptyState
            icon={<GitCompareArrows className="size-5" />}
            title={effectiveSelectedIds.length === 1 ? "Select one more experiment" : "Select two experiments to compare"}
            description="Once two completed runs are selected, the metrics, significance, and disagreement analysis will render here."
          />
        ) : comparisonQuery.isLoading || statsQuery.isLoading ? (
          <div className="space-y-4">
            <SkeletonBlock className="h-[180px]" />
            <SkeletonBlock className="h-[220px]" />
            <SkeletonBlock className="h-[320px]" />
          </div>
        ) : comparisonQuery.error || statsQuery.error || !comparison || !expA || !expB ? (
          <div className="alert alert-danger">
            <ScanSearch className="mt-0.5 size-4 shrink-0" />
            <div className="space-y-1">
              <div className="font-semibold">Comparison failed to load</div>
              <p className="text-sm text-[var(--muted-foreground)]">
                {comparisonQuery.error instanceof Error
                  ? comparisonQuery.error.message
                  : statsQuery.error instanceof Error
                    ? statsQuery.error.message
                    : "The comparison response was incomplete."}
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Panel>
              <PanelHeader
                label="Selected pair"
                title="Active comparison"
                description="The view keeps the model, method, and dataset context attached to the metrics so the screenshot explains itself."
              />
              <div className="panel-body grid gap-3 lg:grid-cols-2">
                {[selectedA, selectedB].map((experiment, index) => (
                  <div
                    key={experiment?.id ?? index}
                    className={cn(
                      "rounded-[18px] border p-4",
                      index === 0
                        ? "border-[color:color-mix(in_oklab,var(--primary)_34%,transparent)] bg-[color:color-mix(in_oklab,var(--primary)_10%,transparent)]"
                        : "border-[color:color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color:color-mix(in_oklab,var(--accent)_10%,transparent)]"
                    )}
                  >
                    <div className="section-label">Experiment {index === 0 ? "A" : "B"}</div>
                    <div className="mt-2 text-xl font-semibold tracking-[-0.04em]">{experiment?.name}</div>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-[var(--muted-foreground)]">
                      <span className="chip">{experiment?.config.reasoning_method.toUpperCase()}</span>
                      <span className="chip">{experiment?.config.model_name.split("/").pop()}</span>
                      <span className="chip">{experiment?.config.dataset_name}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel>
              <PanelHeader label="Metrics" title="Head-to-head performance" description="Compare quality, latency, and cost without leaving the active pair." />
              <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                <MetricCompareCard label="Exact match accuracy" valueA={expA.metrics.quality.accuracy_exact} valueB={expB.metrics.quality.accuracy_exact} />
                <MetricCompareCard label="F1 score" valueA={expA.metrics.quality.accuracy_f1} valueB={expB.metrics.quality.accuracy_f1} />
                <MetricCompareCard label="Substring accuracy" valueA={expA.metrics.quality.accuracy_substring} valueB={expB.metrics.quality.accuracy_substring} />
                <MetricCompareCard label="Latency p50" valueA={expA.metrics.performance.latency_p50} valueB={expB.metrics.performance.latency_p50} format="ms" higherIsBetter={false} />
                <MetricCompareCard label="Latency p95" valueA={expA.metrics.performance.latency_p95} valueB={expB.metrics.performance.latency_p95} format="ms" higherIsBetter={false} />
                <MetricCompareCard
                  label="Total tokens"
                  valueA={(expA.metrics.cost.total_tokens_input || 0) + (expA.metrics.cost.total_tokens_output || 0)}
                  valueB={(expB.metrics.cost.total_tokens_input || 0) + (expB.metrics.cost.total_tokens_output || 0)}
                  format="number"
                  higherIsBetter={false}
                />
              </div>
            </Panel>

            {statsQuery.data ? (
              <div className="grid gap-4 xl:grid-cols-2">
                <SignificancePanel stats={statsQuery.data} />
                <AgreementSummary summary={statsQuery.data.summary} />
              </div>
            ) : null}

            {statsQuery.data ? (
              <DifferencesTable diffs={statsQuery.data.per_example_differences} nameA={selectedA?.name ?? "A"} nameB={selectedB?.name ?? "B"} />
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="page-stack"><SkeletonBlock className="h-[180px]" /><SkeletonBlock className="h-[400px]" /></div>}>
      <ComparePageInner />
    </Suspense>
  );
}

