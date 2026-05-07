"use client";

import { Fragment, useState } from "react";
import { AlertTriangle, ScanSearch, Sigma } from "lucide-react";

import type { StatisticalComparison } from "@/lib/api";
import { EmptyState, MetricBar, Panel, PanelHeader } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

export function formatMetric(value?: number, format: "percent" | "ms" | "number" = "percent") {
  if (value == null) return "--";
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "ms") return `${value.toFixed(0)} ms`;
  return value.toLocaleString();
}

export function getValidSelection(selectedIds: string[], experimentIds: Set<string>) {
  return selectedIds.filter((id) => experimentIds.has(id)).slice(0, 2);
}

export function MetricCompareCard({
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
        <div className="rounded-[14px] border border-(--border) bg-(--surface-2) p-3">
          <div className="section-label">A</div>
          <div className="metric-value mt-2 text-2xl">{formatMetric(valueA, format)}</div>
        </div>
        <div className="rounded-[14px] border border-(--border) bg-(--surface-2) p-3">
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

export function AgreementSummary({ summary }: { summary: StatisticalComparison["summary"] }) {
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
      <PanelHeader label="Agreement" title="Outcome distribution" description="How often the two experiments agree, disagree, or both fail on the same examples." />
      <div className="panel-body space-y-4">
        <div className="flex h-3 overflow-hidden rounded-full bg-(--muted)">
          {segments.map((segment) => (
            <div key={segment.label} style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }} />
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {segments.map((segment) => (
            <div key={segment.label} className="rounded-[16px] border border-(--border) bg-(--surface-2) p-4">
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

export function SignificancePanel({ stats }: { stats: StatisticalComparison }) {
  const warnings = stats.warnings ?? [];
  const notes = stats.methodology_notes ?? [];

  return (
    <Panel>
      <PanelHeader
        label="Significance"
        title="Statistical analysis"
        description="Paired tests and bootstrap intervals are computed on common examples only; read the caveats before treating a difference as meaningful."
      />
      <div className="panel-body space-y-4">
        {warnings.length ? (
          <div className="alert alert-warning">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <div className="space-y-2">
              <div className="font-semibold">Methodology caveats</div>
              <div className="space-y-1 text-sm leading-6 text-(--muted-foreground)">
                {warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>
            </div>
          </div>
        ) : null}
        <div className={cn("alert", stats.mcnemar.is_significant ? "alert-info" : "alert-warning")}>
          <Sigma className="mt-0.5 size-4 shrink-0" />
          <div className="space-y-1">
            <div className="font-semibold">
              {stats.mcnemar.is_significant ? "Difference is statistically significant" : "Difference is not statistically significant"}
            </div>
            <p className="text-sm text-(--muted-foreground)">
              McNemar p-value {stats.mcnemar.p_value < 0.0001 ? "< 0.0001" : stats.mcnemar.p_value.toFixed(4)}
              {" "}from {stats.mcnemar.n} paired examples ({stats.mcnemar.test_type ?? "test"}).
            </p>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="metric-card">
            <div className="metric-label">Common examples</div>
            <div className="metric-value text-2xl">{stats.num_common_examples}</div>
            <div className="metric-caption">Examples shared by both latest attempts</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Sample overlap</div>
            <div className="metric-value text-2xl">{((stats.overlap_ratio ?? 0) * 100).toFixed(1)}%</div>
            <div className="metric-caption">Lower overlap weakens paired conclusions</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Discordant pairs</div>
            <div className="metric-value text-2xl">{stats.mcnemar.b + stats.mcnemar.c}</div>
            <div className="metric-caption">Only disagreements drive the p-value</div>
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
        {notes.length ? (
          <div className="rounded-[16px] border border-(--border) bg-(--surface-2) p-4 text-sm leading-7 text-(--muted-foreground)">
            {notes.map((note) => (
              <p key={note}>{note}</p>
            ))}
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export function DifferencesTable({
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
                    <td className="text-sm text-(--muted-foreground)">{diff.expected}</td>
                  </tr>
                  {isExpanded ? (
                    <tr>
                      <td colSpan={4} className="p-0!">
                        <div className="grid gap-3 border-b border-(--border) bg-(--surface-2) p-4 lg:grid-cols-3">
                          <div className="rounded-[16px] border border-(--border) bg-(--surface-1) p-4">
                            <div className="section-label">{nameA}</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{diff.a_output || "No output"}</p>
                          </div>
                          <div className="rounded-[16px] border border-(--border) bg-(--surface-1) p-4">
                            <div className="section-label">{nameB}</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{diff.b_output || "No output"}</p>
                          </div>
                          <div className="rounded-[16px] border border-(--border) bg-(--surface-1) p-4">
                            <div className="section-label">Expected</div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{diff.expected}</p>
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
