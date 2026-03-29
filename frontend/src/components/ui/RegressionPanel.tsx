"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RefreshCw, LoaderCircle, Shield, GitCompareArrows } from "lucide-react";
import { toast } from "sonner";

import {
  getRegressionReport,
  rerunRegression,
  type RegressionReport,
} from "@/lib/api";
import {
  AnimatedNumber,
  EmptyState,
  MetricCard,
  Panel,
  PanelHeader,
  SkeletonBlock,
} from "@/components/ui/primitives";

/**
 * Self-contained regression report panel.
 *
 * Fetches regression data from raw_metrics and displays:
 * - Verdict + overlap + violations as MetricCards
 * - Violation alerts
 * - Sample regressions/improvements tables
 * - Config diff
 * - Grader summary (both sides)
 */
export function RegressionPanel({ experimentId }: { experimentId: string }) {
  const queryClient = useQueryClient();

  const reportQuery = useQuery({
    queryKey: ["regression", experimentId],
    queryFn: ({ signal }) => getRegressionReport(experimentId, { signal }),
    retry: false,
  });

  const rerunMutation = useMutation({
    mutationFn: () => rerunRegression(experimentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["regression", experimentId] });
      queryClient.invalidateQueries({ queryKey: ["experiment", experimentId] });
      toast.success("Regression check completed");
    },
    onError: (error: Error) => {
      toast.error(`Regression check failed: ${error.message}`);
    },
  });

  // 404 = no regression data → show rerun button only
  if (reportQuery.error) {
    return (
      <Panel>
        <PanelHeader
          label="Regression"
          title="Regression gate"
          description="Compare this experiment against a pinned baseline."
          actions={
            <button
              type="button"
              className="btn-secondary"
              onClick={() => rerunMutation.mutate()}
              disabled={rerunMutation.isPending}
            >
              {rerunMutation.isPending ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <RefreshCw className="size-4" />
              )}
              Run regression check
            </button>
          }
        />
        <div className="panel-body">
          <EmptyState
            icon={<Shield className="size-5" />}
            title="No regression data"
            description="Pin a baseline and run a regression check to compare experiments."
          />
        </div>
      </Panel>
    );
  }

  if (reportQuery.isLoading) {
    return <SkeletonBlock className="h-[320px]" />;
  }

  const report = reportQuery.data;
  if (!report) return null;

  const verdictLabel =
    report.passed === true
      ? "PASS"
      : report.passed === false
        ? "FAIL"
        : "INCONCLUSIVE";
  const verdictTone =
    report.passed === true
      ? "success"
      : report.passed === false
        ? "danger"
        : "warning";

  return (
    <Panel>
      <PanelHeader
        label="Regression"
        title="Regression gate report"
        description={`Compared against baseline (attempt ${report.baseline_attempt}) with ${(report.overlap_ratio * 100).toFixed(0)}% sample overlap.`}
        actions={
          <button
            type="button"
            className="btn-ghost"
            onClick={() => rerunMutation.mutate()}
            disabled={rerunMutation.isPending}
          >
            {rerunMutation.isPending ? (
              <LoaderCircle className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            Rerun
          </button>
        }
      />
      <div className="panel-body space-y-5">
        {/* KPI row */}
        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCard
            label="Verdict"
            tone={verdictTone as "success" | "warning" | "danger"}
            value={
              <span className="text-3xl font-semibold tracking-[-0.06em]">
                {verdictLabel}
              </span>
            }
            detail={`Candidate attempt ${report.candidate_attempt}`}
          />
          <MetricCard
            label="Overlap ratio"
            tone="accent"
            value={
              <AnimatedNumber
                value={report.overlap_ratio * 100}
                suffix="%"
                decimals={1}
                className="text-3xl"
              />
            }
            detail="Common samples between baseline and candidate"
          />
          <MetricCard
            label="Violations"
            tone={report.violations.length > 0 ? "danger" : "success"}
            value={
              <AnimatedNumber
                value={report.violations.length}
                className="text-3xl"
              />
            }
            detail={`${report.sample_regressions_count} regressions · ${report.sample_improvements_count} improvements`}
          />
        </div>

        {/* Violations */}
        {report.violations.length > 0 ? (
          <div className="space-y-2">
            <div className="section-label">Violations</div>
            {report.violations.map((v, i) => (
              <div key={`${v.rule}-${i}`} className="alert alert-danger">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <div className="space-y-0.5">
                  <div className="font-mono text-xs font-semibold">
                    {v.rule}
                  </div>
                  <p className="text-sm leading-7">{v.message}</p>
                </div>
              </div>
            ))}
          </div>
        ) : null}

        {/* Sample regressions / improvements */}
        {(report.sample_regressions_count > 0 ||
          report.sample_improvements_count > 0) && report.statistical ? (
          <SampleTables report={report} />
        ) : null}

        {/* Config diff */}
        {report.config_diff &&
        Object.keys(report.config_diff).length > 0 &&
        !("error" in report.config_diff) ? (
          <div>
            <div className="section-label mb-2">Config diff</div>
            <pre className="code-panel">
              {JSON.stringify(report.config_diff, null, 2)}
            </pre>
          </div>
        ) : null}

        {/* Grader summary */}
        {report.grader_summary &&
        (Object.keys(report.grader_summary.baseline ?? {}).length > 0 ||
          Object.keys(report.grader_summary.candidate ?? {}).length > 0) ? (
          <GraderSummary summary={report.grader_summary} />
        ) : null}
      </div>
    </Panel>
  );
}

function SampleTables({ report }: { report: RegressionReport }) {
  const stats = report.statistical as Record<string, unknown> | undefined;
  const perExample = (stats?.per_example_differences ?? []) as Array<{
    example_id: string;
    a_correct: boolean;
    b_correct: boolean;
    a_score: number;
    b_score: number;
  }>;

  const regressions = perExample.filter((d) => d.a_correct && !d.b_correct);
  const improvements = perExample.filter((d) => !d.a_correct && d.b_correct);

  if (regressions.length === 0 && improvements.length === 0) return null;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {regressions.length > 0 ? (
        <div>
          <div className="section-label mb-2 text-(--destructive)">
            Sample regressions ({regressions.length})
          </div>
          <div className="overflow-x-auto rounded-[18px] border border-(--border) bg-(--surface-2)">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Example</th>
                  <th>Baseline</th>
                  <th>Candidate</th>
                </tr>
              </thead>
              <tbody>
                {regressions.slice(0, 20).map((d) => (
                  <tr key={d.example_id} className="data-row">
                    <td className="font-mono text-xs">{d.example_id}</td>
                    <td className="metric-value text-(--success)">
                      {d.a_score.toFixed(3)}
                    </td>
                    <td className="metric-value text-(--destructive)">
                      {d.b_score.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {improvements.length > 0 ? (
        <div>
          <div className="section-label mb-2 text-(--success)">
            Sample improvements ({improvements.length})
          </div>
          <div className="overflow-x-auto rounded-[18px] border border-(--border) bg-(--surface-2)">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Example</th>
                  <th>Baseline</th>
                  <th>Candidate</th>
                </tr>
              </thead>
              <tbody>
                {improvements.slice(0, 20).map((d) => (
                  <tr key={d.example_id} className="data-row">
                    <td className="font-mono text-xs">{d.example_id}</td>
                    <td className="metric-value text-(--destructive)">
                      {d.a_score.toFixed(3)}
                    </td>
                    <td className="metric-value text-(--success)">
                      {d.b_score.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function GraderSummary({
  summary,
}: {
  summary: Record<string, unknown>;
}) {
  const baseline = (summary.baseline ?? {}) as Record<
    string,
    { pass: number; fail: number; skip: number }
  >;
  const candidate = (summary.candidate ?? {}) as Record<
    string,
    { pass: number; fail: number; skip: number }
  >;

  const allGraders = [
    ...new Set([...Object.keys(baseline), ...Object.keys(candidate)]),
  ];
  if (allGraders.length === 0) return null;

  return (
    <div>
      <div className="section-label mb-2">Grader summary</div>
      <div className="overflow-x-auto rounded-[18px] border border-(--border) bg-(--surface-2)">
        <table className="data-table min-w-[580px]">
          <thead>
            <tr>
              <th>Grader</th>
              <th colSpan={3} className="text-center">
                Baseline
              </th>
              <th colSpan={3} className="text-center">
                Candidate
              </th>
            </tr>
            <tr>
              <th />
              <th className="text-(--success)">Pass</th>
              <th className="text-(--destructive)">Fail</th>
              <th className="text-(--muted-foreground)">Skip</th>
              <th className="text-(--success)">Pass</th>
              <th className="text-(--destructive)">Fail</th>
              <th className="text-(--muted-foreground)">Skip</th>
            </tr>
          </thead>
          <tbody>
            {allGraders.map((name) => {
              const b = baseline[name] ?? { pass: 0, fail: 0, skip: 0 };
              const c = candidate[name] ?? { pass: 0, fail: 0, skip: 0 };
              return (
                <tr key={name} className="data-row">
                  <td className="font-mono text-xs">{name}</td>
                  <td className="metric-value">{b.pass}</td>
                  <td className="metric-value">{b.fail}</td>
                  <td className="metric-value">{b.skip}</td>
                  <td className="metric-value">{c.pass}</td>
                  <td className="metric-value">{c.fail}</td>
                  <td className="metric-value">{c.skip}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
