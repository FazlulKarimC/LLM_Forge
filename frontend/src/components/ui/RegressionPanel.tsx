"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RefreshCw, LoaderCircle, Shield } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
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

  if (reportQuery.isLoading) {
    return <SkeletonBlock className="h-[320px]" />;
  }

  if (reportQuery.error) {
    const error = reportQuery.error;
    const isMissingReport = error instanceof ApiError && error.statusCode === 404;

    return (
      <Panel>
        <PanelHeader
          label="Regression"
          title={isMissingReport ? "Regression gate" : "Regression gate unavailable"}
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
          {isMissingReport ? (
            <EmptyState
              icon={<Shield className="size-5" />}
              title="No regression data"
              description="Pin a baseline and run a regression check to compare experiments."
            />
          ) : (
            <div className="alert alert-danger">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              <p className="text-sm leading-7">{error instanceof Error ? error.message : "Failed to load regression data."}</p>
            </div>
          )}
        </div>
      </Panel>
    );
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

        {report.violations.length > 0 ? (
          <div className="space-y-2">
            <div className="section-label">Violations</div>
            {report.violations.map((violation, index) => (
              <div key={`${violation.rule}-${index}`} className="alert alert-danger">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <div className="space-y-0.5">
                  <div className="font-mono text-xs font-semibold">
                    {violation.rule}
                  </div>
                  <p className="text-sm leading-7">{violation.message}</p>
                </div>
              </div>
            ))}
          </div>
        ) : null}

        {(report.sample_regressions_count > 0 ||
          report.sample_improvements_count > 0) && report.statistical ? (
          <SampleTables report={report} />
        ) : null}

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
  const perExample = (report.statistical?.per_example_differences ?? []) as Array<{
    example_id: string;
    a_correct: boolean;
    b_correct: boolean;
    a_score: number;
    b_score: number;
  }>;

  const regressions = perExample.filter((diff) => diff.a_correct && !diff.b_correct);
  const improvements = perExample.filter((diff) => !diff.a_correct && diff.b_correct);

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {regressions.length > 0 ? (
        <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
          <div className="section-label">Sample regressions</div>
          <div className="mt-3 space-y-2">
            {regressions.slice(0, 10).map((diff) => (
              <div key={diff.example_id} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3 text-sm">
                <div className="font-mono text-xs">{diff.example_id}</div>
                <div className="mt-2 text-(--muted-foreground)">
                  Baseline {diff.a_score.toFixed(2)} {"->"} Candidate {diff.b_score.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {improvements.length > 0 ? (
        <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
          <div className="section-label">Sample improvements</div>
          <div className="mt-3 space-y-2">
            {improvements.slice(0, 10).map((diff) => (
              <div key={diff.example_id} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3 text-sm">
                <div className="font-mono text-xs">{diff.example_id}</div>
                <div className="mt-2 text-(--muted-foreground)">
                  Baseline {diff.a_score.toFixed(2)} {"->"} Candidate {diff.b_score.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function GraderSummary({ summary }: { summary: Record<string, Record<string, { pass: number; fail: number; skip: number }>> }) {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {(["baseline", "candidate"] as const).map((side) => {
        const graders = summary[side] ?? {};
        if (Object.keys(graders).length === 0) return null;

        return (
          <div key={side} className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
            <div className="section-label">{side}</div>
            <div className="mt-3 space-y-2">
              {Object.entries(graders).map(([graderName, counts]) => (
                <div key={`${side}-${graderName}`} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3">
                  <div className="font-mono text-xs">{graderName}</div>
                  <div className="mt-2 flex flex-wrap gap-2 text-sm text-(--muted-foreground)">
                    <span className="chip">Pass {counts.pass}</span>
                    <span className="chip">Fail {counts.fail}</span>
                    <span className="chip">Skip {counts.skip}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
