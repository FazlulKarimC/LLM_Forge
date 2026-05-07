"use client";

import { useQuery } from "@tanstack/react-query";
import { Download, FileText, ScanSearch, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  exportMarkdownReport,
  exportResults,
  getMetrics,
  getProfile,
  getRunGridSummaries,
  type ProfileData,
} from "@/lib/api";
import { formatDuration, LatencyHistogram, RunFilmstrip } from "@/components/experiments/experiment-run-inspection";
import {
  AnimatedNumber,
  EmptyState,
  MetricBar,
  MetricCard,
  Panel,
  PanelHeader,
  SkeletonBlock,
} from "@/components/ui/primitives";

export function ExperimentResultsDashboard({
  experimentId,
  experimentName,
  experimentStatus,
}: {
  experimentId: string;
  experimentName: string;
  experimentStatus: string;
}) {
  const isCompleted = experimentStatus === "completed";
  const immutableStaleTime = isCompleted ? Infinity : 0;

  const metricsQuery = useQuery({
    queryKey: ["metrics", experimentId],
    queryFn: ({ signal }) => getMetrics(experimentId, { signal }),
    staleTime: immutableStaleTime,
  });
  const runsQuery = useQuery({
    queryKey: ["runs-grid", experimentId],
    queryFn: ({ signal }) => getRunGridSummaries(experimentId, { signal }),
    staleTime: immutableStaleTime,
  });

  if (metricsQuery.isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => <SkeletonBlock key={index} className="h-[146px]" />)}
        </div>
        <SkeletonBlock className="h-[320px]" />
      </div>
    );
  }

  if (metricsQuery.error) {
    const error = metricsQuery.error;
    const isMissingResults = error instanceof ApiError && error.statusCode === 404;
    return isMissingResults ? (
      <EmptyState icon={<ScanSearch className="size-5" />} title="Metrics are not ready yet" description="The latest run has not written aggregate metrics yet. Re-open this page once execution completes." />
    ) : (
      <div className="alert alert-danger">
        <TriangleAlert className="mt-0.5 size-4 shrink-0" />
        <p className="text-sm leading-7">{error instanceof Error ? error.message : "Failed to load metrics."}</p>
      </div>
    );
  }

  const metrics = metricsQuery.data;
  const runs = runsQuery.data ?? [];
  const runsError = runsQuery.error;
  const runsMissing = runsError instanceof ApiError && runsError.statusCode === 404;

  if (!metrics) {
    return (
      <EmptyState
        icon={<ScanSearch className="size-5" />}
        title="Metrics are unavailable"
        description="This run has not stored aggregate metrics yet, so only the experiment metadata is currently available."
      />
    );
  }

  const correctRuns = runs.filter((run) => run.is_correct).length;
  const totalRuns = runs.length;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Exact accuracy"
          tone={(metrics.quality.accuracy_exact ?? 0) >= 0.7 ? "success" : (metrics.quality.accuracy_exact ?? 0) >= 0.4 ? "warning" : "danger"}
          value={<AnimatedNumber value={(metrics.quality.accuracy_exact ?? 0) * 100} suffix="%" className="text-4xl" />}
          detail={`${correctRuns}/${totalRuns} runs marked correct`}
        />
        <MetricCard
          label="Mean F1"
          tone="accent"
          value={<AnimatedNumber value={(metrics.quality.accuracy_f1 ?? 0) * 100} suffix="%" className="text-4xl" />}
          detail="Token-level overlap against expected answers"
        />
        <MetricCard
          label="Latency p50"
          value={<AnimatedNumber value={metrics.performance.latency_p50 ?? 0} suffix=" ms" className="text-4xl" />}
          detail={`p95 ${formatDuration(metrics.performance.latency_p95)}`}
        />
        <MetricCard
          label="Total tokens"
          value={<AnimatedNumber value={(metrics.cost.total_tokens_input || 0) + (metrics.cost.total_tokens_output || 0)} className="text-4xl" />}
          detail={metrics.cost.total_cost_usd != null ? `$${metrics.cost.total_cost_usd.toFixed(4)} total estimated cost` : "Cost unavailable on the current provider"}
        />
      </div>

      <Panel>
        <PanelHeader
          label="Summary"
          title="Result snapshot"
          description="Download results as JSON or a formatted Markdown report."
          actions={
            <>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  void toast.promise(exportResults(experimentId, experimentName), {
                    loading: "Exporting JSON...",
                    success: "JSON export downloaded",
                    error: (error) => error instanceof Error ? `Export failed: ${error.message}` : "Export failed",
                  });
                }}
              >
                <Download className="size-4" />
                Export JSON
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => {
                  void toast.promise(exportMarkdownReport(experimentId, experimentName, metrics, runs), {
                    loading: "Building markdown report...",
                    success: "Report downloaded",
                    error: (error) => error instanceof Error ? `Export failed: ${error.message}` : "Export failed",
                  });
                }}
              >
                <FileText className="size-4" />
                Export report
              </button>
            </>
          }
        />
        <div className="panel-body grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
            <div className="section-label">Narrative summary</div>
            <p className="mt-3 text-sm leading-8 text-(--muted-foreground)">{metrics.summary_text || "No generated summary was stored for this experiment. The core metrics below still reflect the latest saved run."}</p>
          </div>
          <div className="space-y-3 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
            <div>
              <div className="section-label">Quality mix</div>
              <div className="mt-2 text-sm text-(--muted-foreground)">Exact accuracy versus mean F1</div>
            </div>
            <div className="space-y-3">
              <div>
                <div className="mb-2 flex items-center justify-between text-sm"><span>Exact match</span><span className="metric-value">{((metrics.quality.accuracy_exact ?? 0) * 100).toFixed(1)}%</span></div>
                <MetricBar value={(metrics.quality.accuracy_exact ?? 0) * 100} />
              </div>
              <div>
                <div className="mb-2 flex items-center justify-between text-sm"><span>Mean F1</span><span className="metric-value">{((metrics.quality.accuracy_f1 ?? 0) * 100).toFixed(1)}%</span></div>
                <MetricBar value={(metrics.quality.accuracy_f1 ?? 0) * 100} />
              </div>
              {(metrics.quality.robustness_safety_score ?? metrics.quality.safety_score) !== undefined ? (
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm"><span>Safety score</span><span className="metric-value">{(((metrics.quality.robustness_safety_score ?? metrics.quality.safety_score ?? 0)) * 100).toFixed(1)}%</span></div>
                  <MetricBar value={((metrics.quality.robustness_safety_score ?? metrics.quality.safety_score ?? 0)) * 100} />
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </Panel>

      {metrics.failure_modes?.total_failures ? (
        <Panel>
          <PanelHeader label="Failures" title={`Failure analysis (${metrics.failure_modes.total_failures})`} description="Breakdown of error types from the latest run." />
          <div className="panel-body grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(metrics.failure_modes.counts).map(([mode, count]) => (
              <div key={mode} className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">{mode.replace(/_/g, " ")}</div>
                <div className="metric-value mt-2 text-3xl">{count}</div>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      {/* Completion Quality — accuracy excluding failures + parse tiers */}
      {metrics.quality.completion_quality || metrics.quality.accuracy_excluding_failures != null ? (
        <Panel>
          <PanelHeader label="Diagnostics" title="Completion quality" description="Model accuracy excluding infrastructure failures and parse degradation tiers." />
          <div className="panel-body grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {metrics.quality.accuracy_excluding_failures != null ? (
              <MetricCard
                label="Accuracy (excl. failures)"
                tone="accent"
                value={<AnimatedNumber value={(metrics.quality.accuracy_excluding_failures ?? 0) * 100} suffix="%" className="text-3xl" />}
                detail={`${metrics.quality.total_excluding_failures ?? "?"} non-failed runs scored`}
              />
            ) : null}
            {metrics.quality.completion_quality ? (
              <MetricCard
                label="Completion tier"
                tone={metrics.quality.completion_quality.label === "full" ? "success" : metrics.quality.completion_quality.label === "partial" ? "warning" : "danger"}
                value={<span className="text-3xl metric-value capitalize">{metrics.quality.completion_quality.label ?? "unknown"}</span>}
                detail={`${metrics.quality.completion_quality.total_failures ?? 0} failures in ${metrics.quality.completion_quality.total_runs ?? 0} runs (${((metrics.quality.completion_quality.failure_rate ?? 0) * 100).toFixed(1)}%)`}
              />
            ) : null}
          </div>
        </Panel>
      ) : null}

      {/* Retrieval Quality — RAG recall@k + evidence hit rate */}
      {metrics.quality.retrieval_quality ? (
        <Panel>
          <PanelHeader label="RAG Diagnostics" title="Retrieval quality" description="How well the retriever found relevant evidence chunks." />
          <div className="panel-body grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {metrics.quality.retrieval_quality.recall_at_k != null ? (
              <MetricCard
                label={`Recall@${metrics.quality.retrieval_quality.k ?? "k"}`}
                tone="accent"
                value={<AnimatedNumber value={(metrics.quality.retrieval_quality.recall_at_k) * 100} suffix="%" className="text-3xl" />}
                detail="Fraction of gold evidence found in top-k chunks"
              />
            ) : null}
            {metrics.quality.retrieval_quality.evidence_hit_rate != null ? (
              <MetricCard
                label="Evidence hit rate"
                tone={metrics.quality.retrieval_quality.evidence_hit_rate >= 0.7 ? "success" : "warning"}
                value={<AnimatedNumber value={(metrics.quality.retrieval_quality.evidence_hit_rate) * 100} suffix="%" className="text-3xl" />}
                detail="Questions where at least one gold keyword matched"
              />
            ) : null}
            {metrics.quality.retrieval_quality.total_evaluated != null ? (
              <MetricCard
                label="Evaluated"
                value={<span className="text-3xl metric-value">{metrics.quality.retrieval_quality.total_evaluated}</span>}
                detail="Questions with gold evidence annotations"
              />
            ) : null}
          </div>
        </Panel>
      ) : null}

      {/* Cost Efficiency — cost_per_sample + accuracy_per_dollar */}
      {metrics.cost.cost_per_sample_usd != null || metrics.cost.accuracy_per_dollar != null ? (
        <Panel>
          <PanelHeader label="Efficiency" title="Cost efficiency" description="Normalized cost metrics for cross-model comparison." />
          <div className="panel-body grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {metrics.cost.cost_per_sample_usd != null ? (
              <MetricCard
                label="Cost per sample"
                value={<span className="text-3xl metric-value">${(metrics.cost.cost_per_sample_usd ?? 0).toFixed(6)}</span>}
                detail="Average cost per example run"
              />
            ) : null}
            {metrics.cost.accuracy_per_dollar != null ? (
              <MetricCard
                label="Accuracy per dollar"
                tone="accent"
                value={<AnimatedNumber value={metrics.cost.accuracy_per_dollar ?? 0} decimals={2} className="text-3xl" />}
                detail="Accuracy points per $1 spent"
              />
            ) : null}
            {metrics.cost.cost_per_correct_answer != null ? (
              <MetricCard
                label="Cost per correct answer"
                value={<span className="text-3xl metric-value">${(metrics.cost.cost_per_correct_answer ?? 0).toFixed(6)}</span>}
                detail="Average cost for each correct result"
              />
            ) : null}
          </div>
        </Panel>
      ) : null}

      {runsError && !runsMissing ? (
        <div className="alert alert-danger">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <p className="text-sm leading-7">{runsError instanceof Error ? runsError.message : "Failed to load per-run logs."}</p>
        </div>
      ) : null}

      {runsQuery.isLoading ? (
        <SkeletonBlock className="h-[420px]" />
      ) : runs.length ? (
        <>
          <RunFilmstrip runs={runs} experimentId={experimentId} />
          <LatencyHistogram runs={runs} />
        </>
      ) : (
        <EmptyState
          icon={<ScanSearch className="size-5" />}
          title={runsMissing ? "No per-run logs" : "Per-run logs unavailable"}
          description={runsMissing ? "Metrics were saved, but no run summaries were returned for the latest attempt." : "The aggregate metrics loaded, but the latest run logs could not be fetched."}
        />
      )}
    </div>
  );
}

export function ExperimentProfileDashboard({ experimentId }: { experimentId: string }) {
  const profileQuery = useQuery({
    queryKey: ["profile", experimentId],
    queryFn: ({ signal }) => getProfile(experimentId, { signal }),
  });

  if (profileQuery.isLoading) {
    return <SkeletonBlock className="h-[220px]" />;
  }

  if (profileQuery.error) {
    const error = profileQuery.error;
    if (error instanceof ApiError && error.statusCode === 404) {
      return null;
    }
    return (
      <div className="alert alert-danger">
        <TriangleAlert className="mt-0.5 size-4 shrink-0" />
        <p className="text-sm leading-7">{error instanceof Error ? error.message : "Failed to load optimization data."}</p>
      </div>
    );
  }

  const profile = profileQuery.data as ProfileData | undefined;
  if (!profile || profile.message) return null;

  const sections = Object.entries(profile.profiling_summary || {});
  const hasCache = profile.cache_stats.hits !== undefined || profile.cache_stats.misses !== undefined;
  const hasBatch = profile.batch_stats.batches_processed !== undefined;

  return (
    <Panel>
      <PanelHeader label="Optimization" title="Execution profile" description="Profiling data from batching, caching, and per-phase execution timing." />
      <div className="panel-body space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard label="Wall time" value={<AnimatedNumber value={(profile.total_wall_time_ms ?? 0) / 1000} decimals={2} suffix=" s" className="text-3xl" />} detail="End-to-end execution time" />
          {hasCache ? <MetricCard label="Cache hit rate" tone="accent" value={<AnimatedNumber value={(profile.cache_stats.hit_rate ?? 0) * 100} suffix="%" className="text-3xl" />} detail={`${profile.cache_stats.hits ?? 0} hits / ${profile.cache_stats.misses ?? 0} misses`} /> : null}
          {hasBatch ? <MetricCard label="Batches" tone="success" value={<AnimatedNumber value={profile.batch_stats.batches_processed ?? 0} className="text-3xl" />} detail={`${profile.batch_stats.total_prompts_batched ?? 0} prompts batched`} /> : null}
        </div>

        {sections.length ? (
          <div className="overflow-x-auto rounded-[18px] border border-(--border) bg-(--surface-2)">
            <table className="data-table min-w-[720px]">
              <thead>
                <tr>
                  <th>Phase</th>
                  <th>Count</th>
                  <th>Total</th>
                  <th>Mean</th>
                  <th>p50</th>
                  <th>p95</th>
                </tr>
              </thead>
              <tbody>
                {sections.map(([name, stats]) => (
                  <tr key={name} className="data-row">
                    <td className="font-mono text-xs">{name.replace(/_/g, " ")}</td>
                    <td className="metric-value">{stats.count}</td>
                    <td className="metric-value">{stats.total_ms.toFixed(0)} ms</td>
                    <td className="metric-value">{stats.mean_ms.toFixed(1)} ms</td>
                    <td className="metric-value">{stats.p50_ms.toFixed(1)} ms</td>
                    <td className="metric-value">{stats.p95_ms.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}
