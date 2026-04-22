"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Download, FileText, ScanSearch, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  exportMarkdownReport,
  exportResults,
  getMetrics,
  getProfile,
  getRunGridSummaries,
  getRunSummaries,
  type ProfileData,
  type RunGridSummary,
} from "@/lib/api";
import {
  AnimatedNumber,
  EmptyState,
  MetricBar,
  MetricCard,
  Panel,
  PanelHeader,
  SkeletonBlock,
  StatusPill,
} from "@/components/ui/primitives";

function formatDuration(ms?: number) {
  if (ms == null) return "--";
  return `${ms.toFixed(0)} ms`;
}

function LatencyHistogram({ runs }: { runs: RunGridSummary[] }) {
  const latencies = runs.map((run) => run.latency_ms).filter((value): value is number => value != null);
  if (!latencies.length) return null;

  const min = Math.min(...latencies);
  const max = Math.max(...latencies);
  const buckets = Array.from({ length: 10 }, (_, index) => {
    const start = min + ((max - min) / 10) * index;
    const end = min + ((max - min) / 10) * (index + 1);
    const count = latencies.filter((latency) => latency >= start && (index === 9 ? latency <= end : latency < end)).length;
    return { label: `${Math.round(start)}`, count };
  });
  const peak = Math.max(...buckets.map((bucket) => bucket.count), 1);

  return (
    <Panel>
      <PanelHeader label="Latency" title="Distribution" description="Response time distribution across all evaluated samples." />
      <div className="panel-body">
        <div className="flex h-48 items-end gap-2">
          {buckets.map((bucket) => (
            <div key={bucket.label} className="flex-1">
              <div className="flex h-40 items-end">
                <div
                  className="w-full rounded-t-[12px] bg-(--accent)"
                  style={{ height: `${Math.max(8, (bucket.count / peak) * 100)}%` }}
                  title={`${bucket.count} runs`}
                />
              </div>
              <div className="mt-2 text-center font-mono text-[11px] text-(--muted-foreground)">{bucket.label}</div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function RunFilmstrip({ runs, experimentId }: { runs: RunGridSummary[]; experimentId: string }) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(runs[0]?.id ?? null);

  const fullRunsQuery = useQuery({
    queryKey: ["runs-full", experimentId],
    queryFn: ({ signal }) => getRunSummaries(experimentId, { signal }),
    enabled: selectedRunId !== null,
    staleTime: Infinity,
  });

  const selectedFullRun = fullRunsQuery.data?.find((run) => run.id === selectedRunId) ?? null;
  const graderEntries = Object.entries(selectedFullRun?.grader_results ?? {});

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          label="Per-run overview"
          title="Evaluation filmstrip"
          description="Each cell represents one sample, color-coded by correctness."
        />
        <div className="panel-body space-y-4">
          <div className="grid grid-cols-8 gap-2 sm:grid-cols-12 xl:grid-cols-10">
            {runs.map((run) => {
              const statusClass = run.failure_mode
                ? "status-failed"
                : run.is_correct
                  ? "status-completed"
                  : "status-failed";

              return (
                <button
                  key={run.id}
                  type="button"
                  className={`min-w-0 rounded-[12px] border p-2 text-left transition-all hover:border-(--border-strong) ${selectedRunId === run.id ? "border-(--accent) bg-[color-mix(in_oklab,var(--accent)_14%,transparent)]" : "border-(--border) bg-(--surface-2)"}`}
                  onClick={() => setSelectedRunId(run.id)}
                  title={run.example_id ?? "Run"}
                >
                  <span className={`status-pill ${statusClass} w-full! justify-center truncate overflow-hidden px-0!`}>{run.example_id ?? "run"}</span>
                </button>
              );
            })}
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="metric-card">
              <div className="metric-label">Correct</div>
              <div className="metric-value text-2xl">{runs.filter((run) => run.is_correct).length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Incorrect</div>
              <div className="metric-value text-2xl">{runs.filter((run) => run.is_correct === false && !run.failure_mode).length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Failures</div>
              <div className="metric-value text-2xl">{runs.filter((run) => run.failure_mode).length}</div>
            </div>
          </div>
        </div>
      </Panel>

      <Panel>
        <PanelHeader
          label="Selected example"
          title={selectedFullRun?.example_id || runs.find((run) => run.id === selectedRunId)?.example_id || "Pick a run"}
          description="Inspect the prompt, model output, and expected answer for this sample."
        />
        <div className="panel-body">
          {fullRunsQuery.isLoading ? (
            <div className="space-y-3">
              <SkeletonBlock className="h-[100px]" />
              <SkeletonBlock className="h-[140px]" />
              <SkeletonBlock className="h-[100px]" />
            </div>
          ) : selectedFullRun ? (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <StatusPill status={selectedFullRun.failure_mode ? "failed" : selectedFullRun.is_correct ? "completed" : "failed"} />
                <span className="chip">F1 {(selectedFullRun.score ?? 0).toFixed(3)}</span>
                <span className="chip">Latency {formatDuration(selectedFullRun.latency_ms)}</span>
                {selectedFullRun.served_provider ? <span className="chip">Provider {selectedFullRun.served_provider}</span> : null}
                {selectedFullRun.routing_reason ? <span className="chip">Route {selectedFullRun.routing_reason}</span> : null}
                {selectedFullRun.cost_usd != null ? <span className="chip">Cost ${selectedFullRun.cost_usd.toFixed(6)}</span> : null}
                {selectedFullRun.failure_mode ? <span className="chip">Failure {selectedFullRun.failure_mode}</span> : null}
              </div>
              <div className="grid gap-4">
                <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                  <div className="section-label">Prompt</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{selectedFullRun.prompt || "No prompt recorded"}</pre>
                </div>
                <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                  <div className="section-label">Model output</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{selectedFullRun.raw_output || "No output recorded"}</pre>
                </div>
                <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                  <div className="section-label">Expected answer</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-(--muted-foreground)">{selectedFullRun.expected_output || "No expected answer recorded"}</pre>
                </div>
                {graderEntries.length > 0 ? (
                  <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                    <div className="section-label">Grader verdicts</div>
                    <div className="mt-3 space-y-2">
                      {graderEntries.map(([graderName, verdict]) => {
                        const verdictData = verdict as Record<string, unknown>;
                        const classification = typeof verdictData.classification === "string" ? verdictData.classification : undefined;
                        const explicitStatus = typeof verdictData.status === "string" ? verdictData.status : undefined;
                        const status = explicitStatus ?? (verdictData.is_safe === true ? "pass" : verdictData.is_safe === false ? "fail" : "skip");
                        const reason = typeof verdictData.reason === "string"
                          ? verdictData.reason
                          : typeof verdictData.reasoning === "string"
                            ? verdictData.reasoning
                            : undefined;
                        return (
                          <div key={graderName} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3">
                            <div className="flex items-center justify-between text-sm">
                              <span className="font-mono text-xs">{graderName}</span>
                              <span
                                className={`status-pill ${
                                  verdictData.status === "pass"
                                    ? "status-completed"
                                    : verdictData.status === "fail"
                                      ? "status-failed"
                                      : "status-pending"
                                }`}
                              >
                                {classification ?? status}
                              </span>
                            </div>
                            {reason ? (
                              <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">{reason}</p>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : null}
                {selectedFullRun.retrieved_chunks?.chunks?.length ? (
                  <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                    <div className="section-label">Retrieved context</div>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-(--muted-foreground)">
                      {selectedFullRun.retrieved_chunks.chunks.map((chunk, index) => (
                        <div key={`${selectedFullRun.id}-${index}`} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3">
                          <div className="font-mono text-[11px] text-(--muted-foreground)">Chunk {index + 1}</div>
                          <p className="mt-2 whitespace-pre-wrap">{chunk.text || chunk.page_content || JSON.stringify(chunk)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
                {selectedFullRun.agent_trace?.steps?.length ? (
                  <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                    <div className="section-label">Agent trace</div>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-(--muted-foreground)">
                      {selectedFullRun.agent_trace.steps.map((step, index) => (
                        <div key={`${selectedFullRun.id}-trace-${index}`} className="rounded-[16px] border border-(--border) bg-(--surface-1) p-3">
                          <div className="font-semibold text-foreground">Step {index + 1}</div>
                          <p className="mt-2 whitespace-pre-wrap"><span className="font-semibold">Thought:</span> {step.thought}</p>
                          {step.action ? (
                            <p className="mt-2 whitespace-pre-wrap">
                              <span className="font-semibold">Action:</span> {step.action} {step.action_input ? `- ${step.action_input}` : ""}
                            </p>
                          ) : null}
                          {step.observation ? (
                            <p className="mt-2 whitespace-pre-wrap"><span className="font-semibold">Observation:</span> {step.observation}</p>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          ) : (
            <EmptyState icon={<ScanSearch className="size-5" />} title="No run selected" description="Choose a cell from the filmstrip to inspect its prompt, output, and supporting context." />
          )}
        </div>
      </Panel>
    </div>
  );
}

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
              {metrics.quality.safety_score !== undefined ? (
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm"><span>Safety score</span><span className="metric-value">{(metrics.quality.safety_score * 100).toFixed(1)}%</span></div>
                  <MetricBar value={metrics.quality.safety_score * 100} />
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
