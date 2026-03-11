"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { use, useState } from "react";
import { Download, FileText, LoaderCircle, Play, ScanSearch, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  exportMarkdownReport,
  exportResults,
  getExperiment,
  getMetrics,
  getProfile,
  getRunSummaries,
  runExperiment,
  type ProfileData,
  type RunSummary,
} from "@/lib/api";
import {
  AnimatedNumber,
  EmptyState,
  MetricBar,
  MetricCard,
  PageHeader,
  Panel,
  PanelHeader,
  SkeletonBlock,
  StatusPill,
} from "@/components/ui/primitives";

interface Props {
  params: Promise<{ id: string }>;
}

function formatDate(value?: string) {
  if (!value) return "--";
  return new Date(value).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(ms?: number) {
  if (ms == null) return "--";
  return `${ms.toFixed(0)} ms`;
}

function LatencyHistogram({ runs }: { runs: RunSummary[] }) {
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
      <PanelHeader label="Latency" title="Distribution" description="A lightweight histogram using only frontend code and the existing run summaries." />
      <div className="panel-body">
        <div className="flex h-48 items-end gap-2">
          {buckets.map((bucket) => (
            <div key={bucket.label} className="flex-1">
              <div className="flex h-40 items-end">
                <div
                  className="w-full rounded-t-[12px] bg-[var(--accent)]"
                  style={{ height: `${Math.max(8, (bucket.count / peak) * 100)}%` }}
                  title={`${bucket.count} runs`}
                />
              </div>
              <div className="mt-2 text-center font-mono text-[11px] text-[var(--muted-foreground)]">{bucket.label}</div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function RunFilmstrip({ runs }: { runs: RunSummary[] }) {
  const [selectedRun, setSelectedRun] = useState<RunSummary | null>(runs[0] ?? null);

  return (
    <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
      <Panel>
        <PanelHeader
          label="Per-run overview"
          title="Evaluation filmstrip"
          description="Each cell is one example, colored by correctness so disagreements and failures pop immediately."
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
                  className={`rounded-[12px] border p-2 text-left transition-all hover:border-[var(--border-strong)] ${selectedRun?.id === run.id ? "border-[var(--accent)] bg-[color:color-mix(in_oklab,var(--accent)_14%,transparent)]" : "border-[var(--border)] bg-[var(--surface-2)]"}`}
                  onClick={() => setSelectedRun(run)}
                  title={run.example_id ?? "Run"}
                >
                  <span className={`status-pill ${statusClass} !w-full justify-center !px-0`}>{run.example_id ?? "run"}</span>
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
          title={selectedRun?.example_id || "Pick a run"}
          description="The inspector uses only current run summary fields: prompt, raw output, expected answer, traces, and retrieval context."
        />
        <div className="panel-body">
          {selectedRun ? (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <StatusPill status={selectedRun.failure_mode ? "failed" : selectedRun.is_correct ? "completed" : "failed"} />
                <span className="chip">F1 {(selectedRun.score ?? 0).toFixed(3)}</span>
                <span className="chip">Latency {formatDuration(selectedRun.latency_ms)}</span>
                {selectedRun.failure_mode ? <span className="chip">Failure {selectedRun.failure_mode}</span> : null}
              </div>
              <div className="grid gap-4">
                <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                  <div className="section-label">Prompt</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{selectedRun.prompt || "No prompt recorded"}</pre>
                </div>
                <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                  <div className="section-label">Model output</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{selectedRun.raw_output || "No output recorded"}</pre>
                </div>
                <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                  <div className="section-label">Expected answer</div>
                  <pre className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--muted-foreground)]">{selectedRun.expected_output || "No expected answer recorded"}</pre>
                </div>
                {selectedRun.retrieved_chunks?.chunks?.length ? (
                  <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                    <div className="section-label">Retrieved context</div>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-[var(--muted-foreground)]">
                      {selectedRun.retrieved_chunks.chunks.map((chunk, index) => (
                        <div key={`${selectedRun.id}-${index}`} className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-1)] p-3">
                          <div className="font-mono text-[11px] text-[var(--muted-foreground)]">Chunk {index + 1}</div>
                          <p className="mt-2 whitespace-pre-wrap">{chunk.text || chunk.page_content || JSON.stringify(chunk)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
                {selectedRun.agent_trace?.steps?.length ? (
                  <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                    <div className="section-label">Agent trace</div>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-[var(--muted-foreground)]">
                      {selectedRun.agent_trace.steps.map((step, index) => (
                        <div key={`${selectedRun.id}-trace-${index}`} className="rounded-[16px] border border-[var(--border)] bg-[var(--surface-1)] p-3">
                          <div className="font-semibold text-[var(--foreground)]">Step {index + 1}</div>
                          <p className="mt-2 whitespace-pre-wrap"><span className="font-semibold">Thought:</span> {step.thought}</p>
                          {step.action ? <p className="mt-2 whitespace-pre-wrap"><span className="font-semibold">Action:</span> {step.action} {step.action_input ? `- ${step.action_input}` : ""}</p> : null}
                          {step.observation ? <p className="mt-2 whitespace-pre-wrap"><span className="font-semibold">Observation:</span> {step.observation}</p> : null}
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

function ResultsDashboard({ experimentId, experimentName }: { experimentId: string; experimentName: string }) {
  const metricsQuery = useQuery({
    queryKey: ["metrics", experimentId],
    queryFn: ({ signal }) => getMetrics(experimentId, { signal }),
  });
  const runsQuery = useQuery({
    queryKey: ["runs", experimentId],
    queryFn: ({ signal }) => getRunSummaries(experimentId, { signal }),
  });

  if (metricsQuery.isLoading || runsQuery.isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => <SkeletonBlock key={index} className="h-[146px]" />)}
        </div>
        <SkeletonBlock className="h-[320px]" />
        <SkeletonBlock className="h-[420px]" />
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
          description="Export JSON or markdown directly from the frontend using the current results endpoint."
          actions={
            <>
              <button type="button" className="btn-secondary" onClick={() => {
                void toast.promise(exportResults(experimentId, experimentName), {
                  loading: "Exporting JSON...",
                  success: "JSON export downloaded",
                  error: (error) => error instanceof Error ? `Export failed: ${error.message}` : "Export failed",
                });
              }}>
                <Download className="size-4" />
                Export JSON
              </button>
              <button type="button" className="btn-primary" onClick={() => {
                void toast.promise(exportMarkdownReport(experimentId, experimentName, metrics, runs), {
                  loading: "Building markdown report...",
                  success: "Report downloaded",
                  error: (error) => error instanceof Error ? `Export failed: ${error.message}` : "Export failed",
                });
              }}>
                <FileText className="size-4" />
                Export report
              </button>
            </>
          }
        />
        <div className="panel-body grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
            <div className="section-label">Narrative summary</div>
            <p className="mt-3 text-sm leading-8 text-[var(--muted-foreground)]">{metrics.summary_text || "No generated summary was stored for this experiment. The core metrics below still reflect the latest saved run."}</p>
          </div>
          <div className="space-y-3 rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
            <div>
              <div className="section-label">Quality mix</div>
              <div className="mt-2 text-sm text-[var(--muted-foreground)]">Exact accuracy versus mean F1</div>
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
          <PanelHeader label="Failures" title={`Failure analysis (${metrics.failure_modes.total_failures})`} description="Sampled from the existing failure mode counts returned by the backend." />
          <div className="panel-body grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(metrics.failure_modes.counts).map(([mode, count]) => (
              <div key={mode} className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">{mode.replace(/_/g, " ")}</div>
                <div className="metric-value mt-2 text-3xl">{count}</div>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      {runs.length ? <RunFilmstrip runs={runs} /> : <EmptyState icon={<ScanSearch className="size-5" />} title="No per-run logs" description="Metrics were saved, but no run summaries were returned for the latest attempt." />}
      {runs.length ? <LatencyHistogram runs={runs} /> : null}
    </div>
  );
}

function ProfileDashboard({ experimentId }: { experimentId: string }) {
  const profileQuery = useQuery({
    queryKey: ["profile", experimentId],
    queryFn: ({ signal }) => getProfile(experimentId, { signal }),
  });

  if (profileQuery.isLoading) {
    return <SkeletonBlock className="h-[220px]" />;
  }

  const profile = profileQuery.data as ProfileData | undefined;
  if (!profile || profile.message) return null;

  const sections = Object.entries(profile.profiling_summary || {});
  const hasCache = profile.cache_stats.hits !== undefined || profile.cache_stats.misses !== undefined;
  const hasBatch = profile.batch_stats.batches_processed !== undefined;

  return (
    <Panel>
      <PanelHeader label="Optimization" title="Execution profile" description="These numbers already come from the backend optimization report stored with the run." />
      <div className="panel-body space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard label="Wall time" value={<AnimatedNumber value={(profile.total_wall_time_ms ?? 0) / 1000} decimals={2} suffix=" s" className="text-3xl" />} detail="End-to-end execution time" />
          {hasCache ? <MetricCard label="Cache hit rate" tone="accent" value={<AnimatedNumber value={(profile.cache_stats.hit_rate ?? 0) * 100} suffix="%" className="text-3xl" />} detail={`${profile.cache_stats.hits ?? 0} hits / ${profile.cache_stats.misses ?? 0} misses`} /> : null}
          {hasBatch ? <MetricCard label="Batches" tone="success" value={<AnimatedNumber value={profile.batch_stats.batches_processed ?? 0} className="text-3xl" />} detail={`${profile.batch_stats.total_prompts_batched ?? 0} prompts batched`} /> : null}
        </div>

        {sections.length ? (
          <div className="overflow-x-auto rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)]">
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


export default function ExperimentDetailPage({ params }: Props) {
  const { id } = use(params);
  const queryClient = useQueryClient();

  const experimentQuery = useQuery({
    queryKey: ["experiment", id],
    queryFn: ({ signal }) => getExperiment(id, { signal }),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "queued" ? 3000 : false;
    },
  });

  const runMutation = useMutation({
    mutationFn: () => {
      let customBaseUrl: string | undefined;
      let customApiKey: string | undefined;

      if (experimentQuery.data?.config.model_name && typeof window !== "undefined") {
        try {
          const settings = JSON.parse(localStorage.getItem("customLLMSettings") || "{}");
          const modelSettings = settings[experimentQuery.data.config.model_name];
          if (modelSettings) {
            customBaseUrl = modelSettings.baseUrl;
            customApiKey = modelSettings.apiKey;
          } else if (localStorage.getItem("customModelId") === experimentQuery.data.config.model_name) {
            customBaseUrl = localStorage.getItem("customBaseUrl") || undefined;
            customApiKey = localStorage.getItem("customApiKey") || undefined;
          }
        } catch (error) {
          console.error("Failed to load custom model settings", error);
        }
      }

      return runExperiment(id, customBaseUrl, customApiKey);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experiment", id] });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      toast.success("Experiment started");
    },
    onError: (error: Error) => {
      toast.error(`Failed to start experiment: ${error.message}`);
    },
  });

  if (experimentQuery.isLoading) {
    return <div className="page-stack"><SkeletonBlock className="h-[180px]" /><SkeletonBlock className="h-[520px]" /></div>;
  }

  if (experimentQuery.error || !experimentQuery.data) {
    return (
      <div className="page-stack">
        <div className="alert alert-danger">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <p className="text-sm leading-7">{experimentQuery.error instanceof Error ? experimentQuery.error.message : "Experiment not found."}</p>
        </div>
      </div>
    );
  }

  const experiment = experimentQuery.data;
  const canRun = experiment.status === "pending" || experiment.status === "failed" || experiment.status === "completed";
  const isActive = experiment.status === "running" || experiment.status === "queued";

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={<><ScanSearch className="size-3.5" /> Experiment detail</>}
        title={experiment.name}
        description={experiment.description || "Inspect configuration, metrics, and execution details for this experiment."}
        actions={
          <>
            {experiment.status === "completed" ? <Link href={`/experiments/compare?preselect=${id}`} className="btn-secondary">Compare</Link> : null}
            {canRun ? (
              <button type="button" className="btn-primary" onClick={() => runMutation.mutate()} disabled={runMutation.isPending}>
                {runMutation.isPending ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
                {experiment.status === "completed" ? "Run again" : "Run experiment"}
              </button>
            ) : null}
          </>
        }
      >
        <div className="flex flex-wrap gap-2">
          <StatusPill status={experiment.status} />
          <span className="chip">{experiment.config.reasoning_method.toUpperCase()}</span>
          <span className="chip">{experiment.config.model_name.split("/").pop()}</span>
          <span className="chip">{experiment.config.dataset_name}</span>
          {isActive ? <span className="chip">Auto-refresh every 3s</span> : null}
        </div>
      </PageHeader>

      {experiment.error_message ? (
        <div className="alert alert-danger">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <pre className="whitespace-pre-wrap text-sm leading-7">{experiment.error_message}</pre>
        </div>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-[0.78fr_1.22fr]">
        <Panel>
          <PanelHeader label="Configuration" title="Experiment payload" description="The raw config remains visible so you can verify exactly what was sent to the backend." />
          <div className="panel-body">
            <pre className="code-panel">{JSON.stringify(experiment.config, null, 2)}</pre>
          </div>
        </Panel>

        <Panel>
          <PanelHeader label="Lifecycle" title="Run metadata" description="Execution status and timestamps reflect the current backend experiment model." />
          <div className="panel-body grid gap-4 sm:grid-cols-2">
            <div className="metric-card">
              <div className="metric-label">Created</div>
              <div className="metric-value text-xl">{formatDate(experiment.created_at)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Started</div>
              <div className="metric-value text-xl">{formatDate(experiment.started_at)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Completed</div>
              <div className="metric-value text-xl">{formatDate(experiment.completed_at)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Samples</div>
              <div className="metric-value text-xl">{experiment.config.num_samples ?? "--"}</div>
            </div>
          </div>
        </Panel>
      </section>

      {experiment.status === "completed" ? (
        <>
          <ResultsDashboard experimentId={id} experimentName={experiment.name} />
          <ProfileDashboard experimentId={id} />
        </>
      ) : (
        <EmptyState
          icon={<LoaderCircle className={`size-5 ${isActive ? "animate-spin" : ""}`} />}
          title={isActive ? "Experiment is still running" : "No results yet"}
          description={isActive ? "This screen auto-refreshes while the backend is processing the run." : "Start the experiment to unlock metrics, per-run inspection, and optimization data."}
          action={canRun ? <button type="button" className="btn-primary" onClick={() => runMutation.mutate()} disabled={runMutation.isPending}>Run now</button> : undefined}
        />
      )}
    </div>
  );
}


