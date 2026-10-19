"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ScanSearch } from "lucide-react";

import { getRunDetail, type RunGridSummary } from "@/lib/api";
import { EmptyState, Panel, PanelHeader, SkeletonBlock, StatusPill } from "@/components/ui/primitives";

export function formatDuration(ms?: number) {
  if (ms == null) return "--";
  return `${ms.toFixed(0)} ms`;
}

export function LatencyHistogram({ runs }: { runs: RunGridSummary[] }) {
  const latencies = runs.map((run) => run.latency_ms).filter((value): value is number => value != null);
  if (!latencies.length) return null;

  const min = Math.min(...latencies);
  const max = Math.max(...latencies);
  const buckets = min === max
    ? [{ key: `single-${Math.round(min)}`, label: `${Math.round(min)}`, count: latencies.length }]
    : Array.from({ length: 10 }, (_, index) => {
      const start = min + ((max - min) / 10) * index;
      const end = min + ((max - min) / 10) * (index + 1);
      const count = latencies.filter((latency) => latency >= start && (index === 9 ? latency <= end : latency < end)).length;
      return {
        key: `${index}-${Math.round(start)}-${Math.round(end)}`,
        label: `${Math.round(start)}`,
        count,
      };
    });
  const peak = Math.max(...buckets.map((bucket) => bucket.count), 1);

  return (
    <Panel>
      <PanelHeader label="Latency" title="Distribution" description="Response time distribution for the displayed page." />
      <div className="panel-body">
        <div className="flex h-48 items-end gap-2">
          {buckets.map((bucket) => (
            <div key={bucket.key} className="flex-1">
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

export function RunFilmstrip({ runs, experimentId, attempt }: { runs: RunGridSummary[]; experimentId: string; attempt?: number }) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(runs[0]?.id ?? null);

  const fullRunsQuery = useQuery({
    queryKey: ["run-detail", experimentId, attempt, selectedRunId],
    queryFn: ({ signal }) => getRunDetail(experimentId, selectedRunId!, attempt, { signal }),
    enabled: selectedRunId !== null,
    staleTime: Infinity,
  });

  const selectedFullRun = fullRunsQuery.data ?? null;
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
          <div className="grid gap-3 sm:grid-cols-4">
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
            </div>            <div className="metric-card">
              <div className="metric-label">Unknown score</div>
              <div className="metric-value text-2xl">{runs.filter((run) => run.is_correct == null).length}</div>
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
          ) : fullRunsQuery.error ? (
            <p className="alert alert-danger">Case details unavailable: {fullRunsQuery.error instanceof Error ? fullRunsQuery.error.message : "request failed"}</p>
          ) : selectedFullRun ? (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <StatusPill status={selectedFullRun.failure_mode ? "failed" : selectedFullRun.is_correct ? "completed" : "failed"} />
                <span className="chip">F1 {selectedFullRun.score == null ? "unknown" : selectedFullRun.score.toFixed(3)}</span>
                {selectedFullRun.run_metadata?.retention_truncated ? <span className="chip">Stored text truncated; original hashes are in the export</span> : null}
                <span className="chip">Latency {formatDuration(selectedFullRun.latency_ms)}</span>
                {selectedFullRun.served_provider ? <span className="chip">Provider {selectedFullRun.served_provider}</span> : null}
                {selectedFullRun.routing_reason ? <span className="chip">Route {selectedFullRun.routing_reason}</span> : null}
                {selectedFullRun.cost_usd != null ? <span className="chip">Cost ${selectedFullRun.cost_usd.toFixed(6)}</span> : null}
                {selectedFullRun.failure_mode ? <span className="chip">Failure {selectedFullRun.failure_mode}</span> : null}
              </div>
              {selectedFullRun.run_metadata ? <details><summary>Scorer, model and usage provenance</summary><pre className="code-panel">{JSON.stringify(selectedFullRun.run_metadata, null, 2)}</pre></details> : <p>Case provenance is unknown for this historical record.</p>}
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
