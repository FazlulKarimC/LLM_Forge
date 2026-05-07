"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { use } from "react";
import {
  LoaderCircle,
  Pin,
  PinOff,
  Play,
  ScanSearch,
  SearchX,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  getExperiment,
  resolveRunExperimentCredentials,
  runExperiment,
  setBaseline,
  unsetBaseline,
} from "@/lib/api";
import {
  ExperimentProfileDashboard,
  ExperimentResultsDashboard,
} from "@/components/experiments/experiment-detail-sections";
import {
  EmptyState,
  PageHeader,
  Panel,
  PanelHeader,
  SkeletonBlock,
  StatusPill,
} from "@/components/ui/primitives";
import { RegressionPanel } from "@/components/ui/RegressionPanel";
import { RoutingPanel } from "@/components/ui/RoutingPanel";

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

function regressionBadge(status?: "not_checked" | "pass" | "fail" | "inconclusive") {
  switch (status) {
    case "pass":
      return <span className="status-pill status-completed">Pass</span>;
    case "fail":
      return <span className="status-pill status-failed">Fail</span>;
    case "inconclusive":
      return <span className="status-pill status-queued">Inconclusive</span>;
    default:
      return null;
  }
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
      const credentials = resolveRunExperimentCredentials(experimentQuery.data?.config);
      return runExperiment(id, credentials.customBaseUrl, credentials.customApiKey);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experiment", id] });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      queryClient.invalidateQueries({ queryKey: ["metrics", id] });
      queryClient.invalidateQueries({ queryKey: ["runs-grid", id] });
      queryClient.invalidateQueries({ queryKey: ["runs-full", id] });
      queryClient.invalidateQueries({ queryKey: ["profile", id] });
      queryClient.invalidateQueries({ queryKey: ["routing", id] });
      queryClient.invalidateQueries({ queryKey: ["regression", id] });
      toast.success("Experiment started");
    },
    onError: (error: Error) => {
      toast.error(`Failed to start experiment: ${error.message}`);
    },
  });

  if (experimentQuery.isLoading) {
    return (
      <div className="page-stack">
        <SkeletonBlock className="h-[180px]" />
        <SkeletonBlock className="h-[520px]" />
      </div>
    );
  }

  if (experimentQuery.error || !experimentQuery.data) {
    const err = experimentQuery.error;
    const isNotFound =
      (err instanceof ApiError && (err.statusCode === 404 || err.statusCode === 422 || err.statusCode === 400)) ||
      (err instanceof Error && /uuid|valid|not found/i.test(err.message));

    return (
      <div className="page-stack">
        <PageHeader
          backHref="/experiments"
          backLabel="Back to experiments"
          eyebrow={<><ScanSearch className="size-3.5" /> Experiment detail</>}
          title="Experiment not found"
        />
        {isNotFound ? (
          <EmptyState
            icon={<SearchX className="size-5" />}
            title="This experiment does not exist"
            description="The experiment ID in the URL is invalid or has been deleted. Return to the catalog to find a valid experiment."
            action={<Link href="/experiments" className="btn-primary">Browse experiments</Link>}
          />
        ) : (
          <div className="alert alert-danger">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <p className="text-sm leading-7">{err instanceof Error ? err.message : "Failed to load experiment."}</p>
          </div>
        )}
      </div>
    );
  }

  const experiment = experimentQuery.data;
  const canRun = experiment.status === "pending" || experiment.status === "failed" || experiment.status === "completed";
  const isActive = experiment.status === "running" || experiment.status === "queued";

  return (
    <div className="page-stack">
      <PageHeader
        backHref="/experiments"
        backLabel="Back to experiments"
        eyebrow={<><ScanSearch className="size-3.5" /> Experiment detail</>}
        title={experiment.name}
        description={experiment.description || "Inspect configuration, metrics, and execution details for this experiment."}
      >
        <div className="flex flex-wrap gap-2">
          <StatusPill status={experiment.status} />
          <span className="chip">{experiment.config.reasoning_method.toUpperCase()}</span>
          <span className="chip">{experiment.config.model_name.split("/").pop()}</span>
          <span className="chip">{experiment.config.dataset_name}</span>
          {experiment.is_baseline ? (
            <span className="chip" style={{ color: "color-mix(in oklab, var(--primary) 84%, white 12%)" }}>
              <Pin className="size-3" /> Baseline
            </span>
          ) : null}
          {regressionBadge(experiment.regression_status)}
          {isActive ? <span className="chip">Auto-refresh every 3s</span> : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {experiment.status === "completed" ? <Link href={`/experiments/compare?preselect=${id}`} className="btn-secondary">Compare</Link> : null}
          {experiment.status === "completed" ? (
            experiment.is_baseline ? (
              <button
                type="button"
                className="btn-danger"
                onClick={() => {
                  void unsetBaseline(id).then(() => {
                    queryClient.invalidateQueries({ queryKey: ["experiment", id] });
                    queryClient.invalidateQueries({ queryKey: ["experiments"] });
                    toast.success("Baseline unpinned");
                  }).catch((error: Error) => toast.error(`Failed to unpin: ${error.message}`));
                }}
              >
                <PinOff className="size-4" />
                Unpin baseline
              </button>
            ) : (
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  void setBaseline(id).then(() => {
                    queryClient.invalidateQueries({ queryKey: ["experiment", id] });
                    queryClient.invalidateQueries({ queryKey: ["experiments"] });
                    toast.success("Pinned as baseline");
                  }).catch((error: Error) => toast.error(`Failed to pin: ${error.message}`));
                }}
              >
                <Pin className="size-4" />
                Pin as baseline
              </button>
            )
          ) : null}
          {canRun ? (
            <button type="button" className="btn-primary" onClick={() => runMutation.mutate()} disabled={runMutation.isPending}>
              {runMutation.isPending ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              {experiment.status === "completed" ? "Run again" : "Run experiment"}
            </button>
          ) : null}
        </div>
      </PageHeader>

      {experiment.error_message ? (
        <div className="alert alert-danger">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <pre className="whitespace-pre-wrap text-sm leading-7">{experiment.error_message}</pre>
        </div>
      ) : null}

      <section>
        <Panel>
          <PanelHeader label="Lifecycle" title="Run metadata" description="Lifecycle timestamps for this experiment run." />
          <div className="panel-body grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
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

      <section>
        <Panel>
          <PanelHeader label="Configuration" title="Experiment payload" description="The experiment configuration as submitted." />
          <div className="panel-body">
            <pre className="code-panel">{JSON.stringify(experiment.config, null, 2)}</pre>
          </div>
        </Panel>
      </section>

      {experiment.run_manifest ? (
        <section>
          <Panel>
            <PanelHeader label="Provenance" title="Execution manifest" description="Submitted config plus effective runtime settings captured for reproducibility." />
            <div className="panel-body">
              <pre className="code-panel">{JSON.stringify(experiment.run_manifest, null, 2)}</pre>
            </div>
          </Panel>
        </section>
      ) : null}

      {experiment.status === "completed" ? (
        <>
          <ExperimentResultsDashboard
            experimentId={id}
            experimentName={experiment.name}
            experimentStatus={experiment.status}
          />
          <ExperimentProfileDashboard experimentId={id} />
          <section>
            <RegressionPanel experimentId={id} />
          </section>
          <section>
            <RoutingPanel experimentId={id} />
          </section>
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
