"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Clock3,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCcw,
  Trash2,
  WifiOff,
} from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  deleteExperiment,
  getDashboardStats,
  getReadinessStatus,
  listExperimentsSlim,
  resolveRunExperimentCredentials,
  runExperiment,
  type ExperimentListItem,
} from "@/lib/api";
import {
  AnimatedNumber,
  EmptyState,
  MetricCard,
  PageHeader,
  Panel,
  PanelHeader,
  SkeletonBlock,
  StatusPill,
} from "@/components/ui/primitives";

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DashboardPage() {
  const queryClient = useQueryClient();
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [experimentToDelete, setExperimentToDelete] = useState<{ id: string; name: string } | null>(null);

  const statsQuery = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: ({ signal }) => getDashboardStats({ signal }),
  });

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "recent"],
    queryFn: ({ signal }) => listExperimentsSlim({ limit: 6 }, { signal }),
  });

  const readinessQuery = useQuery({
    queryKey: ["readiness"],
    queryFn: ({ signal }) => getReadinessStatus({ signal }),
    staleTime: 5 * 60_000,
    retry: (failureCount, error) => !(error instanceof ApiError && error.statusCode === 408) && failureCount < 1,
  });

  const runMutation = useMutation({
    mutationFn: (experiment: ExperimentListItem) => {
      setRunningIds((prev) => new Set(prev).add(experiment.id));
      const credentials = resolveRunExperimentCredentials(experiment);
      return runExperiment(experiment.id, credentials.customBaseUrl, credentials.customApiKey);
    },
    onSuccess: (_data, experiment) => {
      setRunningIds((prev) => {
        const next = new Set(prev);
        next.delete(experiment.id);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      toast.success("Experiment started");
    },
    onError: (error: Error, experiment) => {
      setRunningIds((prev) => {
        const next = new Set(prev);
        next.delete(experiment.id);
        return next;
      });
      toast.error(`Failed to start experiment: ${error.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => {
      setDeletingIds((prev) => new Set(prev).add(id));
      return deleteExperiment(id);
    },
    onSuccess: (_data, id) => {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      setExperimentToDelete(null);
      toast.success("Experiment deleted");
    },
    onError: (error: Error, id) => {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setExperimentToDelete(null);
      toast.error(`Failed to delete experiment: ${error.message}`);
    },
  });

  const readinessError = readinessQuery.error;
  const readinessIsWaking = readinessError instanceof ApiError && readinessError.statusCode === 408;
  const readinessMessage = readinessIsWaking
    ? "The backend is waking up. Free-tier cold starts can take around a minute."
    : readinessError instanceof Error
      ? readinessError.message
      : "Readiness checks are unavailable right now.";

  const stats = statsQuery.data;
  const experiments = experimentsQuery.data?.experiments ?? [];
  const loading = statsQuery.isLoading || experimentsQuery.isLoading;
  const summaryCards = useMemo(
    () => [
      {
        label: "Total experiments",
        value: stats?.totalExperiments ?? 0,
        detail: "All tracked runs across the workspace",
      },
      {
        label: "Completed",
        value: stats?.completedExperiments ?? 0,
        detail: "Finished and ready for analysis",
        tone: "success" as const,
      },
      {
        label: "Live queue",
        value: stats?.runningExperiments ?? 0,
        detail: "Queued or currently executing",
        tone: "accent" as const,
      },
      {
        label: "Pending",
        value: stats?.pendingExperiments ?? 0,
        detail: "Configured but not yet started",
        tone: "warning" as const,
      },
    ],
    [stats]
  );

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={<><Activity className="size-3.5" /> Workspace snapshot</>}
        title="Workspace overview"
        description="System readiness, recent runs, and quick actions at a glance."
        actions={
          <>
            <Link href="/experiments" className="btn-secondary">
              Browse experiments
            </Link>
            <Link href="/experiments/new" className="btn-primary">
              New experiment
              <ArrowRight className="size-4" />
            </Link>
          </>
        }
      >
        <div className="flex flex-wrap gap-3 text-sm text-(--muted-foreground)">
          <span className="chip">Dashboard</span>
          <span className="chip">Readiness</span>
          <span className="chip">Queue control</span>
          <span className="chip">Recent activity</span>
        </div>
      </PageHeader>

      {statsQuery.error && experimentsQuery.error ? (
        <div className="alert alert-danger">
          <WifiOff className="mt-0.5 size-4 shrink-0" />
          <div className="flex-1 space-y-1">
            <div className="font-semibold">Backend unreachable</div>
            <p className="text-sm text-(--muted-foreground)">
              Could not connect to the API. The backend may be sleeping, booting, or temporarily down.
              {statsQuery.error instanceof ApiError && statsQuery.error.statusCode === 408
                ? " Free-tier cold starts on Hugging Face Spaces can take up to a minute."
                : ""}
            </p>
          </div>
          <button
            type="button"
            className="btn-secondary shrink-0"
            onClick={() => {
              statsQuery.refetch();
              experimentsQuery.refetch();
              readinessQuery.refetch();
            }}
          >
            <RefreshCcw className="size-4" />
            Retry
          </button>
        </div>
      ) : statsQuery.error ? (
        <div className="alert alert-danger">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div className="flex-1 space-y-1">
            <div className="font-semibold">Dashboard data failed to load</div>
            <p className="text-sm text-(--muted-foreground)">
              {statsQuery.error instanceof Error ? statsQuery.error.message : "Unknown error"}
            </p>
          </div>
          <button type="button" className="btn-secondary shrink-0" onClick={() => statsQuery.refetch()}>
            <RefreshCcw className="size-4" />
            Retry
          </button>
        </div>
      ) : experimentsQuery.error ? (
        <div className="alert alert-danger">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div className="flex-1 space-y-1">
            <div className="font-semibold">Recent experiments failed to load</div>
            <p className="text-sm text-(--muted-foreground)">
              {experimentsQuery.error instanceof Error ? experimentsQuery.error.message : "Unknown error"}
            </p>
          </div>
          <button type="button" className="btn-secondary shrink-0" onClick={() => experimentsQuery.refetch()}>
            <RefreshCcw className="size-4" />
            Retry
          </button>
        </div>
      ) : null}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {loading
          ? Array.from({ length: 4 }).map((_, index) => <SkeletonBlock key={index} className="h-[134px]" />)
          : summaryCards.map((card) => (
              <MetricCard
                key={card.label}
                label={card.label}
                tone={card.tone}
                value={<AnimatedNumber value={card.value} className="text-4xl" />}
                detail={card.detail}
              />
            ))}
      </section>

      <section>
        <Panel>
          <PanelHeader
            label="System state"
            title="Readiness checks"
            description="Live status of API, database, and model provider connections."
            actions={
              <button
                type="button"
                className="btn-secondary"
                onClick={() => readinessQuery.refetch()}
                disabled={readinessQuery.isFetching}
              >
                <RefreshCcw className={cn("size-4", readinessQuery.isFetching ? "animate-spin" : "")} />
                Re-check
              </button>
            }
          />
          <div className="panel-body">
            {readinessQuery.isLoading ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <SkeletonBlock className="h-14" />
                <SkeletonBlock className="h-14" />
                <SkeletonBlock className="h-14" />
              </div>
            ) : readinessQuery.error ? (
              <div className={readinessIsWaking ? "alert alert-warning" : "alert alert-danger"}>
                <Clock3 className="mt-0.5 size-4 shrink-0" />
                <p className="text-sm leading-7">{readinessMessage}</p>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(readinessQuery.data?.checks ?? {}).map(([key, value]) => {
                  const status = String(value);
                  const tone =
                    status === "healthy"
                      ? "status-completed"
                      : status === "not_configured"
                        ? "status-pending"
                        : status.startsWith("archived")
                          ? "status-queued"
                          : "status-failed";

                  return (
                    <div key={key} className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="section-label">{key.replace(/_/g, " ")}</div>
                          <div className="mt-1 font-semibold capitalize">{status.replace(/_/g, " ")}</div>
                        </div>
                        <span className={cn("status-pill", tone)}>
                          <span className="status-dot" />
                          {status === "healthy" ? "ready" : status}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </Panel>
      </section>

      <section>
        <Panel>
          <PanelHeader
            label="Recent runs"
            title="Experiment queue"
            description="Your most recent experiment runs."
            actions={<Link href="/experiments" className="btn-secondary">View all</Link>}
          />
          <div className="panel-body">
            {experimentsQuery.isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, index) => (
                  <SkeletonBlock key={index} className="h-[92px]" />
                ))}
              </div>
            ) : experiments.length === 0 ? (
              <EmptyState
                icon={<FlaskConical className="size-5" />}
                title="No experiments yet"
                description="Start with a baseline run, then compare it against a reasoning or retrieval variant."
                action={<Link href="/experiments/new" className="btn-primary">Create first experiment</Link>}
              />
            ) : (
              <div className="space-y-3">
                {experiments.map((experiment) => (
                  <Link
                    key={experiment.id}
                    href={`/experiments/${experiment.id}`}
                    className="block w-full rounded-[20px] border border-(--border) bg-(--surface-2) p-4 text-left transition-all hover:border-(--border-strong) hover:bg-(--surface-3)"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1 space-y-2">
                        <div className="flex items-center gap-2">
                          <div className="truncate text-lg font-semibold tracking-[-0.03em]">{experiment.name}</div>
                          <StatusPill status={experiment.status} />
                        </div>
                        <div className="flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
                          <span className="chip">{experiment.reasoning_method.toUpperCase()}</span>
                          <span className="chip">{experiment.model_name.split("/").pop()}</span>
                          <span className="chip">{experiment.dataset_name}</span>
                        </div>
                        {experiment.description ? (
                          <p className="line-clamp-2 max-w-2xl text-sm leading-7 text-(--muted-foreground)">{experiment.description}</p>
                        ) : null}
                      </div>
                      <div className="flex shrink-0 items-center gap-2" onClick={(event) => event.preventDefault()}>
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={(event) => { event.preventDefault(); runMutation.mutate(experiment); }}
                          disabled={runningIds.has(experiment.id) || experiment.status === "running" || experiment.status === "queued"}
                        >
                          {runningIds.has(experiment.id) ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
                          {experiment.status === "completed" ? "Run again" : "Start"}
                        </button>
                        <button
                          type="button"
                          className="btn-danger"
                          onClick={(event) => { event.preventDefault(); setExperimentToDelete({ id: experiment.id, name: experiment.name }); }}
                          disabled={deletingIds.has(experiment.id)}
                        >
                          {deletingIds.has(experiment.id) ? <LoaderCircle className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
                          Delete
                        </button>
                      </div>
                    </div>
                    <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-(--muted-foreground)">
                      <span className="mono-caption">Created {formatDate(experiment.created_at)}</span>
                      {experiment.completed_at ? <span className="mono-caption">Completed {formatDate(experiment.completed_at)}</span> : null}
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </Panel>
      </section>

      <AnimatePresence>
        {experimentToDelete ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-70 flex items-center justify-center bg-black/55 px-4 backdrop-blur-sm"
            onClick={() => setExperimentToDelete(null)}
          >
            <motion.div
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 12, scale: 0.98 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] as const }}
              className="panel max-w-lg p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-4">
                <div className="flex size-12 items-center justify-center rounded-[18px] border border-[color-mix(in_oklab,var(--destructive)_38%,transparent)] bg-(--destructive-soft) text-[color-mix(in_oklab,var(--destructive)_84%,white_12%)]">
                  <AlertTriangle className="size-5" />
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="section-label">Destructive action</div>
                    <h2 className="mt-1 text-2xl font-semibold tracking-[-0.04em]">Delete experiment</h2>
                  </div>
                  <p className="text-sm leading-7 text-(--muted-foreground)">
                    Remove <span className="font-semibold text-foreground">{experimentToDelete.name}</span> and its saved metrics from the workspace.
                    This cannot be undone.
                  </p>
                </div>
              </div>
              <div className="mt-6 flex flex-wrap justify-end gap-3">
                <button type="button" className="btn-secondary" onClick={() => setExperimentToDelete(null)} disabled={deleteMutation.isPending}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn-danger"
                  onClick={() => deleteMutation.mutate(experimentToDelete.id)}
                  disabled={deleteMutation.isPending}
                >
                  {deleteMutation.isPending ? <LoaderCircle className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
                  Delete experiment
                </button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}
