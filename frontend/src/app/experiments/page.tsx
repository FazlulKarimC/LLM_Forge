"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Filter,
  GitCompareArrows,
  LoaderCircle,
  Play,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { deleteExperiment, listExperiments, runExperiment, type ListExperimentsParams } from "@/lib/api";
import { EmptyState, PageHeader, Panel, PanelHeader, SkeletonBlock, StatusPill } from "@/components/ui/primitives";

const methodLabels: Record<string, string> = {
  naive: "Naive prompting",
  cot: "Chain of thought",
  react: "ReAct agent",
};

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ExperimentsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState("");
  const [methodFilter, setMethodFilter] = useState("");
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [experimentToDelete, setExperimentToDelete] = useState<{ id: string; name: string } | null>(null);

  const params: ListExperimentsParams = useMemo(() => {
    const next: ListExperimentsParams = { limit: 50 };
    if (statusFilter) next.status = statusFilter;
    if (methodFilter) next.method = methodFilter;
    return next;
  }, [methodFilter, statusFilter]);

  const experimentsQuery = useQuery({
    queryKey: ["experiments", params],
    queryFn: ({ signal }) => listExperiments(params, { signal }),
  });

  const runMutation = useMutation({
    mutationFn: (id: string) => {
      setRunningIds((prev) => new Set(prev).add(id));
      return runExperiment(id);
    },
    onSuccess: (_data, id) => {
      setRunningIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      toast.success("Experiment queued for execution");
    },
    onError: (error: Error, id) => {
      setRunningIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      toast.error(`Failed to run experiment: ${error.message}`);
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

  const experiments = experimentsQuery.data?.experiments ?? [];
  const total = experimentsQuery.data?.total ?? 0;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={<><Filter className="size-3.5" /> Experiment catalog</>}
        title={`All experiments (${total})`}
        description="Filter, rerun, delete, or compare your experiment runs."
        actions={
          <>
            <Link href="/experiments/compare" className="btn-secondary">
              <GitCompareArrows className="size-4" />
              Compare
            </Link>
            <Link href="/experiments/new" className="btn-primary">
              New experiment
              <ArrowRight className="size-4" />
            </Link>
          </>
        }
      />

      {experimentsQuery.error ? (
        <div className="alert alert-danger">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div className="space-y-1">
            <div className="font-semibold">Failed to load experiments</div>
            <p className="text-sm text-(--muted-foreground)">
              {experimentsQuery.error instanceof Error ? experimentsQuery.error.message : "Unknown error"}
            </p>
          </div>
        </div>
      ) : null}

      <section>
        <Panel>
          <PanelHeader
            label="Filters"
            title="Refine the workspace"
            description="Narrow the list by status or reasoning method."
            actions={
              <button type="button" className="btn-secondary" onClick={() => { setStatusFilter(""); setMethodFilter(""); }}>
                <SlidersHorizontal className="size-4" />
                Clear filters
              </button>
            }
          />
          <div className="panel-body">
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-[160px] flex-1 space-y-2">
                <label className="field-label" htmlFor="status-filter">Status</label>
                <select id="status-filter" className="select-shell" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                  <option value="">All statuses</option>
                  <option value="pending">Pending</option>
                  <option value="queued">Queued</option>
                  <option value="running">Running</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                </select>
              </div>

              <div className="min-w-[160px] flex-1 space-y-2">
                <label className="field-label" htmlFor="method-filter">Reasoning method</label>
                <select id="method-filter" className="select-shell" value={methodFilter} onChange={(event) => setMethodFilter(event.target.value)}>
                  <option value="">All methods</option>
                  <option value="naive">Naive</option>
                  <option value="cot">Chain of thought</option>
                  <option value="react">ReAct agent</option>
                </select>
              </div>

              <div className="min-w-[200px] flex-1 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">Current slice</div>
                <div className="mt-1 font-semibold tracking-[-0.03em]">
                  {statusFilter || methodFilter ? "Filtered workspace" : "Full workspace"}
                </div>
                <p className="mt-1 text-sm leading-6 text-(--muted-foreground)">
                  {statusFilter || methodFilter
                    ? `${total} experiments matching filters.`
                    : `Showing all ${total} experiments.`}
                </p>
              </div>
            </div>
          </div>
        </Panel>
      </section>

      <section>
        <Panel>
          <PanelHeader
            label="Catalog"
            title="Runs and configurations"
            description="Experiment runs with configuration, status, and quick actions."
          />
          <div className="panel-body">
            {experimentsQuery.isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, index) => <SkeletonBlock key={index} className="h-[124px]" />)}
              </div>
            ) : experiments.length === 0 ? (
              <EmptyState
                icon={<Filter className="size-5" />}
                title="No experiments match this slice"
                description="Clear the filters or create a new experiment to seed the workspace."
                action={<Link href="/experiments/new" className="btn-primary">Create experiment</Link>}
              />
            ) : (
              <div className="space-y-3">
                {experiments.map((experiment) => (
                  <div
                    key={experiment.id}
                    role="button"
                    tabIndex={0}
                    className="w-full cursor-pointer rounded-[20px] border border-(--border) bg-(--surface-2) p-4 text-left transition-all hover:border-(--border-strong) hover:bg-(--surface-3)"
                    onClick={() => router.push(`/experiments/${experiment.id}`)}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); router.push(`/experiments/${experiment.id}`); } }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1 space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="truncate text-lg font-semibold tracking-[-0.03em]">{experiment.name}</div>
                          <StatusPill status={experiment.status} />
                        </div>
                        {experiment.description ? (
                          <p className="line-clamp-2 max-w-3xl text-sm leading-7 text-(--muted-foreground)">{experiment.description}</p>
                        ) : null}
                        <div className="flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
                          <span className="chip">{methodLabels[experiment.config.reasoning_method] ?? experiment.config.reasoning_method}</span>
                          <span className="chip">{experiment.config.model_name.split("/").pop()}</span>
                          <span className="chip">{experiment.config.dataset_name}</span>
                          <span className="chip">Samples {experiment.config.num_samples ?? "N/A"}</span>
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-2" onClick={(event) => event.stopPropagation()}>
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={() => runMutation.mutate(experiment.id)}
                          disabled={runningIds.has(experiment.id) || experiment.status === "running" || experiment.status === "queued"}
                        >
                          {runningIds.has(experiment.id) ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
                          {experiment.status === "completed" ? "Run again" : "Run"}
                        </button>
                        <button
                          type="button"
                          className="btn-danger"
                          onClick={() => setExperimentToDelete({ id: experiment.id, name: experiment.name })}
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
                  </div>
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
                    Remove <span className="font-semibold text-foreground">{experimentToDelete.name}</span> from the experiment catalog and its saved metrics from the results store.
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

