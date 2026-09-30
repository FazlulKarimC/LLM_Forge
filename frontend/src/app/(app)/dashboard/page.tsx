"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Database,
  FileText,
  FlaskConical,
  RefreshCcw,
} from "lucide-react";
import { toast } from "sonner";
import {
  createDemoExamples,
  listDatasets,
  listEvaluations,
  isActive,
} from "@/lib/evaluation-api";
import { listPrompts } from "@/lib/prompt-api";
import { getReadinessStatus } from "@/lib/api";
import { PageHeader, StatusPill } from "@/components/ui/primitives";

export default function DashboardPage() {
  const router = useRouter();
  const cache = useQueryClient();
  const prompts = useQuery({
    queryKey: ["prompt-library", "", false, 0],
    queryFn: ({ signal }) => listPrompts("", false, 0, signal),
  });
  const datasets = useQuery({
    queryKey: ["datasets", false, 0],
    queryFn: ({ signal }) => listDatasets(false, 0, signal),
  });
  const runs = useQuery({
    queryKey: ["evaluations", 0],
    queryFn: ({ signal }) => listEvaluations(0, signal),
    refetchInterval: (query) =>
      query.state.data?.items.some(isActive) ? 2500 : false,
  });
  const readiness = useQuery({
    queryKey: ["readiness"],
    queryFn: ({ signal }) => getReadinessStatus({ signal }),
    staleTime: 5 * 60_000,
    retry: 1,
  });
  const demo = useMutation({
    mutationFn: createDemoExamples,
    onSuccess: (examples) => {
      cache.invalidateQueries({ queryKey: ["prompt-library"] });
      cache.invalidateQueries({ queryKey: ["datasets"] });
      const params = new URLSearchParams({
        prompt: examples.prompt_id,
        version: examples.prompt_version_id,
        dataset: examples.dataset_id,
        revision: examples.dataset_revision_id,
        view: "new",
      });
      router.push(`/evaluations?${params}`);
    },
    onError: (error: Error) =>
      toast.error(`Could not prepare demo: ${error.message}`),
  });
  const loading = prompts.isLoading || datasets.isLoading || runs.isLoading;
  const empty =
    !loading &&
    prompts.data?.total === 0 &&
    datasets.data?.total === 0 &&
    runs.data?.total === 0;
  const error = [prompts.error, datasets.error, runs.error].find(
    (item) => item instanceof Error,
  );

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Your project"
        title="Overview"
        description="Develop a prompt, test it on fixed cases, then release a version your application can fetch."
        actions={
          <Link href="/prompts/new" className="btn-primary">
            <FileText className="size-4" />
            Create prompt
          </Link>
        }
      />
      {error instanceof Error && (
        <div role="alert" className="alert alert-danger flex-wrap gap-3">
          <span className="flex-1">
            Workspace data could not load: {error.message}
          </span>
          <button
            className="btn-secondary"
            onClick={() => {
              prompts.refetch();
              datasets.refetch();
              runs.refetch();
            }}
          >
            <RefreshCcw className="size-4" />
            Retry
          </button>
        </div>
      )}
      {(readiness.error || readiness.data?.status === "not_ready") && (
        <div role="alert" className="alert alert-danger flex-wrap gap-3">
          <span className="flex-1">
            The API is not ready for new runs. Check Service status below and
            retry after the database or dispatch recovers.
          </span>
          <button className="btn-secondary" onClick={() => readiness.refetch()}>
            <RefreshCcw className="size-4" />
            Recheck
          </button>
        </div>
      )}
      {empty ? (
        <section className="panel p-4">
          <div className="section-label">First run</div>
          <h2 className="mt-1 text-lg font-semibold">
            See a prompt regression in minutes
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-(--muted-foreground)">
            Create an Echo demo prompt and a two-case Greetings dataset in this
            project. The demo echoes the compiled prompt, so it needs no model
            credits and does not measure model quality.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              className="btn-primary"
              disabled={demo.isPending}
              onClick={() => demo.mutate()}
            >
              {demo.isPending ? "Preparing…" : "Create demo examples"}
              <ArrowRight className="size-4" />
            </button>
            <Link className="btn-secondary" href="/prompts/new">
              Start with my own prompt
            </Link>
          </div>
        </section>
      ) : (
        <section className="panel flex flex-wrap items-center justify-between gap-3 p-4">
          <div>
            <h2 className="font-semibold">Continue testing</h2>
            <p className="mt-1 text-sm text-(--muted-foreground)">
              Use a saved prompt version and dataset revision for a reproducible
              evaluation.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link href="/evaluations?view=new" className="btn-primary">
              New evaluation <ArrowRight className="size-4" />
            </Link>
            <button
              className="btn-secondary"
              disabled={demo.isPending}
              onClick={() => demo.mutate()}
            >
              {demo.isPending ? "Preparing…" : "Create demo examples"}
            </button>
          </div>
        </section>
      )}

      <section
        className="grid gap-3 sm:grid-cols-3"
        aria-label="Project inventory"
      >
        {[
          {
            label: "Prompts",
            value: prompts.data?.total,
            href: "/prompts",
            icon: FileText,
          },
          {
            label: "Datasets",
            value: datasets.data?.total,
            href: "/datasets",
            icon: Database,
          },
          {
            label: "Evaluation runs",
            value: runs.data?.total,
            href: "/evaluations",
            icon: FlaskConical,
          },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.label}
              href={item.href}
              className="panel flex items-center justify-between p-3 transition-colors hover:border-(--border-strong)"
            >
              <div>
                <p className="text-sm text-(--muted-foreground)">
                  {item.label}
                </p>
                <p className="mt-1 text-2xl font-semibold tabular-nums">
                  {item.value ?? (loading ? "…" : "—")}
                </p>
              </div>
              <Icon className="size-5 text-(--primary)" />
            </Link>
          );
        })}
      </section>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <section className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-(--border) px-4 py-3">
            <div>
              <h2 className="font-semibold">Recent evaluations</h2>
              <p className="text-sm text-(--muted-foreground)">
                Choose a run to inspect its cases.
              </p>
            </div>
            <Link
              href="/evaluations"
              className="text-sm text-(--primary) hover:underline"
            >
              View all
            </Link>
          </div>
          {runs.isLoading ? (
            <p className="p-5 text-sm">Loading evaluations…</p>
          ) : !runs.data?.items.length ? (
            <div className="p-5 text-sm text-(--muted-foreground)">
              No runs yet.{" "}
              <Link className="underline" href="/evaluations?view=new">
                Start an evaluation
              </Link>{" "}
              after saving a prompt and dataset.
            </div>
          ) : (
            <div className="divide-y divide-(--border)">
              {runs.data.items.slice(0, 6).map((run) => (
                <Link
                  key={run.id}
                  href={`/evaluations?run=${run.id}`}
                  className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 hover:bg-(--surface-2)"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold">
                      {run.config.prompt_name} v{run.config.prompt_version} ·{" "}
                      {run.config.dataset_name} v{run.config.dataset_version}
                    </p>
                    <p className="mt-1 text-xs text-(--muted-foreground)">
                      {new Date(run.created_at).toLocaleString()} ·{" "}
                      {run.config.source === "sdk_submission"
                        ? "External"
                        : run.config.is_mock
                          ? "Demo"
                          : "Live provider"}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs tabular-nums text-(--muted-foreground)">
                      {run.completed}/{run.total} processed · {run.passed}{" "}
                      passed
                    </span>
                    <StatusPill status={run.status} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>
        <section className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-(--border) px-4 py-3">
            <div>
              <h2 className="font-semibold">Recent prompts</h2>
              <p className="text-sm text-(--muted-foreground)">
                Saved versions and release labels.
              </p>
            </div>
            <Link
              href="/prompts"
              className="text-sm text-(--primary) hover:underline"
            >
              View all
            </Link>
          </div>
          {prompts.isLoading ? (
            <p className="p-5 text-sm">Loading prompts…</p>
          ) : !prompts.data?.items.length ? (
            <p className="p-5 text-sm text-(--muted-foreground)">
              No saved prompts yet.
            </p>
          ) : (
            <div className="divide-y divide-(--border)">
              {prompts.data.items.slice(0, 6).map((prompt) => (
                <Link
                  key={prompt.id}
                  href={`/prompts/${prompt.id}`}
                  className="block px-4 py-3 hover:bg-(--surface-2)"
                >
                  <div className="flex justify-between gap-3">
                    <span className="truncate text-sm font-semibold">
                      {prompt.name}
                    </span>
                    <span className="text-xs">v{prompt.latest_version}</span>
                  </div>
                  <p className="mt-1 text-xs text-(--muted-foreground)">
                    {prompt.labels.length
                      ? prompt.labels
                          .map((label) => `${label.label} v${label.version}`)
                          .join(" · ")
                      : "No release yet"}
                  </p>
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>
      <details className="panel p-5">
        <summary className="cursor-pointer text-sm font-semibold">
          Service status
        </summary>
        <p className="mt-2 text-sm text-(--muted-foreground)">
          Database and dispatch are required for runs. Provider and queue
          integrations are optional.
        </p>
        {readiness.isLoading ? (
          <p className="mt-3 text-sm">Checking…</p>
        ) : readiness.error ? (
          <p role="alert" className="mt-3 text-sm text-(--warning)">
            Readiness is unavailable:{" "}
            {readiness.error instanceof Error
              ? readiness.error.message
              : "Unknown error"}
          </p>
        ) : (
          <ul className="mt-3 grid gap-2 sm:grid-cols-2">
            {Object.entries(readiness.data?.checks ?? {}).map(
              ([key, value]) => (
                <li
                  key={key}
                  className="flex justify-between rounded-lg border border-(--border) px-3 py-2 text-sm"
                >
                  <span className="capitalize">{key.replace(/_/g, " ")}</span>
                  <span className="text-(--muted-foreground)">
                    {String(value).replace(/_/g, " ")}
                  </span>
                </li>
              ),
            )}
          </ul>
        )}
        <button
          className="btn-secondary mt-3"
          onClick={() => readiness.refetch()}
          disabled={readiness.isFetching}
        >
          <RefreshCcw className="size-4" />
          Recheck
        </button>
      </details>
    </div>
  );
}
