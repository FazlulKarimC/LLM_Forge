"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  FileText,
  RefreshCcw,
} from "lucide-react";
import {
  listEvaluations,
  isActive,
} from "@/lib/evaluation-api";
import { listPrompts } from "@/lib/prompt-api";
import { getReadinessStatus } from "@/lib/api";
import { PageHeader, StatusPill } from "@/components/ui/primitives";

export default function DashboardPage() {
  const prompts = useQuery({
    queryKey: ["prompt-library", "", false, 0],
    queryFn: ({ signal }) => listPrompts("", false, 0, signal),
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
  const loading = prompts.isLoading || runs.isLoading;
  const empty =
    !loading &&
    prompts.data?.total === 0 &&
    runs.data?.total === 0;
  const error = [prompts.error, runs.error].find(
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
      {empty && <section className="panel p-5 space-y-3">
        <h2 className="font-semibold">Start with a saved prompt</h2>
        <p className="text-sm text-(--muted-foreground)">Keep your test cases and model execution local. Save a prompt, evaluate with the Python SDK and upload a summary here.</p>
        <Link className="btn-secondary" href="/evaluations?view=sdk">Evaluate with SDK</Link>
      </section>}
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
              after evaluating a saved prompt with the SDK.
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
                      {run.config.dataset_name}{run.config.source === "sdk_local" ? " · Local" : " · Legacy hosted"}
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
