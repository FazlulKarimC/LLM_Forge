import Link from "next/link";
import { ArrowLeft, Compass, SearchX } from "lucide-react";

export default function NotFound() {
  return (
    <div className="page-width flex min-h-[75vh] items-center justify-center px-4 py-8 sm:px-6">
      <div className="panel max-w-2xl p-6 sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
          <div className="flex size-14 items-center justify-center rounded-[20px] border border-[var(--border)] bg-[color:color-mix(in_oklab,var(--accent)_16%,transparent)] text-[var(--accent)]">
            <SearchX className="size-6" />
          </div>
          <div className="flex-1 space-y-5">
            <div className="space-y-2">
              <div className="section-label">Route status</div>
              <h1 className="text-3xl font-semibold tracking-[-0.05em]">This page is not in the workspace.</h1>
              <p className="text-sm leading-7 text-[var(--muted-foreground)]">
                The link may be stale, the route may have been removed, or the experiment id no longer exists.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Suggestion</div>
                <div className="mt-2 font-semibold">Return to the dashboard</div>
                <p className="mt-2 text-sm leading-6 text-[var(--muted-foreground)]">Check recent runs and restart from a known entry point.</p>
              </div>
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Alternative</div>
                <div className="mt-2 font-semibold">Open experiments</div>
                <p className="mt-2 text-sm leading-6 text-[var(--muted-foreground)]">Browse existing runs or start a fresh comparison workflow.</p>
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link href="/dashboard" className="btn-primary">
                <Compass className="size-4" />
                Dashboard
              </Link>
              <Link href="/experiments" className="btn-secondary">
                <ArrowLeft className="size-4" />
                Experiments
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
