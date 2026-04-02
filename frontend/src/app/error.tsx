"use client";

import { useEffect } from "react";
import * as Sentry from "@sentry/nextjs";
import { AlertTriangle, RefreshCcw, WifiOff } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string; statusCode?: number; requestId?: string; details?: { field: string; issue: string }[] };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("LLMForge error boundary caught:", error);
    Sentry.captureException(error, {
      extra: { digest: error.digest, requestId: error.requestId },
    });
  }, [error]);

  const isNetworkError = error.message.toLowerCase().includes("failed to fetch");

  return (
    <div className="page-width flex min-h-[70vh] items-center justify-center px-4 py-8 sm:px-6">
      <div className="panel max-w-2xl p-6 sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
          <div className="flex size-14 items-center justify-center rounded-[20px] border border-(--border) bg-(--destructive-soft) text-[color-mix(in_oklab,var(--destructive)_84%,white_12%)]">
            {isNetworkError ? <WifiOff className="size-6" /> : <AlertTriangle className="size-6" />}
          </div>
          <div className="flex-1 space-y-5">
            <div className="space-y-2">
              <div className="section-label">Application state</div>
              <h1 className="text-3xl font-semibold tracking-[-0.05em]">
                {isNetworkError ? "Connection lost" : "A route crashed while rendering"}
              </h1>
              <p className="text-sm leading-7 text-(--muted-foreground)">
                {isNetworkError
                  ? "The frontend could not reach the backend. This usually happens when the API is sleeping, booting, or temporarily unreachable."
                  : error.message || "An unexpected rendering or API error interrupted the current screen."}
              </p>
            </div>

            {error.requestId ? (
              <div className="chip font-mono">Request ID: {error.requestId}</div>
            ) : null}

            {error.details?.length ? (
              <div className="alert alert-danger">
                <div className="space-y-2">
                  <div className="section-label text-[color-mix(in_oklab,var(--destructive)_84%,white_12%)]">Validation details</div>
                  <ul className="space-y-2 text-sm leading-6 text-[color-mix(in_oklab,var(--destructive)_84%,white_12%)]">
                    {error.details.map((detail, index) => (
                      <li key={`${detail.field}-${index}`}>
                        <span className="font-semibold">{detail.field}</span>: {detail.issue}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : null}

            <div className="flex flex-wrap gap-3">
              <button type="button" className="btn-primary" onClick={() => reset()}>
                <RefreshCcw className="size-4" />
                Retry route
              </button>
              <a href="/dashboard" className="btn-secondary">
                Open dashboard
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
