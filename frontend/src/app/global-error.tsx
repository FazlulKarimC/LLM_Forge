"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";
import "./globals.css";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <body>
        <main className="page-width flex min-h-screen items-center justify-center p-6">
          <div className="panel max-w-lg p-6">
            <h1 className="text-2xl font-semibold">The app could not load</h1>
            <p className="mt-3 text-sm text-(--muted-foreground)">
              Try again. If the problem continues, reload the page.
            </p>
            <button type="button" className="btn-primary mt-5" onClick={reset}>
              Try again
            </button>
          </div>
        </main>
      </body>
    </html>
  );
}
