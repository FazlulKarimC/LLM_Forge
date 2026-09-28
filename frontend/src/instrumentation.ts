import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    await import("./sentry.server");
  } else if (process.env.NEXT_RUNTIME === "edge") {
    await import("./sentry.edge");
  }
}

export const onRequestError = Sentry.captureRequestError;
