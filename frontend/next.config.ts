import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig: NextConfig = {
  // Capture identity at build time; runtime environment may belong to another release.
  env: {
    LLMFORGE_FRONTEND_COMMIT: process.env.VERCEL_GIT_COMMIT_SHA || process.env.LLMFORGE_FRONTEND_COMMIT || "",
    LLMFORGE_FRONTEND_BUILD_ID: process.env.VERCEL_DEPLOYMENT_ID || process.env.LLMFORGE_FRONTEND_BUILD_ID || "",
  },
  turbopack: {
    root: __dirname,
  },
};

export default withSentryConfig(nextConfig, {
  // Suppress source map upload logs in build output
  silent: true,

  // Skip telemetry collection
  telemetry: false,
});
