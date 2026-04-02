import type { NextConfig } from "next";
import path from "path";
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig: NextConfig = {
  // Prevent Next.js from walking up to the root package.json
  // (which would cause tailwindcss/module resolution failures in a non-monorepo setup)
  outputFileTracingRoot: path.join(__dirname),
};

export default withSentryConfig(nextConfig, {
  // Suppress source map upload logs in build output
  silent: true,

  // Skip telemetry collection
  telemetry: false,
});
