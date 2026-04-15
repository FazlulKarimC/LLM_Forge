import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig: NextConfig = {};

export default withSentryConfig(nextConfig, {
  // Suppress source map upload logs in build output
  silent: true,

  // Skip telemetry collection
  telemetry: false,
});
