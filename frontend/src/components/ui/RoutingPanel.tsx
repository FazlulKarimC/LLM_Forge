"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Radio } from "lucide-react";

import { ApiError, getRoutingTelemetry, type ProviderStats } from "@/lib/api";
import {
  AnimatedNumber,
  MetricCard,
  Panel,
  PanelHeader,
  SkeletonBlock,
} from "@/components/ui/primitives";

/**
 * Self-contained routing telemetry panel.
 *
 * Fetches per-provider stats from raw_metrics and displays:
 * - Aggregate KPIs (total requests, total errors, error rate)
 * - Per-provider data table (requests, errors, latency, tokens, cost)
 */
export function RoutingPanel({ experimentId }: { experimentId: string }) {
  const telemetryQuery = useQuery({
    queryKey: ["routing", experimentId],
    queryFn: ({ signal }) => getRoutingTelemetry(experimentId, { signal }),
    retry: false,
  });

  if (telemetryQuery.isLoading) return <SkeletonBlock className="h-[260px]" />;

  if (telemetryQuery.error) {
    const error = telemetryQuery.error;
    if (error instanceof ApiError && error.statusCode === 404) {
      return null;
    }
    return (
      <Panel>
        <PanelHeader
          label="Routing"
          title="Provider telemetry unavailable"
          description="Per-provider performance metrics from the adaptive router."
        />
        <div className="panel-body">
          <div className="alert alert-danger">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <p className="text-sm leading-7">{error instanceof Error ? error.message : "Failed to load routing telemetry."}</p>
          </div>
        </div>
      </Panel>
    );
  }

  const data = telemetryQuery.data;
  if (!data || Object.keys(data).length === 0) return null;

  const providers = Object.entries(data);
  const totalRequests = providers.reduce((sum, [, stats]) => sum + stats.total_requests, 0);
  const totalErrors = providers.reduce((sum, [, stats]) => sum + stats.total_errors, 0);
  const overallErrorRate = totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0;

  return (
    <Panel>
      <PanelHeader
        label="Routing"
        title="Provider telemetry"
        description="Per-provider performance metrics from the adaptive router."
      />
      <div className="panel-body space-y-5">
        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCard
            label="Total requests"
            value={<AnimatedNumber value={totalRequests} className="text-3xl" />}
            detail={`Across ${providers.length} provider${providers.length !== 1 ? "s" : ""}`}
          />
          <MetricCard
            label="Total errors"
            tone={totalErrors > 0 ? "danger" : "success"}
            value={<AnimatedNumber value={totalErrors} className="text-3xl" />}
          />
          <MetricCard
            label="Error rate"
            tone={overallErrorRate > 5 ? "danger" : "success"}
            value={<AnimatedNumber value={overallErrorRate} suffix="%" decimals={1} className="text-3xl" />}
          />
        </div>

        <div className="overflow-x-auto rounded-[18px] border border-(--border) bg-(--surface-2)">
          <table className="data-table min-w-[720px]">
            <thead>
              <tr>
                <th>Provider</th>
                <th>Requests</th>
                <th>Errors</th>
                <th>Error %</th>
                <th>Mean latency</th>
                <th>p95 latency</th>
                <th>Total tokens</th>
                <th>Cost/req</th>
              </tr>
            </thead>
            <tbody>
              {providers.map(([name, stats]) => (
                <ProviderRow key={name} name={name} stats={stats} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}

function ProviderRow({
  name,
  stats,
}: {
  name: string;
  stats: ProviderStats;
}) {
  const errorRateClass =
    stats.error_rate > 0.05
      ? "text-(--destructive)"
      : stats.error_rate === 0
        ? "text-(--success)"
        : "text-(--muted-foreground)";

  return (
    <tr className="data-row">
      <td>
        <div className="flex items-center gap-2">
          <Radio className="size-3.5 text-(--accent)" />
          <span className="font-mono text-xs">{name}</span>
        </div>
      </td>
      <td className="metric-value">{stats.total_requests}</td>
      <td className="metric-value">{stats.total_errors}</td>
      <td className={`metric-value ${errorRateClass}`}>
        {(stats.error_rate * 100).toFixed(1)}%
      </td>
      <td className="metric-value">
        {stats.mean_latency_ms != null
          ? `${stats.mean_latency_ms.toFixed(0)} ms`
          : "--"}
      </td>
      <td className="metric-value">
        {stats.p95_latency_ms != null
          ? `${stats.p95_latency_ms.toFixed(0)} ms`
          : "--"}
      </td>
      <td className="metric-value">{stats.total_tokens}</td>
      <td className="metric-value">
        ${stats.cost_per_request.toFixed(6)}
      </td>
    </tr>
  );
}
