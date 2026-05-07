"use client";

import { useQuery } from "@tanstack/react-query";
import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  FlaskConical,
  GitCompareArrows,
  ScanSearch,
} from "lucide-react";

import {
  compareExperiments,
  getStatisticalComparison,
  listExperimentsSlim,
  type ExperimentListItem,
} from "@/lib/api";
import {
  AgreementSummary,
  DifferencesTable,
  getValidSelection,
  MetricCompareCard,
  SignificancePanel,
} from "@/components/experiments/compare-sections";
import { EmptyState, PageHeader, Panel, PanelHeader, SkeletonBlock } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

function ComparePageInner() {
  const searchParams = useSearchParams();
  const preselect = searchParams.get("preselect");
  const [selectedIds, setSelectedIds] = useState<string[]>(() => (preselect ? [preselect] : []));

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "completed"],
    queryFn: ({ signal }) => listExperimentsSlim({ status: "completed", limit: 50 }, { signal }),
  });

  const experiments = useMemo(() => experimentsQuery.data?.experiments ?? [], [experimentsQuery.data?.experiments]);
  const experimentIds = useMemo(() => new Set(experiments.map((experiment) => experiment.id)), [experiments]);
  const effectiveSelectedIds = useMemo(() => getValidSelection(selectedIds, experimentIds), [selectedIds, experimentIds]);

  const comparisonQuery = useQuery({
    queryKey: ["comparison", effectiveSelectedIds],
    queryFn: ({ signal }) => compareExperiments(effectiveSelectedIds, { signal }),
    enabled: effectiveSelectedIds.length === 2,
  });

  const statsQuery = useQuery({
    queryKey: ["statistical", effectiveSelectedIds],
    queryFn: ({ signal }) => getStatisticalComparison(effectiveSelectedIds[0], effectiveSelectedIds[1], { signal }),
    enabled: effectiveSelectedIds.length === 2,
  });

  const selectedA = experiments.find((experiment) => experiment.id === effectiveSelectedIds[0]);
  const selectedB = experiments.find((experiment) => experiment.id === effectiveSelectedIds[1]);
  const comparison = comparisonQuery.data;
  const expA = comparison?.experiments[0];
  const expB = comparison?.experiments[1];

  function setSlotA(id: string) {
    setSelectedIds((current) => {
      const base = getValidSelection(current, experimentIds);
      if (!id) return base.length > 1 ? ["", base[1]] : [];
      return base.length > 1 ? [id, base[1]] : [id];
    });
  }

  function setSlotB(id: string) {
    setSelectedIds((current) => {
      const base = getValidSelection(current, experimentIds);
      if (!id) return base.length > 0 ? [base[0]] : [];
      return [base[0] || "", id];
    });
  }

  return (
    <div className="page-stack">
      <PageHeader
        backHref="/experiments"
        backLabel="Back to experiments"
        eyebrow={<><GitCompareArrows className="size-3.5" /> Comparison workspace</>}
        title="Compare experiments side by side"
        description="Select two completed runs to compare accuracy, latency, cost, and statistical significance."
      />

      <section>
        <Panel>
          <PanelHeader
            label="Selection"
            title="Choose two completed runs"
            description="Pick one experiment for each slot to begin the comparison."
            actions={effectiveSelectedIds.length ? <button type="button" className="btn-secondary" onClick={() => setSelectedIds([])}>Clear</button> : null}
          />
          <div className="panel-body">
            {experimentsQuery.isLoading ? (
              <div className="grid gap-4 sm:grid-cols-2">
                <SkeletonBlock className="h-14" />
                <SkeletonBlock className="h-14" />
              </div>
            ) : experiments.length === 0 ? (
              <EmptyState
                icon={<FlaskConical className="size-5" />}
                title="No completed experiments"
                description="Finish at least two runs before opening the comparison workspace."
                action={<Link href="/experiments/new" className="btn-primary">Create experiment</Link>}
              />
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="field-label" htmlFor="slot-a">Experiment A</label>
                  <select
                    id="slot-a"
                    className="select-shell"
                    value={effectiveSelectedIds[0] ?? ""}
                    onChange={(e) => setSlotA(e.target.value)}
                  >
                    <option value="">Select experiment…</option>
                    {experiments.map((exp: ExperimentListItem) => (
                      <option key={exp.id} value={exp.id} disabled={exp.id === effectiveSelectedIds[1]}>
                        {exp.name} — {exp.reasoning_method.toUpperCase()} · {exp.model_name.split("/").pop()}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="field-label" htmlFor="slot-b">Experiment B</label>
                  <select
                    id="slot-b"
                    className="select-shell"
                    value={effectiveSelectedIds[1] ?? ""}
                    onChange={(e) => setSlotB(e.target.value)}
                  >
                    <option value="">Select experiment…</option>
                    {experiments.map((exp: ExperimentListItem) => (
                      <option key={exp.id} value={exp.id} disabled={exp.id === effectiveSelectedIds[0]}>
                        {exp.name} — {exp.reasoning_method.toUpperCase()} · {exp.model_name.split("/").pop()}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>
        </Panel>
      </section>

      {effectiveSelectedIds.length < 2 || !effectiveSelectedIds[0] || !effectiveSelectedIds[1] ? (
        <EmptyState
          icon={<GitCompareArrows className="size-5" />}
          title={effectiveSelectedIds.filter(Boolean).length === 1 ? "Select one more experiment" : "Select two experiments to compare"}
          description="Once two completed runs are selected, the metrics, significance, and disagreement analysis will render here."
        />
      ) : comparisonQuery.isLoading || statsQuery.isLoading ? (
        <div className="space-y-4">
          <SkeletonBlock className="h-[180px]" />
          <SkeletonBlock className="h-[220px]" />
          <SkeletonBlock className="h-[320px]" />
        </div>
      ) : comparisonQuery.error || statsQuery.error || !comparison || !expA || !expB ? (
        <div className="alert alert-danger">
          <ScanSearch className="mt-0.5 size-4 shrink-0" />
          <div className="space-y-1">
            <div className="font-semibold">Comparison failed to load</div>
            <p className="text-sm text-(--muted-foreground)">
              {comparisonQuery.error instanceof Error
                ? comparisonQuery.error.message
                : statsQuery.error instanceof Error
                  ? statsQuery.error.message
                  : "The comparison response was incomplete."}
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <Panel>
            <PanelHeader
              label="Selected pair"
              title="Active comparison"
              description="Model, method, and dataset details for each run at a glance."
            />
            <div className="panel-body grid gap-3 lg:grid-cols-2">
              {[selectedA, selectedB].map((experiment, index) => (
                <div
                  key={experiment?.id ?? index}
                  className={cn(
                    "rounded-[18px] border p-4",
                    index === 0
                      ? "border-[color-mix(in_oklab,var(--primary)_34%,transparent)] bg-[color-mix(in_oklab,var(--primary)_10%,transparent)]"
                      : "border-[color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color-mix(in_oklab,var(--accent)_10%,transparent)]"
                  )}
                >
                  <div className="section-label">Experiment {index === 0 ? "A" : "B"}</div>
                  <div className="mt-2 text-xl font-semibold tracking-[-0.04em]">{experiment?.name}</div>
                  <div className="mt-3 flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
                    <span className="chip">{experiment?.reasoning_method.toUpperCase()}</span>
                    <span className="chip">{experiment?.model_name.split("/").pop()}</span>
                    <span className="chip">{experiment?.dataset_name}</span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Metrics" title="Head-to-head performance" description="Quality, latency, and cost metrics compared between the two runs." />
            <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              <MetricCompareCard label="Exact match accuracy" valueA={expA.metrics.quality.accuracy_exact} valueB={expB.metrics.quality.accuracy_exact} />
              <MetricCompareCard label="F1 score" valueA={expA.metrics.quality.accuracy_f1} valueB={expB.metrics.quality.accuracy_f1} />
              <MetricCompareCard label="Substring accuracy" valueA={expA.metrics.quality.accuracy_substring} valueB={expB.metrics.quality.accuracy_substring} />
              <MetricCompareCard label="Latency p50" valueA={expA.metrics.performance.latency_p50} valueB={expB.metrics.performance.latency_p50} format="ms" higherIsBetter={false} />
              <MetricCompareCard label="Latency p95" valueA={expA.metrics.performance.latency_p95} valueB={expB.metrics.performance.latency_p95} format="ms" higherIsBetter={false} />
              <MetricCompareCard
                label="Total tokens"
                valueA={(expA.metrics.cost.total_tokens_input || 0) + (expA.metrics.cost.total_tokens_output || 0)}
                valueB={(expB.metrics.cost.total_tokens_input || 0) + (expB.metrics.cost.total_tokens_output || 0)}
                format="number"
                higherIsBetter={false}
              />
            </div>
          </Panel>

          {statsQuery.data ? (
            <div className="grid gap-4 xl:grid-cols-2">
              <SignificancePanel stats={statsQuery.data} />
              <AgreementSummary summary={statsQuery.data.summary} />
            </div>
          ) : null}

          {statsQuery.data ? (
            <DifferencesTable diffs={statsQuery.data.per_example_differences} nameA={selectedA?.name ?? "A"} nameB={selectedB?.name ?? "B"} />
          ) : null}
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="page-stack"><SkeletonBlock className="h-[180px]" /><SkeletonBlock className="h-[400px]" /></div>}>
      <ComparePageInner />
    </Suspense>
  );
}
