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
  MetricCompareCard,
  SignificancePanel,
} from "@/components/experiments/compare-sections";
import { EmptyState, PageHeader, Panel, PanelHeader, SkeletonBlock } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

function ComparePageInner() {
  const searchParams = useSearchParams();
  const preselect = searchParams.get("preselect");
  const [selectedIds, setSelectedIds] = useState<string[]>(() => (preselect ? [preselect] : []));

  const [attemptA, setAttemptA] = useState<number | undefined>(() => /^\d+$/.test(searchParams.get("attempt") ?? "") ? Number(searchParams.get("attempt")) : undefined);
  const [attemptB, setAttemptB] = useState<number | undefined>();
  const [skip, setSkip] = useState(0);

  const experimentsQuery = useQuery({
    queryKey: ["experiments", "completed", skip],
    queryFn: ({ signal }) => listExperimentsSlim({ status: "completed", limit: 25, skip }, { signal }),
  });

  const experiments = useMemo(() => experimentsQuery.data?.experiments ?? [], [experimentsQuery.data?.experiments]);
  const effectiveSelectedIds = useMemo(() => selectedIds.filter(Boolean).slice(0, 2), [selectedIds]);

  const comparisonQuery = useQuery({
    queryKey: ["comparison", effectiveSelectedIds, attemptA, attemptB],
    queryFn: ({ signal }) => compareExperiments(effectiveSelectedIds, { signal }, [attemptA, attemptB]),
    enabled: effectiveSelectedIds.length === 2,
  });

  const statsQuery = useQuery({
    queryKey: ["statistical", effectiveSelectedIds, attemptA, attemptB],
    queryFn: ({ signal }) => getStatisticalComparison(effectiveSelectedIds[0], effectiveSelectedIds[1], { signal }, [attemptA, attemptB]),
    enabled: effectiveSelectedIds.length === 2,
  });

  const comparison = comparisonQuery.data;
  const expA = comparison?.experiments[0];
  const expB = comparison?.experiments[1];

  function setSlotA(id: string) {
    setAttemptA(undefined);
    setSelectedIds((current) => {
      const base = current.slice(0, 2);
      if (!id) return base.length > 1 ? ["", base[1]] : [];
      return base.length > 1 ? [id, base[1]] : [id];
    });
  }

  function setSlotB(id: string) {
    setAttemptB(undefined);
    setSelectedIds((current) => {
      const base = current.slice(0, 2);
      if (!id) return base.length > 0 ? [base[0]] : [];
      return [base[0] || "", id];
    });
  }

  return (
    <div className="page-stack">
      <PageHeader
        backHref="/experiments"
        backLabel="Back to benchmarks"
        eyebrow={<><GitCompareArrows className="size-3.5" /> Comparison workspace</>}
        title="Compare benchmarks side by side"
        description="Select two completed runs to compare accuracy, latency, cost, and statistical significance."
      />

      <section>
        <Panel>
          <PanelHeader
            label="Selection"
            title="Choose two completed runs"
            description="Pick one benchmark for each slot to begin the comparison."
            actions={effectiveSelectedIds.length ? <button type="button" className="btn-secondary" onClick={() => setSelectedIds([])}>Clear</button> : null}
          />
          <div className="panel-body">
            <div className="mb-4 flex items-center gap-3">
              <button className="btn-secondary" disabled={skip === 0} onClick={() => setSkip(Math.max(0, skip - 25))}>Newer benchmarks</button>
              <span>{skip + 1}–{skip + experiments.length} of {experimentsQuery.data?.total ?? "?"}</span>
              <button className="btn-secondary" disabled={skip + 25 >= (experimentsQuery.data?.total ?? 0)} onClick={() => setSkip(skip + 25)}>Older benchmarks</button>
            </div>
            {experimentsQuery.isLoading ? (
              <div className="grid gap-4 sm:grid-cols-2">
                <SkeletonBlock className="h-14" />
                <SkeletonBlock className="h-14" />
              </div>
            ) : experiments.length === 0 ? (
              <EmptyState
                icon={<FlaskConical className="size-5" />}
                title="No completed benchmarks"
                description="Finish at least two runs before opening the comparison workspace."
                action={<Link href="/experiments/new" className="btn-primary">Create benchmark</Link>}
              />
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="field-label" htmlFor="slot-a">Benchmark A</label>
                  <select
                    id="slot-a"
                    className="select-shell"
                    value={effectiveSelectedIds[0] ?? ""}
                    onChange={(e) => setSlotA(e.target.value)}
                  >
                    <option value="">Select benchmark…</option>
                    {effectiveSelectedIds.filter((id) => !experiments.some((exp) => exp.id === id)).map((id) => <option key={id} value={id}>{comparison?.experiments.find((exp) => exp.experiment_id === id)?.experiment_name ?? id} (selected from another page)</option>)}
                    {experiments.map((exp: ExperimentListItem) => (
                      <option key={exp.id} value={exp.id} disabled={exp.id === effectiveSelectedIds[1]}>
                        {exp.name} — {exp.reasoning_method.toUpperCase()} · {exp.model_name.split("/").pop()}
                      </option>
                    ))}
                  </select>
                  <label className="field-label">Attempt (blank = current)<input className="field-input" type="number" min="0" value={attemptA ?? ""} onChange={(event) => setAttemptA(event.target.value === "" ? undefined : Number(event.target.value))} /></label>
                </div>
                <div className="space-y-2">
                  <label className="field-label" htmlFor="slot-b">Benchmark B</label>
                  <select
                    id="slot-b"
                    className="select-shell"
                    value={effectiveSelectedIds[1] ?? ""}
                    onChange={(e) => setSlotB(e.target.value)}
                  >
                    <option value="">Select benchmark…</option>
                    {effectiveSelectedIds.filter((id) => !experiments.some((exp) => exp.id === id)).map((id) => <option key={id} value={id}>{comparison?.experiments.find((exp) => exp.experiment_id === id)?.experiment_name ?? id} (selected from another page)</option>)}
                    {experiments.map((exp: ExperimentListItem) => (
                      <option key={exp.id} value={exp.id} disabled={exp.id === effectiveSelectedIds[0]}>
                        {exp.name} — {exp.reasoning_method.toUpperCase()} · {exp.model_name.split("/").pop()}
                      </option>
                    ))}
                  </select>
                  <label className="field-label">Attempt (blank = current)<input className="field-input" type="number" min="0" value={attemptB ?? ""} onChange={(event) => setAttemptB(event.target.value === "" ? undefined : Number(event.target.value))} /></label>
                </div>
              </div>
            )}
          </div>
        </Panel>
      </section>

      {effectiveSelectedIds.length < 2 || !effectiveSelectedIds[0] || !effectiveSelectedIds[1] ? (
        <EmptyState
          icon={<GitCompareArrows className="size-5" />}
          title={effectiveSelectedIds.filter(Boolean).length === 1 ? "Select one more benchmark" : "Select two benchmarks to compare"}
          description="Once two completed runs are selected, the metrics, significance, and disagreement analysis will render here."
        />
      ) : comparisonQuery.isLoading || statsQuery.isLoading ? (
        <div className="space-y-4">
          <SkeletonBlock className="h-[180px]" />
          <SkeletonBlock className="h-[220px]" />
          <SkeletonBlock className="h-[320px]" />
        </div>
      ) : comparisonQuery.error || !comparison || !expA || !expB ? (
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
          {statsQuery.error ? <p className="alert alert-warning">Paired analysis unavailable for these attempts. Historical aggregates remain descriptive.</p> : null}
          <p>Attempt A: {expA.metrics.attempt ?? "unknown"}; attempt B: {expB.metrics.attempt ?? "unknown"}. Provenance and coverage affect comparability.</p>
          <Panel>
            <PanelHeader
              label="Selected pair"
              title="Active comparison"
              description="Model, method, and dataset details for each run at a glance."
            />
            <div className="panel-body grid gap-3 lg:grid-cols-2">
              {[expA, expB].map((experiment, index) => (
                <div
                  key={experiment.experiment_id}
                  className={cn(
                    "rounded-[18px] border p-4",
                    index === 0
                      ? "border-[color-mix(in_oklab,var(--primary)_34%,transparent)] bg-[color-mix(in_oklab,var(--primary)_10%,transparent)]"
                      : "border-[color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color-mix(in_oklab,var(--accent)_10%,transparent)]"
                  )}
                >
                  <div className="section-label">Benchmark {index === 0 ? "A" : "B"}</div>
                  <div className="mt-2 text-xl font-semibold tracking-[-0.04em]">{experiment.experiment_name}</div>
                  <div className="mt-3 flex flex-wrap gap-2 text-xs text-(--muted-foreground)">
                    <span className="chip">{experiment.method.toUpperCase()}</span>
                    <span className="chip">{experiment.model.split("/").pop()}</span>
                    <span className="chip">{`Attempt ${experiment.metrics.attempt ?? "unknown"}`}</span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel>
            <PanelHeader label="Metrics" title="Head-to-head performance" description="Quality, latency, and cost metrics compared between the two runs." />
            <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              <MetricCompareCard label="Primary accuracy" valueA={expA.metrics.quality.primary_accuracy} valueB={expB.metrics.quality.primary_accuracy} />
              <MetricCompareCard label="Exact match diagnostic" valueA={expA.metrics.quality.accuracy_exact} valueB={expB.metrics.quality.accuracy_exact} />
              <MetricCompareCard label="F1 score" valueA={expA.metrics.quality.accuracy_f1} valueB={expB.metrics.quality.accuracy_f1} />
              <MetricCompareCard label="Substring diagnostic" valueA={expA.metrics.quality.accuracy_substring} valueB={expB.metrics.quality.accuracy_substring} />
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
            <DifferencesTable diffs={statsQuery.data.per_example_differences} nameA={expA.experiment_name} nameB={expB.experiment_name} />
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
