"use client";
import { benchmarkPollDelay } from "@/lib/benchmark-polling";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getEvaluationCases, getEvaluationCase, isActive, type EvaluationSummary,
} from "@/lib/evaluation-api";
import { ResultsGrid } from "@/components/evaluations/evaluation-results";
import { scopedKey } from "@/features/shared/query-keys";
import { PageControls } from "@/components/ui/page-controls";
import { errorText } from "@/components/prompts/prompt-ui";

const PAGE_SIZE = 25;

export function PagedResults({ detail, comparison, filter, caseIndex, selectCase }: {
  detail: EvaluationSummary;
  comparison?: EvaluationSummary;
  filter: string;
  caseIndex: number | null;
  selectCase: (index: number) => void;
}) {
  const scope = `${detail.run.id}:${comparison?.run.id ?? ""}:${filter}`;
  const [paging, setPaging] = useState({ scope, offset: 0 });
  const offset = paging.scope === scope ? paging.offset : 0;
  const setOffset = (offset: number) => setPaging({ scope, offset });
  const rows = useQuery({
    queryKey: scopedKey(["evaluation-cases", detail.run.id, comparison?.run.id, filter, offset, detail.run.completed]),
    queryFn: ({ signal }) => getEvaluationCases(detail.run.id, {
      offset, limit: PAGE_SIZE, filter, reference_id: comparison?.run.id,
    }, signal),
    refetchInterval: isActive(detail.run) ? () => benchmarkPollDelay(detail.run.created_at) : false,
  });
  const selected = caseIndex ?? rows.data?.items[0]?.case_index;
  const candidate = useQuery({
    queryKey: scopedKey(["evaluation-case", detail.run.id, selected]),
    queryFn: ({ signal }) => getEvaluationCase(detail.run.id, selected!, signal),
    enabled: selected != null,
  });
  const reference = useQuery({
    queryKey: scopedKey(["evaluation-case", comparison?.run.id, selected]),
    queryFn: ({ signal }) => getEvaluationCase(comparison!.run.id, selected!, signal),
    enabled: !!comparison && selected != null,
    retry: false,
  });
  useEffect(() => {
    if (candidate.data && caseIndex != null) {
      document.getElementById("selected-case-detail")?.focus();
    }
  }, [candidate.data, caseIndex]);

  const caseTable = (
    <div className="space-y-3">
      {rows.error && <p role="alert">{errorText(rows.error)}</p>}
      {rows.isLoading && <p role="status" className="field-help">Loading case rows…</p>}
      <div className="case-table responsive-table" role="region" aria-label="Evaluation case rows" tabIndex={0}>
        <table className="data-table">
          <thead><tr><th>Case</th><th>Result</th><th>Latency</th><th>Output preview</th></tr></thead>
          <tbody>
            {rows.data?.items.map((row) => (
              <tr key={row.case_index} aria-selected={selected === row.case_index}>
                <td data-label="Case">
                  <button type="button" className="text-(--primary) hover:underline" onClick={() => selectCase(row.case_index)}>
                    {row.name || `Case ${row.case_index + 1}`}
                  </button>
                </td>
                <td data-label="Result">{row.error ? "Error" : row.passed ? "Passed" : "Failed checks"}</td>
                <td data-label="Latency" className="numeric">
                  {row.latency_ms == null ? "—" : `${row.latency_ms.toFixed(1)} ms`}
                </td>
                <td data-label="Output preview" className="max-w-80 truncate">{row.output_preview || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.error && !rows.isLoading && !rows.data?.items.length && <p className="p-4">No matching results yet.</p>}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="field-help">{rows.data?.total ?? 0} matching cases · page {Math.floor(offset / PAGE_SIZE) + 1}</span>
        <PageControls offset={offset} next={offset + PAGE_SIZE < (rows.data?.total ?? 0)} change={setOffset} pageSize={PAGE_SIZE} />
      </div>
    </div>
  );
  const referenceFailed = reference.error instanceof Error && !("statusCode" in reference.error && reference.error.statusCode === 404);

  return (
    <div className="mt-5 grid gap-4">
      {(candidate.error || referenceFailed) && <p role="alert">Could not load selected case details.</p>}
      {candidate.isLoading && <p role="status" className="field-help">Loading case details…</p>}
      <ResultsGrid
        detail={{ ...detail, results: candidate.data ? [candidate.data] : [] }}
        comparison={comparison ? { ...comparison, results: reference.data ? [reference.data] : [] } : undefined}
        filter="all"
        hideTable
        caseTable={caseTable}
        comparisonTotals={rows.data?.comparison_totals ?? undefined}
        selectedCaseIndex={selected}
      />
    </div>
  );
}
