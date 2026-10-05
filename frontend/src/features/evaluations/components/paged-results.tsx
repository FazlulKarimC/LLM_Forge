"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getEvaluationCases, getEvaluationCase, isActive, type EvaluationSummary } from "@/lib/evaluation-api";
import { ResultsGrid } from "@/components/evaluations/evaluation-results";
import { scopedKey } from "@/features/shared/query-keys";
import { errorText } from "@/components/prompts/prompt-ui";

export function PagedResults({ detail, comparison, filter, caseIndex, selectCase }: {
  detail: EvaluationSummary; comparison?: EvaluationSummary; filter: string; caseIndex: number | null; selectCase: (index: number) => void;
}) {
  const scope = `${detail.run.id}:${comparison?.run.id ?? ""}:${filter}`;
  const [paging, setPaging] = useState({ scope, offset: 0 });
  const offset = paging.scope === scope ? paging.offset : 0;
  const setOffset = (offset: number) => setPaging({ scope, offset });
  const rows = useQuery({ queryKey: scopedKey(["evaluation-cases", detail.run.id, comparison?.run.id, filter, offset, detail.run.completed]),
    queryFn: ({ signal }) => getEvaluationCases(detail.run.id, { offset, limit: 25, filter, reference_id: comparison?.run.id }, signal),
    refetchInterval: isActive(detail.run) ? 2500 : false });
  const selected = caseIndex ?? rows.data?.items[0]?.case_index;
  const candidate = useQuery({ queryKey: scopedKey(["evaluation-case", detail.run.id, selected]),
    queryFn: ({ signal }) => getEvaluationCase(detail.run.id, selected!, signal), enabled: selected != null });
  const reference = useQuery({ queryKey: scopedKey(["evaluation-case", comparison?.run.id, selected]),
    queryFn: ({ signal }) => getEvaluationCase(comparison!.run.id, selected!, signal), enabled: !!comparison && selected != null, retry: false });
  useEffect(() => { if (candidate.data && caseIndex != null) document.getElementById("selected-case-detail")?.focus(); }, [candidate.data, caseIndex]);
  return <div className="mt-5 grid gap-4">
    {rows.error && <p role="alert">{errorText(rows.error)}</p>}
    {rows.isLoading && <p role="status" className="field-help">Loading case rows…</p>}
    <div className="case-table"><table className="data-table"><thead><tr><th>Case</th><th>Result</th><th>Latency</th><th>Output preview</th></tr></thead><tbody>
      {rows.data?.items.map(row => <tr key={row.case_index} aria-selected={selected === row.case_index}><td><button className="text-(--primary) hover:underline" onClick={() => selectCase(row.case_index)}>{row.name || `Case ${row.case_index + 1}`}</button></td><td>{row.error ? "Error" : row.passed ? "Passed" : "Failed checks"}</td><td>{row.latency_ms == null ? "—" : `${row.latency_ms.toFixed(1)} ms`}</td><td className="max-w-80 truncate">{row.output_preview || "—"}</td></tr>)}
    </tbody></table>{!rows.isLoading && !rows.data?.items.length && <p className="p-4">No matching results yet.</p>}</div>
    <div className="flex items-center justify-between"><span className="field-help">{rows.data?.total ?? 0} matching cases · page {Math.floor(offset / 25) + 1}</span><div className="flex gap-2"><button className="btn-secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>Previous</button><button className="btn-secondary" disabled={offset + 25 >= (rows.data?.total ?? 0)} onClick={() => setOffset(offset + 25)}>Next</button></div></div>
    {(candidate.error || (reference.error && reference.error instanceof Error && !("statusCode" in reference.error && reference.error.statusCode === 404))) && <p role="alert">Could not load selected case details.</p>}
    {candidate.isLoading && <p role="status" className="field-help">Loading case details…</p>}
    <ResultsGrid detail={{ ...detail, results: candidate.data ? [candidate.data] : [] }} comparison={comparison ? { ...comparison, results: reference.data ? [reference.data] : [] } : undefined} filter="all" hideTable comparisonTotals={rows.data?.comparison_totals ?? undefined} selectedCaseIndex={selected} />
  </div>;
}
