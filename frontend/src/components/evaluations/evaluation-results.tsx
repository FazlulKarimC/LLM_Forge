"use client";

import { useState } from "react";
import type { EvaluationDetail } from "@/lib/evaluation-api";

type CaseResult = EvaluationDetail["results"][number];

export function RunSummary({ detail }: { detail: EvaluationDetail }) {
  const { run } = detail;
  const failed = Math.max(0, run.completed - run.passed - run.errors);
  const passRate = run.completed
    ? `${Math.round((run.passed / run.completed) * 100)}%`
    : "—";
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-semibold">
          {run.config.prompt_name} v{run.config.prompt_version} ·{" "}
          {run.config.dataset_name} v{run.config.dataset_version}
        </h2>
        <span className="text-sm text-(--muted-foreground)">{run.status}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <SummaryMetric label="Pass rate" value={passRate} />
        <SummaryMetric
          label="Processed"
          value={`${run.completed}/${run.total}`}
        />
        <SummaryMetric label="Failed checks" value={String(failed)} />
        <SummaryMetric label="Errors" value={String(run.errors)} />
      </div>
      <p className="text-xs text-(--muted-foreground)">
        {run.passed} passed · {run.config.provider} / {run.config.model} ·{" "}
        {run.config.source === "sdk_submission"
          ? "Outputs supplied by your application"
          : run.config.is_mock
            ? "Echo demo; no generation model called"
            : "Live provider generation"}
        {run.config.judge &&
          ` · Judge: ${run.config.judge.provider} / ${run.config.judge.model}`}
      </p>
      {run.config.metrics && Object.keys(run.config.metrics).length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer">Submitted metrics</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words">
            {JSON.stringify(run.config.metrics, null, 2)}
          </pre>
        </details>
      )}
      {run.error && (
        <p role="alert" className="text-sm text-(--destructive)">
          {run.error}
        </p>
      )}
    </div>
  );
}

function SummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-(--border) px-3 py-2">
      <p className="text-xs text-(--muted-foreground)">{label}</p>
      <p className="font-semibold">{value}</p>
    </div>
  );
}

function transition(current: CaseResult, prior: CaseResult | undefined) {
  if (!prior) return "unmatched";
  if (current.error || prior.error) return "errors";
  if (current.passed && !prior.passed) return "improved";
  if (!current.passed && prior.passed) return "regressed";
  return "unchanged";
}

export function ResultsGrid({
  detail,
  comparison,
  filter,
  selectedCaseIndex,
  onSelectCase,
}: {
  detail: EvaluationDetail;
  comparison?: EvaluationDetail;
  filter: string;
  selectedCaseIndex?: number | null;
  onSelectCase?: (index: number) => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const priorCases = new Map(
    comparison?.results.map((result) => [result.case_index, result]) ?? [],
  );
  const transitions = { improved: 0, regressed: 0, unchanged: 0, errors: 0 };
  if (comparison) {
    for (const current of detail.results) {
      const change = transition(current, priorCases.get(current.case_index));
      if (change !== "unmatched") transitions[change]++;
    }
  }
  const rows = detail.results.filter((result) => {
    if (filter === "all") return true;
    if (filter === "passed") return result.passed;
    if (filter === "errors") return !!result.error;
    if (filter === "comparison_errors")
      return transition(result, priorCases.get(result.case_index)) === "errors";
    if (filter === "failed") return !result.passed && !result.error;
    if (
      filter === "improved" ||
      filter === "regressed" ||
      filter === "unchanged"
    )
      return transition(result, priorCases.get(result.case_index)) === filter;
    return true;
  });
  const selected =
    rows.find(
      (result) => result.case_index === (selectedCaseIndex ?? selectedIndex),
    ) ?? rows[0];
  const prior = selected && priorCases.get(selected.case_index);
  const sameChecks =
    JSON.stringify(detail.run.config.assertions ?? []) ===
    JSON.stringify(comparison?.run.config.assertions ?? []);

  return (
    <div className="mt-5 space-y-4">
      {comparison && (
        <div className="rounded-lg border border-(--border) p-4">
          <p role="status" className="text-sm">
            Compared with the selected run: {transitions.improved} improved,{" "}
            {transitions.regressed} regressed, {transitions.unchanged}{" "}
            unchanged, {transitions.errors} involving errors.
          </p>
          {!sameChecks && (
            <p className="mt-2 text-sm text-(--warning)">
              These runs use different assertions; pass/fail changes may reflect
              the checks rather than the outputs.
            </p>
          )}
        </div>
      )}
      <div className={selected ? "case-inspection" : "space-y-4"}>
        <div className="case-table">
          <table className="w-full min-w-[500px] text-left text-sm">
            <thead>
              <tr className="border-b border-(--border) bg-(--muted)/30">
                <th className="p-3">Case</th>
                <th className="p-3">Result</th>
                {comparison && <th className="p-3">Change</th>}
                <th className="p-3">Latency</th>
                <th className="p-3">Output preview</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((result) => (
                <tr
                  key={result.case_index}
                  aria-selected={selected?.case_index === result.case_index}
                  className="border-b border-(--border) last:border-0"
                >
                  <td className="p-3">
                    <button
                      type="button"
                      className="font-medium text-(--primary) hover:underline"
                      onClick={() => {
                        setSelectedIndex(result.case_index);
                        onSelectCase?.(result.case_index);
                        document
                          .getElementById("selected-case-detail")
                          ?.focus();
                      }}
                    >
                      {result.name || `Case ${result.case_index + 1}`}
                    </button>
                  </td>
                  <td className="p-3">
                    {result.error
                      ? "Error"
                      : result.passed
                        ? "Passed"
                        : "Failed checks"}
                  </td>
                  {comparison && (
                    <td className="p-3 capitalize">
                      {transition(result, priorCases.get(result.case_index))}
                    </td>
                  )}
                  <td className="p-3">
                    {result.latency_ms == null
                      ? "—"
                      : `${result.latency_ms.toFixed(1)} ms`}
                  </td>
                  <td className="max-w-80 truncate p-3">
                    {result.output || (result.error ? "—" : "Empty output")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && (
            <p className="p-4 text-sm text-(--muted-foreground)">
              No matching results yet.
            </p>
          )}
        </div>
        {selected && (
          <section
            className="case-detail"
            id="selected-case-detail"
            tabIndex={-1}
            aria-label="Selected case details"
          >
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="font-semibold">
                {selected.name || `Case ${selected.case_index + 1}`}
              </h3>
              <p className="text-sm text-(--muted-foreground)">
                Case {selected.case_index + 1} · {selected.tokens_input ?? "—"}{" "}
                input tokens · {selected.tokens_output ?? "—"} output tokens
              </p>
            </div>
            <div className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <DetailField
                  label="Inputs"
                  value={JSON.stringify(selected.inputs, null, 2)}
                />
                <DetailField
                  label="Reference output"
                  value={selected.expected_output ?? "No reference"}
                />
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs text-(--muted-foreground)">
                    Candidate output and checks
                  </p>
                  <ResultCell result={selected} />
                </div>
                {comparison && (
                  <div>
                    <p className="mb-1 text-xs text-(--muted-foreground)">
                      Reference run output and checks
                    </p>
                    {prior ? (
                      <ResultCell result={prior} />
                    ) : (
                      <p className="text-sm">Not processed</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="mb-1 text-xs text-(--muted-foreground)">{label}</p>
      <pre className="whitespace-pre-wrap break-words rounded-md bg-(--muted)/30 p-3 text-xs">
        {value}
      </pre>
    </div>
  );
}

function ResultCell({ result }: { result: CaseResult }) {
  return (
    <div className="text-sm">
      <span
        className={
          result.error
            ? "text-(--warning)"
            : result.passed
              ? "text-(--success)"
              : "text-(--destructive)"
        }
      >
        {result.error ? "Error" : result.passed ? "Passed" : "Failed checks"}
      </span>
      {result.error && <p className="mt-2 text-(--warning)">{result.error}</p>}
      <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-(--muted)/30 p-3 text-xs">
        {result.output || "Empty output"}
      </pre>
      {result.checks.map((check, index) => (
        <p key={index} className="mt-2 text-xs">
          {check.passed ? "✓" : "✗"} {check.kind}
          {check.score !== undefined && ` (${check.score.toFixed(2)})`}:{" "}
          {check.reason}
        </p>
      ))}
    </div>
  );
}
