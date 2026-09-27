import type { EvaluationDetail } from "@/lib/evaluation-api";

export function RunSummary({ detail }: { detail: EvaluationDetail }) {
  const run = detail.run;
  return (
    <div>
      <h2 className="font-semibold">
        {run.config.prompt_name} v{run.config.prompt_version} ·{" "}
        {run.config.dataset_name} v{run.config.dataset_version}
      </h2>
      <p className="text-sm text-slate-400">
        {run.status} · {run.completed}/{run.total} processed · {run.passed}{" "}
        passed · {run.completed - run.passed - run.errors} failed checks ·{" "}
        {run.errors} errors · {run.config.provider} / {run.config.model}
      </p>
      <p className="text-sm mt-1">
        {run.config.source === "sdk_submission"
          ? "External evaluation: outputs, checks and metrics supplied by your application."
          : run.config.is_mock
            ? "Demo: output echoes the compiled prompt. No generation model was called."
            : "Live provider generation."}
        {run.config.judge &&
          ` Judged by ${run.config.judge.provider} / ${run.config.judge.model}.`}
      </p>
      {run.config.metrics && Object.keys(run.config.metrics).length > 0 && (
        <pre className="text-xs whitespace-pre-wrap mt-2">
          {JSON.stringify(run.config.metrics, null, 2)}
        </pre>
      )}
      {run.error && (
        <p role="alert" className="text-red-400">
          {run.error}
        </p>
      )}
    </div>
  );
}
export function ResultsGrid({
  detail,
  comparison,
  filter,
}: {
  detail: EvaluationDetail;
  comparison?: EvaluationDetail;
  filter: string;
}) {
  const rows = detail.results.filter(
    (r) =>
      filter === "all" ||
      (filter === "passed" && r.passed) ||
      (filter === "errors" && !!r.error) ||
      (filter === "failed" && !r.passed && !r.error),
  );
  return (
    <div className="overflow-x-auto mt-5">
      {comparison && (
        <div className="mb-3">
          <p className="text-xs text-slate-400">Comparison</p>
          <RunSummary detail={comparison} />
        </div>
      )}
      <table className="w-full min-w-[800px] text-sm text-left">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="p-3">Case / inputs</th>
            <th className="p-3">Reference</th>
            <th className="p-3">Output / checks</th>
            <th className="p-3">Latency / tokens</th>
            {comparison && <th className="p-3">Compared output / checks</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((result) => {
            const compared = comparison?.results.find(
              (r) => r.case_index === result.case_index,
            );
            return (
              <tr
                key={result.case_index}
                className="align-top border-b border-slate-700"
              >
                <td className="p-3 max-w-64">
                  <strong>
                    {result.name || `Case ${result.case_index + 1}`}
                  </strong>
                  <pre className="whitespace-pre-wrap break-words text-xs mt-2">
                    {JSON.stringify(result.inputs, null, 2)}
                  </pre>
                </td>
                <td className="p-3 max-w-64 whitespace-pre-wrap break-words">
                  {result.expected_output ?? "No reference"}
                </td>
                <td className="p-3 max-w-96">
                  <ResultCell result={result} />
                </td>
                <td className="p-3">
                  {result.latency_ms?.toFixed(1) ?? "—"} ms
                  <br />
                  Input: {result.tokens_input ?? "—"}
                  <br />
                  Output: {result.tokens_output ?? "—"}
                </td>
                {comparison && (
                  <td className="p-3 max-w-96">
                    {compared ? (
                      <ResultCell result={compared} />
                    ) : (
                      "Not processed"
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {!rows.length && (
        <p className="text-slate-400 p-3">No matching results yet.</p>
      )}
    </div>
  );
}
function ResultCell({
  result,
}: {
  result: EvaluationDetail["results"][number];
}) {
  return (
    <div>
      <span
        className={
          result.error
            ? "text-amber-400"
            : result.passed
              ? "text-emerald-400"
              : "text-red-400"
        }
      >
        {result.error ? "Error" : result.passed ? "Passed" : "Failed checks"}
      </span>
      {result.error && <p className="text-amber-400 mt-2">{result.error}</p>}
      <pre className="whitespace-pre-wrap break-words text-xs mt-2">
        {result.output}
      </pre>
      {result.checks.map((check, i) => (
        <p key={i} className="text-xs mt-2">
          {check.passed ? "✓" : "✗"} {check.kind}
          {check.score !== undefined && ` (${check.score.toFixed(2)})`}:{" "}
          {check.reason}
        </p>
      ))}
    </div>
  );
}
