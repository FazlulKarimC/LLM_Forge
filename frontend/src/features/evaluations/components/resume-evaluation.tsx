"use client";

import { useState } from "react";
import { resumeEvaluation, type EvaluationSummary } from "@/lib/evaluation-api";
import { useDispatchIntent } from "../hooks/use-dispatch-intent";
import { inputClass } from "@/components/prompts/prompt-ui";
import { evaluationRequestError } from "../lib/request-error";

export function ResumeEvaluation({ summary, resumed }: { summary: EvaluationSummary; resumed: () => void }) {
  const [apiKey, setApiKey] = useState("");
  const [judgeKey, setJudgeKey] = useState("");
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const intent = useDispatchIntent();
  if (!summary.recovery?.available) return null;
  const run = summary.run;
  const calls = summary.recovery.remaining_calls;
  const judges = run.config.evaluators?.filter(item => item.definition.kind === "llm_judge") ?? [];
  return <section className="mt-4 rounded-lg border border-(--warning) p-4 grid gap-3 max-w-2xl" aria-label="Resume interrupted evaluation">
    <h3 className="font-semibold">Execution interrupted</h3>
    <p className="text-sm">Saved outputs and completed checks are preserved. Resume unfinished work using the same pinned versions.</p>
    <p className="field-help">Up to {calls.generation} generation + {calls.judging} judge calls remain. A provider call whose response was lost may be charged again. Keys are used in memory and must be entered again after interruption.</p>
    {run.config.provider !== "mock" && run.config.source !== "score_only" && <label className="text-sm">Generation API key<input type="password" autoComplete="off" className={inputClass} value={apiKey} onChange={e => setApiKey(e.target.value)} /></label>}
    {run.config.judge && <label className="text-sm">Inline judge API key<input type="password" autoComplete="off" className={inputClass} value={judgeKey} onChange={e => setJudgeKey(e.target.value)} /></label>}
    {judges.map(item => <label key={item.version_id} className="text-sm">{item.name} v{item.version} API key<input type="password" autoComplete="off" className={inputClass} value={keys[item.version_id] ?? ""} onChange={e => setKeys({ ...keys, [item.version_id]: e.target.value })} /></label>)}
    {error && <p role="alert" className="text-(--destructive)">{error}</p>}
    <button className="btn-primary justify-self-start" disabled={busy} onClick={async () => {
      setBusy(true); setError("");
      try {
        await resumeEvaluation(run.id, { api_key: apiKey || undefined, judge_key: judgeKey || undefined, evaluator_keys: keys }, intent.key({ run: run.id, attempt: summary.recovery.attempts.length }));
        intent.accepted(); setApiKey(""); setJudgeKey(""); setKeys({}); resumed();
      } catch (failure) { setError(evaluationRequestError(failure, true)); } finally { setBusy(false); }
    }}>{busy ? "Resuming…" : "Resume unfinished work"}</button>
  </section>;
}
