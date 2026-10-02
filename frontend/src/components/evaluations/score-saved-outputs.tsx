"use client";
import { useState } from "react";
import { EvaluatorSelectionFields } from "./evaluator-selection";
import type { EvaluatorSelection } from "@/lib/evaluator-api";
import type { EvaluationRun } from "@/lib/evaluation-api";
import { scoreSavedOutputs } from "@/lib/evaluation-api";
import { errorText, panelClass } from "@/components/prompts/prompt-ui";

export function ScoreSavedOutputs({
  run,
  started,
}: {
  run: EvaluationRun;
  started: (run: EvaluationRun) => void;
}) {
  const [evaluators, setEvaluators] = useState<EvaluatorSelection[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const judges = evaluators.filter((item) => item.kind === "llm_judge").length;
  const calls = judges * run.total;
  return (
    <details className={`${panelClass} mt-4`}>
      <summary className="cursor-pointer font-medium">
        Score saved outputs again
      </summary>
      <div className="grid gap-4 mt-4 max-w-2xl">
        <p className="field-help">
          Creates a separate scoring run linked to this run. Saved outputs and
          previous scores stay intact.
        </p>
        <EvaluatorSelectionFields value={evaluators} onChange={setEvaluators} />
        <p className="text-sm">
          0 generation calls · up to {calls} judge calls for {run.total} saved
          cases. Source cases without outputs are reported as errors.
        </p>
        {error && (
          <p role="alert" className="text-(--destructive)">
            {error}
          </p>
        )}
        <button
          className="btn-primary justify-self-start"
          disabled={
            busy ||
            !evaluators.some((item) => item.required) ||
            judges > 2 ||
            calls > 60 ||
            (judges > 0 && run.total > 20) ||
            evaluators.some(
              (item) => item.kind === "llm_judge" && !item.api_key?.trim(),
            )
          }
          onClick={async () => {
            setBusy(true);
            setError("");
            try {
              const result = await scoreSavedOutputs(run.id, evaluators);
              setEvaluators([]);
              started(result);
            } catch (failure) {
              setError(errorText(failure));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Starting…" : "Start scoring saved outputs"}
        </button>
        <p className="field-help">
          At least one required evaluator; at most two judges and 60 model
          calls. Runs with judges are limited to 20 cases.
        </p>
      </div>
    </details>
  );
}
