"use client";
import { useEffect, useState } from "react";
import type { Provider } from "@/lib/prompt-api";
import type { Assertion, Judge } from "@/lib/evaluation-api";
import type { EvaluatorSelection } from "@/lib/evaluator-api";
export const defaults: Record<Provider, string> = { mock: "demo-model", groq: "openai/gpt-oss-20b", openrouter: "openrouter/free", openai: "gpt-4o-mini" };
export function useEvaluationEditor() {
  const [promptId, setPromptId] = useState("");
  const [promptOffset, setPromptOffset] = useState(0);
  const [versionOffset, setVersionOffset] = useState(0);
  const [versionId, setVersionId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasetOffset, setDatasetOffset] = useState(0);
  const [revisionOffset, setRevisionOffset] = useState(0);
  const [revisionId, setRevisionId] = useState("");
  const [provider, setProvider] = useState<Provider>("mock");
  const [model, setModel] = useState(defaults.mock);
  const [apiKey, setApiKey] = useState("");
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(256);
  const [rules, setRules] = useState<Assertion[]>([
    { kind: "exact_match", value: "", path: "" },
  ]);
  const [judgeEnabled, setJudgeEnabled] = useState(false);
  const [judge, setJudge] = useState<Judge>({
    provider: "groq",
    model: defaults.groq,
    api_key: "",
    rubric:
      "Does the response correctly answer the input and agree with the reference?",
    threshold: 0.7,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [runId, setRunId] = useState("");
  const [compareId, setCompareId] = useState("");
  const [runOffset, setRunOffset] = useState(0);
  const [runSearch, setRunSearch] = useState("");
  const [runStatus, setRunStatus] = useState("");
  const [referenceSearch, setReferenceSearch] = useState("");
  const [referenceOffset, setReferenceOffset] = useState(0);
  const [evaluators, setEvaluators] = useState<EvaluatorSelection[]>([]);
  const [view, setView] = useState<"history" | "new" | "detail">("history");
  const [filter, setFilter] = useState("all");
  const [caseIndex, setCaseIndex] = useState<number | null>(null);
  const [urlReady, setUrlReady] = useState(false);
  useEffect(() => {
    const restore = () => {
      const params = new URLSearchParams(window.location.search);
      setPromptId(params.get("prompt") ?? "");
      setVersionId(params.get("version") ?? "");
      setDatasetId(params.get("dataset") ?? "");
      setRevisionId(params.get("revision") ?? "");
      const restoredRun = params.get("run") ?? "";
      setRunId(restoredRun);
      setCompareId(params.get("compare") ?? "");
      const selectedCase = params.get("case");
      setCaseIndex(
        selectedCase !== null && /^\d+$/.test(selectedCase)
          ? Number(selectedCase)
          : null,
      );
      const savedFilter = params.get("filter");
      setFilter(
        savedFilter &&
          [
            "all",
            "passed",
            "failed",
            "errors",
            "improved",
            "regressed",
            "unchanged",
            "comparison_errors",
          ].includes(savedFilter)
          ? savedFilter
          : "all",
      );
      setView(
        restoredRun
          ? "detail"
          : params.get("view") === "history"
            ? "history"
            : params.get("view") === "new" ||
                !!params.get("prompt") ||
                !!params.get("dataset")
              ? "new"
              : "history",
      );
      setUrlReady(true);
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => {
    if (!urlReady) return;
    const params = new URLSearchParams(window.location.search);
    for (const [key, value] of Object.entries({
      prompt: promptId,
      version: versionId,
      dataset: datasetId,
      revision: revisionId,
      run: runId,
      compare: compareId,
      case: runId && caseIndex !== null ? String(caseIndex) : "",
      filter: runId && filter !== "all" ? filter : "",
    })) {
      if (value) params.set(key, value);
      else params.delete(key);
    }
    if (view === "new" || view === "history") params.set("view", view);
    else params.delete("view");
    const search = params.toString();
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${search ? `?${search}` : ""}`,
    );
  }, [
    urlReady,
    promptId,
    versionId,
    datasetId,
    revisionId,
    runId,
    compareId,
    caseIndex,
    filter,
    view,
  ]);
  return { promptId, setPromptId, promptOffset, setPromptOffset, versionOffset, setVersionOffset, versionId, setVersionId, datasetId, setDatasetId, datasetOffset, setDatasetOffset, revisionOffset, setRevisionOffset, revisionId, setRevisionId, provider, setProvider, model, setModel, apiKey, setApiKey, temperature, setTemperature, maxTokens, setMaxTokens, rules, setRules, judgeEnabled, setJudgeEnabled, judge, setJudge, busy, setBusy, error, setError, runId, setRunId, compareId, setCompareId, runOffset, setRunOffset, runSearch, setRunSearch, runStatus, setRunStatus, referenceSearch, setReferenceSearch, referenceOffset, setReferenceOffset, evaluators, setEvaluators, view, setView, filter, setFilter, caseIndex, setCaseIndex, urlReady, setUrlReady };
}
