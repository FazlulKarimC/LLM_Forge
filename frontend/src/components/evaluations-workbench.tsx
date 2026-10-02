"use client";
import { PageControls } from "@/components/ui/page-controls";
import Link from "next/link";
import { EvaluatorSelectionFields } from "@/components/evaluations/evaluator-selection";
import { ScoreSavedOutputs } from "@/components/evaluations/score-saved-outputs";
import type { EvaluatorSelection } from "@/lib/evaluator-api";
import {
  ResultsGrid,
  RunSummary,
} from "@/components/evaluations/evaluation-results";
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { PageHeader, StatusPill } from "@/components/ui/primitives";
import { listPrompts, getVersions, type Provider } from "@/lib/prompt-api";
import {
  listDatasets,
  listRevisions,
  startEvaluation,
  listEvaluations,
  getEvaluation,
  cancelEvaluation,
  isActive,
  exportJson,
  type Assertion,
  type Judge,
} from "@/lib/evaluation-api";
import {
  inputClass,
  buttonClass,
  errorText,
  panelClass,
} from "@/components/prompts/prompt-ui";

const defaults: Record<Provider, string> = {
  mock: "demo-model",
  groq: "openai/gpt-oss-20b",
  openrouter: "openrouter/free",
  openai: "gpt-4o-mini",
};
const ruleNames: Record<Assertion["kind"], string> = {
  exact_match: "Exact match to reference",
  contains: "Contains text",
  regex: "Regex search",
  json_valid: "Valid JSON",
  json_equals: "JSON equals",
  json_reference: "JSON matches each reference",
  json_path: "JSON path equals",
};

export function EvaluationsWorkbench() {
  const cache = useQueryClient();
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
  const prompts = useQuery({
    queryKey: ["prompt-library", "", false, promptOffset],
    queryFn: ({ signal }) => listPrompts("", false, promptOffset, signal),
  });
  const versions = useQuery({
    queryKey: ["eval-versions", promptId, versionOffset],
    queryFn: ({ signal }) => getVersions(promptId, versionOffset, signal),
    enabled: !!promptId,
  });
  const datasets = useQuery({
    queryKey: ["datasets", false, datasetOffset],
    queryFn: ({ signal }) => listDatasets(false, datasetOffset, signal),
  });
  const revisions = useQuery({
    queryKey: ["dataset-history", datasetId, revisionOffset],
    queryFn: ({ signal }) => listRevisions(datasetId, revisionOffset, signal),
    enabled: !!datasetId,
  });
  const runs = useQuery({
    queryKey: ["evaluations", runOffset, runSearch, runStatus],
    queryFn: ({ signal }) =>
      listEvaluations(runOffset, signal, {
        search: runSearch,
        ...(runStatus ? { status: runStatus } : {}),
      }),
    refetchInterval: (query) =>
      query.state.data?.items.some(isActive) ? 2500 : false,
  });
  const detail = useQuery({
    queryKey: ["evaluation", runId],
    queryFn: ({ signal }) => getEvaluation(runId, signal),
    enabled: !!runId,
    refetchInterval: (query) =>
      query.state.data && isActive(query.state.data.run) ? 2500 : false,
  });
  const comparison = useQuery({
    queryKey: ["evaluation", compareId],
    queryFn: ({ signal }) => getEvaluation(compareId, signal),
    enabled: !!compareId,
    refetchInterval: (query) =>
      query.state.data && isActive(query.state.data.run) ? 2500 : false,
  });
  const references = useQuery({
    queryKey: [
      "evaluation-references",
      detail.data?.run.dataset_revision_id,
      referenceSearch,
      referenceOffset,
    ],
    queryFn: ({ signal }) =>
      listEvaluations(referenceOffset, signal, {
        dataset_revision_id: detail.data!.run.dataset_revision_id,
        search: referenceSearch,
        status: "completed",
      }),
    enabled: !!detail.data,
  });
  const selectedVersion = versions.data?.find((item) => item.id === versionId);
  const selectedRevision = revisions.data?.find(
    (item) => item.id === revisionId,
  );
  const judgeCount =
    Number(judgeEnabled) +
    evaluators.filter((item) => item.kind === "llm_judge").length;
  const casesCount = selectedRevision?.cases.length;
  const generationCalls = provider === "mock" ? 0 : casesCount;
  const judgeCalls = casesCount == null ? undefined : casesCount * judgeCount;
  async function act(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  const updateRule = (index: number, changes: Partial<Assertion>) =>
    setRules(
      rules.map((rule, i) => (i === index ? { ...rule, ...changes } : rule)),
    );
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Test and compare"
        title="Evaluations"
        description="Run saved prompt versions against fixed dataset revisions, then inspect every case."
        actions={
          <>
            <Link className="btn-secondary" href="/evaluations/evaluators">
              Evaluators
            </Link>
            <button
              className="btn-secondary"
              onClick={() => {
                setView("history");
                setRunId("");
                setCompareId("");
              }}
            >
              Run history
            </button>
            <button
              className="btn-primary"
              onClick={() => {
                setView("new");
                setRunId("");
                setCompareId("");
              }}
            >
              New evaluation
            </button>
          </>
        }
      />
      {error && (
        <p role="alert" className="text-(--destructive)">
          {error}
        </p>
      )}
      {[
        prompts.error,
        datasets.error,
        versions.error,
        revisions.error,
        runs.error,
        detail.error,
        comparison.error,
      ]
        .filter(Boolean)
        .map((e, i) => (
          <p key={i} role="alert" className="text-(--destructive)">
            {errorText(e)}
          </p>
        ))}
      {view === "new" && (
        <section className="evaluation-setup">
          <div className={`${panelClass} evaluation-fields`}>
            <h2 className="font-semibold">Start an evaluation</h2>
            <div className="evaluation-fields">
              <div className="form-section">
                <h3>1. Prompt &amp; dataset</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm">
                      Prompt
                      <select
                        aria-label="Evaluation prompt"
                        className={inputClass}
                        value={promptId}
                        onChange={(e) => {
                          setPromptId(e.target.value);
                          setVersionId("");
                          setVersionOffset(0);
                        }}
                      >
                        <option value="">Select a prompt</option>
                        {prompts.data?.items.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                        {promptId &&
                          !prompts.data?.items.some(
                            (p) => p.id === promptId,
                          ) && (
                            <option value={promptId}>
                              Selected prompt ({promptId})
                            </option>
                          )}
                      </select>
                    </label>
                    <PageControls
                      offset={promptOffset}
                      next={
                        !!prompts.data && promptOffset + 50 < prompts.data.total
                      }
                      change={(offset) => {
                        setPromptOffset(offset);
                        setPromptId("");
                        setVersionId("");
                      }}
                    />
                    <label className="block text-sm">
                      Saved prompt version
                      <select
                        aria-label="Prompt version"
                        className={inputClass}
                        value={versionId}
                        onChange={(e) => setVersionId(e.target.value)}
                        disabled={!promptId || versions.isLoading}
                      >
                        <option value="">Select a version</option>
                        {versions.data?.map((v) => (
                          <option key={v.id} value={v.id}>
                            v{v.version} ·{" "}
                            {v.variables.join(", ") || "No variables"}
                          </option>
                        ))}
                        {versionId &&
                          !versions.data?.some((v) => v.id === versionId) && (
                            <option value={versionId}>
                              Selected saved version ({versionId})
                            </option>
                          )}
                      </select>
                    </label>
                    <PageControls
                      offset={versionOffset}
                      next={versions.data?.length === 50}
                      change={(offset) => {
                        setVersionOffset(offset);
                        setVersionId("");
                      }}
                    />
                  </div>
                  <div>
                    <label className="block text-sm">
                      Dataset
                      <select
                        aria-label="Evaluation dataset"
                        className={inputClass}
                        value={datasetId}
                        onChange={(e) => {
                          setDatasetId(e.target.value);
                          setRevisionId("");
                          setRevisionOffset(0);
                        }}
                      >
                        <option value="">Select a dataset</option>
                        {datasets.data?.items.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.name}
                          </option>
                        ))}
                        {datasetId &&
                          !datasets.data?.items.some(
                            (d) => d.id === datasetId,
                          ) && (
                            <option value={datasetId}>
                              Selected dataset ({datasetId})
                            </option>
                          )}
                      </select>
                    </label>
                    <PageControls
                      offset={datasetOffset}
                      next={
                        !!datasets.data &&
                        datasetOffset + 50 < datasets.data.total
                      }
                      change={(offset) => {
                        setDatasetOffset(offset);
                        setDatasetId("");
                        setRevisionId("");
                      }}
                    />
                    <label className="block text-sm">
                      Dataset revision
                      <select
                        aria-label="Dataset revision"
                        className={inputClass}
                        value={revisionId}
                        onChange={(e) => setRevisionId(e.target.value)}
                        disabled={!datasetId || revisions.isLoading}
                      >
                        <option value="">Select a revision</option>
                        {revisions.data?.map((r) => (
                          <option key={r.id} value={r.id}>
                            v{r.version} · {r.cases.length} cases
                          </option>
                        ))}
                        {revisionId &&
                          !revisions.data?.some((r) => r.id === revisionId) && (
                            <option value={revisionId}>
                              Selected revision ({revisionId})
                            </option>
                          )}
                      </select>
                    </label>
                    <PageControls
                      offset={revisionOffset}
                      next={revisions.data?.length === 50}
                      change={(offset) => {
                        setRevisionOffset(offset);
                        setRevisionId("");
                      }}
                    />
                  </div>
                </div>
              </div>
              <div className="form-section space-y-3">
                <h3>2. Generation</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <label className="text-sm">
                    Provider
                    <select
                      aria-label="Evaluation provider"
                      className={inputClass}
                      value={provider}
                      onChange={(e) => {
                        const p = e.target.value as Provider;
                        setProvider(p);
                        setModel(defaults[p]);
                        setApiKey("");
                      }}
                    >
                      <option value="mock">Demo (echo compiled prompt)</option>
                      <option value="groq">Groq</option>
                      <option value="openrouter">OpenRouter</option>
                      <option value="openai">OpenAI</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    Model
                    <input
                      aria-label="Evaluation model"
                      className={inputClass}
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      maxLength={255}
                    />
                  </label>
                  {provider !== "mock" && (
                    <label className="text-sm">
                      Provider API key
                      <input
                        aria-label="Evaluation API key"
                        type="password"
                        autoComplete="off"
                        className={inputClass}
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        maxLength={512}
                      />
                    </label>
                  )}
                </div>
                <div className="flex flex-wrap gap-4">
                  <label className="text-sm">
                    Temperature
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      className={inputClass}
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                    />
                  </label>
                  <label className="text-sm">
                    Max output tokens
                    <input
                      type="number"
                      min="1"
                      max="2048"
                      className={inputClass}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                    />
                  </label>
                </div>
              </div>
              <div className="form-section space-y-3">
                <h3>3. Checks &amp; judge</h3>
                <EvaluatorSelectionFields
                  value={evaluators}
                  onChange={setEvaluators}
                />
                <div className="space-y-3">
                  <h2 className="font-medium">Assertions</h2>
                  {rules.map((rule, index) => (
                    <div key={index} className="flex flex-wrap gap-3 items-end">
                      <label className="text-sm">
                        Rule {index + 1}
                        <select
                          className={inputClass}
                          value={rule.kind}
                          onChange={(e) =>
                            updateRule(index, {
                              kind: e.target.value as Assertion["kind"],
                            })
                          }
                        >
                          {Object.entries(ruleNames).map(([value, title]) => (
                            <option value={value} key={value}>
                              {title}
                            </option>
                          ))}
                        </select>
                      </label>
                      {![
                        "exact_match",
                        "json_valid",
                        "json_reference",
                      ].includes(rule.kind) && (
                        <label className="flex-1 text-sm">
                          {rule.kind.startsWith("json_")
                            ? "Expected JSON value"
                            : "Text / pattern"}
                          <input
                            className={inputClass}
                            value={rule.value}
                            onChange={(e) =>
                              updateRule(index, { value: e.target.value })
                            }
                            maxLength={2000}
                          />
                        </label>
                      )}
                      {rule.kind === "json_path" && (
                        <label className="text-sm">
                          Path (e.g. answer.label)
                          <input
                            className={inputClass}
                            value={rule.path}
                            onChange={(e) =>
                              updateRule(index, { path: e.target.value })
                            }
                            maxLength={200}
                          />
                        </label>
                      )}
                      <button
                        aria-label={`Remove rule ${index + 1}`}
                        className={buttonClass}
                        onClick={() =>
                          setRules(rules.filter((_, i) => i !== index))
                        }
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button
                    disabled={rules.length >= 10}
                    className={buttonClass}
                    onClick={() =>
                      setRules([
                        ...rules,
                        { kind: "contains", value: "", path: "" },
                      ])
                    }
                  >
                    Add assertion
                  </button>
                  <p className="text-xs text-(--muted-foreground)">
                    All checks must pass. Exact match includes whitespace. Regex
                    uses search. JSON comparisons ignore object key order; dot
                    paths access keys and numeric array indexes.
                  </p>
                </div>
                <label className="text-sm flex gap-2">
                  <input
                    type="checkbox"
                    checked={judgeEnabled}
                    onChange={(e) => {
                      setJudgeEnabled(e.target.checked);
                      setJudge({ ...judge, api_key: "" });
                    }}
                  />{" "}
                  Add LLM judge (one additional provider call per case)
                </label>
                {judgeEnabled && (
                  <div className="space-y-3 border border-(--border) rounded-xl p-4">
                    <p className="text-xs text-(--muted-foreground)">
                      Judge scores are model opinions from 0–1. They may vary
                      and can be influenced by candidate content. Review reasons
                      alongside deterministic checks.
                    </p>
                    <div className="grid md:grid-cols-3 gap-4">
                      <label>
                        Judge provider
                        <select
                          className={inputClass}
                          value={judge.provider}
                          onChange={(e) => {
                            const p = e.target.value as Judge["provider"];
                            setJudge({
                              ...judge,
                              provider: p,
                              model: defaults[p],
                              api_key: "",
                            });
                          }}
                        >
                          <option value="groq">Groq</option>
                          <option value="openrouter">OpenRouter</option>
                          <option value="openai">OpenAI</option>
                        </select>
                      </label>
                      <label>
                        Judge model
                        <input
                          className={inputClass}
                          value={judge.model}
                          onChange={(e) =>
                            setJudge({ ...judge, model: e.target.value })
                          }
                          maxLength={255}
                        />
                      </label>
                      <label>
                        Judge API key
                        <input
                          aria-label="Judge API key"
                          type="password"
                          autoComplete="off"
                          className={inputClass}
                          value={judge.api_key}
                          onChange={(e) =>
                            setJudge({ ...judge, api_key: e.target.value })
                          }
                          maxLength={512}
                        />
                      </label>
                    </div>
                    <label className="block">
                      Rubric
                      <textarea
                        className={inputClass}
                        value={judge.rubric}
                        onChange={(e) =>
                          setJudge({ ...judge, rubric: e.target.value })
                        }
                        maxLength={4000}
                      />
                    </label>
                    <label>
                      Passing score
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        className={inputClass}
                        value={judge.threshold}
                        onChange={(e) =>
                          setJudge({
                            ...judge,
                            threshold: Number(e.target.value),
                          })
                        }
                      />
                    </label>
                  </div>
                )}
                <p className="text-xs text-(--muted-foreground)">
                  Limits: 100 demo cases, 50 live cases, or 20 with judging; two
                  active runs per project. Keys stay in memory and are cleared
                  from this form after starting. Cancellation stops subsequent
                  cases; an in-flight call may finish.
                </p>
              </div>
            </div>
          </div>
          <aside
            className={`${panelClass} evaluation-summary`}
            aria-label="Evaluation configuration"
          >
            <div className="rounded-lg border border-(--border) p-4 text-sm">
              <h3 className="font-medium">Run configuration</h3>
              <button
                type="button"
                className="btn-secondary"
                disabled={!selectedVersion?.config}
                onClick={() => {
                  const config = selectedVersion?.config ?? {};
                  if (
                    typeof config.temperature === "number" &&
                    config.temperature >= 0 &&
                    config.temperature <= 2 &&
                    Number.isFinite(config.temperature)
                  )
                    setTemperature(config.temperature);
                  if (
                    typeof config.max_tokens === "number" &&
                    Number.isInteger(config.max_tokens) &&
                    config.max_tokens >= 1 &&
                    config.max_tokens <= 2048
                  )
                    setMaxTokens(config.max_tokens);
                }}
              >
                Use saved prompt settings
              </button>
              <p className="field-help">
                Effective generation settings: temperature {temperature},
                maximum {maxTokens} output tokens. Saved settings apply
                supported temperature/token values only; edit the generation
                fields to override them.
              </p>
              <p className="mt-1 text-(--muted-foreground)">
                Prompt{" "}
                {versions.data?.find((entry) => entry.id === versionId)?.version
                  ? `v${versions.data.find((entry) => entry.id === versionId)?.version}`
                  : "not selected"}{" "}
                · Dataset{" "}
                {revisions.data?.find((entry) => entry.id === revisionId)
                  ?.version
                  ? `v${revisions.data.find((entry) => entry.id === revisionId)?.version}`
                  : "not selected"}{" "}
                · {provider} / {model || "no model"} · {rules.length} checks
                {judgeEnabled ? " + judge" : ""}
                {evaluators.length
                  ? ` + ${evaluators.length} saved evaluators`
                  : ""}
              </p>
              <p className="text-sm">
                {casesCount == null
                  ? "Select a dataset revision to see call counts."
                  : `${casesCount} cases · up to ${generationCalls} generation + ${judgeCalls} judge calls`}
              </p>
              <p className="field-help">
                Sequential execution. At most two judges, 60 model calls and 20
                cases when judging; two active runs per project.
              </p>
            </div>
            <button
              className="btn-primary"
              disabled={
                busy ||
                !versionId ||
                !revisionId ||
                !model.trim() ||
                (!rules.length &&
                  !judgeEnabled &&
                  !evaluators.some((item) => item.required)) ||
                judgeCount > 2 ||
                (generationCalls ?? 0) + (judgeCalls ?? 0) > 60 ||
                (judgeCount > 0 && (casesCount ?? 0) > 20) ||
                evaluators.some(
                  (item) => item.kind === "llm_judge" && !item.api_key?.trim(),
                ) ||
                (provider !== "mock" && !apiKey.trim()) ||
                (judgeEnabled &&
                  (!judge.api_key.trim() ||
                    !judge.model.trim() ||
                    !judge.rubric.trim()))
              }
              onClick={() =>
                act(async () => {
                  const run = await startEvaluation({
                    prompt_version_id: versionId,
                    dataset_revision_id: revisionId,
                    provider,
                    model,
                    temperature,
                    max_tokens: maxTokens,
                    ...(provider !== "mock" ? { api_key: apiKey } : {}),
                    assertions: rules,
                    ...(evaluators.length ? { evaluators } : {}),
                    ...(judgeEnabled ? { judge } : {}),
                  });
                  setApiKey("");
                  setJudge({ ...judge, api_key: "" });
                  setEvaluators(
                    evaluators.map((item) => ({ ...item, api_key: "" })),
                  );
                  setRunOffset(0);
                  setRunId(run.id);
                  setView("detail");
                  setCompareId("");
                  setFilter("all");
                  await cache.invalidateQueries({ queryKey: ["evaluations"] });
                })
              }
            >
              {busy ? "Starting…" : "Start evaluation"}
            </button>
          </aside>
        </section>
      )}
      {view === "history" && (
        <section className={panelClass}>
          <h2 className="font-semibold">Run history</h2>
          <div className="table-toolbar">
            <label className="text-sm flex-1">
              Search prompt names
              <input
                className={inputClass}
                value={runSearch}
                onChange={(event) => {
                  setRunSearch(event.target.value);
                  setRunOffset(0);
                }}
              />
            </label>
            <label className="text-sm">
              Run status
              <select
                className={inputClass}
                value={runStatus}
                onChange={(event) => {
                  setRunStatus(event.target.value);
                  setRunOffset(0);
                }}
              >
                <option value="">All statuses</option>
                {["queued", "running", "completed", "failed", "cancelled"].map(
                  (status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ),
                )}
              </select>
            </label>
          </div>
          <p className="mt-1 text-sm text-(--muted-foreground)">
            Every row is a saved run. Choose one to inspect its outputs and
            checks.
          </p>
          {runs.isLoading && <p className="mt-4">Loading runs…</p>}
          {runs.data?.items.length === 0 && (
            <p className="mt-4 text-(--muted-foreground)">
              No runs yet. Create a prompt and dataset, or use the demo examples
              on Overview.
            </p>
          )}
          {!!runs.data?.items.length && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[860px] text-left text-sm">
                <thead className="border-b border-(--border) text-xs uppercase text-(--muted-foreground)">
                  <tr>
                    <th className="p-3">Created</th>
                    <th className="p-3">Prompt</th>
                    <th className="p-3">Dataset</th>
                    <th className="p-3">Source</th>
                    <th className="p-3">Provider / model</th>
                    <th className="p-3">Progress</th>
                    <th className="p-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-(--border)">
                  {runs.data.items.map((run) => (
                    <tr key={run.id} className="hover:bg-(--surface-2)">
                      <td className="p-3 text-(--muted-foreground)">
                        {new Date(run.created_at).toLocaleString()}
                      </td>
                      <td className="p-3">
                        <button
                          className="font-semibold text-(--primary) hover:underline"
                          onClick={() => {
                            setRunId(run.id);
                            setCompareId("");
                            setFilter("all");
                            setView("detail");
                          }}
                        >
                          {run.config.prompt_name} v{run.config.prompt_version}
                        </button>
                      </td>
                      <td className="p-3">
                        {run.config.dataset_name} v{run.config.dataset_version}
                      </td>
                      <td className="p-3">
                        {run.config.source === "score_only"
                          ? "Saved-output scoring"
                          : run.config.source === "sdk_submission"
                            ? "External"
                            : run.config.is_mock
                              ? "Demo"
                              : "Live"}
                      </td>
                      <td className="p-3">
                        {run.config.provider} / {run.config.model}
                      </td>
                      <td className="p-3 tabular-nums">
                        {run.completed}/{run.total} · {run.passed} passed ·{" "}
                        {run.errors} errors
                      </td>
                      <td className="p-3">
                        <StatusPill status={run.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <PageControls
            offset={runOffset}
            next={!!runs.data && runOffset + 50 < runs.data.total}
            change={setRunOffset}
          />
        </section>
      )}
      {view === "detail" && detail.isLoading && runId && (
        <p>Loading results…</p>
      )}
      {view === "detail" && detail.data && (
        <section className={panelClass}>
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="space-y-2">
              <p className="text-xs font-semibold text-(--muted-foreground)">
                {comparison.data ? "Candidate run" : "Selected run"}
              </p>
              <RunSummary detail={detail.data} />
            </div>
            {comparison.data && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-(--muted-foreground)">
                  Reference run
                </p>
                <RunSummary detail={comparison.data} />
              </div>
            )}
          </div>
          <div className="table-toolbar mt-4">
            <div className="min-w-48 flex-1 text-xs grid gap-2">
              <label>
                Find reference by prompt name
                <input
                  className={inputClass}
                  value={referenceSearch}
                  onChange={(event) => {
                    setReferenceSearch(event.target.value);
                    setReferenceOffset(0);
                  }}
                />
              </label>
              <label>
                Reference run for this candidate
                <select
                  className={inputClass}
                  value={compareId}
                  onChange={(event) => {
                    setCompareId(event.target.value);
                    setFilter("all");
                  }}
                >
                  <option value="">No comparison</option>
                  {references.data?.items
                    .filter((run) => run.id !== runId)
                    .map((run) => (
                      <option key={run.id} value={run.id}>
                        {run.config.prompt_name} v{run.config.prompt_version} ·{" "}
                        {new Date(run.created_at).toLocaleString()} ·{" "}
                        {run.config.source === "score_only"
                          ? "Scoring"
                          : run.config.source === "sdk_submission"
                            ? "External"
                            : run.config.is_mock
                              ? "Demo"
                              : "Live"}
                        {run.config.evaluators?.length
                          ? ` · ${run.config.evaluators.map((item) => `${item.name} v${item.version}`).join(", ")}`
                          : ""}
                      </option>
                    ))}
                  {compareId &&
                    !references.data?.items.some(
                      (run) => run.id === compareId,
                    ) && <option value={compareId}>Selected comparison</option>}
                </select>
              </label>
              <PageControls
                offset={referenceOffset}
                next={referenceOffset + 50 < (references.data?.total ?? 0)}
                change={setReferenceOffset}
              />
              {references.error && (
                <span role="alert">{errorText(references.error)}</span>
              )}
            </div>
            {isActive(detail.data.run) && (
              <button
                disabled={busy}
                className={buttonClass}
                onClick={() =>
                  act(async () => {
                    await cancelEvaluation(runId);
                    await cache.invalidateQueries({
                      queryKey: ["evaluation", runId],
                    });
                    await cache.invalidateQueries({
                      queryKey: ["evaluations"],
                    });
                  })
                }
              >
                Cancel run
              </button>
            )}
            <button
              className={buttonClass}
              onClick={() => exportJson("evaluation-results.json", detail.data)}
            >
              Export results JSON
            </button>
            <label className="text-sm">
              Cases
              <select
                className={inputClass}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                <option value="all">All</option>
                <option value="failed">Failed checks</option>
                <option value="errors">Errors</option>
                <option value="passed">Passed</option>
                {comparison.data?.run.dataset_revision_id ===
                  detail.data.run.dataset_revision_id && (
                  <>
                    <option value="improved">Improved</option>
                    <option value="regressed">Regressed</option>
                    <option value="unchanged">Unchanged</option>
                    <option value="comparison_errors">
                      Change involved error
                    </option>
                  </>
                )}
              </select>
            </label>
          </div>
          {comparison.data &&
            comparison.data.run.dataset_revision_id !==
              detail.data.run.dataset_revision_id && (
              <p role="status" className="text-(--warning) mt-4">
                These runs use different dataset revisions. Select the same
                revision for case-by-case comparison.
              </p>
            )}
          <ResultsGrid
            detail={detail.data}
            comparison={
              comparison.data?.run.dataset_revision_id ===
              detail.data.run.dataset_revision_id
                ? comparison.data
                : undefined
            }
            filter={filter}
            selectedCaseIndex={caseIndex}
            onSelectCase={setCaseIndex}
          />
          {detail.data.run.status === "completed" && (
            <ScoreSavedOutputs
              key={detail.data.run.id}
              run={detail.data.run}
              started={(run) => {
                setRunId(run.id);
                setCompareId("");
                setFilter("all");
                setCaseIndex(null);
                setReferenceOffset(0);
                cache.invalidateQueries({ queryKey: ["evaluations"] });
              }}
            />
          )}
        </section>
      )}
    </div>
  );
}
