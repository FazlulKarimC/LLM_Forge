"use client";
import { PageControls } from "@/components/ui/page-controls";
import {
  ResultsGrid,
  RunSummary,
} from "@/components/evaluations/evaluation-results";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
  const [filter, setFilter] = useState("all");
  const [urlReady, setUrlReady] = useState(false);
  useEffect(() => {
    const restore = () => {
      const params = new URLSearchParams(window.location.search);
      setPromptId(params.get("prompt") ?? "");
      setVersionId(params.get("version") ?? "");
      setDatasetId(params.get("dataset") ?? "");
      setRevisionId(params.get("revision") ?? "");
      setRunId(params.get("run") ?? "");
      setCompareId(params.get("compare") ?? "");
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
      prompt: promptId, version: versionId, dataset: datasetId,
      revision: revisionId, run: runId, compare: compareId,
    })) {
      if (value) params.set(key, value);
      else params.delete(key);
    }
    const search = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${search ? `?${search}` : ""}`);
  }, [urlReady, promptId, versionId, datasetId, revisionId, runId, compareId]);
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
    queryKey: ["evaluations", runOffset],
    queryFn: ({ signal }) => listEvaluations(runOffset, signal),
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
    <div className="space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl font-semibold">Evaluations</h1>
        <p className="text-sm text-(--muted-foreground)">
          Compare saved prompt versions against fixed test cases.{" "}
          <Link className="underline" href="/datasets">
            Manage datasets
          </Link>
        </p>
      </div>
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
      <details open className={panelClass}>
        <summary className="font-semibold cursor-pointer">
          Start an evaluation
        </summary>
        <div className="space-y-5 mt-4">
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
                  {promptId && !prompts.data?.items.some((p) => p.id === promptId) &&
                    <option value={promptId}>Selected prompt ({promptId})</option>}
                </select>
              </label>
              <PageControls
                offset={promptOffset}
                next={!!prompts.data && promptOffset + 50 < prompts.data.total}
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
                      v{v.version} · {v.variables.join(", ") || "No variables"}
                    </option>
                  ))}
                  {versionId && !versions.data?.some((v) => v.id === versionId) &&
                    <option value={versionId}>Selected saved version ({versionId})</option>}
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
                  {datasetId && !datasets.data?.items.some((d) => d.id === datasetId) &&
                    <option value={datasetId}>Selected dataset ({datasetId})</option>}
                </select>
              </label>
              <PageControls
                offset={datasetOffset}
                next={
                  !!datasets.data && datasetOffset + 50 < datasets.data.total
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
                  {revisionId && !revisions.data?.some((r) => r.id === revisionId) &&
                    <option value={revisionId}>Selected revision ({revisionId})</option>}
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
                {!["exact_match", "json_valid", "json_reference"].includes(rule.kind) && (
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
                  onClick={() => setRules(rules.filter((_, i) => i !== index))}
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              disabled={rules.length >= 10}
              className={buttonClass}
              onClick={() =>
                setRules([...rules, { kind: "contains", value: "", path: "" }])
              }
            >
              Add assertion
            </button>
            <p className="text-xs text-(--muted-foreground)">
              All checks must pass. Exact match includes whitespace. Regex uses
              search. JSON comparisons ignore object key order; dot paths access
              keys and numeric array indexes.
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
                Judge scores are model opinions from 0–1. They may vary and can
                be influenced by candidate content. Review reasons alongside
                deterministic checks.
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
                    setJudge({ ...judge, threshold: Number(e.target.value) })
                  }
                />
              </label>
            </div>
          )}
          <p className="text-xs text-(--muted-foreground)">
            Limits: 100 demo cases, 50 live cases, or 20 with judging; two
            active runs per project. Keys stay in memory and are cleared from
            this form after starting. Cancellation stops subsequent cases; an
            in-flight call may finish.
          </p>
          <button
            className="btn-primary"
            disabled={
              busy ||
              !versionId ||
              !revisionId ||
              !model.trim() ||
              (!rules.length && !judgeEnabled) ||
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
                  ...(judgeEnabled ? { judge } : {}),
                });
                setApiKey("");
                setJudge({ ...judge, api_key: "" });
                setRunOffset(0);
                setRunId(run.id);
                setCompareId("");
                setFilter("all");
                await cache.invalidateQueries({ queryKey: ["evaluations"] });
              })
            }
          >
            {busy ? "Starting…" : "Start evaluation"}
          </button>
        </div>
      </details>
      <section className={panelClass}>
        <h2 className="font-semibold">Run history</h2>
        {runs.isLoading && <p>Loading runs…</p>}
        {runs.data?.items.length === 0 && (
          <p className="text-(--muted-foreground) mt-3">
            No runs yet. Try the sample dataset with a {"{{query}}"} prompt.
          </p>
        )}
        <div className="grid md:grid-cols-2 gap-3 mt-4">
          <label className="text-sm">
            View run
            <select
              aria-label="View evaluation run"
              className={inputClass}
              value={runId}
              onChange={(e) => {
                setRunId(e.target.value);
                setFilter("all");
                if (e.target.value === compareId) setCompareId("");
              }}
            >
              <option value="">Select a run</option>
              {runs.data?.items.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.config.prompt_name} v{r.config.prompt_version} ·{" "}
                  {r.config.dataset_name} v{r.config.dataset_version} ·{" "}
                  {r.status} · {new Date(r.created_at).toLocaleString()}
                </option>
              ))}
              {runId && !runs.data?.items.some((r) => r.id === runId) &&
                <option value={runId}>{detail.data ? `${detail.data.run.config.prompt_name} · ${detail.data.run.status}` : `Selected run (${runId})`}</option>}
            </select>
          </label>
          <label className="text-sm">
            Compare with
            <select
              aria-label="Compare evaluation run"
              className={inputClass}
              value={compareId}
              onChange={(e) => setCompareId(e.target.value)}
            >
              <option value="">No comparison</option>
              {runs.data?.items
                .filter((r) => r.id !== runId)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.config.prompt_name} v{r.config.prompt_version} ·{" "}
                    {r.config.model} · {new Date(r.created_at).toLocaleString()}
                  </option>
                ))}
              {compareId && !runs.data?.items.some((r) => r.id === compareId) &&
                <option value={compareId}>{comparison.data ? `${comparison.data.run.config.prompt_name} · ${comparison.data.run.status}` : `Selected comparison (${compareId})`}</option>}
            </select>
          </label>
        </div>
        <PageControls
          offset={runOffset}
          next={!!runs.data && runOffset + 50 < runs.data.total}
          change={setRunOffset}
        />
      </section>
      {detail.isLoading && runId && <p>Loading results…</p>}
      {detail.data && (
        <section className={panelClass}>
          <RunSummary detail={detail.data} />
          <div className="flex flex-wrap gap-3 mt-3">
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
          />
        </section>
      )}
    </div>
  );
}
