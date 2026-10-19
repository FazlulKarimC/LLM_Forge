"use client";
import { benchmarkPollDelay } from "@/lib/benchmark-polling";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { scopedKey } from "@/features/shared/query-keys";
import { PageHeader, StatusPill } from "@/components/ui/primitives";
import { PageControls } from "@/components/ui/page-controls";
import { SdkEvaluationGuide } from "@/components/sdk-evaluation-guide";
import { RunSummary } from "./evaluation-results";
import { PagedResults } from "./paged-results";
import { ResumeEvaluation } from "./resume-evaluation";
import { ScoreSavedOutputs } from "./score-saved-outputs";
import { useEvaluationSummary } from "../hooks/use-evaluation-summary";
import { listEvaluations, isActive, exportJson, getEvaluation, cancelEvaluation } from "@/lib/evaluation-api";
import { errorText } from "@/components/prompts/prompt-ui";

export function EvaluationsWorkbench() {
  const [ready, setReady] = useState(false);
  const [runId, setRunId] = useState("");
  const [compareId, setCompareId] = useState("");
  const [caseIndex, setCaseIndex] = useState<number | null>(null);
  const [filter, setFilter] = useState("all");
  const [sdk, setSdk] = useState(false);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const [referenceOffset, setReferenceOffset] = useState(0);
  const [referenceSearch, setReferenceSearch] = useState("");
  const [actionError, setActionError] = useState("");
  useEffect(() => {
    const restore = () => {
      const params = new URLSearchParams(window.location.search);
      setRunId(params.get("run") || "");
      setCompareId(params.get("compare") || "");
      const index = params.get("case");
      setCaseIndex(index != null && /^\d+$/.test(index) ? Number(index) : null);
      const savedFilter = params.get("filter") || "all";
      setFilter(["all","passed","failed","errors","improved","regressed","unchanged","comparison_errors"].includes(savedFilter) ? savedFilter : "all");
      setSdk(["new","sdk"].includes(params.get("view") || "") || params.has("prompt"));
      setReady(true);
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => {
    if (!ready) return;
    const params = new URLSearchParams();
    if (runId) params.set("run", runId);
    if (compareId && runId) params.set("compare", compareId);
    if (caseIndex != null && runId) params.set("case", String(caseIndex));
    if (filter !== "all" && runId) params.set("filter", filter);
    if (sdk && !runId) params.set("view", "sdk");
    window.history.replaceState(null, "", "/evaluations" + (params.size ? "?" + params.toString() : ""));
  }, [ready, runId, compareId, caseIndex, filter, sdk]);
  const runs = useQuery({
    queryKey: scopedKey(["evaluations", offset, search, status]),
    queryFn: ({signal}) => listEvaluations(offset, signal, {search, ...(status ? {status} : {})}),
    refetchInterval: query => query.state.data?.items.some(isActive) ? benchmarkPollDelay(query.state.data.items.find(isActive)?.created_at) : false,
  });
  const detail = useEvaluationSummary(runId);
  const comparison = useEvaluationSummary(compareId);
  const references = useQuery({
    queryKey: scopedKey(["evaluation-references", runId, referenceSearch, referenceOffset]),
    queryFn: ({signal}) => listEvaluations(referenceOffset, signal, {reference_run_id: runId, search: referenceSearch, status: "completed"}),
    enabled: !!detail.data,
  });
  const local = detail.data?.run.config.source === "sdk_local";
  const comparable = !comparison.data || !detail.data ? true : local
    ? comparison.data.run.config.source === "sdk_local"
      && detail.data.run.config.dataset_fingerprint === comparison.data.run.config.dataset_fingerprint
      && detail.data.run.config.scoring_fingerprint === comparison.data.run.config.scoring_fingerprint
    : comparison.data.run.config.source !== "sdk_local" && detail.data.run.dataset_revision_id === comparison.data.run.dataset_revision_id;
  async function cancel() {
    try {
      setActionError("");
      await cancelEvaluation(runId);
      await Promise.all([detail.refetch(), runs.refetch()]);
    } catch (error) { setActionError(errorText(error)); }
  }
  async function download() {
    try {
      setActionError("");
      exportJson("evaluation.json", local ? detail.data : await getEvaluation(runId));
    } catch (error) { setActionError(errorText(error)); }
  }
  return <div className="page-stack">
    <PageHeader eyebrow="Test and compare" title="Evaluations"
      description="Track SDK evaluations of saved prompts. Test data and model execution stay in your application."
      actions={<><Link className="btn-secondary" href="/evaluations/evaluators">Evaluators</Link>
        <button className="btn-secondary" onClick={() => {setRunId(""); setCompareId(""); setSdk(false);}}>Run history</button>
        <button className="btn-primary" onClick={() => {setRunId(""); setCompareId(""); setSdk(true);}}>Evaluate with SDK</button></>} />
    {[runs.error, detail.error, comparison.error, references.error].filter(Boolean).map((error, index) => <p role="alert" key={index}>{errorText(error)}</p>)}
    {actionError && <p role="alert">{actionError}</p>}
    {!runId && sdk && <SdkEvaluationGuide />}
    {!runId && !sdk && <section className="panel p-4 space-y-3">
      <div className="table-toolbar">
        <label className="field-label flex-1">Search prompts<input className="input-shell" value={search} onChange={e => {setSearch(e.target.value); setOffset(0);}} /></label>
        <label className="field-label">Status<select className="select-shell" value={status} onChange={e => {setStatus(e.target.value); setOffset(0);}}><option value="">All statuses</option>{["queued","running","completed","failed","cancelled"].map(value => <option key={value}>{value}</option>)}</select></label>
      </div>
      {runs.isLoading ? <p role="status">Loading evaluations…</p> : !runs.error && !runs.data?.items.length ? <div className="collection-empty">
        <h2>{search || status ? "No runs match your filters" : "No evaluations yet"}</h2>
        {search || status ? <button className="btn-secondary" onClick={() => {setSearch(""); setStatus(""); setOffset(0);}}>Clear filters</button> : <><p className="field-help">Run your local cases with the SDK and upload a summary.</p><button className="btn-primary" onClick={() => setSdk(true)}>Evaluate with SDK</button></>}
      </div> : <div className="case-table responsive-table" role="region" aria-label="Evaluation history" tabIndex={0}><table className="data-table">
        <thead><tr><th>Prompt</th><th>Test source</th><th>Status</th><th>Passed</th><th>Processed</th></tr></thead>
        <tbody>{runs.data?.items.map(run => <tr key={run.id}>
          <td data-label="Prompt"><button className="text-(--accent)" onClick={() => {setRunId(run.id); setCompareId(""); setCaseIndex(null); setFilter("all"); setReferenceOffset(0);}}>{run.config.prompt_name} v{run.config.prompt_version}</button></td>
          <td data-label="Test source">{run.config.dataset_name}{run.config.source === "sdk_local" ? " · Local" : " · Legacy hosted"}</td>
          <td data-label="Status"><StatusPill status={run.status} /></td>
          <td data-label="Passed" className="numeric">{run.passed}/{run.total}</td>
          <td data-label="Processed" className="numeric">{run.completed}/{run.total}</td>
        </tr>)}</tbody></table></div>}
      <p className="field-help">{runs.data?.total ?? 0} runs</p>
      <PageControls offset={offset} next={offset + 50 < (runs.data?.total ?? 0)} change={setOffset} />
    </section>}
    {runId && detail.isLoading && <p role="status">Loading evaluation…</p>}
    {runId && detail.data && <>
      <section className="panel p-4 space-y-4">
        <RunSummary detail={detail.data} />
        <button className="btn-secondary" onClick={download}>Export report</button>
        <label className="field-label">Reference search<input className="input-shell" value={referenceSearch} onChange={e => {setReferenceSearch(e.target.value); setReferenceOffset(0);}} /></label>
        <label className="field-label">Reference run<select className="select-shell" value={compareId} onChange={e => {setCompareId(e.target.value); setFilter("all");}}>
          <option value="">No reference</option>
          {compareId && !references.data?.items.some(run => run.id === compareId) && <option value={compareId}>Selected reference</option>}
          {references.data?.items.filter(run => run.id !== runId).map(run => <option key={run.id} value={run.id}>{run.config.prompt_name} v{run.config.prompt_version} · {run.id.slice(0,8)}</option>)}
        </select></label>
        <PageControls offset={referenceOffset} next={referenceOffset + 50 < (references.data?.total ?? 0)} change={setReferenceOffset} />
        {!comparable && <p role="alert">These runs use different test data or scoring definitions and cannot be compared.</p>}
        {comparison.data && comparable && <div className="border-t border-(--border) pt-4"><h3 className="font-semibold">Reference</h3><RunSummary detail={comparison.data} /></div>}
      </section>
      {local ? <section className="panel p-4 space-y-2">
        <h2 className="font-semibold">Summary-only report</h2>
        <p className="field-help">Inputs, reference answers, outputs and per-case checks remain in your local evaluation.json. This dashboard cannot identify individual regressions or re-score missing outputs. Re-run your application with the SDK.</p>
        <p className="field-help">Dataset fingerprint</p><code className="block break-all text-xs">{detail.data.run.config.dataset_fingerprint}</code>
        <p className="field-help">Scoring fingerprint</p><code className="block break-all text-xs">{detail.data.run.config.scoring_fingerprint}</code>
        <p className="field-help">Scores and fingerprints are supplied by your SDK; matching fingerprints do not independently verify execution.</p>
      </section> : <>
        <p className="field-help">Legacy hosted evaluation. Saved history remains readable; new tests should run locally with the SDK.</p>
        {isActive(detail.data.run) && <button className="btn-secondary" onClick={cancel}>Cancel run</button>}
        <ResumeEvaluation summary={detail.data} resumed={() => {void detail.refetch(); void runs.refetch();}} />
        <ScoreSavedOutputs run={detail.data.run} started={run => {setRunId(run.id); setCompareId("");}} />
        <label className="field-label">Case filter<select className="select-shell" value={filter} onChange={e => setFilter(e.target.value)}>{["all","passed","failed","errors",...(compareId && comparable ? ["improved","regressed","unchanged","comparison_errors"] : [])].map(value => <option key={value}>{value}</option>)}</select></label>
        <PagedResults detail={detail.data} comparison={comparable ? comparison.data : undefined} filter={filter} caseIndex={caseIndex} selectCase={setCaseIndex} />
      </>}
    </>}
  </div>;
}
