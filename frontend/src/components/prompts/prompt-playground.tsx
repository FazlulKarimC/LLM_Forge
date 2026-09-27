"use client";
import { useEffect, useRef, useState } from "react";
import { FlaskConical, Plus, X } from "lucide-react";
import { draftVariables, compileDraft, runPlayground, type Provider, type PlaygroundResult, type VersionDraft } from "@/lib/prompt-api";
import { ErrorMessage, errorText, inputClass, panelClass } from "./prompt-ui";

type Target = { id: number; provider: Provider; model: string; key: string };
type Outcome = { target: Target; result?: PlaygroundResult; error?: string };
const defaultModels: Record<Provider, string> = { mock: "demo-model", groq: "openai/gpt-oss-20b", openrouter: "openrouter/free", openai: "gpt-4o-mini" };

export function PromptPlayground({ draft }: { draft: VersionDraft }) {
  const names = draftVariables(draft.template_text, draft.template_format);
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<Target[]>([{ id: 1, provider: "mock", model: "demo-model", key: "" }]);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const [compiled, setCompiled] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(2);
  useEffect(() => () => controller.current?.abort(), []);

  function changeTarget(id: number, changes: Partial<Target>) { setTargets((current) => current.map((target) => target.id === id ? { ...target, ...changes } : target)); }
  async function run(preview: boolean) {
    if (pending) return;
    setPending(true); setError(null);
    const abort = new AbortController(); controller.current = abort;
    const inputs = Object.fromEntries(names.map((name) => [name, variables[name] || ""]));
    try {
      const response = await compileDraft(draft, inputs);
      if (abort.signal.aborted) return;
      setCompiled(response.compiled_prompt);
      if (!preview) {
        setOutcomes([]);
        const results = await Promise.all(targets.map(async (target): Promise<Outcome> => {
          try { return { target: { ...target, key: "" }, result: await runPlayground(draft, inputs, target.provider, target.model, target.key, temperature, maxTokens, abort.signal) }; }
          catch (err) { return { target: { ...target, key: "" }, error: errorText(err) }; }
        }));
        if (!abort.signal.aborted) setOutcomes(results);
      }
    } catch (err) { if (!abort.signal.aborted) setError(errorText(err)); }
    finally { if (!abort.signal.aborted) setPending(false); }
  }

  return <section className={`${panelClass} space-y-5`} aria-label="Prompt playground">
    <div><h2 className="flex items-center gap-2 text-xl font-semibold"><FlaskConical className="size-5 text-(--primary)" />Playground</h2><p className="mt-2 text-sm text-(--muted-foreground)">Test the editor draft before saving. Demo mode makes no model call. Provider keys stay in this page&apos;s memory and are used for the current request only.</p></div>
    {names.length ? <div className="grid gap-3 sm:grid-cols-2">{names.map((name) => <label key={name} className="text-sm">{name}<textarea className={inputClass} rows={2} maxLength={10_000} value={variables[name] || ""} disabled={pending} onChange={(event) => setVariables({ ...variables, [name]: event.target.value })} placeholder={`Sample value for ${name}`} /></label>)}</div> : <p className="text-sm text-(--muted-foreground)">Add a variable such as <code>{"{{query}}"}</code> to make this template reusable.</p>}
    <div className="space-y-3">{targets.map((target, index) => <div key={target.id} className="rounded-xl border border-(--border) bg-(--surface-2) p-4">
      <div className="flex items-center justify-between"><h3 className="text-sm font-medium">Model {index + 1}</h3>{targets.length > 1 ? <button aria-label={`Remove model ${index + 1}`} disabled={pending} onClick={() => setTargets(targets.filter((value) => value.id !== target.id))}><X className="size-4" /></button> : null}</div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2"><label className="text-sm">Provider<select aria-label={`Provider ${index + 1}`} className={inputClass} value={target.provider} disabled={pending} onChange={(event) => { const provider = event.target.value as Provider; changeTarget(target.id, { provider, model: defaultModels[provider], key: "" }); }}><option value="mock">Demo (no API key)</option><option value="groq">Groq</option><option value="openrouter">OpenRouter</option><option value="openai">OpenAI</option></select></label><label className="text-sm">Model ID<input aria-label={`Model ID ${index + 1}`} className={inputClass} value={target.model} maxLength={255} disabled={pending} onChange={(event) => changeTarget(target.id, { model: event.target.value })} /></label></div>
      {target.provider !== "mock" ? <label className="mt-3 block text-sm">Provider API key<input aria-label={`Provider API key ${index + 1}`} type="password" autoComplete="off" className={inputClass} value={target.key} maxLength={512} disabled={pending} onChange={(event) => changeTarget(target.id, { key: event.target.value })} placeholder="Used for this request; never saved" /></label> : null}
    </div>)}</div>
    {targets.length < 3 ? <button className="btn-secondary" disabled={pending} onClick={() => setTargets([...targets, { id: nextId.current++, provider: "mock", model: "demo-model", key: "" }])}><Plus className="size-4" />Compare another model</button> : null}
    <div className="grid gap-3 sm:grid-cols-2"><label className="text-sm">Temperature<input className={inputClass} type="number" min={0} max={2} step={0.1} value={temperature} disabled={pending} onChange={(event) => setTemperature(Number(event.target.value))} /></label><label className="text-sm">Maximum output tokens<input className={inputClass} type="number" min={1} max={2048} value={maxTokens} disabled={pending} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label></div>
    <div className="flex flex-wrap gap-2"><button className="btn-secondary" disabled={pending || !draft.template_text.trim()} onClick={() => run(true)}>Preview compiled prompt</button><button className="btn-primary" disabled={pending || !draft.template_text.trim() || targets.some((target) => !target.model.trim() || (target.provider !== "mock" && !target.key.trim()))} onClick={() => run(false)}>{pending ? "Running…" : targets.length > 1 ? "Run comparison" : "Run prompt"}</button></div>
    <ErrorMessage message={error} />
    {compiled !== null ? <details open className="rounded-xl border border-(--border) p-4"><summary className="cursor-pointer text-sm font-medium">Compiled prompt from the last preview or run</summary><pre className="mt-3 max-h-60 overflow-auto whitespace-pre-wrap break-words text-sm text-(--muted-foreground)">{compiled}</pre></details> : null}
    {outcomes.length ? <div className="grid gap-3 xl:grid-cols-2">{outcomes.map((outcome) => <div key={outcome.target.id} className="rounded-xl border border-(--border) p-4"><h3 className="break-all text-sm font-semibold">{outcome.target.provider} · {outcome.target.model}</h3>{outcome.error ? <div className="mt-3"><ErrorMessage message={outcome.error} /></div> : outcome.result ? <><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-sm">{outcome.result.output || "[Empty response]"}</pre><p className="mt-4 text-xs text-(--muted-foreground)">{outcome.result.is_mock ? "Demo output" : `${outcome.result.tokens_input ?? "Unknown"} input / ${outcome.result.tokens_output ?? "Unknown"} output tokens`} · {outcome.result.latency_ms.toFixed(0)} ms · {outcome.result.finish_reason}</p></> : null}</div>)}</div> : null}
  </section>;
}
