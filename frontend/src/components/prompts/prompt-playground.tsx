"use client";
import { useEffect, useRef, useState } from "react";
import { FlaskConical, Plus, X } from "lucide-react";
import {
  promptDraftVariables,
  compileDraft,
  runPlayground,
  type Provider,
  type PlaygroundResult,
  type VersionDraft,
} from "@/lib/prompt-api";
import { SectionHeading } from "@/components/ui/primitives";
import { ErrorMessage, errorText, inputClass, panelClass } from "./prompt-ui";
type Target = { id: number; provider: Provider; model: string; key: string };
type Outcome = { target: Target; result?: PlaygroundResult; error?: string };
const defaultModels: Record<Provider, string> = {
  mock: "demo-model",
  groq: "openai/gpt-oss-20b",
  openrouter: "openrouter/free",
  openai: "gpt-4o-mini",
};
export function PromptPlayground({
  draft,
  onCompareChange,
}: {
  draft: VersionDraft;
  onCompareChange?: (value: boolean) => void;
}) {
  const names = promptDraftVariables(draft);
  const contentReady =
    draft.prompt_type === "chat"
      ? (draft.messages ?? []).some((message) => message.content.trim())
      : !!draft.template_text.trim();
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<Target[]>([
    { id: 1, provider: "mock", model: "demo-model", key: "" },
  ]);
  const [temperature, setTemperature] = useState(
    typeof draft.config?.temperature === "number" &&
      draft.config.temperature >= 0 &&
      draft.config.temperature <= 2
      ? draft.config.temperature
      : 0.7,
  );
  const [maxTokens, setMaxTokens] = useState(
    typeof draft.config?.max_tokens === "number" &&
      Number.isInteger(draft.config.max_tokens) &&
      draft.config.max_tokens > 0 &&
      draft.config.max_tokens <= 2048
      ? draft.config.max_tokens
      : 256,
  );
  const [compiled, setCompiled] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(2);
  useEffect(() => () => controller.current?.abort(), []);
  function changeTarget(id: number, changes: Partial<Target>) {
    setTargets((current) =>
      current.map((target) =>
        target.id === id ? { ...target, ...changes } : target,
      ),
    );
  }
  async function run(preview: boolean) {
    if (pending) return;
    setPending(true);
    setError(null);
    const abort = new AbortController();
    controller.current = abort;
    const inputs = Object.fromEntries(
      names.map((name) => [name, variables[name] || ""]),
    );
    try {
      const response = await compileDraft(draft, inputs);
      if (abort.signal.aborted) return;
      setCompiled(response.compiled_prompt);
      if (!preview) {
        setOutcomes([]);
        const results = await Promise.all(
          targets.map(async (target): Promise<Outcome> => {
            try {
              return {
                target: { ...target, key: "" },
                result: await runPlayground(
                  draft,
                  inputs,
                  target.provider,
                  target.model,
                  target.key,
                  temperature,
                  maxTokens,
                  abort.signal,
                ),
              };
            } catch (err) {
              return { target: { ...target, key: "" }, error: errorText(err) };
            }
          }),
        );
        if (!abort.signal.aborted) setOutcomes(results);
      }
    } catch (err) {
      if (!abort.signal.aborted) setError(errorText(err));
    } finally {
      if (!abort.signal.aborted) setPending(false);
    }
  }
  return (
    <section
      className={`${panelClass} space-y-4`}
      aria-label="Prompt playground"
    >
      <SectionHeading
        title="Playground"
        description="Run the editor draft. Demo mode uses no model credits."
        actions={
          <button
            className="btn-primary"
            disabled={
              pending ||
              !contentReady ||
              targets.some(
                (target) =>
                  !target.model.trim() ||
                  (target.provider !== "mock" && !target.key.trim()),
              )
            }
            onClick={() => run(false)}
          >
            <FlaskConical className="size-3.5" />
            {pending
              ? "Running…"
              : targets.length > 1
                ? "Run comparison"
                : "Run prompt"}
          </button>
        }
      />
      {names.length ? (
        <details open>
          <summary className="cursor-pointer text-xs font-medium">
            Test inputs · {names.length} variable{names.length === 1 ? "" : "s"}
          </summary>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {names.map((name) => (
              <label key={name} className="text-xs">
                {name}
                <textarea
                  className={inputClass}
                  rows={2}
                  maxLength={10_000}
                  value={variables[name] || ""}
                  disabled={pending}
                  onChange={(event) =>
                    setVariables({ ...variables, [name]: event.target.value })
                  }
                  placeholder={`Sample value for ${name}`}
                />
              </label>
            ))}
          </div>
        </details>
      ) : (
        <p className="text-xs text-(--muted-foreground)">
          Add a variable such as <code>{"{{query}}"}</code> to test different
          inputs.
        </p>
      )}
      <div className="playground-models" data-count={targets.length}>
        {targets.map((target, index) => {
          const outcome = outcomes.find(
            (value) => value.target.id === target.id,
          );
          return (
            <div
              key={target.id}
              className="min-w-0 rounded-lg border border-(--border) overflow-hidden"
            >
              <div className="space-y-3 bg-(--surface-2) p-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold">Model {index + 1}</h3>
                  {targets.length > 1 ? (
                    <button
                      className="btn-ghost px-1!"
                      aria-label={`Remove model ${index + 1}`}
                      disabled={pending}
                      onClick={() => {
                        const remaining = targets.filter(
                          (value) => value.id !== target.id,
                        );
                        setTargets(remaining);
                        onCompareChange?.(remaining.length > 1);
                      }}
                    >
                      <X className="size-3.5" />
                    </button>
                  ) : null}
                </div>
                <label className="block text-xs">
                  Provider
                  <select
                    aria-label={`Provider ${index + 1}`}
                    className={inputClass}
                    value={target.provider}
                    disabled={pending}
                    onChange={(event) => {
                      const provider = event.target.value as Provider;
                      changeTarget(target.id, {
                        provider,
                        model: defaultModels[provider],
                        key: "",
                      });
                    }}
                  >
                    <option value="mock">Demo (no API key)</option>
                    <option value="groq">Groq</option>
                    <option value="openrouter">OpenRouter</option>
                    <option value="openai">OpenAI</option>
                  </select>
                </label>
                <label className="block text-xs">
                  Model ID
                  <input
                    aria-label={`Model ID ${index + 1}`}
                    className={inputClass}
                    value={target.model}
                    maxLength={255}
                    disabled={pending}
                    onChange={(event) =>
                      changeTarget(target.id, { model: event.target.value })
                    }
                  />
                </label>
                {target.provider !== "mock" ? (
                  <label className="block text-xs">
                    Provider API key
                    <input
                      aria-label={`Provider API key ${index + 1}`}
                      type="password"
                      autoComplete="off"
                      className={inputClass}
                      value={target.key}
                      maxLength={512}
                      disabled={pending}
                      onChange={(event) =>
                        changeTarget(target.id, { key: event.target.value })
                      }
                      placeholder="Request only; never saved"
                    />
                  </label>
                ) : null}
              </div>
              <div
                className="space-y-2 border-t border-(--border) p-3"
                aria-live="polite"
              >
                <p className="text-xs text-(--muted-foreground)">
                  {outcome ? "Last run output" : "Output"}
                  {outcome && ` · ${outcome.target.model}`}
                </p>
                {outcome?.error ? (
                  <ErrorMessage message={outcome.error} />
                ) : (
                  <pre className="playground-output">
                    {outcome?.result
                      ? outcome.result.output || "[Empty response]"
                      : pending
                        ? "Generating…"
                        : "Run this draft to inspect the response here."}
                  </pre>
                )}
                {outcome?.result && (
                  <p className="border-t border-(--border) pt-2 text-xs text-(--muted-foreground)">
                    {outcome.result.is_mock
                      ? "Demo output"
                      : `${outcome.result.tokens_input ?? "Unknown"} input / ${outcome.result.tokens_output ?? "Unknown"} output tokens`}{" "}
                    · {outcome.result.latency_ms.toFixed(0)} ms ·{" "}
                    {outcome.result.finish_reason}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {targets.length < 3 && (
          <button
            className="btn-secondary"
            disabled={pending}
            onClick={() => {
              setTargets([
                ...targets,
                {
                  id: nextId.current++,
                  provider: "mock",
                  model: "demo-model",
                  key: "",
                },
              ]);
              onCompareChange?.(true);
            }}
          >
            <Plus className="size-3.5" />
            Compare another model
          </button>
        )}
        <button
          className="btn-ghost"
          disabled={pending || !contentReady}
          onClick={() => run(true)}
        >
          Preview compiled prompt
        </button>
      </div>
      <details className="border-t border-(--border) pt-3">
        <summary className="cursor-pointer text-xs font-medium">
          Generation settings
        </summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="text-xs">
            Temperature
            <input
              className={inputClass}
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              disabled={pending}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
          </label>
          <label className="text-xs">
            Maximum output tokens
            <input
              className={inputClass}
              type="number"
              min={1}
              max={2048}
              value={maxTokens}
              disabled={pending}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
            />
          </label>
        </div>
        <p className="mt-2 text-xs text-(--muted-foreground)">
          Settings apply to all models. Provider keys stay in memory for this
          page only.
        </p>
      </details>
      <ErrorMessage message={error} />
      {compiled !== null && (
        <details className="rounded-lg border border-(--border) p-3">
          <summary className="cursor-pointer text-xs font-medium">
            Compiled prompt from the last preview or run
          </summary>
          <pre className="mt-3 max-h-60 overflow-auto whitespace-pre-wrap break-words text-xs text-(--muted-foreground)">
            {compiled}
          </pre>
        </details>
      )}
    </section>
  );
}
