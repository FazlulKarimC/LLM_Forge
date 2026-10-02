"use client";

import Link from "next/link";
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/ui/primitives";
import { PageControls } from "@/components/ui/page-controls";
import {
  inputClass,
  panelClass,
  errorText,
} from "@/components/prompts/prompt-ui";
import {
  listEvaluators,
  getEvaluator,
  listEvaluatorVersions,
  createEvaluator,
  saveEvaluatorVersion,
  archiveEvaluator,
  testEvaluator,
  defaultDefinition,
  defaultScore,
  type EvaluatorDefinition,
  type Evaluator,
  type ScoreDefinition,
} from "@/lib/evaluator-api";
import { canonicalJSON } from "@/lib/prompt-api";
import { useUnsavedChanges } from "@/lib/use-unsaved-changes";
import type { Assertion, CaseResult } from "@/lib/evaluation-api";

const checkKinds: Assertion["kind"][] = [
  "exact_match",
  "contains",
  "regex",
  "json_valid",
  "json_equals",
  "json_reference",
  "json_path",
];

function DefinitionFields({
  value,
  change,
}: {
  value: EvaluatorDefinition;
  change: (value: EvaluatorDefinition) => void;
}) {
  const scoreChange = (index: number, fields: Partial<ScoreDefinition>) =>
    change({
      ...value,
      outputs: value.outputs.map((score, position) =>
        position === index ? { ...score, ...fields } : score,
      ),
    });
  return (
    <div className="grid gap-4">
      <label className="text-sm">
        Evaluation method
        <select
          className={inputClass}
          value={value.kind}
          onChange={(event) =>
            change({
              ...defaultDefinition(),
              kind: event.target.value as EvaluatorDefinition["kind"],
              ...(event.target.value === "llm_judge"
                ? {
                    assertion: null,
                    model: "openai/gpt-oss-20b",
                    rubric: "Does the candidate correctly answer the input?",
                    outputs: [{ ...defaultScore(), data_type: "numeric" }],
                  }
                : {}),
            })
          }
        >
          <option value="builtin">Built-in check</option>
          <option value="llm_judge">LLM judge</option>
        </select>
      </label>
      {value.kind === "builtin" && value.assertion ? (
        <div className="grid gap-3">
          <label className="text-sm">
            Check
            <select
              className={inputClass}
              value={value.assertion.kind}
              onChange={(event) =>
                change({
                  ...value,
                  assertion: {
                    ...value.assertion!,
                    kind: event.target.value as Assertion["kind"],
                  },
                })
              }
            >
              {checkKinds.map((kind) => (
                <option key={kind} value={kind}>
                  {kind.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
          {["contains", "regex", "json_equals", "json_path"].includes(
            value.assertion.kind,
          ) && (
            <label className="text-sm">
              Check value
              <input
                className={inputClass}
                value={value.assertion.value}
                onChange={(event) =>
                  change({
                    ...value,
                    assertion: {
                      ...value.assertion!,
                      value: event.target.value,
                    },
                  })
                }
              />
            </label>
          )}
          {value.assertion.kind === "json_path" && (
            <label className="text-sm">
              JSON path
              <input
                className={inputClass}
                value={value.assertion.path}
                onChange={(event) =>
                  change({
                    ...value,
                    assertion: {
                      ...value.assertion!,
                      path: event.target.value,
                    },
                  })
                }
              />
            </label>
          )}
        </div>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="text-sm">
              Judge provider
              <select
                className={inputClass}
                value={value.provider}
                onChange={(event) =>
                  change({
                    ...value,
                    provider: event.target
                      .value as EvaluatorDefinition["provider"],
                  })
                }
              >
                <option value="groq">Groq</option>
                <option value="openrouter">OpenRouter</option>
                <option value="openai">OpenAI</option>
              </select>
            </label>
            <label className="text-sm">
              Judge model
              <input
                className={inputClass}
                value={value.model}
                onChange={(event) =>
                  change({ ...value, model: event.target.value })
                }
              />
            </label>
          </div>
          <label className="text-sm">
            Rubric
            <textarea
              className={inputClass}
              rows={5}
              value={value.rubric}
              onChange={(event) =>
                change({ ...value, rubric: event.target.value })
              }
            />
          </label>
          <details>
            <summary className="cursor-pointer text-sm">
              Evaluator inputs
            </summary>
            <div className="grid gap-2 mt-3">
              {Object.entries(value.mapping).map(([name, field]) => (
                <label key={name} className="text-sm">
                  {name}
                  <select
                    className={inputClass}
                    value={field}
                    onChange={(event) =>
                      change({
                        ...value,
                        mapping: {
                          ...value.mapping,
                          [name]: event.target.value as typeof field,
                        },
                      })
                    }
                  >
                    <option value="inputs">Case inputs</option>
                    <option value="output">Candidate output</option>
                    <option value="expected_output">Expected output</option>
                  </select>
                </label>
              ))}
            </div>
          </details>
        </>
      )}
      <h3 className="font-medium">Score outputs</h3>
      {value.outputs.map((score, index) => (
        <div
          key={index}
          className="grid gap-3 rounded-lg border border-(--border) p-3"
        >
          <label className="text-sm">
            Score {index + 1} name
            <input
              className={inputClass}
              value={score.name}
              onChange={(event) =>
                scoreChange(index, { name: event.target.value })
              }
            />
            <span className="field-help">
              Start with a letter; use letters, numbers, dots, underscores or
              hyphens.
            </span>
          </label>
          {value.kind === "llm_judge" && (
            <label className="text-sm">
              Score {index + 1} type
              <select
                className={inputClass}
                value={score.data_type}
                onChange={(event) =>
                  scoreChange(index, {
                    data_type: event.target
                      .value as ScoreDefinition["data_type"],
                  })
                }
              >
                <option value="boolean">Boolean</option>
                <option value="numeric">Numeric</option>
                <option value="categorical">Categorical</option>
              </select>
            </label>
          )}
          {score.data_type === "numeric" && (
            <div className="grid gap-2 sm:grid-cols-3">
              {(["minimum", "maximum", "threshold"] as const).map((field) => (
                <label key={field} className="text-sm">
                  Score {index + 1} {field}
                  <input
                    type="number"
                    step="any"
                    className={inputClass}
                    value={score[field]}
                    onChange={(event) =>
                      scoreChange(index, {
                        [field]: Number(event.target.value),
                      })
                    }
                  />
                </label>
              ))}
            </div>
          )}
          {score.data_type === "categorical" && (
            <>
              <label className="text-sm">
                Score {index + 1} categories
                <input
                  className={inputClass}
                  value={score.categories.join(", ")}
                  onChange={(event) =>
                    scoreChange(index, {
                      categories: event.target.value
                        .split(",")
                        .map((item) => item.trim()),
                    })
                  }
                />
              </label>
              <label className="text-sm">
                Score {index + 1} passing categories
                <input
                  className={inputClass}
                  value={score.passing_categories.join(", ")}
                  onChange={(event) =>
                    scoreChange(index, {
                      passing_categories: event.target.value
                        .split(",")
                        .map((item) => item.trim()),
                    })
                  }
                />
                <span className="field-help">
                  Comma-separated values from the declared categories.
                </span>
              </label>
            </>
          )}
          {value.outputs.length > 1 && (
            <button
              type="button"
              className="btn-ghost justify-self-start"
              onClick={() =>
                change({
                  ...value,
                  outputs: value.outputs.filter(
                    (_, position) => position !== index,
                  ),
                })
              }
            >
              Remove score {index + 1}
            </button>
          )}
        </div>
      ))}
      {value.kind === "llm_judge" && (
        <button
          type="button"
          className="btn-secondary justify-self-start"
          disabled={value.outputs.length >= 5}
          onClick={() =>
            change({
              ...value,
              outputs: [
                ...value.outputs,
                {
                  ...defaultScore(),
                  name: `score_${value.outputs.length + 1}`,
                },
              ],
            })
          }
        >
          Add score output
        </button>
      )}
    </div>
  );
}

function EvaluatorEditor({
  item,
  saved,
}: {
  item: Evaluator | null;
  saved: (id: string) => void;
}) {
  const cache = useQueryClient();
  const [name, setName] = useState(item?.name ?? "");
  const [description, setDescription] = useState(item?.description ?? "");
  const [definition, setDefinition] = useState(
    item?.latest.definition ?? defaultDefinition(),
  );
  const [notes, setNotes] = useState("");
  const [version, setVersion] = useState(item?.latest);
  const [offset, setOffset] = useState(0);
  const [sample, setSample] = useState('{"query":"Hello"}');
  const [output, setOutput] = useState("Hello");
  const [expected, setExpected] = useState("Hello");
  const [noReference, setNoReference] = useState(false);
  const [key, setKey] = useState("");
  const [preview, setPreview] = useState<CaseResult["checks"] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const history = useQuery({
    queryKey: ["evaluator-versions", item?.id, offset],
    queryFn: ({ signal }) => listEvaluatorVersions(item!.id, offset, signal),
    enabled: !!item,
  });
  const dirty =
    !version || canonicalJSON(definition) !== canonicalJSON(version.definition);
  useUnsavedChanges(
    item
      ? dirty || !!notes.trim()
      : !!name.trim() ||
          !!description.trim() ||
          !!notes.trim() ||
          canonicalJSON(definition) !== canonicalJSON(defaultDefinition()),
  );
  const edit = (next: EvaluatorDefinition) => {
    setDefinition(next);
    setPreview(null);
    setKey("");
  };
  async function action(task: () => Promise<void>) {
    setError("");
    setBusy(true);
    try {
      await task();
    } catch (failure) {
      setError(errorText(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="grid gap-4 xl:grid-cols-2 items-start">
      <section className={`${panelClass} grid gap-4 min-w-0`}>
        <h2 className="font-semibold">
          {item ? `${item.name} · v${version?.version}` : "New evaluator"}
        </h2>
        <label className="text-sm">
          Evaluator name
          <input
            className={inputClass}
            value={name}
            disabled={!!item}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label className="text-sm">
          Description
          <textarea
            className={inputClass}
            value={description}
            disabled={!!item}
            onChange={(event) => setDescription(event.target.value)}
          />
        </label>
        {item && (
          <>
            <label className="text-sm">
              Saved evaluator version
              <select
                className={inputClass}
                value={version?.id ?? ""}
                onChange={(event) => {
                  const selected = history.data?.find(
                    (entry) => entry.id === event.target.value,
                  );
                  if (!selected) return;
                  if (
                    dirty &&
                    !window.confirm("Discard the unsaved evaluator definition?")
                  )
                    return;
                  setVersion(selected);
                  setDefinition(selected.definition);
                  setNotes("");
                  setKey("");
                  setPreview(null);
                }}
              >
                {history.data?.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    v{entry.version}
                    {entry.version === item.latest_version ? " · latest" : ""}
                  </option>
                ))}
                {version &&
                  !history.data?.some((entry) => entry.id === version.id) && (
                    <option value={version.id}>v{version.version}</option>
                  )}
              </select>
            </label>
            <PageControls
              offset={offset}
              next={history.data?.length === 50}
              change={setOffset}
            />
          </>
        )}
        <DefinitionFields value={definition} change={edit} />
        {version?.notes && (
          <p className="field-help break-words">
            Saved version notes: {version.notes}
          </p>
        )}
        <label className="text-sm">
          Version notes
          <textarea
            className={inputClass}
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
          />
        </label>
        <p className="field-help">
          Saving creates an immutable definition. Existing runs retain their
          pinned version.
        </p>
        {error && (
          <p role="alert" className="text-(--destructive)">
            {error}
          </p>
        )}
        {history.error && (
          <p role="alert" className="text-(--destructive)">
            {errorText(history.error)}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <button
            className="btn-primary"
            disabled={busy || !name.trim() || !dirty || item?.archived}
            onClick={() =>
              action(async () => {
                if (item)
                  await saveEvaluatorVersion(
                    item.id,
                    definition,
                    notes,
                    item.latest_version,
                  );
                else {
                  const result = await createEvaluator(
                    name,
                    description,
                    definition,
                    notes,
                  );
                  saved(result.id);
                }
                await cache.invalidateQueries({ queryKey: ["evaluators"] });
                if (item)
                  await cache.invalidateQueries({
                    queryKey: ["evaluator", item.id],
                  });
              })
            }
          >
            {busy
              ? "Saving…"
              : item
                ? "Save evaluator version"
                : "Create evaluator"}
          </button>
          {item && (
            <button
              className="btn-secondary"
              disabled={busy}
              onClick={() =>
                action(async () => {
                  await archiveEvaluator(item.id, !item.archived);
                  await cache.invalidateQueries({ queryKey: ["evaluators"] });
                  await cache.invalidateQueries({
                    queryKey: ["evaluator", item.id],
                  });
                })
              }
            >
              {item.archived ? "Restore evaluator" : "Archive evaluator"}
            </button>
          )}
          {item && (
            <button
              className="btn-secondary"
              onClick={() => {
                setVersion(item.latest);
                setDefinition(item.latest.definition);
                setNotes("");
                setPreview(null);
                setKey("");
              }}
            >
              Reload latest / discard draft
            </button>
          )}
        </div>
      </section>
      <section className={`${panelClass} grid gap-4 min-w-0`}>
        <h2 className="font-semibold">Test saved evaluator</h2>
        <p className="field-help">
          Preview one sample before evaluating a dataset. Tests do not enter run
          history or saved scores.
        </p>
        <label className="text-sm">
          Sample inputs JSON
          <textarea
            className={inputClass}
            rows={4}
            value={sample}
            onChange={(event) => {
              setSample(event.target.value);
              setPreview(null);
            }}
          />
        </label>
        <label className="text-sm">
          Candidate output
          <textarea
            className={inputClass}
            rows={5}
            value={output}
            onChange={(event) => {
              setOutput(event.target.value);
              setPreview(null);
            }}
          />
        </label>
        <label className="text-sm flex gap-2 items-center">
          <input
            type="checkbox"
            checked={noReference}
            onChange={(event) => {
              setNoReference(event.target.checked);
              setPreview(null);
            }}
          />
          No expected output
        </label>
        {!noReference && (
          <label className="text-sm">
            Expected output
            <textarea
              className={inputClass}
              rows={3}
              value={expected}
              onChange={(event) => {
                setExpected(event.target.value);
                setPreview(null);
              }}
            />
          </label>
        )}
        {definition.kind === "llm_judge" && (
          <label className="text-sm">
            Test judge API key
            <input
              type="password"
              autoComplete="off"
              className={inputClass}
              value={key}
              onChange={(event) => setKey(event.target.value)}
            />
            <span className="field-help">
              One provider call. Key is request-only.
            </span>
          </label>
        )}
        <button
          className="btn-primary justify-self-start"
          disabled={
            busy ||
            dirty ||
            !version ||
            item?.archived ||
            (definition.kind === "llm_judge" && !key.trim())
          }
          onClick={() =>
            action(async () => {
              setPreview(null);
              const inputs = JSON.parse(sample);
              if (
                !inputs ||
                Array.isArray(inputs) ||
                typeof inputs !== "object" ||
                Object.values(inputs).some((value) => typeof value !== "string")
              )
                throw new Error(
                  "Sample inputs must be an object of string values",
                );
              const result = await testEvaluator(version!.id, {
                inputs,
                output,
                expected_output: noReference ? null : expected,
                ...(definition.kind === "llm_judge" ? { api_key: key } : {}),
              });
              setKey("");
              setPreview(result.checks);
            })
          }
        >
          Test evaluator
        </button>
        {dirty && (
          <p className="field-help">
            Save your definition to test its exact version.
          </p>
        )}
        {preview && (
          <div role="status" className="grid gap-3">
            {preview.map((check) => (
              <div
                key={check.kind}
                className="rounded-lg border border-(--border) p-3"
              >
                <p className="font-medium">
                  {check.kind}: {String(check.value ?? check.passed)} ·{" "}
                  {check.passed ? "Passed" : "Failed"}
                </p>
                <p className="text-sm">{check.reason}</p>
                {check.latency_ms != null && (
                  <p className="field-help">
                    {check.latency_ms.toFixed(0)} ms ·{" "}
                    {check.tokens_input ?? "—"} input /{" "}
                    {check.tokens_output ?? "—"} output tokens
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export function EvaluatorsWorkbench() {
  const [search, setSearch] = useState("");
  const [archived, setArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const [id, setId] = useState("");
  const [creating, setCreating] = useState(false);
  const catalog = useQuery({
    queryKey: ["evaluators", search, archived, offset],
    queryFn: ({ signal }) => listEvaluators(search, archived, offset, signal),
  });
  const detail = useQuery({
    queryKey: ["evaluator", id],
    queryFn: ({ signal }) => getEvaluator(id, signal),
    enabled: !!id,
  });
  return (
    <div className="page-stack">
      <PageHeader
        title="Evaluators"
        description="Define reusable checks and judges. Every saved definition has an immutable version."
        actions={
          <>
            <Link className="btn-secondary" href="/evaluations">
              Runs
            </Link>
            <button
              className="btn-primary"
              onClick={() => {
                if (
                  (id || creating) &&
                  !window.confirm(
                    "Leave the current editor? Any unsaved draft will be discarded.",
                  )
                )
                  return;
                setId("");
                setCreating(true);
              }}
            >
              New evaluator
            </button>
          </>
        }
      />
      <section className={panelClass}>
        <div className="table-toolbar">
          <label className="text-sm flex-1">
            Search evaluators
            <input
              className={inputClass}
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setOffset(0);
              }}
            />
          </label>
          <label className="text-sm flex gap-2 items-center">
            <input
              type="checkbox"
              checked={archived}
              onChange={(event) => {
                setArchived(event.target.checked);
                setOffset(0);
              }}
            />
            Archived
          </label>
        </div>
        {catalog.isLoading && <p>Loading evaluators…</p>}
        {catalog.error && <p role="alert">{errorText(catalog.error)}</p>}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Evaluator</th>
                <th>Method</th>
                <th>Latest version</th>
                <th>Scores</th>
              </tr>
            </thead>
            <tbody>
              {catalog.data?.items.map((item) => (
                <tr key={item.id} className="data-row">
                  <td>
                    <button
                      className="text-(--accent)"
                      onClick={() => {
                        if (
                          ((id && id !== item.id) || creating) &&
                          !window.confirm(
                            "Leave the current editor? Any unsaved draft will be discarded.",
                          )
                        )
                          return;
                        setCreating(false);
                        setId(item.id);
                      }}
                    >
                      {item.name}
                    </button>
                  </td>
                  <td>
                    {item.latest.definition.kind === "builtin"
                      ? "Built-in"
                      : "LLM judge"}
                  </td>
                  <td>v{item.latest_version}</td>
                  <td>
                    {item.latest.definition.outputs
                      .map((score) => score.name)
                      .join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {catalog.data?.total === 0 && (
          <p className="field-help">
            No matching evaluators. Create a built-in check to try the workflow
            without model calls.
          </p>
        )}
        <PageControls
          offset={offset}
          next={offset + 50 < (catalog.data?.total ?? 0)}
          change={setOffset}
        />
      </section>
      {detail.isLoading && <p>Loading evaluator…</p>}
      {detail.error && <p role="alert">{errorText(detail.error)}</p>}
      {creating ? (
        <EvaluatorEditor
          key="new"
          item={null}
          saved={(identifier) => {
            setCreating(false);
            setId(identifier);
          }}
        />
      ) : (
        detail.data && (
          <EvaluatorEditor
            key={`${detail.data.id}:${detail.data.latest_version}:${detail.data.archived}`}
            item={detail.data}
            saved={setId}
          />
        )
      )}
    </div>
  );
}
