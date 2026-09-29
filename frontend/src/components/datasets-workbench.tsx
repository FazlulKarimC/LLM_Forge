"use client";
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useUnsavedChanges } from "@/lib/use-unsaved-changes";
import {
  createDataset,
  getDataset,
  importDataset,
  listDatasets,
  listRevisions,
  saveRevision,
  updateDataset,
  exportJson,
  type DatasetDetail,
  type DatasetCase,
} from "@/lib/evaluation-api";
import {
  errorText,
  inputClass,
  buttonClass,
} from "@/components/prompts/prompt-ui";
import { PageHeader } from "@/components/ui/primitives";

const sample: DatasetCase[] = [
  { name: "Greeting", inputs: { query: "Hello" }, expected_output: "Hello" },
  {
    name: "Farewell",
    inputs: { query: "Goodbye" },
    expected_output: "Goodbye",
  },
];
const pretty = (cases: DatasetCase[]) => JSON.stringify(cases, null, 2);
// Ignore JSON whitespace and object-key order without changing case order.
const sameCases = (text: string, saved: DatasetCase[]) => {
  try {
    const ordered = (_key: string, value: unknown) =>
      value && typeof value === "object" && !Array.isArray(value)
        ? Object.fromEntries(
            Object.entries(value).sort(([a], [b]) => a.localeCompare(b)),
          )
        : value;
    return (
      JSON.stringify(JSON.parse(text), ordered) ===
      JSON.stringify(saved, ordered)
    );
  } catch {
    return false;
  }
};

export function DatasetsWorkbench() {
  const router = useRouter();
  const cache = useQueryClient();
  const [archived, setArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<DatasetDetail | null>(null);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [text, setText] = useState(pretty(sample));
  const [format, setFormat] = useState<"json" | "csv">("json");
  const [importText, setImportText] = useState("");
  const [importPreview, setImportPreview] = useState<DatasetCase[] | null>(
    null,
  );
  const [caseView, setCaseView] = useState<"table" | "json">("table");
  const [caseIndex, setCaseIndex] = useState<number | null>(null);
  const [caseDraft, setCaseDraft] = useState<{
    name: string;
    inputs: string;
    expected_output: string;
  } | null>(null);
  const [historyOffset, setHistoryOffset] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const list = useQuery({
    queryKey: ["datasets", archived, offset],
    queryFn: ({ signal }) => listDatasets(archived, offset, signal),
  });
  const history = useQuery({
    queryKey: ["dataset-history", selected?.dataset.id, historyOffset],
    queryFn: ({ signal }) =>
      listRevisions(selected!.dataset.id, historyOffset, signal),
    enabled: !!selected,
  });
  const cases = (() => {
    try {
      const parsed: unknown = JSON.parse(text);
      return Array.isArray(parsed) &&
        parsed.every(
          (row) =>
            row &&
            typeof row === "object" &&
            !Array.isArray(row) &&
            typeof row.inputs === "object" &&
            row.inputs !== null &&
            !Array.isArray(row.inputs),
        )
        ? (parsed as DatasetCase[])
        : null;
    } catch {
      return null;
    }
  })();
  const caseDraftDirty =
    caseIndex !== null &&
    caseDraft !== null &&
    cases !== null &&
    (caseDraft.name !== (cases[caseIndex]?.name ?? "") ||
      caseDraft.inputs !==
        JSON.stringify(cases[caseIndex]?.inputs ?? {}, null, 2) ||
      caseDraft.expected_output !== (cases[caseIndex]?.expected_output ?? ""));
  const hasDraftChanges =
    caseDraftDirty ||
    !sameCases(text, selected?.revision.cases ?? sample) ||
    name !== (selected?.dataset.name ?? "") ||
    description !== (selected?.dataset.description ?? "");
  const dirty = editing && hasDraftChanges;
  const restoringRevision =
    !!selected && selected.revision.version !== selected.dataset.latest_version;
  useUnsavedChanges(dirty);
  const load = (detail: DatasetDetail | null) => {
    const url = detail
      ? `/datasets?${new URLSearchParams({ dataset: detail.dataset.id, revision: detail.revision.id })}`
      : "/datasets";
    window.history.replaceState(null, "", url);
    setSelected(detail);
    setName(detail?.dataset.name ?? "");
    setDescription(detail?.dataset.description ?? "");
    setText(pretty(detail?.revision.cases ?? sample));
    setHistoryOffset(0);
    setEditing(true);
    setCaseIndex(null);
    setCaseDraft(null);
    setCaseView("table");
    setImportPreview(null);
    setError("");
    setMessage("");
  };
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("dataset");
    if (!id) return;
    let active = true;
    getDataset(id)
      .then(async (detail) => {
        if (!active) return;
        const revisionId = params.get("revision");
        if (revisionId && detail.revision.id !== revisionId) {
          let found = false;
          for (let offset = 0; offset < 500 && !found; offset += 50) {
            const page = await listRevisions(id, offset);
            const revision = page.find((entry) => entry.id === revisionId);
            if (revision) {
              detail = { ...detail, revision };
              found = true;
            }
            if (page.length < 50) break;
          }
          if (!found)
            throw new Error("The linked dataset revision was not found.");
        }
        if (active) load(detail);
      })
      .catch((cause) => {
        if (active) setError(errorText(cause));
      });
    return () => {
      active = false;
    };
    // Restore a deep link on first mount only; later selection is managed in this workbench.
  }, []);
  function updateCases(next: DatasetCase[]) {
    setText(pretty(next));
    setCaseIndex(null);
    setCaseDraft(null);
  }
  function openCase(index: number) {
    if (caseDraftDirty || !cases?.[index]) return;
    const row = cases[index];
    setCaseIndex(index);
    setCaseDraft({
      name: row.name ?? "",
      inputs: JSON.stringify(row.inputs, null, 2),
      expected_output: row.expected_output ?? "",
    });
  }
  function applyCase() {
    if (caseIndex === null || !caseDraft || !cases) return;
    let inputs: unknown;
    try {
      inputs = JSON.parse(caseDraft.inputs);
    } catch {
      setError("Case inputs must be valid JSON.");
      return;
    }
    if (
      !inputs ||
      typeof inputs !== "object" ||
      Array.isArray(inputs) ||
      Object.values(inputs).some((value) => typeof value !== "string")
    ) {
      setError("Case inputs must be a JSON object with string values.");
      return;
    }
    updateCases(
      cases.map((row, index) =>
        index === caseIndex
          ? {
              ...row,
              name: caseDraft.name,
              inputs: inputs as Record<string, string>,
              expected_output: caseDraft.expected_output || null,
            }
          : row,
      ),
    );
    setError("");
  }
  async function act(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await task();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    await act(async () => {
      let cases: DatasetCase[];
      try {
        cases = JSON.parse(text);
      } catch {
        throw new Error("Cases must be valid JSON");
      }
      if (!Array.isArray(cases)) throw new Error("Cases must be a JSON array");
      if (!selected) load(await createDataset(name, description, cases));
      else {
        // Save cases first; metadata failure can be retried without duplicating a revision.
        let revision = selected.revision;
        if (
          selected.revision.version !== selected.dataset.latest_version ||
          !sameCases(text, selected.revision.cases)
        )
          revision = await saveRevision(
            selected.dataset.id,
            cases,
            selected.dataset.latest_version,
          );
        const dataset = {
          ...selected.dataset,
          latest_version: revision.version,
        };
        setSelected({ dataset, revision });
        setText(pretty(revision.cases));
        const updated = await updateDataset(dataset.id, { name, description });
        load({ dataset: updated, revision });
      }
      await cache.invalidateQueries({ queryKey: ["datasets"] });
      await cache.invalidateQueries({ queryKey: ["dataset-history"] });
      setMessage(
        "Dataset saved. Existing revisions and evaluation results are preserved.",
      );
    });
  }
  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Test cases"
        title="Datasets"
        description="Version your test cases, then evaluate saved prompts against them."
        actions={
          <button
            className={buttonClass}
            disabled={busy || dirty}
            onClick={() => load(null)}
          >
            New dataset
          </button>
        }
      />
      {error && (
        <p role="alert" className="text-(--destructive)">
          {error}
        </p>
      )}
      {message && (
        <p role="status" className="text-(--success)">
          {message}
        </p>
      )}
      <div className="grid lg:grid-cols-[280px_1fr] gap-6">
        <aside className="space-y-3">
          <label className="text-sm flex gap-2">
            <input
              type="checkbox"
              checked={archived}
              onChange={(e) => {
                setArchived(e.target.checked);
                setOffset(0);
              }}
            />{" "}
            Show archived
          </label>
          {list.isLoading && <p>Loading datasets…</p>}
          {list.error && (
            <p role="alert">
              {errorText(list.error)}{" "}
              <button className="btn-ghost" onClick={() => list.refetch()}>
                Retry
              </button>
            </p>
          )}
          {list.data?.items.length === 0 && (
            <p className="text-(--muted-foreground)">
              No datasets yet. Create one or import your cases.
            </p>
          )}
          {list.data?.items.map((item) => (
            <button
              className="w-full rounded-xl border border-(--border) p-3 text-left disabled:opacity-40"
              key={item.id}
              disabled={busy || dirty}
              onClick={() => act(async () => load(await getDataset(item.id)))}
            >
              <span className="block font-medium break-words">{item.name}</span>
              <span className="text-xs text-(--muted-foreground)">
                Revision {item.latest_version}
                {item.archived && " · Archived"}
              </span>
            </button>
          ))}
          <div className="flex gap-2">
            <button
              className="btn-secondary"
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 50))}
            >
              Previous
            </button>
            <button
              className="btn-secondary"
              disabled={!list.data || offset + 50 >= list.data.total}
              onClick={() => setOffset(offset + 50)}
            >
              Next
            </button>
          </div>
        </aside>
        {editing ? (
          <section className="space-y-4 rounded-xl border border-(--border) p-5">
            <div className="flex flex-wrap gap-3 items-center">
              <h2 className="font-semibold">
                {selected
                  ? `Revision ${selected.revision.version} · ${selected.revision.cases.length} cases`
                  : "New dataset"}
              </h2>
              {selected && (
                <button
                  disabled={busy}
                  className={buttonClass}
                  onClick={() =>
                    act(async () => {
                      load(await getDataset(selected.dataset.id));
                      setMessage(
                        "Loaded the latest revision; draft discarded.",
                      );
                    })
                  }
                >
                  Reload latest / discard draft
                </button>
              )}
              {!selected && dirty && (
                <button className="btn-secondary" onClick={() => load(null)}>
                  Discard draft
                </button>
              )}
            </div>
            <label className="block text-sm">
              Dataset name
              <input
                aria-label="Dataset name"
                className={inputClass}
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={255}
                disabled={selected?.dataset.archived || busy}
              />
            </label>
            <label className="block text-sm">
              Description
              <textarea
                className={inputClass}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                maxLength={4000}
                disabled={selected?.dataset.archived || busy}
              />
            </label>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-(--border) pt-4">
              <div>
                <h3 className="font-medium">Cases</h3>
                <p className="text-xs text-(--muted-foreground)">
                  Edit the draft, then save it as a new revision.
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  className={buttonClass}
                  aria-pressed={caseView === "table"}
                  onClick={() => setCaseView("table")}
                >
                  Case table
                </button>
                <button
                  type="button"
                  className={buttonClass}
                  aria-pressed={caseView === "json"}
                  disabled={caseDraftDirty}
                  onClick={() => setCaseView("json")}
                >
                  Advanced JSON
                </button>
              </div>
            </div>
            {caseView === "json" ? (
              <label className="block text-sm">
                Cases (JSON array)
                <textarea
                  aria-label="Dataset cases"
                  className={`${inputClass} min-h-72 font-mono text-xs`}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  disabled={selected?.dataset.archived || busy}
                />
              </label>
            ) : (
              <div className="space-y-3">
                {!cases && (
                  <p role="alert" className="text-sm text-(--destructive)">
                    The draft is not a JSON array. Correct it in Advanced JSON.
                  </p>
                )}
                {cases && (
                  <div className="overflow-x-auto rounded-lg border border-(--border)">
                    <table className="w-full min-w-[560px] text-left text-sm">
                      <thead>
                        <tr className="border-b border-(--border)">
                          <th className="p-3">Case</th>
                          <th className="p-3">Inputs</th>
                          <th className="p-3">Reference</th>
                          <th className="p-3">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {cases.map((row, index) => (
                          <tr
                            key={index}
                            className="border-b border-(--border) last:border-0"
                          >
                            <td className="p-3">
                              <button
                                type="button"
                                className="font-medium text-(--primary) hover:underline"
                                disabled={caseDraftDirty && caseIndex !== index}
                                onClick={() => openCase(index)}
                              >
                                {row.name || `Case ${index + 1}`}
                              </button>
                            </td>
                            <td className="max-w-64 truncate p-3 font-mono text-xs">
                              {JSON.stringify(row.inputs)}
                            </td>
                            <td className="max-w-64 truncate p-3">
                              {row.expected_output ?? "No reference"}
                            </td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button
                                  type="button"
                                  className="btn-ghost min-h-8! px-2! text-xs!"
                                  disabled={
                                    !!selected?.dataset.archived ||
                                    busy ||
                                    caseDraftDirty
                                  }
                                  onClick={() =>
                                    updateCases([
                                      ...cases.slice(0, index + 1),
                                      structuredClone(row),
                                      ...cases.slice(index + 1),
                                    ])
                                  }
                                >
                                  Duplicate
                                </button>
                                <button
                                  type="button"
                                  className="btn-ghost min-h-8! px-2! text-xs!"
                                  disabled={
                                    !!selected?.dataset.archived ||
                                    busy ||
                                    caseDraftDirty
                                  }
                                  onClick={() =>
                                    updateCases(
                                      cases.filter(
                                        (_, rowIndex) => rowIndex !== index,
                                      ),
                                    )
                                  }
                                >
                                  Remove
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {cases && (
                  <button
                    type="button"
                    className={buttonClass}
                    disabled={
                      !!selected?.dataset.archived ||
                      busy ||
                      caseDraftDirty ||
                      cases.length >= 100
                    }
                    onClick={() => {
                      const next = [
                        ...cases,
                        { name: "", inputs: {}, expected_output: null },
                      ];
                      updateCases(next);
                    }}
                  >
                    Add case
                  </button>
                )}
                {caseIndex !== null && caseDraft && (
                  <div className="space-y-3 rounded-lg border border-(--border) p-4">
                    <h4 className="font-medium">Edit case {caseIndex + 1}</h4>
                    <label className="block text-sm">
                      Name
                      <input
                        className={inputClass}
                        value={caseDraft.name}
                        onChange={(event) =>
                          setCaseDraft({
                            ...caseDraft,
                            name: event.target.value,
                          })
                        }
                        disabled={!!selected?.dataset.archived || busy}
                      />
                    </label>
                    <label className="block text-sm">
                      Inputs (JSON object with string values)
                      <textarea
                        className={`${inputClass} min-h-24 font-mono text-xs`}
                        value={caseDraft.inputs}
                        onChange={(event) =>
                          setCaseDraft({
                            ...caseDraft,
                            inputs: event.target.value,
                          })
                        }
                        disabled={!!selected?.dataset.archived || busy}
                      />
                    </label>
                    <label className="block text-sm">
                      Reference output
                      <textarea
                        className={inputClass}
                        value={caseDraft.expected_output}
                        onChange={(event) =>
                          setCaseDraft({
                            ...caseDraft,
                            expected_output: event.target.value,
                          })
                        }
                        disabled={!!selected?.dataset.archived || busy}
                      />
                    </label>
                    <div className="flex gap-3">
                      <button
                        type="button"
                        className={buttonClass}
                        disabled={
                          !!selected?.dataset.archived ||
                          busy ||
                          !caseDraftDirty
                        }
                        onClick={applyCase}
                      >
                        Apply case
                      </button>
                      <button
                        type="button"
                        className="btn-ghost"
                        disabled={busy}
                        onClick={() => {
                          setCaseIndex(null);
                          setCaseDraft(null);
                        }}
                      >
                        Discard row edits
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
            <p className="text-xs text-(--muted-foreground)">
              Each case has inputs (string values), expected_output (string or
              null), and an optional name. Maximum 100 cases / 1 MB. Exact match
              requires a reference for every case.
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                className={buttonClass}
                disabled={
                  busy ||
                  caseDraftDirty ||
                  selected?.dataset.archived ||
                  !name.trim() ||
                  (!!selected && !dirty && !restoringRevision)
                }
                onClick={save}
              >
                {busy
                  ? "Saving…"
                  : selected
                    ? "Save changes"
                    : "Create dataset"}
              </button>
              {selected && (
                <>
                  <button
                    className={buttonClass}
                    disabled={busy || dirty}
                    onClick={() =>
                      act(async () => {
                        const updated = await updateDataset(
                          selected.dataset.id,
                          { archived: !selected.dataset.archived },
                        );
                        setSelected({ ...selected, dataset: updated });
                        await cache.invalidateQueries({
                          queryKey: ["datasets"],
                        });
                      })
                    }
                  >
                    {selected.dataset.archived ? "Restore" : "Archive"}
                  </button>
                  <button
                    className={buttonClass}
                    onClick={() =>
                      exportJson(
                        `dataset-v${selected.revision.version}.json`,
                        selected.revision.cases,
                      )
                    }
                  >
                    Export saved revision
                  </button>
                  <button
                    className={buttonClass}
                    disabled={
                      busy || hasDraftChanges || selected.dataset.archived
                    }
                    onClick={() =>
                      router.push(
                        `/evaluations?${new URLSearchParams({ view: "new", dataset: selected.dataset.id, revision: selected.revision.id })}`,
                      )
                    }
                  >
                    Evaluate this revision
                  </button>
                  {hasDraftChanges && (
                    <p className="text-xs text-(--muted-foreground)">
                      Save or discard your draft before evaluating the selected
                      revision.
                    </p>
                  )}
                </>
              )}
            </div>
            {!selected?.dataset.archived && (
              <details className="border-t border-(--border) pt-3">
                <summary>Import CSV or JSON into this draft</summary>
                <div className="space-y-3 mt-3">
                  <select
                    aria-label="Import format"
                    className={inputClass}
                    value={format}
                    onChange={(e) =>
                      setFormat(e.target.value as "json" | "csv")
                    }
                  >
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                  </select>
                  <p className="text-xs text-(--muted-foreground)">
                    CSV example: input.query,expected_output,name followed by
                    Hello,Hello,Greeting. Alternatively use an inputs column
                    containing a quoted JSON object.
                  </p>
                  <input
                    type="file"
                    accept=".csv,.json"
                    disabled={busy}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      if (file.size > 1_000_000) {
                        setError("File exceeds 1 MB");
                        return;
                      }
                      act(async () => {
                        setImportText(await file.text());
                        setFormat(
                          file.name.toLowerCase().endsWith(".csv")
                            ? "csv"
                            : "json",
                        );
                      });
                      e.target.value = "";
                    }}
                  />
                  <textarea
                    aria-label="Import content"
                    className={`${inputClass} min-h-32 font-mono text-xs`}
                    value={importText}
                    onChange={(e) => {
                      setImportText(e.target.value);
                      setImportPreview(null);
                    }}
                  />
                  <button
                    disabled={busy || !importText}
                    className={buttonClass}
                    onClick={() =>
                      act(async () => {
                        const parsed = await importDataset(format, importText);
                        setImportPreview(parsed.cases);
                        setMessage(
                          `Validated ${parsed.cases.length} cases. Review the preview before replacing your draft.`,
                        );
                      })
                    }
                  >
                    Validate and preview
                  </button>
                  {importPreview && (
                    <div className="rounded-lg border border-(--border) p-3 text-sm">
                      <p className="font-medium">
                        Import preview · {importPreview.length} cases
                      </p>
                      <ul className="mt-2 max-h-40 overflow-auto text-xs">
                        {importPreview.slice(0, 10).map((row, index) => (
                          <li key={index}>
                            {row.name || `Case ${index + 1}`} ·{" "}
                            {JSON.stringify(row.inputs)} ·{" "}
                            {row.expected_output ?? "No reference"}
                          </li>
                        ))}
                      </ul>
                      {importPreview.length > 10 && (
                        <p className="mt-2 text-xs">Showing first 10 cases</p>
                      )}
                      <button
                        type="button"
                        className={`${buttonClass} mt-3`}
                        disabled={caseDraftDirty}
                        onClick={() => {
                          updateCases(importPreview);
                          setImportPreview(null);
                          setMessage(
                            "Imported cases into the draft. Save to create a revision.",
                          );
                        }}
                      >
                        Replace draft cases
                      </button>
                    </div>
                  )}
                </div>
              </details>
            )}
            {selected && (
              <div className="border-t border-(--border) pt-3">
                <h3 className="font-medium">Saved revisions</h3>
                {history.error && (
                  <p role="alert">{errorText(history.error)}</p>
                )}
                <div className="flex flex-wrap gap-2 mt-2">
                  {history.data?.map((revision) => (
                    <button
                      key={revision.id}
                      disabled={dirty || busy}
                      className={buttonClass}
                      onClick={() => {
                        setSelected({ ...selected, revision });
                        setText(pretty(revision.cases));
                        setCaseIndex(null);
                        setCaseDraft(null);
                        window.history.replaceState(
                          null,
                          "",
                          `/datasets?${new URLSearchParams({ dataset: selected.dataset.id, revision: revision.id })}`,
                        );
                      }}
                    >
                      v{revision.version} · {revision.cases.length} cases
                    </button>
                  ))}
                </div>
                <p className="text-xs text-(--muted-foreground) mt-2">
                  Selecting an older revision loads it as a draft; saving
                  creates a new revision. Discard unsaved changes before
                  switching.
                </p>
                <div className="flex gap-3">
                  <button
                    className="btn-secondary"
                    disabled={!historyOffset}
                    onClick={() =>
                      setHistoryOffset(Math.max(0, historyOffset - 50))
                    }
                  >
                    Newer revisions
                  </button>
                  <button
                    className="btn-secondary"
                    disabled={history.data?.length !== 50}
                    onClick={() => setHistoryOffset(historyOffset + 50)}
                  >
                    Older revisions
                  </button>
                </div>
              </div>
            )}
          </section>
        ) : (
          <p className="text-(--muted-foreground)">
            Select a dataset or create one. For the demo, use the sample cases
            with a prompt containing only {"{{query}}"}.
          </p>
        )}
      </div>
    </div>
  );
}
