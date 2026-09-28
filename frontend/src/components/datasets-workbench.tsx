"use client";
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
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
        ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)))
        : value;
    return JSON.stringify(JSON.parse(text), ordered) === JSON.stringify(saved, ordered);
  } catch {
    return false;
  }
};

export function DatasetsWorkbench() {
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
    queryFn: ({ signal }) => listRevisions(selected!.dataset.id, historyOffset, signal),
    enabled: !!selected,
  });
  const dirty =
    editing &&
    ((!!selected &&
      selected.revision.version !== selected.dataset.latest_version) ||
      !sameCases(text, selected?.revision.cases ?? sample) ||
      name !== (selected?.dataset.name ?? "") ||
      description !== (selected?.dataset.description ?? ""));
  useUnsavedChanges(dirty);
  const load = (detail: DatasetDetail | null) => {
    setSelected(detail);
    setName(detail?.dataset.name ?? "");
    setDescription(detail?.dataset.description ?? "");
    setText(pretty(detail?.revision.cases ?? sample));
    setHistoryOffset(0);
    setEditing(true);
    setError("");
    setMessage("");
  };
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
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      <div className="flex justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Datasets</h1>
          <p className="text-sm text-(--muted-foreground)">
            Version your test cases, then evaluate saved prompts against them.
          </p>
        </div>
        <button
          className={buttonClass}
          disabled={busy || dirty}
          onClick={() => load(null)}
        >
          New dataset
        </button>
      </div>
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
              <button onClick={() => list.refetch()}>Retry</button>
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
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 50))}
            >
              Previous
            </button>
            <button
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
                <button onClick={() => load(null)}>Discard draft</button>
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
                  selected?.dataset.archived ||
                  !name.trim() ||
                  (!!selected && !dirty)
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
                  <Link className={buttonClass} href="/evaluations">
                    Evaluate prompts
                  </Link>
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
                    onChange={(e) => setImportText(e.target.value)}
                  />
                  <button
                    disabled={busy || !importText}
                    className={buttonClass}
                    onClick={() =>
                      act(async () => {
                        const parsed = await importDataset(format, importText);
                        setText(pretty(parsed.cases));
                        setMessage(
                          `Imported ${parsed.cases.length} cases into draft. Save to create a revision.`,
                        );
                      })
                    }
                  >
                    Validate and replace draft cases
                  </button>
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
                    disabled={!historyOffset}
                    onClick={() =>
                      setHistoryOffset(Math.max(0, historyOffset - 50))
                    }
                  >
                    Newer revisions
                  </button>
                  <button
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
