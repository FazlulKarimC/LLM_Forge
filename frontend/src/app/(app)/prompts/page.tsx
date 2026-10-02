"use client";
import Link from "next/link";
import { useDeferredValue, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus, Search } from "lucide-react";
import { PageHeader, TableToolbar } from "@/components/ui/primitives";
import {
  ErrorMessage,
  errorText,
  inputClass,
} from "@/components/prompts/prompt-ui";
import { listPrompts } from "@/lib/prompt-api";
export default function PromptsPage() {
  const [search, setSearch] = useState("");
  const [archived, setArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const [tag, setTag] = useState("");
  const [folder, setFolder] = useState("");
  const deferredTag = useDeferredValue(tag.trim());
  const deferredFolder = useDeferredValue(folder.trim());
  const deferredSearch = useDeferredValue(search);
  const query = useQuery({
    queryKey: [
      "prompt-library",
      deferredSearch,
      archived,
      offset,
      deferredTag,
      deferredFolder,
    ],
    queryFn: ({ signal }) =>
      listPrompts(deferredSearch, archived, offset, signal, {
        tag: deferredTag,
        folder: deferredFolder,
      }),
  });
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Prompt management"
        title="Prompts"
        description="Develop, test, and release versions your application can fetch."
        actions={
          <Link href="/prompts/new" className="btn-primary">
            <Plus className="size-4" />
            New prompt
          </Link>
        }
      />
      <TableToolbar>
        <label className="min-w-48 flex-1 text-sm">
          <span className="inline-flex items-center gap-2">
            <Search className="size-4" />
            Search prompts
          </span>
          <input
            className={inputClass}
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setOffset(0);
            }}
            placeholder="Find a prompt by name"
            maxLength={255}
          />
        </label>
        <label className="text-sm">
          Tag
          <input
            className={inputClass}
            value={tag}
            onChange={(event) => {
              setTag(event.target.value);
              setOffset(0);
            }}
            maxLength={64}
            placeholder="Exact tag"
          />
        </label>
        <label className="text-sm">
          Folder
          <input
            className={inputClass}
            value={folder}
            onChange={(event) => {
              setFolder(event.target.value);
              setOffset(0);
            }}
            maxLength={255}
            placeholder="support"
          />
        </label>
        <label className="text-sm">
          Show
          <select
            className={inputClass}
            value={String(archived)}
            onChange={(event) => {
              setArchived(event.target.value === "true");
              setOffset(0);
            }}
          >
            <option value="false">Active prompts</option>
            <option value="true">Archived prompts</option>
          </select>
        </label>
      </TableToolbar>
      <ErrorMessage message={query.error ? errorText(query.error) : null} />
      {query.error && (
        <button
          className="btn-secondary self-start"
          onClick={() => query.refetch()}
        >
          Try again
        </button>
      )}
      {query.isPending ? (
        <p role="status" className="panel p-5">
          Loading prompts…
        </p>
      ) : query.data?.items.length ? (
        <div className="panel overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="border-b border-(--border) text-xs uppercase tracking-wide text-(--muted-foreground)">
              <tr>
                <th className="p-4">Prompt</th>
                <th className="p-4">Latest</th>
                <th className="p-4">Staging</th>
                <th className="p-4">Production</th>
                <th className="p-4">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-(--border)">
              {query.data.items.map((prompt) => (
                <tr key={prompt.id} className="hover:bg-(--surface-2)">
                  <td className="p-4">
                    <Link
                      href={`/prompts/${prompt.id}`}
                      className="font-semibold hover:text-(--primary) hover:underline"
                    >
                      {prompt.name}
                    </Link>
                    <p className="mt-1 max-w-md truncate text-xs text-(--muted-foreground)">
                      {prompt.description || "No description"}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1 text-xs text-(--muted-foreground)">
                      <span className="chip">
                        {prompt.prompt_type ?? "text"}
                      </span>
                      {prompt.tags?.map((tag) => (
                        <button
                          key={tag}
                          className="chip hover:underline"
                          onClick={() => {
                            setTag(tag);
                            setOffset(0);
                          }}
                        >
                          {tag}
                        </button>
                      ))}
                      {prompt.labels
                        .filter(
                          (label) =>
                            !["latest", "staging", "production"].includes(
                              label.label,
                            ),
                        )
                        .map((label) => (
                          <span key={label.label} className="chip">
                            {label.label} · v{label.version}
                          </span>
                        ))}
                    </div>
                  </td>
                  <td className="p-4">v{prompt.latest_version}</td>
                  <td className="p-4">
                    {prompt.labels.find((label) => label.label === "staging")
                      ?.version
                      ? `v${prompt.labels.find((label) => label.label === "staging")?.version}`
                      : "—"}
                  </td>
                  <td className="p-4">
                    {prompt.labels.find((label) => label.label === "production")
                      ?.version
                      ? `v${prompt.labels.find((label) => label.label === "production")?.version}`
                      : "—"}
                  </td>
                  <td className="p-4 text-(--muted-foreground)">
                    {new Date(prompt.updated_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !query.error && (
          <div className="panel py-14 text-center">
            <h2 className="text-xl font-semibold">
              {search || tag || folder
                ? "No matching prompts"
                : archived
                  ? "No archived prompts"
                  : "Your first prompt starts here"}
            </h2>
            <p className="mt-2 text-sm text-(--muted-foreground)">
              {search || tag || folder
                ? "Try another name, tag, or folder."
                : "Write a reusable template and test it before releasing it."}
            </p>
            {!archived && !search && !tag && !folder && (
              <Link href="/prompts/new" className="btn-primary mt-6">
                Create a prompt
              </Link>
            )}
          </div>
        )
      )}
      {query.data && query.data.total > 50 && (
        <div className="flex items-center justify-between gap-4 text-sm">
          <span>
            {offset + 1}–{Math.min(offset + 50, query.data.total)} of{" "}
            {query.data.total}
          </span>
          <div className="flex gap-2">
            <button
              className="btn-secondary"
              disabled={offset === 0}
              onClick={() => setOffset(offset - 50)}
            >
              Previous
            </button>
            <button
              className="btn-secondary"
              disabled={offset + 50 >= query.data.total}
              onClick={() => setOffset(offset + 50)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
