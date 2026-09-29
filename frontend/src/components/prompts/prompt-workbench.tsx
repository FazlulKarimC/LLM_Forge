"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, History, Save, Send } from "lucide-react";
import { PageHeader } from "@/components/ui/primitives";
import { getApiBaseUrl } from "@/lib/api-client";
import { useUnsavedChanges } from "@/lib/use-unsaved-changes";
import {
  archivePrompt,
  createPrompt,
  draftVariables,
  getPrompt,
  getVersions,
  promoteVersion,
  removeLabel,
  saveVersion,
  updatePrompt,
  type PromptDetail,
  type PromptVersion,
  type ReleaseLabel,
  type TemplateFormat,
} from "@/lib/prompt-api";
import { PromptPlayground } from "./prompt-playground";
import { ErrorMessage, errorText, inputClass, panelClass } from "./prompt-ui";

const DEFAULT_TEMPLATE =
  "Answer the following question clearly and concisely.\n\nQuestion: {{query}}\nAnswer:";

export function PromptWorkbench({ initial }: { initial?: PromptDetail }) {
  const router = useRouter();
  const cache = useQueryClient();
  const [prompt, setPrompt] = useState(initial?.prompt ?? null);
  const [selected, setSelected] = useState<PromptVersion | null>(
    initial?.version ?? null,
  );
  const [name, setName] = useState(initial?.prompt.name ?? "");
  const [description, setDescription] = useState(
    initial?.prompt.description ?? "",
  );
  const [template, setTemplate] = useState(
    initial?.version.template_text ?? DEFAULT_TEMPLATE,
  );
  const [format, setFormat] = useState<TemplateFormat>(
    initial?.version.template_format ?? "mustache",
  );
  const [notes, setNotes] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [archiveRequested, setArchiveRequested] = useState(false);
  const [releaseRequested, setReleaseRequested] = useState<ReleaseLabel | null>(
    null,
  );
  const [tab, setTab] = useState<
    "editor" | "versions" | "releases" | "integrate" | "settings"
  >("editor");
  const versions = useInfiniteQuery({
    queryKey: ["prompt-versions", prompt?.id],
    enabled: !!prompt,
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) =>
      getVersions(prompt!.id, pageParam, signal),
    getNextPageParam: (lastPage, pages) =>
      lastPage.length === 50 ? pages.length * 50 : undefined,
  });
  const draft = {
    template_text: template,
    template_format: format,
    description: notes,
  };
  const dirty = selected
    ? template !== selected.template_text || format !== selected.template_format
    : true;
  const metadataDirty =
    !!prompt && (name !== prompt.name || description !== prompt.description);
  useUnsavedChanges(
    prompt
      ? dirty || metadataDirty || !!notes.trim()
      : !!name.trim() ||
          !!description.trim() ||
          template !== DEFAULT_TEMPLATE ||
          format !== "mustache",
  );
  const variables = draftVariables(template, format);

  async function action(kind: string, work: () => Promise<void>) {
    if (pending) return;
    setPending(kind);
    setError(null);
    setNotice(null);
    try {
      await work();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setPending(null);
    }
  }
  async function refresh() {
    await Promise.all([
      cache.invalidateQueries({ queryKey: ["prompt-library"] }),
      cache.invalidateQueries({ queryKey: ["prompt-detail", prompt?.id] }),
      cache.invalidateQueries({ queryKey: ["prompt-versions", prompt?.id] }),
    ]);
  }
  function chooseVersion(version: PromptVersion) {
    setSelected(version);
    setTemplate(version.template_text);
    setFormat(version.template_format);
    setNotes("");
    setReleaseRequested(null);
    setError(null);
    setNotice(null);
  }
  async function reloadLatest() {
    if (!prompt) return;
    await action("reload", async () => {
      const latest = await getPrompt(prompt.id);
      setPrompt(latest.prompt);
      setName(latest.prompt.name);
      setDescription(latest.prompt.description);
      chooseVersion(latest.version);
      cache.setQueryData(["prompt-detail", prompt.id], latest);
      await Promise.all([
        cache.invalidateQueries({ queryKey: ["prompt-library"] }),
        cache.invalidateQueries({ queryKey: ["prompt-versions", prompt.id] }),
      ]);
    });
  }
  async function save() {
    await action("save", async () => {
      if (prompt) {
        const saved = await saveVersion(prompt.id, {
          ...draft,
          base_version: prompt.latest_version,
        });
        setPrompt({ ...prompt, latest_version: saved.version });
        chooseVersion(saved);
        await refresh();
      } else {
        const created = await createPrompt(name.trim(), {
          ...draft,
          description,
        });
        await cache.invalidateQueries({ queryKey: ["prompt-library"] });
        router.push(`/prompts/${created.prompt.id}`);
      }
    });
  }
  async function promote(label: ReleaseLabel) {
    if (!prompt || !selected) return;
    await action(label, async () => {
      setPrompt(await promoteVersion(prompt.id, label, selected.id));
      setReleaseRequested(null);
      setNotice(`${label} now points to v${selected.version}.`);
      await refresh();
    });
  }

  return (
    <div className="space-y-6">
      <PageHeader
        backHref="/prompts"
        backLabel="All prompts"
        eyebrow={prompt?.archived ? "Archived prompt" : "Prompt engineering"}
        title={prompt ? prompt.name : "Create a prompt"}
        description="Saved versions are immutable. Test your draft in the playground, then choose which saved version your application receives."
        actions={
          <>
            <button
              className="btn-primary"
              disabled={
                !!pending ||
                !!prompt?.archived ||
                !name.trim() ||
                !template.trim() ||
                (!!prompt && !dirty)
              }
              onClick={save}
            >
              <Save className="size-4" />
              {pending === "save"
                ? "Saving…"
                : prompt
                  ? "Save new version"
                  : "Create prompt"}
            </button>
            {prompt && selected && (
              <Link
                className="btn-secondary"
                href={`/evaluations?${new URLSearchParams({ prompt: prompt.id, version: selected.id, view: "new" })}`}
              >
                Evaluate saved v{selected.version}
              </Link>
            )}
          </>
        }
      />
      <ErrorMessage message={error} />
      {error && prompt ? (
        <button
          className="btn-secondary"
          disabled={!!pending}
          onClick={reloadLatest}
        >
          Reload latest version (discard draft)
        </button>
      ) : null}
      {notice ? (
        <p
          role="status"
          className="rounded-xl border border-(--primary)/30 bg-(--primary)/10 p-4 text-sm"
        >
          {notice}
        </p>
      ) : null}
      {prompt && (
        <nav
          aria-label="Prompt workspace sections"
          className="flex flex-wrap gap-2 border-b border-(--border) pb-3"
        >
          {(
            [
              ["editor", "Editor & playground"],
              ["versions", "Versions"],
              ["releases", "Releases"],
              ["integrate", "Integrate"],
              ["settings", "Settings"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-current={tab === value ? "page" : undefined}
              className={tab === value ? "btn-primary" : "btn-secondary"}
              onClick={() => setTab(value)}
            >
              {label}
            </button>
          ))}
        </nav>
      )}
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="space-y-6">
          {tab === "editor" && (
            <section className={`${panelClass} space-y-4`}>
              <h2 className="text-xl font-semibold">
                {selected
                  ? `Editor · v${selected.version}${dirty ? " · unsaved changes" : ""}`
                  : "Prompt editor"}
              </h2>
              <label className="block text-sm">
                Prompt name
                <input
                  className={inputClass}
                  value={name}
                  maxLength={255}
                  onChange={(event) => setName(event.target.value)}
                  disabled={!!pending || !!prompt?.archived}
                  placeholder="support-answer"
                />
              </label>
              <label className="block text-sm">
                Description
                <textarea
                  className={inputClass}
                  value={description}
                  rows={2}
                  maxLength={4000}
                  onChange={(event) => setDescription(event.target.value)}
                  disabled={!!pending || !!prompt?.archived}
                  placeholder="What this prompt is used for"
                />
              </label>
              {prompt ? (
                <button
                  className="btn-secondary"
                  disabled={
                    !!pending ||
                    prompt.archived ||
                    (name === prompt.name &&
                      description === prompt.description) ||
                    !name.trim()
                  }
                  onClick={() =>
                    action("metadata", async () => {
                      setPrompt(
                        await updatePrompt(prompt.id, {
                          name: name.trim(),
                          description,
                        }),
                      );
                      setNotice("Prompt details updated.");
                      await refresh();
                    })
                  }
                >
                  {pending === "metadata"
                    ? "Updating…"
                    : "Update name and description"}
                </button>
              ) : null}
              <label className="block text-sm">
                Template format
                <select
                  className={inputClass}
                  value={format}
                  onChange={(event) =>
                    setFormat(event.target.value as TemplateFormat)
                  }
                  disabled={!!pending || !!prompt?.archived}
                >
                  <option value="mustache">Mustache · {"{{variable}}"}</option>
                  <option value="fstring">Brace · {"{variable}"}</option>
                </select>
              </label>
              <label className="block text-sm">
                Template
                <textarea
                  aria-label="Prompt template"
                  className={`${inputClass} min-h-64 font-mono leading-6`}
                  value={template}
                  maxLength={50_000}
                  spellCheck={false}
                  onChange={(event) => setTemplate(event.target.value)}
                  disabled={!!pending || !!prompt?.archived}
                />
              </label>
              <div className="flex flex-wrap gap-2 text-xs">
                {variables.length ? (
                  variables.map((variable) => (
                    <span
                      key={variable}
                      className="rounded-lg bg-(--surface-2) px-2 py-1 font-mono"
                    >
                      {variable}
                    </span>
                  ))
                ) : (
                  <span className="text-(--muted-foreground)">
                    No template variables
                  </span>
                )}
                <span className="ml-auto text-(--muted-foreground)">
                  {template.length.toLocaleString()} / 50,000 characters
                </span>
              </div>
              <p className="text-xs text-(--muted-foreground)">
                {format === "mustache"
                  ? "Use simple names inside double braces. JSON braces remain literal. Expressions and sections are not supported."
                  : "Use simple names inside single braces. Escape literal braces as {{ and }}. Expressions and formatting directives are not supported."}
              </p>
              {prompt ? (
                <label className="block text-sm">
                  Version notes
                  <textarea
                    className={inputClass}
                    value={notes}
                    rows={2}
                    maxLength={4000}
                    onChange={(event) => setNotes(event.target.value)}
                    disabled={!!pending || prompt.archived}
                    placeholder="What changed in this version?"
                  />
                </label>
              ) : null}
              {selected ? (
                <p className="break-all text-xs text-(--muted-foreground)">
                  Saved snapshot SHA-256: <code>{selected.sha256_hash}</code>
                </p>
              ) : null}
            </section>
          )}
          {tab === "versions" && prompt ? (
            <section className={`${panelClass} space-y-4`}>
              <h2 className="flex items-center gap-2 text-xl font-semibold">
                <History className="size-5" />
                Version history
              </h2>
              <p className="text-sm text-(--muted-foreground)">
                Select a snapshot to inspect or promote it. Editing an older
                snapshot creates a new version.
              </p>
              <ErrorMessage
                message={versions.error ? errorText(versions.error) : null}
              />
              {versions.isPending ? (
                <p role="status">Loading versions…</p>
              ) : null}
              <div className="max-h-72 space-y-2 overflow-auto">
                {versions.data?.pages.flat().map((version) => (
                  <button
                    key={version.id}
                    aria-pressed={selected?.id === version.id}
                    disabled={!!pending || dirty}
                    className={`w-full rounded-xl border p-3 text-left text-sm ${selected?.id === version.id ? "border-(--primary) bg-(--primary)/5" : "border-(--border)"}`}
                    onClick={() => chooseVersion(version)}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">v{version.version}</span>
                      {prompt.labels
                        .filter((label) => label.version_id === version.id)
                        .map((label) => (
                          <span
                            key={label.label}
                            className="rounded-md bg-(--surface-2) px-2 py-0.5 text-xs"
                          >
                            {label.label}
                          </span>
                        ))}
                      <span className="ml-auto text-xs text-(--muted-foreground)">
                        {new Date(version.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p className="mt-1 truncate text-(--muted-foreground)">
                      {version.description || "No version notes"}
                    </p>
                  </button>
                ))}
              </div>
              {dirty && selected ? (
                <button
                  className="btn-secondary"
                  disabled={!!pending}
                  onClick={() => chooseVersion(selected)}
                >
                  Discard draft changes
                </button>
              ) : null}
              {versions.hasNextPage ? (
                <button
                  className="btn-secondary"
                  disabled={versions.isFetchingNextPage}
                  onClick={() => versions.fetchNextPage()}
                >
                  Load older versions
                </button>
              ) : null}
            </section>
          ) : null}
        </div>
        <div className="space-y-6">
          {tab === "editor" && <PromptPlayground draft={draft} />}
          {tab === "releases" && prompt && selected ? (
            <section className={`${panelClass} space-y-4`}>
              <h2 className="flex items-center gap-2 text-xl font-semibold">
                <Send className="size-5" />
                Releases
              </h2>
              <p className="text-sm text-(--muted-foreground)">
                Release saved v{selected.version}. Saving a version never
                changes production automatically.
              </p>
              <Link
                className="btn-secondary"
                href={`/evaluations?${new URLSearchParams({ prompt: prompt.id, version: selected.id, view: "new" })}`}
              >
                Evaluate saved v{selected.version}
              </Link>
              {(["staging", "production"] as ReleaseLabel[]).map((label) => (
                <div
                  key={label}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-(--border) p-3"
                >
                  <div>
                    <h3 className="text-sm font-medium capitalize">{label}</h3>
                    <p className="mt-1 text-xs text-(--muted-foreground)">
                      {prompt.labels.find((release) => release.label === label)
                        ? `v${prompt.labels.find((release) => release.label === label)!.version}`
                        : "Not released"}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      className="btn-secondary"
                      disabled={!!pending || prompt.archived || dirty}
                      onClick={() => setReleaseRequested(label)}
                    >
                      Promote v{selected.version}
                    </button>
                    {prompt.labels.some(
                      (release) => release.label === label,
                    ) ? (
                      <button
                        aria-label={`Remove ${label} release`}
                        className="text-xs text-(--muted-foreground) underline"
                        disabled={!!pending || prompt.archived}
                        onClick={() =>
                          action(`remove-${label}`, async () => {
                            await removeLabel(prompt.id, label);
                            setPrompt({
                              ...prompt,
                              labels: prompt.labels.filter(
                                (release) => release.label !== label,
                              ),
                            });
                            setNotice(`${label} release removed.`);
                            await refresh();
                          })
                        }
                      >
                        Remove
                      </button>
                    ) : null}
                  </div>
                </div>
              ))}
              {releaseRequested ? (
                <div className="rounded-xl border border-(--primary) p-4">
                  <p className="text-sm">
                    Point <strong>{releaseRequested}</strong> to saved{" "}
                    <strong>v{selected.version}</strong>? The next SDK fetch
                    using this label will receive that snapshot.
                  </p>
                  <div className="mt-3 flex gap-2">
                    <button
                      className="btn-primary"
                      disabled={!!pending}
                      onClick={() => promote(releaseRequested)}
                    >
                      Confirm promotion
                    </button>
                    <button
                      className="btn-secondary"
                      disabled={!!pending}
                      onClick={() => setReleaseRequested(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : null}
              {dirty ? (
                <p className="text-xs text-(--muted-foreground)">
                  Save or discard your draft before promoting a saved version.
                </p>
              ) : null}
            </section>
          ) : null}
          {tab === "settings" && prompt ? (
            <section className={`${panelClass} space-y-3`}>
              <h2 className="flex items-center gap-2 text-lg font-semibold">
                <Archive className="size-4" />
                {prompt.archived ? "Archived" : "Archive prompt"}
              </h2>
              <p className="text-sm text-(--muted-foreground)">
                Archiving removes the prompt from SDK fetches and the active
                list. Its saved snapshots remain available for reproducibility.
              </p>
              {prompt.archived ? (
                <button
                  className="btn-secondary"
                  disabled={!!pending}
                  onClick={() =>
                    action("restore", async () => {
                      setPrompt(
                        await updatePrompt(prompt.id, { archived: false }),
                      );
                      setNotice("Prompt restored.");
                      await refresh();
                    })
                  }
                >
                  Restore prompt
                </button>
              ) : archiveRequested ? (
                <div className="flex gap-2">
                  <button
                    className="btn-secondary"
                    disabled={!!pending}
                    onClick={() =>
                      action("archive", async () => {
                        await archivePrompt(prompt.id);
                        setPrompt({ ...prompt, archived: true });
                        setArchiveRequested(false);
                        await refresh();
                      })
                    }
                  >
                    Confirm archive
                  </button>
                  <button
                    className="btn-secondary"
                    disabled={!!pending}
                    onClick={() => setArchiveRequested(false)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="btn-secondary"
                  disabled={!!pending}
                  onClick={() => setArchiveRequested(true)}
                >
                  Archive prompt
                </button>
              )}
            </section>
          ) : null}
        </div>
      </div>
      {tab === "integrate" && prompt && (
        <section className={`${panelClass} space-y-4`}>
          <h2 className="text-lg font-semibold">Fetch a released prompt</h2>
          <p className="text-sm text-(--muted-foreground)">
            Create a read-only project key in{" "}
            <Link href="/settings" className="underline">
              Settings
            </Link>
            . Promote a saved version to production before fetching it. Store
            the key in an environment variable.
          </p>
          <h3 className="text-sm font-semibold">Python SDK</h3>
          <pre className="overflow-auto rounded-xl bg-(--surface-2) p-4 text-xs">
            <code>{`from llmforge import LLMForge\n\nwith LLMForge() as forge:\n    prompt = forge.get_prompt(${JSON.stringify(prompt.name)}, label="production")\n    print(prompt.compile(**${JSON.stringify(Object.fromEntries((selected?.variables ?? []).map((variable) => [variable, "example"])))})`}</code>
          </pre>
          <h3 className="text-sm font-semibold">HTTP</h3>
          <pre className="overflow-auto rounded-xl bg-(--surface-2) p-4 text-xs">
            <code>{`curl -H "Authorization: Bearer $LLMFORGE_API_KEY" "${getApiBaseUrl()}/sdk/prompts/${encodeURIComponent(prompt.name)}?label=production"`}</code>
          </pre>
        </section>
      )}
    </div>
  );
}
