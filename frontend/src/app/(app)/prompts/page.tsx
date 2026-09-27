"use client";
import Link from "next/link";
import { useDeferredValue, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileText, Plus, Search } from "lucide-react";
import { PageHeader } from "@/components/ui/primitives";
import { ErrorMessage, errorText, inputClass, panelClass } from "@/components/prompts/prompt-ui";
import { listPrompts } from "@/lib/prompt-api";

export default function PromptsPage() {
  const [search, setSearch] = useState("");
  const [archived, setArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const deferredSearch = useDeferredValue(search);
  const query = useQuery({ queryKey: ["prompt-library", deferredSearch, archived, offset], queryFn: () => listPrompts(deferredSearch, archived, offset) });
  return <div className="space-y-6">
    <PageHeader eyebrow="Prompt engineering" title="Prompts" description="Develop a prompt, test it with sample inputs, and release a version your application can fetch."
      actions={<Link href="/prompts/new" className="btn-primary"><Plus className="size-4" />New prompt</Link>} />
    <div className="flex flex-wrap items-end gap-4">
      <label className="min-w-48 flex-1 text-sm"><span className="inline-flex items-center gap-2"><Search className="size-4" />Search prompts</span><input className={inputClass} value={search} onChange={(event) => { setSearch(event.target.value); setOffset(0); }} placeholder="Find a prompt by name" maxLength={255} /></label>
      <label className="text-sm">Show<select className={inputClass} value={String(archived)} onChange={(event) => { setArchived(event.target.value === "true"); setOffset(0); }}><option value="false">Active prompts</option><option value="true">Archived prompts</option></select></label>
    </div>
    <ErrorMessage message={query.error ? errorText(query.error) : null} />
    {query.error ? <button className="btn-secondary" onClick={() => query.refetch()}>Try again</button> : null}
    {query.isPending ? <p role="status" className={panelClass}>Loading prompts…</p> : query.data?.items.length ? <div className="grid gap-4 lg:grid-cols-2">
      {query.data.items.map((prompt) => <Link key={prompt.id} href={`/prompts/${prompt.id}`} className={`${panelClass} transition-colors hover:border-(--primary)`}>
        <div className="flex items-start justify-between gap-3"><div className="flex min-w-0 items-center gap-3"><FileText className="size-5 shrink-0 text-(--primary)" /><h2 className="truncate text-lg font-semibold">{prompt.name}</h2></div><span className="shrink-0 rounded-lg bg-(--surface-2) px-2 py-1 text-xs">v{prompt.latest_version}</span></div>
        <p className="mt-3 line-clamp-2 text-sm text-(--muted-foreground)">{prompt.description || "No description yet"}</p>
        <div className="mt-5 flex flex-wrap items-center gap-2 text-xs">{prompt.labels.length ? prompt.labels.map((label) => <span key={label.label} className="rounded-lg border border-(--border) px-2 py-1">{label.label} · v{label.version}</span>) : <span className="text-(--muted-foreground)">No releases yet</span>}<span className="ml-auto text-(--muted-foreground)">Updated {new Date(prompt.updated_at).toLocaleDateString()}</span></div>
      </Link>)}
    </div> : !query.error ? <div className={`${panelClass} py-14 text-center`}><FileText className="mx-auto size-8 text-(--muted-foreground)" /><h2 className="mt-4 text-xl font-semibold">{search ? "No matching prompts" : archived ? "No archived prompts" : "Your first prompt starts here"}</h2><p className="mt-2 text-sm text-(--muted-foreground)">{search ? "Try another name." : "Write a reusable template and test it before releasing it."}</p>{!archived && !search ? <Link href="/prompts/new" className="btn-primary mt-6">Create a prompt</Link> : null}</div> : null}
    {query.data && query.data.total > 50 ? <div className="flex items-center justify-between gap-4 text-sm"><span>{offset + 1}–{Math.min(offset + 50, query.data.total)} of {query.data.total}</span><div className="flex gap-2"><button className="btn-secondary" disabled={offset === 0} onClick={() => setOffset(offset - 50)}>Previous</button><button className="btn-secondary" disabled={offset + 50 >= query.data.total} onClick={() => setOffset(offset + 50)}>Next</button></div></div> : null}
  </div>;
}
