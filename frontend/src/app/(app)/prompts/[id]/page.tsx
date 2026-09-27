"use client";
import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import { PromptWorkbench } from "@/components/prompts/prompt-workbench";
import { ErrorMessage, errorText, panelClass } from "@/components/prompts/prompt-ui";
import { getPrompt } from "@/lib/prompt-api";

export default function PromptPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const query = useQuery({ queryKey: ["prompt-detail", id], queryFn: () => getPrompt(id) });
  if (query.isPending) return <p role="status" className={panelClass}>Loading prompt…</p>;
  if (query.error) return <div className="space-y-4"><ErrorMessage message={errorText(query.error)} /><button className="btn-secondary" onClick={() => query.refetch()}>Try again</button></div>;
  return <PromptWorkbench key={query.data.version.id} initial={query.data} />;
}
