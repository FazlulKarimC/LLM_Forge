"use client";
import { useState } from "react";
import { parsePromptConfig, type PromptDetail, type PromptVersion, type ReleaseLabel, type TemplateFormat, type PromptType, type ChatMessage } from "@/lib/prompt-api";
export const DEFAULT_TEMPLATE =
  "Answer the following question clearly and concisely.\n\nQuestion: {{query}}\nAnswer:";
export function usePromptEditor(initial?: PromptDetail) {
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
  const [promptType, setPromptType] = useState<PromptType>(
    initial?.version.prompt_type ?? "text",
  );
  const [messages, setMessages] = useState<ChatMessage[]>(
    initial?.version.messages ?? [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "{{query}}" },
    ],
  );
  const [configText, setConfigText] = useState(
    JSON.stringify(initial?.version.config ?? {}, null, 2),
  );
  const [tagsText, setTagsText] = useState(
    (initial?.prompt.tags ?? []).join(", "),
  );
  const [customLabel, setCustomLabel] = useState("");
  let config: Record<string, unknown> = {};
  let configError: string | null = null;
  try {
    config = parsePromptConfig(configText);
  } catch {
    configError = "Configuration must be a valid JSON object.";
  }
  const [widePlayground, setWidePlayground] = useState(false);
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
  return { prompt, setPrompt, selected, setSelected, name, setName, description, setDescription, template, setTemplate, format, setFormat, promptType, setPromptType, messages, setMessages, configText, setConfigText, tagsText, setTagsText, customLabel, setCustomLabel, widePlayground, setWidePlayground, notes, setNotes, pending, setPending, error, setError, notice, setNotice, archiveRequested, setArchiveRequested, releaseRequested, setReleaseRequested, tab, setTab, config, configError };
}
