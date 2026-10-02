import { fetchAPI } from "./api-client";

export type TemplateFormat = "mustache" | "fstring";
export type ReleaseLabel = string;
export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};
export type PromptType = "text" | "chat";
export type Prompt = {
  id: string;
  name: string;
  prompt_type?: PromptType;
  tags?: string[];
  description: string;
  archived: boolean;
  latest_version: number;
  created_at: string;
  updated_at: string;
  labels: {
    label: ReleaseLabel;
    version_id: string;
    version: number;
    updated_at: string;
  }[];
};
export type PromptVersion = {
  id: string;
  prompt_id: string;
  name: string;
  template_text: string;
  prompt_type?: PromptType;
  messages?: ChatMessage[];
  config?: Record<string, unknown>;
  created_by?: string;
  template_format: TemplateFormat;
  version: number;
  sha256_hash: string;
  parent_id: string | null;
  description: string | null;
  created_at: string;
  variables: string[];
};
export type PromptDetail = { prompt: Prompt; version: PromptVersion };
export type VersionDraft = {
  template_text: string;
  template_format: TemplateFormat;
  description: string;
  base_version?: number;
  prompt_type?: PromptType;
  messages?: ChatMessage[];
  config?: Record<string, unknown>;
  tags?: string[];
};
export type Provider = "mock" | "groq" | "openrouter" | "openai";
export type PlaygroundResult = {
  output: string;
  compiled_prompt: string;
  provider: Provider;
  model: string;
  is_mock: boolean;
  latency_ms: number;
  tokens_input: number | null;
  tokens_output: number | null;
  finish_reason: string;
};
export type ProjectKey = {
  id: string;
  name: string;
  prefix: string;
  scope: string;
  scopes?: string[];
  created_at: string;
  revoked_at: string | null;
};

const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
});
export const listPrompts = (
  search: string,
  archived: boolean,
  offset: number,
  signal?: AbortSignal,
  filters?: { tag?: string; folder?: string; label?: string },
) =>
  fetchAPI<{ items: Prompt[]; total: number }>(
    `/prompt-library?${new URLSearchParams({ search, archived: String(archived), offset: String(offset), ...filters })}`,
    { signal },
  );
export const getPrompt = (id: string, signal?: AbortSignal) =>
  fetchAPI<PromptDetail>(`/prompt-library/${id}`, { signal });
export const getVersions = (id: string, offset = 0, signal?: AbortSignal) =>
  fetchAPI<PromptVersion[]>(
    `/prompt-library/${id}/versions?offset=${offset}&limit=50`,
    { signal },
  );
export const createPrompt = (name: string, draft: VersionDraft) =>
  fetchAPI<PromptDetail>("/prompt-library", json("POST", { name, ...draft }));
export const saveVersion = (id: string, draft: VersionDraft) =>
  fetchAPI<PromptVersion>(
    `/prompt-library/${id}/versions`,
    json("POST", draft),
  );
export const updatePrompt = (
  id: string,
  changes: Partial<Pick<Prompt, "name" | "description" | "archived" | "tags">>,
) => fetchAPI<Prompt>(`/prompt-library/${id}`, json("PATCH", changes));
export const archivePrompt = (id: string) =>
  fetchAPI<void>(`/prompt-library/${id}`, { method: "DELETE" });
export const promoteVersion = (
  id: string,
  label: ReleaseLabel,
  versionId: string,
) =>
  fetchAPI<Prompt>(
    `/prompt-library/${id}/labels/${label}`,
    json("PUT", { version_id: versionId }),
  );
export const removeLabel = (id: string, label: ReleaseLabel) =>
  fetchAPI<void>(`/prompt-library/${id}/labels/${label}`, { method: "DELETE" });
export const compileDraft = (
  draft: VersionDraft,
  variables: Record<string, string>,
) =>
  fetchAPI<{ compiled_prompt: string }>(
    "/prompt-library/compile",
    json("POST", { ...draft, variables }),
  );
export const runPlayground = (
  draft: VersionDraft,
  variables: Record<string, string>,
  provider: Provider,
  model: string,
  apiKey: string,
  temperature: number,
  maxTokens: number,
  signal: AbortSignal,
) =>
  fetchAPI<PlaygroundResult>(
    "/prompt-library/playground",
    {
      ...json("POST", {
        ...draft,
        variables,
        provider,
        model,
        ...(provider !== "mock" ? { api_key: apiKey } : {}),
        temperature,
        max_tokens: maxTokens,
      }),
      signal,
    },
    { timeoutMs: 35_000, maxRetries: 0 },
  );
export const listProjectKeys = () => fetchAPI<ProjectKey[]>("/project-keys");
export const createProjectKey = (
  name: string,
  evaluations = false,
  promptWrites = false,
) =>
  fetchAPI<ProjectKey & { secret: string }>(
    "/project-keys",
    json("POST", {
      name,
      ...(evaluations || promptWrites
        ? {
            scopes: [
              "prompts:read",
              ...(evaluations ? ["evaluations:write"] : []),
              ...(promptWrites ? ["prompts:write"] : []),
            ],
          }
        : {}),
    }),
  );
export const revokeProjectKey = (id: string) =>
  fetchAPI<void>(`/project-keys/${id}`, { method: "DELETE" });

// Server validation remains authoritative; this supplies inputs while typing.
export function draftVariables(
  template: string,
  format: TemplateFormat,
): string[] {
  // Consume escaped braces before placeholders. In {{{query}}}, the outer
  // braces are escaped and the inner {query} remains a real variable.
  const pattern =
    format === "mustache"
      ? /{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}/g
      : /{{|}}|{([A-Za-z_][A-Za-z0-9_]*)}/g;
  return [
    ...new Set(
      Array.from(template.matchAll(pattern), (match) => match[1]).filter(
        (name): name is string => !!name,
      ),
    ),
  ].sort();
}

export function promptDraftVariables(draft: VersionDraft): string[] {
  return draft.prompt_type === "chat"
    ? [
        ...new Set(
          (draft.messages ?? []).flatMap((message) =>
            draftVariables(message.content, draft.template_format),
          ),
        ),
      ].sort()
    : draftVariables(draft.template_text, draft.template_format);
}

export function canonicalJSON(value: unknown): string {
  return JSON.stringify(value, (_key, item) =>
    item && typeof item === "object" && !Array.isArray(item)
      ? Object.fromEntries(
          Object.keys(item)
            .sort()
            .map((key) => [key, item[key]]),
        )
      : item,
  );
}

export function parsePromptConfig(text: string): Record<string, unknown> {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Configuration must be a JSON object.");
  JSON.stringify(value, (_key, item) => {
    if (typeof item === "number" && !Number.isFinite(item))
      throw new Error("Configuration numbers must be finite.");
    return item;
  });
  return value as Record<string, unknown>;
}
