import { fetchAPI } from "./api-client";

export type TemplateFormat = "mustache" | "fstring";
export type ReleaseLabel = "staging" | "production";
export type Prompt = {
  id: string;
  name: string;
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
) =>
  fetchAPI<{ items: Prompt[]; total: number }>(
    `/prompt-library?${new URLSearchParams({ search, archived: String(archived), offset: String(offset) })}`,
  );
export const getPrompt = (id: string) =>
  fetchAPI<PromptDetail>(`/prompt-library/${id}`);
export const getVersions = (id: string, offset = 0) =>
  fetchAPI<PromptVersion[]>(
    `/prompt-library/${id}/versions?offset=${offset}&limit=50`,
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
  changes: Partial<Pick<Prompt, "name" | "description" | "archived">>,
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
export const createProjectKey = (name: string, evaluations = false) =>
  fetchAPI<ProjectKey & { secret: string }>(
    "/project-keys",
    json("POST", {
      name,
      ...(evaluations ? { scopes: ["prompts:read", "evaluations:write"] } : {}),
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
