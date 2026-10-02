import { fetchAPI } from "./api-client";
import type { Assertion, CaseResult } from "./evaluation-api";

export type ScoreDefinition = {
  name: string;
  data_type: "boolean" | "numeric" | "categorical";
  minimum: number;
  maximum: number;
  threshold: number;
  categories: string[];
  passing_categories: string[];
};
export type EvaluatorDefinition = {
  kind: "builtin" | "llm_judge";
  assertion: Assertion | null;
  provider: "groq" | "openrouter" | "openai";
  model: string;
  rubric: string;
  outputs: ScoreDefinition[];
  mapping: Record<string, "inputs" | "output" | "expected_output">;
};
export type EvaluatorVersion = {
  id: string;
  evaluator_id: string;
  version: number;
  definition: EvaluatorDefinition;
  notes: string;
};
export type Evaluator = {
  id: string;
  name: string;
  description: string;
  archived: boolean;
  latest_version: number;
  latest: EvaluatorVersion;
};
export type EvaluatorSelection = {
  version_id: string;
  required: boolean;
  api_key?: string;
  kind?: "builtin" | "llm_judge";
  name?: string;
  version?: number;
};
export const selectionPayload = (items: EvaluatorSelection[]) =>
  items.map(({ version_id, required, api_key }) => ({
    version_id,
    required,
    ...(api_key ? { api_key } : {}),
  }));
export type EvaluatorSnapshot = {
  version_id: string;
  name: string;
  version: number;
  required: boolean;
  definition: EvaluatorDefinition;
};
export const defaultScore = (): ScoreDefinition => ({
  name: "correctness",
  data_type: "boolean",
  minimum: 0,
  maximum: 1,
  threshold: 0.7,
  categories: [],
  passing_categories: [],
});
export const defaultDefinition = (): EvaluatorDefinition => ({
  kind: "builtin",
  assertion: { kind: "exact_match", value: "", path: "" },
  provider: "groq",
  model: "",
  rubric: "",
  outputs: [defaultScore()],
  mapping: {
    inputs: "inputs",
    candidate: "output",
    reference: "expected_output",
  },
});
const json = (method: string, body: unknown) => ({
  method,
  body: JSON.stringify(body),
});
export const listEvaluators = (
  search = "",
  archived = false,
  offset = 0,
  signal?: AbortSignal,
) =>
  fetchAPI<{ items: Evaluator[]; total: number }>(
    `/evaluators?${new URLSearchParams({ search, archived: String(archived), offset: String(offset), limit: "50" })}`,
    { signal },
  );
export const createEvaluator = (
  name: string,
  description: string,
  definition: EvaluatorDefinition,
  notes: string,
) =>
  fetchAPI<Evaluator>(
    "/evaluators",
    json("POST", { name, description, definition, notes }),
  );
export const getEvaluator = (id: string, signal?: AbortSignal) =>
  fetchAPI<Evaluator>(`/evaluators/${id}`, { signal });
export const listEvaluatorVersions = (
  id: string,
  offset = 0,
  signal?: AbortSignal,
) =>
  fetchAPI<EvaluatorVersion[]>(
    `/evaluators/${id}/versions?offset=${offset}&limit=50`,
    { signal },
  );
export const saveEvaluatorVersion = (
  id: string,
  definition: EvaluatorDefinition,
  notes: string,
  base_version: number,
) =>
  fetchAPI<EvaluatorVersion>(
    `/evaluators/${id}/versions`,
    json("POST", { definition, notes, base_version }),
  );
export const archiveEvaluator = (id: string, archived: boolean) =>
  fetchAPI<Evaluator>(`/evaluators/${id}`, json("PATCH", { archived }));
export const testEvaluator = (
  id: string,
  sample: {
    inputs: Record<string, string>;
    output: string;
    expected_output: string | null;
    api_key?: string;
  },
) =>
  fetchAPI<{ checks: CaseResult["checks"]; preview: boolean }>(
    `/evaluators/versions/${id}/test`,
    json("POST", sample),
    { timeoutMs: 40_000, maxRetries: 0 },
  );
