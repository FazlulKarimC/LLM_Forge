import { fetchAPI } from "./api-client";
import type { Provider } from "./prompt-api";

export type DatasetCase = {
  inputs: Record<string, string>;
  expected_output: string | null;
  name: string;
};
export type Dataset = {
  id: string;
  name: string;
  description: string;
  archived: boolean;
  latest_version: number;
  created_at: string;
};
export type DatasetRevision = {
  id: string;
  dataset_id: string;
  version: number;
  cases: DatasetCase[];
  created_at: string;
};
export type DatasetDetail = { dataset: Dataset; revision: DatasetRevision };
export type Assertion = {
  kind:
    | "exact_match"
    | "contains"
    | "regex"
    | "json_valid"
    | "json_equals"
    | "json_path";
  value: string;
  path: string;
};
export type Judge = {
  provider: Exclude<Provider, "mock">;
  model: string;
  api_key: string;
  rubric: string;
  threshold: number;
};
export type EvaluationRequest = {
  prompt_version_id: string;
  dataset_revision_id: string;
  provider: Provider;
  model: string;
  api_key?: string;
  temperature: number;
  max_tokens: number;
  assertions: Assertion[];
  judge?: Judge;
};
export type EvaluationRun = {
  id: string;
  prompt_version_id: string;
  dataset_revision_id: string;
  config: {
    provider: Provider | "external";
    source?: "sdk_submission";
    metrics?: Record<string, number>;
    model: string;
    prompt_name: string;
    prompt_version: number;
    dataset_name: string;
    dataset_version: number;
    is_mock: boolean;
    judge?: Omit<Judge, "api_key">;
    assertions: Assertion[];
  };
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  total: number;
  completed: number;
  passed: number;
  errors: number;
  error: string | null;
  created_at: string;
};
export type CaseResult = {
  case_index: number;
  name: string;
  inputs: Record<string, string>;
  expected_output: string | null;
  output: string | null;
  passed: boolean;
  error: string | null;
  checks: { kind: string; passed: boolean; reason: string; score?: number }[];
  latency_ms: number | null;
  tokens_input: number | null;
  tokens_output: number | null;
};
export type EvaluationDetail = { run: EvaluationRun; results: CaseResult[] };
const json = (method: string, body: unknown) => ({
  method,
  body: JSON.stringify(body),
});
export const listDatasets = (archived = false, offset = 0) =>
  fetchAPI<{ items: Dataset[]; total: number }>(
    `/datasets?archived=${archived}&offset=${offset}&limit=50`,
  );
export const getDataset = (id: string) =>
  fetchAPI<DatasetDetail>(`/datasets/${id}`);
export const createDataset = (
  name: string,
  description: string,
  cases: DatasetCase[],
) =>
  fetchAPI<DatasetDetail>(
    "/datasets",
    json("POST", { name, description, cases }),
  );
export const updateDataset = (
  id: string,
  changes: Partial<Pick<Dataset, "name" | "description" | "archived">>,
) => fetchAPI<Dataset>(`/datasets/${id}`, json("PATCH", changes));
export const importDataset = (format: "csv" | "json", content: string) =>
  fetchAPI<{ cases: DatasetCase[] }>(
    "/datasets/import",
    json("POST", { format, content }),
  );
export const listRevisions = (id: string, offset = 0) =>
  fetchAPI<DatasetRevision[]>(
    `/datasets/${id}/revisions?offset=${offset}&limit=50`,
  );
export const saveRevision = (
  id: string,
  cases: DatasetCase[],
  baseVersion: number,
) =>
  fetchAPI<DatasetRevision>(
    `/datasets/${id}/revisions`,
    json("POST", { cases, base_version: baseVersion }),
  );
export const startEvaluation = (request: EvaluationRequest) =>
  fetchAPI<EvaluationRun>("/evaluations", json("POST", request));
export const listEvaluations = (offset = 0) =>
  fetchAPI<{ items: EvaluationRun[]; total: number }>(
    `/evaluations?offset=${offset}&limit=50`,
  );
export const getEvaluation = (id: string) =>
  fetchAPI<EvaluationDetail>(`/evaluations/${id}`);
export const cancelEvaluation = (id: string) =>
  fetchAPI<EvaluationRun>(`/evaluations/${id}/cancel`, { method: "POST" });
export const isActive = (run: EvaluationRun) =>
  run.status === "queued" || run.status === "running";

export function exportJson(filename: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
