import type { components } from "@/lib/generated/api-schema";
import { fetchAPI } from "@/lib/api-client";
import type { Provider } from "@/lib/prompt-api";
import {
  selectionPayload,
  type EvaluatorSelection,
  type EvaluatorSnapshot,
} from "@/lib/evaluator-api";

export * from "@/features/datasets/api/dataset-api";
export type Assertion = Required<components["schemas"]["Assertion"]>;
export type Judge = Required<components["schemas"]["JudgeConfig"]>;
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
  evaluators?: EvaluatorSelection[];
};
type RunWire = components["schemas"]["EvaluationRunResponse"];
export type EvaluationRun = Omit<RunWire, "config" | "source_run_id" | "updated_at"> & {
  id: string;
  prompt_version_id: string;
  dataset_revision_id: string;
  source_run_id?: string | null;
  config: {
    provider: Provider | "external";
    source?: "sdk_submission" | "score_only";
    evaluators?: EvaluatorSnapshot[];
    call_budget?: {
      generation: number;
      judging: number;
      total: number;
      maximum: number;
    };
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
export type CaseResult = Omit<components["schemas"]["CaseResponse"], "name" | "expected_output" | "checks"> & {
  name: string;
  expected_output: string | null;
  checks: (components["schemas"]["CheckResponse"] & {
    score?: number;
    name?: string;
    value?: string | number | boolean;
    data_type?: string;
    required?: boolean;
    evaluator_name?: string;
    evaluator_version?: number;
    evaluator_version_id?: string;
    assignment?: string;
    error?: string;
    latency_ms?: number;
    tokens_input?: number | null;
    tokens_output?: number | null;
  })[];
};
export type ScoreSummary = components["schemas"]["ScoreSummaryResponse"];
export type EvaluationDetail = {
  run: EvaluationRun;
  results: CaseResult[];
  score_summary?: ScoreSummary[];
};
export type DemoExamples = {
  prompt_id: string;
  prompt_version_id: string;
  dataset_id: string;
  dataset_revision_id: string;
};
const json = (method: string, body: unknown) => ({
  method,
  body: JSON.stringify(body),
});
export const startEvaluation = (request: EvaluationRequest, idempotencyKey?: string) =>
  fetchAPI<EvaluationRun>(
    "/evaluations",
    { ...json("POST", {
      ...request,
      ...(request.evaluators
        ? { evaluators: selectionPayload(request.evaluators) }
        : {}),
    }), headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined },
  );
export const scoreSavedOutputs = (
  runId: string,
  evaluators: EvaluatorSelection[],
  idempotencyKey?: string,
) =>
  fetchAPI<EvaluationRun>(
    `/evaluations/${runId}/score`,
    { ...json("POST", { evaluators: selectionPayload(evaluators) }), headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined },
  );
export const createDemoExamples = () =>
  fetchAPI<DemoExamples>("/demo/setup", { method: "POST" });
export const listEvaluations = (
  offset = 0,
  signal?: AbortSignal,
  filters: {
    dataset_revision_id?: string;
    search?: string;
    status?: string;
  } = {},
) =>
  fetchAPI<{ items: EvaluationRun[]; total: number }>(
    `/evaluations?${new URLSearchParams({ offset: String(offset), limit: "50", ...filters })}`,
    { signal },
  );
export const getEvaluation = (id: string, signal?: AbortSignal) =>
  fetchAPI<EvaluationDetail>(`/evaluations/${id}`, { signal });
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

export type EvaluationSummary = EvaluationDetail & {
  recovery: components["schemas"]["RecoveryResponse"];
  usage: components["schemas"]["EvaluationUsage"];
};
export type EvaluationCasePage = components["schemas"]["CasePageResponse"];
export const getEvaluationSummary = (id: string, signal?: AbortSignal) => fetchAPI<EvaluationSummary>(`/evaluations/${id}/summary`, { signal });
export const getEvaluationCases = (id: string, params: { offset: number; limit: number; filter: string; reference_id?: string }, signal?: AbortSignal) => {
  const query = new URLSearchParams({ offset: String(params.offset), limit: String(params.limit), filter: params.filter });
  if (params.reference_id) query.set("reference_id", params.reference_id);
  return fetchAPI<EvaluationCasePage>(`/evaluations/${id}/cases?${query}`, { signal });
};
export const getEvaluationCase = (id: string, index: number, signal?: AbortSignal) => fetchAPI<CaseResult>(`/evaluations/${id}/cases/${index}`, { signal });
export const resumeEvaluation = (id: string, credentials: { api_key?: string; judge_key?: string; evaluator_keys: Record<string, string> }, idempotencyKey?: string) => fetchAPI<EvaluationRun>(`/evaluations/${id}/resume`, {
  method: "POST", body: JSON.stringify(credentials), headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined,
});
