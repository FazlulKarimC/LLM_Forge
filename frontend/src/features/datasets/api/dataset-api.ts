import type { components } from "@/lib/generated/api-schema";
import { fetchAPI } from "@/lib/api-client";
export type DatasetCase = Omit<components["schemas"]["DatasetCase"], "name" | "expected_output"> & {
  name: string;
  expected_output: string | null;
};
export type Dataset = components["schemas"]["DatasetResponse"];
export type DatasetRevision = Omit<components["schemas"]["RevisionResponse"], "cases"> & { cases: DatasetCase[] };
export type DatasetDetail = { dataset: Dataset; revision: DatasetRevision };
const json = (method: string, body: unknown) => ({ method, body: JSON.stringify(body) });
export const listDatasets = (
  archived = false,
  offset = 0,
  signal?: AbortSignal,
) =>
  fetchAPI<{ items: Dataset[]; total: number }>(
    `/datasets?archived=${archived}&offset=${offset}&limit=50`,
    { signal },
  );
export const getDataset = (id: string, signal?: AbortSignal) =>
  fetchAPI<DatasetDetail>(`/datasets/${id}`, { signal });
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
export const listRevisions = (id: string, offset = 0, signal?: AbortSignal) =>
  fetchAPI<DatasetRevision[]>(
    `/datasets/${id}/revisions?offset=${offset}&limit=50`,
    { signal },
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
