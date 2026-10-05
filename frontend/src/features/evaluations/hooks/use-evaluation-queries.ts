"use client";
import { useQuery } from "@tanstack/react-query";
import { scopedKey } from "@/features/shared/query-keys";
import { listPrompts, getVersions } from "@/lib/prompt-api";
import { listDatasets, listRevisions, listEvaluations, isActive } from "@/lib/evaluation-api";
import { useEvaluationSummary } from "./use-evaluation-summary";
export function useEvaluationQueries({ promptOffset, promptId, versionOffset, datasetOffset, datasetId, revisionOffset, runOffset, runSearch, runStatus, runId, compareId, referenceSearch, referenceOffset }: { promptOffset: number, promptId: string, versionOffset: number, datasetOffset: number, datasetId: string, revisionOffset: number, runOffset: number, runSearch: string, runStatus: string, runId: string, compareId: string, referenceSearch: string, referenceOffset: number }) {
  const prompts = useQuery({
    queryKey: scopedKey(["prompt-library", "", false, promptOffset]),
    queryFn: ({ signal }) => listPrompts("", false, promptOffset, signal),
  });
  const versions = useQuery({
    queryKey: scopedKey(["eval-versions", promptId, versionOffset]),
    queryFn: ({ signal }) => getVersions(promptId, versionOffset, signal),
    enabled: !!promptId,
  });
  const datasets = useQuery({
    queryKey: scopedKey(["datasets", false, datasetOffset]),
    queryFn: ({ signal }) => listDatasets(false, datasetOffset, signal),
  });
  const revisions = useQuery({
    queryKey: scopedKey(["dataset-history", datasetId, revisionOffset]),
    queryFn: ({ signal }) => listRevisions(datasetId, revisionOffset, signal),
    enabled: !!datasetId,
  });
  const runs = useQuery({
    queryKey: scopedKey(["evaluations", runOffset, runSearch, runStatus]),
    queryFn: ({ signal }) =>
      listEvaluations(runOffset, signal, {
        search: runSearch,
        ...(runStatus ? { status: runStatus } : {}),
      }),
    refetchInterval: (query) =>
      query.state.data?.items.some(isActive) ? 2500 : false,
  });
  const detail = useEvaluationSummary(runId);
  const comparison = useEvaluationSummary(compareId);
  const references = useQuery({
    queryKey: scopedKey([
      "evaluation-references",
      detail.data?.run.dataset_revision_id,
      referenceSearch,
      referenceOffset,
    ]),
    queryFn: ({ signal }) =>
      listEvaluations(referenceOffset, signal, {
        dataset_revision_id: detail.data!.run.dataset_revision_id,
        search: referenceSearch,
        status: "completed",
      }),
    enabled: !!detail.data,
  });
  return { prompts, versions, datasets, revisions, runs, detail, comparison, references };
}
