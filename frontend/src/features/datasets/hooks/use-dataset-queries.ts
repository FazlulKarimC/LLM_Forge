"use client";
import { useQuery } from "@tanstack/react-query";
import { scopedKey } from "@/features/shared/query-keys";
import { listDatasets, listRevisions, type DatasetDetail } from "@/lib/evaluation-api";
export function useDatasetQueries(archived: boolean, offset: number, selected: DatasetDetail | null, historyOffset: number) {
  const list = useQuery({
    queryKey: scopedKey(["datasets", archived, offset]),
    queryFn: ({ signal }) => listDatasets(archived, offset, signal),
  });
  const history = useQuery({
    queryKey: scopedKey(["dataset-history", selected?.dataset.id, historyOffset]),
    queryFn: ({ signal }) =>
      listRevisions(selected!.dataset.id, historyOffset, signal),
    enabled: !!selected,
  });
  return { list, history };
}
