"use client";
import { useQuery } from "@tanstack/react-query";
import { getEvaluationSummary, isActive } from "@/lib/evaluation-api";
import { scopedKey } from "@/features/shared/query-keys";

export function useEvaluationSummary(runId: string) {
  return useQuery({
    queryKey: scopedKey(["evaluation", runId]),
    queryFn: ({ signal }) => getEvaluationSummary(runId, signal),
    enabled: !!runId,
    refetchInterval: query => query.state.data && isActive(query.state.data.run) ? 2500 : false,
  });
}
