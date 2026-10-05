"use client";
import { useInfiniteQuery } from "@tanstack/react-query";
import { scopedKey } from "@/features/shared/query-keys";
import { getVersions, type Prompt } from "@/lib/prompt-api";
export function usePromptVersions(prompt: Prompt | null) {
  const versions = useInfiniteQuery({
    queryKey: scopedKey(["prompt-versions", prompt?.id]),
    enabled: !!prompt,
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) =>
      getVersions(prompt!.id, pageParam, signal),
    getNextPageParam: (lastPage, pages) =>
      lastPage.length === 50 ? pages.length * 50 : undefined,
  });
  return versions;
}
