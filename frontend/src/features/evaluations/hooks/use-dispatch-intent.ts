"use client";
import { useRef } from "react";
import { canonicalJSON } from "@/lib/prompt-api";

/** An uncertain transport result retains the same token for the same intent. */
export function useDispatchIntent() {
  const current = useRef<{ fingerprint: string; key: string } | null>(null);
  return {
    key(value: unknown) {
      const nonSecret = (item: unknown): unknown => Array.isArray(item) ? item.map(nonSecret) : item && typeof item === "object" ? Object.fromEntries(Object.entries(item).filter(([key]) => !["api_key", "judge_key", "evaluator_keys"].includes(key)).map(([key, value]) => [key, nonSecret(value)])) : item;
      const fingerprint = canonicalJSON(nonSecret(value));
      if (current.current?.fingerprint !== fingerprint) current.current = { fingerprint, key: crypto.randomUUID() };
      return current.current!.key;
    },
    accepted() { current.current = null; },
  };
}
