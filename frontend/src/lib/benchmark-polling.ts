export function benchmarkPollDelay(startedAt?: string): number | false {
  if (typeof document !== "undefined" && document.visibilityState === "hidden") return false;
  const elapsed = startedAt ? Date.now() - Date.parse(startedAt) : 0;
  return elapsed < 60_000 ? 3_000 : elapsed < 300_000 ? 10_000 : 30_000;
}
