import { afterEach, expect, it, vi } from "vitest";
import { benchmarkPollDelay } from "./benchmark-polling";

afterEach(() => vi.restoreAllMocks());

it("backs off active work and pauses hidden tabs", () => {
  const now = Date.parse("2026-01-01T00:10:00Z");
  vi.spyOn(Date, "now").mockReturnValue(now);
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
  expect(benchmarkPollDelay(new Date(now - 10_000).toISOString())).toBe(3_000);
  expect(benchmarkPollDelay(new Date(now - 90_000).toISOString())).toBe(10_000);
  expect(benchmarkPollDelay(new Date(now - 400_000).toISOString())).toBe(30_000);
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
  expect(benchmarkPollDelay()).toBe(false);
});
