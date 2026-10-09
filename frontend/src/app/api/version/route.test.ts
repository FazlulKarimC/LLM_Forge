import { afterEach, expect, it, vi } from "vitest";
import { GET } from "./route";
afterEach(() => vi.unstubAllEnvs());

it("returns only designated frontend identity", async () => {
  vi.stubEnv("LLMFORGE_FRONTEND_COMMIT", "a".repeat(40));
  vi.stubEnv("LLMFORGE_FRONTEND_BUILD_ID", "dpl_example");
  vi.stubEnv("CLERK_SECRET_KEY", "never-expose");
  expect(await GET().json()).toEqual({ service: "llmforge-frontend", commit: "a".repeat(40), build_id: "dpl_example" });
  vi.stubEnv("LLMFORGE_FRONTEND_COMMIT", "");
  vi.stubEnv("LLMFORGE_FRONTEND_BUILD_ID", "key=secret");
  expect(await GET().json()).toEqual({ service: "llmforge-frontend", commit: null, build_id: null });
});
