import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { testEvaluator } from "./evaluator-api";
import { scoreSavedOutputs } from "./evaluation-api";
import { setApiContext } from "./request-context";

vi.mock("@sentry/nextjs", () => ({ addBreadcrumb: vi.fn() }));
beforeEach(() => setApiContext(null));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("evaluator request contracts", () => {
  it("allows a judge preview to finish beyond the normal mutation timeout", async () => {
    vi.useFakeTimers();
    const fetch = vi.fn(
      (_url: string, options: RequestInit) =>
        new Promise<Response>((resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
          setTimeout(
            () =>
              resolve(
                new Response(JSON.stringify({ preview: true, checks: [] }), {
                  headers: { "Content-Type": "application/json" },
                }),
              ),
            20_000,
          );
        }),
    );
    vi.stubGlobal("fetch", fetch);
    const pending = testEvaluator("version", {
      inputs: {},
      output: "Hello",
      expected_output: null,
      api_key: "request-only",
    });
    await vi.advanceTimersByTimeAsync(20_000);
    await expect(pending).resolves.toMatchObject({ preview: true });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("sends only the pinned selection and request key, excluding display metadata", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ id: "scored" }), {
          headers: { "Content-Type": "application/json" },
        }),
      );
    vi.stubGlobal("fetch", fetch);
    await scoreSavedOutputs("source", [
      {
        version_id: "version",
        required: false,
        kind: "llm_judge",
        name: "Quality",
        version: 2,
        api_key: "request-only",
      },
    ]);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      evaluators: [
        { version_id: "version", required: false, api_key: "request-only" },
      ],
    });
  });
});
