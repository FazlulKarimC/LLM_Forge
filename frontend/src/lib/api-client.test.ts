import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, fetchWithHandling } from "./api-client";
import { setApiContext } from "./request-context";

vi.mock("@sentry/nextjs", () => ({ addBreadcrumb: vi.fn() }));

beforeEach(() => setApiContext(null));
afterEach(() => vi.unstubAllGlobals());

describe("read request transport", () => {
  it("does not retry a permanent authentication error", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ message: "Sign in" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    await expect(fetchWithHandling("http://localhost:8000/api/v1/prompts"))
      .rejects.toMatchObject({ statusCode: 401 } satisfies Partial<ApiError>);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("retries a transient server response once", async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response("Unavailable", { status: 503 }))
      .mockResolvedValueOnce(new Response("OK", { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const response = await fetchWithHandling("http://localhost:8000/api/v1/prompts");
    expect(response.status).toBe(200);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("stops a pending request when its caller aborts", async () => {
    let markStarted: () => void = () => undefined;
    const started = new Promise<void>((resolve) => { markStarted = resolve; });
    const fetch = vi.fn((_url: string, options: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        markStarted();
        options.signal?.addEventListener("abort", () =>
          reject(new DOMException("Aborted", "AbortError")));
      }));
    vi.stubGlobal("fetch", fetch);
    const controller = new AbortController();
    const pending = fetchWithHandling(
      "http://localhost:8000/api/v1/prompts",
      { signal: controller.signal },
    );
    await started;
    controller.abort();
    await expect(pending).rejects.toHaveProperty("name", "AbortError");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
