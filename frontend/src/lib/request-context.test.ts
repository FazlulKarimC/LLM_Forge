import { afterEach, describe, expect, it, vi } from "vitest";
import { getContextHeaders, setApiContext } from "@/lib/request-context";

afterEach(() => setApiContext(null));
describe("workspace request identity", () => {
  it("retrieves a fresh token and includes the selected project", async () => {
    const getToken = vi.fn().mockResolvedValueOnce("first").mockResolvedValueOnce("second");
    setApiContext({ getToken, projectId: "project-a" });
    expect(await getContextHeaders()).toEqual({ Authorization: "Bearer first", "X-Project-ID": "project-a" });
    expect(await getContextHeaders()).toEqual({ Authorization: "Bearer second", "X-Project-ID": "project-a" });
  });
  it("keeps a pending request scoped to the project at its start", async () => {
    let resolve!: (value: string) => void;
    setApiContext({ getToken: () => new Promise((r) => { resolve = r; }), projectId: "project-a" });
    const pending = getContextHeaders();
    setApiContext({ getToken: async () => "b", projectId: "project-b" });
    resolve("a");
    expect(await pending).toEqual({ Authorization: "Bearer a", "X-Project-ID": "project-a" });
    expect(await getContextHeaders()).toEqual({ Authorization: "Bearer b", "X-Project-ID": "project-b" });
  });
  it("clears identity on logout and rejects expired sessions", async () => {
    setApiContext({ getToken: async () => null, projectId: "project-a" });
    await expect(getContextHeaders()).rejects.toThrow(/session has expired/);
    setApiContext(null);
    expect(await getContextHeaders()).toEqual({});
  });
});
