import { beforeEach, describe, expect, it } from "vitest";

import {
  getStoredCustomProviderCredentials,
  persistCustomProviderCredentials,
  resolveRunExperimentCredentials,
} from "@/lib/api";

describe("custom provider credential helpers", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("removes legacy credentials from both storage locations", () => {
    window.sessionStorage.setItem("customLLMSettings", JSON.stringify({model: {apiKey: "sentinel"}}));
    window.localStorage.setItem("customApiKey", "sentinel");
    persistCustomProviderCredentials();
    expect(getStoredCustomProviderCredentials()).toEqual({});
    expect(window.sessionStorage.getItem("customLLMSettings")).toBeNull();
    expect(window.localStorage.getItem("customApiKey")).toBeNull();
  });
  it("requires fresh custom endpoint input for each run", () => {
    expect(() => resolveRunExperimentCredentials({provider: "custom"})).toThrow(/Enter endpoint credentials/);
    expect(resolveRunExperimentCredentials({provider: "custom"}, {customBaseUrl: "http://localhost/v1", customApiKey: "sentinel"})).toEqual({customBaseUrl: "http://localhost/v1", customApiKey: "sentinel"});
    expect(window.sessionStorage.length).toBe(0);
  });
});
