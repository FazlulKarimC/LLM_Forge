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

  it("persists and restores session-scoped credentials", () => {
    persistCustomProviderCredentials("custom-model", "http://localhost:8000/v1", "secret-key");

    expect(getStoredCustomProviderCredentials("custom-model")).toEqual({
      customBaseUrl: "http://localhost:8000/v1",
      customApiKey: "secret-key",
    });
  });

  it("throws when a custom-provider run has no saved base url", () => {
    expect(() =>
      resolveRunExperimentCredentials({ provider: "custom", model_name: "missing-model" }),
    ).toThrow(/No saved custom endpoint settings/i);
  });
});
