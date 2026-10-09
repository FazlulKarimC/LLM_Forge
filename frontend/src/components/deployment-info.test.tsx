import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { DeploymentInfo } from "./deployment-info";
import { fetchAPI } from "@/lib/api-client";
vi.mock("@/lib/api-client", () => ({ fetchAPI: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); vi.unstubAllGlobals(); });

it("keeps frontend/backend identities separate", async () => {
  vi.mocked(fetchAPI).mockResolvedValue({ api_version: "0.1.0", commit: "backend-source", build_id: "backend-build" });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ commit: "frontend-source", build_id: "frontend-build" }) }));
  render(<DeploymentInfo />);
  expect(await screen.findByText("0.1.0 / backend-source / backend-build")).toBeInTheDocument();
  expect(screen.getByText("frontend-source / frontend-build")).toBeInTheDocument();
});
it("keeps unavailable metadata explicit", async () => {
  vi.mocked(fetchAPI).mockRejectedValue(new Error("Not found"));
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ commit: null, build_id: null }) }));
  render(<DeploymentInfo />);
  expect(await screen.findByText("unavailable / unknown / unknown")).toBeInTheDocument();
  expect(screen.getByText("unknown / unknown")).toBeInTheDocument();
});
