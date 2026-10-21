import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, type ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import ExperimentDetailPage from "./page";

const api = vi.hoisted(() => ({ getExperiment: vi.fn(), unsetBaseline: vi.fn(), setBaseline: vi.fn() }));
vi.mock("@/lib/api", async () => ({ ...await vi.importActual<typeof import("@/lib/api")>("@/lib/api"), ...api }));
vi.mock("next/link", () => ({ default: ({ children, href }: { children: ReactNode; href: string }) => <a href={href}>{children}</a> }));
vi.mock("@/components/experiments/attempt-selector", () => ({ AttemptSelector: () => null }));
vi.mock("@/components/experiments/experiment-detail-sections", () => ({ ExperimentProfileDashboard: () => null, ExperimentResultsDashboard: () => null }));
vi.mock("@/components/ui/RegressionPanel", () => ({ RegressionPanel: () => null }));
vi.mock("@/components/ui/RoutingPanel", () => ({ RoutingPanel: () => null }));

function fixture(status: string, baseline: boolean) {
  return { id: "baseline-review", name: "Pinned historical attempt", status, is_baseline: baseline,
    current_attempt: 4, pinned_attempt: baseline ? 3 : null,
    config: { model_name: "mock-review", reasoning_method: "naive", provider: "auto", num_samples: 1 } };
}

async function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const params = Promise.resolve({ id: "baseline-review" });
  await act(async () => {
    render(<QueryClientProvider client={client}><Suspense fallback="Loading"><ExperimentDetailPage params={params} /></Suspense></QueryClientProvider>);
  });
}

beforeEach(() => {
  vi.resetAllMocks();
  api.unsetBaseline.mockResolvedValue({});
});

it.each(["failed", "running", "queued", "completed"])("keeps Unpin available for an existing baseline with a %s current attempt", async (status) => {
  api.getExperiment.mockResolvedValue(fixture(status, true));
  await show();
  expect(await screen.findByRole("button", { name: "Unpin baseline" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: "Pin as baseline" })).toBeNull();
});

it("unpins after an interrupted rerun and refreshes the baseline controls", async () => {
  api.getExperiment.mockResolvedValueOnce(fixture("failed", true)).mockResolvedValue(fixture("failed", false));
  await show();
  fireEvent.click(await screen.findByRole("button", { name: "Unpin baseline" }));
  await waitFor(() => expect(api.unsetBaseline).toHaveBeenCalledWith("baseline-review"));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Unpin baseline" })).toBeNull());
  expect(screen.queryByRole("button", { name: "Pin as baseline" })).toBeNull();
});

it.each(["failed", "running", "queued"])("does not offer a new pin for a %s non-baseline", async (status) => {
  api.getExperiment.mockResolvedValue(fixture(status, false));
  await show();
  await screen.findByRole("heading", { name: "Pinned historical attempt" });
  expect(screen.queryByRole("button", { name: "Unpin baseline" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Pin as baseline" })).toBeNull();
});

it("still offers a new pin for a completed non-baseline", async () => {
  api.getExperiment.mockResolvedValue(fixture("completed", false));
  await show();
  expect(await screen.findByRole("button", { name: "Pin as baseline" })).toBeEnabled();
});
