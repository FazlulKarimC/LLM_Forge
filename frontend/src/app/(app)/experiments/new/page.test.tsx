import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import NewExperimentPage from "@/app/(app)/experiments/new/page";

const push = vi.fn();
const apiMocks = vi.hoisted(() => ({
  createExperiment: vi.fn(),
  getAvailableModels: vi.fn(),
  runExperiment: vi.fn(),
  resolveRunExperimentCredentials: vi.fn(() => ({})),
  persistCustomProviderCredentials: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    createExperiment: apiMocks.createExperiment,
    getAvailableModels: apiMocks.getAvailableModels,
    runExperiment: apiMocks.runExperiment,
    resolveRunExperimentCredentials: apiMocks.resolveRunExperimentCredentials,
    persistCustomProviderCredentials: apiMocks.persistCustomProviderCredentials,
  };
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <NewExperimentPage />
    </QueryClientProvider>,
  );
}

describe("NewExperimentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.getAvailableModels.mockResolvedValue({
      models: [
        {
          value: "meta-llama/Llama-3.2-1B-Instruct",
          label: "Llama 3.2 (1B)",
          description: "Fast, efficient default",
        },
      ],
    });
    apiMocks.createExperiment.mockResolvedValue({ id: "exp-1" });
    apiMocks.runExperiment.mockResolvedValue({ id: "exp-1" });
  });

  it("validates an empty experiment name", async () => {
    const { container } = renderPage();

    fireEvent.submit(container.querySelector("form")!);

    expect(await screen.findByText("Experiment name is required.")).toBeInTheDocument();
    expect(apiMocks.createExperiment).not.toHaveBeenCalled();
  });

  it("enforces backend-aligned cache bounds", async () => {
    const { container } = renderPage();

    fireEvent.change(screen.getByLabelText(/experiment name/i), { target: { value: "Cache bounds" } });
    fireEvent.click(screen.getByLabelText(/enable caching/i));
    fireEvent.change(screen.getByLabelText(/cache size/i), { target: { value: "10" } });
    fireEvent.submit(container.querySelector("form")!);

    expect(await screen.findByText("Cache size must be between 16 and 2,048.")).toBeInTheDocument();
    expect(apiMocks.createExperiment).not.toHaveBeenCalled();
  });

  it("shows custom provider fields when the dev-only endpoint is selected", async () => {
    renderPage();

    fireEvent.change(screen.getByLabelText(/^provider$/i), { target: { value: "custom" } });

    await waitFor(() => {
      expect(screen.getByLabelText(/base url/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/api key/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/model id/i)).toBeInTheDocument();
    });
  });

  it("submits strict comparison routing by default for auto provider experiments", async () => {
    const { container } = renderPage();

    expect(screen.getByLabelText(/strict comparison mode/i)).toBeChecked();

    fireEvent.change(screen.getByLabelText(/experiment name/i), {
      target: { value: "Strict routing" },
    });
    fireEvent.submit(container.querySelector("form")!);

    await waitFor(() => expect(apiMocks.createExperiment).toHaveBeenCalledTimes(1));

    const request = apiMocks.createExperiment.mock.calls[0][0];
    expect(request.config.provider).toBe("auto");
    expect(request.config.routing).toMatchObject({
      policy: "fallback_chain",
      epsilon: 0.15,
      exploration_window: 10,
      strict_comparison: true,
    });
  });
});
