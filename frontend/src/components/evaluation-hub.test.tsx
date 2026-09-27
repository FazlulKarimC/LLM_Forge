import {
  fireEvent,
  render,
  screen,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetsWorkbench } from "./datasets-workbench";
import { EvaluationsWorkbench } from "./evaluations-workbench";

const mocks = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  getDataset: vi.fn(),
  createDataset: vi.fn(),
  updateDataset: vi.fn(),
  listRevisions: vi.fn(),
  saveRevision: vi.fn(),
  importDataset: vi.fn(),
  listEvaluations: vi.fn(),
  getEvaluation: vi.fn(),
  startEvaluation: vi.fn(),
  cancelEvaluation: vi.fn(),
  listPrompts: vi.fn(),
  getVersions: vi.fn(),
}));
vi.mock("@/lib/evaluation-api", async (original) => ({
  ...(await original<object>()),
  ...mocks,
}));
vi.mock("@/lib/prompt-api", async (original) => ({
  ...(await original<object>()),
  listPrompts: mocks.listPrompts,
  getVersions: mocks.getVersions,
}));
const cases = [
  { inputs: { query: "Hello" }, expected_output: "Hello", name: "Greeting" },
];
const dataset = {
  id: "dataset",
  name: "Greetings",
  description: "",
  latest_version: 1,
  archived: false,
};
const revision = { id: "revision", dataset_id: "dataset", version: 1, cases };
const run = {
  id: "run",
  prompt_version_id: "version",
  dataset_revision_id: "revision",
  config: {
    prompt_name: "Echo",
    prompt_version: 1,
    dataset_name: "Greetings",
    dataset_version: 1,
    model: "demo-model",
    provider: "mock",
    is_mock: true,
  },
  status: "completed",
  total: 1,
  completed: 1,
  passed: 1,
  errors: 0,
  created_at: "2026-09-27T00:00:00Z",
  error: null,
};
const result = {
  case_index: 0,
  ...cases[0],
  output: "Hello",
  passed: true,
  error: null,
  checks: [{ kind: "exact_match", passed: true, reason: "Passed" }],
  latency_ms: 1,
  tokens_input: null,
  tokens_output: null,
};
function mount(element: React.ReactNode) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({
          defaultOptions: {
            queries: { retry: false },
            mutations: { retry: false },
          },
        })
      }
    >
      {element}
    </QueryClientProvider>,
  );
}
afterEach(cleanup);
beforeEach(() => {
  vi.resetAllMocks();
  mocks.listDatasets.mockResolvedValue({ items: [dataset], total: 1 });
  mocks.getDataset.mockResolvedValue({ dataset, revision });
  mocks.createDataset.mockResolvedValue({ dataset, revision });
  mocks.updateDataset.mockResolvedValue(dataset);
  mocks.listRevisions.mockResolvedValue([revision]);
  mocks.listPrompts.mockResolvedValue({
    items: [{ id: "prompt", name: "Echo" }],
    total: 1,
  });
  mocks.getVersions.mockResolvedValue([
    { id: "version", version: 1, variables: ["query"] },
  ]);
  mocks.listEvaluations.mockResolvedValue({ items: [run], total: 1 });
  mocks.getEvaluation.mockResolvedValue({ run, results: [result] });
  mocks.startEvaluation.mockResolvedValue(run);
});

describe("dataset editor", () => {
  it("restores old cases as a new revision using the latest base version", async () => {
    const latest = {
      ...revision,
      id: "revision2",
      version: 2,
      cases: [{ ...cases[0], expected_output: "Hi" }],
    };
    mocks.getDataset.mockResolvedValue({
      dataset: { ...dataset, latest_version: 2 },
      revision: latest,
    });
    mocks.listRevisions.mockResolvedValue([latest, revision]);
    mocks.saveRevision.mockResolvedValue({
      ...revision,
      id: "revision3",
      version: 3,
    });
    mocks.updateDataset.mockResolvedValue({ ...dataset, latest_version: 3 });
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    fireEvent.click(await screen.findByText("v1 · 1 cases"));
    fireEvent.click(screen.getByText("Save changes"));
    await waitFor(() =>
      expect(mocks.saveRevision).toHaveBeenCalledWith("dataset", cases, 2),
    );
    expect(await screen.findByText(/Dataset saved/)).toBeInTheDocument();
  });
  it("creates a dataset from editable cases", async () => {
    mount(<DatasetsWorkbench />);
    fireEvent.click(screen.getByText("New dataset"));
    fireEvent.change(screen.getByLabelText("Dataset name"), {
      target: { value: "New cases" },
    });
    fireEvent.change(screen.getByLabelText("Dataset cases"), {
      target: { value: JSON.stringify(cases) },
    });
    fireEvent.click(screen.getByText("Create dataset"));
    await waitFor(() =>
      expect(mocks.createDataset).toHaveBeenCalledWith("New cases", "", cases),
    );
    expect(await screen.findByText(/Dataset saved/)).toBeInTheDocument();
  });
  it("retains a draft after a stale revision conflict and reloads explicitly", async () => {
    mocks.saveRevision.mockRejectedValue(new Error("Dataset changed; reload"));
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    const revised = [{ ...cases[0], expected_output: "Hi" }];
    fireEvent.change(await screen.findByLabelText("Dataset cases"), {
      target: { value: JSON.stringify(revised) },
    });
    fireEvent.click(screen.getByText("Save changes"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Dataset changed",
    );
    expect(mocks.saveRevision).toHaveBeenCalledWith("dataset", revised, 1);
    expect(screen.getByLabelText("Dataset cases")).toHaveValue(
      JSON.stringify(revised),
    );
    fireEvent.click(screen.getByText("Reload latest / discard draft"));
    await waitFor(() =>
      expect(screen.getByLabelText("Dataset cases")).toHaveValue(
        JSON.stringify(cases, null, 2),
      ),
    );
  });
});

describe("evaluation UI", () => {
  it("starts a live run with saved IDs, clears the key, and displays results", async () => {
    mount(<EvaluationsWorkbench />);
    await screen.findByRole("option", { name: "Echo" });
    fireEvent.change(screen.getByLabelText("Evaluation prompt"), {
      target: { value: "prompt" },
    });
    await screen.findByRole("option", { name: /v1 · query/ });
    fireEvent.change(screen.getByLabelText("Prompt version"), {
      target: { value: "version" },
    });
    fireEvent.change(screen.getByLabelText("Evaluation dataset"), {
      target: { value: "dataset" },
    });
    await screen.findByRole("option", { name: /v1 · 1 cases/ });
    fireEvent.change(screen.getByLabelText("Dataset revision"), {
      target: { value: "revision" },
    });
    fireEvent.change(screen.getByLabelText("Evaluation provider"), {
      target: { value: "groq" },
    });
    fireEvent.change(screen.getByLabelText("Evaluation API key"), {
      target: { value: "secret" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start evaluation" }));
    await waitFor(() =>
      expect(mocks.startEvaluation).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt_version_id: "version",
          dataset_revision_id: "revision",
          provider: "groq",
          api_key: "secret",
          assertions: [{ kind: "exact_match", value: "", path: "" }],
        }),
      ),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Evaluation API key")).toHaveValue(""),
    );
    expect(await screen.findByText(/1\/1 processed/)).toBeInTheDocument();
    expect(screen.getByText("✓ exact_match: Passed")).toBeInTheDocument();
  });
  it("keeps results visible and warns when comparison datasets differ", async () => {
    const other = { ...run, id: "other", dataset_revision_id: "different" };
    mocks.listEvaluations.mockResolvedValue({ items: [run, other], total: 2 });
    mocks.getEvaluation.mockImplementation((id) =>
      Promise.resolve({ run: id === "other" ? other : run, results: [result] }),
    );
    mount(<EvaluationsWorkbench />);
    await waitFor(() =>
      expect(
        screen.getByLabelText("View evaluation run").querySelectorAll("option"),
      ).toHaveLength(3),
    );
    fireEvent.change(screen.getByLabelText("View evaluation run"), {
      target: { value: "run" },
    });
    fireEvent.change(screen.getByLabelText("Compare evaluation run"), {
      target: { value: "other" },
    });
    expect(
      await screen.findByText(/different dataset revisions/),
    ).toBeInTheDocument();
    expect(screen.getByText("✓ exact_match: Passed")).toBeInTheDocument();
  });
});
