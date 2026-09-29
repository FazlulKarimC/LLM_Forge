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
  push: vi.fn(),
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
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }));
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
  window.history.replaceState(null, "", "/evaluations");
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
  it("updates metadata without a revision when cases are only reformatted", async () => {
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    fireEvent.change(await screen.findByLabelText("Dataset name"), {
      target: { value: "Renamed greetings" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Advanced JSON" }));
    fireEvent.change(screen.getByLabelText("Dataset cases"), {
      target: {
        value:
          '[{"name":"Greeting","expected_output":"Hello","inputs":{"query":"Hello"}}]',
      },
    });
    fireEvent.click(screen.getByText("Save changes"));
    await waitFor(() =>
      expect(mocks.updateDataset).toHaveBeenCalledWith(
        "dataset",
        expect.objectContaining({ name: "Renamed greetings" }),
      ),
    );
    expect(mocks.saveRevision).not.toHaveBeenCalled();
  });

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
  it("can evaluate an older saved revision without creating a new one", async () => {
    const latest = { ...revision, id: "revision2", version: 2 };
    mocks.getDataset.mockResolvedValue({
      dataset: { ...dataset, latest_version: 2 },
      revision: latest,
    });
    mocks.listRevisions.mockResolvedValue([latest, revision]);
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    fireEvent.click(
      await screen.findByRole("button", { name: "v1 · 1 cases" }),
    );
    const evaluate = screen.getByRole("button", {
      name: "Evaluate this revision",
    });
    expect(evaluate).not.toBeDisabled();
    fireEvent.click(evaluate);
    expect(mocks.push).toHaveBeenCalledWith(
      "/evaluations?view=new&dataset=dataset&revision=revision",
    );
  });
  it("edits a case in the table and saves the resulting revision", async () => {
    mocks.saveRevision.mockResolvedValue({
      ...revision,
      id: "revision2",
      version: 2,
      cases: [{ ...cases[0], expected_output: "Hi" }],
    });
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    fireEvent.click(await screen.findByRole("button", { name: "Greeting" }));
    fireEvent.change(screen.getByLabelText("Reference output"), {
      target: { value: "Hi" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply case" }));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() =>
      expect(mocks.saveRevision).toHaveBeenCalledWith(
        "dataset",
        [{ ...cases[0], expected_output: "Hi" }],
        1,
      ),
    );
  });
  it("previews an import before replacing draft cases", async () => {
    const imported = [
      { name: "Farewell", inputs: { query: "Bye" }, expected_output: "Bye" },
    ];
    mocks.importDataset.mockResolvedValue({ cases: imported });
    mount(<DatasetsWorkbench />);
    fireEvent.click(await screen.findByText("Greetings"));
    await screen.findByRole("button", { name: "Greeting" });
    fireEvent.click(screen.getByText("Import CSV or JSON into this draft"));
    fireEvent.change(screen.getByLabelText("Import content"), {
      target: { value: JSON.stringify(imported) },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Validate and preview" }),
    );
    expect(
      await screen.findByText("Import preview · 1 cases"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Greeting" }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Replace draft cases" }),
    );
    expect(
      screen.getByRole("button", { name: "Farewell" }),
    ).toBeInTheDocument();
  });
  it("creates a dataset from editable cases", async () => {
    mount(<DatasetsWorkbench />);
    fireEvent.click(screen.getByText("New dataset"));
    fireEvent.change(screen.getByLabelText("Dataset name"), {
      target: { value: "New cases" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Advanced JSON" }));
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
    await screen.findByLabelText("Dataset name");
    fireEvent.click(screen.getByRole("button", { name: "Advanced JSON" }));
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
    await screen.findByText(/Loaded the latest revision/);
    fireEvent.click(
      await screen.findByRole("button", { name: "Advanced JSON" }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Dataset cases")).toHaveValue(
        JSON.stringify(cases, null, 2),
      ),
    );
  });
});

describe("evaluation UI", () => {
  it("offers per-case JSON references without a global comparison value", () => {
    mount(<EvaluationsWorkbench />);
    fireEvent.click(screen.getByRole("button", { name: "New evaluation" }));
    fireEvent.change(screen.getByLabelText("Rule 1"), {
      target: { value: "json_reference" },
    });
    expect(screen.getByLabelText("Rule 1")).toHaveValue("json_reference");
    expect(screen.queryByText("Expected JSON value")).not.toBeInTheDocument();
  });
  it("restores and updates a shareable evaluation selection", async () => {
    window.history.replaceState(
      null,
      "",
      "/evaluations?prompt=prompt&version=version&dataset=dataset&revision=revision&run=run",
    );
    mount(<EvaluationsWorkbench />);
    await screen.findByText("1/1");
    expect(window.location.search).toContain("run=run");
    fireEvent.click(screen.getByRole("button", { name: "Run history" }));
    await waitFor(() => expect(window.location.search).not.toContain("run="));
    fireEvent.click(screen.getByRole("button", { name: "New evaluation" }));
    await waitFor(() =>
      expect(screen.getByLabelText("Prompt version")).toHaveValue("version"),
    );
    expect(screen.getByLabelText("Dataset revision")).toHaveValue("revision");
    expect(window.location.search).toContain("version=version");
  });
  it("starts a live run with saved IDs, clears the key, and displays results", async () => {
    mount(<EvaluationsWorkbench />);
    fireEvent.click(screen.getByRole("button", { name: "New evaluation" }));
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
    await waitFor(() => expect(window.location.search).toContain("run=run"));
    expect(await screen.findByText("1/1")).toBeInTheDocument();
    expect(screen.getByText("✓ exact_match: Passed")).toBeInTheDocument();
  });
  it("keeps results visible and warns when comparison datasets differ", async () => {
    const other = { ...run, id: "other", dataset_revision_id: "different" };
    mocks.listEvaluations.mockResolvedValue({ items: [run, other], total: 2 });
    mocks.getEvaluation.mockImplementation((id) =>
      Promise.resolve({ run: id === "other" ? other : run, results: [result] }),
    );
    mount(<EvaluationsWorkbench />);
    await screen.findAllByRole("button", { name: "Echo v1" });
    fireEvent.click(screen.getAllByRole("button", { name: "Echo v1" })[0]);
    fireEvent.change(
      await screen.findByLabelText("Reference run for this candidate"),
      {
        target: { value: "other" },
      },
    );
    expect(
      await screen.findByText(/different dataset revisions/),
    ).toBeInTheDocument();
    expect(screen.getByText("✓ exact_match: Passed")).toBeInTheDocument();
  });
  it("shows aligned case changes against a comparable run", async () => {
    const oldRun = {
      ...run,
      id: "old",
      config: {
        ...run.config,
        assertions: [{ kind: "exact_match", value: "", path: "" }],
      },
    };
    const currentRun = { ...run, config: oldRun.config, passed: 0 };
    const failedResult = { ...result, passed: false };
    mocks.listEvaluations.mockResolvedValue({
      items: [currentRun, oldRun],
      total: 2,
    });
    mocks.getEvaluation.mockImplementation((id) =>
      Promise.resolve({
        run: id === "old" ? oldRun : currentRun,
        results: [id === "old" ? result : failedResult],
      }),
    );
    mount(<EvaluationsWorkbench />);
    await screen.findAllByRole("button", { name: "Echo v1" });
    fireEvent.click(screen.getAllByRole("button", { name: "Echo v1" })[0]);
    fireEvent.change(
      await screen.findByLabelText("Reference run for this candidate"),
      { target: { value: "old" } },
    );
    expect(
      await screen.findByText(/0 improved, 1 regressed/),
    ).toBeInTheDocument();
  });
});
