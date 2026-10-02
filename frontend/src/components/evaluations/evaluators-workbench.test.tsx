import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluatorsWorkbench } from "./evaluators-workbench";
import { ScoreSavedOutputs } from "./score-saved-outputs";
import { ResultsGrid, RunSummary } from "./evaluation-results";
import { defaultDefinition } from "@/lib/evaluator-api";
import type { EvaluatorSelection } from "@/lib/evaluator-api";
import { useState } from "react";
import { EvaluatorSelectionFields } from "./evaluator-selection";
import type { EvaluationDetail, EvaluationRun } from "@/lib/evaluation-api";

const mocks = vi.hoisted(() => ({
  listEvaluators: vi.fn(),
  getEvaluator: vi.fn(),
  listEvaluatorVersions: vi.fn(),
  createEvaluator: vi.fn(),
  saveEvaluatorVersion: vi.fn(),
  archiveEvaluator: vi.fn(),
  testEvaluator: vi.fn(),
  scoreSavedOutputs: vi.fn(),
}));
vi.mock("@/lib/evaluator-api", async (original) => ({
  ...(await original<object>()),
  ...mocks,
}));
vi.mock("@/lib/evaluation-api", async (original) => ({
  ...(await original<object>()),
  scoreSavedOutputs: mocks.scoreSavedOutputs,
}));
const version = {
  id: "ev1",
  evaluator_id: "e1",
  version: 1,
  definition: defaultDefinition(),
  notes: "",
};
const evaluator = {
  id: "e1",
  name: "Correctness",
  description: "",
  archived: false,
  latest_version: 1,
  latest: version,
};
const run: EvaluationRun = {
  id: "source",
  prompt_version_id: "prompt",
  dataset_revision_id: "dataset",
  config: {
    prompt_name: "Echo",
    prompt_version: 1,
    dataset_name: "Greetings",
    dataset_version: 1,
    model: "demo",
    provider: "mock",
    is_mock: true,
    assertions: [],
  },
  status: "completed",
  total: 2,
  completed: 2,
  passed: 2,
  errors: 0,
  created_at: "2026-10-02T00:00:00Z",
  error: null,
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
  mocks.listEvaluators.mockResolvedValue({ items: [evaluator], total: 1 });
  mocks.getEvaluator.mockResolvedValue(evaluator);
  mocks.listEvaluatorVersions.mockResolvedValue([version]);
  mocks.testEvaluator.mockResolvedValue({
    preview: true,
    checks: [
      { kind: "correctness", value: true, passed: true, reason: "Passed" },
    ],
  });
});

describe("evaluator workflows", () => {
  it("retains selected names and judge credentials when the setup section remounts", async () => {
    const judgeVersion = {
      ...version,
      definition: { ...version.definition, kind: "llm_judge", assertion: null },
    };
    mocks.listEvaluators.mockResolvedValue({
      items: [{ ...evaluator, latest: judgeVersion }],
      total: 1,
    });
    mocks.listEvaluatorVersions.mockResolvedValue([judgeVersion]);
    function Setup() {
      const [shown, setShown] = useState(true);
      const [selected, setSelected] = useState<EvaluatorSelection[]>([]);
      return (
        <>
          <button onClick={() => setShown(!shown)}>Toggle setup</button>
          {shown && (
            <EvaluatorSelectionFields value={selected} onChange={setSelected} />
          )}
        </>
      );
    }
    mount(<Setup />);
    await screen.findByRole("option", { name: "Correctness" });
    fireEvent.change(screen.getByLabelText("Evaluator"), {
      target: { value: "e1" },
    });
    await screen.findByRole("option", { name: "v1 · LLM judge" });
    fireEvent.change(screen.getByLabelText("Add saved version"), {
      target: { value: "ev1" },
    });
    fireEvent.change(screen.getByLabelText(/Judge key for Correctness/), {
      target: { value: "request-only" },
    });
    fireEvent.click(screen.getByText("Toggle setup"));
    fireEvent.click(screen.getByText("Toggle setup"));
    expect(screen.getByText("Correctness v1")).toBeInTheDocument();
    expect(screen.getByLabelText(/Judge key for Correctness/)).toHaveValue(
      "request-only",
    );
  });
  it("tests the exact saved version and blocks testing an unsaved definition", async () => {
    mount(<EvaluatorsWorkbench />);
    fireEvent.click(await screen.findByRole("button", { name: "Correctness" }));
    const test = await screen.findByRole("button", { name: "Test evaluator" });
    fireEvent.click(test);
    expect(await screen.findByRole("status")).toHaveTextContent(
      "correctness: true",
    );
    expect(mocks.testEvaluator).toHaveBeenCalledWith("ev1", {
      inputs: { query: "Hello" },
      output: "Hello",
      expected_output: "Hello",
    });
    fireEvent.change(screen.getByLabelText("Check"), {
      target: { value: "contains" },
    });
    expect(test).toBeDisabled();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    mocks.saveEvaluatorVersion.mockRejectedValue(
      new Error("Evaluator changed; reload latest"),
    );
    fireEvent.change(screen.getByLabelText("Check value"), {
      target: { value: "Hello" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Save evaluator version" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Evaluator changed",
    );
    expect(screen.getByLabelText("Check value")).toHaveValue("Hello");
    expect(mocks.saveEvaluatorVersion).toHaveBeenCalledWith(
      "e1",
      expect.objectContaining({
        assertion: expect.objectContaining({
          kind: "contains",
          value: "Hello",
        }),
      }),
      "",
      1,
    );
  });

  it("starts a separate scoring run with an explicit saved version and a required gate", async () => {
    const started = vi.fn();
    mocks.scoreSavedOutputs.mockResolvedValue({
      ...run,
      id: "scored",
      source_run_id: "source",
    });
    mount(<ScoreSavedOutputs run={run} started={started} />);
    fireEvent.click(screen.getByText("Score saved outputs again"));
    await screen.findByRole("option", { name: "Correctness" });
    fireEvent.change(await screen.findByLabelText("Evaluator"), {
      target: { value: "e1" },
    });
    await screen.findByRole("option", { name: "v1 · Built-in" });
    fireEvent.change(screen.getByLabelText("Add saved version"), {
      target: { value: "ev1" },
    });
    const button = screen.getByRole("button", {
      name: "Start scoring saved outputs",
    });
    fireEvent.click(screen.getByLabelText("Required to pass"));
    expect(button).toBeDisabled();
    fireEvent.click(screen.getByLabelText("Required to pass"));
    fireEvent.click(button);
    await waitFor(() =>
      expect(mocks.scoreSavedOutputs).toHaveBeenCalledWith("source", [
        expect.objectContaining({
          version_id: "ev1",
          required: true,
          kind: "builtin",
        }),
      ]),
    );
    expect(started).toHaveBeenCalledWith(
      expect.objectContaining({ id: "scored", source_run_id: "source" }),
    );
    expect(mocks.testEvaluator).not.toHaveBeenCalled();
  });

  it("warns when only the judge rubric changes and suppresses misleading score deltas", () => {
    const judge = {
      provider: "groq" as const,
      model: "judge",
      rubric: "Correctness",
      threshold: 0.7,
    };
    const metric = {
      assignment: "inline:0",
      evaluator_name: null,
      evaluator_version_id: null,
      name: "llm_judge",
      data_type: "numeric" as const,
      required: true,
      count: 2,
      errors: 0,
      missing: 0,
      passed: 2,
      mean: 0.9,
      categories: {},
    };
    const detail: EvaluationDetail = {
      run: { ...run, config: { ...run.config, judge } },
      results: [],
      score_summary: [metric],
    };
    mount(
      <ResultsGrid
        detail={detail}
        comparison={{
          ...detail,
          run: {
            ...run,
            config: { ...run.config, judge: { ...judge, rubric: "Style" } },
          },
        }}
        filter="all"
      />,
    );
    expect(
      screen.getByText(/different scoring configurations/),
    ).toBeInTheDocument();
    expect(screen.getByText("Not comparable")).toBeInTheDocument();
  });

  it("uses all cases for provisional pass rates", () => {
    mount(
      <RunSummary
        detail={{
          run: { ...run, status: "running", completed: 1, passed: 1 },
          results: [],
        }}
      />,
    );
    expect(
      screen.getByText("Provisional pass rate · all cases"),
    ).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
    expect(screen.getByText("1/2")).toBeInTheDocument();
  });
});
