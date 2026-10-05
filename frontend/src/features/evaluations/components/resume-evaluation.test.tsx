import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ResumeEvaluation } from "./resume-evaluation";
import type { EvaluationSummary } from "@/lib/evaluation-api";

const resume = vi.hoisted(() => vi.fn());
vi.mock("@/lib/evaluation-api", () => ({ resumeEvaluation: resume }));
afterEach(() => { cleanup(); resume.mockReset(); });
const summary = {
  run: { id: "run-1", config: { provider: "groq", judge: { provider: "groq" }, evaluators: [] } },
  recovery: { available: true, attempts: [{ attempt: 1 }], remaining_calls: { generation: 1, judging: 2, total: 3 } },
} as unknown as EvaluationSummary;

it("retains dispatch identity after an uncertain response, re-enters keys and clears them on acceptance", async () => {
  resume.mockRejectedValueOnce(new Error("Response lost")).mockResolvedValueOnce({});
  const resumed = vi.fn();
  render(<ResumeEvaluation summary={summary} resumed={resumed} />);
  expect(screen.getByText(/may be charged again/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Generation API key"), { target: { value: "generation-secret" } });
  fireEvent.change(screen.getByLabelText("Inline judge API key"), { target: { value: "judge-secret" } });
  fireEvent.click(screen.getByRole("button", { name: "Resume unfinished work" }));
  await screen.findByRole("alert");
  const firstKey = resume.mock.calls[0][2];
  expect(firstKey).toEqual(expect.any(String));
  fireEvent.change(screen.getByLabelText("Generation API key"), { target: { value: "replacement-secret" } });
  fireEvent.click(screen.getByRole("button", { name: "Resume unfinished work" }));
  await waitFor(() => expect(resumed).toHaveBeenCalledOnce());
  expect(resume.mock.calls[1]).toEqual(["run-1", { api_key: "replacement-secret", judge_key: "judge-secret", evaluator_keys: {} }, firstKey]);
  expect((screen.getByLabelText("Generation API key") as HTMLInputElement).value).toBe("");
  expect((screen.getByLabelText("Inline judge API key") as HTMLInputElement).value).toBe("");
});

it("does not offer recovery for ordinary terminal failures", () => {
  const { container } = render(<ResumeEvaluation summary={{ ...summary, recovery: { ...summary.recovery, available: false } }} resumed={vi.fn()} />);
  expect(container.textContent).toBe("");
});
