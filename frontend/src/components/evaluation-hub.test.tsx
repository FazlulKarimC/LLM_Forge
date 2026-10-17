import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { EvaluationsWorkbench } from "./evaluations-workbench";
import { PagedResults } from "@/features/evaluations/components/paged-results";
const mocks = vi.hoisted(() => ({ listEvaluations: vi.fn(), getEvaluationSummary: vi.fn(), getEvaluationCases: vi.fn(), getEvaluationCase: vi.fn() }));
vi.mock("@/lib/evaluation-api", async original => ({...await original<object>(), ...mocks}));
const run = {id: "run", prompt_version_id: "version", dataset_revision_id: null,
  config: {source: "sdk_local" as const, reporting: "summary" as const, prompt_name: "Echo", prompt_version: 1, dataset_name: "Private tests", dataset_version: null, dataset_fingerprint: "sha256:aaa", scoring_fingerprint: "sha256:bbb", model: "app", provider: "external" as const, is_mock: false, assertions: []},
  status: "completed" as const, total: 1000, completed: 1000, passed: 990, errors: 2, source_run_id: null, updated_at: "2026-10-17T00:00:00Z", error: null, created_at: "2026-10-17T00:00:00Z"};
const summary = {run, results: [], score_summary: [], recovery: {available: false, attempts: [], failure_code: null, remaining_calls: {generation: 0, judging: 0, total: 0}}, usage: {generation: {tokens_input: 0, tokens_output: 0}, judging: {tokens_input: 0, tokens_output: 0}}};
function mount(node: React.ReactNode) { return render(<QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}>{node}</QueryClientProvider>); }
beforeEach(() => {
  vi.clearAllMocks(); window.history.replaceState(null,"","/evaluations");
  mocks.listEvaluations.mockResolvedValue({items:[run],total:1});
  mocks.getEvaluationSummary.mockResolvedValue(summary);
  mocks.getEvaluationCases.mockResolvedValue({items:[],total:0});
});
afterEach(cleanup);
it("shows local history without requesting hosted dataset management", async () => {
  mount(<EvaluationsWorkbench />);
  fireEvent.click(await screen.findByRole("button", {name:"Echo v1"}));
  expect(await screen.findByText("Summary-only report")).toBeInTheDocument();
  expect(screen.getByText(/Inputs, reference answers, outputs/)).toBeInTheDocument();
  expect(mocks.getEvaluationCases).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", {name:/Start evaluation/})).not.toBeInTheDocument();
  expect(screen.queryByText(/Score saved outputs again/)).not.toBeInTheDocument();
  await waitFor(() => expect(mocks.listEvaluations).toHaveBeenCalledWith(0, expect.anything(), expect.objectContaining({reference_run_id:"run"})));
});
it("opens SDK instructions from old new-run links without dataset selectors", async () => {
  window.history.replaceState(null,"","/evaluations?prompt=prompt&view=new");
  mount(<EvaluationsWorkbench />);
  expect(await screen.findByText("Evaluate locally, track results here")).toBeInTheDocument();
  expect(screen.getByLabelText("Local SDK example")).toHaveTextContent("upload=True");
  expect(screen.queryByLabelText(/Dataset revision/)).not.toBeInTheDocument();
});
it("distinguishes filtered empty results and resets search", async () => {
  mocks.listEvaluations.mockResolvedValue({items:[],total:0});
  mount(<EvaluationsWorkbench />);
  expect(await screen.findByText("No evaluations yet")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Search prompts"),{target:{value:"missing"}});
  expect(await screen.findByText("No runs match your filters")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button",{name:"Clear filters"}));
  expect(await screen.findByText("No evaluations yet")).toBeInTheDocument();
});
it("rejects a URL reference with mismatched local fingerprints", async () => {
  window.history.replaceState(null,"","/evaluations?run=run&compare=other");
  mocks.getEvaluationSummary.mockImplementation((id:string) => Promise.resolve(id==="other" ? {...summary,run:{...run,id:"other",config:{...run.config,dataset_fingerprint:"different"}}}:summary));
  mount(<EvaluationsWorkbench />);
  expect(await screen.findByText(/different test data or scoring definitions/)).toBeInTheDocument();
});
it("preserves focused paged inspection for legacy case results", async () => {
  const result = {case_index:0,name:"Greeting",inputs:{query:"Hello"},expected_output:"Hello",output:"Hello",passed:true,error:null,checks:[],latency_ms:1,tokens_input:null,tokens_output:null};
  mocks.getEvaluationCases.mockResolvedValue({items:[{...result,output_preview:"Hello"}],total:1});
  mocks.getEvaluationCase.mockResolvedValue(result);
  const select = vi.fn();
  mount(<PagedResults detail={{...summary,run:{...run,dataset_revision_id:"revision",config:{...run.config,source:"sdk_submission" as const}}}} filter="all" caseIndex={0} selectCase={select}/>);
  await waitFor(() => expect(document.getElementById("selected-case-detail")).toHaveFocus());
  expect(screen.queryByRole("button",{name:"Next"})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button",{name:"Greeting"}));
  expect(select).toHaveBeenCalledWith(0);
});
