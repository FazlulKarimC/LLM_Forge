import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PromptWorkbench } from "./prompt-workbench";
import { PromptPlayground } from "./prompt-playground";
import { ProjectAPIKeys } from "./project-api-keys";
import type { PromptDetail } from "@/lib/prompt-api";

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  createPrompt: vi.fn(),
  getPrompt: vi.fn(),
  saveVersion: vi.fn(),
  getVersions: vi.fn(),
  promoteVersion: vi.fn(),
  updatePrompt: vi.fn(),
  archivePrompt: vi.fn(),
  removeLabel: vi.fn(),
  compileDraft: vi.fn(),
  runPlayground: vi.fn(),
  listProjectKeys: vi.fn(),
  createProjectKey: vi.fn(),
  revokeProjectKey: vi.fn(),
}));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }));
vi.mock("@/lib/prompt-api", async () => ({
  ...(await vi.importActual<typeof import("@/lib/prompt-api")>(
    "@/lib/prompt-api",
  )),
  ...mocks,
}));

const first = {
  id: "v1",
  prompt_id: "p1",
  name: "support",
  template_text: "Answer {{query}}",
  template_format: "mustache" as const,
  version: 1,
  sha256_hash: "a".repeat(64),
  parent_id: null,
  description: "Initial",
  created_at: "2026-09-27T00:00:00Z",
  variables: ["query"],
};
const detail: PromptDetail = {
  prompt: {
    id: "p1",
    name: "support",
    description: "Support answers",
    archived: false,
    latest_version: 1,
    created_at: first.created_at,
    updated_at: first.created_at,
    labels: [],
  },
  version: first,
};
function wrap(child: React.ReactNode) {
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
      {child}
    </QueryClientProvider>,
  );
}
beforeEach(() => {
  vi.clearAllMocks();
  mocks.getVersions.mockResolvedValue([first]);
  mocks.getPrompt.mockResolvedValue(detail);
  mocks.promoteVersion.mockResolvedValue({
    ...detail.prompt,
    labels: [
      {
        label: "production",
        version_id: "v1",
        version: 1,
        updated_at: first.created_at,
      },
    ],
  });
  mocks.compileDraft.mockResolvedValue({ compiled_prompt: "Answer sample" });
  mocks.runPlayground.mockResolvedValue({
    output: "Generated answer",
    compiled_prompt: "Answer sample",
    provider: "mock",
    model: "demo-model",
    is_mock: true,
    latency_ms: 1,
    tokens_input: null,
    tokens_output: null,
    finish_reason: "demo",
  });
  mocks.listProjectKeys.mockResolvedValue([]);
});
afterEach(cleanup);

describe("prompt workbench", () => {
  it("creates a stable prompt and navigates to its editor", async () => {
    mocks.createPrompt.mockResolvedValue(detail);
    wrap(<PromptWorkbench />);
    fireEvent.change(screen.getByLabelText("Prompt name"), {
      target: { value: "support" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create prompt" }));
    await waitFor(() =>
      expect(mocks.createPrompt).toHaveBeenCalledWith(
        "support",
        expect.objectContaining({ template_format: "mustache" }),
      ),
    );
    expect(mocks.push).toHaveBeenCalledWith("/prompts/p1");
  });
  it("keeps unsaved edits out of releases and sends the current base version when saving", async () => {
    mocks.saveVersion.mockRejectedValue(
      new Error("A newer version was saved. Reload the prompt."),
    );
    wrap(<PromptWorkbench initial={detail} />);
    fireEvent.change(screen.getByLabelText("Prompt template"), {
      target: { value: "Improve {{query}}" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Releases" }));
    screen
      .getAllByRole("button", { name: "Promote v1" })
      .forEach((button) => expect(button).toBeDisabled());
    fireEvent.click(
      screen.getByRole("button", { name: "Editor & playground" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save new version" }));
    await waitFor(() =>
      expect(mocks.saveVersion).toHaveBeenCalledWith(
        "p1",
        expect.objectContaining({
          base_version: 1,
          template_text: "Improve {{query}}",
        }),
      ),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("newer version");
    expect(screen.getByLabelText("Prompt template")).toHaveValue(
      "Improve {{query}}",
    );
  });
  it("promotes a saved snapshot only after the release confirmation", async () => {
    wrap(<PromptWorkbench initial={detail} />);
    fireEvent.click(screen.getByRole("button", { name: "Releases" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Promote v1" })[1]);
    expect(mocks.promoteVersion).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Confirm promotion" }));
    await waitFor(() =>
      expect(mocks.promoteVersion).toHaveBeenCalledWith(
        "p1",
        "production",
        "v1",
      ),
    );
    expect(await screen.findByRole("status")).toHaveTextContent(
      "production now points to v1",
    );
  });
  it("shows a runnable Python example for the saved prompt", () => {
    wrap(<PromptWorkbench initial={detail} />);
    fireEvent.click(screen.getByRole("button", { name: "Integrate" }));
    expect(screen.getByText(/print\(prompt\.compile/)).toHaveTextContent(
      'print(prompt.compile(**{"query":"example"}))',
    );
  });
  it("reloads authoritative metadata, releases and template after a stale save", async () => {
    mocks.saveVersion.mockRejectedValueOnce(
      new Error("A newer version was saved."),
    );
    const second = {
      ...first,
      id: "v2",
      version: 2,
      template_text: "Latest {{query}}",
    };
    mocks.getPrompt.mockResolvedValue({
      prompt: {
        ...detail.prompt,
        name: "renamed",
        description: "Updated purpose",
        latest_version: 2,
        labels: [
          {
            label: "production",
            version_id: "v2",
            version: 2,
            updated_at: first.created_at,
          },
        ],
      },
      version: second,
    });
    wrap(<PromptWorkbench initial={detail} />);
    fireEvent.change(screen.getByLabelText("Prompt template"), {
      target: { value: "Stale {{query}}" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save new version" }));
    await screen.findByRole("alert");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Reload latest version (discard draft)",
      }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Prompt template")).toHaveValue(
        "Latest {{query}}",
      ),
    );
    expect(screen.getByLabelText("Prompt name")).toHaveValue("renamed");
    expect(screen.getByLabelText("Description")).toHaveValue("Updated purpose");
    expect(
      screen.getByRole("button", { name: "Save new version" }),
    ).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Prompt template"), {
      target: { value: "Newest {{query}}" },
    });
    mocks.saveVersion.mockResolvedValue({
      ...second,
      id: "v3",
      version: 3,
      template_text: "Newest {{query}}",
    });
    fireEvent.click(screen.getByRole("button", { name: "Save new version" }));
    await waitFor(() =>
      expect(mocks.saveVersion).toHaveBeenLastCalledWith(
        "p1",
        expect.objectContaining({ base_version: 2 }),
      ),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save new version" }),
      ).toBeDisabled(),
    );
  });
});

describe("prompt playground", () => {
  it("compiles sample variables and displays partial model failures without losing successful output", async () => {
    mocks.runPlayground
      .mockRejectedValueOnce(new Error("Provider unavailable"))
      .mockResolvedValueOnce({
        output: "Other model works",
        compiled_prompt: "Answer sample",
        provider: "mock",
        model: "demo-model",
        is_mock: true,
        latency_ms: 1,
        tokens_input: null,
        tokens_output: null,
        finish_reason: "demo",
      });
    wrap(
      <PromptPlayground
        draft={{
          template_text: "Answer {{query}}",
          template_format: "mustache",
          description: "",
        }}
      />,
    );
    fireEvent.change(screen.getByLabelText("query"), {
      target: { value: "sample" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Compare another model" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Run comparison" }));
    expect(await screen.findByText("Other model works")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Provider unavailable");
    expect(mocks.compileDraft).toHaveBeenCalledWith(expect.anything(), {
      query: "sample",
    });
    expect(mocks.runPlayground).toHaveBeenCalledTimes(2);
  });
  it("clears provider credentials when changing providers and never runs a paid provider without a key", () => {
    wrap(
      <PromptPlayground
        draft={{
          template_text: "Static",
          template_format: "mustache",
          description: "",
        }}
      />,
    );
    fireEvent.change(screen.getByLabelText("Provider 1"), {
      target: { value: "groq" },
    });
    expect(screen.getByRole("button", { name: "Run prompt" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Provider API key 1"), {
      target: { value: "private-key" },
    });
    expect(
      screen.getByRole("button", { name: "Run prompt" }),
    ).not.toBeDisabled();
    fireEvent.change(screen.getByLabelText("Provider 1"), {
      target: { value: "openrouter" },
    });
    expect(screen.getByLabelText("Provider API key 1")).toHaveValue("");
  });
});

describe("project SDK keys", () => {
  it("requires opting in before creating an evaluation-enabled key", async () => {
    mocks.createProjectKey.mockResolvedValue({
      id: "k1",
      secret: "lf_live_private",
      name: "CI",
    });
    wrap(<ProjectAPIKeys />);
    const capability = screen.getByLabelText(
      "Allow evaluations for CI (evaluations:write)",
    );
    expect(capability).not.toBeChecked();
    fireEvent.click(capability);
    fireEvent.change(screen.getByLabelText("Key name"), {
      target: { value: "CI" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create API key" }));
    await waitFor(() =>
      expect(mocks.createProjectKey).toHaveBeenCalledWith("CI", true),
    );
    await waitFor(() => expect(capability).not.toBeChecked());
  });
  it("shows a new key once and removes it from the page when dismissed", async () => {
    mocks.createProjectKey.mockResolvedValue({
      id: "k1",
      secret: "lf_live_private",
      name: "Local",
    });
    wrap(<ProjectAPIKeys />);
    fireEvent.change(screen.getByLabelText("Key name"), {
      target: { value: "Local" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create API key" }));
    expect(await screen.findByText("lf_live_private")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss key" }));
    expect(screen.queryByText("lf_live_private")).not.toBeInTheDocument();
    expect(window.localStorage.getItem("LLMFORGE_API_KEY")).toBeNull();
  });
});
