import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useQueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspaceProvider, WorkspaceSwitcher, useWorkspace } from "./workspace-provider";
import { getContextHeaders } from "@/lib/request-context";

const mocks = vi.hoisted(() => ({
  auth: { isLoaded: true, isSignedIn: true, userId: "user_a", getToken: vi.fn() },
  fetch: vi.fn(), push: vi.fn(),
}));
vi.mock("@clerk/nextjs", () => ({ useAuth: () => mocks.auth, RedirectToSignIn: () => <div>Sign in required</div> }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }));
vi.mock("@/lib/api-client", () => ({ fetchAPI: mocks.fetch }));

function Probe() {
  const { project } = useWorkspace();
  const client = useQueryClient();
  return <>
    <WorkspaceSwitcher />
    <div>Current: {project.name}</div>
    <button onClick={() => client.setQueryData(["records"], ["private data"])}>Cache record</button>
    <button onClick={() => expect(client.getQueryData(["records"])).toBeUndefined()}>Check empty cache</button>
  </>;
}
const snapshot = {
  user: { id: "local_a", display_name: "Developer" },
  organizations: [{ id: "org_a", name: "Personal", slug: "personal", role: "owner", projects: [
    { id: "project_a", name: "First", slug: "first", organization_id: "org_a" },
    { id: "project_b", name: "Second", slug: "second", organization_id: "org_a" },
  ] }],
};
beforeEach(() => {
  vi.clearAllMocks(); window.localStorage.clear();
  mocks.auth.isSignedIn = true; mocks.auth.userId = "user_a";
  mocks.auth.getToken.mockResolvedValue("token_a");
  mocks.fetch.mockResolvedValue(snapshot);
});
afterEach(cleanup);

describe("workspace boundary", () => {
  it("switches request identity and clears cached records when changing projects", async () => {
    render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    await screen.findByText("Current: First");
    fireEvent.click(screen.getByText("Cache record"));
    fireEvent.change(screen.getByLabelText("Project"), { target: { value: "project_b" } });
    await screen.findByText("Current: Second");
    fireEvent.click(screen.getByText("Check empty cache"));
    expect(await getContextHeaders()).toEqual({ Authorization: "Bearer token_a", "X-Project-ID": "project_b" });
    expect(mocks.push).toHaveBeenCalledWith("/dashboard");
    expect(window.localStorage.getItem("llmforge.project.user_a")).toBe("project_b");
  });
  it("discards an inaccessible saved project and does not render private children after logout", async () => {
    window.localStorage.setItem("llmforge.project.user_a", "foreign_project");
    const view = render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    await screen.findByText("Current: First");
    mocks.auth.isSignedIn = false;
    view.rerender(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    expect(screen.getByText("Sign in required")).toBeInTheDocument();
    expect(screen.queryByText("Current: First")).not.toBeInTheDocument();
    expect(await getContextHeaders()).toEqual({});
  });
  it("keeps the app gated on bootstrap failure and can retry", async () => {
    mocks.fetch.mockRejectedValueOnce(new Error("Backend unavailable"));
    render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    expect(await screen.findByRole("alert")).toHaveTextContent("Backend unavailable");
    expect(screen.queryByLabelText("Project")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Try again"));
    await waitFor(() => expect(screen.getByText("Current: First")).toBeInTheDocument());
  });
});
