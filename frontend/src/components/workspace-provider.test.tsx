import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useQueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspaceProvider, WorkspaceSwitcher, useWorkspace } from "./workspace-provider";
import { getContextHeaders, SessionExpiredError } from "@/lib/request-context";
import { ApiError } from "@/lib/api-client";

const mocks = vi.hoisted(() => ({
  auth: { isLoaded: true, isSignedIn: true, userId: "user_a", getToken: vi.fn() },
  fetch: vi.fn(), push: vi.fn(), signOut: vi.fn(),
}));
vi.mock("@clerk/nextjs", () => ({ useAuth: () => mocks.auth, useClerk: () => ({ signOut: mocks.signOut }), RedirectToSignIn: () => <div>Sign in required</div> }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }));
vi.mock("@/lib/api-client", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/api-client")>(), fetchAPI: mocks.fetch }));

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
  mocks.signOut.mockResolvedValue(undefined);
  window.history.replaceState(null, "", "/dashboard");
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
  it.each([new ApiError("Invalid or expired session", 401), new SessionExpiredError()])("refreshes rejected or missing sessions with a fresh token", async (error) => {
    mocks.fetch.mockRejectedValueOnce(error).mockImplementationOnce(async () => {
      await getContextHeaders();
      return snapshot;
    });
    render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    await screen.findByRole("button", { name: "Refresh session" });
    expect(screen.queryByText("Current: First")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh session" }));
    await screen.findByText("Current: First");
    expect(mocks.auth.getToken).toHaveBeenCalledWith({ skipCache: true });
    expect(window.location.pathname).toBe("/dashboard");
  });
  it("ends a persistently rejected session and retains the exact workspace destination", async () => {
    window.history.replaceState(null, "", "/evaluations?run=fixture&case=2#details");
    mocks.fetch.mockRejectedValue(new ApiError("Invalid or expired session", 401));
    render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Refresh session" }));
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledTimes(2));
    fireEvent.click(await screen.findByRole("button", { name: "Sign in again" }));
    await waitFor(() => expect(mocks.signOut).toHaveBeenCalledWith({
      redirectUrl: "/sign-in?redirect_url=%2Fevaluations%3Frun%3Dfixture%26case%3D2%23details",
    }));
    expect(await getContextHeaders()).toEqual({});
    expect(screen.queryByLabelText("Project")).not.toBeInTheDocument();
  });
  it("does not offer session reset for a service configuration failure", async () => {
    mocks.fetch.mockRejectedValue(new ApiError("Authentication service unavailable", 503));
    render(<WorkspaceProvider><Probe /></WorkspaceProvider>);
    await screen.findByRole("button", { name: "Try again" });
    expect(screen.queryByRole("button", { name: "Sign in again" })).not.toBeInTheDocument();
  });
});
