"use client";
import { useAuth, useClerk, RedirectToSignIn } from "@clerk/nextjs";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Providers } from "@/app/providers";
import { ApiError, fetchAPI } from "@/lib/api-client";
import { SessionExpiredError, setApiContext } from "@/lib/request-context";
import { clearCustomProviderCredentials } from "@/lib/custom-provider";
import { Panel } from "@/components/ui/primitives";
import type { components } from "@/lib/generated/api-schema";
export type WorkspaceProject = components["schemas"]["ProjectResponse"];
export type WorkspaceOrganization = components["schemas"]["OrganizationResponse"];
type Snapshot = components["schemas"]["WorkspaceResponse"];
type Workspace = {
  organizations: WorkspaceOrganization[];
  organization: WorkspaceOrganization;
  project: WorkspaceProject;
  switchProject: (id: string) => boolean;
  refresh: (selectedId?: string) => Promise<void>;
};
const WorkspaceContext = createContext<Workspace | null>(null);
export function useWorkspace() {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error("Workspace context is unavailable");
  return value;
}
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn, userId, getToken } = useAuth();
  if (!isLoaded) return <WorkspaceMessage title="Loading your session…" />;
  if (!isSignedIn || !userId) return <RedirectToSignIn />;
  // Changing accounts unmounts all workspace state and query caches.
  return <AuthenticatedWorkspace key={userId} userId={userId} getToken={getToken}>{children}</AuthenticatedWorkspace>;
}
function WorkspaceMessage({ title, error, children }: { title: string; error?: string; children?: ReactNode }) {
  return <main className="page-width flex min-h-screen items-center justify-center px-6"><Panel className="w-full max-w-lg"><div className="panel-body flex flex-col gap-4">
    <h1 className="text-2xl font-semibold">{title}</h1>
    {error ? <p role="alert" className="text-(--muted-foreground)">{error}</p> : null}
    {children}
  </div></Panel></main>;
}
function AuthenticatedWorkspace({ children, userId, getToken }: { children: ReactNode; userId: string; getToken: (options?: { skipCache?: boolean }) => Promise<string | null> }) {
  const router = useRouter();
  const { signOut } = useClerk();
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [recovering, setRecovering] = useState(false);
  const recordError = useCallback((err: unknown) => setError(err instanceof Error ? err : new Error("Could not load your workspace")), []);
  const storageKey = `llmforge.project.${userId}`;
  const refresh = useCallback(async (selectedId?: string, signal?: AbortSignal, freshSession = false) => {
    if (freshSession) setApiContext({ getToken: () => getToken({ skipCache: true }) });
    const result = await fetchAPI<Snapshot>("/workspaces", { signal });
    if (signal?.aborted) return;
    const projects = result.organizations.flatMap((org) => org.projects);
    const preferred = selectedId || window.localStorage.getItem(storageKey);
    const selected = projects.find((p) => p.id === preferred) || projects[0];
    if (!selected) throw new Error("No projects are available. Contact your workspace owner.");
    setApiContext({ getToken, projectId: selected.id });
    window.localStorage.setItem(storageKey, selected.id);
    setSnapshot(result);
    setProjectId(selected.id);
    setError(null);
  }, [getToken, storageKey]);
  useEffect(() => {
    clearCustomProviderCredentials();
    const controller = new AbortController();
    setApiContext({ getToken });
    Promise.resolve().then(() => refresh(undefined, controller.signal)).catch((err: unknown) => {
      if (!controller.signal.aborted) recordError(err);
    });
    return () => { controller.abort(); setApiContext(null); };
  }, [getToken, refresh, recordError]);
  function switchProject(id: string) {
    if (!snapshot?.organizations.some((org) => org.projects.some((p) => p.id === id))) return false;
    if (id === projectId) return true;
    if (!window.dispatchEvent(new Event("llmforge:before-workspace-switch", { cancelable: true }))) return false;
    clearCustomProviderCredentials();
    setApiContext({ getToken, projectId: id });
    window.localStorage.setItem(storageKey, id);
    setProjectId(id);
    // Do not retain a detail URL belonging to the old project.
    router.push("/dashboard");
    return true;
  }
  if (error) {
    const authenticationError = error instanceof SessionExpiredError || (error instanceof ApiError && error.statusCode === 401);
    const recover = async (signIn = false) => {
      setRecovering(true);
      try {
        if (signIn) {
          // End the rejected Clerk session so /sign-in cannot redirect it back.
          // Construct the destination locally; never accept an external return URL.
          const destination = window.location.pathname + window.location.search + window.location.hash;
          const returnTo = destination.startsWith("/") && !destination.startsWith("//") ? destination : "/dashboard";
          setApiContext(null);
          await signOut({ redirectUrl: `/sign-in?redirect_url=${encodeURIComponent(returnTo)}` });
        } else {
          await refresh(undefined, undefined, authenticationError);
        }
      } catch (err) {
        // A failed sign-out still offers both recovery actions.
        if (signIn) setError(new SessionExpiredError());
        else recordError(err);
      } finally {
        setRecovering(false);
      }
    };
    return <WorkspaceMessage title="Your workspace could not load" error={error.message}>
      <div className="flex flex-wrap gap-3">
        <button disabled={recovering} onClick={() => void recover()} className="btn-primary">{recovering ? "Recovering…" : authenticationError ? "Refresh session" : "Try again"}</button>
        {authenticationError ? <button disabled={recovering} onClick={() => void recover(true)} className="btn-secondary">Sign in again</button> : null}
      </div>
      {authenticationError ? <p className="field-help">Sign in again ends this session and returns you to this page. If a fresh session is also rejected, contact the deployment owner.</p> : null}
    </WorkspaceMessage>;
  }
  const organization = snapshot?.organizations.find((org) => org.projects.some((p) => p.id === projectId));
  const project = organization?.projects.find((p) => p.id === projectId);
  if (!snapshot || !organization || !project) return <WorkspaceMessage title="Preparing your workspace…" />;
  return <WorkspaceContext.Provider value={{ organizations: snapshot.organizations, organization, project, switchProject, refresh }}>
    <Providers key={`${userId}:${project.id}`}>{children}</Providers>
  </WorkspaceContext.Provider>;
}
export function WorkspaceSwitcher() {
  const { organizations, organization, project, switchProject } = useWorkspace();
  return <div className="workspace-switcher">
    <select aria-label="Organization" className="max-w-48 rounded-xl border border-(--border) bg-(--surface-1) px-3 py-2 text-sm" value={organization.id}
      onChange={(event) => { const first = organizations.find((org) => org.id === event.target.value)?.projects[0]; if (!first || !switchProject(first.id)) event.target.value = organization.id; }}>
      {organizations.map((org) => <option key={org.id} value={org.id}>{org.name}</option>)}
    </select>
    <span className="text-(--muted-foreground)">/</span>
    <select aria-label="Project" className="max-w-48 rounded-xl border border-(--border) bg-(--surface-1) px-3 py-2 text-sm" value={project.id} onChange={(event) => { if (!switchProject(event.target.value)) event.target.value = project.id; }}>
      {organization.projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select>
  </div>;
}
