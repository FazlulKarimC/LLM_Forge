"use client";

import { useAuth, RedirectToSignIn } from "@clerk/nextjs";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Providers } from "@/app/providers";
import { fetchAPI } from "@/lib/api-client";
import { setApiContext } from "@/lib/request-context";

export type WorkspaceProject = { id: string; name: string; slug: string; organization_id: string };
export type WorkspaceOrganization = { id: string; name: string; slug: string; role: "owner" | "member"; projects: WorkspaceProject[] };
type Snapshot = { user: { id: string; display_name: string }; organizations: WorkspaceOrganization[] };
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

function WorkspaceMessage({ title, error, retry }: { title: string; error?: string; retry?: () => void }) {
  return <main className="page-width flex min-h-screen items-center justify-center px-6"><div className="max-w-lg rounded-3xl border border-(--border) bg-(--surface-1) p-8">
    <h1 className="text-2xl font-semibold">{title}</h1>
    {error ? <p role="alert" className="mt-3 text-(--muted-foreground)">{error}</p> : null}
    {retry ? <button onClick={retry} className="btn-primary mt-5">Try again</button> : null}
  </div></main>;
}

function AuthenticatedWorkspace({ children, userId, getToken }: { children: ReactNode; userId: string; getToken: () => Promise<string | null> }) {
  const router = useRouter();
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const storageKey = `llmforge.project.${userId}`;

  const refresh = useCallback(async (selectedId?: string, signal?: AbortSignal) => {
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
    const controller = new AbortController();
    setApiContext({ getToken });
    Promise.resolve().then(() => refresh(undefined, controller.signal)).catch((err: unknown) => {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : "Could not load your workspace");
    });
    return () => { controller.abort(); setApiContext(null); };
  }, [getToken, refresh]);

  function switchProject(id: string) {
    if (!snapshot?.organizations.some((org) => org.projects.some((p) => p.id === id))) return false;
    if (id === projectId) return true;
    if (!window.dispatchEvent(new Event("llmforge:before-workspace-switch", { cancelable: true }))) return false;
    setApiContext({ getToken, projectId: id });
    window.localStorage.setItem(storageKey, id);
    setProjectId(id);
    // Do not retain a detail URL belonging to the old project.
    router.push("/dashboard");
    return true;
  }

  if (error) return <WorkspaceMessage title="Your workspace could not load" error={error} retry={() => { refresh().catch((err: unknown) => setError(err instanceof Error ? err.message : "Could not load your workspace")); }} />;
  const organization = snapshot?.organizations.find((org) => org.projects.some((p) => p.id === projectId));
  const project = organization?.projects.find((p) => p.id === projectId);
  if (!snapshot || !organization || !project) return <WorkspaceMessage title="Preparing your workspace…" />;

  return <WorkspaceContext.Provider value={{ organizations: snapshot.organizations, organization, project, switchProject, refresh }}>
    <Providers key={`${userId}:${project.id}`}>{children}</Providers>
  </WorkspaceContext.Provider>;
}

export function WorkspaceSwitcher() {
  const { organizations, organization, project, switchProject } = useWorkspace();
  return <div className="flex min-w-0 flex-wrap items-center gap-2">
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
