"use client";

import { useState, type FormEvent } from "react";
import { UserButton } from "@clerk/nextjs";
import { fetchAPI } from "@/lib/api-client";
import { useWorkspace } from "@/components/workspace-provider";
import { ProjectAPIKeys } from "@/components/prompts/project-api-keys";
import { inputClass } from "@/components/prompts/prompt-ui";
import { PageHeader } from "@/components/ui/primitives";

export default function SettingsPage() {
  const { organization, project, refresh } = useWorkspace();
  const [organizationName, setOrganizationName] = useState("");
  const [projectName, setProjectName] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function create(event: FormEvent, kind: "organization" | "project") {
    event.preventDefault();
    if (pending) return;
    setPending(kind);
    setError(null);
    try {
      const result = await fetchAPI<{ id: string; project_id?: string }>(
        kind === "organization"
          ? "/workspaces/organizations"
          : `/workspaces/organizations/${organization.id}/projects`,
        {
          method: "POST",
          body: JSON.stringify({
            name:
              kind === "organization"
                ? organizationName.trim()
                : projectName.trim(),
          }),
        },
      );
      await refresh(result.project_id || result.id);
      setOrganizationName("");
      setProjectName("");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Workspace creation failed",
      );
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="page-stack settings-layout">
      <PageHeader
        eyebrow="Workspace"
        title="Settings"
        description="Manage project access, keys, and workspaces."
      />
      {error ? (
        <p
          role="alert"
          className="alert alert-danger"
        >
          {error}
        </p>
      ) : null}
      <div>
        <h2 className="text-lg font-semibold">Project and API keys</h2>
        <p className="mt-1 text-sm text-(--muted-foreground)">
          Keys belong to the selected project. Create a read-only key for SDK
          prompt fetches, or opt in to evaluation access for CI.
        </p>
      </div>
      {organization.role === "owner" ? (
        <ProjectAPIKeys />
      ) : (
        <p className="text-sm text-(--muted-foreground)">
          The organization owner manages project keys.
        </p>
      )}
      <div className="border-t border-(--border) pt-6">
        <h2 className="text-lg font-semibold">Workspace and account</h2>
        <p className="mt-1 text-sm text-(--muted-foreground)">
          Create projects in this organization or start a separate organization.
        </p>
      </div>
      <section className="panel workbench-panel">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold">{organization.name}</h3>
            <p className="mt-1 text-sm text-(--muted-foreground)">
              {project.name} · {organization.role}
            </p>
          </div>
          <UserButton />
        </div>
        <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
          <div>
            <dt className="text-(--muted-foreground)">Organization ID</dt>
            <dd className="mt-1 break-all font-mono">{organization.id}</dd>
          </div>
          <div>
            <dt className="text-(--muted-foreground)">Project ID</dt>
            <dd className="mt-1 break-all font-mono">{project.id}</dd>
          </div>
        </dl>
      </section>
      <div className="grid gap-6 md:grid-cols-2">
        <form
          onSubmit={(event) => create(event, "project")}
          className="panel workbench-panel"
        >
          <h2 className="text-xl font-semibold">Create a project</h2>
          <p className="mt-2 text-sm text-(--muted-foreground)">
            Keep prompts and experiments for an application together.
          </p>
          <label className="mt-4 block text-sm">
            Project name
            <input
              className={inputClass}
              required
              maxLength={120}
              value={projectName}
              onChange={(event) => setProjectName(event.target.value)}
              placeholder="Support assistant"
              disabled={organization.role !== "owner"}
            />
          </label>
          <button
            className="btn-primary mt-5"
            disabled={
              !!pending || !projectName.trim() || organization.role !== "owner"
            }
          >
            {pending === "project" ? "Creating…" : "Create project"}
          </button>
          {organization.role !== "owner" ? (
            <p className="mt-3 text-sm text-(--muted-foreground)">
              Your workspace owner can create projects.
            </p>
          ) : null}
        </form>
        <form
          onSubmit={(event) => create(event, "organization")}
          className="panel workbench-panel"
        >
          <h2 className="text-xl font-semibold">Create an organization</h2>
          <p className="mt-2 text-sm text-(--muted-foreground)">
            Start a separate workspace with its own default project.
          </p>
          <label className="mt-4 block text-sm">
            Organization name
            <input
              className={inputClass}
              required
              maxLength={120}
              value={organizationName}
              onChange={(event) => setOrganizationName(event.target.value)}
              placeholder="My workspace"
            />
          </label>
          <button
            className="btn-secondary mt-5"
            disabled={!!pending || !organizationName.trim()}
          >
            {pending === "organization" ? "Creating…" : "Create organization"}
          </button>
        </form>
      </div>
    </div>
  );
}
