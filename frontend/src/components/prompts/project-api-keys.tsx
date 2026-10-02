"use client";
import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound } from "lucide-react";
import {
  createProjectKey,
  listProjectKeys,
  revokeProjectKey,
} from "@/lib/prompt-api";
import { ErrorMessage, errorText, inputClass, panelClass } from "./prompt-ui";

export function ProjectAPIKeys() {
  const cache = useQueryClient();
  const query = useQuery({
    queryKey: ["project-api-keys"],
    queryFn: listProjectKeys,
  });
  const [name, setName] = useState("");
  const [evaluations, setEvaluations] = useState(false);
  const [promptWrites, setPromptWrites] = useState(false);
  const [secret, setSecret] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  async function create(event: FormEvent) {
    event.preventDefault();
    if (pending) return;
    setPending(true);
    setError(null);
    setSecret(null);
    setCopied(false);
    try {
      const created = promptWrites
        ? await createProjectKey(name.trim(), evaluations, true)
        : evaluations
          ? await createProjectKey(name.trim(), true)
          : await createProjectKey(name.trim());
      setSecret(created.secret);
      setName("");
      setEvaluations(false);
      setPromptWrites(false);
      await cache.invalidateQueries({ queryKey: ["project-api-keys"] });
    } catch (err) {
      setError(errorText(err));
    } finally {
      setPending(false);
    }
  }
  async function revoke(id: string) {
    setPending(true);
    setError(null);
    setSecret(null);
    try {
      await revokeProjectKey(id);
      await cache.invalidateQueries({ queryKey: ["project-api-keys"] });
    } catch (err) {
      setError(errorText(err));
    } finally {
      setPending(false);
    }
  }
  return (
    <section className={`${panelClass} space-y-4`}>
      <h2 className="flex items-center gap-2 text-xl font-semibold">
        <KeyRound className="size-5" />
        Project API keys
      </h2>
      <p className="text-sm text-(--muted-foreground)">
        Keys read prompts in this project by default. Enable only the extra
        capabilities your application needs. Evaluation access runs checks;
        prompt write access creates versions and moves release labels. Keys
        cannot manage keys or edit datasets.
      </p>
      <label className="flex gap-2 text-sm">
        <input
          type="checkbox"
          checked={evaluations}
          disabled={pending}
          onChange={(event) => setEvaluations(event.target.checked)}
        />
        Allow evaluations for CI (evaluations:write)
      </label>
      <label className="flex gap-2 text-sm">
        <input
          type="checkbox"
          checked={promptWrites}
          disabled={pending}
          onChange={(event) => setPromptWrites(event.target.checked)}
        />
        Allow prompt creation and label changes (prompts:write)
      </label>
      <form onSubmit={create} className="flex flex-wrap items-end gap-3">
        <label className="min-w-48 flex-1 text-sm">
          Key name
          <input
            className={inputClass}
            required
            maxLength={120}
            value={name}
            disabled={pending}
            onChange={(event) => setName(event.target.value)}
            placeholder="Local development"
          />
        </label>
        <button className="btn-primary" disabled={pending || !name.trim()}>
          {pending ? "Working…" : "Create API key"}
        </button>
      </form>
      <ErrorMessage
        message={error || (query.error ? errorText(query.error) : null)}
      />
      {secret ? (
        <div className="rounded-xl border border-(--primary) p-4">
          <h3 className="text-sm font-semibold">Copy this key now</h3>
          <p className="mt-2 text-xs text-(--muted-foreground)">
            It is shown once. Store it in your application&apos;s environment
            and keep it out of source control.
          </p>
          <code className="mt-3 block break-all select-all rounded-lg bg-(--surface-2) p-3 text-xs">
            {secret}
          </code>
          <div className="mt-3 flex gap-2">
            <button
              className="btn-secondary"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(secret);
                  setCopied(true);
                } catch {
                  setError(
                    "Clipboard access was denied. Select and copy the key manually.",
                  );
                }
              }}
            >
              {copied ? "Copied" : "Copy key"}
            </button>
            <button className="btn-secondary" onClick={() => setSecret(null)}>
              Dismiss key
            </button>
          </div>
        </div>
      ) : null}
      {query.isPending ? (
        <p role="status" className="text-sm">
          Loading keys…
        </p>
      ) : query.data?.length ? (
        <div className="divide-y divide-(--border)">
          {query.data.map((key) => (
            <div
              key={key.id}
              className="flex flex-wrap items-center justify-between gap-3 py-3"
            >
              <div>
                <h3 className="text-sm font-medium">{key.name}</h3>
                <p className="mt-1 text-xs text-(--muted-foreground)">
                  <code>{key.prefix}…</code> ·{" "}
                  {(key.scopes ?? ["prompts:read"]).join(", ")} ·{" "}
                  {key.revoked_at ? "Revoked" : "Active"}
                </p>
              </div>
              {!key.revoked_at ? (
                <button
                  className="btn-secondary"
                  disabled={pending}
                  onClick={() => revoke(key.id)}
                >
                  Revoke
                </button>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-(--muted-foreground)">No API keys yet.</p>
      )}
    </section>
  );
}
