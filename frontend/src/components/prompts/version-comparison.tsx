"use client";
import { useState } from "react";
import { canonicalJSON, type PromptVersion } from "@/lib/prompt-api";
import { inputClass } from "./prompt-ui";

function content(version: PromptVersion) {
  return version.prompt_type === "chat"
    ? (version.messages ?? [])
        .map((message) => `[${message.role}]\n${message.content}`)
        .join("\n\n")
    : version.template_text;
}

export function VersionComparison({ versions }: { versions: PromptVersion[] }) {
  const [referenceId, setReferenceId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  if (versions.length < 2)
    return (
      <p className="text-sm text-(--muted-foreground)">
        Save another version to compare content and configuration.
      </p>
    );
  const reference =
    versions.find((version) => version.id === referenceId) ?? versions[1];
  const candidate =
    versions.find((version) => version.id === candidateId) ?? versions[0];
  const before = content(reference).split("\n"),
    after = content(candidate).split("\n");
  let prefix = 0,
    suffix = 0;
  while (
    prefix < Math.min(before.length, after.length) &&
    before[prefix] === after[prefix]
  )
    prefix++;
  while (
    suffix < Math.min(before.length, after.length) - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  )
    suffix++;
  const hasChanges = content(reference) !== content(candidate);
  return (
    <section
      aria-label="Version comparison"
      className="space-y-4 border-t border-(--border) pt-4"
    >
      <h3 className="text-sm font-semibold">Compare saved versions</h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {[
          { label: "Reference", id: reference.id, change: setReferenceId },
          { label: "Candidate", id: candidate.id, change: setCandidateId },
        ].map(({ label, id, change }) => (
          <label key={label} className="text-xs">
            {label} version
            <select
              className={inputClass}
              value={id}
              onChange={(event) => change(event.target.value)}
            >
              {versions.map((version) => (
                <option key={version.id} value={version.id}>
                  v{version.version} · {version.description || "No notes"}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <p className="text-xs text-(--muted-foreground)">
        Changes between v{reference.version} and v{candidate.version}; unchanged
        leading and trailing lines are omitted.
      </p>
      {hasChanges ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="min-w-0">
            <h4 className="mb-2 text-xs font-medium">
              Removed / changed · reference v{reference.version}
            </h4>
            <pre className="whitespace-pre-wrap break-words rounded-lg border border-(--destructive)/30 bg-(--destructive)/5 p-3 text-xs max-h-80 overflow-auto">
              {before.slice(prefix, before.length - suffix).join("\n") ||
                "[No removed lines]"}
            </pre>
          </div>
          <div className="min-w-0">
            <h4 className="mb-2 text-xs font-medium">
              Added / changed · candidate v{candidate.version}
            </h4>
            <pre className="whitespace-pre-wrap break-words rounded-lg border border-(--success)/30 bg-(--success)/5 p-3 text-xs max-h-80 overflow-auto">
              {after.slice(prefix, after.length - suffix).join("\n") ||
                "[No added lines]"}
            </pre>
          </div>
        </div>
      ) : (
        <p className="text-xs">Prompt content is unchanged.</p>
      )}
      {reference.template_format !== candidate.template_format && (
        <p className="text-xs">
          Template format: {reference.template_format} →{" "}
          {candidate.template_format}
        </p>
      )}
      {canonicalJSON(reference.config ?? {}) !==
        canonicalJSON(candidate.config ?? {}) && (
        <div className="grid gap-3 sm:grid-cols-2">
          {[
            { label: "Reference config", config: reference.config },
            { label: "Candidate config", config: candidate.config },
          ].map(({ label, config }) => (
            <div key={label} className="min-w-0">
              <h4 className="mb-2 text-xs font-medium">{label}</h4>
              <pre className="whitespace-pre-wrap break-words rounded-lg bg-(--surface-2) p-3 text-xs max-h-64 overflow-auto">
                {JSON.stringify(config ?? {}, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
