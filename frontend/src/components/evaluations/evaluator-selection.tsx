"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import {
  listEvaluators,
  listEvaluatorVersions,
  type EvaluatorSelection,
} from "@/lib/evaluator-api";
import { PageControls } from "@/components/ui/page-controls";
import { inputClass, errorText } from "@/components/prompts/prompt-ui";

export function EvaluatorSelectionFields({
  value,
  onChange,
}: {
  value: EvaluatorSelection[];
  onChange: (items: EvaluatorSelection[]) => void;
}) {
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [identifier, setIdentifier] = useState("");
  const [versionOffset, setVersionOffset] = useState(0);
  const catalog = useQuery({
    queryKey: ["evaluators", search, offset],
    queryFn: ({ signal }) => listEvaluators(search, false, offset, signal),
  });
  const versions = useQuery({
    queryKey: ["evaluator-versions", identifier, versionOffset],
    queryFn: ({ signal }) =>
      listEvaluatorVersions(identifier, versionOffset, signal),
    enabled: !!identifier,
  });
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="font-medium">Saved evaluators</h4>
        <Link
          className="text-(--accent) text-sm"
          href="/evaluations/evaluators"
        >
          Manage evaluators
        </Link>
      </div>
      <label className="text-sm">
        Find evaluator
        <input
          className={inputClass}
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setOffset(0);
          }}
        />
      </label>
      <label className="text-sm">
        Evaluator
        <select
          className={inputClass}
          value={identifier}
          onChange={(event) => {
            setIdentifier(event.target.value);
            setVersionOffset(0);
          }}
        >
          <option value="">Select evaluator</option>
          {catalog.data?.items.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      <PageControls
        offset={offset}
        next={offset + 50 < (catalog.data?.total ?? 0)}
        change={setOffset}
      />
      {identifier && (
        <>
          <label className="text-sm">
            Add saved version
            <select
              className={inputClass}
              value=""
              onChange={(event) => {
                const version = versions.data?.find(
                  (item) => item.id === event.target.value,
                );
                const item = catalog.data?.items.find(
                  (item) => item.id === identifier,
                );
                if (
                  !version ||
                  !item ||
                  value.some((entry) => entry.version_id === version.id)
                )
                  return;
                onChange([
                  ...value,
                  {
                    version_id: version.id,
                    required: true,
                    kind: version.definition.kind,
                    name: item.name,
                    version: version.version,
                  },
                ]);
              }}
              disabled={value.length >= 10}
            >
              <option value="">Choose version to add</option>
              {versions.data?.map((version) => (
                <option
                  key={version.id}
                  value={version.id}
                  disabled={value.some(
                    (item) => item.version_id === version.id,
                  )}
                >{`v${version.version} · ${version.definition.kind === "llm_judge" ? "LLM judge" : "Built-in"}`}</option>
              ))}
            </select>
          </label>
          <PageControls
            offset={versionOffset}
            next={versions.data?.length === 50}
            change={setVersionOffset}
          />
        </>
      )}
      {catalog.isLoading && <p className="field-help">Loading evaluators…</p>}
      {catalog.data?.total === 0 && (
        <p className="field-help">
          No matching saved evaluators. Create one in the evaluator library.
        </p>
      )}
      {[catalog.error, versions.error].filter(Boolean).map((error, index) => (
        <p key={index} role="alert" className="text-(--destructive)">
          {errorText(error)}
        </p>
      ))}
      {value.map((item, index) => (
        <div
          key={item.version_id}
          className="rounded-lg border border-(--border) p-3 grid gap-2"
        >
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-sm font-medium">
              {item.name ?? "Evaluator"} v{item.version ?? "—"}
            </span>
            <button
              type="button"
              className="btn-ghost"
              aria-label={`Remove evaluator ${index + 1}`}
              onClick={() =>
                onChange(
                  value.filter((entry) => entry.version_id !== item.version_id),
                )
              }
            >
              Remove
            </button>
          </div>
          <label className="text-sm flex items-center gap-2">
            <input
              type="checkbox"
              checked={item.required}
              onChange={(event) =>
                onChange(
                  value.map((entry) =>
                    entry.version_id === item.version_id
                      ? { ...entry, required: event.target.checked }
                      : entry,
                  ),
                )
              }
            />
            Required to pass
          </label>
          {item.kind === "llm_judge" && (
            <label className="text-sm">
              Judge key for {item.name ?? "evaluator"}
              <input
                type="password"
                autoComplete="off"
                className={inputClass}
                value={item.api_key ?? ""}
                onChange={(event) =>
                  onChange(
                    value.map((entry) =>
                      entry.version_id === item.version_id
                        ? { ...entry, api_key: event.target.value }
                        : entry,
                    ),
                  )
                }
              />
              <span className="field-help">
                Request-only. At most two judges per run, including the inline
                judge.
              </span>
            </label>
          )}
        </div>
      ))}
      <p className="field-help">
        Versions are pinned when the run starts. Informational evaluators add
        scores without deciding pass/fail.
      </p>
    </div>
  );
}
