"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { exportResults, getBenchmarkAttempts, purgeBenchmarkAttempt } from "@/lib/api";
import { Panel } from "@/components/ui/primitives";

export function AttemptSelector({ id, current, selected, status, onSelect }: { id: string; current: number; selected: number; status: string; onSelect: (n: number) => void }) {
  const [skip, setSkip] = useState(0);
  const [busy, setBusy] = useState(false);
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["attempts", id, current, skip, status], queryFn: ({ signal }) => getBenchmarkAttempts(id, skip, { signal }), retry: false });
  const snapshot = query.data?.attempts.find((row) => row.number === selected);
  return <Panel><div className="panel-body space-y-4">
    <div className="flex flex-wrap items-center gap-3">
      <label>Attempt <select aria-label="Result attempt" value={selected} onChange={(event) => onSelect(Number(event.target.value))}>
        {!query.data?.attempts.some((row) => row.number === selected) ? <option value={selected}>{selected} (current selection)</option> : null}
        {query.data?.attempts.map((row) => <option key={row.number} value={row.number}>{row.number === 0 ? "Legacy aggregate (unbound)" : row.number} · {row.status} · {row.provenance}</option>)}
      </select></label>
      <button className="btn-secondary" disabled={skip === 0} onClick={() => setSkip(Math.max(0, skip - 25))}>Newer attempts</button>
      <button className="btn-secondary" disabled={skip + 25 >= (query.data?.total ?? 0)} onClick={() => setSkip(skip + 25)}>Older attempts</button>
      <button className="btn-secondary" onClick={() => onSelect(current)}>Current attempt {current}</button>
      <button className="btn-secondary" disabled={busy} onClick={() => void exportResults(id, undefined, selected, "jsonl").catch((e: Error) => toast.error(e.message))}>Export complete JSONL</button>
      {snapshot && !["queued", "running"].includes(snapshot.status) ? <button className="btn-danger" disabled={busy} onClick={async () => {
        setBusy(true);
        try {
          await exportResults(id, undefined, selected, "jsonl");
          if (!window.confirm(`The export has downloaded. Permanently purge attempt ${selected}? Pinned or referenced attempts are protected.`)) return;
          await purgeBenchmarkAttempt(id, selected);
          await Promise.all(["attempts", "experiment", "metrics", "runs-grid", "run-detail", "profile", "routing", "regression"].map((key) => client.invalidateQueries({ queryKey: [key, id] })));
          onSelect(current);
          toast.success("Historical attempt purged");
        } catch (error) { toast.error(error instanceof Error ? error.message : "Purge failed"); }
        finally { setBusy(false); }
      }}>Export then purge</button> : null}
    </div>
    <p className="text-sm text-(--muted-foreground)">Selected attempt {selected}: {snapshot?.status ?? "unknown"}; provenance {snapshot?.provenance ?? "unknown"}; {snapshot?.selected ?? "unknown"} selected cases. Legacy provenance cannot support regression gates.</p>
    {snapshot?.run_manifest ? <details><summary>Frozen attempt manifest</summary><pre className="code-panel">{JSON.stringify(snapshot.run_manifest, null, 2)}</pre></details> : null}
  </div></Panel>;
}
