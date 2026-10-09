"use client";

import { useEffect, useState } from "react";
import { fetchAPI } from "@/lib/api-client";
import type { components } from "@/lib/generated/api-schema";

type Backend = components["schemas"]["VersionMetadata"];
type Frontend = { commit: string | null; build_id: string | null };
export function DeploymentInfo() {
  const [backend, setBackend] = useState<Backend | null>(null);
  const [frontend, setFrontend] = useState<Frontend | null>(null);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    Promise.allSettled([
      fetchAPI<Backend>("/version", { signal: controller.signal }),
      fetch("/api/version", { signal: controller.signal }).then((res) => { if (!res.ok) throw new Error("Version unavailable"); return res.json() as Promise<Frontend>; }),
    ]).then(([api, web]) => {
      if (controller.signal.aborted) return;
      if (api.status === "fulfilled") setBackend(api.value);
      if (web.status === "fulfilled") setFrontend(web.value);
      setLoaded(true);
    });
    return () => controller.abort();
  }, []);
  return <details className="text-sm">
    <summary>Deployment identity</summary>
    <dl className="flex flex-col gap-2 py-3 break-all">
      <div><dt>Backend API / source commit / build</dt><dd>{loaded ? `${backend?.api_version ?? "unavailable"} / ${backend?.commit ?? "unknown"} / ${backend?.build_id ?? "unknown"}` : "Loading…"}</dd></div>
      <div><dt>Frontend source commit / build</dt><dd>{loaded ? `${frontend?.commit ?? "unknown"} / ${frontend?.build_id ?? "unknown"}` : "Loading…"}</dd></div>
    </dl>
    <p className="field-help">Record SDK package and source versions separately. Unknown build identity does not establish source parity.</p>
  </details>;
}
