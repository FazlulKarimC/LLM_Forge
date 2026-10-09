"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchAPI } from "@/lib/api-client";
import type { components } from "@/lib/generated/api-schema";

type Capabilities = components["schemas"]["EvaluationCapabilities"];
const names: Record<string, string> = { mock: "Demo (mock)", groq: "Groq", openrouter: "OpenRouter", openai: "OpenAI" };

export function ServerCapabilities() {
  const [data, setData] = useState<Capabilities | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetchAPI<Capabilities>("/capabilities", { signal: controller.signal }).then((result) => {
      if (!controller.signal.aborted) setData(result);
    }).catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, []);
  return <div className="flex flex-col gap-2 text-sm" aria-label="Server evaluation capabilities">
    <p role="status" className="text-(--muted-foreground)">
      {data ? `Server generation: ${data.generation_providers.map((p) => names[p] ?? p).join(", ")}. Judges: ${data.judge_providers.map((p) => names[p] ?? p).join(", ")}.`
        : failed ? "Server capabilities are unavailable. Check the deployed API schema before choosing a provider." : "Loading server provider choices…"}
    </p>
    {data && !data.generation_providers.some((p: string) => p === "gemini") ? <p>For Gemini, generate in your own application and submit results with <code>submit_evaluation</code>. <Link className="text-(--accent) underline" href="/docs#external">External results guide</Link></p> : null}
    <p className="field-help">Provider support describes the API contract; live generation and judges require your provider credentials.</p>
  </div>;
}
