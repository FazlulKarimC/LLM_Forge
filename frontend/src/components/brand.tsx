import { FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";

/** The existing sidebar mark is the single source of product identity. */
export function BrandGlyph({ className }: { className?: string }) {
  return <FlaskConical aria-hidden="true" className={cn("size-5", className)} />;
}

export function Brand({ compact = false, subtitle = "Prompt workspace" }: {
  compact?: boolean; subtitle?: string;
}) {
  return <span className="brand" aria-label={compact ? "LLMForge home" : undefined}>
    <span className="brand-mark"><BrandGlyph /></span>
    {!compact && <span className="brand-wordmark">
      <span className="brand-name">LLMForge</span>
      <span className="brand-subtitle">{subtitle}</span>
    </span>}
  </span>;
}
