import type { ReactNode } from "react";

/** Secondary table fields remain available without crowding narrow row layouts. */
export function TableMetadata({ label, children }: { label: string; children: ReactNode }) {
  return <>
    <span className="table-metadata-desktop">{children}</span>
    <details className="table-metadata-mobile">
      <summary>{label}</summary>
      <div className="pt-1">{children}</div>
    </details>
  </>;
}
