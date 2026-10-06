"use client";

import Link from "next/link";
import { Menu } from "lucide-react";
import { useRef } from "react";

export function PublicMobileNav() {
  const menu = useRef<HTMLDetailsElement>(null);
  function close() { if (menu.current) menu.current.open = false; }
  return <details ref={menu} className="public-mobile-nav" onKeyDown={(event) => {
    if (event.key === "Escape") { close(); menu.current?.querySelector("summary")?.focus(); }
  }}>
    <summary className="btn-ghost" aria-label="Navigation"><Menu aria-hidden="true" className="size-5" /></summary>
    <nav aria-label="Mobile navigation" onClick={close}>
      <Link href="/#product">Product</Link><Link href="/#workflow">Workflow</Link>
      <Link href="/docs">Docs</Link>
      <a href="https://github.com/FazlulKarimC/LLM_Forge" target="_blank" rel="noreferrer">GitHub</a>
      <Link href="/dashboard">Open workspace</Link>
    </nav>
  </details>;
}
