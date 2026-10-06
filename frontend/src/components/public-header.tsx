import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Brand } from "@/components/brand";
import { ThemeToggle } from "@/components/theme-provider";
import { PublicMobileNav } from "@/components/public-mobile-nav";

export const repositoryUrl = "https://github.com/FazlulKarimC/LLM_Forge";

export function PublicHeader() {
  return <header className="public-header">
    <div className="page-width public-header-inner">
      <Link href="/" aria-label="LLMForge home"><Brand /></Link>
      <nav aria-label="Main navigation" className="public-desktop-nav">
        <Link href="/#product">Product</Link><Link href="/#workflow">Workflow</Link>
        <Link href="/docs">Docs</Link>
        <a href={repositoryUrl} target="_blank" rel="noreferrer">GitHub <ArrowUpRight className="size-3" aria-hidden="true" /></a>
      </nav>
      <div className="flex items-center gap-2">
        <ThemeToggle />
        <Link href="/dashboard" className="btn-primary public-workspace-link">Open workspace</Link>
        <PublicMobileNav />
      </div>
    </div>
  </header>;
}
