import Link from "next/link";
import { Brand } from "@/components/brand";
import { repositoryUrl } from "@/components/public-header";

export function PublicFooter() {
  return <footer className="public-footer">
    <div className="page-width public-footer-inner">
      <div className="flex flex-col gap-3"><Link href="/" aria-label="LLMForge home"><Brand /></Link>
        <p>A personal engineering project. Built in the open.</p></div>
      <nav aria-label="Footer navigation" className="flex flex-wrap gap-6">
        <Link href="/docs">Documentation</Link><Link href="/docs#demo">Demo guide</Link>
        <a href={repositoryUrl} target="_blank" rel="noreferrer">GitHub</a>
      </nav>
    </div>
  </footer>;
}
