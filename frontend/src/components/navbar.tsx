"use client";

import Link from "next/link";
import Image from "next/image";

export function Navbar() {
    return (
        <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-md border-b border-[#E0DEDB]">
            <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 group">
                    <div className="relative size-8 rounded-lg overflow-hidden border border-[#E0DEDB] shadow-sm group-hover:shadow-md transition-shadow">
                        <Image src="/logo.png" alt="LlmForge Logo" fill className="object-cover" />
                    </div>
                    <span className="font-serif text-xl text-[#37322F] tracking-tight">LlmForge</span>
                </Link>

                <nav className="flex items-center gap-6">
                    <Link
                        href="/dashboard"
                        className="text-sm font-medium text-[#605A57] hover:text-[#37322F] transition-colors"
                    >
                        Dashboard
                    </Link>
                    <Link
                        href="/experiments"
                        className="text-sm font-medium text-[#605A57] hover:text-[#37322F] transition-colors"
                    >
                        Experiments
                    </Link>
                    <a
                        href="https://github.com/FazlulKarimC/LLM_Forge"
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm font-medium text-[#605A57] hover:text-[#37322F] transition-colors"
                    >
                        GitHub
                    </a>
                </nav>
            </div>
        </header>
    );
}
