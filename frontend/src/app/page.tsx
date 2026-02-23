"use client";

import { motion, Variants } from "framer-motion";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, BarChart3, Binary, Share2, Sparkles, Zap, Search, ShieldCheck } from "lucide-react";

const fadeIn: Variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
};

const staggerContainer: Variants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.15
        }
    }
};

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-(--bg-page) selection:bg-[#37322F] selection:text-[#F7F5F3] overflow-hidden">
            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 px-6">
                <div className="max-w-7xl mx-auto flex flex-col items-center text-center">
                    <motion.div
                        initial="hidden"
                        animate="visible"
                        variants={staggerContainer}
                        className="max-w-3xl"
                    >
                        <motion.div variants={fadeIn} className="mb-6 flex justify-center">
                            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#E0DEDB] text-xs font-medium text-[#37322F] shadow-sm">
                                <span className="flex size-2 bg-green-500 rounded-full animate-pulse"></span>
                                Free & Open Source Learning Project
                            </span>
                        </motion.div>

                        <motion.h1
                            variants={fadeIn}
                            className="text-[52px] md:text-[80px] font-serif leading-[1.05] text-[#37322F] tracking-tight mb-8"
                        >
                            Systematic LLM <br />
                            <span className="text-[#605A57]">Experimentation.</span>
                        </motion.h1>

                        <motion.p
                            variants={fadeIn}
                            className="text-lg md:text-xl text-[#605A57] leading-relaxed mb-10 max-w-2xl mx-auto"
                        >
                            A config-driven platform to systematically compare reasoning strategies like Naive Prompting, Chain-of-Thought, RAG, and ReAct Agents with comprehensive metrics tracking.
                        </motion.p>

                        <motion.div
                            variants={fadeIn}
                            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
                        >
                            <Link
                                href="/dashboard"
                                className="group relative inline-flex items-center justify-center gap-2 px-8 py-4 bg-[#37322F] text-white rounded-full font-medium overflow-hidden transition-transform hover:scale-105 active:scale-95"
                            >
                                <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                                <span className="relative">Launch Dashboard</span>
                                <ArrowRight className="relative size-4 group-hover:translate-x-1 transition-transform" />
                            </Link>
                            <a
                                href="https://github.com/FazlulKarimC/LLM_Forge"
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white border border-[#E0DEDB] text-[#37322F] rounded-full font-medium hover:bg-[#F7F5F3] shadow-[0px_2px_0px_0px_rgba(55,50,47,0.04)] transition-all hover:-translate-y-0.5"
                            >
                                View on GitHub
                            </a>
                        </motion.div>
                    </motion.div>
                </div>

                {/* Abstract design elements */}
                <div className="absolute top-1/2 left-0 -translate-y-1/2 w-64 h-64 bg-linear-to-tr from-[#E0DEDB]/40 to-transparent rounded-full blur-3xl -z-10" />
                <div className="absolute bottom-0 right-10 w-96 h-96 bg-linear-to-bl from-[#605A57]/10 to-transparent rounded-full blur-3xl -z-10" />
            </section>

            {/* Features Grid */}
            <section className="py-24 bg-white border-y border-[#E0DEDB]">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="mb-16 md:text-center max-w-2xl mx-auto">
                        <h2 className="text-3xl md:text-4xl font-serif text-[#37322F] mb-4">Research-grade methodology, accessible to everyone.</h2>
                        <p className="text-[#605A57] text-lg">Designed to deeply understand the trade-offs in accuracy, latency, and tokens across different AI architectures.</p>
                    </div>

                    <motion.div
                        initial="hidden"
                        whileInView="visible"
                        viewport={{ once: true, margin: "-100px" }}
                        variants={staggerContainer}
                        className="grid grid-cols-1 md:grid-cols-3 gap-8"
                    >
                        {features.map((feature, i) => (
                            <motion.div
                                key={i}
                                variants={fadeIn}
                                className="group p-8 rounded-2xl bg-[#F7F5F3] border border-[#E0DEDB] hover:border-[#605A57]/30 transition-colors"
                            >
                                <div className="size-12 rounded-xl bg-white border border-[#E0DEDB] flex items-center justify-center mb-6 shadow-sm group-hover:scale-110 transition-transform duration-300">
                                    <feature.icon className="size-5 text-[#37322F]" />
                                </div>
                                <h3 className="text-xl font-serif text-[#37322F] mb-3">{feature.title}</h3>
                                <p className="text-[#605A57] leading-relaxed">{feature.description}</p>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* Orchestration Concept Section */}
            <section className="py-24 px-6 overflow-hidden">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center gap-16">
                    <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8 }}
                        className="flex-1"
                    >
                        <h2 className="text-4xl md:text-5xl font-serif text-[#37322F] mb-6 leading-tight">
                            Compare strategies side-by-side.
                        </h2>
                        <p className="text-lg text-[#605A57] mb-8 leading-relaxed">
                            Don't guess what works best. Forge clear hypotheses and run direct technical comparisons. Uncover the exact latency overhead of Chain-of-Thought versus the accuracy gain it yields on complex benchmarks.
                        </p>
                        <ul className="space-y-4">
                            {['Naive Prompting Models', 'Iterative Chain-of-Thought', 'RAG with Vector Retrieval', 'ReAct Tool-use Agents'].map((item, i) => (
                                <li key={i} className="flex items-center gap-3 text-[#37322F] font-medium">
                                    <div className="size-5 rounded-full bg-[#E0DEDB] flex items-center justify-center">
                                        <div className="size-2 rounded-full bg-[#37322F]" />
                                    </div>
                                    {item}
                                </li>
                            ))}
                        </ul>
                    </motion.div>
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8 }}
                        className="flex-1 w-full relative"
                    >
                        <div className="aspect-4/3 rounded-2xl bg-white border border-[#E0DEDB] shadow-xl overflow-hidden flex flex-col">
                            <div className="h-12 border-b border-[#E0DEDB] bg-[#F7F5F3] flex items-center px-4 gap-2">
                                <div className="flex gap-1.5">
                                    <div className="size-3 rounded-full bg-red-400" />
                                    <div className="size-3 rounded-full bg-amber-400" />
                                    <div className="size-3 rounded-full bg-green-400" />
                                </div>
                            </div>
                            <div className="flex-1 p-6 flex flex-col gap-4 bg-[#F7F5F3]/50">
                                <div className="h-4 w-32 bg-[#E0DEDB] rounded-full" />
                                <div className="flex-1 grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-white border border-[#E0DEDB] rounded-lg shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
                                        <div className="h-3 w-16 bg-blue-100 rounded-full" />
                                        <div>
                                            <div className="text-3xl font-serif text-[#37322F] mb-1">92%</div>
                                            <div className="text-xs text-[#605A57]">CoT Accuracy</div>
                                        </div>
                                    </div>
                                    <div className="p-4 bg-white border border-[#E0DEDB] rounded-lg shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
                                        <div className="h-3 w-16 bg-gray-200 rounded-full" />
                                        <div>
                                            <div className="text-3xl font-serif text-[#37322F] mb-1">68%</div>
                                            <div className="text-xs text-[#605A57]">Naive Accuracy</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="h-24 bg-white border border-[#E0DEDB] rounded-lg p-4 flex gap-4 items-end">
                                    <div className="w-1/4 bg-[#E0DEDB] rounded-t-sm" style={{ height: '40%' }} />
                                    <div className="w-1/4 bg-[#37322F] rounded-t-sm" style={{ height: '80%' }} />
                                    <div className="w-1/4 bg-[#E0DEDB] rounded-t-sm" style={{ height: '55%' }} />
                                    <div className="w-1/4 bg-[#605A57] rounded-t-sm" style={{ height: '95%' }} />
                                </div>
                            </div>
                        </div>

                        {/* Decorative dots grid behind screen */}
                        <div className="absolute -bottom-8 -right-8 w-48 h-48 -z-10 grid grid-cols-6 grid-rows-6 gap-2 opacity-20">
                            {Array.from({ length: 36 }).map((_, i) => (
                                <div key={i} className="size-2 rounded-full bg-[#37322F]" />
                            ))}
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* CTA Footer */}
            <footer className="py-16 text-center border-t border-[#E0DEDB] bg-white">
                <h2 className="text-2xl font-serif text-[#37322F] mb-6">Start your first experiment today.</h2>
                <Link
                    href="/dashboard"
                    className="inline-flex items-center justify-center px-8 py-3 bg-[#37322F] text-white rounded-full font-medium transition-transform hover:scale-105"
                >
                    Open Dashboard
                </Link>
                <p className="mt-12 text-sm text-[#605A57]">
                    Built as a personal learning project by Fazlul Karim. <br className="md:hidden" /> Open source and free to explore.
                </p>
            </footer>
        </div>
    );
}

const features = [
    {
        icon: Binary,
        title: "Methodology Comparison",
        description: "Run head-to-head tests between standard prompting, Chain-of-Thought, RAG pipelines, and ReAct agent workflows."
    },
    {
        icon: BarChart3,
        title: "Deep Analytics",
        description: "Track model accuracy, performance latency, exact matches, token-level F1 scores, and inference cost natively."
    },
    {
        icon: Share2,
        title: "Orchestrated Execution",
        description: "Submit experiments matching datasets to models. Background queues handle API failures and batch processing reliably."
    },
    {
        icon: Zap,
        title: "Fast & Interactive",
        description: "Designed using React 19, Next.js 16, and FastAPI. Zero database locks on long generations thanks to asyncio threading."
    },
    {
        icon: Search,
        title: "Retrieval Testing",
        description: "Examine how knowledge graph injection and dense vector retrieval modify response accuracy compared to zero-shot approaches."
    },
    {
        icon: ShieldCheck,
        title: "Guardrails & Alignment",
        description: "Implement custom moderation prompts to test boundaries safely, comparing how strictly different LLMs adhere to system instructions."
    }
];
