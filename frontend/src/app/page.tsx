"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Command,
  Cpu,
  GitCompareArrows,
  Github,
  LayoutDashboard,
  ShieldCheck,
  TimerReset,
} from "lucide-react";

const cardVariant = {
  hidden: { opacity: 0, y: 18, filter: "blur(8px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] as const },
  },
};

const features = [
  {
    icon: LayoutDashboard,
    title: "Operational dashboard",
    description: "Experiment status, throughput, latency, and failure analysis — all in one dense, keyboard-navigable view.",
  },
  {
    icon: GitCompareArrows,
    title: "Side-by-side comparison",
    description: "Accuracy, statistical significance, disagreement cases, and output diffs between any two runs.",
  },
  {
    icon: ShieldCheck,
    title: "Multi-method evaluation",
    description: "Benchmark Naive, Chain-of-Thought, ReAct, and RAG pipelines against the same dataset in one workspace.",
  },
  {
    icon: TimerReset,
    title: "Cost and latency tracking",
    description: "Wall time, token usage, caching efficiency, and per-sample cost alongside every quality metric.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-(--border) bg-[color-mix(in_oklab,var(--background)_86%,transparent)] backdrop-blur-xl">
        <div className="page-width flex items-center justify-between gap-4 px-4 py-5 sm:px-6">
          <Link href="/" className="flex items-center gap-3 min-w-0">
            <div className="flex shrink-0 size-11 items-center justify-center rounded-[18px] border border-(--border) bg-(--surface-2) text-(--primary)">
              <Activity className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold uppercase tracking-[0.18em] text-(--muted-foreground)">LLMForge</div>
              <div className="truncate text-lg font-semibold tracking-[-0.04em]">Evaluation Console</div>
            </div>
          </Link>
          <div className="flex shrink-0 items-center gap-2">
            <a href="https://github.com/FazlulKarimC/LLM_Forge" target="_blank" rel="noreferrer" className="btn-ghost hidden sm:inline-flex">
              Repository
            </a>
            <Link href="/dashboard" className="btn-primary">
              <span className="hidden sm:inline">Launch app</span>
              <span className="sm:hidden">Launch</span>
              <ArrowRight className="size-4" />
            </Link>
          </div>
        </div>
      </header>

      <main className="page-width px-4 py-10 sm:px-6 lg:py-16">
        <section className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <motion.div initial="hidden" animate="visible" variants={cardVariant} className="space-y-6">
            <div className="page-eyebrow">
              <Command className="size-3.5" />
              LLM Evaluation Platform
            </div>
            <div className="space-y-4">
              <h1 className="text-[clamp(2.5rem,8vw,5.75rem)] font-semibold leading-[0.92] tracking-[-0.06em]">
                Run, compare, and optimize LLM experiments.
              </h1>
              <p className="max-w-2xl text-lg leading-8 text-(--muted-foreground)">
                Configure reasoning methods, run side-by-side A/B comparisons, inspect statistical significance, and track latency and token cost — all from one console.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 w-full sm:w-auto">
              <Link href="/dashboard" className="btn-primary w-full sm:w-auto justify-center">
                Open dashboard
                <ArrowRight className="size-4" />
              </Link>
              <Link href="/experiments/new" className="btn-secondary w-full sm:w-auto justify-center">
                Create experiment
              </Link>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-(--muted-foreground)">
              <span className="chip">Chain-of-Thought</span>
              <span className="chip">A/B Comparison</span>
              <span className="chip">Metrics Dashboard</span>
              <span className="chip">RAG Evaluation</span>
            </div>
          </motion.div>

          <motion.div initial="hidden" animate="visible" variants={cardVariant} transition={{ delay: 0.08 }} className="panel overflow-hidden">
            <div className="border-b border-(--border) px-5 py-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="section-label">Live preview</div>
                  <div className="section-title">Comparison workspace</div>
                </div>
                <div className="chip">Naive vs CoT</div>
              </div>
            </div>
            <div className="grid gap-4 p-5 lg:grid-cols-[1fr_80px_1fr]">
              <div className="space-y-4 rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="section-label">Experiment A</div>
                    <div className="font-semibold">Naive baseline</div>
                  </div>
                  <span className="status-pill status-completed">completed</span>
                </div>
                <div className="grid gap-3">
                  <div className="metric-card">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value text-3xl">68.4%</div>
                    <div className="metric-caption">24 / 35 exact matches</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Latency p50</div>
                    <div className="metric-value text-3xl">1380 ms</div>
                    <div className="metric-caption">Fast but low reasoning depth</div>
                  </div>
                </div>
              </div>

              <div className="hidden items-center justify-center lg:flex">
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 rounded-[18px] border border-(--border) bg-(--surface-2) p-3">
                  <div className="section-label">Delta</div>
                  <div className="metric-value text-3xl text-(--accent)">+22%</div>
                  <div className="h-full w-2 rounded-full bg-(--muted)">
                    <div className="h-[78%] rounded-full bg-(--accent)" />
                  </div>
                </div>
              </div>

              <div className="space-y-4 rounded-[18px] border border-[color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color-mix(in_oklab,var(--accent)_10%,transparent)] p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="section-label">Experiment B</div>
                    <div className="font-semibold">Chain-of-thought</div>
                  </div>
                  <span className="status-pill status-running">running</span>
                </div>
                <div className="grid gap-3">
                  <div className="metric-card border-[color-mix(in_oklab,var(--accent)_32%,transparent)]">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value text-3xl">90.1%</div>
                    <div className="metric-caption">31 / 35 exact matches</div>
                  </div>
                  <div className="metric-card border-[color-mix(in_oklab,var(--primary)_24%,transparent)]">
                    <div className="metric-label">Latency p50</div>
                    <div className="metric-value text-3xl">2140 ms</div>
                    <div className="metric-caption">Higher cost, higher reasoning payoff</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        <section className="mt-10 grid gap-4 lg:grid-cols-4">
          {features.map((feature, index) => (
            <motion.div key={feature.title} initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={cardVariant} transition={{ delay: index * 0.05 }} className="panel p-5">
              <div className="flex size-11 items-center justify-center rounded-[18px] border border-(--border) bg-(--surface-2) text-(--accent)">
                <feature.icon className="size-5" />
              </div>
              <h2 className="mt-5 text-xl font-semibold tracking-[-0.04em]">{feature.title}</h2>
              <p className="mt-3 text-sm leading-7 text-(--muted-foreground)">{feature.description}</p>
            </motion.div>
          ))}
        </section>

        <section className="mt-10 panel overflow-hidden">
          <div className="grid gap-0 lg:grid-cols-[0.95fr_1.05fr]">
            <div className="border-b border-(--border) p-6 lg:border-b-0 lg:border-r">
              <div className="section-label">Platform</div>
              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.05em]">Everything you need to evaluate LLMs.</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-(--muted-foreground)">
                From experiment configuration to statistical comparison — a single workspace for the entire evaluation lifecycle.
              </p>
            </div>
            <div className="grid gap-4 p-6 sm:grid-cols-2">
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">Configure</div>
                <div className="mt-2 text-lg font-semibold">Structured experiment setup</div>
                <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">Pick a model, reasoning method, dataset, and run parameters from a structured form — no YAML guesswork.</p>
              </div>
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">Compare</div>
                <div className="mt-2 text-lg font-semibold">Side-by-side analysis</div>
                <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">View accuracy, latency, token cost, and statistical significance between any two completed runs.</p>
              </div>
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">Monitor</div>
                <div className="mt-2 text-lg font-semibold">Real-time observability</div>
                <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">Readiness checks, queue status, cold-start handling, and per-experiment progress from one dashboard.</p>
              </div>
              <div className="rounded-[18px] border border-(--border) bg-(--surface-2) p-4">
                <div className="section-label">Inspect</div>
                <div className="mt-2 text-lg font-semibold">Reasoning trace viewer</div>
                <p className="mt-2 text-sm leading-7 text-(--muted-foreground)">Drill into Chain-of-Thought, ReAct, and naive outputs at the individual sample level to understand why.</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-(--border) bg-[color-mix(in_oklab,var(--surface-1)_60%,transparent)]">
        <div className="page-width px-4 py-16 sm:px-6 lg:py-20">
          <div className="space-y-8">
            <div className="space-y-3">
              <div className="text-[clamp(3rem,7vw,5.5rem)] font-semibold leading-[0.92] tracking-[-0.06em]">
                LLMForge
              </div>
              <p className="max-w-lg text-lg leading-8 text-(--muted-foreground)">
                Open-source LLM evaluation console.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-4">
              <Link href="/dashboard" className="btn-ghost">
                Dashboard
              </Link>
              <Link href="/experiments" className="btn-ghost">
                Experiments
              </Link>
              <Link href="/experiments/new" className="btn-ghost">
                New Run
              </Link>
              <a href="https://github.com/FazlulKarimC/LLM_Forge" target="_blank" rel="noreferrer" className="btn-ghost">
                <Github className="size-4" />
                Source Code
              </a>
            </div>
            <div className="border-t border-(--border) pt-6">
              <div className="flex flex-wrap items-center justify-between gap-4 text-sm text-(--muted-foreground)">
                <span>Built with Next.js, FastAPI, and Framer Motion</span>
                <span className="flex items-center gap-2">
                  <Cpu className="size-3.5" />
                  A Fazlul Karim project
                </span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

