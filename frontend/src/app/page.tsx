"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Command,
  GitCompareArrows,
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
    title: "Evidence-first overview",
    description: "Watch experiment status, throughput, latency, and failure analysis without leaving the dashboard.",
  },
  {
    icon: GitCompareArrows,
    title: "A/B comparison that explains why",
    description: "Line up metrics, statistical significance, disagreement cases, and output differences in one flow.",
  },
  {
    icon: ShieldCheck,
    title: "RAG, reasoning, and safety coverage",
    description: "Benchmark naive prompting, CoT, ReAct agents, and adversarial datasets against the same shell.",
  },
  {
    icon: TimerReset,
    title: "Performance-aware experimentation",
    description: "Track wall time, batching, caching, and token cost with the same rigor as quality metrics.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-[var(--border)] bg-[color:color-mix(in_oklab,var(--background)_86%,transparent)] backdrop-blur-xl">
        <div className="page-width flex items-center justify-between gap-4 px-4 py-5 sm:px-6">
          <Link href="/" className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]">
              <Activity className="size-5" />
            </div>
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[var(--muted-foreground)]">LLMForge</div>
              <div className="text-lg font-semibold tracking-[-0.04em]">Evaluation Console</div>
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <a href="https://github.com/FazlulKarimC/LLM_Forge" target="_blank" rel="noreferrer" className="btn-ghost hidden sm:inline-flex">
              Repository
            </a>
            <Link href="/dashboard" className="btn-primary">
              Launch app
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
              Built for evaluation, not demos
            </div>
            <div className="space-y-4">
              <h1 className="text-[clamp(3rem,6vw,5.75rem)] font-semibold leading-[0.92] tracking-[-0.06em]">
                Precision UI for LLM experiments.
              </h1>
              <p className="max-w-2xl text-lg leading-8 text-[var(--muted-foreground)]">
                LLMForge lets you configure, run, compare, and inspect reasoning experiments with the feel of a real dev tool.
                It is opinionated about evidence, dense metrics, and operational feedback because those are the details recruiters notice.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Link href="/dashboard" className="btn-primary">
                Open dashboard
                <ArrowRight className="size-4" />
              </Link>
              <Link href="/experiments/new" className="btn-secondary">
                Create experiment
              </Link>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-[var(--muted-foreground)]">
              <span className="chip">Reasoning methods</span>
              <span className="chip">Statistical comparison</span>
              <span className="chip">Optimization profiling</span>
              <span className="chip">Static-first frontend</span>
            </div>
          </motion.div>

          <motion.div initial="hidden" animate="visible" variants={cardVariant} transition={{ delay: 0.08 }} className="panel overflow-hidden">
            <div className="border-b border-[var(--border)] px-5 py-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="section-label">Live preview</div>
                  <div className="section-title">Comparison workspace</div>
                </div>
                <div className="chip">Naive vs CoT</div>
              </div>
            </div>
            <div className="grid gap-4 p-5 lg:grid-cols-[1fr_80px_1fr]">
              <div className="space-y-4 rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
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
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-3">
                  <div className="section-label">Delta</div>
                  <div className="metric-value text-3xl text-[var(--accent)]">+22%</div>
                  <div className="h-full w-2 rounded-full bg-[var(--muted)]">
                    <div className="h-[78%] rounded-full bg-[var(--accent)]" />
                  </div>
                </div>
              </div>

              <div className="space-y-4 rounded-[18px] border border-[color:color-mix(in_oklab,var(--accent)_34%,transparent)] bg-[color:color-mix(in_oklab,var(--accent)_10%,transparent)] p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="section-label">Experiment B</div>
                    <div className="font-semibold">Chain-of-thought</div>
                  </div>
                  <span className="status-pill status-running">running</span>
                </div>
                <div className="grid gap-3">
                  <div className="metric-card border-[color:color-mix(in_oklab,var(--accent)_32%,transparent)]">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value text-3xl">90.1%</div>
                    <div className="metric-caption">31 / 35 exact matches</div>
                  </div>
                  <div className="metric-card border-[color:color-mix(in_oklab,var(--primary)_24%,transparent)]">
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
              <div className="flex size-11 items-center justify-center rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] text-[var(--accent)]">
                <feature.icon className="size-5" />
              </div>
              <h2 className="mt-5 text-xl font-semibold tracking-[-0.04em]">{feature.title}</h2>
              <p className="mt-3 text-sm leading-7 text-[var(--muted-foreground)]">{feature.description}</p>
            </motion.div>
          ))}
        </section>

        <section className="mt-10 panel overflow-hidden">
          <div className="grid gap-0 lg:grid-cols-[0.95fr_1.05fr]">
            <div className="border-b border-[var(--border)] p-6 lg:border-b-0 lg:border-r">
              <div className="section-label">Why it lands</div>
              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.05em]">Recruiters only need ninety seconds.</h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-[var(--muted-foreground)]">
                In that window, the product has to signal real systems thinking: configuration discipline, operational polish,
                strong information hierarchy, and enough motion to feel deliberate without becoming decorative noise.
              </p>
            </div>
            <div className="grid gap-4 p-6 sm:grid-cols-2">
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Signal 01</div>
                <div className="mt-2 text-lg font-semibold">Keyboard-first shell</div>
                <p className="mt-2 text-sm leading-7 text-[var(--muted-foreground)]">Persistent navigation, command palette, and dense panels make the app feel like a tool, not a landing page with forms.</p>
              </div>
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Signal 02</div>
                <div className="mt-2 text-lg font-semibold">Evidence over decoration</div>
                <p className="mt-2 text-sm leading-7 text-[var(--muted-foreground)]">Accuracy, latency, failure modes, and disagreements carry the visual emphasis rather than gradients and badges.</p>
              </div>
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Signal 03</div>
                <div className="mt-2 text-lg font-semibold">Operational empathy</div>
                <p className="mt-2 text-sm leading-7 text-[var(--muted-foreground)]">Cold-start banners, empty states, and inline recovery paths show maturity beyond the happy path.</p>
              </div>
              <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="section-label">Signal 04</div>
                <div className="mt-2 text-lg font-semibold">Implementation discipline</div>
                <p className="mt-2 text-sm leading-7 text-[var(--muted-foreground)]">Everything here is React, Tailwind, and Motion. No paid assets, no backend-only theatrics, no fake complexity.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

