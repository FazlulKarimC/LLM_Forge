"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
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

/* ─── animated mesh-gradient shader (minimalistic, subtle) ─── */

const VERTEX_SRC = `
  attribute vec2 a_position;
  void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
`;

const FRAGMENT_SRC = `
  precision mediump float;
  uniform float u_time;
  uniform vec2  u_resolution;
  uniform vec3  u_color_a;
  uniform vec3  u_color_b;
  uniform vec3  u_bg;

  void main() {
    vec2 p = (gl_FragCoord.xy / u_resolution.xy) - 0.5;
    p.x *= u_resolution.x / u_resolution.y;

    float t = u_time * 0.15;
    float v = 0.0;
    vec2 c = p * 2.0;

    for (float i = 1.0; i < 4.0; i++) {
        c.x += sin(t * 0.5 * i + c.y * i + t) * 0.5;
        c.y += cos(t * 0.4 * i + c.x * i - t) * 0.5;
        v += sin(c.x + t) * cos(c.y + t);
    }

    v = smoothstep(-1.5, 1.5, v);

    // Blend between two theme-aware accent colors over the background
    vec3 color = mix(u_color_a, u_color_b, v);
    // Mix with background so the effect is subtle — 18% color, 82% bg
    color = mix(u_bg, color, 0.18);
    gl_FragColor = vec4(color, 1.0);
  }
`;

function getThemeColors() {
  if (typeof window === "undefined") {
    return { colorA: [0.851, 0.549, 0.337] as [number, number, number], colorB: [0.337, 0.635, 0.620] as [number, number, number], bg: [0.035, 0.035, 0.043] as [number, number, number] };
  }

  let isLight = false;
  try {
    isLight = JSON.parse(window.localStorage.getItem("llmforge.theme.light") ?? "false");
  } catch {
    // ignore
  }

  return {
    colorA: [0.851, 0.549, 0.337] as [number, number, number], // Primary
    colorB: [0.337, 0.635, 0.620] as [number, number, number], // Accent
    bg: isLight 
      ? [0.988, 0.988, 0.992] as [number, number, number]  // Light background
      : [0.035, 0.035, 0.043] as [number, number, number], // Dark background
  };
}

function useShader(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) return;

    const createShader = (type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    };

    const vShader = createShader(gl.VERTEX_SHADER, VERTEX_SRC);
    const fShader = createShader(gl.FRAGMENT_SHADER, FRAGMENT_SRC);
    if (!vShader || !fShader) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vShader);
    gl.attachShader(program, fShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    // Full-screen quad
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, "u_time");
    const uRes = gl.getUniformLocation(program, "u_resolution");
    const uColorA = gl.getUniformLocation(program, "u_color_a");
    const uColorB = gl.getUniformLocation(program, "u_color_b");
    const uBg = gl.getUniformLocation(program, "u_bg");

    // Push current theme colors into the shader
    const syncThemeColors = () => {
      const { colorA, colorB, bg } = getThemeColors();
      gl.uniform3f(uColorA, colorA[0], colorA[1], colorA[2]);
      gl.uniform3f(uColorB, colorB[0], colorB[1], colorB[2]);
      gl.uniform3f(uBg, bg[0], bg[1], bg[2]);
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio, 1.0);
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };

    resize();
    syncThemeColors();
    window.addEventListener("resize", resize);

    // Re-sync colors when theme toggles (observed via attribute change)
    const observer = new MutationObserver(() => syncThemeColors());
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    let frameId: number;
    const start = performance.now();

    const tick = () => {
      gl.uniform1f(uTime, (performance.now() - start) / 1000);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      frameId = requestAnimationFrame(tick);
    };

    tick();

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, []);
}

/* ─── animation variants ─── */

const fadeUp = {
  hidden: { opacity: 0, y: 20, filter: "blur(6px)" },
  visible: (delay: number) => ({
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] as const },
  }),
};

const wordPull = {
  hidden: { opacity: 0, y: 14, filter: "blur(4px)" },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.5, delay: 0.15 + i * 0.06, ease: [0.16, 1, 0.3, 1] as const },
  }),
};

const headlineWords = ["Run,", "compare,", "and", "optimize", "LLM", "experiments."];

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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useShader(canvasRef);

  return (
    <div className="min-h-screen">
      {/* navbar */}
      <header className="relative z-10 border-b border-(--border) bg-[color-mix(in_oklab,var(--background)_60%,transparent)] backdrop-blur-xl">
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

      {/* hero section with shader background */}
      <section className="relative min-h-[min(85vh,820px)] flex items-center overflow-hidden">
        {/* shader canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full"
          style={{ 
            filter: "blur(30px) saturate(1.4)",
          }}
          aria-hidden="true"
        />
        {/* gradient overlays for blending into page */}
        <div className="pointer-events-none absolute inset-0 bg-linear-to-b from-background via-transparent to-background" style={{ opacity: 0.5 }} />
        <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-32 bg-linear-to-t from-background to-transparent" />

        <div className="relative z-10 page-width w-full px-4 py-20 sm:px-6 lg:py-28">
          <div className="mx-auto max-w-4xl text-center space-y-8">
            {/* eyebrow */}
            <motion.div
              initial="hidden"
              animate="visible"
              custom={0}
              variants={fadeUp}
              className="inline-flex items-center gap-2 page-eyebrow"
            >
              <Command className="size-3.5" />
              LLM Evaluation Platform
            </motion.div>

            {/* headline — word-by-word animated */}
            <h1 className="text-[clamp(2.8rem,9vw,6.5rem)] font-semibold leading-[0.9] tracking-[-0.06em]">
              {headlineWords.map((word, i) => (
                <motion.span
                  key={word}
                  initial="hidden"
                  animate="visible"
                  custom={i}
                  variants={wordPull}
                  className="inline-block mr-[0.28em]"
                  style={
                    word === "optimize" || word === "LLM"
                      ? { color: "color-mix(in oklab, var(--primary) 90%, white 10%)" }
                      : undefined
                  }
                >
                  {word}
                </motion.span>
              ))}
            </h1>

            {/* subtitle */}
            <motion.p
              initial="hidden"
              animate="visible"
              custom={0.55}
              variants={fadeUp}
              className="mx-auto max-w-2xl text-lg leading-8 text-(--muted-foreground)"
            >
              Configure reasoning methods, run side-by-side A/B comparisons,
              inspect statistical significance, and track latency and token cost
              — all from one console.
            </motion.p>

            {/* buttons */}
            <motion.div
              initial="hidden"
              animate="visible"
              custom={0.7}
              variants={fadeUp}
              className="flex flex-col sm:flex-row items-center justify-center gap-3"
            >
              <Link href="/dashboard" className="btn-primary w-full sm:w-auto justify-center">
                Open dashboard
                <ArrowRight className="size-4" />
              </Link>
              <Link href="/experiments/new" className="btn-secondary w-full sm:w-auto justify-center">
                Create experiment
              </Link>
            </motion.div>

            {/* chips */}
            <motion.div
              initial="hidden"
              animate="visible"
              custom={0.85}
              variants={fadeUp}
              className="flex flex-wrap justify-center gap-3 text-sm text-(--muted-foreground)"
            >
              <span className="chip">Chain-of-Thought</span>
              <span className="chip">A/B Comparison</span>
              <span className="chip">Metrics Dashboard</span>
              <span className="chip">RAG Evaluation</span>
            </motion.div>
          </div>
        </div>
      </section>

      {/* features grid */}
      <main className="page-width px-4 sm:px-6">
        <section className="grid gap-4 lg:grid-cols-4">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-80px" }}
              custom={index * 0.06}
              variants={fadeUp}
              className="panel p-5"
            >
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

      {/* footer */}
      <footer className="mt-16 border-t border-(--border) bg-[color-mix(in_oklab,var(--surface-1)_60%,transparent)]">
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
                New experiment
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
