"use client";

import { useEffect, useRef, useState, type HTMLAttributes, type ReactNode } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowLeft, CheckCircle2, CircleDashed, LoaderCircle, PauseCircle } from "lucide-react";

import { cn } from "@/lib/utils";

type PanelProps = HTMLAttributes<HTMLDivElement>;

export function Panel({ className, ...props }: PanelProps) {
  return <div className={cn("panel", className)} {...props} />;
}

export function PanelHeader({
  label,
  title,
  description,
  actions,
}: {
  label?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="panel-header">
      <div className="space-y-1.5">
        {label ? <div className="section-label">{label}</div> : null}
        <div className="section-title">{title}</div>
        {description ? <p className="section-description">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function PageHeader({
  backHref,
  backLabel,
  eyebrow,
  title,
  description,
  actions,
  children,
}: {
  backHref?: string;
  backLabel?: string;
  eyebrow?: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="page-header">
      {backHref ? (
        <Link href={backHref} className="inline-flex items-center gap-1.5 text-sm font-medium text-(--muted-foreground) transition-colors hover:text-foreground">
          <ArrowLeft className="size-3.5" />
          {backLabel || "Back"}
        </Link>
      ) : null}
      {eyebrow ? <div className="page-eyebrow">{eyebrow}</div> : null}
      <div className="page-header-row">
        <div className="space-y-3">
          <h1 className="page-title">{title}</h1>
          {description ? <p className="page-description">{description}</p> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </div>
      {children}
    </div>
  );
}

const statusConfig = {
  pending: { icon: CircleDashed, className: "status-pending" },
  queued: { icon: PauseCircle, className: "status-queued" },
  running: { icon: LoaderCircle, className: "status-running" },
  completed: { icon: CheckCircle2, className: "status-completed" },
  failed: { icon: AlertTriangle, className: "status-failed" },
} as const;

export function StatusPill({ status }: { status: keyof typeof statusConfig | string }) {
  const config = statusConfig[status as keyof typeof statusConfig] ?? statusConfig.pending;
  const Icon = config.icon;

  return (
    <span className={cn("status-pill", config.className)}>
      <Icon className={cn("size-3.5", status === "running" ? "animate-spin" : undefined)} />
      {status}
    </span>
  );
}

export function MetricCard({
  label,
  value,
  detail,
  tone,
  className,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: "default" | "accent" | "success" | "warning" | "danger";
  className?: string;
}) {
  const toneClass =
    tone === "accent"
      ? "border-[color:color-mix(in_oklab,var(--accent)_32%,transparent)]"
      : tone === "success"
        ? "border-[color:color-mix(in_oklab,var(--success)_34%,transparent)]"
        : tone === "warning"
          ? "border-[color:color-mix(in_oklab,var(--warning)_34%,transparent)]"
          : tone === "danger"
            ? "border-[color:color-mix(in_oklab,var(--destructive)_34%,transparent)]"
            : "";

  return (
    <div className={cn("metric-card", toneClass, className)}>
      <div className="metric-label">{label}</div>
      <div className="text-3xl font-semibold tracking-[-0.06em] text-foreground">{value}</div>
      {detail ? <div className="metric-caption">{detail}</div> : null}
    </div>
  );
}

/**
 * AnimatedNumber — uses requestAnimationFrame instead of framer-motion.
 * Animates from 0 → value with an ease-out curve over 700ms.
 */
export function AnimatedNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  className,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}) {
  const [display, setDisplay] = useState(`${prefix}${value.toFixed(decimals)}${suffix}`);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const duration = 700;
    const startTime = performance.now();
    const from = 0;

    function easeOut(t: number): number {
      // Same bezier feel as [0.16, 1, 0.3, 1] — fast start, decelerate
      return 1 - Math.pow(1 - t, 3);
    }

    function tick(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easeOut(progress);
      const current = from + (value - from) * eased;
      setDisplay(`${prefix}${current.toFixed(decimals)}${suffix}`);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, decimals, prefix, suffix]);

  return <span className={cn("metric-value", className)}>{display}</span>;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("empty-state", className)}>
      <div className="empty-state-icon">{icon}</div>
      <div className="space-y-2">
        <div className="section-title">{title}</div>
        <p className="section-description">{description}</p>
      </div>
      {action}
    </div>
  );
}

export function SkeletonBlock({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded-[16px]", className)} aria-hidden="true" />;
}

/**
 * MetricBar — uses CSS transition instead of framer-motion.
 * The bar width animates via `transition: width 700ms`.
 */
export function MetricBar({
  value,
  className,
}: {
  value: number;
  className?: string;
}) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    // Trigger the CSS transition by setting width on next frame
    const raf = requestAnimationFrame(() => {
      setWidth(Math.max(0, Math.min(100, value)));
    });
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return (
    <div className={cn("h-2 rounded-full bg-(--muted)", className)}>
      <div
        className="h-full rounded-full bg-(--accent)"
        style={{
          width: `${width}%`,
          transition: "width 700ms cubic-bezier(0.16, 1, 0.3, 1)",
        }}
      />
    </div>
  );
}

export function Keycap({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-7 items-center justify-center rounded-[10px] border border-(--border) bg-(--surface-2) px-2 py-1 font-mono text-[11px] text-(--muted-foreground)">
      {children}
    </kbd>
  );
}
