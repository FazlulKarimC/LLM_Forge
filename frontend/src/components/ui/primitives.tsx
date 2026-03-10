"use client";

import { useEffect, useState, type HTMLAttributes, type ReactNode } from "react";
import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { AlertTriangle, CheckCircle2, CircleDashed, LoaderCircle, PauseCircle } from "lucide-react";

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
  eyebrow,
  title,
  description,
  actions,
  children,
}: {
  eyebrow?: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="page-header">
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
      <div className="text-3xl font-semibold tracking-[-0.06em] text-[var(--foreground)]">{value}</div>
      {detail ? <div className="metric-caption">{detail}</div> : null}
    </div>
  );
}

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
  const motionValue = useMotionValue(0);
  const rounded = useTransform(motionValue, (latest) => latest.toFixed(decimals));
  const [formatted, setFormatted] = useState(`${prefix}${value.toFixed(decimals)}${suffix}`);

  useEffect(() => {
    const controls = animate(motionValue, value, {
      duration: 0.7,
      ease: [0.16, 1, 0.3, 1] as const,
      onUpdate: () => {
        setFormatted(`${prefix}${rounded.get()}${suffix}`);
      },
    });

    return () => controls.stop();
  }, [decimals, motionValue, prefix, rounded, suffix, value]);

  return <span className={cn("metric-value", className)}>{formatted}</span>;
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

export function MetricBar({
  value,
  className,
}: {
  value: number;
  className?: string;
}) {
  return (
    <div className={cn("h-2 rounded-full bg-[var(--muted)]", className)}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] as const }}
        className="h-full rounded-full bg-[var(--accent)]"
      />
    </div>
  );
}

export function Keycap({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-7 items-center justify-center rounded-[10px] border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 font-mono text-[11px] text-[var(--muted-foreground)]">
      {children}
    </kbd>
  );
}

