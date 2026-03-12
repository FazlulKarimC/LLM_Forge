"use client";

import { useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  Command,
  FlaskConical,
  GitCompareArrows,
  Github,
  Home,
  LayoutDashboard,
  MoonStar,
  Plus,
  Search,
  SunMedium,
  X,
} from "lucide-react";

import { Keycap } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

type NavItem = {
  href: string;
  label: string;
  icon: typeof LayoutDashboard;
  match: (pathname: string) => boolean;
};

const navItems: NavItem[] = [
  {
    href: "/dashboard",
    label: "Overview",
    icon: LayoutDashboard,
    match: (pathname) => pathname === "/dashboard",
  },
  {
    href: "/experiments",
    label: "Experiments",
    icon: FlaskConical,
    match: (pathname) =>
      pathname === "/experiments" ||
      (pathname.startsWith("/experiments/") &&
        !pathname.startsWith("/experiments/compare") &&
        !pathname.startsWith("/experiments/new")),
  },
  {
    href: "/experiments/compare",
    label: "Compare",
    icon: GitCompareArrows,
    match: (pathname) => pathname.startsWith("/experiments/compare"),
  },
  {
    href: "/experiments/new",
    label: "New Run",
    icon: Plus,
    match: (pathname) => pathname.startsWith("/experiments/new"),
  },
];

function usePersistentState(key: string, initialValue: boolean) {
  const subscribe = (onStoreChange: () => void) => {
    if (typeof window === "undefined") {
      return () => undefined;
    }

    const handleStorage = (event: Event) => {
      if (event instanceof StorageEvent) {
        if (event.key !== null && event.key !== key) {
          return;
        }
      } else {
        const customEvent = event as CustomEvent<string>;
        if (customEvent.detail && customEvent.detail !== key) {
          return;
        }
      }

      onStoreChange();
    };

    window.addEventListener("storage", handleStorage);
    window.addEventListener("llmforge-storage", handleStorage as EventListener);

    return () => {
      window.removeEventListener("storage", handleStorage);
      window.removeEventListener("llmforge-storage", handleStorage as EventListener);
    };
  };

  const getSnapshot = () => {
    if (typeof window === "undefined") {
      return initialValue;
    }

    const stored = window.localStorage.getItem(key);
    return stored == null ? initialValue : stored === "true";
  };

  const value = useSyncExternalStore(subscribe, getSnapshot, () => initialValue);

  const setValue = (nextValue: boolean | ((current: boolean) => boolean)) => {
    if (typeof window === "undefined") {
      return;
    }

    const resolved = typeof nextValue === "function" ? nextValue(getSnapshot()) : nextValue;
    window.localStorage.setItem(key, String(resolved));
    window.dispatchEvent(new CustomEvent("llmforge-storage", { detail: key }));
  };

  return [value, setValue] as const;
}

function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");

  const actions = useMemo(
    () => [
      { label: "Go to dashboard", shortcut: "G D", action: () => router.push("/dashboard") },
      { label: "Browse experiments", shortcut: "G E", action: () => router.push("/experiments") },
      { label: "Create new experiment", shortcut: "G N", action: () => router.push("/experiments/new") },
      { label: "Compare completed runs", shortcut: "G C", action: () => router.push("/experiments/compare") },
      { label: "Open landing page", shortcut: "G H", action: () => router.push("/") },
      {
        label: "Open GitHub repository",
        shortcut: "G G",
        action: () => window.open("https://github.com/FazlulKarimC/LLM_Forge", "_blank", "noreferrer"),
      },
    ],
    [router]
  );

  const filtered = actions.filter((item) => item.label.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-70 flex items-start justify-center bg-black/55 px-4 pt-[12vh] backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] as const }}
            className="w-full max-w-2xl overflow-hidden rounded-[26px] border border-(--border) bg-(--surface-1) shadow-(--shadow-overlay)"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center gap-3 border-b border-(--border) px-5 py-4">
              <Search className="size-4 text-(--muted-foreground)" />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Jump to a route or action"
                className="w-full bg-transparent text-sm outline-none placeholder:text-(--muted-foreground)"
              />
              <Keycap>Esc</Keycap>
            </div>
            <div className="max-h-[420px] overflow-y-auto p-3">
              {filtered.map((item) => (
                <button
                  key={item.label}
                  onClick={() => {
                    item.action();
                    onClose();
                  }}
                  className="flex w-full items-center justify-between rounded-[16px] px-4 py-3 text-left transition-colors hover:bg-(--surface-2)"
                >
                  <span className="font-medium">{item.label}</span>
                  <span className="mono-caption">{item.shortcut}</span>
                </button>
              ))}
              {!filtered.length ? (
                <div className="rounded-[16px] border border-dashed border-(--border-strong) px-4 py-8 text-center text-sm text-(--muted-foreground)">
                  No matching routes.
                </div>
              ) : null}
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "/";
  const isAppRoute = pathname.startsWith("/dashboard") || pathname.startsWith("/experiments");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [collapsed, setCollapsed] = usePersistentState("llmforge.sidebar.collapsed", false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [isLightTheme, setIsLightTheme] = usePersistentState("llmforge.theme.light", false);

  useEffect(() => {
    document.documentElement.dataset.theme = isLightTheme ? "light" : "dark";
  }, [isLightTheme]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }

      if (event.key === "Escape") {
        setPaletteOpen(false);
        setMobileNavOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  if (!isAppRoute) {
    return (
      <>
        {children}
        <CommandPalette key={paletteOpen ? "palette-open" : "palette-closed"} open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      </>
    );
  }

  const sidebar = (
    <aside
      className={cn(
        "flex h-screen flex-col border-r border-(--border) bg-[color-mix(in_oklab,var(--surface-1)_92%,transparent)] backdrop-blur",
        collapsed ? "w-[88px]" : "w-(--sidebar-width)"
      )}
    >
      <div className="flex items-center justify-between gap-3 border-b border-(--border) px-4 py-4">
        <Link href="/" className="flex min-w-0 items-center gap-3" onClick={() => setMobileNavOpen(false)}>
          <div className="flex size-11 items-center justify-center rounded-[18px] border border-(--border) bg-(--surface-2) text-(--primary)">
            <FlaskConical className="size-5" />
          </div>
          {!collapsed ? (
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold uppercase tracking-[0.18em] text-(--muted-foreground)">
                LLMForge
              </div>
              <div className="truncate text-lg font-semibold tracking-[-0.04em]">Evaluation Console</div>
            </div>
          ) : null}
        </Link>
        <button
          type="button"
          className="hidden lg:inline-flex btn-ghost size-10 rounded-[14px]! px-0!"
          onClick={() => setCollapsed((value) => !value)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="size-4" /> : <ChevronLeft className="size-4" />}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-6 px-3 py-4">
        <div className="space-y-2">
          {!collapsed ? <div className="px-3 text-xs font-semibold uppercase tracking-[0.18em] text-(--muted-foreground)">Workspace</div> : null}
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = item.match(pathname);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileNavOpen(false)}
                className={cn(
                  "flex items-center gap-3 rounded-[18px] border px-3 py-3 transition-all",
                  active
                    ? "border-[color-mix(in_oklab,var(--primary)_38%,transparent)] bg-[color-mix(in_oklab,var(--primary)_14%,transparent)] text-foreground"
                    : "border-transparent text-(--muted-foreground) hover:border-(--border) hover:bg-(--surface-2) hover:text-foreground",
                  collapsed ? "justify-center" : ""
                )}
              >
                <Icon className="size-4 shrink-0" />
                {!collapsed ? <span className="font-medium">{item.label}</span> : null}
              </Link>
            );
          })}
        </div>

        {!collapsed ? (
          <div className="rounded-[20px] border border-(--border) bg-(--surface-2) p-4">
            <div className="section-label">Shortcut</div>
            <div className="mt-2 text-sm font-medium">Command palette</div>
            <p className="mt-1 text-sm text-(--muted-foreground)">
              Jump between experiment flows without leaving the keyboard.
            </p>
            <button type="button" className="btn-secondary mt-4 w-full justify-between" onClick={() => setPaletteOpen(true)}>
              <span className="inline-flex items-center gap-2">
                <Command className="size-4" />
                Open palette
              </span>
              <span className="inline-flex items-center gap-1">
                <Keycap>Ctrl</Keycap>
                <Keycap>K</Keycap>
              </span>
            </button>
          </div>
        ) : null}
      </div>

      <div className="space-y-2 border-t border-(--border) px-3 py-4">
        <button
          type="button"
          className={cn("btn-secondary w-full justify-start", collapsed ? "px-0 justify-center" : "")}
          onClick={() => setIsLightTheme((value) => !value)}
        >
          {isLightTheme ? <MoonStar className="size-4" /> : <SunMedium className="size-4" />}
          {!collapsed ? <span>{isLightTheme ? "Dark mode" : "Light mode"}</span> : null}
        </button>
        <a
          href="https://github.com/FazlulKarimC/LLM_Forge"
          target="_blank"
          rel="noreferrer"
          className={cn("btn-ghost w-full justify-start", collapsed ? "px-0 justify-center" : "")}
        >
          <Github className="size-4" />
          {!collapsed ? <span>Repository</span> : null}
        </a>
      </div>
    </aside>
  );

  return (
    <>
      <div className="app-shell min-h-screen lg:grid lg:grid-cols-[auto_minmax(0,1fr)]">
        <div className="hidden lg:sticky lg:top-0 lg:h-screen lg:block">{sidebar}</div>
        <div className="min-w-0 overflow-y-auto">
          <header className="sticky top-0 z-40 border-b border-(--border) bg-[color-mix(in_oklab,var(--background)_78%,transparent)] backdrop-blur-xl">
            <div className="page-width flex items-center justify-between gap-3 px-4 py-4 sm:px-6">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="btn-secondary lg:hidden size-11 rounded-[16px]! px-0!"
                  onClick={() => setMobileNavOpen(true)}
                  aria-label="Open navigation"
                >
                  <Command className="size-4" />
                </button>
                <button type="button" className="btn-secondary" onClick={() => setPaletteOpen(true)}>
                  <Search className="size-4" />
                  Search
                  <span className="hidden items-center gap-1 sm:inline-flex">
                    <Keycap>Ctrl</Keycap>
                    <Keycap>K</Keycap>
                  </span>
                </button>
              </div>
              <div className="hidden items-center gap-2 sm:flex">
                <Link href="/" className="btn-ghost">
                  <Home className="size-4" />
                  Home
                </Link>
                <a href="https://github.com/FazlulKarimC/LLM_Forge" target="_blank" rel="noreferrer" className="btn-ghost">
                  <Github className="size-4" />
                  GitHub
                </a>
              </div>
            </div>
          </header>
          <main className="page-width px-4 py-6 sm:px-6 lg:py-8">{children}</main>
        </div>
      </div>

      <AnimatePresence>
        {mobileNavOpen ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-60 bg-black/55 backdrop-blur-sm lg:hidden"
            onClick={() => setMobileNavOpen(false)}
          >
            <motion.div
              initial={{ x: -28, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -28, opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] as const }}
              className="h-full w-[min(90vw,320px)]"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex h-full flex-col border-r border-(--border) bg-(--surface-1) shadow-(--shadow-overlay)">
                <div className="flex items-center justify-end px-3 py-3">
                  <button type="button" className="btn-ghost size-10 rounded-[14px]! px-0!" onClick={() => setMobileNavOpen(false)}>
                    <X className="size-4" />
                  </button>
                </div>
                <div className="flex-1">{sidebar}</div>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <CommandPalette key={paletteOpen ? "palette-open" : "palette-closed"} open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}

