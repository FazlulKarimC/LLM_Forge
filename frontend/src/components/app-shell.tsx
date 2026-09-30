"use client";
import {
  useEffect,
  useMemo,
  useRef,
  type RefObject,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  Command,
  FlaskConical,
  Github,
  LayoutDashboard,
  Menu,
  MoonStar,
  Search,
  SunMedium,
  X,
} from "lucide-react";
import { Keycap } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import { UserButton } from "@clerk/nextjs";
import { WorkspaceSwitcher } from "@/components/workspace-provider";
import { FileText, Database, Settings } from "lucide-react";
type NavItem = {
  href: string;
  label: string;
  icon: typeof LayoutDashboard;
  group: "Build" | "Benchmarks" | "Workspace";
  match: (pathname: string) => boolean;
};
const navItems: NavItem[] = [
  {
    href: "/dashboard",
    label: "Overview",
    icon: LayoutDashboard,
    group: "Build",
    match: (pathname) => pathname === "/dashboard",
  },
  {
    href: "/prompts",
    label: "Prompts",
    icon: FileText,
    group: "Build",
    match: (p) => p.startsWith("/prompts"),
  },
  {
    href: "/datasets",
    label: "Datasets",
    icon: Database,
    group: "Build",
    match: (p) => p.startsWith("/datasets"),
  },
  {
    href: "/evaluations",
    label: "Evaluations",
    icon: FlaskConical,
    group: "Build",
    match: (p) => p.startsWith("/evaluations"),
  },
  {
    href: "/experiments",
    label: "Benchmarks",
    icon: FlaskConical,
    group: "Benchmarks",
    match: (pathname) => pathname.startsWith("/experiments"),
  },
  {
    href: "/settings",
    label: "Settings",
    icon: Settings,
    group: "Workspace",
    match: (p) => p.startsWith("/settings"),
  },
  {
    href: "/docs",
    label: "Docs",
    icon: FileText,
    group: "Workspace",
    match: (p) => p.startsWith("/docs"),
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
      window.removeEventListener(
        "llmforge-storage",
        handleStorage as EventListener,
      );
    };
  };
  const getSnapshot = () => {
    if (typeof window === "undefined") {
      return initialValue;
    }
    const stored = window.localStorage.getItem(key);
    return stored == null ? initialValue : stored === "true";
  };
  const value = useSyncExternalStore(
    subscribe,
    getSnapshot,
    () => initialValue,
  );
  const setValue = (nextValue: boolean | ((current: boolean) => boolean)) => {
    if (typeof window === "undefined") {
      return;
    }
    const resolved =
      typeof nextValue === "function" ? nextValue(getSnapshot()) : nextValue;
    window.localStorage.setItem(key, String(resolved));
    window.dispatchEvent(new CustomEvent("llmforge-storage", { detail: key }));
  };
  return [value, setValue] as const;
}
function useOverlayFocus(open: boolean, ref: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusable = () =>
      Array.from(
        ref.current?.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex="0"]',
        ) ?? [],
      ).filter((element) => element.getClientRects().length > 0);
    const frame = requestAnimationFrame(() => focusable()[0]?.focus());
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const elements = focusable();
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", trap);
      document.body.style.overflow = overflow;
      if (previous?.isConnected) previous.focus();
    };
  }, [open, ref]);
}
function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const dialog = useRef<HTMLDivElement>(null);
  useOverlayFocus(open, dialog);
  const [query, setQuery] = useState("");
  const actions = useMemo(
    () => [
      { label: "Open overview", action: () => router.push("/dashboard") },
      { label: "Browse prompts", action: () => router.push("/prompts") },
      { label: "Create prompt", action: () => router.push("/prompts/new") },
      { label: "Browse datasets", action: () => router.push("/datasets") },
      { label: "Open evaluations", action: () => router.push("/evaluations") },
      { label: "Browse benchmarks", action: () => router.push("/experiments") },
      {
        label: "Create benchmark",
        action: () => router.push("/experiments/new"),
      },
      {
        label: "Compare benchmarks",
        action: () => router.push("/experiments/compare"),
      },
      { label: "Open settings", action: () => router.push("/settings") },
      { label: "Open docs", action: () => router.push("/docs") },
      { label: "Open landing page", action: () => router.push("/") },
      {
        label: "Open GitHub repository",
        action: () =>
          window.open(
            "https://github.com/FazlulKarimC/LLM_Forge",
            "_blank",
            "noreferrer",
          ),
      },
    ],
    [router],
  );
  const filtered = actions.filter((item) =>
    item.label.toLowerCase().includes(query.trim().toLowerCase()),
  );
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
            ref={dialog}
            role="dialog"
            aria-modal="true"
            aria-label="Commands"
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
                aria-label="Find a page or action"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Find a page or action"
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
  const isAppRoute = true;
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [collapsed, setCollapsed] = usePersistentState(
    "llmforge.sidebar.collapsed",
    false,
  );
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const mobileDialog = useRef<HTMLDivElement>(null);
  useOverlayFocus(mobileNavOpen, mobileDialog);
  const [isLightTheme, setIsLightTheme] = usePersistentState(
    "llmforge.theme.light",
    false,
  );
  useEffect(() => {
    document.documentElement.dataset.theme = isLightTheme ? "light" : "dark";
  }, [isLightTheme]);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setMobileNavOpen(false);
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
        <CommandPalette
          key={paletteOpen ? "palette-open" : "palette-closed"}
          open={paletteOpen}
          onClose={() => setPaletteOpen(false)}
        />
      </>
    );
  }
  const sidebar = (mobile = false) => {
    const isCollapsed = collapsed && !mobile;
    return (
      <aside
        className={cn(
          "flex h-full flex-col border-r border-(--border) bg-[color-mix(in_oklab,var(--surface-1)_92%,transparent)] backdrop-blur",
          isCollapsed ? "w-[64px]" : "w-(--sidebar-width)",
        )}
      >
        <div className="flex items-center justify-between gap-3 border-b border-(--border) px-3 py-3">
          <Link
            href="/"
            className="flex min-w-0 items-center gap-3"
            onClick={() => setMobileNavOpen(false)}
          >
            <div className="flex size-8 items-center justify-center rounded-lg border border-(--border) bg-(--surface-2) text-(--primary)">
              <FlaskConical className="size-5" />
            </div>
            {!isCollapsed ? (
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold uppercase tracking-[0.18em] text-(--muted-foreground)">
                  LLMForge
                </div>
                <div className="truncate text-sm font-medium">
                  Prompt workspace
                </div>
              </div>
            ) : null}
          </Link>
          <button
            type="button"
            className="hidden! lg:inline-flex! btn-ghost size-7! min-h-0! px-0!"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isCollapsed ? (
              <ChevronRight className="size-4" />
            ) : (
              <ChevronLeft className="size-4" />
            )}
          </button>
        </div>
        <div className="sidebar-navigation flex flex-1 flex-col gap-5 overflow-y-auto px-2 py-4">
          {(["Build", "Benchmarks", "Workspace"] as const).map((group) => (
            <div
              key={group}
              className={cn("space-y-1", group === "Workspace" && "mt-auto")}
            >
              {!isCollapsed ? (
                <div className="px-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-(--muted-foreground)">
                  {group}
                </div>
              ) : null}
              {navItems
                .filter((item) => item.group === group)
                .map((item) => {
                  const Icon = item.icon;
                  const active = item.match(pathname);
                  return (
                    <Link
                      key={item.href}
                      aria-current={active ? "page" : undefined}
                      title={isCollapsed ? item.label : undefined}
                      href={item.href}
                      onClick={() => setMobileNavOpen(false)}
                      className={cn(
                        "sidebar-link flex items-center gap-2.5 rounded-md border px-3 py-2 transition-colors",
                        active
                          ? "border-[color-mix(in_oklab,var(--primary)_38%,transparent)] bg-[color-mix(in_oklab,var(--primary)_14%,transparent)] text-foreground"
                          : "border-transparent text-(--muted-foreground) hover:border-(--border) hover:bg-(--surface-2) hover:text-foreground",
                        isCollapsed ? "justify-center" : "",
                      )}
                    >
                      <Icon className="size-4 shrink-0" />
                      {!isCollapsed ? (
                        <span className="font-medium">{item.label}</span>
                      ) : null}
                    </Link>
                  );
                })}
            </div>
          ))}
        </div>
        <div className="space-y-1 border-t border-(--border) px-2 py-2">
          <button
            type="button"
            className="btn-ghost w-full justify-start"
            aria-label="Open command palette"
            onClick={() => {
              setMobileNavOpen(false);
              setPaletteOpen(true);
            }}
          >
            <Command className="size-4" />
            {!isCollapsed && (
              <>
                <span>Commands</span>
                <span className="ml-auto">
                  <Keycap>Ctrl K</Keycap>
                </span>
              </>
            )}
          </button>
          <button
            type="button"
            className={cn(
              "btn-ghost w-full justify-start",
              isCollapsed ? "px-0! justify-center!" : "",
            )}
            onClick={() => setIsLightTheme((value) => !value)}
          >
            {isLightTheme ? (
              <MoonStar className="size-4" />
            ) : (
              <SunMedium className="size-4" />
            )}
            {!isCollapsed ? (
              <span>{isLightTheme ? "Dark mode" : "Light mode"}</span>
            ) : null}
          </button>
          <a
            href="https://github.com/FazlulKarimC/LLM_Forge"
            target="_blank"
            rel="noreferrer"
            className={cn(
              "btn-ghost w-full justify-start",
              isCollapsed ? "px-0! justify-center!" : "",
            )}
          >
            <Github className="size-4" />
            {!isCollapsed ? <span>Repository</span> : null}
          </a>
        </div>
      </aside>
    );
  };

  return (
    <>
      <div className="app-shell min-h-screen lg:grid lg:grid-cols-[auto_minmax(0,1fr)]">
        <div className="hidden lg:sticky lg:top-0 lg:h-screen lg:block">
          {sidebar()}
        </div>
        <div className="min-w-0">
          <header className="workspace-topbar sticky top-0 z-40 border-b border-(--border) bg-(--background)">
            <div className="flex min-h-14 items-center justify-between gap-2 px-4 sm:px-5">
              <div className="flex min-w-0 items-center gap-2">
                <button
                  type="button"
                  className="btn-ghost lg:hidden! px-2!"
                  onClick={() => setMobileNavOpen(true)}
                  aria-label="Open navigation"
                >
                  <Menu className="size-4" />
                </button>
                <WorkspaceSwitcher />
              </div>
              <div className="flex shrink-0 items-center gap-3">
                <button
                  type="button"
                  className="btn-ghost hidden! sm:inline-flex!"
                  onClick={() => setPaletteOpen(true)}
                >
                  <Search className="size-3.5" />
                  <span>Go to</span>
                  <Keycap>Ctrl K</Keycap>
                </button>
                <UserButton />
              </div>
            </div>
          </header>
          <main className="page-width px-4 py-5 sm:px-5">{children}</main>
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
              ref={mobileDialog}
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              initial={{ x: -28, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -28, opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] as const }}
              className="app-shell h-full w-(--sidebar-width)"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex h-full flex-col border-r border-(--border) bg-(--surface-1) shadow-(--shadow-overlay)">
                <div className="flex items-center justify-end px-3 py-3">
                  <button
                    type="button"
                    className="btn-ghost size-10 rounded-[14px]! px-0!"
                    aria-label="Close navigation"
                    onClick={() => setMobileNavOpen(false)}
                  >
                    <X className="size-4" />
                  </button>
                </div>
                <div className="flex-1 overflow-hidden">{sidebar(true)}</div>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <CommandPalette
        key={paletteOpen ? "palette-open" : "palette-closed"}
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
      />
    </>
  );
}
