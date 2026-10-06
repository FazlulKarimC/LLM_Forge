"use client";

import { createContext, useContext, useEffect, useSyncExternalStore, type ReactNode } from "react";
import { MoonStar, SunMedium } from "lucide-react";
import { cn } from "@/lib/utils";

import { THEME_KEY } from "@/lib/theme";
const THEME_EVENT = "llmforge-theme";
const ThemeContext = createContext<{ light: boolean; toggle: () => void } | null>(null);

function snapshot() {
  try { return window.localStorage.getItem(THEME_KEY) === "true"; }
  catch { return document.documentElement.dataset.theme === "light"; }
}
function subscribe(notify: () => void) {
  const changed = (event: Event) => {
    if (event instanceof StorageEvent && event.key !== null && event.key !== THEME_KEY) return;
    notify();
  };
  window.addEventListener("storage", changed);
  window.addEventListener(THEME_EVENT, changed);
  return () => { window.removeEventListener("storage", changed); window.removeEventListener(THEME_EVENT, changed); };
}
function toggle() {
  const next = !snapshot();
  document.documentElement.dataset.theme = next ? "light" : "dark";
  try { window.localStorage.setItem(THEME_KEY, String(next)); } catch { /* Theme still works without storage. */ }
  window.dispatchEvent(new Event(THEME_EVENT));
}
export function ThemeProvider({ children }: { children: ReactNode }) {
  const light = useSyncExternalStore(subscribe, snapshot, () => false);
  // Read the live preference during hydration so the server's dark snapshot
  // cannot overwrite the theme already applied by the before-paint bootstrap.
  useEffect(() => { document.documentElement.dataset.theme = snapshot() ? "light" : "dark"; }, [light]);
  return <ThemeContext.Provider value={{ light, toggle }}>{children}</ThemeContext.Provider>;
}
export function useTheme() {
  const theme = useContext(ThemeContext);
  if (!theme) throw new Error("Theme controls require ThemeProvider");
  return theme;
}
export function ThemeToggle({ className, showLabel = false }: { className?: string; showLabel?: boolean }) {
  const { light, toggle } = useTheme();
  const label = light ? "Dark mode" : "Light mode";
  return <button type="button" onClick={toggle} aria-label={label} title={label}
    className={cn("btn-ghost theme-toggle", className)}>
    {light ? <MoonStar aria-hidden="true" className="size-4" /> : <SunMedium aria-hidden="true" className="size-4" />}
    {showLabel && <span>{label}</span>}
  </button>;
}
