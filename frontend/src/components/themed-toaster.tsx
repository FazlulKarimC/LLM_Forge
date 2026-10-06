"use client";

import { Toaster } from "sonner";
import { useTheme } from "@/components/theme-provider";

export function ThemedToaster() {
  const { light } = useTheme();
  return <Toaster position="bottom-right" richColors theme={light ? "light" : "dark"}
    toastOptions={{ style: { background: "var(--surface-1)", border: "1px solid var(--border)",
      color: "var(--foreground)", borderRadius: "7px" } }} />;
}
