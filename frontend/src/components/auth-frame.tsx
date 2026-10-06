import type { ReactNode } from "react";
import { Brand } from "@/components/brand";
import { PublicHeader } from "@/components/public-header";
import { PublicFooter } from "@/components/public-footer";

export function AuthFrame({ children }: { children: ReactNode }) {
  return <div className="public-shell"><PublicHeader />
    <main className="auth-page"><Brand />{children}</main><PublicFooter />
  </div>;
}

export const authAppearance = {
  variables: {
    colorPrimary: "var(--primary)", colorBackground: "var(--surface-1)",
    colorForeground: "var(--foreground)", colorMutedForeground: "var(--muted-foreground)",
    colorInput: "var(--surface-2)", colorInputForeground: "var(--foreground)",
    colorPrimaryForeground: "var(--primary-foreground)", borderRadius: "7px",
    fontFamily: "var(--font-body)",
  },
  elements: { card: "border border-(--border) shadow-none", footer: "bg-(--surface-1)",
    formButtonPrimary: "shadow-none", logoBox: "hidden" },
};
