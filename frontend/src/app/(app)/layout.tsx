import { AppShell } from "@/components/app-shell";

/**
 * (app) route-group layout — wraps all dashboard and experiment routes
 * with the sidebar shell and toast provider. The root layout stays minimal
 * so the landing page (`/`) doesn't pay for these client-side imports.
 */
export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <AppShell>{children}</AppShell>
  );
}
