import { AppShell } from "@/components/app-shell";
import { AuthSetup } from "@/components/auth-setup";
import { WorkspaceProvider } from "@/components/workspace-provider";

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
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return <AuthSetup />;
  return <WorkspaceProvider><AppShell>{children}</AppShell></WorkspaceProvider>;
}
