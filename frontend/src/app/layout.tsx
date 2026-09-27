import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";
import { ClerkProvider } from "@clerk/nextjs";

export const metadata: Metadata = {
  title: {
    default: "LLMForge",
    template: "%s | LLMForge",
  },
  description: "Version prompts, evaluate fixed test cases, and release tested prompts through a Python SDK and CI quality gates.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning data-theme="dark">
      <body className="antialiased">
        <Providers>
            {process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY ? (
              <ClerkProvider signInUrl="/sign-in" signUpUrl="/sign-up" signInFallbackRedirectUrl="/dashboard" signUpFallbackRedirectUrl="/dashboard">{children}</ClerkProvider>
            ) : children}
          <Toaster
            position="bottom-right"
            richColors
            theme="dark"
            toastOptions={{
              style: {
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
                color: "var(--foreground)",
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}
