import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { ThemedToaster } from "@/components/themed-toaster";
import { ClerkProvider } from "@clerk/nextjs";
import { themeBootstrap } from "@/lib/theme";

export const metadata: Metadata = {
  title: {
    default: "LLMForge",
    template: "%s | LLMForge",
  },
  description: "Compare model benchmarks, develop versioned prompts, and evaluate outputs in one workspace with LLMForge.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning data-theme="dark">
      <head><script dangerouslySetInnerHTML={{ __html: themeBootstrap }} /></head>
      <body className="antialiased">
        <Providers>
            {process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY ? (
              <ClerkProvider signInUrl="/sign-in" signUpUrl="/sign-up" signInFallbackRedirectUrl="/dashboard" signUpFallbackRedirectUrl="/dashboard">{children}</ClerkProvider>
            ) : children}
          <ThemedToaster />
        </Providers>
      </body>
    </html>
  );
}
