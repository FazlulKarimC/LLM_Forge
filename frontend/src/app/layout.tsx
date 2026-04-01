import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: {
    default: "LLMForge",
    template: "%s | LLMForge",
  },
  description: "A precision console for systematic LLM evaluation, comparison, and optimization.",
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
          {children}
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
