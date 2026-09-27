import Link from "next/link";

export function AuthSetup() {
  return <main className="page-width flex min-h-screen items-center justify-center px-6">
    <div className="max-w-xl rounded-3xl border border-(--border) bg-(--surface-1) p-8">
      <div className="section-label">Workspace setup</div>
      <h1 className="mt-3 text-3xl font-semibold">Connect authentication to launch LLMForge</h1>
      <p className="mt-4 text-(--muted-foreground)">Create a Clerk application, add its publishable and secret keys to the frontend environment, and configure its issuer URL in the backend. Then restart the app.</p>
      <p className="mt-3 text-sm text-(--muted-foreground)">See docs/PHASE_1_SETUP.md in the repository for the complete setup.</p>
      <div className="mt-6 flex gap-3"><a href="https://dashboard.clerk.com" target="_blank" rel="noreferrer" className="btn-primary">Open Clerk</a><Link href="/" className="btn-secondary">Back home</Link></div>
    </div>
  </main>;
}
