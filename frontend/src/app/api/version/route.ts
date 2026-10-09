export const dynamic = "force-static";

export function GET() {
  const commit = process.env.LLMFORGE_FRONTEND_COMMIT || "";
  const buildId = process.env.LLMFORGE_FRONTEND_BUILD_ID || "";
  return Response.json({
    service: "llmforge-frontend",
    commit: /^[a-fA-F0-9]{40}$/.test(commit) ? commit : null,
    build_id: /^[A-Za-z0-9._-]{1,80}$/.test(buildId) ? buildId : null,
  });
}
