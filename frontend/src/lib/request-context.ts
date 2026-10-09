// The workspace gate supplies context; session tokens are never persisted.
type ApiContext = { getToken: () => Promise<string | null>; projectId?: string };
let context: ApiContext | null = null;

export class SessionExpiredError extends Error {
  constructor() {
    super("Your session has expired. Refresh your session or sign in again.");
    this.name = "SessionExpiredError";
  }
}

export function getProjectCacheScope() { return context?.projectId ?? "unselected"; }

export function setApiContext(value: ApiContext | null) {
  context = value;
}

export async function getContextHeaders(): Promise<Record<string, string>> {
  const current = context;
  if (!current) return {};
  const token = await current.getToken();
  if (!token) throw new SessionExpiredError();
  return { Authorization: `Bearer ${token}`, ...(current.projectId ? { "X-Project-ID": current.projectId } : {}) };
}
