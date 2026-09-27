// The workspace gate supplies context; session tokens are never persisted.
type ApiContext = { getToken: () => Promise<string | null>; projectId?: string };
let context: ApiContext | null = null;

export function setApiContext(value: ApiContext | null) {
  context = value;
}

export async function getContextHeaders(): Promise<Record<string, string>> {
  const current = context;
  if (!current) return {};
  const token = await current.getToken();
  if (!token) throw new Error("Your session has expired. Sign in again.");
  return { Authorization: `Bearer ${token}`, ...(current.projectId ? { "X-Project-ID": current.projectId } : {}) };
}
