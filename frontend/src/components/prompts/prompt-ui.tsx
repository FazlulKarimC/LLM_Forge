export const inputClass = "mt-2 w-full rounded-xl border border-(--border) bg-(--surface-2) px-3 py-2.5 text-sm outline-none focus:border-(--primary) disabled:opacity-60";
export const panelClass = "rounded-2xl border border-(--border) bg-(--surface-1) p-5 sm:p-6";
export const buttonClass = "btn-secondary disabled:opacity-40 disabled:cursor-not-allowed";

export function ErrorMessage({ message }: { message: string | null }) {
  return message ? <p role="alert" className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm">{message}</p> : null;
}

export function errorText(error: unknown) { return error instanceof Error ? error.message : "The request failed. Try again."; }
