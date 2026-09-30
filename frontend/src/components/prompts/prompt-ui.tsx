export const inputClass = "workbench-input mt-1.5 w-full disabled:opacity-60";
export const panelClass = "panel workbench-panel";
export const buttonClass =
  "btn-secondary disabled:opacity-40 disabled:cursor-not-allowed";

export function ErrorMessage({ message }: { message: string | null }) {
  return message ? (
    <p role="alert" className="alert alert-danger text-sm">
      {message}
    </p>
  ) : null;
}

export function errorText(error: unknown) {
  return error instanceof Error
    ? error.message
    : "The request failed. Try again.";
}
