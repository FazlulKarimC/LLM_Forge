export function PageControls({
  offset,
  next,
  change,
}: {
  offset: number;
  next: boolean;
  change: (value: number) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      <button
        type="button"
        className="btn-secondary"
        disabled={!offset}
        onClick={() => change(Math.max(0, offset - 50))}
      >
        Previous
      </button>
      <button
        type="button"
        className="btn-secondary"
        disabled={!next}
        onClick={() => change(offset + 50)}
      >
        Next
      </button>
    </div>
  );
}
