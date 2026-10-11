export function PageControls({
  offset,
  next,
  change,
  pageSize = 50,
}: {
  offset: number;
  next: boolean;
  change: (value: number) => void;
  pageSize?: number;
}) {
  if (!offset && !next) return null;
  return (
    <div className="table-pagination">
      <button
        type="button"
        className="btn-secondary"
        disabled={!offset}
        onClick={() => change(Math.max(0, offset - pageSize))}
      >
        Previous
      </button>
      <button
        type="button"
        className="btn-secondary"
        disabled={!next}
        onClick={() => change(offset + pageSize)}
      >
        Next
      </button>
    </div>
  );
}
