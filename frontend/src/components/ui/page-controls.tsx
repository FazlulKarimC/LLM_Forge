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
    <div className="flex gap-3 text-xs mt-2">
      <button
        disabled={!offset}
        onClick={() => change(Math.max(0, offset - 50))}
      >
        Previous
      </button>
      <button disabled={!next} onClick={() => change(offset + 50)}>
        Next
      </button>
    </div>
  );
}
