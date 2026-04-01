import { SkeletonBlock } from "@/components/ui/primitives";

export default function ExperimentsLoading() {
  return (
    <div className="page-stack">
      {/* Page header skeleton */}
      <SkeletonBlock className="h-[160px]" />

      {/* Filter panel */}
      <SkeletonBlock className="h-[140px]" />

      {/* Experiment rows */}
      <div className="panel">
        <div className="panel-body space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonBlock key={i} className="h-[124px]" />
          ))}
        </div>
      </div>
    </div>
  );
}
