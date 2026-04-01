import { SkeletonBlock } from "@/components/ui/primitives";

export default function ExperimentDetailLoading() {
  return (
    <div className="page-stack">
      {/* Page header with back link */}
      <SkeletonBlock className="h-[200px]" />

      {/* Metric cards grid */}
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonBlock key={i} className="h-[134px]" />
        ))}
      </section>

      {/* Results panel */}
      <SkeletonBlock className="h-[500px]" />
    </div>
  );
}
