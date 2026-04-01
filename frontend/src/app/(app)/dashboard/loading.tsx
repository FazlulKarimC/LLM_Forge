import { SkeletonBlock } from "@/components/ui/primitives";

export default function DashboardLoading() {
  return (
    <div className="page-stack">
      {/* Page header skeleton */}
      <SkeletonBlock className="h-[180px]" />

      {/* Metric cards grid */}
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonBlock key={i} className="h-[134px]" />
        ))}
      </section>

      {/* Readiness panel */}
      <SkeletonBlock className="h-[180px]" />

      {/* Recent experiments panel */}
      <SkeletonBlock className="h-[400px]" />
    </div>
  );
}
