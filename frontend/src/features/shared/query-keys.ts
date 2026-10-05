import { getProjectCacheScope } from "@/lib/request-context";

/** Keep feature prefixes stable for invalidation; partition all reads by project. */
export function scopedKey(parts: readonly unknown[]) {
  return [parts[0], { project: getProjectCacheScope() }, ...parts.slice(1)] as const;
}
