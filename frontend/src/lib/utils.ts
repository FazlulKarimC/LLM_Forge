import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utility to merge Tailwind CSS classes, handling conflicts intelligently.
 */
export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}
