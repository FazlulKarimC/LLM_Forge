import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

const protectedRoute = createRouteMatcher(["/dashboard(.*)", "/experiments(.*)", "/prompts(.*)", "/datasets(.*)", "/evaluations(.*)", "/settings(.*)"]);
const clerkProxy = clerkMiddleware(async (auth, request) => {
  if (protectedRoute(request)) await auth.protect();
}, { signInUrl: "/sign-in", signUpUrl: "/sign-up" });

export default function proxy(...args: Parameters<typeof clerkProxy>) {
  // Missing keys render setup instructions in the layout, never private data.
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return NextResponse.next();
  return clerkProxy(...args);
}

export const config = {
  matcher: ["/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)", "/(api|trpc)(.*)"],
};
