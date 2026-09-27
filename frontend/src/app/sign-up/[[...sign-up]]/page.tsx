import { SignUp } from "@clerk/nextjs";
import { AuthSetup } from "@/components/auth-setup";

export default function SignUpPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return <AuthSetup />;
  return <main className="flex min-h-screen items-center justify-center px-4"><SignUp routing="path" path="/sign-up" signInUrl="/sign-in" fallbackRedirectUrl="/dashboard" /></main>;
}
