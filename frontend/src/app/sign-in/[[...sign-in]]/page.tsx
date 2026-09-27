import { SignIn } from "@clerk/nextjs";
import { AuthSetup } from "@/components/auth-setup";

export default function SignInPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return <AuthSetup />;
  return <main className="flex min-h-screen items-center justify-center px-4"><SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" fallbackRedirectUrl="/dashboard" /></main>;
}
