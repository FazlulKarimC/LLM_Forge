import { SignUp } from "@clerk/nextjs";
import { AuthSetup } from "@/components/auth-setup";
import { AuthFrame, authAppearance } from "@/components/auth-frame";

export default function SignUpPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return <AuthSetup />;
  return <AuthFrame><SignUp appearance={authAppearance} routing="path" path="/sign-up" signInUrl="/sign-in" fallbackRedirectUrl="/dashboard" /></AuthFrame>;
}
