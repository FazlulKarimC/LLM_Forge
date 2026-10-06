import { SignIn } from "@clerk/nextjs";
import { AuthSetup } from "@/components/auth-setup";
import { AuthFrame, authAppearance } from "@/components/auth-frame";

export default function SignInPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) return <AuthSetup />;
  return <AuthFrame><SignIn appearance={authAppearance} routing="path" path="/sign-in" signUpUrl="/sign-up" fallbackRedirectUrl="/dashboard" /></AuthFrame>;
}
