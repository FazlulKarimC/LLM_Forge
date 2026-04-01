/**
 * Route transition template — uses a pure CSS fade-in animation
 * instead of framer-motion to avoid loading the motion runtime
 * on every route change.
 */
export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <div className="route-enter w-full h-full">
      {children}
    </div>
  );
}
