import type { Metadata } from "next";

// The /app shell is the workspace - authed-only behaviour, dynamic
// project state, no organic-traffic value. Noindex it so Google
// spends its crawl budget on the marketing + content routes (/,
// /pricing, /guide, /p/[id], policies).
export const metadata: Metadata = {
  title: "Workspace",
  robots: { index: false, follow: false },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
