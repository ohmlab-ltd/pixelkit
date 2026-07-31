"use client";

import { SessionProvider } from "next-auth/react";
import type { Session } from "next-auth";

// Portable build: single local user, no accounts. Components throughout the
// app still read useSession() (plan gates, username display, backendToken
// attachment), so until that plumbing is removed in the frontend-slimming
// phase we provide one static, always-authenticated session. refetch is
// fully disabled — there is no /api/auth backend to talk to.
const LOCAL_SESSION: Session = {
  user: {
    id: "local",
    name: "Local",
    username: "local",
    email: "local@pixelkit.local",
    backendToken: "",
    subscriptionPlan: "mega",
    subscriptionStatus: "active",
  },
  expires: "2099-12-31T23:59:59.999Z",
};

export function SessionProviderWrapper({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider
      session={LOCAL_SESSION}
      refetchInterval={0}
      refetchOnWindowFocus={false}
      refetchWhenOffline={false}
    >
      {children}
    </SessionProvider>
  );
}
