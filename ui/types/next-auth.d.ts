// Session type augmentation. This lived in auth.ts in the SaaS build; the
// portable build has no NextAuth config, but components still type against
// these fields until the session plumbing is removed in the frontend-slimming
// phase.
import type { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      username?: string | null;
      // SaaS-era HS256 JWT for the FastAPI backend. Always empty in the
      // portable build — the local engine does not require auth.
      backendToken?: string | null;
      subscriptionPlan?: string | null;
      subscriptionStatus?: string | null;
      betaExpiresAt?: string | null;
    } & DefaultSession["user"];
  }
}
