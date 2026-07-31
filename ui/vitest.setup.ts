import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// ── global mocks for the external/service edges ────────────────────────
// Route-handler and component tests should never reach a real DB, Stripe,
// email provider, captcha verifier, or analytics sink. Individual tests
// override these with `vi.mocked(...)` / `vi.spyOn(...)` as needed.

vi.mock("@/lib/prisma", () => ({
  prisma: new Proxy(
    {},
    {
      // Any model access (prisma.user, prisma.emailVerification, …) returns
      // a vi.fn() per method so tests can stub return values explicitly.
      get: () =>
        new Proxy(
          {},
          { get: () => vi.fn() },
        ),
    },
  ),
}));

vi.mock("@/lib/turnstile", () => ({
  verifyTurnstile: vi.fn(async () => ({ ok: true })),
}));

vi.mock("@/lib/stripe", () => ({
  stripe: {
    webhooks: { constructEvent: vi.fn() },
    checkout: { sessions: { retrieve: vi.fn() } },
    subscriptions: { retrieve: vi.fn() },
  },
}));

vi.mock("resend", () => ({
  Resend: vi.fn(() => ({ emails: { send: vi.fn(async () => ({ data: {}, error: null })) } })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
  notFound: vi.fn(),
}));

vi.mock("posthog-js", () => ({
  default: { init: vi.fn(), capture: vi.fn(), identify: vi.fn(), reset: vi.fn() },
}));
