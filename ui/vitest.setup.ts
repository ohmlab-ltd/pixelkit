import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// next/navigation needs the Next.js runtime; tests override it with vi.mocked or vi.spyOn.

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
  notFound: vi.fn(),
}));
