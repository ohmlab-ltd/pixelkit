// Global backend-auth wiring (Phase 0 security).
//
// The backend now enforces ownership/privacy on every project route. The app
// historically called many of those routes with plain `fetch(${API}/...)`
// (no bearer) and loaded private images via `<img src>` (which can't carry an
// Authorization header). Rather than touch dozens of call sites, we:
//
//   1. installApiAuth(): patch window.fetch so EVERY request to the backend
//      origin carries the bearer (the same token apiFetch uses). Scoped to
//      API-origin URLs; all other fetches pass through untouched. Idempotent.
//   2. ensureAuthCookie(): ask our own Next.js server to set an httpOnly
//      `pk_auth` cookie scoped to `.pixelkit.ai` carrying the backend token,
//      so same-site `<img>` loads to api.pixelkit.ai authenticate too.
//
// The backend reads either the Authorization header OR the pk_auth cookie
// (see gd/auth.py _token_from_request), so both paths satisfy the guards.

"use client";

import { getBackendToken } from "./apiFetch";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

let _installed = false;

function urlOf(input: RequestInfo | URL): string {
  try {
    if (typeof input === "string") return input;
    if (input instanceof URL) return input.href;
    return (input as Request).url;
  } catch {
    return "";
  }
}

/** Patch window.fetch to attach the backend bearer to every API-origin call,
 *  covering legacy plain-fetch call sites. No-op off the browser; idempotent. */
export function installApiAuth(): void {
  if (_installed || typeof window === "undefined") return;
  _installed = true;
  const orig = window.fetch.bind(window);

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = urlOf(input);
    // Only touch calls to our backend; everything else is untouched.
    if (!url || !url.startsWith(API)) {
      return orig(input as RequestInfo | URL, init);
    }
    // Merge headers from init (or from a Request input), then add the bearer
    // if the caller didn't already (apiFetch already sets it -> we skip).
    const base = init?.headers ?? (input instanceof Request ? input.headers : undefined);
    const headers = new Headers(base || {});
    if (!headers.has("Authorization")) {
      try {
        const token = await getBackendToken();
        if (token) headers.set("Authorization", `Bearer ${token}`);
      } catch {
        /* no token -> request goes unauthenticated; public routes still work */
      }
    }
    if (input instanceof Request) {
      // Re-issue by URL so the merged headers take effect (a Request's headers
      // are otherwise immutable once constructed).
      return orig(url, {
        method: input.method,
        body: init?.body,
        mode: input.mode,
        credentials: input.credentials,
        cache: input.cache,
        redirect: input.redirect,
        ...init,
        headers,
      });
    }
    return orig(input as RequestInfo | URL, { ...init, headers });
  };
}

/** Set the httpOnly pk_auth cookie (via our Next server) so <img> image-serve
 *  requests to the backend authenticate. Safe to call repeatedly. */
export async function ensureAuthCookie(): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    await fetch("/api/auth/cookie", { method: "POST" });
  } catch {
    /* best-effort; private <img> loads just fall back to 404 until next try */
  }
}

/** Clear the pk_auth cookie on sign-out. */
export async function clearAuthCookie(): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    await fetch("/api/auth/cookie", { method: "DELETE" });
  } catch {
    /* ignore */
  }
}
