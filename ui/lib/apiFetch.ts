// Backend fetch helpers. Attach the NextAuth-issued backend bearer
// token (signed with BACKEND_AUTH_SECRET) to every call so the
// FastAPI app can verify identity + ownership.
//
// Two flavours:
//   `apiFetch(path, init)`, drop-in fetch replacement; uses
//      `getSession()` to look up the current bearer just-in-time, no
//      hooks required. Use from anywhere (event handlers, async
//      flows, component bodies).
//   `apiFetchWithUser(user, path, init)`, same idea but reads the
//      token off a session.user you already have. Saves a function
//      call inside hot loops, and works in tests where getSession()
//      isn't available.
//
// Anonymous endpoints (demo, references/process, embed_crops, ...)
// are fine to call with plain `fetch`, the bearer is just ignored
// when the backend route doesn't have a `current_user` dependency.

import { getSession } from "next-auth/react";

import { perfTimeFetch } from "./perf";

type AuthedUser = { backendToken?: string | null } | null | undefined;

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

// Optional third-arg config.
//   - `noDedup`: bypass the in-flight GET coalesce when the caller
//     needs independent timing per request even when two chunks
//     share a URL key.
export type ApiFetchOpts = {
  noDedup?: boolean;
};

function withApi(path: string): string {
  return path.startsWith("http") ? path : `${API}${path}`;
}

function attachBearer(headersInit: HeadersInit | undefined, token: string | null | undefined): Headers {
  const headers = new Headers(headersInit || {});
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return headers;
}

// Backend-token cache. CRITICAL for project-open latency: next-auth's
// getSession() is NOT cached — it fires a fresh /api/auth/session
// round-trip on EVERY call. Without this cache, every apiFetch paid
// that round-trip BEFORE its backend request even left the browser, so
// opening a project (≈6 apiFetch calls) meant ≈6 serialised
// /api/auth/session hits gating the real /initial + /overview +
// /dataset-stats calls — which is why those arrived at the server
// 10-15s late. We resolve the token once, reuse it for TOKEN_TTL_MS,
// and coalesce concurrent first-callers onto one in-flight lookup.
let _cachedToken: string | null = null;
let _cachedTokenAt = 0; // 0 = never resolved
let _inflightToken: Promise<string | null> | null = null;
const TOKEN_TTL_MS = 60_000;

// Seed the token cache from a session the app already holds (e.g. the
// app shell's useSession()). Called when auth resolves so the FIRST
// apiFetch on project-open skips the getSession() round-trip entirely
// and the backend request fires immediately on click.
export function primeBackendToken(token: string | null | undefined): void {
  _cachedToken = token ?? null;
  _cachedTokenAt = Date.now();
}

// Drop the cached token (call on sign-out so a stale bearer can't ride
// the next request).
export function clearBackendToken(): void {
  _cachedToken = null;
  _cachedTokenAt = 0;
  _inflightToken = null;
}

export async function getBackendToken(): Promise<string | null> {
  const now = Date.now();
  if (_cachedTokenAt !== 0 && now - _cachedTokenAt < TOKEN_TTL_MS) {
    return _cachedToken;
  }
  if (_inflightToken) return _inflightToken;
  _inflightToken = getSession()
    .then((session) => {
      _cachedToken = (session?.user as AuthedUser)?.backendToken ?? null;
      _cachedTokenAt = Date.now();
      _inflightToken = null;
      return _cachedToken;
    })
    .catch(() => {
      // Network blip on /api/auth/session — fall back to the last
      // known token rather than dropping the bearer (which would 401).
      _inflightToken = null;
      return _cachedToken;
    });
  return _inflightToken;
}

// In-flight GET dedup. Multiple components on the same page often
// fetch the same /overview / /annotations URL within milliseconds of
// each other (refs hydration, label-aliases hook, app-shell mount).
// Without dedup the browser fires N parallel requests and Cloudflare
// passes them all through to the origin (cf-cache-status: DYNAMIC),
// so N round-trips happen instead of one. Coalescing them here means
// the second-through-Nth callers piggyback on the first request and
// each get an independent Response (via .clone()) to read.
//
// Only GETs with no body are deduped, anything with a payload or
// non-GET method is assumed to mutate and goes straight through.
const _inflightGets = new Map<string, Promise<Response>>();

function isDedupableGet(init: RequestInit): boolean {
  if (init.body != null) return false;
  if (init.method && init.method.toUpperCase() !== "GET") return false;
  // `cache: "no-store"` only tells the browser HTTP cache to skip
  // storing the response. The in-flight dedup is an in-memory promise
  // share, two concurrent fetches to the same URL can safely await
  // the same network call and each receive a fresh Response.clone()
  // without ever touching the HTTP cache.
  return true;
}

// Single fetch attempt with a given bearer. Factored out of apiFetch so the
// 401 auto-retry can re-run it with a freshly-minted token.
async function _doFetch(
  path: string,
  init: RequestInit,
  opts: ApiFetchOpts,
  token: string | null,
): Promise<Response> {
  const url = withApi(path);
  const finalInit: RequestInit = { ...init, headers: attachBearer(init.headers, token) };
  return perfTimeFetch(path, async () => {
    if (opts.noDedup || !isDedupableGet(init)) {
      return fetch(url, finalInit);
    }
    // Bearer is part of the dedup key so a re-auth mid-page-load
    // doesn't have a stale token piggyback onto a fresh request.
    const key = `${url}|${token ?? ""}`;
    const existing = _inflightGets.get(key);
    if (existing) {
      const resp = await existing;
      return resp.clone();
    }
    const p = fetch(url, finalInit);
    _inflightGets.set(key, p);
    p.finally(() => {
      if (_inflightGets.get(key) === p) _inflightGets.delete(key);
    });
    const resp = await p;
    return resp.clone();
  });
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
  opts: ApiFetchOpts = {},
): Promise<Response> {
  // Resolve the bearer through the TTL cache so we don't pay a
  // getSession() → /api/auth/session round-trip per call (next-auth
  // does NOT cache getSession; see getBackendToken above).
  const token = await getBackendToken();
  const resp = await _doFetch(path, init, opts, token);

  // Auto-heal a stale/expired bearer. A 401 on an authed route means the
  // token the backend saw was rejected (the 60 s-cached token aged past the
  // backend JWT's life, or a refresh hasn't propagated yet). Drop the cache,
  // force a fresh token via getSession (which re-mints the 12 h backend JWT),
  // and retry ONCE. Without this, an expired bearer made auth-gated calls like
  // /api/containers silently return 401 → the Projects row rendered empty as
  // if the user had no projects. If the session is genuinely gone the fresh
  // token is null/unchanged and we surface the original 401 for the caller.
  if (resp.status === 401) {
    clearBackendToken();
    const fresh = await getBackendToken();
    if (fresh && fresh !== token) {
      return _doFetch(path, init, opts, fresh);
    }
  }
  return resp;
}

export function apiFetchWithUser(
  user: AuthedUser,
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  return fetch(withApi(path), {
    ...init,
    headers: attachBearer(init.headers, user?.backendToken ?? null),
  });
}

export function authHeader(user: AuthedUser): Record<string, string> {
  const token = user?.backendToken;
  return token ? { Authorization: `Bearer ${token}` } : {};
}
