// Backend fetch helpers (portable build: single local user, no auth).
//
// `apiFetch(path, init)` is a drop-in fetch replacement that prefixes
// the engine base URL and coalesces concurrent identical GETs.

import { perfTimeFetch } from "./perf";

// Engine base URL. Empty string = same-origin: in the packaged build the
// engine serves the static UI itself, so relative paths hit it directly.
// The :3000 check keeps `next dev` working against a locally-running
// engine on its default port.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// Optional third-arg config.
//   - `noDedup`: bypass the in-flight GET coalesce when the caller
//     needs independent timing per request even when two chunks
//     share a URL key.
export type ApiFetchOpts = {
  noDedup?: boolean;
};

function withApi(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

// In-flight GET dedup. Multiple components on the same page often
// fetch the same /overview / /annotations URL within milliseconds of
// each other (refs hydration, label-aliases hook, app-shell mount).
// Without dedup the browser fires N parallel requests, so N
// round-trips happen instead of one. Coalescing them here means
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

export async function apiFetch(
  path: string,
  init: RequestInit = {},
  opts: ApiFetchOpts = {},
): Promise<Response> {
  const url = withApi(path);
  return perfTimeFetch(path, async () => {
    if (opts.noDedup || !isDedupableGet(init)) {
      return fetch(url, init);
    }
    const existing = _inflightGets.get(url);
    if (existing) {
      const resp = await existing;
      return resp.clone();
    }
    const p = fetch(url, init);
    _inflightGets.set(url, p);
    p.finally(() => {
      if (_inflightGets.get(url) === p) _inflightGets.delete(url);
    });
    const resp = await p;
    return resp.clone();
  });
}
