// Engine fetch: adds the base URL and coalesces concurrent identical GETs.

// Empty means same origin, where the engine serves the UI. Port 3000 is next dev.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// noDedup skips GET coalescing, for independent timing per request.
type ApiFetchOpts = {
  noDedup?: boolean;
};

function withApi(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

// In-flight GETs by URL; each caller reads its own clone of the Response.
const _inflightGets = new Map<string, Promise<Response>>();

function isDedupableGet(init: RequestInit): boolean {
  if (init.body != null) return false;
  if (init.method && init.method.toUpperCase() !== "GET") return false;
  // cache: "no-store" affects only the HTTP cache, so it does not stop coalescing.
  return true;
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
  opts: ApiFetchOpts = {},
): Promise<Response> {
  const url = withApi(path);
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
  // then(done, done) rather than finally: a failed GET rejects only for its callers.
  const done = () => {
    if (_inflightGets.get(url) === p) _inflightGets.delete(url);
  };
  p.then(done, done);
  const resp = await p;
  return resp.clone();
}
