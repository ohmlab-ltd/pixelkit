// Lightweight performance instrumentation. Cheap when disabled; only
// runs the observers + batch writer when NEXT_PUBLIC_PERF_LOG === "1".
//
// What gets collected:
//   - long-task spans (PerformanceObserver, entryTypes: ["longtask"])
//   - fetch marks (callers wrap apiFetch results)
//   - render marks (callers wrap heavy commits)
//   - core web vitals (LCP, INP, CLS via PerformanceObserver — no
//     third-party deps; the numbers aren't perfectly Lighthouse-
//     accurate but match closely enough for trend analysis)
//
// Where it goes:
//   - in-memory ring buffer (last N events)
//   - flushed to POST /api/perf/log every ~30s and on visibility-
//     change → hidden so a closing tab still ships its data
//
// Designed to be safe to import from anywhere; the observers + flush
// loop only spin up once on first use.

const ENABLED = process.env.NEXT_PUBLIC_PERF_LOG === "1";

type EventKind =
  | "long-task"
  | "fetch"
  | "render"
  | "web-vital"
  | "phase"
  | "custom";

export type PerfEvent = {
  kind: EventKind;
  ts: number;
  data?: Record<string, unknown>;
};

const _buffer: PerfEvent[] = [];
const _BUFFER_CAP = 400;
const _FLUSH_INTERVAL_MS = 30_000;
let _flushTimer: number | null = null;
let _observersStarted = false;

// Single session id per page load so we can correlate events from
// the same browsing session in the logs.
function makeSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
const SESSION_ID =
  typeof window === "undefined" ? "ssr" : makeSessionId();

function pushEvent(e: PerfEvent): void {
  if (!ENABLED) return;
  _buffer.push(e);
  // Bounded ring buffer; drop oldest on overflow rather than block.
  if (_buffer.length > _BUFFER_CAP) {
    _buffer.shift();
  }
}

async function flush(): Promise<void> {
  if (!ENABLED) return;
  if (_buffer.length === 0) return;
  // Splice the entire buffer into a local copy and clear so events
  // collected while the POST is in flight don't get dropped.
  const events = _buffer.splice(0);
  try {
    // Best-effort POST. Don't await the response; we don't care
    // about correctness here, and a failed POST shouldn't surface
    // anywhere visible. Use sendBeacon when the page is unloading
    // because that's the only reliable transport during pagehide.
    const body = JSON.stringify({ events, session: SESSION_ID });
    const path = "/api/perf/log";
    // Same base logic as lib/apiFetch.ts (not imported — apiFetch
    // imports this module, so importing back would be circular).
    const apiBase =
      process.env.NEXT_PUBLIC_API_URL ??
      (typeof window !== "undefined" && window.location.port === "3000"
        ? "http://localhost:8001"
        : "");
    const url = `${apiBase}${path}`;
    if (
      document.visibilityState === "hidden" &&
      "sendBeacon" in navigator
    ) {
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon(url, blob);
      return;
    }
    // No bearer here — perf endpoint is intentionally anonymous so
    // we never miss data from an unauthenticated session.
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    });
  } catch {
    /* swallow */
  }
}

function startFlushLoop(): void {
  if (_flushTimer !== null) return;
  _flushTimer = window.setInterval(flush, _FLUSH_INTERVAL_MS);
  // Visibility + pagehide give us a last shot at shipping data
  // before the tab dies. pagehide fires on bf-cache too.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") void flush();
  });
  window.addEventListener("pagehide", () => {
    void flush();
  });
}

function startObservers(): void {
  if (_observersStarted) return;
  if (typeof PerformanceObserver === "undefined") return;
  _observersStarted = true;

  // Long-task spans (>50 ms by spec).
  try {
    const ltObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        pushEvent({
          kind: "long-task",
          ts: Date.now(),
          data: {
            duration: entry.duration,
            startTime: entry.startTime,
            name: entry.name,
          },
        });
      }
    });
    ltObserver.observe({ entryTypes: ["longtask"] });
  } catch {
    /* not supported on this browser */
  }

  // LCP. Latest entry wins; we just push whichever is current so
  // the log shows the trajectory if it kept growing.
  try {
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1] as PerformanceEntry & {
        startTime: number;
      };
      if (last) {
        pushEvent({
          kind: "web-vital",
          ts: Date.now(),
          data: { metric: "LCP", value: last.startTime },
        });
      }
    });
    lcpObserver.observe({ entryTypes: ["largest-contentful-paint"] });
  } catch {
    /* not supported */
  }

  // CLS. Sum of layout-shift entries that weren't user-initiated.
  let clsValue = 0;
  try {
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as unknown as Array<{
        hadRecentInput?: boolean;
        value: number;
      }>) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      pushEvent({
        kind: "web-vital",
        ts: Date.now(),
        data: { metric: "CLS", value: clsValue },
      });
    });
    clsObserver.observe({ entryTypes: ["layout-shift"] });
  } catch {
    /* not supported */
  }
}

function ensureStarted(): void {
  if (!ENABLED) return;
  if (typeof window === "undefined") return;
  startObservers();
  startFlushLoop();
}

// ─── Public API ─────────────────────────────────────────────────────

export function perfMark(name: string, data?: Record<string, unknown>): void {
  if (!ENABLED) return;
  ensureStarted();
  pushEvent({ kind: "custom", ts: Date.now(), data: { name, ...data } });
}

// Wrap a fetch (or any Promise) and log how long it took.
export async function perfTimeFetch<T>(
  path: string,
  body: () => Promise<T>,
  extra?: Record<string, unknown>,
): Promise<T> {
  if (!ENABLED) return body();
  ensureStarted();
  const t0 = performance.now();
  try {
    const out = await body();
    pushEvent({
      kind: "fetch",
      ts: Date.now(),
      data: {
        path,
        elapsed_ms: performance.now() - t0,
        ok: true,
        ...extra,
      },
    });
    return out;
  } catch (e) {
    pushEvent({
      kind: "fetch",
      ts: Date.now(),
      data: {
        path,
        elapsed_ms: performance.now() - t0,
        ok: false,
        error: e instanceof Error ? e.message : String(e),
        ...extra,
      },
    });
    throw e;
  }
}

// Wrap a heavy synchronous compute and log its duration. Useful for
// flagging render / normalisation work that's eating frames.
export function perfTimeSync<T>(name: string, fn: () => T): T {
  if (!ENABLED) return fn();
  ensureStarted();
  const t0 = performance.now();
  const out = fn();
  pushEvent({
    kind: "render",
    ts: Date.now(),
    data: { name, elapsed_ms: performance.now() - t0 },
  });
  return out;
}

// Log a project-load phase event so dashboards can correlate the
// progress-bar phases with real loading timings.
export function perfPhase(phase: string, status: "start" | "end" | "fail", data?: Record<string, unknown>): void {
  if (!ENABLED) return;
  ensureStarted();
  pushEvent({
    kind: "phase",
    ts: Date.now(),
    data: { phase, status, ...data },
  });
}

export function perfEnabled(): boolean {
  return ENABLED;
}

export function perfSession(): string {
  return SESSION_ID;
}
