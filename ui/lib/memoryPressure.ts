// Polls performance.memory and fires a callback when the heap
// crosses a configurable fraction of its limit. Used by the V2
// project view to evict in-RAM mask polygons from imports that
// aren't currently visible — the existing 30 s TTL strips them
// based on user navigation, this is the proactive complement
// driven by actual pressure.
//
// performance.memory is Chromium-only (Chrome, Edge, Brave, Opera).
// Firefox + Safari don't expose it; on those browsers the subscribe
// call is a no-op so the project keeps the in-RAM behaviour it has
// today (the TTL still runs).
//
// Flag: NEXT_PUBLIC_MEM_PRESSURE=1.

type Listener = () => void;

type ChromiumMemory = {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
};

function readMemory(): ChromiumMemory | null {
  if (typeof performance === "undefined") return null;
  const mem = (performance as unknown as { memory?: ChromiumMemory }).memory;
  if (
    !mem ||
    typeof mem.usedJSHeapSize !== "number" ||
    typeof mem.jsHeapSizeLimit !== "number"
  ) {
    return null;
  }
  return mem;
}

export function isMemoryApiAvailable(): boolean {
  return readMemory() !== null;
}

export const MEM_PRESSURE_ENABLED =
  typeof process !== "undefined" &&
  process.env.NEXT_PUBLIC_MEM_PRESSURE === "1";

const POLL_MS = 5000;
// Cool-down between consecutive fires — a single eviction pass
// can take a beat to land back into state; firing again 5 s later
// while the heap hasn't moved yet would do nothing useful.
const COOLDOWN_MS = 20000;

// Subscribe to a pressure threshold. `threshold` is a 0..1 fraction
// of the heap limit; when usedJSHeapSize crosses it, the listener
// fires (with a cool-down to avoid storms). Returns an unsubscribe
// fn.
export function subscribePressure(
  threshold: number,
  listener: Listener,
): () => void {
  if (!MEM_PRESSURE_ENABLED) return () => { /* no-op */ };
  if (typeof window === "undefined") return () => { /* no-op */ };
  if (!isMemoryApiAvailable()) return () => { /* no-op */ };

  let lastFiredAt = 0;
  let cancelled = false;
  const tick = () => {
    if (cancelled) return;
    const mem = readMemory();
    if (!mem) return;
    const ratio = mem.usedJSHeapSize / Math.max(1, mem.jsHeapSizeLimit);
    if (ratio >= threshold) {
      const now = Date.now();
      if (now - lastFiredAt >= COOLDOWN_MS) {
        lastFiredAt = now;
        try {
          listener();
        } catch (e) {
          console.warn("[mem-pressure] listener threw:", e);
        }
      }
    }
  };
  const handle = window.setInterval(tick, POLL_MS);
  return () => {
    cancelled = true;
    window.clearInterval(handle);
  };
}
