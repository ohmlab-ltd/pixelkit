// Singleton wrapper around annotations.worker.ts. One persistent
// worker for the whole tab - workers are expensive to spawn
// (~5–10 ms) and pin a few MB of RAM, so a single long-lived
// instance handling all annotations / labelStats traffic is the
// right shape. Re-creating per call would lose more than it saves.
//
// Three public methods. Each:
//   - generates a message id,
//   - posts the request (with ArrayBuffer as transferable when
//     applicable so the main thread doesn't retain the bytes),
//   - returns a Promise that resolves when the worker echoes the
//     same id back.
//
// Behind NEXT_PUBLIC_ANNOT_WORKER - when off, ANNOT_WORKER_ENABLED
// flips false and the call sites stay on their main-thread paths.
//
// SSR-safe: getWorker() bails on `typeof window === "undefined"`.
// Importers must check ANNOT_WORKER_ENABLED before calling so they
// keep an inline fallback for the flag-off path.

type ParsedDetection = {
  box: [number, number, number, number];
  mask: unknown;
  embedding: number[] | undefined;
  gdLabel: string | null;
  gdVariant: string | null;
  gdScore: number | null;
  vlmLabel: string | null;
  vlmScore: number | null;
  vlmMs: number | null;
  embedNearestLabel: string | null;
  embedSimilarity: number | null;
  embedSimilarityForLabel: number | null;
  embedMargin: number | null;
  predLabel: string | null;
  predSource: string | null;
  rejected: boolean;
  rejectReason: string | null;
  ambiguous: boolean;
  vlmAction: string | null;
  embedSims: unknown;
  embedSimsDino: unknown;
  embedSimsSiglip: unknown;
  siglipWeight: number | null;
  cropDataUrl: string;
};

export type ParsedEditedBox = {
  id?: string;
  label?: string;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  score?: number;
  mask?: unknown;
  validation?: unknown;
};

export type ParsedRow = {
  detections: ParsedDetection[];
  editedBoxes: ParsedEditedBox[] | null;
  timings: unknown;
};

export const ANNOT_WORKER_ENABLED =
  process.env.NEXT_PUBLIC_ANNOT_WORKER === "1";

// Skip the worker hop for tiny payloads - the postMessage round-
// trip costs more than the inline parse on small JSON.
export const ANNOT_WORKER_BUFFER_THRESHOLD = 50 * 1024; // 50 KB

let _worker: Worker | null = null;
let _nextMsgId = 1;
const _pending = new Map<number, {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
}>();

function getWorker(): Worker | null {
  if (typeof window === "undefined") return null;
  if (_worker) return _worker;
  try {
    // Next.js 13+ supports the standard URL+import.meta.url worker
    // construction - Webpack/Turbopack rewrite the relative URL into
    // a bundled chunk URL at build time.
    _worker = new Worker(
      new URL("./annotations.worker.ts", import.meta.url),
      { type: "module" },
    );
    _worker.addEventListener("message", (e: MessageEvent<{
      id: number;
      ok: boolean;
      result?: unknown;
      error?: string;
    }>) => {
      const { id, ok, result, error } = e.data;
      const entry = _pending.get(id);
      if (!entry) return;
      _pending.delete(id);
      if (ok) entry.resolve(result);
      else entry.reject(new Error(error || "worker error"));
    });
    _worker.addEventListener("error", (e) => {
      // Failure-mode: reject all in-flight requests so callers fall
      // back to the inline path on the next call.
      const err = new Error(`annotations worker error: ${e.message || "unknown"}`);
      for (const [id, entry] of _pending) {
        entry.reject(err);
        _pending.delete(id);
      }
      _worker = null;
    });
    return _worker;
  } catch (err) {
    console.warn("[annotations-worker] failed to spawn:", err);
    return null;
  }
}

function postWithId<T>(
  op: string,
  payload: Record<string, unknown>,
  transfer: Transferable[] = [],
): Promise<T> {
  const w = getWorker();
  if (!w) return Promise.reject(new Error("worker unavailable"));
  const id = _nextMsgId++;
  return new Promise<T>((resolve, reject) => {
    _pending.set(id, {
      resolve: (v) => resolve(v as T),
      reject,
    });
    w.postMessage({ op, id, ...payload }, transfer);
  });
}

export function parseViewportBatchInWorker(
  buffer: ArrayBuffer,
): Promise<{ imports: Record<string, ParsedRow> }> {
  return postWithId("parseViewport", { buffer }, [buffer]);
}

// P6: msgpack variant. Same return shape; the worker decodes the
// binary payload + unwraps wire detections in one pass off-thread.
export function parseViewportBatchMsgpackInWorker(
  buffer: ArrayBuffer,
): Promise<{ imports: Record<string, ParsedRow> }> {
  return postWithId("parseViewportMsgpack", { buffer }, [buffer]);
}

export const BINARY_WIRE_ENABLED =
  process.env.NEXT_PUBLIC_BINARY_WIRE === "1";

export function parseSingleAnnotationInWorker(
  buffer: ArrayBuffer,
): Promise<ParsedRow> {
  return postWithId("parseSingle", { buffer }, [buffer]);
}

export function aggregateLabelStatsInWorker(
  slim: { id: string; labelStats?: Record<string, number> }[],
): Promise<Record<string, number>> {
  return postWithId("labelStats", { slim });
}
