// IndexedDB cache for per-image annotation geometry. Caches the
// heavy fields (mask polygons, editedBoxes) keyed by
// (projectId, importId) so revisits and back/forward navigation
// paint from IDB instead of refetching tens of KB of JSON per
// image.
//
// Designed conservatively after the localStorage importTiles cache
// caused phantom-duplicate tiles when stale ids leaked into the
// gallery list. Two safeguards make this cache safer:
//   1. It NEVER caches the imports list itself, only per-id
//      geometry. /overview remains the source of truth for which
//      imports exist.
//   2. Every read is gated on `manifestUpdatedAt` — the BE updates
//      this on every write to the manifest, so a stale entry never
//      lands in state. We compare against the value the FE currently
//      knows (set from the latest /overview response) and treat a
//      mismatch as a cache miss.
//
// Flag: NEXT_PUBLIC_IDB_CACHE=1.

const DB_NAME = "pixelkit-v2";
const DB_VERSION = 1;
const STORE = "annotations";
const PROJECT_INDEX = "by_project";

export const IDB_CACHE_ENABLED =
  typeof process !== "undefined" &&
  process.env.NEXT_PUBLIC_IDB_CACHE === "1";

// Hard cap on cached rows. ~10 KB per row on average → ~50 MB at
// the cap. The browser's IDB quota is usually 100s of MB but other
// origin storage shares the budget, so being polite here.
const MAX_ROWS = 5000;

export type CachedAnnotation = {
  projectId: string;
  importId: string;
  detections: unknown[];
  editedBoxes: unknown[] | null;
  timings: unknown;
  manifestUpdatedAt: string | null;
  fetchedAt: number;
};

let _dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (typeof window === "undefined" || !("indexedDB" in window)) {
    return Promise.resolve(null);
  }
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise<IDBDatabase | null>((resolve) => {
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch (e) {
      console.warn("[idb] open threw:", e);
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        // Compound key [projectId, importId] keeps writes idempotent
        // per record without us having to hash a string key.
        const store = db.createObjectStore(STORE, {
          keyPath: ["projectId", "importId"],
        });
        // by_project index powers a per-project clear (used after
        // /v3/dedupe-imports apply or any other mass mutation).
        store.createIndex(PROJECT_INDEX, "projectId", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => {
      console.warn("[idb] open failed:", req.error);
      resolve(null);
    };
    req.onblocked = () => {
      console.warn("[idb] upgrade blocked by another tab");
      resolve(null);
    };
  });
  return _dbPromise;
}

function withStore<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => Promise<T> | T,
): Promise<T | null> {
  return openDb().then((db) => {
    if (!db) return null as T | null;
    return new Promise<T | null>((resolve) => {
      let tx: IDBTransaction;
      try {
        tx = db.transaction(STORE, mode);
      } catch (e) {
        console.warn("[idb] tx failed:", e);
        resolve(null);
        return;
      }
      const store = tx.objectStore(STORE);
      let result: T | null = null;
      Promise.resolve(fn(store)).then((r) => { result = r; }).catch((e) => {
        console.warn("[idb] op error:", e);
      });
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => {
        console.warn("[idb] tx error:", tx.error);
        resolve(null);
      };
      tx.onabort = () => {
        console.warn("[idb] tx aborted:", tx.error);
        resolve(null);
      };
    });
  });
}

// Returns the cached entry if it's still fresh (manifestUpdatedAt
// matches the caller's expected value). Returns null on miss or
// stale.
export async function getCachedAnnotation(
  projectId: string,
  importId: string,
  expectedManifestUpdatedAt: string | null,
): Promise<CachedAnnotation | null> {
  if (!IDB_CACHE_ENABLED) return null;
  return (await withStore("readonly", (store) =>
    new Promise<CachedAnnotation | null>((resolve) => {
      const req = store.get([projectId, importId]);
      req.onsuccess = () => {
        const row = req.result as CachedAnnotation | undefined;
        if (!row) {
          resolve(null);
          return;
        }
        if (row.manifestUpdatedAt !== expectedManifestUpdatedAt) {
          resolve(null);
          return;
        }
        resolve(row);
      };
      req.onerror = () => resolve(null);
    }),
  )) ?? null;
}

// Batched lookup. Returns a map keyed by importId; absent entries
// are misses (caller should fetch them over the network).
export async function getCachedAnnotationBatch(
  projectId: string,
  importIds: string[],
  expectedManifestUpdatedAt: string | null,
): Promise<Record<string, CachedAnnotation>> {
  if (!IDB_CACHE_ENABLED || importIds.length === 0) return {};
  return (await withStore("readonly", (store) =>
    new Promise<Record<string, CachedAnnotation>>((resolve) => {
      const out: Record<string, CachedAnnotation> = {};
      let remaining = importIds.length;
      if (remaining === 0) {
        resolve(out);
        return;
      }
      for (const id of importIds) {
        const req = store.get([projectId, id]);
        req.onsuccess = () => {
          const row = req.result as CachedAnnotation | undefined;
          if (
            row &&
            row.manifestUpdatedAt === expectedManifestUpdatedAt
          ) {
            out[id] = row;
          }
          remaining -= 1;
          if (remaining === 0) resolve(out);
        };
        req.onerror = () => {
          remaining -= 1;
          if (remaining === 0) resolve(out);
        };
      }
    }),
  )) ?? {};
}

export async function putCachedAnnotation(
  projectId: string,
  importId: string,
  row: {
    detections: unknown[];
    editedBoxes: unknown[] | null;
    timings: unknown;
  },
  manifestUpdatedAt: string | null,
): Promise<void> {
  if (!IDB_CACHE_ENABLED) return;
  await withStore("readwrite", (store) => {
    store.put({
      projectId,
      importId,
      detections: row.detections,
      editedBoxes: row.editedBoxes,
      timings: row.timings,
      manifestUpdatedAt,
      fetchedAt: Date.now(),
    } satisfies CachedAnnotation);
  });
}

export async function putCachedAnnotationBatch(
  projectId: string,
  rows: { importId: string; detections: unknown[]; editedBoxes: unknown[] | null; timings: unknown }[],
  manifestUpdatedAt: string | null,
): Promise<void> {
  if (!IDB_CACHE_ENABLED || rows.length === 0) return;
  await withStore("readwrite", (store) => {
    for (const r of rows) {
      store.put({
        projectId,
        importId: r.importId,
        detections: r.detections,
        editedBoxes: r.editedBoxes,
        timings: r.timings,
        manifestUpdatedAt,
        fetchedAt: Date.now(),
      } satisfies CachedAnnotation);
    }
  });
}

// Drops every cached annotation for one project. Called after the
// dedupe endpoint runs OR any mass-mutation flow.
export async function clearCachedAnnotationsForProject(
  projectId: string,
): Promise<void> {
  if (!IDB_CACHE_ENABLED) return;
  await withStore("readwrite", (store) => {
    return new Promise<void>((resolve) => {
      const idx = store.index(PROJECT_INDEX);
      const req = idx.openCursor(IDBKeyRange.only(projectId));
      req.onsuccess = () => {
        const cur = req.result;
        if (!cur) {
          resolve();
          return;
        }
        cur.delete();
        cur.continue();
      };
      req.onerror = () => resolve();
    });
  });
}

// LRU eviction. Scans the store, drops the oldest entries until we
// fit under MAX_ROWS. Cheap when the cache is already small, runs
// once on app load (called from a setup site below) rather than
// on every write.
let _evictionScheduled = false;
export function scheduleLruEviction(): void {
  if (!IDB_CACHE_ENABLED || _evictionScheduled) return;
  _evictionScheduled = true;
  // Run after the page settles. requestIdleCallback isn't on every
  // browser; setTimeout 4s is the polite fallback.
  const start = () => {
    void evictLruNow().catch(() => { /* swallow */ });
  };
  if (typeof window === "undefined") return;
  if ("requestIdleCallback" in window) {
    (window as unknown as { requestIdleCallback: (cb: () => void) => void })
      .requestIdleCallback(start);
  } else {
    setTimeout(start, 4000);
  }
}

async function evictLruNow(): Promise<void> {
  await withStore("readwrite", (store) =>
    new Promise<void>((resolve) => {
      const countReq = store.count();
      countReq.onsuccess = () => {
        const n = countReq.result;
        if (n <= MAX_ROWS) {
          resolve();
          return;
        }
        const toDrop = n - MAX_ROWS;
        // Scan all rows in insertion order (== key order on the
        // compound [projectId, importId]). That's not strict LRU
        // by access time, but close enough — drop the first `toDrop`
        // rows we see whose fetchedAt is the oldest.
        const fetchedTimes: { key: IDBValidKey; t: number }[] = [];
        const cursorReq = store.openCursor();
        cursorReq.onsuccess = () => {
          const cur = cursorReq.result;
          if (!cur) {
            // Sort by oldest, drop the first toDrop.
            fetchedTimes.sort((a, b) => a.t - b.t);
            const drop = fetchedTimes.slice(0, toDrop);
            for (const d of drop) store.delete(d.key);
            resolve();
            return;
          }
          const v = cur.value as CachedAnnotation;
          fetchedTimes.push({ key: cur.primaryKey, t: v.fetchedAt ?? 0 });
          cur.continue();
        };
        cursorReq.onerror = () => resolve();
      };
      countReq.onerror = () => resolve();
    }),
  );
}
