// Normalised per-project store. Replaces the flat ImportedMedia[]
// arrays sitting in ProjectViewV2Stub's useState with an id-keyed
// Map plus a small set of viewport-scoped projections.
//
// Why this exists:
//   - Today, ProjectViewV2Stub holds `imports: ImportedMedia[]` in
//     useState. Every setImports(...) regenerates the array reference,
//     so the whole DatasetGallery re-renders even when one tile
//     changed. On a 9k-image project that's ~9k <DatasetThumb> commits
//     for a single PUT response.
//   - Zustand gives us O(1) per-id reads with stable references plus
//     fine-grained selector subscriptions, so a tile only re-renders
//     when its OWN id's record changes.
//
// Migration is gated on NEXT_PUBLIC_STORE_V2 - when the flag is off,
// nothing is wired into this store and the existing useState path
// drives everything. When the flag is on, ProjectViewV2Stub mirror-
// writes every setImports into bulkUpsert here, and DatasetThumb
// reads from useImport(id). Migration is incremental: any reader
// that hasn't been ported yet keeps reading from the prop.

import { useMemo } from "react";
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

// ─── Public shape ────────────────────────────────────────────────────

// Subset of ImportedMedia carried in the store. Stays NARROW on
// purpose - keeping every transient flag here would defeat the
// re-render-isolation win. Anything chunky (full polygon arrays,
// embedding sims) lives off the side in geometryCache.
export type StoreImport = {
  id: string;
  backendId?: string;
  filename?: string;
  preview: string;
  blurhash?: string | null;
  status: "processing" | "ready" | "failed";
  width?: number;
  height?: number;
  createdAt?: number | string | null;
  sourceUrl?: string | null;
  nAugmentations?: number;
  labelStats?: Record<string, number>;
  detectionCount?: number;
  // For a derived ("new labels") crop: the parent's original label, shown as a
  // reference while the user assigns their own fresh label. Null on normal imports.
  derivedLabel?: string | null;
  // hasGeometry tracks "has the per-image /annotations fetch
  // resolved for this id". Drives the viewer's loading spinner +
  // skips redundant fetches.
  hasGeometry?: boolean;
  // Verdict (good/bad/unsure) and edit dirty flags surface here so
  // selectors that ONLY care about the verdict (filter pills,
  // review header) don't re-render on unrelated edits.
  verdict?: "good" | "bad" | "unsure" | null;
  hasEdits?: boolean;
  labelledAt?: number;
};

// Heavy per-id geometry. Lives separately so a tile in viewport but
// without geometry doesn't carry mask polygons around.
export type StoreGeometry = {
  // 0 = none, 1 = bboxes only, 2 = simplified polygons, 3 = full.
  level: number;
  // Opaque blob; deserialised by the consumer (BoxEditor).
  detections: unknown[];
  editedBoxes?: unknown[];
};

type ProjectStoreState = {
  projectId: string | null;
  importsById: Map<string, StoreImport>;
  // ID list in display order (createdAt desc). Single source of
  // truth for DatasetGallery iteration. Stable reference: only
  // replaced when the actual order changes.
  orderedIds: string[];
  visibleIds: string[];
  geometryCache: Map<string, StoreGeometry>;
  // Project-wide label-count aggregate, summed from per-import
  // labelStats. Drives the dataset-stats card's label distribution
  // panel without iterating every import on every render.
  labelStats: Record<string, number>;

  // ─── Mutations ─────────────────────────────────────────────────
  reset: (projectId: string | null) => void;
  upsertImport: (row: StoreImport) => void;
  bulkUpsert: (rows: StoreImport[]) => void;
  removeImport: (id: string) => void;
  setVisibleIds: (ids: string[]) => void;
  setGeometry: (id: string, geom: StoreGeometry) => void;
  clearGeometry: (id: string) => void;
  // Authoritative project-wide labelStats aggregate, computed
  // off-thread by the annotations worker (P4). bulkUpsert maintains
  // the delta-updated value inline for optimistic UI; this setter
  // lands when the worker finishes a from-scratch recompute and is
  // a no-op when the worker's result already matches.
  setLabelStatsAggregate: (stats: Record<string, number>) => void;
};

// subscribeWithSelector lets external (non-React) consumers like the
// perf logger watch a specific slice without subscribing to the
// whole store. Cheap; no Provider tree.
export const useProjectStore = create<ProjectStoreState>()(
  subscribeWithSelector((set, get) => ({
    projectId: null,
    importsById: new Map(),
    orderedIds: [],
    visibleIds: [],
    geometryCache: new Map(),
    labelStats: {},

    reset: (projectId) => {
      // Don't bail on identical projectId - callers might call
      // reset to clear state mid-session (e.g. error recovery).
      set({
        projectId,
        importsById: new Map(),
        orderedIds: [],
        visibleIds: [],
        geometryCache: new Map(),
        labelStats: {},
      });
    },

    upsertImport: (row) => {
      const cur = get();
      // Merge into the existing record; the incoming row may be
      // partial (e.g. a /annotations fetch result patching just
      // hasGeometry + detectionCount).
      const existing = cur.importsById.get(row.id);
      const next: StoreImport = existing ? { ...existing, ...row } : row;
      const nextMap = new Map(cur.importsById);
      nextMap.set(row.id, next);
      // orderedIds only changes when this is a brand-new id OR
      // createdAt changed enough to shuffle order. Recompute lazily
      // by deferring until bulkUpsert finishes, callers should
      // prefer bulkUpsert for hot paths.
      const wasNew = !existing;
      const orderedIds = wasNew
        ? [...cur.orderedIds, row.id]
        : cur.orderedIds;
      // Label-stats aggregate: subtract old, add new.
      const labelStats = recomputeLabelStatsDelta(
        cur.labelStats, existing?.labelStats, next.labelStats,
      );
      set({ importsById: nextMap, orderedIds, labelStats });
    },

    bulkUpsert: (rows) => {
      if (rows.length === 0) return;
      const cur = get();
      // Mutate a fresh Map so subscribers see one update, not N.
      const nextMap = new Map(cur.importsById);
      let aggDelta: Record<string, number> = {};
      const newIds: string[] = [];
      for (const row of rows) {
        const existing = nextMap.get(row.id);
        const next: StoreImport = existing ? { ...existing, ...row } : row;
        nextMap.set(row.id, next);
        if (!existing) newIds.push(row.id);
        // Roll up labelStats delta in one pass.
        aggDelta = applyLabelStatsDelta(
          aggDelta, existing?.labelStats, next.labelStats,
        );
      }
      const orderedIds = newIds.length === 0
        ? cur.orderedIds
        : sortIdsByCreatedAt(
            [...cur.orderedIds, ...newIds], nextMap,
          );
      const labelStats = mergeAggregate(cur.labelStats, aggDelta);
      set({ importsById: nextMap, orderedIds, labelStats });
    },

    removeImport: (id) => {
      const cur = get();
      if (!cur.importsById.has(id)) return;
      const removed = cur.importsById.get(id);
      const nextMap = new Map(cur.importsById);
      nextMap.delete(id);
      const orderedIds = cur.orderedIds.filter((x) => x !== id);
      const labelStats = recomputeLabelStatsDelta(
        cur.labelStats, removed?.labelStats, undefined,
      );
      const geometryCache = cur.geometryCache.has(id)
        ? (() => {
            const c = new Map(cur.geometryCache);
            c.delete(id);
            return c;
          })()
        : cur.geometryCache;
      set({ importsById: nextMap, orderedIds, labelStats, geometryCache });
    },

    setVisibleIds: (ids) => {
      // Cheap reference equality check - gallery scroll fires this
      // every frame, no point invalidating subscribers when nothing
      // actually changed.
      const cur = get();
      if (
        cur.visibleIds.length === ids.length &&
        cur.visibleIds.every((id, i) => id === ids[i])
      ) return;
      set({ visibleIds: ids });
    },

    setGeometry: (id, geom) => {
      const cur = get();
      const nextMap = new Map(cur.geometryCache);
      nextMap.set(id, geom);
      // Also flip hasGeometry on the import record so selectors
      // that gate the viewer's spinner can short-circuit on it.
      const imp = cur.importsById.get(id);
      const importsById = imp && !imp.hasGeometry
        ? (() => {
            const m = new Map(cur.importsById);
            m.set(id, { ...imp, hasGeometry: true });
            return m;
          })()
        : cur.importsById;
      set({ geometryCache: nextMap, importsById });
    },

    clearGeometry: (id) => {
      const cur = get();
      if (!cur.geometryCache.has(id)) return;
      const nextMap = new Map(cur.geometryCache);
      nextMap.delete(id);
      // Reset hasGeometry on the import so the next view re-fetches.
      const imp = cur.importsById.get(id);
      const importsById = imp && imp.hasGeometry
        ? (() => {
            const m = new Map(cur.importsById);
            m.set(id, { ...imp, hasGeometry: false });
            return m;
          })()
        : cur.importsById;
      set({ geometryCache: nextMap, importsById });
    },

    setLabelStatsAggregate: (stats) => {
      const cur = get();
      // Reference-equal when keys + values match - skip the set so
      // subscribers don't churn. Cheap structural compare given
      // labels are usually <20.
      const a = cur.labelStats;
      const aKeys = Object.keys(a);
      const bKeys = Object.keys(stats);
      if (aKeys.length === bKeys.length) {
        let same = true;
        for (const k of aKeys) {
          if (a[k] !== stats[k]) { same = false; break; }
        }
        if (same) return;
      }
      set({ labelStats: stats });
    },
  })),
);

// ─── Selector helpers ────────────────────────────────────────────────

// Drop-in for `imports.find(m => m.id === id)`. Re-renders only
// when this id's record changes.
export function useImport(id: string | null | undefined): StoreImport | undefined {
  return useProjectStore((s) =>
    id ? s.importsById.get(id) : undefined,
  );
}

// Slim selector for the chip rail. Avoids re-rendering on
// labelledAt / verdict changes that don't affect chip count.
export function useImportLabelStats(
  id: string | null | undefined,
): Record<string, number> | undefined {
  return useProjectStore((s) =>
    id ? s.importsById.get(id)?.labelStats : undefined,
  );
}

// Used by DatasetGallery to drive its tile-iteration loop. Stable
// reference unless ids actually change.
export function useOrderedIds(): string[] {
  return useProjectStore((s) => s.orderedIds);
}

export function useVisibleIds(): string[] {
  return useProjectStore((s) => s.visibleIds);
}

// Total label-count aggregate across the whole dataset. Backs the
// stats card's label distribution panel.
export function useDatasetLabelStats(): Record<string, number> {
  return useProjectStore((s) => s.labelStats);
}

// Per-id geometry. Returns undefined until /annotations resolves
// for this import.
export function useImportGeometry(id: string | null | undefined): StoreGeometry | undefined {
  return useProjectStore((s) =>
    id ? s.geometryCache.get(id) : undefined,
  );
}

// Memoised mutator handles. Components that only mutate (not read)
// should pull these via this hook to avoid re-rendering on store
// changes.
export function useProjectStoreActions() {
  return useMemo(() => {
    const s = useProjectStore.getState();
    return {
      reset: s.reset,
      upsertImport: s.upsertImport,
      bulkUpsert: s.bulkUpsert,
      removeImport: s.removeImport,
      setVisibleIds: s.setVisibleIds,
      setGeometry: s.setGeometry,
      clearGeometry: s.clearGeometry,
      setLabelStatsAggregate: s.setLabelStatsAggregate,
    };
  }, []);
}

// Bare object for use from non-React code (apiFetch interceptors,
// background prefetchers). All methods are stable function refs.
export const ProjectStore = {
  get state() {
    return useProjectStore.getState();
  },
  reset: (...args: Parameters<ProjectStoreState["reset"]>) =>
    useProjectStore.getState().reset(...args),
  upsertImport: (...args: Parameters<ProjectStoreState["upsertImport"]>) =>
    useProjectStore.getState().upsertImport(...args),
  bulkUpsert: (...args: Parameters<ProjectStoreState["bulkUpsert"]>) =>
    useProjectStore.getState().bulkUpsert(...args),
  removeImport: (...args: Parameters<ProjectStoreState["removeImport"]>) =>
    useProjectStore.getState().removeImport(...args),
  setVisibleIds: (...args: Parameters<ProjectStoreState["setVisibleIds"]>) =>
    useProjectStore.getState().setVisibleIds(...args),
  setGeometry: (...args: Parameters<ProjectStoreState["setGeometry"]>) =>
    useProjectStore.getState().setGeometry(...args),
  clearGeometry: (...args: Parameters<ProjectStoreState["clearGeometry"]>) =>
    useProjectStore.getState().clearGeometry(...args),
  setLabelStatsAggregate: (...args: Parameters<ProjectStoreState["setLabelStatsAggregate"]>) =>
    useProjectStore.getState().setLabelStatsAggregate(...args),
};

// ─── Feature flag ────────────────────────────────────────────────────

export const STORE_V2_ENABLED =
  process.env.NEXT_PUBLIC_STORE_V2 === "1";

// ─── Internal helpers ────────────────────────────────────────────────

// Diff the label-count delta for one record's labelStats change.
function recomputeLabelStatsDelta(
  agg: Record<string, number>,
  before: Record<string, number> | undefined,
  after: Record<string, number> | undefined,
): Record<string, number> {
  if (!before && !after) return agg;
  const next = { ...agg };
  if (before) {
    for (const [lab, n] of Object.entries(before)) {
      next[lab] = Math.max(0, (next[lab] ?? 0) - n);
      if (next[lab] === 0) delete next[lab];
    }
  }
  if (after) {
    for (const [lab, n] of Object.entries(after)) {
      next[lab] = (next[lab] ?? 0) + n;
    }
  }
  return next;
}

function applyLabelStatsDelta(
  delta: Record<string, number>,
  before: Record<string, number> | undefined,
  after: Record<string, number> | undefined,
): Record<string, number> {
  if (!before && !after) return delta;
  const next = { ...delta };
  if (before) {
    for (const [lab, n] of Object.entries(before)) {
      next[lab] = (next[lab] ?? 0) - n;
    }
  }
  if (after) {
    for (const [lab, n] of Object.entries(after)) {
      next[lab] = (next[lab] ?? 0) + n;
    }
  }
  return next;
}

function mergeAggregate(
  base: Record<string, number>,
  delta: Record<string, number>,
): Record<string, number> {
  const next = { ...base };
  let mutated = false;
  for (const [lab, n] of Object.entries(delta)) {
    const v = (next[lab] ?? 0) + n;
    if (v <= 0) {
      if (lab in next) {
        delete next[lab];
        mutated = true;
      }
    } else if (next[lab] !== v) {
      next[lab] = v;
      mutated = true;
    }
  }
  return mutated ? next : base;
}

function sortIdsByCreatedAt(
  ids: string[],
  byId: Map<string, StoreImport>,
): string[] {
  // De-dupe, then sort by createdAt desc (newest first), matching
  // the existing gallery order. Falls back to id-lex for missing
  // createdAt so the sort is deterministic across reloads.
  const unique = Array.from(new Set(ids));
  unique.sort((a, b) => {
    const ra = byId.get(a)?.createdAt;
    const rb = byId.get(b)?.createdAt;
    const ta = toMs(ra);
    const tb = toMs(rb);
    if (ta !== tb) return tb - ta;
    return a.localeCompare(b);
  });
  return unique;
}

function toMs(v: number | string | null | undefined): number {
  if (typeof v === "number") return v;
  if (typeof v === "string" && v) {
    const t = Date.parse(v);
    if (Number.isFinite(t)) return t;
  }
  return 0;
}
