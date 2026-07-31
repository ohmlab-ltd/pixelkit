"use client";

// Dataset stats card, sits at the top of the project page's Dataset
// tab. Pulls /api/v2/projects/{id}/dataset-stats which folds together:
//   • counts        (imports / detections / augmentations / unsure)
//   • label dist    (one bar per label, sorted desc)
//   • health score  (0-100, blended from multiple quality signals)
//   • variation     (2D projection of whole-image features
//                    + flagged near-duplicate ids)
//
// Default state is COLLAPSED, only a compact summary row is visible
// (health badge + a few counters). Expanding the card opens the full
// three-column layout (counters / label distribution / variation plot).
// This keeps the page short while still surfacing the headline number
// at a glance.

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "../../lib/apiFetch";
import { buildProjectLabelColourMap, colourForLabelStable, readableTextForBg } from "./OnboardLabelsV2";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

export type StatsPoint = {
  id: string;
  filename: string | null;
  x: number;
  y: number;
  label: string | null;
  n_detections: number;
};

type DatasetStats = {
  counts: {
    imports: number;
    with_detections: number;
    detections: number;
    unsure_detections: number;
    augmentations: number;
    near_duplicates: number;
    embeddings_ready: number;
  };
  labels: { label: string; count: number }[];
  health: {
    score: number;
    factors: {
      balance: number;
      coverage: number;
      confidence: number;
      uniqueness: number;
    };
  };
  variation: {
    points: StatsPoint[];
    augmentations?: AugmentPoint[];
    near_duplicate_ids: string[];
  };
};

export type AugmentPoint = {
  id: string;
  source_id: string;
  filename: string | null;
  x: number;
  y: number;
  label: string | null;
};

export function DatasetStatsCard({
  projectId,
  labels,
  labelColours,
  refreshSignal = 0,
  onJumpToImport,
  seedStats,
  readOnly = false,
}: {
  projectId: string | null;
  labels: string[];
  labelColours?: Record<string, string> | null;
  refreshSignal?: number;
  /** Called when a dot on the variation plot is clicked. `kind` tells
      the parent whether the click was an original image or one of its
      augmentations, the parent uses that to either just highlight
      the thumbnail or also flash an "augmentation" badge over it. */
  onJumpToImport?: (importId: string, kind: "image" | "augmentation") => void;
  /** Lite stats handed down from the parent's /initial fetch. When
      present, the card paints the summary row + label distribution
      immediately on mount instead of waiting on its own
      /dataset-stats?lite=true round-trip. Treated as a seed: the
      full-payload fetch still runs afterwards to add the variation
      plot + near-duplicates + 4-factor health score. */
  seedStats?: DatasetStats | null;
  /** True when the card is rendered in the public read-only view ,
      hides destructive actions like "Review duplicates". */
  readOnly?: boolean;
}) {
  // Internal store for stats payloads fetched by this component
  // (lite + full). `stats` below is the value used by every render
  // expression in the card, it falls back to `seedStats` when the
  // internal store is empty, so a seed arriving as a fresh prop is
  // visible to the render in the SAME tick (no one-render gap
  // waiting for a sync useEffect to copy seedStats into _stats).
  // The previous "sync into internal state" useEffect was what made
  // the stats card appear empty for ~1 frame after the project
  // loader faded on a hard refresh, on a large project that one
  // frame coincided with the gallery's heavy commit pass and was
  // perceptible to the user.
  const [_stats, setStats] = useState<DatasetStats | null>(seedStats ?? null);
  const stats = _stats ?? (seedStats as DatasetStats | null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const colourFor = useMemo(() => {
    const map = buildProjectLabelColourMap(labels, labelColours ?? null);
    return (lab: string | null): string => {
      if (!lab) return "var(--muted)";
      const key = lab.trim().toLowerCase();
      return map.get(key) ?? colourForLabelStable(lab);
    };
  }, [labels, labelColours]);

  // The /initial seed is only fresh until the first refreshSignal bump (a
  // mutation: augment / import / label edit). Track the mount value so the
  // lite effect stops short-circuiting on the stale seed afterwards and
  // refetches - otherwise the always-visible summary row froze at the
  // pre-mutation snapshot until the card was re-expanded or reloaded.
  const mountSignalRef = useRef(refreshSignal);

  useEffect(() => {
    if (!projectId) return;
    let alive = true;
    setLoading(true);
    setError(null);
    (async () => {
      // Lite-only on mount. The full payload (PCA + near-duplicates +
      // 4-factor health) is multi-second on big projects and only
      // matters for the expanded variation plot, see the open-gated
      // useEffect below. The summary row + label distribution paint
      // from the lite snapshot alone.
      //
      // When the parent has already handed us a fresh seed (the
      // /initial payload's `stats` field, served from the same disk
      // sidecar that backs /dataset-stats?lite=true), skip the lite
      // round-trip entirely, same bytes, one fewer fetch.
      if (seedStats && refreshSignal === mountSignalRef.current) {
        setLoading(false);
        return;
      }
      try {
        const liteUrl = `/api/v2/projects/${projectId}/dataset-stats?lite=true&v=${refreshSignal}`;
        const lr = await apiFetch(liteUrl, { cache: "no-store" });
        if (!lr.ok) throw new Error(`http ${lr.status}`);
        const liteData = (await lr.json()) as DatasetStats;
        if (!alive) return;
        setStats(liteData);
        setLoading(false);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [projectId, refreshSignal, seedStats]);

  // Re-fetch the lite stats every 2.5 s while embeddings are still
  // backfilling so the variation-plot spinner shows live progress
  // instead of being stuck at the seed value. Stops as soon as ready
  // catches up with imports OR points land (full payload arrived).
  const isComputing = !!stats
    && stats.counts.imports > 0
    && stats.counts.embeddings_ready < stats.counts.imports
    && stats.variation.points.length === 0;
  useEffect(() => {
    if (!isComputing || !projectId) return;
    let alive = true;
    let timer: number | null = null;
    const tick = async () => {
      try {
        const liteUrl = `/api/v2/projects/${projectId}/dataset-stats?lite=true&v=${refreshSignal}&_=${Date.now()}`;
        const r = await apiFetch(liteUrl, { cache: "no-store" });
        if (!alive || !r.ok) return;
        const fresh = (await r.json()) as DatasetStats;
        if (!alive) return;
        setStats(fresh);
      } catch { /* keep last good snapshot */ }
      if (!alive) return;
      timer = window.setTimeout(tick, 2500);
    };
    timer = window.setTimeout(tick, 2500);
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [isComputing, projectId, refreshSignal]);

  // Full payload, variation plot, near-dup flags, 4-factor health.
  // Deferred until the user actually opens the card. On a 964-image
  // project the full compute is multi-second (load 964 embeddings →
  // PCA → pairwise cosine for near-dup detection), so firing it on
  // mount taxed the request thread + browser even when the user
  // never expanded the section. Ref-guarded so multiple opens within
  // the same refreshSignal share one network call.
  const fullFetchedKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!open || !projectId) return;
    const key = `${projectId}::${refreshSignal}`;
    if (fullFetchedKeyRef.current === key) return;
    fullFetchedKeyRef.current = key;
    let alive = true;
    (async () => {
      try {
        const url = `/api/v2/projects/${projectId}/dataset-stats?v=${refreshSignal}`;
        const r = await apiFetch(url, { cache: "no-store" });
        if (!r.ok) return;
        const data = (await r.json()) as DatasetStats;
        if (!alive) return;
        setStats(data);
      } catch {
        /* keep the lite snapshot on full-fetch failure */
      }
    })();
    return () => { alive = false; };
  }, [open, projectId, refreshSignal]);

  if (!projectId) return null;

  const totalLabelCount = stats?.labels.reduce((s, l) => s + l.count, 0) ?? 0;
  const maxLabelCount = stats?.labels[0]?.count ?? 0;
  const hasImports = (stats?.counts.imports ?? 0) > 0;

  return (
    <section className="mx-auto max-w-6xl px-6 pt-4 pb-1">
      {/* Ambient: no border, no background tint, no shadow. The
          card reads as quiet metadata above the page; the only
          visual chrome is the chevron + health badge in the
          summary row. */}
      <div className="rounded-2xl overflow-hidden">
        {/* Header / summary row, always visible. Click toggles the
            full breakdown below. The headline metric (health score)
            and the top-three counters live in this row so the user
            gets the at-a-glance read without expanding. */}
        {/* Header row. min-h reserves the full height of the
            (taller) health-badge state so the row doesn't grow when
            stats finally land. Sub-info + badge share a single
            `data-ready` gate, fade in together once the response
            resolves, and live in inline-fixed-height containers
            so the layout is pixel-stable from first paint. */}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="w-full px-5 min-h-[3.5rem] flex items-center justify-between gap-4 text-left hover:bg-foreground/[0.02] transition-colors"
        >
          <div className="flex items-center gap-3 min-w-0">
            <svg
              viewBox="0 0 24 24"
              className="h-4 w-4 text-foreground/55 shrink-0"
              fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform 220ms cubic-bezier(0.2,0.7,0.2,1)" }}
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
            <h2 className="text-base font-medium tracking-tight text-[var(--foreground)] truncate">
              Dataset stats
            </h2>
            {/* Sub-info wrapper renders unconditionally so it doesn't
                pop into existence after the title, it just fades its
                contents in once data lands. Coordinated with the
                health badge by a shared 220ms ease and a single
                `data-ready` gate keyed off `stats && hasImports`. */}
            <span
              className="hidden sm:flex items-center gap-3 ml-3 text-xs text-foreground/65 flex-wrap"
              style={{
                opacity: stats && hasImports ? 1 : 0,
                transition: "opacity 240ms ease 60ms",
              }}
              aria-hidden={!stats}
            >
              {stats && hasImports && (
                <>
                  <span className="font-mono tabular-nums">
                    <span className="text-foreground/45">{stats.counts.imports} </span>
                    image{stats.counts.imports === 1 ? "" : "s"}
                  </span>
                  <span className="font-mono tabular-nums">
                    <span className="text-foreground/45">{stats.counts.detections} </span>
                    detection{stats.counts.detections === 1 ? "" : "s"}
                  </span>
                  {stats.labels.length > 0 && (
                    <span className="font-mono tabular-nums">
                      <span className="text-foreground/45">{stats.labels.length} </span>
                      label{stats.labels.length === 1 ? "" : "s"}
                    </span>
                  )}
                  {stats.counts.augmentations > 0 && (
                    <span className="font-mono tabular-nums">
                      <span className="text-foreground/45">{stats.counts.augmentations} </span>
                      augmentation{stats.counts.augmentations === 1 ? "" : "s"}
                    </span>
                  )}
                  {stats.counts.near_duplicates > 0 && (
                    <span className="font-mono tabular-nums text-amber-700 dark:text-amber-300">
                      {stats.counts.near_duplicates} near-dup
                    </span>
                  )}
                </>
              )}
            </span>
          </div>
          {/* Health badge wrapper has a fixed min-height matching the
              rendered badge, so the header row pre-allocates the
              space and doesn't jump when stats arrive. Same fade-in
              timing as the sub-info → both lights up together. */}
          <span
            className="inline-flex items-center justify-end min-h-[2rem]"
            style={{
              opacity: stats && hasImports ? 1 : 0,
              transition: "opacity 240ms ease 60ms",
            }}
            aria-hidden={!stats}
          >
            {stats && hasImports && (
              <HealthBadge
                score={stats.health.score}
                factors={stats.health.factors}
                counts={stats.counts}
              />
            )}
          </span>
        </button>

        {/* Collapsible body. Animated via the same grid-rows trick
            used elsewhere on the page so the transition is smooth
            with no fixed-height fudge. */}
        <div
          className="grid transition-[grid-template-rows] duration-300 ease-out"
          style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
        >
          <div className="min-h-0 overflow-hidden">
            <div className="px-5 pb-5 pt-2 border-t border-foreground/[0.06]">
              {loading && !stats && (
                <div className="text-sm text-[var(--muted)] py-4">Loading stats…</div>
              )}
              {error && !stats && (
                <div className="text-sm text-red-700 dark:text-red-300 py-4">
                  Couldn&rsquo;t load stats, {error}
                </div>
              )}

              {stats && (
                <div className="grid gap-6 lg:grid-cols-3 pt-3">
                  {/* Counters column */}
                  <div className="space-y-2 lg:col-span-1">
                    <CounterRow label="Images" value={stats.counts.imports} />
                    <CounterRow label="With detections" value={stats.counts.with_detections} />
                    <CounterRow label="Total detections" value={stats.counts.detections} />
                    <CounterRow label="Unsure detections" value={stats.counts.unsure_detections} tone="warn" />
                    <CounterRow label="Augmentations" value={stats.counts.augmentations} />
                    <CounterRow
                      label="Near-duplicates"
                      value={stats.counts.near_duplicates}
                      tone={stats.counts.near_duplicates > 0 ? "warn" : undefined}
                    />
                    {stats.counts.near_duplicates > 0 && projectId && !readOnly && (
                      <RemoveDuplicatesAction projectId={projectId} />
                    )}
                  </div>

                  {/* Label distribution column.
                      Capped at ~14 rem then scrolled, projects with
                      dozens of labels used to push the variation
                      plot way down the page. The mask-image fade is
                      only applied when the content actually overflows
                      the cap (LabelsList measures itself + toggles
                      the mask), so short label sets stay flat-edged. */}
                  <div className="lg:col-span-1">
                    <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-mono mb-3">
                      Label distribution
                    </div>
                    {stats.labels.length === 0 ? (
                      <p className="text-sm text-[var(--muted)]">No annotations yet.</p>
                    ) : (
                      <LabelsList maxHeight="14rem">
                        <ul className="space-y-2 py-1">
                          {stats.labels.map((row) => {
                            const pct = totalLabelCount > 0 ? (row.count / totalLabelCount) * 100 : 0;
                            const fill = maxLabelCount > 0 ? (row.count / maxLabelCount) * 100 : 0;
                            const colour = colourFor(row.label);
                            return (
                              <li key={row.label} className="grid gap-1">
                                <div className="flex items-baseline justify-between gap-2 text-xs">
                                  <span className="inline-flex items-center gap-1.5 text-[var(--foreground)]">
                                    <span
                                      aria-hidden
                                      className="inline-block h-2 w-2 rounded-full shrink-0"
                                      style={{ backgroundColor: colour }}
                                    />
                                    <span className="truncate">{row.label}</span>
                                  </span>
                                  <span className="font-mono tabular-nums text-foreground/65 shrink-0">
                                    {row.count}
                                    <span className="text-foreground/30 ml-1">({pct.toFixed(0)}%)</span>
                                  </span>
                                </div>
                                <div className="h-1.5 rounded-full bg-foreground/[0.08] overflow-hidden">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${fill}%`,
                                      backgroundColor: colour,
                                      transition: "width 240ms ease-out",
                                    }}
                                  />
                                </div>
                              </li>
                            );
                          })}
                        </ul>
                      </LabelsList>
                    )}
                  </div>

                  {/* Variation plot column */}
                  <div className="lg:col-span-1">
                    <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-mono mb-3 flex items-center justify-between gap-2">
                      <span>Image variation</span>
                      {stats.counts.embeddings_ready > 0 && (
                        <span className="text-[10px] font-normal text-foreground/40 normal-case tracking-normal">
                          {stats.counts.embeddings_ready} of {stats.counts.imports} embedded
                        </span>
                      )}
                    </div>
                    <VariationPlot
                      points={stats.variation.points}
                      augmentations={stats.variation.augmentations ?? []}
                      colourFor={colourFor}
                      nearDuplicateIds={new Set(stats.variation.near_duplicate_ids)}
                      onJumpToImport={onJumpToImport}
                      // Surface "embeddings in progress" with a spinner
                      // inside the plot rather than a quiet line below.
                      // Computing = there are images but not every one
                      // is embedded yet.
                      computing={
                        stats.counts.imports > 0
                        && stats.counts.embeddings_ready < stats.counts.imports
                      }
                      totalImports={stats.counts.imports}
                      embeddingsReady={stats.counts.embeddings_ready}
                    />
                    {stats.variation.points.length === 0
                      && stats.counts.imports === 0 && (
                      <p className="mt-2 text-xs text-[var(--muted)]">
                        Add images to see how the dataset spreads.
                      </p>
                    )}
                    {stats.variation.points.length > 0 && stats.variation.points.length < 15 && (
                      <p className="mt-2 text-xs text-[var(--muted)]">
                        The variation plot becomes more useful as your dataset grows.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

type DupDropMember = { id: string; filename: string | null };
type DupGroup = {
  keep: string;
  keep_filename: string | null;
  drop: DupDropMember[];
  key: string;
};

// Review-modal flow under the Near-duplicates counter. The previous
// single-click "Delete all" hit a stale-cache race (the deduped count
// flashed correctly post-reload, then snapped back to the original as
// SWR served the pre-delete cached payload), fixed server-side by
// invalidating the project's payload cache + sidecars in the commit
// path. The UI is now a portal-rendered modal listing each group's
// thumbnails so the user can confirm and act per-group: "Delete
// duplicates" prunes the non-keepers, "Not duplicates" persists the
// group to `manifest.ignored_near_dups` so it stops showing up.
function RemoveDuplicatesAction({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [groups, setGroups] = useState<DupGroup[]>([]);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  // Cross-group multi-select for bulk delete. Holds the import ids
  // of non-keepers the user has ticked. Wiped on rescan / close.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  // "near" → embedding-cosine clustering (catches near-identical
  // re-encodes). "exact" → byte-hash clustering (only literally the
  // same file uploaded twice). Stored at component scope so a switch
  // can re-fetch with the new strategy.
  const [mode, setMode] = useState<"near" | "exact">("near");

  // Lock the page scroll while the review modal is open so the
  // background can't move under the user's pointer. Stashes the
  // previous overflow so we restore exactly what was there before.
  useEffect(() => {
    if (!open || typeof document === "undefined") return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  // Run the dedupe preview with the current mode. Extracted so the
  // initial open AND the in-modal toggle can both invoke it without
  // duplicating fetch logic.
  const runScan = async (m: "near" | "exact") => {
    setLoading(true);
    setError(null);
    setGroups([]);
    setSelectedIds(new Set());
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/dedupe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "preview",
          // "exact" → byte-hash only (literal duplicates).
          // "near" → embedding cosine clustering.
          strategy: m === "exact" ? "exact" : "near",
          // Match the dataset-stats card's near-duplicate threshold
          // (0.95) so the modal surfaces every pair the stats counter
          // is reporting. Using a stricter 0.98 here used to make the
          // counter say "13 near-duplicates" while the modal opened
          // empty, same data source, different cutoffs.
          threshold: 0.95,
        }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const j = (await r.json()) as { groups?: DupGroup[] };
      setGroups(j.groups ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const openModal = async () => {
    setOpen(true);
    await runScan(mode);
  };

  const switchMode = async (next: "near" | "exact") => {
    if (next === mode || loading) return;
    setMode(next);
    await runScan(next);
  };

  const deleteGroup = async (group: DupGroup) => {
    setBusyKey(group.keep);
    setError(null);
    try {
      const ids = group.drop.map((d) => d.id).filter(Boolean);
      if (ids.length === 0) return;
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/imports/delete_batch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        },
      );
      if (!r.ok) throw new Error(`http ${r.status}`);
      // Remove this group from the modal so the user can move on to
      // the next without re-running preview.
      // Filter by keeper id, each group has a unique keeper, but
      // the `key` field can collide across groups when several
      // clusters happen to share the same min_sim string (e.g.
      // multiple "min_sim=1.000" pairs). Without the keep-based
      // filter, deleting one group dropped every sibling cluster
      // with the same min_sim from the modal in one shot.
      setGroups((cur) => cur.filter((g) => g.keep !== group.keep));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyKey(null);
    }
  };

  const ignoreGroup = async (group: DupGroup) => {
    setBusyKey(group.keep);
    setError(null);
    try {
      const ids = [group.keep, ...group.drop.map((d) => d.id)].filter(Boolean);
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/imports/dedupe/ignore`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        },
      );
      if (!r.ok) throw new Error(`http ${r.status}`);
      // Filter by keeper id, each group has a unique keeper, but
      // the `key` field can collide across groups when several
      // clusters happen to share the same min_sim string (e.g.
      // multiple "min_sim=1.000" pairs). Without the keep-based
      // filter, deleting one group dropped every sibling cluster
      // with the same min_sim from the modal in one shot.
      setGroups((cur) => cur.filter((g) => g.keep !== group.keep));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyKey(null);
    }
  };

  const closeAndRefresh = () => {
    setOpen(false);
    // Hard-reload so gallery + stats sidecars repaint from post-action
    // truth. The cache-invalidation we added server-side guarantees
    // the next fetches return fresh data, so no need for a state-
    // surgery dance here.
    if (typeof window !== "undefined") window.location.reload();
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // "Select all non-keepers across every group", gives the user a
  // one-click bulk path for a fresh project's flood of near-dups.
  const allDropIds = groups.flatMap((g) => g.drop.map((d) => d.id).filter(Boolean));
  const allSelected = allDropIds.length > 0 && allDropIds.every((id) => selectedIds.has(id));
  const toggleSelectAll = () => {
    setSelectedIds(() => (allSelected ? new Set() : new Set(allDropIds)));
  };

  const bulkDelete = async () => {
    if (selectedIds.size === 0 || bulkBusy) return;
    setBulkBusy(true);
    setError(null);
    try {
      const ids = Array.from(selectedIds);
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/imports/delete_batch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        },
      );
      if (!r.ok) throw new Error(`http ${r.status}`);
      // Splice the deleted ids out of each group's drop list. A group
      // whose drops are all gone collapses to just its keeper, drop
      // those groups entirely so the modal doesn't show a lone "Keep"
      // tile with no duplicates next to it.
      const gone = new Set(ids);
      setGroups((cur) =>
        cur
          .map((g) => ({ ...g, drop: g.drop.filter((d) => !gone.has(d.id)) }))
          .filter((g) => g.drop.length > 0),
      );
      setSelectedIds(new Set());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBulkBusy(false);
    }
  };

  return (
    <>
      <div className="mt-1.5" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          onClick={openModal}
          className="text-[11px] uppercase tracking-wider font-mono text-amber-700 dark:text-amber-300 hover:text-amber-600 dark:hover:text-amber-200 transition-colors"
        >
          Review duplicates →
        </button>
      </div>
      {open && typeof window !== "undefined" && createPortal(
        <>
          <div
            className="fixed inset-0 z-[1000] bg-white/65 dark:bg-black/65"
            onClick={closeAndRefresh}
            aria-hidden
            style={{
              backdropFilter: "blur(14px) saturate(120%)",
              WebkitBackdropFilter: "blur(14px) saturate(120%)",
            }}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Review duplicate images"
            className="fixed inset-0 z-[1001] grid place-items-center p-6 pointer-events-none"
          >
            <div
              className="pointer-events-auto w-full max-w-3xl max-h-[88vh] flex flex-col rounded-2xl border border-foreground/10 overflow-hidden"
              style={{ background: "rgb(var(--surface-rgb))", boxShadow: "var(--shadow-strong)" }}
              onClick={(e) => e.stopPropagation()}
            >
              <header className="px-5 py-4 border-b border-foreground/[0.08] flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="text-base font-medium tracking-tight text-[var(--foreground)]">Review duplicate images</h2>
                  <p className="text-[12px] text-foreground/55 mt-0.5">
                    Each group is a cluster of duplicates. Choose one of:
                  </p>
                  {/* Mode toggle. Near → embedding-cosine clusters of
                      visually-similar images. Exact → byte-hash
                      clusters, only literal duplicates of the same
                      file. No threshold text exposed, the user
                      picks intent, the backend picks the cutoff. */}
                  <div
                    role="tablist"
                    aria-label="Duplicate type"
                    className="mt-3 inline-flex rounded-full border border-foreground/15 p-0.5 bg-foreground/[0.025]"
                  >
                    {([
                      { key: "near", label: "Near duplicates" },
                      { key: "exact", label: "100% duplicates" },
                    ] as const).map(({ key, label }) => {
                      const active = mode === key;
                      return (
                        <button
                          key={key}
                          type="button"
                          role="tab"
                          aria-selected={active}
                          onClick={() => void switchMode(key)}
                          disabled={loading && !active}
                          className={[
                            "rounded-full px-3 py-1 text-[11px] uppercase tracking-wider font-mono transition-colors",
                            active
                              ? "bg-foreground text-background"
                              : "text-foreground/65 hover:text-foreground",
                          ].join(" ")}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={closeAndRefresh}
                  className="text-[11px] uppercase tracking-wider font-mono text-foreground/55 hover:text-foreground transition-colors shrink-0"
                >
                  Close
                </button>
              </header>
              {!loading && groups.length > 0 && (
                <div className="px-5 py-2.5 border-b border-foreground/[0.08] flex items-center justify-between gap-3 bg-foreground/[0.015]">
                  <button
                    type="button"
                    onClick={toggleSelectAll}
                    disabled={bulkBusy}
                    className="text-[11px] uppercase tracking-wider font-mono text-foreground/70 hover:text-foreground transition-colors disabled:opacity-50"
                  >
                    {allSelected ? "Clear selection" : "Select all duplicates"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void bulkDelete()}
                    disabled={selectedIds.size === 0 || bulkBusy}
                    className="text-[11px] uppercase tracking-wider font-mono px-3 py-1 rounded-full border border-red-500/40 text-red-700 dark:text-red-300 hover:bg-red-500/[0.08] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {bulkBusy
                      ? "Deleting…"
                      : selectedIds.size === 0
                        ? "Delete selected"
                        : `Delete ${selectedIds.size} selected`}
                  </button>
                </div>
              )}
              <div className="px-5 py-4 overflow-y-auto flex-1">
                {loading && (
                  <div className="text-sm text-foreground/55 py-6">
                    {mode === "exact" ? "Scanning for exact duplicates…" : "Scanning for near duplicates…"}
                  </div>
                )}
                {!loading && !error && groups.length === 0 && (
                  <div className="text-sm text-foreground/55 py-6">No duplicate groups remain.</div>
                )}
                {error && (
                  <div className="text-sm text-red-700 dark:text-red-300 py-4">{error}</div>
                )}
                <ul className="grid gap-5">
                  {groups.map((g, gi) => (
                    <li key={g.keep} className="border border-foreground/[0.08] rounded-xl p-3">
                      <div className="flex items-center justify-between gap-3 mb-3">
                        <div className="text-[11px] uppercase tracking-wider font-mono text-foreground/60">
                          Group {gi + 1} · {g.drop.length + 1} images
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            disabled={busyKey === g.keep}
                            onClick={() => deleteGroup(g)}
                            className="text-[11px] uppercase tracking-wider font-mono px-2.5 py-1 rounded-full border border-red-500/40 text-red-700 dark:text-red-300 hover:bg-red-500/[0.08] disabled:opacity-50 transition-colors"
                          >
                            {busyKey === g.keep ? "…" : `Delete ${g.drop.length}`}
                          </button>
                          <button
                            type="button"
                            disabled={busyKey === g.keep}
                            onClick={() => ignoreGroup(g)}
                            className="text-[11px] uppercase tracking-wider font-mono px-2.5 py-1 rounded-full border border-foreground/15 text-foreground/70 hover:text-foreground hover:bg-foreground/[0.04] disabled:opacity-50 transition-colors"
                          >
                            Not duplicates
                          </button>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
                        {[
                          { id: g.keep, filename: g.keep_filename, isKeeper: true },
                          ...g.drop.map((d) => ({ id: d.id, filename: d.filename, isKeeper: false })),
                        ].map((m) => {
                          const isSelected = !m.isKeeper && selectedIds.has(m.id);
                          return (
                            <button
                              type="button"
                              key={m.id}
                              disabled={m.isKeeper || bulkBusy}
                              onClick={() => !m.isKeeper && toggleSelect(m.id)}
                              className={`relative aspect-square rounded-lg overflow-hidden border text-left transition-shadow ${
                                m.isKeeper
                                  ? "border-emerald-500/40 ring-1 ring-emerald-500/30 cursor-default"
                                  : isSelected
                                    ? "border-red-500/70 ring-2 ring-red-500/55 cursor-pointer"
                                    : "border-foreground/10 hover:border-foreground/30 cursor-pointer"
                              }`}
                            >
                              {m.filename && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img
                                  src={`${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(m.filename)}`}
                                  alt=""
                                  loading="lazy"
                                  decoding="async"
                                  className={`absolute inset-0 w-full h-full object-cover transition-opacity ${
                                    isSelected ? "opacity-55" : "opacity-100"
                                  }`}
                                />
                              )}
                              {m.isKeeper && (
                                <span className="absolute bottom-1 left-1 text-[10px] uppercase tracking-wider font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/85 text-white">
                                  Keep
                                </span>
                              )}
                              {!m.isKeeper && (
                                <span
                                  className={`absolute top-1.5 right-1.5 h-5 w-5 rounded-md grid place-items-center text-[10px] font-bold border transition-colors ${
                                    isSelected
                                      ? "bg-red-500 border-red-500 text-white"
                                      : "bg-black/50 border-white/60 text-white/0"
                                  }`}
                                  aria-hidden
                                >
                                  ✓
                                </span>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
              <footer className="px-5 py-3 border-t border-foreground/[0.08] flex items-center justify-end">
                <button
                  type="button"
                  onClick={closeAndRefresh}
                  className="text-[11px] uppercase tracking-wider font-mono px-3 py-1.5 rounded-full bg-foreground/[0.06] text-[var(--foreground)] hover:bg-foreground/[0.1] transition-colors"
                >
                  Done
                </button>
              </footer>
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}

function CounterRow({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "warn";
}) {
  const valueClass = tone === "warn" && value > 0
    ? "text-amber-700 dark:text-amber-300"
    : "text-[var(--foreground)]";
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5 border-b border-foreground/[0.06] last:border-b-0">
      <span className="text-sm text-foreground/70">{label}</span>
      <span className={`text-base font-mono tabular-nums ${valueClass}`}>{value}</span>
    </div>
  );
}

// Hoverable, portal-anchored explainer for the health score. Mirrors
// the Target Input Shape help bubble: hovering surfaces a floating
// card with the four sub-scores and a short paragraph on how the
// number is computed. Click-through is blocked while open so the
// parent button (which toggles the card) doesn't fire when the user
// is actually trying to read the explainer.
function HealthBadge({
  score,
  factors,
  counts,
}: {
  score: number;
  factors: { balance: number; coverage: number; confidence: number; uniqueness: number };
  counts: {
    imports: number;
    with_detections: number;
    detections: number;
    unsure_detections: number;
    near_duplicates: number;
  };
}) {
  const tone =
    score >= 80
      ? "emerald"
      : score >= 55
      ? "amber"
      : "red";
  const toneClasses =
    tone === "emerald"
      ? "border-emerald-500/45 bg-emerald-500/[0.10] text-emerald-800 dark:text-emerald-200"
      : tone === "amber"
      ? "border-amber-500/50 bg-amber-500/[0.12] text-amber-800 dark:text-amber-200"
      : "border-red-500/50 bg-red-500/[0.10] text-red-800 dark:text-red-200";

  const triggerRef = useRef<HTMLSpanElement | null>(null);
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);
  const timerRef = useRef<number | null>(null);
  const onEnter = () => {
    if (timerRef.current) { window.clearTimeout(timerRef.current); timerRef.current = null; }
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) {
      // Anchor the tooltip just under the badge, right-aligned so
      // it doesn't dangle off the page on narrow viewports.
      setAnchor({ x: Math.max(12, rect.right - 320), y: rect.bottom + 8 });
    }
    setOpen(true);
  };
  const onLeave = () => {
    timerRef.current = window.setTimeout(() => setOpen(false), 80);
  };
  useEffect(() => () => { if (timerRef.current) window.clearTimeout(timerRef.current); }, []);

  return (
    <span
      ref={triggerRef}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onClick={(e) => e.stopPropagation()}
      className={`relative inline-flex items-center gap-3 rounded-full border px-3 py-1.5 ${toneClasses}`}
    >
      <span className="text-[10px] uppercase tracking-[0.18em] font-mono opacity-80">Health</span>
      <span className="text-base font-mono tabular-nums">{score}</span>
      {open && anchor && typeof window !== "undefined" && createPortal(
        <>
          <div
            className="fixed inset-0 z-[1000] pointer-events-none"
            style={{ background: "rgb(var(--background-rgb) / 0.35)" }}
            aria-hidden="true"
          />
          <div
            role="tooltip"
            className="fixed z-[1002] w-80 rounded-2xl border border-foreground/10 p-4 pointer-events-none"
            style={{
              left: anchor.x,
              top: anchor.y,
              background: "rgb(var(--surface-rgb))",
              boxShadow: "var(--shadow-strong)",
            }}
          >
            <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
              Dataset health · {score}/100
            </h3>
            <p className="text-xs text-foreground/70 leading-relaxed mb-3">
              The score blends four equally-weighted factors. Lower factors are usually the cheapest wins, fix the lowest one first.
            </p>
            <ul className="grid gap-1.5">
              <FactorRow
                name="Balance"
                value={factors.balance}
                hint="Even spread of detections across labels"
              />
              <FactorRow
                name="Coverage"
                value={factors.coverage}
                hint={`${counts.with_detections}/${counts.imports} images annotated`}
              />
              <FactorRow
                name="Confidence"
                value={factors.confidence}
                hint={`${counts.unsure_detections} unsure of ${counts.detections}`}
              />
              <FactorRow
                name="Uniqueness"
                value={factors.uniqueness}
                hint={`${counts.near_duplicates} near-duplicate image${counts.near_duplicates === 1 ? "" : "s"}`}
              />
            </ul>
          </div>
        </>,
        document.body,
      )}
    </span>
  );
}

function FactorRow({
  name,
  value,
  hint,
}: {
  name: string;
  value: number;
  hint: string;
}) {
  const pct = Math.round(value * 100);
  // Match the badge tone bands per-factor so the user can see at a
  // glance which sub-score is dragging the headline number down.
  const tone =
    pct >= 80 ? "emerald" : pct >= 55 ? "amber" : "red";
  const bar =
    tone === "emerald" ? "bg-emerald-500/70"
    : tone === "amber" ? "bg-amber-500/80"
    : "bg-red-500/75";
  return (
    <li className="grid gap-1">
      <div className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="text-foreground/85 font-medium">{name}</span>
        <span className="font-mono tabular-nums text-foreground/55">{pct}%</span>
      </div>
      <div className="h-1 rounded-full bg-foreground/[0.08] overflow-hidden">
        <div className={`h-full rounded-full ${bar}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[10px] text-foreground/45 leading-relaxed">{hint}</p>
    </li>
  );
}

// ─── Labels list with conditional edge fade ──────────────────────
// Scroll container that only paints the top/bottom mask-image fade
// when the content actually overflows its cap. Without this every
// label set, even short ones that fit the cap with room to spare ,
// got the soft-edge mask, which read as a styled fade-out for no
// reason. Watches both the container and its child via a single
// ResizeObserver so a window resize or a labels-list change
// re-evaluates without needing a parent re-render.

function LabelsList({
  children,
  maxHeight,
}: {
  children: React.ReactNode;
  maxHeight: string;
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [overflowing, setOverflowing] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      // 1px slack so subpixel rounding doesn't flicker the fade
      // on near-fitting content.
      setOverflowing(el.scrollHeight - el.clientHeight > 1);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    // Also watch the inner content, adding/removing labels changes
    // scrollHeight without changing the container's own box size.
    if (el.firstElementChild instanceof Element) {
      ro.observe(el.firstElementChild);
    }
    return () => ro.disconnect();
  }, []);

  const mask = overflowing
    ? "linear-gradient(to bottom, transparent 0, #000 20px, #000 calc(100% - 20px), transparent 100%)"
    : undefined;

  return (
    <div
      ref={scrollRef}
      className="overflow-y-auto pr-1"
      style={{
        maxHeight,
        maskImage: mask,
        WebkitMaskImage: mask,
      }}
    >
      {children}
    </div>
  );
}

export function VariationPlot({
  points,
  augmentations,
  colourFor,
  nearDuplicateIds,
  onJumpToImport,
  computing = false,
  totalImports = 0,
  embeddingsReady = 0,
}: {
  points: StatsPoint[];
  augmentations: AugmentPoint[];
  colourFor: (label: string | null) => string;
  nearDuplicateIds: Set<string>;
  onJumpToImport?: (importId: string, kind: "image" | "augmentation") => void;
  /** True when the dataset has images that don't yet have embeddings.
      The plot can't render until at least one point exists, so the
      empty-state turns into a spinner with progress copy instead of
      the dashed "no embeddings yet" tile. */
  computing?: boolean;
  totalImports?: number;
  embeddingsReady?: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<StatsPoint | null>(null);

  if (points.length === 0) {
    if (computing) {
      const pct = totalImports > 0
        ? Math.min(100, Math.round((embeddingsReady / totalImports) * 100))
        : 0;
      // Conic progress ring, fills clockwise from 12 o'clock as the
      // backend completes embeddings. The CSS-keyframe spinner runs
      // on top as a "still alive" affordance even when pct doesn't
      // advance for a few seconds.
      const ringSize = 56;
      return (
        <div
          ref={wrapRef}
          className="relative aspect-square rounded-xl border border-foreground/10 bg-foreground/[0.02] grid place-items-center"
        >
          <style>{`
            @keyframes pk-stats-spin { to { transform: rotate(360deg); } }
            .pk-stats-spin { animation: pk-stats-spin 1.1s linear infinite; transform-origin: center; }
          `}</style>
          <div className="flex flex-col items-center gap-3 text-center px-6">
            <div className="relative grid place-items-center" style={{ width: ringSize, height: ringSize }}>
              {/* Progress ring, conic gradient driven by the live
                  embeddings_ready / imports ratio. Filled portion in
                  foreground, remainder in foreground/10. */}
              <div
                aria-hidden
                className="absolute inset-0 rounded-full"
                style={{
                  background: `conic-gradient(rgb(var(--foreground-rgb)) ${pct * 3.6}deg, rgb(var(--foreground-rgb) / 0.10) ${pct * 3.6}deg)`,
                  // Knock out the centre to reveal the value text.
                  mask: "radial-gradient(circle, transparent 0 18px, #000 19px)",
                  WebkitMask: "radial-gradient(circle, transparent 0 18px, #000 19px)",
                }}
              />
              {/* Live spinning arc on top, uses CSS keyframes (SMIL
                  animateTransform didn't apply to the bare SVG path
                  in every browser). */}
              <svg
                viewBox="0 0 24 24"
                className="pk-stats-spin h-7 w-7 text-foreground/55"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              >
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            </div>
            <span className="text-xs text-foreground/65">
              Computing embeddings
            </span>
          </div>
        </div>
      );
    }
    return (
      <div
        ref={wrapRef}
        className="relative aspect-square grid place-items-center rounded-xl border border-foreground/10 bg-foreground/[0.02]"
      >
        {/* Loading wheel rather than a "no embeddings" message, so the plot
            reads as still loading while the embeddings/projection arrive. */}
        <span className="h-7 w-7 animate-spin rounded-full border-2 border-foreground/15 border-t-[var(--accent-orange)]" aria-label="Loading image variation" role="status" />
      </div>
    );
  }

  const inset = 8;
  const project = (v: number): number => inset + ((v + 1) / 2) * (100 - inset * 2);

  return (
    <div
      ref={wrapRef}
      className="relative aspect-square rounded-xl border border-foreground/10 bg-foreground/[0.02] overflow-hidden"
    >
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        {[25, 50, 75].map((g) => (
          <g key={g} stroke="rgb(var(--foreground-rgb) / 0.06)" strokeWidth="0.4">
            <line x1={g} y1="0" x2={g} y2="100" />
            <line x1="0" y1={g} x2="100" y2={g} />
          </g>
        ))}
        {/* Augmentations first so the originals paint on top. */}
        {augmentations.map((p) => {
          const cx = project(p.x);
          const cy = project(p.y);
          const colour = colourFor(p.label);
          return (
            <circle
              key={p.id}
              cx={cx}
              cy={cy}
              r={1}
              fill={colour}
              fillOpacity={0.55}
              // Same contrast-stroke trick as the originals, keeps
              // the augment dots visible when the label colour is
              // near-white in light mode or near-black in dark.
              stroke={readableTextForBg(colour)}
              strokeOpacity={0.55}
              strokeWidth={0.35}
              style={{ cursor: onJumpToImport ? "pointer" : "default" }}
              onClick={(e) => {
                if (!onJumpToImport) return;
                e.stopPropagation();
                onJumpToImport(p.source_id, "augmentation");
              }}
              onPointerEnter={() => setHover({
                id: p.id,
                filename: p.filename,
                x: p.x,
                y: p.y,
                label: p.label,
                n_detections: 0,
              })}
              onPointerLeave={() => setHover((cur) => (cur?.id === p.id ? null : cur))}
            />
          );
        })}
        {points.map((p) => {
          const cx = project(p.x);
          const cy = project(p.y);
          const r = 1.0 + Math.min(2.0, p.n_detections * 0.35);
          const isDup = nearDuplicateIds.has(p.id);
          const colour = colourFor(p.label);
          return (
            <g
              key={p.id}
              onPointerEnter={() => setHover(p)}
              onPointerLeave={() => setHover((cur) => (cur?.id === p.id ? null : cur))}
              onClick={(e) => {
                if (!onJumpToImport) return;
                e.stopPropagation();
                onJumpToImport(p.id, "image");
              }}
              style={{ cursor: onJumpToImport ? "pointer" : "default" }}
            >
              {isDup && (
                <circle cx={cx} cy={cy} r={r + 1.6} fill="none" stroke="rgb(245 158 11 / 0.7)" strokeWidth="0.5" />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={colour}
                // Outline contrasts with the dot's OWN fill: a white
                // label dot gets a dark ring, a near-black label dot
                // gets a light ring. Works on both light- and dark-
                // mode plot backgrounds since the visibility comes
                // from the colour-versus-fill contrast, not the bg.
                stroke={readableTextForBg(colour)}
                strokeWidth="0.45"
                strokeOpacity="0.6"
              />
            </g>
          );
        })}
      </svg>
      {hover && (
        <div
          className="absolute bottom-2 left-2 right-2 px-2 py-1.5 rounded-md text-[11px] font-mono pointer-events-none truncate text-[var(--foreground)]"
          style={{
            // Translucent + blurred, sits over the plot without
            // hard-fading the dots underneath.
            background: "rgb(var(--background-rgb) / 0.55)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            border: "1px solid rgb(var(--foreground-rgb) / 0.08)",
          }}
        >
          <span className="truncate">{hover.filename ?? hover.id}</span>
          {hover.label && <span className="opacity-70 ml-2">· {hover.label}</span>}
          {hover.n_detections > 0 && <span className="opacity-70 ml-2">· {hover.n_detections} det.</span>}
          {nearDuplicateIds.has(hover.id) && <span className="opacity-80 ml-2 text-amber-700 dark:text-amber-300">· near-dup</span>}
        </div>
      )}
    </div>
  );
}
