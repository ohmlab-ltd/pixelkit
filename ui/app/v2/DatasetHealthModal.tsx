"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "../../lib/apiFetch";
import { colourForLabelStable } from "./OnboardLabelsV2";
import { VariationPlot, type StatsPoint, type AugmentPoint } from "./DatasetStatsCard";
import { FactorRadar, FACTOR_INFO, FACTOR_ORDER } from "./HealthRadar";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

// One modal, two equal-height columns: dataset HEALTH on the left, the
// near-duplicate REVIEW (already expanded) on the right. No pop-up within a
// pop-up: the duplicates review is inline, not a second modal. Opaque panel
// over a soft white-blur backdrop.

type HealthStats = {
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
  health: { score: number; factors: { balance: number; coverage: number; confidence: number; uniqueness: number } };
  variation?: { points: StatsPoint[]; augmentations?: AugmentPoint[]; near_duplicate_ids: string[] };
};

type DupGroup = { keep: string; keep_filename: string | null; drop: { id: string; filename: string | null }[]; key: string };

function Spinner({ label }: { label?: string }) {
  return (
    <div className="m-auto flex flex-col items-center gap-3 py-10 text-[13px] text-foreground/55">
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-foreground/20 border-t-[var(--accent-orange)]" />
      {label && <span>{label}</span>}
    </div>
  );
}

function scoreTone(score: number): string {
  if (score >= 75) return "var(--success)";
  if (score >= 45) return "var(--warning)";
  return "var(--destructive)";
}

// ─── Left column: health summary ───
function HealthSummary({ stats, labelColours }: { stats: HealthStats; labelColours?: Record<string, string> | null }) {
  const counters: { label: string; value: number; warn?: boolean }[] = [
    { label: "Images", value: stats.counts.imports },
    { label: "With detections", value: stats.counts.with_detections },
    { label: "Total detections", value: stats.counts.detections },
    { label: "Unsure detections", value: stats.counts.unsure_detections, warn: stats.counts.unsure_detections > 0 },
    { label: "Augmentations", value: stats.counts.augmentations },
    { label: "Near-duplicates", value: stats.counts.near_duplicates, warn: stats.counts.near_duplicates > 0 },
  ];
  const total = stats.labels.reduce((a, l) => a + l.count, 0);
  const max = stats.labels.reduce((a, l) => Math.max(a, l.count), 0);
  const tone = scoreTone(stats.health.score);
  const colourFor = (lab: string | null) => (lab ? (labelColours?.[lab] ?? colourForLabelStable(lab)) : "#9ca3af");
  const factorVals = FACTOR_ORDER.map((k) => ({ key: k as string, value: Math.max(0, Math.min(1, stats.health.factors?.[k] ?? 0)) }));
  const variation = stats.variation;

  return (
    <div className="space-y-6">
      {/* Headline score */}
      <div>
        <div className="flex items-end gap-3">
          <span className="text-[44px] font-semibold leading-none tracking-tight text-foreground tabular-nums">
            {Math.round(stats.health.score)}
          </span>
          <span className="mb-1.5 text-[13px] font-medium text-foreground/55">/ 100 health</span>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-foreground/[0.08]">
          <div className="h-full rounded-full" style={{ width: `${Math.max(2, Math.min(100, stats.health.score))}%`, background: tone, transition: "width 320ms ease-out" }} />
        </div>
      </div>

      {/* Factor spider + per-factor explanations */}
      <div className="flex flex-col items-center gap-4 sm:flex-row">
        <FactorRadar values={factorVals} />
        <ul className="min-w-0 flex-1 space-y-2.5">
          {factorVals.map((f) => (
            <li key={f.key}>
              <div className="flex items-baseline justify-between gap-2 text-[12.5px]">
                <span className="font-medium text-foreground/85">{FACTOR_INFO[f.key].label}</span>
                <span className="shrink-0 tabular-nums font-semibold text-foreground/60">{Math.round(f.value * 100)}%</span>
              </div>
              <p className="mt-0.5 text-[11.5px] leading-snug text-foreground/55">{FACTOR_INFO[f.key].desc}</p>
            </li>
          ))}
        </ul>
      </div>

      {/* Image variation (embeddings projection) */}
      <div>
        <div className="mb-2.5 flex items-center justify-between gap-2">
          <h4 className="text-[12px] font-semibold uppercase tracking-wider text-foreground/55">Image variation</h4>
          {stats.counts.embeddings_ready > 0 && (
            <span className="text-[11px] text-foreground/45">{stats.counts.embeddings_ready}/{stats.counts.imports} embedded</span>
          )}
        </div>
        <VariationPlot
          points={variation?.points ?? []}
          augmentations={variation?.augmentations ?? []}
          colourFor={colourFor}
          nearDuplicateIds={new Set(variation?.near_duplicate_ids ?? [])}
          computing={stats.counts.imports > 0 && stats.counts.embeddings_ready < stats.counts.imports}
          totalImports={stats.counts.imports}
          embeddingsReady={stats.counts.embeddings_ready}
        />
      </div>

      {/* Counters */}
      <div className="grid grid-cols-2 gap-2">
        {counters.map((c) => (
          <div key={c.label} className="rounded-xl border border-foreground/[0.08] bg-foreground/[0.02] px-3 py-2.5">
            <div className={`text-[18px] font-semibold leading-none tabular-nums ${c.warn ? "text-[var(--warning)]" : "text-foreground"}`}>{c.value}</div>
            <div className="mt-1 text-[11.5px] text-foreground/60">{c.label}</div>
          </div>
        ))}
      </div>

      {/* Label distribution */}
      <div>
        <h4 className="mb-2.5 text-[12px] font-semibold uppercase tracking-wider text-foreground/55">Label distribution</h4>
        {stats.labels.length === 0 ? (
          <p className="text-[13px] text-foreground/55">No annotations yet.</p>
        ) : (
          <ul className="space-y-2">
            {stats.labels.map((row) => {
              const pct = total > 0 ? (row.count / total) * 100 : 0;
              const fill = max > 0 ? (row.count / max) * 100 : 0;
              const colour = labelColours?.[row.label] ?? colourForLabelStable(row.label);
              return (
                <li key={row.label} className="grid gap-1">
                  <div className="flex items-baseline justify-between gap-2 text-[12.5px]">
                    <span className="inline-flex min-w-0 items-center gap-1.5 text-foreground/85">
                      <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ background: colour }} />
                      <span className="truncate">{row.label}</span>
                    </span>
                    <span className="shrink-0 tabular-nums font-medium text-foreground/60">
                      {row.count} <span className="text-foreground/40">({pct.toFixed(0)}%)</span>
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-foreground/[0.08]">
                    <div className="h-full rounded-full" style={{ width: `${fill}%`, background: colour, transition: "width 240ms ease-out" }} />
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

// ─── Right column: inline duplicates review ───
function DuplicatesReviewPanel({ projectId, onChanged }: { projectId: string; onChanged?: () => void }) {
  const [mode, setMode] = useState<"near" | "exact">("near");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [groups, setGroups] = useState<DupGroup[]>([]);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);

  const scan = useCallback(async (m: "near" | "exact") => {
    setLoading(true);
    setError(null);
    setGroups([]);
    setSelected(new Set());
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/dedupe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "preview", strategy: m === "exact" ? "exact" : "near", threshold: 0.95 }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const j = (await r.json()) as { groups?: DupGroup[] };
      setGroups(j.groups ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { void scan("near"); }, [scan]);

  const switchMode = (next: "near" | "exact") => {
    if (next === mode || loading) return;
    setMode(next);
    void scan(next);
  };

  const deleteGroup = async (g: DupGroup) => {
    const ids = g.drop.map((d) => d.id).filter(Boolean);
    if (ids.length === 0) return;
    setBusyKey(g.keep);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/delete_batch`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      setGroups((cur) => cur.filter((x) => x.keep !== g.keep));
      onChanged?.();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusyKey(null); }
  };

  const ignoreGroup = async (g: DupGroup) => {
    const ids = [g.keep, ...g.drop.map((d) => d.id)].filter(Boolean);
    setBusyKey(g.keep);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/dedupe/ignore`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      setGroups((cur) => cur.filter((x) => x.keep !== g.keep));
      onChanged?.();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusyKey(null); }
  };

  const toggleSelect = (id: string) => setSelected((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const bulkDelete = async () => {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    setBulkBusy(true);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/delete_batch`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const gone = new Set(ids);
      setGroups((cur) => cur.map((g) => ({ ...g, drop: g.drop.filter((d) => !gone.has(d.id)) })).filter((g) => g.drop.length > 0));
      setSelected(new Set());
      onChanged?.();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBulkBusy(false); }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 pb-3">
        <div role="tablist" aria-label="Duplicate type" className="inline-flex rounded-full border border-foreground/15 bg-foreground/[0.025] p-0.5">
          {([{ key: "near", label: "Near duplicates" }, { key: "exact", label: "100% duplicates" }] as const).map(({ key, label }) => {
            const active = mode === key;
            return (
              <button key={key} type="button" role="tab" aria-selected={active} disabled={loading && !active}
                onClick={() => switchMode(key)}
                className={`rounded-full px-3 py-1 text-[12px] font-medium transition-colors ${active ? "bg-foreground text-background" : "text-foreground/65 hover:text-foreground"}`}>
                {label}
              </button>
            );
          })}
        </div>
        {selected.size > 0 && (
          <button type="button" onClick={() => void bulkDelete()} disabled={bulkBusy}
            className="shrink-0 rounded-full bg-[var(--destructive)] px-3 py-1.5 text-[12px] font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-50">
            {bulkBusy ? "Deleting…" : `Delete ${selected.size} selected`}
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        {loading ? (
          <Spinner label={mode === "exact" ? "Scanning for exact duplicates…" : "Scanning for near duplicates…"} />
        ) : error ? (
          <p className="py-6 text-[13px] text-[var(--destructive)]">{error}</p>
        ) : groups.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-1.5 py-10 text-center">
            <span className="grid h-10 w-10 place-items-center rounded-full bg-foreground/[0.06] text-[var(--success)]">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12.5l5 5 11-11" /></svg>
            </span>
            <p className="text-[13px] font-medium text-foreground/70">No duplicates to review</p>
            <p className="text-[12px] text-foreground/55">This dataset looks clean.</p>
          </div>
        ) : (
          <ul className="space-y-4">
            {groups.map((g, gi) => (
              <li key={g.keep} className="rounded-xl border border-foreground/[0.08] p-3">
                <div className="mb-2.5 flex items-center justify-between gap-3">
                  <span className="text-[12px] font-medium text-foreground/65">Group {gi + 1} · {g.drop.length + 1} images</span>
                  <div className="flex items-center gap-2">
                    <button type="button" disabled={busyKey === g.keep} onClick={() => void deleteGroup(g)}
                      className="rounded-full border border-[var(--destructive)]/40 px-2.5 py-1 text-[11.5px] font-medium text-[var(--destructive)] transition-colors hover:bg-[var(--destructive)]/[0.08] disabled:opacity-50">
                      {busyKey === g.keep ? "…" : `Delete ${g.drop.length}`}
                    </button>
                    <button type="button" disabled={busyKey === g.keep} onClick={() => void ignoreGroup(g)}
                      className="rounded-full border border-foreground/15 px-2.5 py-1 text-[11.5px] font-medium text-foreground/70 transition-colors hover:bg-foreground/[0.04] disabled:opacity-50">
                      Not duplicates
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                  {[{ id: g.keep, filename: g.keep_filename, keeper: true }, ...g.drop.map((d) => ({ id: d.id, filename: d.filename, keeper: false }))].map((m) => {
                    const sel = !m.keeper && selected.has(m.id);
                    return (
                      <button key={m.id} type="button" disabled={m.keeper || bulkBusy} onClick={() => !m.keeper && toggleSelect(m.id)}
                        className={`relative aspect-square overflow-hidden rounded-lg border text-left transition-shadow ${m.keeper ? "cursor-default border-[var(--success)]/40 ring-1 ring-[var(--success)]/30" : sel ? "cursor-pointer border-[var(--destructive)]/70 ring-2 ring-[var(--destructive)]/55" : "cursor-pointer border-foreground/10 hover:border-foreground/30"}`}>
                        {m.filename && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={`${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(m.filename)}`} alt="" loading="lazy" decoding="async"
                            className={`absolute inset-0 h-full w-full object-cover ${sel ? "opacity-55" : "opacity-100"}`} />
                        )}
                        {m.keeper && <span className="absolute bottom-1 left-1 rounded-full bg-[var(--success)] px-1.5 py-0.5 text-[10px] font-semibold text-white">Keep</span>}
                      </button>
                    );
                  })}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function DatasetHealthModal({
  open,
  onClose,
  projectId,
  labelColours,
  refreshSignal = 0,
}: {
  open: boolean;
  onClose: () => void;
  projectId: string | null | undefined;
  labelColours?: Record<string, string> | null;
  refreshSignal?: number;
}) {
  const [stats, setStats] = useState<HealthStats | null>(null);
  const [loading, setLoading] = useState(false);
  const restoreFocus = useRef<HTMLElement | null>(null);

  const load = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/dataset-stats?v=${refreshSignal}`, { cache: "no-store" });
      if (r.ok) setStats((await r.json()) as HealthStats);
    } catch { /* keep seed */ }
    finally { setLoading(false); }
  }, [projectId, refreshSignal]);

  useEffect(() => {
    if (!open) return;
    restoreFocus.current = (document.activeElement as HTMLElement) ?? null;
    void load();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    document.addEventListener("keydown", onKey, true);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = prev;
      restoreFocus.current?.focus?.();
    };
  }, [open, load, onClose]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="presentation"
      style={{ background: "rgb(var(--surface-rgb) / 0.55)", backdropFilter: "blur(14px) saturate(115%)", WebkitBackdropFilter: "blur(14px) saturate(115%)" }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Dataset health"
        className="pk-pop flex max-h-[88vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-foreground/10"
        style={{ background: "rgb(var(--surface-rgb))", boxShadow: "var(--shadow-strong)" }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 border-b border-foreground/[0.08] px-5 py-4">
          <h2 className="text-[16px] font-semibold tracking-tight text-foreground">Dataset health</h2>
          <button type="button" onClick={onClose} aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-full text-foreground/55 outline-none transition-colors hover:bg-foreground/[0.08] hover:text-foreground focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]">
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-2">
          {/* LEFT: health */}
          <div className="min-h-0 overflow-y-auto px-5 py-5 lg:border-r lg:border-foreground/[0.08]">
            {stats ? <HealthSummary stats={stats} labelColours={labelColours} /> : <Spinner label={loading ? "Loading dataset health…" : undefined} />}
          </div>
          {/* RIGHT: duplicates review, inline + expanded */}
          <div className="flex min-h-0 flex-col px-5 py-5">
            <h3 className="mb-1 text-[13px] font-semibold tracking-tight text-foreground">Review duplicates</h3>
            <p className="mb-3 text-[12px] text-foreground/55">Each group is a cluster of duplicates. Keep one, delete the rest, or mark them as not duplicates.</p>
            {projectId && <DuplicatesReviewPanel projectId={projectId} onChanged={load} />}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
