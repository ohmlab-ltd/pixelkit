"use client";

import { useEffect, useMemo, useState } from "react";
import { BlurhashCanvas } from "react-blurhash";

import { apiFetch } from "../../lib/apiFetch";
import { colourForLabelStable } from "./OnboardLabelsV2";
import { FactorRadar, FACTOR_ORDER, FACTOR_INFO, factorColour } from "./HealthRadar";
import { VariationPlot, type StatsPoint, type AugmentPoint } from "./DatasetStatsCard";
import type { StoreImport } from "./store/projectStore";

// Project Overview body. Answers "how is this project doing?" at a glance:
// 5 stat cards (incl. a Health card) -> Insights & suggestions -> Recent images
// + Recent activity side by side -> a bottom analytics row (Dataset health
// spider + factors / Label distribution / Image variation). The Derived
// datasets strip is rendered by the parent below. No dataset/upload controls
// live here. Width is capped so it stays natural on very wide screens.

type Stats = {
  counts?: { imports?: number; with_detections?: number; detections?: number; augmentations?: number; near_duplicates?: number; embeddings_ready?: number };
  labels?: { label: string; count: number }[];
  health?: { score?: number; factors?: Record<string, number> };
  variation?: { points: StatsPoint[]; augmentations?: AugmentPoint[]; near_duplicate_ids: string[] };
} | null;

type RefImage = { referenceId?: string; filename?: string | null; preview: string; blurhash?: string | null };

function toMs(v: number | string | null | undefined): number {
  if (v == null) return 0;
  if (typeof v === "number") return v < 1e12 ? v * 1000 : v;
  const t = Date.parse(v);
  return Number.isNaN(t) ? 0 : t;
}

function relTime(ms: number): string {
  if (!ms) return "";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d ago`;
  const mo = Math.floor(d / 30);
  if (mo < 12) return `${mo}mo ago`;
  return `${Math.floor(mo / 12)}y ago`;
}

function Donut({ slices }: { slices: { label: string; count: number; colour: string }[] }) {
  const total = slices.reduce((a, s) => a + s.count, 0);
  const r = 34;
  const sw = 13;
  const C = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg viewBox="0 0 88 88" className="h-[84px] w-[84px] shrink-0 -rotate-90">
      <circle cx="44" cy="44" r={r} fill="none" stroke="currentColor" strokeWidth={sw} className="text-foreground/[0.08]" />
      {total > 0 &&
        slices.map((s) => {
          const len = (s.count / total) * C;
          const seg = (
            <circle key={s.label} cx="44" cy="44" r={r} fill="none" stroke={s.colour} strokeWidth={sw} strokeDasharray={`${len} ${C - len}`} strokeDashoffset={-offset} strokeLinecap="butt" />
          );
          offset += len;
          return seg;
        })}
    </svg>
  );
}

function Card({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="pk-card flex min-h-[120px] flex-col rounded-md p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="pk-eyebrow">{title}</h3>
        {action}
      </div>
      <div className="flex flex-1 flex-col">{children}</div>
    </div>
  );
}

// Quiet tool-style section header: 11px uppercase mono, optional count,
// optional right-aligned action. No accent decoration — the accent is
// reserved for the primary action / active states.
function SectionHeader({ children, count, action }: { children: React.ReactNode; count?: number; action?: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="pk-micro flex items-baseline gap-2 text-[var(--fg-soft)]">
        {children}
        {typeof count === "number" && (
          <span className="tabular-nums text-[var(--fg-dim)]">{count}</span>
        )}
      </h2>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

function HeaderLink({ onClick, children }: { onClick?: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="font-mono text-[11px] uppercase tracking-wider text-foreground/55 outline-none transition-colors hover:text-foreground focus-visible:underline"
    >
      {children}
    </button>
  );
}

function InsightCard({ tone, title, desc, onClick }: { tone: "warn" | "good" | "info"; title: string; desc: string; onClick?: () => void }) {
  // Tone drives the icon colour + its chip wash so a warning reads at a glance
  // without a loud full-card tint — amber = attention, emerald = healthy,
  // neutral = still computing.
  const palette =
    tone === "warn"
      ? { text: "text-[var(--warn)]", chip: "bg-foreground/[0.05]" }
      : tone === "good"
        ? { text: "text-[var(--ok)]", chip: "bg-foreground/[0.05]" }
        : { text: "text-foreground/55", chip: "bg-foreground/[0.05]" };
  const icon =
    tone === "warn" ? (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 9v4M12 17h.01M10.3 4.3 2.5 18a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z" /></svg>
    ) : tone === "good" ? (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12.5l5 5 11-11" /></svg>
    ) : (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9" /><path d="M12 8h.01M11 12h1v4h1" /></svg>
    );
  const base = "flex w-full items-start gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] p-3.5 text-left";
  const inner = (
    <>
      <span className={`mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-md ${palette.chip} ${palette.text}`}>{icon}</span>
      <span className="min-w-0">
        <span className="block text-[13px] font-semibold leading-tight text-foreground">{title}</span>
        <span className="mt-1 block text-[12px] leading-snug text-foreground/55">{desc}</span>
      </span>
    </>
  );
  if (onClick) {
    return (
      <button type="button" onClick={onClick} className={`${base} outline-none transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]`}>
        {inner}
      </button>
    );
  }
  return <div className={base}>{inner}</div>;
}

export function OverviewPanel({
  imports,
  importsTotal,
  // `labels` (the label names) is still accepted by callers but the panel no
  // longer renders a Labels count — that moved to the hero stat strip — so it's
  // intentionally not destructured here.
  labelColours,
  projectId,
  refreshSignal = 0,
  refs = [],
  showReferences = true,
  seedStats,
  onOpenHealth,
  onOpenReferences,
  onOpenDataset,
  onJumpToImport,
}: {
  imports: StoreImport[];
  importsTotal: number | null;
  labels: string[];
  labelColours?: Record<string, string> | null;
  projectId: string | null | undefined;
  refreshSignal?: number;
  refs?: RefImage[];
  /** Hidden for derived datasets — they don't manage their own
      references. Everywhere else the section shows (references are
      optional on any dataset). */
  showReferences?: boolean;
  seedStats: Stats;
  onOpenHealth: () => void;
  onOpenReferences?: () => void;
  onOpenDataset?: () => void;
  onJumpToImport?: (id: string) => void;
}) {
  // Seed from a localStorage cache so the health bars, spider and variation
  // plot paint INSTANTLY (possibly slightly stale) on load, then refresh in the
  // background. Avoids the ~1s blank-then-pop when waiting on /dataset-stats.
  const statsKey = projectId ? `pk_stats:${projectId}` : null;
  const [full, setFull] = useState<Stats>(() => {
    if (typeof window === "undefined" || !statsKey) return null;
    try { const raw = window.localStorage.getItem(statsKey); return raw ? (JSON.parse(raw) as Stats) : null; } catch { return null; }
  });
  useEffect(() => {
    if (!projectId || !statsKey) return;
    let alive = true;
    // Re-seed from cache when the project changes (the initial state only ran
    // for the first projectId).
    try { const raw = window.localStorage.getItem(statsKey); if (raw && alive) setFull(JSON.parse(raw) as Stats); } catch {}
    apiFetch(`/api/v2/projects/${projectId}/dataset-stats?v=${refreshSignal}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!alive || !d) return;
        setFull(d as Stats);
        try { window.localStorage.setItem(statsKey, JSON.stringify(d)); } catch { /* quota */ }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [projectId, statsKey, refreshSignal]);

  const images = importsTotal ?? imports.length;
  const score = full?.health?.score ?? seedStats?.health?.score;
  const factors = full?.health?.factors;
  const withDet = full?.counts?.with_detections;
  const nearDups = full?.counts?.near_duplicates;
  const variation = full?.variation;

  const dist = useMemo(() => {
    const map = new Map<string, number>();
    const src = full?.labels ?? seedStats?.labels;
    if (src?.length) {
      for (const l of src) map.set(l.label, l.count);
    } else {
      for (const im of imports) for (const [k, v] of Object.entries(im.labelStats ?? {})) map.set(k, (map.get(k) ?? 0) + (v ?? 0));
    }
    return Array.from(map.entries())
      .filter(([, c]) => c > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([label, count]) => ({ label, count, colour: labelColours?.[label] ?? colourForLabelStable(label) }));
  }, [full, seedStats, imports, labelColours]);

  const activity = useMemo(() => {
    const ev: { id: string; importId: string; kind: "added" | "labelled"; name: string; at: number }[] = [];
    for (const im of imports) {
      const name = im.filename ?? "image";
      const c = toMs(im.createdAt);
      if (c) ev.push({ id: `${im.id}-a`, importId: im.id, kind: "added", name, at: c });
      const l = toMs(im.labelledAt);
      if (l) ev.push({ id: `${im.id}-l`, importId: im.id, kind: "labelled", name, at: l });
    }
    return ev.sort((a, b) => b.at - a.at).slice(0, 6);
  }, [imports]);

  const recent = useMemo(
    () => [...imports].sort((a, b) => toMs(b.createdAt) - toMs(a.createdAt)).slice(0, 5),
    [imports],
  );

  const factorVals = FACTOR_ORDER.map((k) => ({ key: k as string, value: Math.max(0, Math.min(1, factors?.[k] ?? 0)) }));
  const statsLoading = full === null;
  const healthLabel = typeof score === "number" ? (score >= 75 ? "Healthy" : score >= 45 ? "Fair" : "Needs work") : "";
  const colourFor = (lab: string | null) => (lab ? (labelColours?.[lab] ?? colourForLabelStable(lab)) : "#9ca3af");

  // Insights
  const noDet = typeof withDet === "number" ? Math.max(0, images - withDet) : null;
  const under = (() => {
    if (dist.length < 2) return null;
    const maxC = dist[0].count;
    const minRow = dist[dist.length - 1];
    return minRow.count < maxC * 0.4 ? minRow : null;
  })();

  return (
    <section className="space-y-7 px-6 pt-5 pb-12 lg:px-10">
      {/* Insights — fast rule-based signals. */}
      <div>
        <SectionHeader>Insights</SectionHeader>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {statsLoading || nearDups == null ? (
            <InsightCard tone="info" title="Duplicates" desc="Checking your images for near-duplicates." />
          ) : nearDups > 0 ? (
            <InsightCard tone="warn" title={`${nearDups} near-duplicate ${nearDups === 1 ? "image" : "images"}`} desc="Open dataset health to review and remove them." onClick={onOpenHealth} />
          ) : (
            <InsightCard tone="good" title="No duplicates" desc="Your images all look distinct." />
          )}
          {noDet == null ? (
            <InsightCard tone="info" title="Detection coverage" desc="Checking which images have detections." />
          ) : noDet > 0 ? (
            <InsightCard tone="warn" title={`${noDet} ${noDet === 1 ? "image has" : "images have"} no detections`} desc="Label or remove them to improve coverage." />
          ) : (
            <InsightCard tone="good" title="Full coverage" desc="Every image has at least one detection." />
          )}
          {dist.length === 0 ? (
            <InsightCard tone="info" title="Label balance" desc="Add detections to see how your labels balance." />
          ) : under ? (
            <InsightCard tone="warn" title={`"${under.label}" is underrepresented`} desc={`Only ${under.count} so far; add more examples to balance training.`} />
          ) : (
            <InsightCard tone="good" title="Balanced labels" desc="Your labels are reasonably well represented." />
          )}
        </div>
      </div>

      {/* Dataset health — the focal panel: big score + radar + factor bars. */}
      <div>
        <SectionHeader action={<HeaderLink onClick={onOpenHealth}>Open health →</HeaderLink>}>Dataset health</SectionHeader>
        <button
          type="button"
          onClick={onOpenHealth}
          className="pk-card pk-card-hover w-full rounded-md p-4 text-left outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)] sm:p-5"
        >
          {typeof score !== "number" ? (
            <div className="flex items-center justify-center gap-2.5 py-12 text-sm text-foreground/50">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-foreground/20 border-t-[var(--accent)]" />
              Computing dataset health…
            </div>
          ) : (
            <div className="flex flex-col gap-7 sm:flex-row sm:items-center sm:gap-9">
              <div className="flex items-center gap-5">
                <FactorRadar values={factorVals} size={172} />
                <div>
                  <div className="text-4xl font-medium tracking-tight leading-none tabular-nums" style={{ color: factorColour((score ?? 0) / 100) }}>
                    {Math.round(score)}
                    <span className="ml-1 align-top text-sm font-medium text-foreground/40">/100</span>
                  </div>
                  <div className="pk-eyebrow mt-2" style={{ color: factorColour((score ?? 0) / 100) }}>{healthLabel}</div>
                </div>
              </div>
              <div className="grid flex-1 grid-cols-1 gap-x-8 gap-y-3.5 sm:grid-cols-2">
                {factorVals.map((f) => (
                  <div key={f.key}>
                    <div className="flex items-baseline justify-between gap-2 text-[13px]">
                      <span className="text-foreground/75">{FACTOR_INFO[f.key]?.label ?? f.key}</span>
                      <span className="tabular-nums font-medium" style={{ color: factorColour(f.value) }}>{Math.round(f.value * 100)}%</span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-foreground/[0.08]">
                      <div className="h-full rounded-full" style={{ width: `${f.value * 100}%`, background: factorColour(f.value) }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </button>
      </div>

      {/* References — optional on every dataset (hidden only for derived
          datasets, which don't manage their own). */}
      {showReferences && (
        <div>
          <SectionHeader count={refs.length} action={onOpenReferences ? <HeaderLink onClick={onOpenReferences}>View all →</HeaderLink> : undefined}>References</SectionHeader>
          <div className="pk-card rounded-md p-4">
            {refs.length === 0 ? (
              <p className="text-[13px] text-foreground/55">No reference images yet — optional, but a few clear examples per label improve label matching.</p>
            ) : (
              <div className="grid grid-cols-4 gap-2.5 sm:grid-cols-6 lg:grid-cols-10">
                {refs.slice(0, 10).map((r, i) => (
                  <button
                    key={r.referenceId ?? r.filename ?? `ref-${i}`}
                    type="button"
                    onClick={onOpenReferences}
                    className="group relative aspect-[4/3] overflow-hidden rounded-md bg-foreground/[0.06] outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]"
                    title={r.filename ?? undefined}
                  >
                    {r.blurhash && <BlurhashCanvas hash={r.blurhash} width={48} height={36} className="absolute inset-0 h-full w-full" />}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={r.preview} alt={r.filename ?? "reference"} loading="lazy" className="absolute inset-0 h-full w-full object-cover opacity-0 transition-[opacity,transform] duration-300 group-hover:scale-[1.04]" onLoad={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "1"; }} />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Composition — how the dataset breaks down across labels + visual variety. */}
      <div>
        <SectionHeader>Composition</SectionHeader>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Label distribution">
            {dist.length === 0 ? (
              <p className="m-auto text-[13px] text-foreground/55">No detections yet.</p>
            ) : (
              <div className="flex flex-1 items-center gap-5">
                <Donut slices={dist} />
                <ul className="min-w-0 flex-1 space-y-2">
                  {dist.slice(0, 6).map((s) => (
                    <li key={s.label} className="flex items-center gap-2.5 text-[13px]">
                      <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: s.colour }} />
                      <span className="min-w-0 flex-1 truncate text-foreground/85">{s.label}</span>
                      <span className="shrink-0 tabular-nums font-medium text-foreground/60">{s.count}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>

          <Card title="Image variation" action={typeof full?.counts?.embeddings_ready === "number" && (full?.counts?.embeddings_ready ?? 0) > 0 ? (
            <span className="text-[11px] text-foreground/45">{full?.counts?.embeddings_ready}/{full?.counts?.imports ?? images} embedded</span>
          ) : undefined}>
            <div className="mx-auto w-full max-w-[260px]">
              <VariationPlot
                points={variation?.points ?? []}
                augmentations={variation?.augmentations ?? []}
                colourFor={colourFor}
                nearDuplicateIds={new Set(variation?.near_duplicate_ids ?? [])}
                computing={(full?.counts?.imports ?? 0) > 0 && (full?.counts?.embeddings_ready ?? 0) < (full?.counts?.imports ?? 0)}
                totalImports={full?.counts?.imports ?? images}
                embeddingsReady={full?.counts?.embeddings_ready ?? 0}
              />
            </div>
          </Card>
        </div>
      </div>

      {/* Recent — newest images + the activity feed. */}
      <div>
        <SectionHeader action={onOpenDataset ? <HeaderLink onClick={onOpenDataset}>View all →</HeaderLink> : undefined}>Recent</SectionHeader>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <Card title="Images">
              {recent.length === 0 ? (
                <p className="m-auto text-[13px] text-foreground/55">No images yet.</p>
              ) : (
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
                  {recent.map((im) => (
                    <button key={im.id} type="button" onClick={() => onJumpToImport?.(im.id)} className="group block text-left outline-none" title={im.filename ?? undefined}>
                      <span className="relative block aspect-[4/3] overflow-hidden rounded-md bg-foreground/[0.06] ring-offset-2 ring-offset-[var(--background)] group-focus-visible:ring-2 group-focus-visible:ring-[var(--focus-ring)]">
                        {im.blurhash && <BlurhashCanvas hash={im.blurhash} width={48} height={36} className="absolute inset-0 h-full w-full" />}
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={im.preview} alt={im.filename ?? "image"} loading="lazy" className="absolute inset-0 h-full w-full object-cover opacity-0 transition-[opacity,transform] duration-300 group-hover:scale-[1.04]" onLoad={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "1"; }} />
                      </span>
                      <span className="mt-1.5 block truncate text-[11px] text-foreground/60">{im.filename ?? "image"}</span>
                    </button>
                  ))}
                </div>
              )}
            </Card>
          </div>
          <div className="lg:col-span-1">
            <Card title="Activity">
              {activity.length === 0 ? (
                <p className="m-auto text-[13px] text-foreground/55">Activity will appear here.</p>
              ) : (
                <ol className="relative ml-1 flex flex-col gap-3.5 border-l border-foreground/10 pl-4">
                  {activity.map((a) => (
                    <li key={a.id} className="relative text-[12.5px] leading-snug text-foreground/70">
                      <span
                        className="absolute -left-5 top-1 h-2 w-2 rounded-full bg-[var(--fg-faint)] ring-2 ring-[var(--surface)]"
                        aria-hidden
                      />
                      <button
                        type="button"
                        onClick={() => onJumpToImport?.(a.importId)}
                        className="flex w-full items-baseline justify-between gap-2 text-left outline-none focus-visible:underline"
                      >
                        <span className="min-w-0 flex-1 truncate">
                          <span className="font-medium text-foreground/90">{a.kind === "labelled" ? "Labelled " : "Added "}</span>
                          <span className="text-foreground/60">{a.name}</span>
                        </span>
                        <span className="shrink-0 whitespace-nowrap text-xs text-foreground/40">{relTime(a.at)}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
}
