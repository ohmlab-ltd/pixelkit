"use client";

import { useEffect, useMemo, useState } from "react";

export type CascadeMember = {
  box_id: string;
  image: string;
  label: string;
  box_xyxy: number[];
  size_frac: number;
  similarity: number;
  is_pivot?: boolean;
};

export type CascadeGroup = {
  id: string;
  members: CascadeMember[];
  label_counts: Record<string, number>;
  label_diversity: number;
};

type Props = {
  apiBase: string;
  projectId: string;
  groups: CascadeGroup[];
  projectTags: string[];
  onClose: () => void;
  /** Mirror of the per-edit cascade callback, applies a new label
      across the listed (image, box_id) pairs in the parent's local
      `editedBoxes` state so the editor is in sync without waiting
      for the next manifest poll. */
  onApplied: (targets: { image: string; box_id: string }[], newLabel: string) => void;
};


// Label Cascade Review, paginates through the project-wide groups
// of visually-similar boxes returned by /embeddings/scan. For each
// group the user picks the label that should apply across all
// selected members (or skips the group). Behaves like a one-shot
// triage queue rather than a per-rename suggestion.
export function LabelCascadeReviewModal({
  apiBase,
  projectId,
  groups,
  projectTags,
  onClose,
  onApplied,
}: Props) {
  const [groupIdx, setGroupIdx] = useState(0);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [labelChoice, setLabelChoice] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const group = groups[groupIdx];

  const labelOptionsRaw = useMemo(() => {
    if (!group) return [] as string[];
    const seen = new Set<string>();
    const out: string[] = [];
    // Group's own labels first (most relevant).
    for (const m of group.members) {
      if (m.label && !seen.has(m.label)) {
        seen.add(m.label);
        out.push(m.label);
      }
    }
    // Then project tags so the user can also pick a label that
    // nobody in the group currently uses (catches the case where
    // the entire cluster is misclassified).
    for (const t of projectTags) {
      if (t && !seen.has(t)) {
        seen.add(t);
        out.push(t);
      }
    }
    return out;
  }, [group, projectTags]);

  // When we move to a new group, default-select every member and
  // pre-fill the label dropdown with the most common existing label
  // (the user's likely "majority wins" choice).
  useEffect(() => {
    if (!group) return;
    const sel: Record<string, boolean> = {};
    for (const m of group.members) sel[memberKey(m)] = true;
    setSelected(sel);
    const dominant = pickDominantLabel(group);
    setLabelChoice(dominant ?? group.members[0]?.label ?? "");
    setError(null);
  }, [group]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") goNext();
      if (e.key === "ArrowLeft") goPrev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupIdx, groups.length]);

  if (!group) {
    return null;
  }

  const goNext = () => {
    if (groupIdx < groups.length - 1) setGroupIdx((i) => i + 1);
  };
  const goPrev = () => {
    if (groupIdx > 0) setGroupIdx((i) => i - 1);
  };
  const skipGroup = () => {
    if (groupIdx < groups.length - 1) goNext();
    else onClose();
  };

  const apply = async () => {
    if (busy) return;
    const newLabel = labelChoice.trim();
    if (!newLabel) {
      setError("Pick a label first.");
      return;
    }
    const targets = group.members
      .filter((m) => selected[memberKey(m)] && m.label !== newLabel)
      .map((m) => ({ image: m.image, box_id: m.box_id }));
    if (targets.length === 0) {
      // Nothing to do, every selected member is already on the
      // chosen label. Treat as a skip.
      skipGroup();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(`${apiBase}/api/projects/${projectId}/embeddings/relabel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets, new_label: newLabel }),
      });
      if (!r.ok) {
        let msg = `http ${r.status}`;
        try {
          const body = await r.json();
          if (body && typeof body.detail === "string") msg = body.detail;
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      onApplied(targets, newLabel);
      if (groupIdx < groups.length - 1) goNext();
      else onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const dontAskAgain = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await fetch(`${apiBase}/api/projects/${projectId}/embeddings/ignore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: group.members.map((m) => ({ image: m.image, box_id: m.box_id })),
        }),
      });
      if (groupIdx < groups.length - 1) goNext();
      else onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const selectedCount = group.members.filter((m) => selected[memberKey(m)]).length;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center backdrop-blur-md bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-4xl rounded-2xl border border-foreground/10 bg-[var(--background)]/95 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        <header className="px-5 py-4 border-b border-foreground/[0.06] flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span
                className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300/40 bg-emerald-300/[0.08] text-emerald-100 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider"
                aria-hidden
              >
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18M6 12h12M9 18h6" />
                </svg>
                Label Cascade
              </span>
              <span className="text-xs text-foreground/45 font-mono tabular-nums">
                {groupIdx + 1} / {groups.length}
              </span>
            </div>
            <h2 className="text-base font-semibold text-[var(--foreground)] tracking-tight">
              {group.members.length} visually-similar boxes
            </h2>
            <p className="mt-1 text-xs text-foreground/55 leading-relaxed">
              {group.label_diversity > 1
                ? <>These look like the same kind of object but carry <span className="text-foreground/85">{group.label_diversity}</span> different labels. Pick one to apply across the group.</>
                : <>All currently labelled the same. Confirm or change to a new label.</>
              }
            </p>
            {Object.keys(group.label_counts).length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {Object.entries(group.label_counts)
                  .sort(([, a], [, b]) => b - a)
                  .map(([lab, count]) => (
                    <span
                      key={lab || "_blank"}
                      className="inline-flex items-center gap-1.5 rounded-full bg-foreground/[0.04] border border-foreground/10 px-2 py-0.5 text-[10px] font-mono text-foreground/70"
                    >
                      <span>{lab || "(no label)"}</span>
                      <span className="text-foreground/40">×{count}</span>
                    </span>
                  ))}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-foreground/40 hover:text-foreground text-2xl leading-none px-1 shrink-0"
          >
            ×
          </button>
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4">
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {group.members.map((m) => {
              const key = memberKey(m);
              const isOn = !!selected[key];
              return (
                <li key={key}>
                  <button
                    type="button"
                    onClick={() => setSelected((s) => ({ ...s, [key]: !isOn }))}
                    className={[
                      "w-full text-left rounded-xl border overflow-hidden transition-colors",
                      isOn
                        ? "border-emerald-400/50 bg-emerald-400/[0.07]"
                        : "border-foreground/10 hover:border-foreground/25 bg-foreground/[0.02]",
                    ].join(" ")}
                    aria-pressed={isOn}
                  >
                    <ImageWithBox
                      apiBase={apiBase}
                      projectId={projectId}
                      image={m.image}
                      box={m.box_xyxy}
                    />
                    <div className="flex items-center justify-between gap-2 px-3 py-2">
                      <div className="min-w-0">
                        <div className="text-xs font-mono text-foreground/85 truncate">
                          {m.label || <span className="text-foreground/40 italic">no label</span>}
                        </div>
                        <div className="text-[10px] text-foreground/40 truncate mt-0.5">
                          {m.image}
                          <span className="text-foreground/30">
                            {" · "}
                            {Math.round(m.similarity * 100)}%
                          </span>
                        </div>
                      </div>
                      <span
                        aria-hidden
                        className={[
                          "h-5 w-5 grid place-items-center rounded-full border shrink-0",
                          isOn
                            ? "bg-emerald-400/80 border-emerald-300/80 text-black"
                            : "border-foreground/25 text-transparent",
                        ].join(" ")}
                      >
                        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 8.5l3 3 7-7" />
                        </svg>
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
          {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
        </div>

        <footer className="px-5 py-4 border-t border-foreground/[0.06] flex flex-wrap items-center gap-2">
          <button
            onClick={dontAskAgain}
            disabled={busy}
            title="Lock these matches so Label Cascade never proposes them again."
            className="rounded-full border border-foreground/10 bg-transparent hover:bg-foreground/[0.04] hover:border-foreground/25 px-3 py-1.5 text-xs text-foreground/55 hover:text-foreground/85 transition-colors disabled:opacity-50"
          >
            Don&rsquo;t ask again
          </button>
          <div className="ml-auto flex items-center gap-2 flex-wrap">
            <button
              onClick={goPrev}
              disabled={busy || groupIdx === 0}
              className="rounded-full border border-foreground/15 bg-foreground/[0.03] hover:bg-foreground/[0.08] px-3 py-1.5 text-xs text-foreground/75 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              aria-label="Previous group"
            >
              ← Prev
            </button>
            <button
              onClick={skipGroup}
              disabled={busy}
              className="rounded-full border border-foreground/15 bg-foreground/[0.03] hover:bg-foreground/[0.08] px-3 py-1.5 text-xs text-foreground/75 transition-colors disabled:opacity-50"
            >
              Skip
            </button>
            <span className="hidden sm:inline-block w-px h-5 bg-foreground/10 mx-1" aria-hidden />
            <input
              type="text"
              value={labelChoice}
              onChange={(e) => setLabelChoice(e.target.value)}
              list={`cascade-labels-${group.id}`}
              placeholder="label"
              className="rounded-md border border-foreground/15 bg-foreground/[0.03] focus:border-foreground/40 outline-none px-2.5 py-1.5 text-xs font-mono text-[var(--foreground)] w-40"
            />
            <datalist id={`cascade-labels-${group.id}`}>
              {labelOptionsRaw.map((l) => (
                <option key={l} value={l} />
              ))}
            </datalist>
            <button
              onClick={apply}
              disabled={busy || !labelChoice.trim() || selectedCount === 0}
              className="rounded-full bg-foreground text-background px-4 py-1.5 text-xs font-medium hover:bg-zinc-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {busy ? "Updating…" : `Apply to ${selectedCount}`}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}


function memberKey(m: CascadeMember): string {
  return `${m.image}::${m.box_id}`;
}

function pickDominantLabel(group: CascadeGroup): string | null {
  let best: string | null = null;
  let bestCount = 0;
  for (const [lab, count] of Object.entries(group.label_counts)) {
    if (!lab) continue;
    if (count > bestCount) {
      best = lab;
      bestCount = count;
    }
  }
  return best;
}


function ImageWithBox({
  apiBase,
  projectId,
  image,
  box,
}: {
  apiBase: string;
  projectId: string;
  image: string;
  box: number[];
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [x0, y0, x1, y1] = box;
  return (
    <div className="relative aspect-video bg-[var(--background)] overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`${apiBase}/api/projects/${projectId}/originals/${encodeURIComponent(image)}`}
        alt=""
        loading="lazy"
        decoding="async"
        className="absolute inset-0 h-full w-full object-cover"
        onLoad={(e) => {
          const img = e.currentTarget;
          setSize({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
      {size && (
        <div
          className="absolute pointer-events-none border-2 border-emerald-300/90 shadow-[0_0_14px_rgba(110,231,183,0.45)] rounded-[2px]"
          style={{
            left: `${(x0 / size.w) * 100}%`,
            top: `${(y0 / size.h) * 100}%`,
            width: `${((x1 - x0) / size.w) * 100}%`,
            height: `${((y1 - y0) / size.h) * 100}%`,
          }}
        />
      )}
    </div>
  );
}
