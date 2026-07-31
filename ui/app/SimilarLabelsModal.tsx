"use client";

import { useEffect, useMemo, useState } from "react";

export type SimilarMatch = {
  box_id: string;
  image: string;
  label: string;
  box_xyxy: number[];
  similarity: number;
  // present in the metadata but unused by the modal:
  id?: string;
  poly_count?: number;
  updated_at?: string;
};

type Props = {
  apiBase: string;
  projectId: string;
  oldLabel: string;
  newLabel: string;
  matches: SimilarMatch[];
  onClose: () => void;
  /** Parent updates its own boxes state with the relabelled set so
      the editor reflects the change without a full reload. The
      second argument is the new label that was applied, saves the
      parent from having to re-derive it. */
  onRelabelled: (changed: { image: string; box_id: string }[], newLabel: string) => void;
};

// Two-stage modal: a compact prompt with Ignore / Review buttons,
// then an expanded grid showing every similar candidate. The grid
// is animated in via a CSS height transition so the expansion
// reads as a continuation of the same surface, not a second popup.
export function SimilarLabelsModal({
  apiBase,
  projectId,
  oldLabel,
  newLabel,
  matches,
  onClose,
  onRelabelled,
}: Props) {
  const [reviewing, setReviewing] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dontAskAgain = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      // Lock every candidate the modal proposed, they'll never be
      // surfaced again as a Label Cascade match (regardless of
      // which trigger box prompted the search).
      await fetch(`${apiBase}/api/projects/${projectId}/embeddings/ignore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: matches.map((m) => ({ image: m.image, box_id: m.box_id })),
        }),
      });
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Default everything to selected when the user enters review, the
  // expected case is "yes, fix all of them"; deselecting a single
  // mistake is one click instead of selecting every match individually.
  useEffect(() => {
    if (reviewing) {
      const next: Record<string, boolean> = {};
      for (const m of matches) next[matchKey(m)] = true;
      setSelected(next);
    }
  }, [reviewing, matches]);

  // Esc closes the dialog. Unlike the verdict modal we don't gate
  // this since the action here is always a discardable suggestion.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const selectedCount = useMemo(
    () => matches.filter((m) => selected[matchKey(m)]).length,
    [matches, selected],
  );

  const apply = async () => {
    if (busy) return;
    const targets = matches
      .filter((m) => selected[matchKey(m)])
      .map((m) => ({ image: m.image, box_id: m.box_id }));
    if (targets.length === 0) {
      onClose();
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
      onRelabelled(targets, newLabel);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center backdrop-blur-md bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div
        className={[
          "w-full max-w-3xl rounded-2xl border border-foreground/10 bg-[var(--background)]/95 shadow-2xl overflow-hidden",
          "transition-all duration-300 ease-out",
        ].join(" ")}
      >
        <header className="px-5 py-4 border-b border-foreground/[0.06] flex items-start justify-between gap-3">
          <div>
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
            </div>
            <h2 className="text-base font-semibold text-[var(--foreground)] tracking-tight">
              {matches.length} match{matches.length === 1 ? "" : "es"} found
            </h2>
            <p className="mt-1 text-xs text-foreground/55 leading-relaxed">
              Other regions look visually similar to the one you just renamed.
              They&rsquo;re currently labelled
              <span className="font-mono text-foreground/85"> {truncate(oldLabel, 24)}</span>{" "}
              and might also belong to
              <span className="font-mono text-foreground/85"> {truncate(newLabel, 24)}</span>.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-foreground/40 hover:text-foreground text-2xl leading-none px-1"
          >
            ×
          </button>
        </header>

        <div
          className="grid transition-[grid-template-rows] duration-300 ease-out"
          style={{ gridTemplateRows: reviewing ? "1fr" : "0fr" }}
        >
          <div className="overflow-hidden">
            {reviewing && (
              <div className="px-5 py-4 max-h-[55vh] overflow-y-auto">
                <ul className="grid gap-3 sm:grid-cols-2">
                  {matches.map((m) => {
                    const key = matchKey(m);
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
                                {m.image}
                              </div>
                              <div className="text-[10px] text-foreground/50 mt-0.5">
                                Currently{" "}
                                <span className="font-mono text-foreground/75">
                                  {truncate(m.label, 20)}
                                </span>
                                {" · "}
                                <span className="text-foreground/40 tabular-nums">
                                  {Math.round(m.similarity * 100)}% match
                                </span>
                              </div>
                            </div>
                            <span
                              aria-hidden
                              className={[
                                "h-5 w-5 grid place-items-center rounded-full border",
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
                {error && (
                  <p className="mt-3 text-xs text-red-400">{error}</p>
                )}
              </div>
            )}
          </div>
        </div>

        <footer className="px-5 py-4 border-t border-foreground/[0.06] flex flex-wrap items-center justify-end gap-2">
          {!reviewing ? (
            <>
              <button
                onClick={dontAskAgain}
                disabled={busy}
                title="Lock these matches so Label Cascade never proposes them again."
                className="mr-auto rounded-full border border-foreground/10 bg-transparent hover:bg-foreground/[0.04] hover:border-foreground/25 px-3 py-1.5 text-xs text-foreground/55 hover:text-foreground/85 transition-colors disabled:opacity-50"
              >
                Don&rsquo;t ask again
              </button>
              <button
                onClick={onClose}
                disabled={busy}
                className="rounded-full border border-foreground/15 bg-foreground/[0.03] hover:bg-foreground/[0.08] hover:border-foreground/30 px-4 py-1.5 text-sm text-foreground/85 transition-colors disabled:opacity-50"
              >
                Ignore
              </button>
              <button
                onClick={() => setReviewing(true)}
                disabled={busy}
                className="rounded-full bg-foreground text-background px-4 py-1.5 text-sm font-medium hover:bg-zinc-200 transition-colors disabled:opacity-50"
              >
                Review
              </button>
            </>
          ) : (
            <>
              <span className="text-xs text-foreground/55 mr-auto">
                {selectedCount} of {matches.length} selected
              </span>
              <button
                onClick={dontAskAgain}
                disabled={busy}
                title="Lock these matches so Label Cascade never proposes them again."
                className="rounded-full border border-foreground/10 bg-transparent hover:bg-foreground/[0.04] hover:border-foreground/25 px-3 py-1.5 text-xs text-foreground/55 hover:text-foreground/85 transition-colors disabled:opacity-50"
              >
                Don&rsquo;t ask again
              </button>
              <button
                onClick={onClose}
                disabled={busy}
                className="rounded-full border border-foreground/15 bg-foreground/[0.03] hover:bg-foreground/[0.08] hover:border-foreground/30 px-4 py-1.5 text-sm text-foreground/85 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={apply}
                disabled={busy || selectedCount === 0}
                className="rounded-full bg-foreground text-background px-4 py-1.5 text-sm font-medium hover:bg-zinc-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {busy ? "Updating…" : `Relabel ${selectedCount} as "${truncate(newLabel, 16)}"`}
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}


function matchKey(m: SimilarMatch): string {
  return `${m.image}::${m.box_id}`;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}


// Renders the original image with a single bounding box drawn over
// it, scaled to fit a 16:9 thumbnail. We don't know the image
// dimensions upfront so we let the <img> load naturally and apply
// the box overlay as percentage-based positioning once we know
// the natural width/height.
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
