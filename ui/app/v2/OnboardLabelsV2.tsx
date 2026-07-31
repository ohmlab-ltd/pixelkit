"use client";

import { useEffect, useState } from "react";

import { containsProfanity } from "../profanity";

// Deterministic palette so each label keeps the same colour across
// the onboarding flow + the editor. Same family used by the project
// card chips in V1 so the visual identity carries through.
export const LABEL_COLOURS = [
  "#fb923c", // orange-400
  "#60a5fa", // blue-400
  "#a78bfa", // violet-400
  "#34d399", // emerald-400
  "#f472b6", // pink-400
  "#facc15", // yellow-400
  "#2dd4bf", // teal-400
  "#f87171", // red-400
];

export function colourForLabel(label: string, all: string[]): string {
  const idx = all.indexOf(label);
  return LABEL_COLOURS[(idx >= 0 ? idx : Math.abs(_hash(label))) % LABEL_COLOURS.length];
}

// Colour for a label that's STABLE across views, the same label
// string in the SAME project always maps to the same colour
// regardless of which surface renders it (workspace card, public
// feed, project page). Hashes the canonical label name, not the
// displayed alias, so a rename doesn't reshuffle existing chips.
export function colourForLabelStable(label: string): string {
  const key = (label || "").trim().toLowerCase();
  return LABEL_COLOURS[Math.abs(_hash(key)) % LABEL_COLOURS.length];
}

// Per-project palette assignment with collision resolution. Two
// labels in the same project that would otherwise hash to the same
// (or visually-adjacent) palette slot get pushed to the slot that
// maximises distance from every already-taken slot, so no two
// chips in a single card look the same colour. Same project +
// same canonical label set yields the same map on every render,
// which is what keeps the colour consistent across the workspace
// card, public-feed card, and the project page.
//
// Assignment order is INSERTION order (the project's tags array)
// so adding a new label appends to whatever's free without
// reshuffling existing ones.
export function buildProjectLabelColourMap(
  allLabels: string[],
  /** Per-label colour overrides (canonical-lower label → #rrggbb).
      When set for a label, the override wins over the hash-derived
      palette slot and the slot stays free for another label. */
  overrides?: Record<string, string> | null,
): Map<string, string> {
  const N = LABEL_COLOURS.length;
  const taken = new Set<number>();
  const out = new Map<string, string>();
  // Apply explicit overrides first so they reserve no palette slot ,
  // a user-picked colour shouldn't crowd out another label's hash
  // assignment. Slot-tracking only ever covers the auto-assigned
  // labels below.
  const ov = overrides ? new Map<string, string>(
    Object.entries(overrides).map(([k, v]) => [k.trim().toLowerCase(), v]),
  ) : null;
  for (const raw of allLabels) {
    const key = (raw || "").trim().toLowerCase();
    if (!key || out.has(key)) continue;
    if (ov && ov.has(key)) {
      out.set(key, ov.get(key)!);
      continue;
    }
    const pref = Math.abs(_hash(key)) % N;
    let slot = pref;
    if (taken.has(slot)) {
      // Pick the free slot whose minimum circular distance to any
      // already-taken slot is largest. Falls through to a linear
      // probe if everything's somehow tied.
      let bestSlot = -1;
      let bestMinDist = -1;
      for (let s = 0; s < N; s++) {
        if (taken.has(s)) continue;
        let minDist = N;
        for (const t of taken) {
          const raw_d = Math.abs(s - t);
          const d = Math.min(raw_d, N - raw_d);
          if (d < minDist) minDist = d;
        }
        if (minDist > bestMinDist) {
          bestMinDist = minDist;
          bestSlot = s;
        }
      }
      if (bestSlot >= 0) slot = bestSlot;
      else {
        // Palette exhausted (more labels than colours), wrap with a
        // linear probe so we at least avoid an exact-slot collision.
        let s = (pref + 1) % N;
        while (taken.has(s) && s !== pref) s = (s + 1) % N;
        slot = s;
      }
    }
    taken.add(slot);
    out.set(key, LABEL_COLOURS[slot]);
  }
  return out;
}

function _hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

// Pick a readable text colour for a chip given its background hex.
// YIQ luminance is good enough for the small palette + the user's
// custom picks, keeps the call cheap on every render. Threshold
// of 150 keeps mid-orange/yellow on black text and pushes deep
// reds / navy / black-ish picks onto white.
export function readableTextForBg(hex: string | null | undefined): string {
  if (!hex || typeof hex !== "string") return "#000";
  let s = hex.trim().replace(/^#/, "");
  if (s.length === 3) s = s.split("").map((c) => c + c).join("");
  if (s.length !== 6) return "#000";
  const r = parseInt(s.slice(0, 2), 16);
  const g = parseInt(s.slice(2, 4), 16);
  const b = parseInt(s.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return "#000";
  const yiq = (r * 299 + g * 587 + b * 114) / 1000;
  return yiq >= 150 ? "#000" : "#fff";
}

// Stage A of the V2 create flow.
//
//   ┌────────────────────────────────────────────┐
//   │  What do you want to detect?               │
//   │  Add a list of labels for the objects.     │
//   │  You can add more later.                   │
//   │                                            │
//   │  [Project name…]                           │
//   │  [type a label  ↵]   [dog] [cat] [car]     │
//   │                                            │
//   │                       [Skip]   [ Done ]    │
//   └────────────────────────────────────────────┘
//
// Skip → proceed with no labels. Done → proceed with the labels list.
// The orchestrator (CreateFlowV2) animates this whole panel up + the
// labels slide into the header position when the user clicks Done.
export function OnboardLabelsV2({
  onDone,
  onSkip,
  onClose,
}: {
  /** Returns the user's project name, the labels list, and the random
      colour assignment chosen during onboarding so the project's UI
      keeps the exact swatches the user previewed. Colour map keys are
      canonical (trimmed + lowercased) so the backend can re-key
      against tags directly. */
  onDone: (
    name: string,
    labels: string[],
    labelColours: Record<string, string>,
  ) => void;
  onSkip: (name: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [labels, setLabels] = useState<string[]>([]);
  // Per-label random palette pick made the moment a label is added.
  // Picks from LABEL_COLOURS without replacement until exhausted, then
  // allows reuse, same colour can never collide with the previous N-1
  // labels for a palette of size N.
  const [labelColours, setLabelColours] = useState<Record<string, string>>({});
  const [input, setInput] = useState("");
  const [mounted, setMounted] = useState(false);

  // Mount-in animation, same easing the home page uses for project
  // cards so the entrance feels consistent.
  useEffect(() => {
    const t = window.setTimeout(() => setMounted(true), 20);
    return () => window.clearTimeout(t);
  }, []);

  // Esc closes the overlay (cancel), only when not actively typing
  // a label. Without this guard, a half-typed chip is lost on stray
  // Escape presses.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !input) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [input, onClose]);

  // Sample a colour that isn't currently in use by any label still
  // in the list. Falls back to a fully random pick once the palette
  // is saturated so the chip never goes blank.
  const sampleColour = (taken: Set<string>): string => {
    const free = LABEL_COLOURS.filter((c) => !taken.has(c));
    const pool = free.length > 0 ? free : LABEL_COLOURS;
    return pool[Math.floor(Math.random() * pool.length)];
  };

  const [labelError, setLabelError] = useState<string | null>(null);
  const addLabel = (raw?: string) => {
    const v = (raw ?? input).trim().toLowerCase();
    if (!v) return;
    if (labels.includes(v)) {
      setInput("");
      return;
    }
    // Client-side profanity gate so the user can't sneak a blocked
    // term past the backend's assert_clean. Explicit message ,
    // "can't be used as a label" alone left users wondering why it
    // was rejected.
    if (containsProfanity(v)) {
      setLabelError(`"${v}" is blocked by the profanity filter and can't be used as a label.`);
      return;
    }
    setLabelError(null);
    const taken = new Set(Object.values(labelColours));
    const colour = sampleColour(taken);
    setLabels([...labels, v]);
    setLabelColours((m) => ({ ...m, [v]: colour }));
    setInput("");
  };

  const removeLabel = (lab: string) => {
    setLabels(labels.filter((l) => l !== lab));
    setLabelColours((m) => {
      const next = { ...m };
      delete next[lab];
      return next;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === "," || e.key === ".") {
      e.preventDefault();
      addLabel();
    } else if (e.key === "Backspace" && !input && labels.length) {
      // Pop the most recent chip when backspacing over an empty input.
      // Same UX as Gmail recipient chips.
      setLabels(labels.slice(0, -1));
    }
  };

  const canSubmit = name.trim().length > 0;
  const allLabels = labels;

  return (
    <div
      className="pk-backdrop fixed inset-0 z-[300] overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-label="Create project, what do you want to detect"
    >
      <div className="min-h-full grid place-items-center p-6">
        <div
          className={[
            "w-full max-w-2xl transition-all duration-700 ease-out",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6",
          ].join(" ")}
        >
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight text-[var(--foreground)]">
            What do you want to detect?
          </h1>
          <p className="mt-3 text-base text-foreground/60 leading-relaxed">
            Add a list of labels for the objects you want to detect. You can add
            more later.
          </p>

          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Project name"
            autoFocus
            className="mt-10 w-full bg-transparent border-b border-[var(--line)] py-3 text-xl text-[var(--foreground)] placeholder:text-foreground/25 transition-colors"
          />

          <div className="mt-8">
            <span className="text-[11px] uppercase tracking-wider text-foreground/40 font-mono">
              Labels
            </span>
            <div
              className="mt-2 min-h-[44px] flex flex-wrap items-center gap-2 border-b border-[var(--line)] focus-within:border-[var(--accent)] transition-colors py-2"
              onClick={(e) => {
                // Click anywhere in the chip strip focuses the input ,
                // a pattern lifted from Gmail / Notion tag fields.
                const target = e.currentTarget.querySelector("input");
                if (target) (target as HTMLInputElement).focus();
              }}
            >
              {allLabels.map((lab, i) => {
                const bg = labelColours[lab] ?? LABEL_COLOURS[i % LABEL_COLOURS.length];
                return (
                  <span
                    key={lab}
                    // Neutral chip: hairline border, transparent surface,
                    // the label's colour carried by the swatch dot only.
                    className="inline-flex h-6 items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent px-2 font-mono text-[12px] text-[var(--fg-soft)] animate-[fadeIn_180ms_ease-out]"
                  >
                    <span
                      aria-hidden
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: bg }}
                    />
                    {lab}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeLabel(lab);
                      }}
                      aria-label={`Remove ${lab}`}
                      className="inline-flex h-4 w-4 shrink-0 items-center justify-center text-[var(--fg-dim)] transition-colors hover:text-[var(--bad)]"
                    >
                      <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden>
                        <path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                    </button>
                  </span>
                );
              })}
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onBlur={() => input && addLabel()}
                placeholder={
                  allLabels.length === 0 ? "Type a label, press Enter" : "Add another…"
                }
                className="flex-1 min-w-[10rem] bg-transparent py-1 text-base text-[var(--foreground)] placeholder:text-foreground/30"
              />
            </div>
            {labelError && (
              <p className="mt-2 text-[11px] text-red-700 dark:text-red-300">
                {labelError}
              </p>
            )}
            {allLabels.length > 0 && !labelError && (
              <p className="mt-2 text-[11px] text-foreground/35">
                Press Enter, comma, or full stop to add a label. Backspace
                deletes the most recent.
              </p>
            )}
          </div>

          <div className="mt-12 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={() => canSubmit && onSkip(name.trim())}
              disabled={!canSubmit}
              className="px-5 py-2.5 text-sm text-foreground/55 hover:text-foreground/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Skip
            </button>
            <button
              type="button"
              onClick={() => canSubmit && onDone(name.trim(), allLabels, labelColours)}
              disabled={!canSubmit || allLabels.length === 0}
              className="pk-btn pk-btn-primary"
            >
              Done
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(2px) scale(0.96); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </div>
  );
}
