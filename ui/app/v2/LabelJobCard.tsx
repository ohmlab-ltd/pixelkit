"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { PixelKitLoader } from "./PixelKitLoader";

// V1's whimsical labelling-status phrases, copy-pasted verbatim from
// app/ProjectView.tsx. Same vibe, V2 styling.
const LABEL_PHRASES = [
  "Scanning for suspiciously object-shaped pixels…",
  "Asking the model “what vibes do you detect?”",
  "Proposing candidate regions like a very confident intern…",
  "Generating bounding boxes with questionable confidence…",
  "Looking for things that might be things…",
  "Detecting objects (or at least trying really hard to)",
  "Guessing where stuff might be hiding…",
  "Turning pixels into hypotheses…",
  "Finding blobs that could legally count as objects…",
  "Running zero-shot detection (and hoping for the best)",
  "Putting borders on things…",
  "Drawing lines where the model feels emotionally confident…",
  "Separating object from not-object…",
  "Carving reality into neat little masks…",
  "Turning blobs into shapes…",
  "Refining edges like a perfectionist…",
  "Segmenting with surgical optimism…",
  "Making masks that almost make sense…",
  "Naming things with varying degrees of confidence…",
  "Applying labels and mild overconfidence…",
  "Making educated guesses about reality…",
  "Translating pixels into words and words into pixels…",
  "Arguing internally…",
  "Assigning labels that seem defensible…",
  "Performing semantic guesswork…",
  "Choosing labels and standing by them (for now)",
  "Converting shapes into nouns…",
  "Comparing things to other things we’ve seen…",
  "Looking for familiar-looking objects…",
  "Finding visually similar troublemakers…",
  "Finding objects that pass the “same-ish” test…",
  "Recalling past visual experiences…",
  "Doing computer vision things…",
  "Making sense of chaos…",
  "Turning images into structured optimism…",
  "Working through pixels one decision at a time…",
  "Progress is happening, we promise…",
  "Applying questionable intelligence at scale…",
  "Optimising reality…",
  "Crunching pixels responsibly…",
  "Finding shapes…",
  "Figuring out what’s what…",
  "Comparing things to things…",
  "Fixing similar ones…",
  "Making better guesses…",
  "Inspecting pixels with cautious optimism…",
];

export type LabelJobState = {
  jobId: string;
  /** running|done|failed|cancelled, mapped from backend status */
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  index: number;
  total: number;
  /** Epoch ms, when the user first saw the job switch to running.
      Used purely for the FE ETA estimate; backend's startedAt is the
      authoritative timestamp but we don't need it for display. */
  startedAt?: number;
  /** Filename of the image currently being labelled (set by the
      runner's pre-emit, cleared between images). The dataset gallery
      paints a "Labelling…" overlay on the matching tile. */
  currentImage?: string | null;
};

// "1m 23s" / "12s", kept compact so the ETA fits in the card header.
function formatEtaSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "almost done";
  const s = Math.max(1, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mr = m % 60;
  return mr ? `${h}h ${mr}m` : `${h}h`;
}

export function LabelJobCard({
  state,
  onClose,
  onCancel,
  headlines,
  phrases,
  doneMessage,
}: {
  state: LabelJobState | null;
  onClose: () => void;
  /** Optional: when supplied, the X button on a still-running job
      calls this BEFORE dismissing so the parent can hit the
      backend's cancel endpoint. Falls back to a plain dismiss
      when the job is already in a terminal state. */
  onCancel?: () => Promise<void> | void;
  /** Optional copy override so callers can repurpose the same
      card chrome for a different job kind (e.g. augmentation
      generation). Missing keys fall back to the labelling copy. */
  headlines?: Partial<{
    running: string;
    done: string;
    failed: string;
    cancelled: string;
  }>;
  /** Optional phrase rotator pool. Defaults to the labelling
      vibe phrases above. */
  phrases?: string[];
  /** Optional copy shown when the job's status hits "done". */
  doneMessage?: string;
}) {
  const phrasePool = phrases && phrases.length > 0 ? phrases : LABEL_PHRASES;
  // Phrase rotator, same cadence as V1 (~3.2 s) with a brief opacity
  // dip between swaps so the change reads as intentional rather than
  // a flicker.
  const [msg, setMsg] = useState(() =>
    phrasePool[Math.floor(Math.random() * phrasePool.length)],
  );
  const [phraseVisible, setPhraseVisible] = useState(true);
  useEffect(() => {
    if (!state || state.status !== "running") return;
    let mounted = true;
    const tick = () => {
      if (!mounted) return;
      setPhraseVisible(false);
      window.setTimeout(() => {
        if (!mounted) return;
        setMsg((cur) => {
          let next = phrasePool[Math.floor(Math.random() * phrasePool.length)];
          if (next === cur) {
            next = phrasePool[
              (phrasePool.indexOf(next) + 1) % phrasePool.length
            ];
          }
          return next;
        });
        setPhraseVisible(true);
      }, 220);
    };
    const id = window.setInterval(tick, 3200);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, [state?.status]);

  // Progress percentage. Defaults to indeterminate at 0/0 (job just
  // created, no progress events yet) so the bar doesn't collapse
  // visually.
  const pct = useMemo(() => {
    if (!state || state.total <= 0) return null;
    return Math.min(100, Math.max(0, Math.round((state.index / state.total) * 100)));
  }, [state]);

  // ETA, derived from how long we've been running and how many
  // images we've completed. Recomputes on a 1 s tick so the countdown
  // doesn't sit stale between progress events.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!state || state.status !== "running") return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [state?.status]);
  const eta = useMemo(() => {
    if (!state || state.status !== "running") return null;
    if (!state.startedAt || state.index <= 0 || state.total <= 0) return null;
    const elapsed = (now - state.startedAt) / 1000;
    if (elapsed < 1) return null;
    const perImage = elapsed / state.index;
    const remaining = (state.total - state.index) * perImage;
    if (!Number.isFinite(remaining) || remaining < 0) return null;
    return formatEtaSeconds(remaining);
  }, [state, now]);

  // Auto-dismiss after success, keep the success state visible just
  // long enough to read, then fade out via the parent's onClose.
  // onClose is stashed in a ref so a fresh function reference on each
  // render doesn't tear down the timer before it fires.
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);
  useEffect(() => {
    if (!state) return;
    if (state.status !== "done") return;
    const id = window.setTimeout(() => onCloseRef.current?.(), 2400);
    return () => window.clearTimeout(id);
  }, [state?.status]);

  if (!state) return null;

  return (
    <div
      // Animated spawn-in: scale + fade + slide. The keyframe lands
      // on `fadeIn` (already in globals.css) for opacity and on a
      // bespoke transform for the lift.
      // Themable surface + soft shadow so the card matches the
      // other rounded cards on the page. The previous heavy black
      // shadow read as a sharp grey rectangle in light mode.
      className="relative rounded-2xl border border-foreground/10 bg-[var(--surface)]/95 backdrop-blur-md shadow-[var(--shadow-strong)] overflow-hidden animate-[fadeIn_320ms_ease-out]"
      style={{ animation: "labelJobIn 360ms cubic-bezier(0.2,0.7,0.2,1) both" }}
    >
      <style>{`
        @keyframes labelJobIn {
          0%   { opacity: 0; transform: translateY(8px) scale(0.985); }
          100% { opacity: 1; transform: translateY(0)   scale(1);     }
        }
      `}</style>

      <div className="px-5 py-4 flex items-center gap-4">
        <div className="shrink-0">
          <PixelKitLoader size={56} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="text-[15px] font-medium text-[var(--foreground)]">
              {state.status === "done"
                ? (headlines?.done ?? "Labelling complete")
                : state.status === "failed"
                ? (headlines?.failed ?? "Labelling failed")
                : state.status === "cancelled"
                ? (headlines?.cancelled ?? "Labelling cancelled")
                : (headlines?.running ?? "Labelling images")}
            </h3>
            <span className="text-[11px] text-foreground/45 font-mono tabular-nums">
              {state.total > 0
                ? `${state.index} / ${state.total}${eta ? ` · ${eta} left` : ""}`
                : "starting…"}
            </span>
          </div>
          <p
            className={[
              "mt-1 text-[12.5px] text-foreground/55 leading-snug min-h-[1.2em] transition-opacity duration-200",
              phraseVisible ? "opacity-100" : "opacity-0",
            ].join(" ")}
          >
            {state.status === "done"
              ? (doneMessage ?? "All images labelled. Refreshing the gallery…")
              : state.status === "failed"
              ? "Something went wrong, check the backend logs."
              : msg}
          </p>
        </div>
        {/* X button. While the job is running it doubles as a
            cancel, calls onCancel (which hits the backend) before
            dismissing. After the job is in a terminal state it's a
            plain dismiss. */}
        <button
          type="button"
          onClick={async () => {
            if (state.status === "running" || state.status === "queued") {
              if (onCancel) {
                try { await onCancel(); } catch { /* ignore, UI still dismisses */ }
              }
            }
            onClose();
          }}
          className="shrink-0 text-foreground/45 hover:text-foreground text-xl leading-none px-2"
          aria-label={
            state.status === "running" || state.status === "queued" ? "Cancel job" : "Close"
          }
          title={
            state.status === "running" || state.status === "queued" ? "Cancel job" : "Close"
          }
        >
          ×
        </button>
      </div>

      {/* Progress bar, solid fill for determinate progress, an
          animated indeterminate sweep when total isn't known yet. */}
      <div className="h-[3px] w-full bg-foreground/[0.06] relative overflow-hidden">
        {state.status === "running" && pct === null ? (
          <div className="indeterminate-bar absolute inset-y-0 w-1/3 bg-foreground/55 rounded-full" />
        ) : (
          <div
            className="h-full bg-gradient-to-r from-orange-400 to-orange-500 transition-[width] duration-400 ease-out"
            style={{ width: `${pct ?? 0}%` }}
          />
        )}
      </div>
    </div>
  );
}
