"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { PixelKitLoader } from "./PixelKitLoader";

const LABEL_PHRASES = [
  "Finding objects…",
  "Detecting objects with SAM 3…",
  "Proposing candidate regions…",
  "Drawing bounding boxes…",
  "Scanning each image…",
  "Searching for your labels…",
  "Locating objects…",
  "Checking candidate regions…",
  "Filtering weak detections…",
  "Running detection…",
  "Outlining objects…",
  "Drawing mask outlines…",
  "Separating objects from the background…",
  "Drawing masks…",
  "Converting masks to polygons…",
  "Refining mask edges…",
  "Segmenting objects…",
  "Cleaning up masks…",
  "Assigning labels…",
  "Applying labels…",
  "Scoring detections…",
  "Matching detections to label names…",
  "Resolving overlapping boxes…",
  "Checking label assignments…",
  "Choosing the best label…",
  "Confirming labels…",
  "Naming each detection…",
  "Comparing detections across labels…",
  "Checking each label in turn…",
  "Finding similar objects…",
  "Comparing image embeddings…",
  "Measuring similarity…",
  "Processing images…",
  "Analysing images…",
  "Structuring annotations…",
  "Working through the images…",
  "Labelling images…",
  "Labelling the dataset…",
  "Checking confidence scores…",
  "Saving annotations…",
  "Finding object shapes…",
  "Sorting detections by label…",
  "Comparing detections…",
  "Removing duplicate boxes…",
  "Flagging borderline detections…",
  "Inspecting pixels…",
];

export type LabelJobState = {
  jobId: string;
  /** Mapped from the backend job status. */
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  index: number;
  total: number;
  /** Epoch ms when the job was first seen running, for the ETA. */
  startedAt?: number;
  /** Filename of the image being labelled; null between images. */
  currentImage?: string | null;
};

// "1m 23s", "12s".
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
  /** Called by the X button before it dismisses a running or queued job. */
  onCancel?: () => Promise<void> | void;
  /** Headlines for other job kinds; missing keys use the labelling text. */
  headlines?: Partial<{
    running: string;
    done: string;
    failed: string;
    cancelled: string;
  }>;
  /** Rotating status phrases; LABEL_PHRASES when unset. */
  phrases?: string[];
  doneMessage?: string;
}) {
  const phrasePool = phrases && phrases.length > 0 ? phrases : LABEL_PHRASES;
  const [msg, setMsg] = useState(() =>
    phrasePool[Math.floor(Math.random() * phrasePool.length)],
  );
  const [phraseVisible, setPhraseVisible] = useState(true);
  // The pool in a ref: a new phrases array each render must not reset the rotation.
  const phrasePoolRef = useRef(phrasePool);
  useEffect(() => {
    phrasePoolRef.current = phrasePool;
  }, [phrasePool]);
  const running = state?.status === "running";
  const done = state?.status === "done";
  useEffect(() => {
    if (!running) return;
    let mounted = true;
    const tick = () => {
      if (!mounted) return;
      setPhraseVisible(false);
      window.setTimeout(() => {
        if (!mounted) return;
        setMsg((cur) => {
          const pool = phrasePoolRef.current;
          let next = pool[Math.floor(Math.random() * pool.length)];
          if (next === cur) {
            next = pool[(pool.indexOf(next) + 1) % pool.length];
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
  }, [running]);

  const pct = useMemo(() => {
    if (!state || state.total <= 0) return null;
    return Math.min(100, Math.max(0, Math.round((state.index / state.total) * 100)));
  }, [state]);

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running]);
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

  // onClose in a ref: a new function each render must not reset the dismiss timer.
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);
  useEffect(() => {
    if (!done) return;
    const id = window.setTimeout(() => onCloseRef.current?.(), 2400);
    return () => window.clearTimeout(id);
  }, [done]);

  if (!state) return null;

  return (
    <div
      className="relative rounded-md border border-[var(--line)] bg-[var(--surface)]/95 backdrop-blur-md shadow-[var(--shadow-strong)] overflow-hidden animate-[fadeIn_320ms_ease-out]"
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
              ? "Something went wrong. Check the engine log."
              : msg}
          </p>
        </div>
        <button
          type="button"
          onClick={async () => {
            if (state.status === "running" || state.status === "queued") {
              if (onCancel) {
                try { await onCancel(); } catch { /* the card closes anyway */ }
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

      <div className="h-[3px] w-full bg-foreground/[0.06] relative overflow-hidden">
        {state.status === "running" && pct === null ? (
          <div className="indeterminate-bar absolute inset-y-0 w-1/3 bg-[var(--accent)] rounded-full" />
        ) : (
          <div
            className="h-full bg-[var(--accent)] transition-[width] duration-[400ms] ease-out"
            style={{ width: `${pct ?? 0}%` }}
          />
        )}
      </div>
    </div>
  );
}
