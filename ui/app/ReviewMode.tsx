"use client";

import { useEffect, useState } from "react";
import { BoxEditor, type EditableBox, type MaskShape } from "./BoxEditor";

type Verdict = "good" | "bad";
export type ReviewScope = "unrated" | "vlm" | "good" | "bad" | "all";

type ResultLite = {
  image: string;
  size: { width: number; height: number };
};

type Props = {
  results: ResultLite[];
  boxesByImage: Record<string, EditableBox[]>;
  jobId: string; // project name
  apiBase: string;
  /** kept for compatibility; no-op */
  urlMode?: "projects" | "jobs";
  verdicts: Record<string, Verdict>;
  onVerdict: (image: string, verdict: Verdict) => void;
  onClose: () => void;
  /** What slice of results we're walking, only used for the header label. */
  scope?: ReviewScope;
  /** Edits made on the top card flow back through here. */
  onBoxesChange?: (image: string, next: EditableBox[]) => void;
  /** Segmentation / classification callbacks forwarded to the embedded editor. */
  onSegmentBox?: (image: string, box: EditableBox) => Promise<MaskShape | null>;
  onClassifyBox?: (image: string, box: EditableBox) => Promise<{ label: string | null; score: number | null } | null>;
  onPointDetect?: (image: string, point: { x: number; y: number }) => Promise<{
    box_xyxy: number[];
    mask?: MaskShape | null;
    label?: string | null;
    score?: number | null;
  } | null>;
  projectTags?: string[];
  /** When false, Escape no longer closes the modal, used for the
   *  forced preference-model review where the user must commit a
   *  Skip / Finish action instead of dismissing the gate. */
  escapeClosable?: boolean;
};

const SCOPE_LABEL: Record<ReviewScope, string> = {
  unrated: "Unrated",
  vlm: "AI-rejected",
  good: "Marked good",
  bad: "Marked bad",
  all: "All images",
};

const STACK_DEPTH = 3; // top + 2 behind
const PRELOAD_AHEAD = 3; // images beyond the visible deck to warm in cache

export function ReviewMode({
  results,
  boxesByImage,
  jobId,
  apiBase,
  verdicts,
  onVerdict,
  onClose,
  scope = "unrated",
  onBoxesChange,
  onSegmentBox,
  onClassifyBox,
  onPointDetect,
  projectTags = [],
  escapeClosable = true,
}: Props) {
  // For "unrated" we still skip ahead to the first un-judged image; for any
  // other scope we walk the list as-given so the user can re-review every
  // entry without it auto-skipping.
  const firstUnrated = results.findIndex((r) => !verdicts[r.image]);
  const startIdx = scope === "unrated" ? (firstUnrated === -1 ? 0 : firstUnrated) : 0;
  const [idx, setIdx] = useState(startIdx);
  const [animating, setAnimating] = useState<"left" | "right" | null>(null);

  const current = results[idx];
  const isDone = idx >= results.length;

  const classify = (verdict: Verdict) => {
    if (!current || animating) return;
    setAnimating(verdict === "bad" ? "left" : "right");
    window.setTimeout(() => {
      onVerdict(current.image, verdict);
      setIdx((i) => i + 1);
      setAnimating(null);
    }, 260);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack arrow keys while the user is typing into the editor's
      // label picker (or any other input), they're cursor moves there.
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      const editing = tag === "input" || tag === "textarea" || (document.activeElement as HTMLElement | null)?.isContentEditable;
      if (e.key === "ArrowLeft" && !editing) {
        e.preventDefault();
        classify("bad");
      } else if (e.key === "ArrowRight" && !editing) {
        e.preventDefault();
        classify("good");
      } else if (e.key === "Escape" && !editing && escapeClosable) {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animating, current?.image, isDone, escapeClosable]);

  if (isDone) {
    const goods = Object.values(verdicts).filter((v) => v === "good").length;
    const bads = Object.values(verdicts).filter((v) => v === "bad").length;
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-md bg-black/70 p-6">
        <div className="bg-[var(--background)]/95 rounded-2xl border border-[var(--border)] p-10 text-center max-w-md">
          <h2 className="text-2xl font-semibold">Review complete</h2>
          <p className="mt-3 text-[var(--muted)]">
            <span className="text-green-400 font-mono">{goods}</span> good ·{" "}
            <span className="text-red-400 font-mono">{bads}</span> bad
          </p>
          <div className="mt-8 flex justify-center">
            <button
              onClick={onClose}
              className="rounded-full bg-foreground text-background px-5 py-2.5 text-sm hover:bg-zinc-200"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  const visible = results.slice(idx, Math.min(idx + STACK_DEPTH, results.length));
  // Images just beyond the visible deck, eagerly fetched and decoded so
  // they don't pop in when the stack advances.
  const preload = results.slice(
    idx + STACK_DEPTH,
    Math.min(idx + STACK_DEPTH + PRELOAD_AHEAD, results.length),
  );

  return (
    <div
      className="fixed inset-0 z-50 backdrop-blur-md bg-black/80 flex flex-col"
      role="dialog"
      aria-modal="true"
    >
      <header className="flex items-center justify-between px-6 py-4">
        <div className="flex items-baseline gap-3">
          <span className="text-xs uppercase tracking-wider text-[var(--muted)]">
            {SCOPE_LABEL[scope]}
          </span>
          <span className="text-sm font-mono">
            {idx + 1} <span className="text-[var(--muted)]">/ {results.length}</span>
          </span>
          <span className="font-mono text-xs text-[var(--muted)] truncate max-w-[40vw]">
            {current.image}
          </span>
          {verdicts[current.image] && (
            <span
              className={[
                "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                verdicts[current.image] === "good"
                  ? "bg-green-500/25 border border-green-400/60 text-green-200"
                  : "bg-red-500/25 border border-red-400/60 text-red-200",
              ].join(" ")}
            >
              {verdicts[current.image]}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-2xl px-3 leading-none text-[var(--muted)] hover:text-foreground"
          aria-label="close"
        >
          ×
        </button>
      </header>

      <div className="flex-1 flex items-center justify-center px-6 overflow-hidden">
        <div
          className="relative"
          style={{
            width: "min(96vw, 1500px)",
            height: "78vh",
          }}
        >
          {visible.map((r, i) => {
            const isOutgoing = animating !== null && i === 0;
            // when animating, behind cards advance forward by one slot
            const slot = animating !== null && !isOutgoing ? i - 1 : i;

            let transform: string;
            let opacity = 1;
            if (isOutgoing) {
              transform =
                animating === "left"
                  ? "translateX(-80vw) rotate(-12deg)"
                  : "translateX(80vw) rotate(12deg)";
              opacity = 0;
            } else if (slot <= 0) {
              transform = "translate(0, 0) rotate(0deg) scale(1)";
            } else if (slot === 1) {
              transform = "translate(-14px, 18px) rotate(-4deg) scale(0.95)";
            } else {
              transform = "translate(16px, 34px) rotate(5deg) scale(0.9)";
            }
            const zIndex = 30 - i * 10;

            const W = r.size.width;
            const H = r.size.height;
            const boxes = boxesByImage[r.image] ?? [];
            // The "top" card needs the heavy interactive editor; behind cards
            // get the lightweight static preview. Crucially, the card that is
            // *about* to become top (i === 1 while a swipe is in flight) is
            // also rendered as the editor, so when the deck advances there's
            // no remount and no layout jump.
            const isTop = animating ? i === 1 || i === 0 : i === 0;
            const isInteractiveTop = i === 0 && !animating;

            return (
              <div
                key={r.image}
                className="absolute inset-0 transition-all duration-[280ms] ease-out"
                style={{
                  transform,
                  transformOrigin: "center center",
                  opacity,
                  zIndex,
                  pointerEvents: isInteractiveTop ? "auto" : "none",
                  filter: i === 0 ? "none" : "brightness(0.85)",
                }}
              >
                <div className="relative w-full h-full rounded-xl overflow-hidden bg-[var(--background)] border border-[var(--border)] shadow-2xl">
                  {isTop ? (
                    <BoxEditor
                      imageUrl={`${apiBase}/api/projects/${jobId}/originals/${encodeURIComponent(r.image)}`}
                      imageWidth={W}
                      imageHeight={H}
                      boxes={boxes}
                      colorMode="review"
                      projectTags={projectTags}
                      onChange={(next) => onBoxesChange?.(r.image, next)}
                      onBoxDrawn={onSegmentBox ? (b) => onSegmentBox(r.image, b) : undefined}
                      onClassifyBox={onClassifyBox ? (b) => onClassifyBox(r.image, b) : undefined}
                      onPointDetect={onPointDetect ? (p) => onPointDetect(r.image, p) : undefined}
                    />
                  ) : (
                    <DeckPreview
                      apiBase={apiBase}
                      jobId={jobId}
                      image={r.image}
                      width={W}
                      height={H}
                      boxes={boxes}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
        {/* Hidden preload bank, keeps the next few images warm in the
            browser cache so advancing the deck never waits on a network
            fetch. eager + async decode happens off the main paint path. */}
        <div aria-hidden className="absolute pointer-events-none opacity-0" style={{ width: 0, height: 0, overflow: "hidden" }}>
          {preload.map((r) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={r.image}
              src={`${apiBase}/api/projects/${jobId}/originals/${encodeURIComponent(r.image)}`}
              alt=""
              width={r.size.width}
              height={r.size.height}
              loading="eager"
              decoding="async"
              fetchPriority="high"
            />
          ))}
        </div>
      </div>

      <footer className="px-6 py-6 flex items-center justify-center gap-6 relative z-50">
        <button
          onClick={() => classify("bad")}
          className="rounded-full bg-red-500/15 border border-red-500/60 text-red-200 px-6 py-3 text-sm hover:bg-red-500/25 transition-colors"
        >
          ← Bad
        </button>
        <span className="text-xs text-[var(--muted)] font-mono hidden sm:inline">
          ←/→ keys · Esc to close
        </span>
        <button
          onClick={() => classify("good")}
          className="rounded-full bg-green-500/15 border border-green-500/60 text-green-200 px-6 py-3 text-sm hover:bg-green-500/25 transition-colors"
        >
          Good →
        </button>
      </footer>
    </div>
  );
}

function DeckPreview({
  apiBase,
  jobId,
  image,
  width,
  height,
  boxes,
}: {
  apiBase: string;
  jobId: string;
  image: string;
  width: number;
  height: number;
  boxes: EditableBox[];
}) {
  const fontSize = Math.max(width, height) * 0.022;
  const strokeWidth = Math.max(width, height) * 0.0035;
  const charW = fontSize * 0.6;
  const padX = fontSize * 0.35;
  const labelH = fontSize * 1.3;
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div
        className="relative"
        style={{
          height: "100%",
          maxWidth: "100%",
          aspectRatio: `${width} / ${height}`,
        }}
      >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`${apiBase}/api/projects/${jobId}/originals/${encodeURIComponent(image)}`}
        alt=""
        width={width}
        height={height}
        decoding="async"
        className="absolute inset-0 w-full h-full"
      />
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="absolute inset-0 w-full h-full pointer-events-none"
      >
        {boxes.map((b) => {
          const w = b.x1 - b.x0;
          const h = b.y1 - b.y0;
          const rejected = b.validation && b.validation.match === false;
          const stroke = rejected ? "rgba(248,113,113,1)" : "rgba(74,222,128,1)";
          const fill = rejected ? "rgba(248,113,113,0.95)" : "rgba(74,222,128,0.95)";
          const maskFill = rejected ? "rgba(248,113,113,0.30)" : "rgba(74,222,128,0.28)";
          const polys = b.mask?.polygons ?? [];
          const labelText = (rejected ? "⚠ " : "") + (b.label || "label");
          const maxChars = Math.max(8, Math.floor((w + fontSize * 4) / charW));
          const display = labelText.length > maxChars ? labelText.slice(0, maxChars - 1) + "…" : labelText;
          const labelW = display.length * charW + padX * 2;
          const labelY = b.y0 < labelH ? b.y0 + strokeWidth : b.y0 - labelH;
          return (
            <g key={b.id}>
              {polys.map((pts, pi) => (
                <polygon
                  key={pi}
                  points={pts.map((p) => `${p[0]},${p[1]}`).join(" ")}
                  fill={maskFill}
                  stroke={stroke}
                  strokeWidth={strokeWidth * 0.6}
                  opacity={0.9}
                />
              ))}
              <rect
                x={b.x0}
                y={b.y0}
                width={w}
                height={h}
                fill="transparent"
                stroke={stroke}
                strokeWidth={rejected ? strokeWidth * 1.7 : strokeWidth}
              />
              <rect x={b.x0} y={labelY} width={labelW} height={labelH} fill={fill} />
              <text
                x={b.x0 + padX}
                y={labelY + labelH * 0.74}
                fontSize={fontSize}
                fontFamily="ui-monospace, monospace"
                fill="#0a0a0a"
              >
                {display}
              </text>
            </g>
          );
        })}
      </svg>
      </div>
    </div>
  );
}
