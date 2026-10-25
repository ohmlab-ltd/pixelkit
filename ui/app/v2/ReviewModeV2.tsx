"use client";

// Review mode: rates images good, bad or unsure by button, arrow key or swipe.

import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildProjectLabelColourMap,
  colourForLabelStable,
  readableTextForBg,
} from "./labelColours";

type Verdict = "good" | "bad" | "unsure";
type ReviewScope = "unrated" | "unsure" | "good" | "bad" | "all";

type ReviewDetection = {
  box?: number[] | null;        // [x0, y0, x1, y1], intrinsic px
  pred_label?: string | null;
  polygons?: number[][][] | null; // outer/inner polygons in intrinsic px
  score?: number | null;
};

type ReviewItem = {
  id: string;
  filename: string;
  preview: string;
  width?: number;
  height?: number;
  detections?: ReviewDetection[];
};

const STACK_DEPTH = 3; // top + 2 behind
const PRELOAD_AHEAD = 8;
const SWIPE_COMMIT_PX = 120;
const SWIPE_TINT_PX = 40;

const SCOPE_LABEL: Record<ReviewScope, string> = {
  unrated: "Unrated",
  unsure: "Unsure",
  good: "Good",
  bad: "Bad",
  all: "All",
};

export function ReviewModeV2({
  items,
  verdicts,
  onVerdict,
  onClose,
  scope = "unrated",
  projectLabels,
  labelAliases = {},
  labelColours = null,
  onRequestAnnotations,
}: {
  items: ReviewItem[];
  verdicts: Record<string, Verdict>;
  onVerdict: (id: string, verdict: Verdict) => void;
  onClose: () => void;
  scope?: ReviewScope;
  /** Project labels, for the label palette. */
  projectLabels?: string[];
  labelAliases?: Record<string, string>;
  labelColours?: Record<string, string> | null;
  /** Asks the parent to load mask polygons for an image near the current one. */
  onRequestAnnotations?: (backendId: string) => void;
}) {
  // Items frozen at open: a verdict removes the image from the parent's
  // filter, which would shift the index. Detection data still merges live.
  const [snapshot] = useState<ReviewItem[]>(() => items);
  const liveById = useMemo(() => {
    const m = new Map<string, ReviewItem>();
    for (const it of items) m.set(it.id, it);
    return m;
  }, [items]);

  // Least confident first: ordered by each image's lowest detection score,
  // with unscored images last.
  const [byConfidence, setByConfidence] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem("pk-review-confidence-sort") === "1";
    } catch {
      return false;
    }
  });
  const minScore = (r: ReviewItem): number => {
    let min = Infinity;
    for (const d of r.detections ?? []) {
      if (typeof d.score === "number") min = Math.min(min, d.score);
    }
    return min;
  };
  // The order is frozen when the sort turns on, so live score updates do
  // not move the top card.
  const orderRef = useRef<string[] | null>(null);
  const list = useMemo(() => {
    const merged = snapshot.map((r) => liveById.get(r.id) ?? r);
    if (!byConfidence) return merged;
    if (orderRef.current === null) {
      orderRef.current = [...merged]
        .sort((a, b) => minScore(a) - minScore(b))
        .map((r) => r.id);
    }
    const byId = new Map(merged.map((r) => [r.id, r]));
    return orderRef.current
      .map((id) => byId.get(id))
      .filter((r): r is ReviewItem => !!r);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot, liveById, byConfidence]);

  const firstUnrated = list.findIndex((r) => !verdicts[r.id]);
  const startIdx = scope === "unrated" ? (firstUnrated === -1 ? 0 : firstUnrated) : 0;
  const [idx, setIdx] = useState(startIdx);
  const toggleConfidenceSort = () => {
    const next = !byConfidence;
    orderRef.current = null; // re-freeze from current scores on next render
    try {
      window.localStorage.setItem("pk-review-confidence-sort", next ? "1" : "0");
    } catch {
      /* the toggle still works for this session */
    }
    // Lands on the first unrated card of the new order.
    const merged = snapshot.map((r) => liveById.get(r.id) ?? r);
    const ordered = next
      ? [...merged].sort((a, b) => minScore(a) - minScore(b))
      : merged;
    const first = scope === "unrated" ? ordered.findIndex((r) => !verdicts[r.id]) : 0;
    setByConfidence(next);
    setIdx(first === -1 ? 0 : first);
  };
  const [animating, setAnimating] = useState<"left" | "right" | null>(null);
  const [drag, setDrag] = useState(0);
  // Sidebar annotation under the pointer, highlighted in the image.
  const [hoveredAnnot, setHoveredAnnot] = useState<number | null>(null);
  useEffect(() => setHoveredAnnot(null), [idx]);
  const dragStartRef = useRef<{ x: number; pointerId: number } | null>(null);

  const current = list[idx];
  const isDone = idx >= list.length;

  // Requests masks for the current image and its neighbours.
  useEffect(() => {
    if (!onRequestAnnotations) return;
    const want = (it: ReviewItem | undefined) => {
      if (!it) return;
      const hasMask = (it.detections ?? []).some((d) => d?.polygons && d.polygons.length > 0);
      if (hasMask) return;
      onRequestAnnotations(it.id);
    };
    want(list[idx]);
    want(list[idx + 1]);
    want(list[idx + 2]);
    want(list[idx - 1]);
  }, [idx, list, onRequestAnnotations]);

  const colourFor = useMemo(() => {
    const map = buildProjectLabelColourMap(projectLabels ?? [], labelColours ?? null);
    return (lab: string | null): string => {
      if (!lab) return "#a3a3a3";
      const key = lab.trim().toLowerCase();
      return map.get(key) ?? colourForLabelStable(lab);
    };
  }, [projectLabels, labelColours]);

  const displayLabel = (lab: string | null): string => {
    if (!lab) return "No label";
    const k = lab.trim().toLowerCase();
    return labelAliases[k] || lab;
  };

  const classify = (verdict: Verdict) => {
    if (!current || animating) return;
    setAnimating(verdict === "bad" ? "left" : "right");
    setDrag(0);
    window.setTimeout(() => {
      onVerdict(current.id, verdict);
      setIdx((i) => i + 1);
      setAnimating(null);
    }, 260);
  };

  // Pointer drag, for mouse, touch and pen.
  const onPointerDown = (e: React.PointerEvent) => {
    if (animating) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragStartRef.current = { x: e.clientX, pointerId: e.pointerId };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragStartRef.current || dragStartRef.current.pointerId !== e.pointerId) return;
    setDrag(e.clientX - dragStartRef.current.x);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    if (!dragStartRef.current || dragStartRef.current.pointerId !== e.pointerId) return;
    const total = e.clientX - dragStartRef.current.x;
    dragStartRef.current = null;
    if (Math.abs(total) >= SWIPE_COMMIT_PX) {
      classify(total < 0 ? "bad" : "good");
    } else {
      setDrag(0);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      const editing = tag === "input" || tag === "textarea"
        || (document.activeElement as HTMLElement | null)?.isContentEditable;
      if (editing) return;
      // A held key must not rate unseen cards.
      if (e.repeat) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        classify("bad");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        classify("good");
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        classify("unsure");
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animating, current?.id, isDone]);

  if (isDone) {
    const goods = Object.values(verdicts).filter((v) => v === "good").length;
    const bads = Object.values(verdicts).filter((v) => v === "bad").length;
    const unsures = Object.values(verdicts).filter((v) => v === "unsure").length;
    return (
      <div
        // Fills the shell's content area, leaving the app chrome visible.
        className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[1200] grid place-items-center bg-[var(--background)] p-6"
        role="dialog"
        aria-modal="true"
      >
        <div className="max-w-md rounded-md border border-[var(--line)] bg-[var(--panel)] px-8 py-7 text-center">
          <h2 className="text-[16px] font-medium tracking-tight text-[var(--foreground)]">
            Review complete
          </h2>
          <p className="mt-3 text-foreground/55 text-sm tabular-nums">
            <span className="font-mono text-[var(--ok)]">{goods}</span> good ·{" "}
            <span className="font-mono text-[var(--bad)]">{bads}</span> bad ·{" "}
            <span className="font-mono text-[var(--warn)]">{unsures}</span> unsure
          </p>
          <button onClick={onClose} className="pk-btn mt-6">
            Back to dataset
          </button>
        </div>
      </div>
    );
  }

  const visible = list.slice(idx, Math.min(idx + STACK_DEPTH, list.length));
  const preload = list.slice(
    idx + STACK_DEPTH,
    Math.min(idx + STACK_DEPTH + PRELOAD_AHEAD, list.length),
  );

  const tintIntensity = Math.min(1, Math.abs(drag) / SWIPE_TINT_PX);
  const tintColour = drag < 0 ? "239, 68, 68" : "34, 197, 94";
  const tintOpacity = animating ? 0 : Math.min(0.18, tintIntensity * 0.18);

  return (
    <div
      // Fills the shell's content area, leaving the app chrome visible.
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[1200] flex flex-col bg-[var(--background)]"
      role="dialog"
      aria-modal="true"
      aria-label="Fast review"
    >
      {/* Direction wash, only visible during a pointer drag */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 transition-opacity duration-150"
        style={{
          background: `radial-gradient(ellipse at center, rgba(${tintColour}, ${tintOpacity}), transparent 70%)`,
        }}
      />

      <header className="relative flex items-center gap-4 px-6 py-3 border-b border-[var(--line)]">
        <button type="button" onClick={onClose} className="pk-btn shrink-0" aria-label="Close review">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M19 12H5" /><path d="m12 19-7-7 7-7" />
          </svg>
          Back
        </button>
        <div className="flex min-w-0 items-baseline gap-3">
          <span className="pk-micro">{SCOPE_LABEL[scope]}</span>
          <span className="text-sm font-mono text-[var(--foreground)] tabular-nums">
            {idx + 1}
            <span className="text-foreground/35"> / {list.length}</span>
          </span>
          <span className="font-mono text-xs text-foreground/45 truncate max-w-[40vw]">
            {current.filename}
          </span>
          {verdicts[current.id] && (
            <VerdictPill v={verdicts[current.id]} />
          )}
        </div>
        <button
          type="button"
          onClick={toggleConfidenceSort}
          title="Order the deck by each image's weakest detection score"
          className={[
            "ml-auto shrink-0 rounded-md border px-2.5 py-1 text-[11px] font-mono uppercase tracking-[0.08em] transition-colors",
            byConfidence
              ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-dim)]"
              : "border-[var(--line)] text-[var(--fg-dim)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
          ].join(" ")}
        >
          Least confident first
        </button>
      </header>

      <div className="relative flex-1 flex items-stretch overflow-hidden">
        <div className="flex-1 relative px-6 py-6 flex items-center justify-center min-w-0">
          <div
            className="relative w-full h-full"
            style={{ maxWidth: "min(96%, 1500px)", maxHeight: "100%" }}
          >
            {visible.map((r, i) => {
              const isOutgoing = animating !== null && i === 0;
              const slot = animating !== null && !isOutgoing ? i - 1 : i;

              let transform: string;
              let opacity = 1;
              if (isOutgoing) {
                transform = animating === "left"
                  ? "translateX(-80vw) rotate(-12deg)"
                  : "translateX(80vw) rotate(12deg)";
                opacity = 0;
              } else if (slot <= 0) {
                const tilt = i === 0 ? drag * 0.05 : 0;
                transform = `translate(${i === 0 ? drag : 0}px, 0) rotate(${tilt}deg) scale(1)`;
              } else if (slot === 1) {
                transform = "translate(-14px, 18px) rotate(-4deg) scale(0.95)";
              } else {
                transform = "translate(16px, 34px) rotate(5deg) scale(0.9)";
              }
              const zIndex = 30 - i * 10;
              const isTop = i === 0 && !animating;
              const transition = isTop && dragStartRef.current
                ? "none"
                : "transform 260ms cubic-bezier(0.2,0.7,0.2,1), opacity 260ms ease";

              return (
                <div
                  key={r.id}
                  className="absolute inset-0 rounded-md overflow-hidden border border-[var(--line)] bg-[var(--surface)] shadow-[var(--shadow-strong)]"
                  style={{
                    transform,
                    transformOrigin: "center center",
                    opacity,
                    zIndex,
                    pointerEvents: isTop ? "auto" : "none",
                    filter: i === 0 ? "none" : "brightness(0.92)",
                    cursor: isTop ? "grab" : "default",
                    touchAction: "none",
                    transition,
                  }}
                  onPointerDown={isTop ? onPointerDown : undefined}
                  onPointerMove={isTop ? onPointerMove : undefined}
                  onPointerUp={isTop ? onPointerUp : undefined}
                  onPointerCancel={isTop ? onPointerUp : undefined}
                >
                  <ReviewCanvas
                    item={r}
                    colourFor={colourFor}
                    displayLabel={displayLabel}
                    highlightIdx={isTop ? hoveredAnnot : null}
                    onHighlightChange={isTop ? setHoveredAnnot : undefined}
                  />
                  {isTop && Math.abs(drag) > 20 && (
                    <span
                      className={[
                        "pointer-events-none absolute top-6 rounded-md border-2 bg-black/55 px-4 py-2 font-mono text-sm font-medium uppercase tracking-wider backdrop-blur-md",
                        drag < 0
                          ? "left-6 -rotate-12 border-[var(--bad)] text-[var(--bad)]"
                          : "right-6 rotate-12 border-[var(--ok)] text-[var(--ok)]",
                      ].join(" ")}
                      style={{ opacity: tintIntensity }}
                    >
                      {drag < 0 ? "Bad" : "Good"}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          <div
            aria-hidden
            className="absolute pointer-events-none opacity-0"
            style={{ width: 0, height: 0, overflow: "hidden" }}
          >
            {preload.map((r) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={r.id}
                src={r.preview}
                alt=""
                width={r.width ?? 64}
                height={r.height ?? 64}
                loading="eager"
                decoding="async"
                fetchPriority="high"
              />
            ))}
          </div>
        </div>

        {/* Annotation panel */}
        <aside className="hidden lg:flex flex-col w-72 border-l border-[var(--line)] bg-[var(--panel)]">
          <div className="px-4 py-3 border-b border-foreground/[0.07] flex items-baseline justify-between gap-2">
            <h3 className="pk-micro text-[var(--fg-soft)]">Annotations</h3>
            <span className="text-[10px] font-mono text-foreground/40 tabular-nums">
              {current.detections?.length ?? 0}
            </span>
          </div>
          <div className="flex-1 overflow-y-auto px-3 py-3">
            {(current.detections ?? []).length === 0 ? (
              <p className="text-[12px] text-foreground/45 px-1 py-2">
                No detections on this image.
              </p>
            ) : (
              <ul className="grid gap-1.5">
                {(current.detections ?? []).map((d, i) => {
                  const bg = colourFor(d.pred_label ?? null);
                  const isHovered = hoveredAnnot === i;
                  return (
                    <li
                      key={i}
                      onMouseEnter={() => setHoveredAnnot(i)}
                      onMouseLeave={() => setHoveredAnnot((cur) => (cur === i ? null : cur))}
                      onFocus={() => setHoveredAnnot(i)}
                      onBlur={() => setHoveredAnnot((cur) => (cur === i ? null : cur))}
                      tabIndex={0}
                      className={[
                        "flex items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 cursor-default transition-colors",
                        isHovered
                          ? "border-[var(--line-strong)] bg-[var(--surface-hover)]"
                          : "border-[var(--line)] bg-transparent hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
                      ].join(" ")}
                    >
                      <span
                        className="inline-flex max-w-[8.5rem] items-center gap-1.5 truncate font-mono text-[11px] text-[var(--fg-soft)]"
                        title={d.pred_label ?? ""}
                      >
                        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: bg }} aria-hidden />
                        {displayLabel(d.pred_label ?? null)}
                      </span>
                      {typeof d.score === "number" && (
                        <span className="font-mono text-[10px] text-foreground/55 tabular-nums shrink-0">
                          {(d.score * 100).toFixed(0)}%
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>
      </div>

      <footer className="relative px-6 py-4 border-t border-[var(--line)]">
        <div className="flex items-center justify-center gap-6">
          <ControlButton tone="bad" hint="← key" onClick={() => classify("bad")}>
            ← Bad
          </ControlButton>
          <ControlButton tone="unsure" hint="↓ key" onClick={() => classify("unsure")} small>
            Unsure
          </ControlButton>
          <ControlButton tone="good" hint="→ key" onClick={() => classify("good")}>
            Good →
          </ControlButton>
        </div>
        <p className="mt-2 text-center text-[10px] font-mono uppercase tracking-[0.18em] text-foreground/40">
          ← bad · ↓ unsure · → good · drag to swipe · esc to close
        </p>
      </footer>
    </div>
  );
}

function VerdictPill({ v }: { v: Verdict }) {
  const tone = v === "good" ? "text-[var(--ok)]" : v === "bad" ? "text-[var(--bad)]" : "text-[var(--warn)]";
  return (
    <span className={`rounded-md border border-[var(--line)] bg-transparent px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider ${tone}`}>
      {v}
    </span>
  );
}

function ControlButton({
  children,
  onClick,
  tone,
  hint,
  small = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  tone: "bad" | "good" | "unsure";
  hint: string;
  small?: boolean;
}) {
  const tones: Record<typeof tone, string> = {
    bad: "text-[var(--bad)]",
    good: "text-[var(--ok)]",
    unsure: "text-[var(--warn)]",
  };
  return (
    <div className="grid place-items-center gap-1">
      <button
        onClick={onClick}
        className={[
          "inline-flex items-center gap-2 rounded-md border border-[var(--line)] bg-transparent font-medium transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
          small ? "px-4 py-1.5 text-xs" : "px-5 py-2 text-sm",
          tones[tone],
        ].join(" ")}
      >
        {children}
      </button>
      <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-foreground/35">
        {hint}
      </span>
    </div>
  );
}

function ReviewCanvas({
  item,
  colourFor,
  displayLabel,
  highlightIdx = null,
  onHighlightChange,
}: {
  item: ReviewItem;
  colourFor: (lab: string | null) => string;
  displayLabel: (lab: string | null) => string;
  /** Sidebar annotation to highlight. */
  highlightIdx?: number | null;
  /** Reports the annotation under the pointer in the image. */
  onHighlightChange?: (idx: number | null) => void;
}) {
  // The manifest size until the image decodes, then its natural size,
  // which can differ.
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(() =>
    item.width && item.height ? { w: item.width, h: item.height } : null,
  );
  useEffect(() => {
    setNatural(item.width && item.height ? { w: item.width, h: item.height } : null);
  }, [item.id, item.width, item.height]);
  const W = natural?.w ?? 0;
  const H = natural?.h ?? 0;
  // Container size, to keep labels in screen pixels.
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setContainerSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // Screen pixels per viewBox unit under xMidYMid meet.
  const pxPerUnit =
    W > 0 && H > 0 && containerSize.w > 0 && containerSize.h > 0
      ? Math.min(containerSize.w / W, containerSize.h / H)
      : 0;
  // One SVG holds the image and the overlays, so both share the viewBox.
  return (
    <div ref={containerRef} className="relative w-full h-full select-none bg-black">
      {W > 0 && H > 0 ? (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          <image
            href={item.preview}
            x="0"
            y="0"
            width={W}
            height={H}
            preserveAspectRatio="xMidYMid meet"
            onLoad={(e) => {
              const img = e.currentTarget as unknown as SVGImageElement & {
                naturalWidth?: number; naturalHeight?: number;
              };
              const nw = img.naturalWidth ?? 0;
              const nh = img.naturalHeight ?? 0;
              if (nw > 0 && nh > 0 && (nw !== W || nh !== H)) {
                setNatural({ w: nw, h: nh });
              }
            }}
          />
          {/* Glow for the hovered detection */}
          <defs>
            <filter id={`rev-pop-${item.id}`} x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation={2.2} result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* Largest first, so smaller detections paint on top. */}
          {(item.detections ?? [])
            .map((d, i) => ({ d, i }))
            .sort((a, b) => {
              const aBox = a.d.box;
              const bBox = b.d.box;
              const aArea = aBox ? (aBox[2] - aBox[0]) * (aBox[3] - aBox[1]) : 0;
              const bArea = bBox ? (bBox[2] - bBox[0]) * (bBox[3] - bBox[1]) : 0;
              return bArea - aArea;
            })
            .map(({ d, i }) => {
            const box = d.box;
            const polys = d.polygons ?? null;
            const colour = colourFor(d.pred_label ?? null);
            const fg = readableTextForBg(colour);
            // Screen pixels.
            const STROKE_PX = 1.8;
            const LABEL_FONT_PX = 13;
            const labelText = displayLabel(d.pred_label ?? null);
            const fontSize = pxPerUnit > 0 ? LABEL_FONT_PX / pxPerUnit : 0;
            const padX = fontSize * 0.45;
            const labelH = fontSize * 1.4;
            const isHovered = i === highlightIdx;
            const fillRgba = hexToRgba(colour, isHovered ? 0.4 : 0.22);
            const lx = box ? box[0] : 0;
            const ly = box ? box[1] : 0;
            const labelW = labelText.length * fontSize * 0.6 + padX * 2;
            // Stroke width in viewBox units.
            const strokeOffset = pxPerUnit > 0 ? STROKE_PX / pxPerUnit : 0;
            const labelY = ly < labelH ? ly + strokeOffset : ly - labelH;
            return (
              <g
                key={i}
                // Hovering highlights the sidebar row. Events still bubble
                // to the card, so a drag starts here too.
                style={{
                  pointerEvents: "auto",
                  cursor: "default",
                }}
                onPointerEnter={() => onHighlightChange?.(i)}
                onPointerLeave={() => onHighlightChange?.(null)}
                filter={isHovered ? `url(#rev-pop-${item.id})` : undefined}
              >
                {polys && polys.length > 0 && polys.map((poly, pi) => (
                  <polygon
                    key={pi}
                    points={poly.map((p) => `${p[0]},${p[1]}`).join(" ")}
                    fill={fillRgba}
                    stroke={colour}
                    strokeWidth={STROKE_PX}
                    vectorEffect="non-scaling-stroke"
                    strokeLinejoin="round"
                  />
                ))}
                {box && (
                  <rect
                    x={box[0]} y={box[1]}
                    width={Math.max(0, box[2] - box[0])}
                    height={Math.max(0, box[3] - box[1])}
                    fill={polys && polys.length > 0 ? "transparent" : fillRgba}
                    stroke={colour}
                    strokeWidth={STROKE_PX}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {d.pred_label && box && fontSize > 0 && (
                  <g>
                    <rect
                      x={lx}
                      y={labelY}
                      width={labelW}
                      height={labelH}
                      rx={labelH * 0.5}
                      ry={labelH * 0.5}
                      fill={colour}
                    />
                    <text
                      x={lx + padX}
                      y={labelY + labelH * 0.72}
                      fontSize={fontSize}
                      fontFamily="ui-sans-serif, system-ui, sans-serif"
                      fontWeight={600}
                      fill={fg}
                    >
                      {labelText}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      ) : (
        // A plain img until the size is known.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={item.preview}
          alt={item.filename}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          draggable={false}
          decoding="async"
          onLoad={(e) => {
            const img = e.currentTarget;
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
              setNatural({ w: img.naturalWidth, h: img.naturalHeight });
            }
          }}
        />
      )}
    </div>
  );
}

function hexToRgba(hex: string, alpha: number): string {
  let s = hex.trim().replace(/^#/, "");
  if (s.length === 3) s = s.split("").map((c) => c + c).join("");
  if (s.length !== 6) return `rgba(160, 160, 160, ${alpha})`;
  const r = parseInt(s.slice(0, 2), 16);
  const g = parseInt(s.slice(2, 4), 16);
  const b = parseInt(s.slice(4, 6), 16);
  if ([r, g, b].some((n) => Number.isNaN(n))) return `rgba(160, 160, 160, ${alpha})`;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
