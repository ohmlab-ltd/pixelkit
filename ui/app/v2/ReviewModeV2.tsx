"use client";

// Fast image-level triage for the V2 project page. Lifted from V1's
// ReviewMode and rebuilt to match V2's chrome:
//
//   • Image fills the card edge-to-edge (no letterbox gutter on the
//     primary review surface).
//   • Annotation list rides the right side so the user can read what
//     the segmenter picked while triaging the whole image.
//   • Theme-aware backdrop: dark blur over a dark wash in dark mode,
//     and the same blur over a white wash with white framing around
//     the image in light mode.
//   • Mask polygons + boxes render over the image with per-label
//     palette tints. Polygon paths preload from the manifest so the
//     overlay is in place before the JPEG decode finishes.
//
// Controls the user can drive:
//   • Swipe-style buttons:  "← Bad" / "Unsure" / "Good →"
//   • Arrow keys:           ← (bad) · ↓ (unsure) · → (good)
//   • Pointer drag:         throw past SWIPE_COMMIT_PX to commit
//   • Esc:                  closes the overlay
//
// Caching: PRELOAD_AHEAD images past the visible deck get an eager
// off-screen <img> with loading="eager" + fetchPriority="high" so
// flicking through a long unrated set never waits on the network.

import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildProjectLabelColourMap,
  colourForLabelStable,
  readableTextForBg,
} from "./OnboardLabelsV2";

export type Verdict = "good" | "bad" | "unsure";
export type ReviewScope = "unrated" | "unsure" | "good" | "bad" | "all";

export type ReviewDetection = {
  box?: number[] | null;        // [x0, y0, x1, y1] (intrinsic px)
  pred_label?: string | null;
  polygons?: number[][][] | null; // outer/inner polygons in intrinsic px
  score?: number | null;
};

export type ReviewItem = {
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
  /** Project tag list, used for the per-project palette so the
      review chips line up with the rest of the page. */
  projectLabels?: string[];
  labelAliases?: Record<string, string>;
  labelColours?: Record<string, string> | null;
  /** Called as the user navigates so the parent can pull per-image
      annotations (mask polygons) for the current + nearby items.
      Without this, /overview / /initial only carry placeholder
      detections without masks, and the review canvas renders boxes
      only, no segmentation. */
  onRequestAnnotations?: (backendId: string) => void;
}) {
  // Snapshot the items list at modal open. The parent filters by
  // verdict (unrated / good / bad / unsure), and committing a
  // verdict immediately removes the just-rated image from that
  // filter, which would shift the index by one in the same tick.
  // Walking a frozen list keeps the deck stable: the user just
  // moves forward, no "jumps to a different image and then snaps
  // back" effect. New mask/detection data still flows in because we
  // overlay it from the live items map on each render.
  const [snapshot] = useState<ReviewItem[]>(() => items);
  // When fresh detection data arrives via /annotations, merge it
  // into our snapshot by id so each card picks up its real
  // polygons + scores without re-snapshotting the list itself.
  const liveById = useMemo(() => {
    const m = new Map<string, ReviewItem>();
    for (const it of items) m.set(it.id, it);
    return m;
  }, [items]);
  const list = useMemo(
    () => snapshot.map((r) => liveById.get(r.id) ?? r),
    [snapshot, liveById],
  );

  const firstUnrated = list.findIndex((r) => !verdicts[r.id]);
  const startIdx = scope === "unrated" ? (firstUnrated === -1 ? 0 : firstUnrated) : 0;
  const [idx, setIdx] = useState(startIdx);
  const [animating, setAnimating] = useState<"left" | "right" | null>(null);
  const [drag, setDrag] = useState(0);
  // Index of the annotation the user is hovering in the sidebar.
  // Drives a "pulse + brighten" treatment of the matching segmentation
  // overlay so the user can map a chip in the list back to a region
  // in the image. Reset on every image advance.
  const [hoveredAnnot, setHoveredAnnot] = useState<number | null>(null);
  useEffect(() => setHoveredAnnot(null), [idx]);
  const dragStartRef = useRef<{ x: number; pointerId: number } | null>(null);

  const current = list[idx];
  const isDone = idx >= list.length;

  // Pull per-image annotations (mask polygons) for the current item
  // and a small window around it. The list comes through with
  // placeholder detections from /overview that carry boxes + labels
  // but no mask polygons, without this fetch the canvas would draw
  // only the rectangles. Mirrors DatasetViewer's neighbour-prefetch
  // pattern so flicking through the deck stays smooth.
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

  // Stable colour map for this project's labels, same palette the
  // dataset gallery chips use.
  const colourFor = useMemo(() => {
    const map = buildProjectLabelColourMap(projectLabels ?? [], labelColours ?? null);
    return (lab: string | null): string => {
      if (!lab) return "#a3a3a3";
      const key = lab.trim().toLowerCase();
      return map.get(key) ?? colourForLabelStable(lab);
    };
  }, [projectLabels, labelColours]);

  const displayLabel = (lab: string | null): string => {
    if (!lab) return ",";
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

  // Pointer-drag throw. Same flow for mouse / touch / pen.
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
        className="fixed inset-0 z-[1200] grid place-items-center p-6"
        // Themable wash: white frosted-glass in light mode, near-black
        // in dark. Reads cleanly on either palette without overriding
        // the page's accent.
        style={{
          background: "rgb(var(--background-rgb) / 0.7)",
          backdropFilter: "blur(18px) saturate(140%)",
          WebkitBackdropFilter: "blur(18px) saturate(140%)",
        }}
      >
        <div className="rounded-2xl border border-foreground/15 bg-[var(--surface)] px-10 py-9 text-center max-w-md shadow-[var(--shadow-strong)]">
          <h2 className="text-2xl font-semibold tracking-tight text-[var(--foreground)]">
            Review complete
          </h2>
          <p className="mt-3 text-foreground/55 text-sm">
            <span className="text-emerald-600 dark:text-emerald-400 font-mono">{goods}</span> good ·{" "}
            <span className="text-rose-600 dark:text-rose-400 font-mono">{bads}</span> bad ·{" "}
            <span className="text-amber-600 dark:text-amber-400 font-mono">{unsures}</span> unsure
          </p>
          <button
            onClick={onClose}
            className="mt-7 rounded-full bg-foreground text-background px-5 py-2.5 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            Close
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
      className="fixed inset-0 z-[1200] flex flex-col"
      role="dialog"
      aria-modal="true"
      aria-label="Fast review"
      style={{
        // Themable wash. --background-rgb is white (light) / near-black
        // (dark), so this reads as a frosted-glass surface in either
        // theme. Blur + saturate match the workspace navigation
        // chrome so the overlay feels like part of the app.
        background: "rgb(var(--background-rgb) / 0.72)",
        backdropFilter: "blur(20px) saturate(140%)",
        WebkitBackdropFilter: "blur(20px) saturate(140%)",
      }}
    >
      {/* Direction wash, only visible during a pointer drag */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 transition-opacity duration-150"
        style={{
          background: `radial-gradient(ellipse at center, rgba(${tintColour}, ${tintOpacity}), transparent 70%)`,
        }}
      />

      <header className="relative flex items-center justify-between gap-4 px-6 py-3.5 border-b border-foreground/10">
        <div className="flex items-baseline gap-3 min-w-0">
          <span className="text-[10px] uppercase tracking-[0.18em] text-foreground/45">
            {SCOPE_LABEL[scope]}
          </span>
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
          onClick={onClose}
          className="text-foreground/45 hover:text-foreground text-2xl leading-none px-2"
          aria-label="Close review"
        >
          ×
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
                  className="absolute inset-0 rounded-2xl overflow-hidden border border-foreground/15 bg-[var(--surface)] shadow-2xl"
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
                        "pointer-events-none absolute top-6 px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wider border-2 backdrop-blur-md",
                        drag < 0
                          ? "left-6 -rotate-12 border-rose-500 text-rose-700 dark:text-rose-200 bg-rose-500/15"
                          : "right-6 rotate-12 border-emerald-500 text-emerald-700 dark:text-emerald-200 bg-emerald-500/15",
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

        {/* Annotation panel on the right. Lists every detection on the
            current image with its label chip + confidence. Reads as
            a quiet sidebar, same surface tone as the card behind. */}
        <aside className="hidden lg:flex flex-col w-72 border-l border-foreground/10 bg-[var(--surface)]/60 backdrop-blur-md">
          <div className="px-4 py-3 border-b border-foreground/[0.07] flex items-baseline justify-between gap-2">
            <h3 className="text-[12px] font-semibold tracking-[-0.01em] text-[var(--foreground)]">
              Annotations
            </h3>
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
                  const fg = readableTextForBg(bg);
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
                        "flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 cursor-default transition-colors",
                        isHovered
                          ? "border-foreground/30 bg-foreground/[0.08]"
                          : "border-foreground/[0.07] bg-foreground/[0.02] hover:border-foreground/20 hover:bg-foreground/[0.05]",
                      ].join(" ")}
                    >
                      <span
                        className="inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium truncate max-w-[8.5rem]"
                        style={{ backgroundColor: bg, color: fg }}
                        title={d.pred_label ?? ""}
                      >
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

      <footer className="relative px-6 py-4 border-t border-foreground/10">
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
  const styles = v === "good"
    ? "bg-emerald-500/15 border-emerald-400/60 text-emerald-700 dark:text-emerald-200"
    : v === "bad"
      ? "bg-rose-500/15 border-rose-400/60 text-rose-700 dark:text-rose-200"
      : "bg-amber-500/15 border-amber-400/60 text-amber-700 dark:text-amber-200";
  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider border ${styles}`}>
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
    bad: "bg-rose-500/15 border-rose-400/70 text-rose-700 dark:text-rose-200 hover:bg-rose-500/25",
    good: "bg-emerald-500/15 border-emerald-400/70 text-emerald-700 dark:text-emerald-200 hover:bg-emerald-500/25",
    unsure: "bg-amber-500/15 border-amber-400/70 text-amber-700 dark:text-amber-200 hover:bg-amber-500/25",
  };
  return (
    <div className="grid place-items-center gap-1">
      <button
        onClick={onClick}
        className={[
          "inline-flex items-center gap-2 rounded-full border transition-colors font-medium",
          small ? "px-4 py-2 text-xs" : "px-5 py-2.5 text-sm",
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
  /** Index of the annotation the user is hovering in the sidebar.
      The matching overlay brightens + animates so the user can
      visually map a chip back to the region it describes. */
  highlightIdx?: number | null;
  /** Mirror image-side hover back up to the parent so the sidebar
      annotation row matching the segmentation under the cursor
      lights up at the same time. */
  onHighlightChange?: (idx: number | null) => void;
}) {
  // Use the image's NATURAL dimensions for the SVG viewBox, the
  // manifest's stored width/height occasionally drift from the actual
  // file (re-bake, resize) and a mismatch puts the boxes / polygons
  // in the wrong place. Falls back to manifest dims on first paint,
  // then snaps to the loaded image's real dims once decode finishes.
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(() =>
    item.width && item.height ? { w: item.width, h: item.height } : null,
  );
  useEffect(() => {
    setNatural(item.width && item.height ? { w: item.width, h: item.height } : null);
  }, [item.id, item.width, item.height]);
  const W = natural?.w ?? 0;
  const H = natural?.h ?? 0;
  // Track the container's on-screen size so label fontSize + chip
  // dimensions can be pinned to screen pixels rather than scaled
  // with the image. Without this, portrait images get visibly
  // smaller labels than landscape ones because viewBox->screen
  // scale changes with aspect ratio under preserveAspectRatio=meet.
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
  // Screen-pixels-per-viewBox-unit. With preserveAspectRatio="xMidYMid
  // meet" the limiting axis sets the scale, so min() of the two
  // ratios. Used to invert image-space sizes back to screen pixels.
  const pxPerUnit =
    W > 0 && H > 0 && containerSize.w > 0 && containerSize.h > 0
      ? Math.min(containerSize.w / W, containerSize.h / H)
      : 0;
  // Single SVG renders BOTH the image and the detection overlays so
  // they share one coordinate system. Previously the image was an
  // <img> with object-contain and the detections were in a separate
  // <svg> with preserveAspectRatio="xMidYMid meet", when the
  // manifest's stored dimensions diverged from the actual file,
  // those two layers fit-meet'd differently and the overlay landed
  // off the rendered image. With a single SVG <image>, both are
  // scaled by the same viewBox transform.
  return (
    <div ref={containerRef} className="relative w-full h-full select-none bg-white dark:bg-black">
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
          {/* Soft glow filter applied to the hovered detection so it
              "pops" with a brighter halo, mirroring BoxEditorV2's
              bx-select-glow. Replaces the previous spotlight-dim
              approach which read as a pulse because every hover
              swap re-mounted a big translucent <rect>. */}
          <defs>
            <filter id={`rev-pop-${item.id}`} x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation={2.2} result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* Render detections sorted by area descending, large
              boxes paint first so smaller ones stack on top and
              stay readable / clickable. Matches BoxEditorV2's
              z-order rule. */}
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
            // Uniform stroke width across boxes + polygons so every
            // detection reads as the same line weight regardless of
            // image scale. vector-effect locks it to screen pixels.
            const STROKE_PX = 1.8;
            const LABEL_FONT_PX = 13;
            const labelText = displayLabel(d.pred_label ?? null);
            const fontSize = pxPerUnit > 0 ? LABEL_FONT_PX / pxPerUnit : 0;
            const padX = fontSize * 0.45;
            const labelH = fontSize * 1.4;
            const isHovered = i === highlightIdx;
            // Hovered detection brightens its fill + opacity (the
            // "pop") and gets a soft glow via the filter above. Idle
            // detections stay at their resting opacity so the page
            // doesn't pulse on hover.
            const fillRgba = hexToRgba(colour, isHovered ? 0.4 : 0.22);
            const lx = box ? box[0] : 0;
            const ly = box ? box[1] : 0;
            const labelW = labelText.length * fontSize * 0.6 + padX * 2;
            // Stroke offset for labelY in viewBox units, so the chip
            // sits flush above/below the box border without overlap.
            const strokeOffset = pxPerUnit > 0 ? STROKE_PX / pxPerUnit : 0;
            const labelY = ly < labelH ? ly + strokeOffset : ly - labelH;
            return (
              <g
                key={i}
                // pointer-events: auto on the overlay group lets the
                // user hover the segmentation directly to highlight
                // the matching sidebar annotation. The pointer
                // events still bubble up to the card so the drag
                // gesture starts on pointerdown, segmentation hover
                // doesn't swallow the swipe.
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
        // Manifest dimensions missing, fall back to a plain img so
        // the user still sees the photo while waiting for the natural
        // size to land (will re-render into the SVG path on next
        // mount once width/height arrive).
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
