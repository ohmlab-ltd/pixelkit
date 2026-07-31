"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { BlurhashCanvas } from "react-blurhash";
import { MaskPainter, type MaskPainterHandle, type PaintMode } from "./MaskPainter";
import { containsProfanity } from "./profanity";
import { buildProjectLabelColourMap, colourForLabelStable } from "./v2/OnboardLabelsV2";

export type MaskShape = { polygons: number[][][] };

export type Validation = {
  match: boolean;
  confidence: number;
  reason: string;
  model?: string | null;
  // "auto", written by the validator pass. "manual", user verified by
  // hand. "cascade", applied via the Label Cascade modal (similarity
  // match the user accepted). The chip swaps colour to reflect the
  // provenance.
  source?: "auto" | "manual" | "cascade";
  // Stage tag used for the rejection chip's subtext, plus an
  // "unsure" variant the V2 pipeline emits when the resolver kept
  // the label but flagged it ambiguous (low detector score, detector
  // ≠ embed nearest, or tight top-1/top-2 margin), surfaces an amber
  // pill distinct from the green Verified one without forcing a flip
  // through the rejected path.
  kind?: "vlm" | "cascade" | "unsure";
};

export type EditableBox = {
  id: string;
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  score: number | null;
  mask?: MaskShape | null;
  /** True once the user hand-painted / edited the mask in MaskPainter, so
      a later box move or resize must NOT auto-overwrite it by re-running
      segmentation. Persisted so the provenance survives reloads. */
  maskEdited?: boolean;
  segmenting?: boolean;
  classifying?: boolean;
  /** One-click detection in flight, gets the special pulsing "AI" overlay
      instead of the plain label chip. */
  detecting?: { cx: number; cy: number } | boolean;
  validation?: Validation | null;
};

// FE-only UI state that must NEVER reach the manifest. The auto-
// PUT and the /annotations fetches both shuttle EditableBox arrays
// across the wire, and without this strip a click-to-detect that
// got auto-PUT'd mid-flight persisted `detecting: true` forever —
// subsequent fetches returned it and the box was stuck in the
// pulsing "Detecting…" overlay even after Phase 2 had locally
// cleared the flag.
export function stripTransientBoxFlags(box: EditableBox): EditableBox {
  const out: EditableBox = {
    id: box.id,
    label: box.label,
    x0: box.x0,
    y0: box.y0,
    x1: box.x1,
    y1: box.y1,
    score: box.score,
  };
  if (box.mask !== undefined) out.mask = box.mask;
  if (box.maskEdited !== undefined) out.maskEdited = box.maskEdited;
  if (box.validation !== undefined) out.validation = box.validation;
  return out;
}

type Props = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  /** Optional downscaled "display" URL shown FIRST for fast paint (e.g. a 2560px
      variant of a 4K original). The full `imageUrl` is preloaded in the
      background and swapped in once decoded, so zoom stays pixel-sharp. Null =
      load `imageUrl` directly. */
  previewUrl?: string | null;
  /** Optional BlurHash for the image. Painted instantly behind the <img> as a
      low-quality preview (with a spinner) so a large/4K original doesn't show a
      blank grey canvas while it streams in. */
  blurhash?: string | null;
  boxes: EditableBox[];
  onChange: (next: EditableBox[]) => void;
  onBoxDrawn?: (box: EditableBox) => Promise<MaskShape | null>;
  /** Auto-classify a freshly drawn box against the project's tags. Returns
      the chosen label (or null when no tag matches). When provided, the
      rename picker stays closed until classification finishes. */
  onClassifyBox?: (box: EditableBox) => Promise<{ label: string | null; score: number | null } | null>;
  /** One-shot add-box: segment + classify in a single backend call.
      When provided, BoxEditor uses THIS instead of the
      onBoxDrawn + onClassifyBox pair for the add-box gesture, saving
      one network round-trip per drag. The legacy pair stays available
      for callers (references, demo) that don't have a combined
      backend route. */
  onAddBoxDetect?: (box: EditableBox) => Promise<{
    mask?: MaskShape | null;
    label?: string | null;
    score?: number | null;
  } | null>;
  /** One-click detection: segmentation from the clicked point, then
      classification on the derived crop. Returns the full detection or
      null if nothing was found. Enables the "Click to detect" toolbar mode. */
  onPointDetect?: (point: { x: number; y: number }) => Promise<{
    box_xyxy: number[];
    mask?: MaskShape | null;
    label?: string | null;
    score?: number | null;
  } | null>;
  /** Project tags. Surfaced in the label picker so users pick from the
      project's vocabulary instead of free-typing every time. */
  projectTags?: string[];
  /** Per-label colour overrides (canonical_lower → #rrggbb). When set,
      the user's chosen colour replaces the hash-derived palette slot
      so segmentation tints + outline colours match the chip colours
      everywhere else in the app. */
  labelColours?: Record<string, string> | null;
  /** "palette" → per-id rainbow palette (default, image viewer).
      "review" → all OK boxes use a single green; rejected stays red. */
  colorMode?: "palette" | "review";
  /** Per-box size verdict against the project's target input shape. When
      provided, overrides the palette/review colour: ok = green, warn = amber,
      fail = red. Lets the workspace editor immediately show which labels
      will be detected and which won't. */
  sizeStatuses?: Record<string, "ok" | "warn" | "fail">;
  /** When true, boxes flagged as warn/fail by `sizeStatuses` render heavily
      transparent and ignore pointer events; the right-hand list shows them
      greyed and disabled. They're still in the manifest, just visually muted
      so the user can focus on viable labels. */
  muteSizeWarnings?: boolean;
  /** Size-warning filter shown in the toolbar when any box is warn/fail.
      Lifted up to the parent because the filter selects which boxes the
      parent passes back into `boxes`. */
  sizeFilter?: "all" | "hide" | "only";
  onSizeFilterChange?: (f: "all" | "hide" | "only") => void;
  /** When true, renders boxes/masks for viewing only. Disables zoom/pan,
      drawing, dragging, mask painting, deleting, and labelling. */
  readOnly?: boolean;
  /** Fires after the user renames an existing box from one non-empty
      label to another. Used to trigger the similar-label suggestion
      flow (similarity lookup), the parent decides what to do with it. */
  onLabelRenamed?: (boxId: string, oldLabel: string, newLabel: string) => void;
  /** Optional canonical → display mapping for labels. When set,
      every label render site (sidebar list, canvas chip, picker
      suggestions) shows the display value while box.label stays
      canonical. Lets a project-level rename surface inside the
      BoxEditor without changing the underlying detection records. */
  displayLabel?: (label: string) => string;
  /** When both are passed, the toolbar renders a "Size warnings"
      toggle on the right that the parent can use to flip
      sizeStatuses on/off without owning its own button. Defaults
      `undefined` (toggle hidden) so existing call sites that don't
      care keep behaving the same. */
  sizeColoringOn?: boolean;
  onSizeColoringToggle?: () => void;
  /** Optional spinner in the Labels sidebar while the per-image
      /annotations fetch is in flight. The parent sets this to true
      while the network request is outstanding; the sidebar header
      shows a small circular spinner so users know labels are
      coming, even when there are zero boxes to render yet. */
  loadingLabels?: boolean;
  /** External hover trigger. When set, the matching box is highlighted
      using the same dim-the-others spotlight as a real mouse-hover
      would produce, driven from outside the component (e.g. a parent
      detections list that wants to show "this row → that box"). The
      user's actual mouse hover takes precedence when no value is passed. */
  emphasizedBoxId?: string | null;
  /** Fires whenever the cursor enters / leaves a box on the canvas.
      Lets the parent wire keyboard shortcuts to "the box under the
      cursor" (e.g. press 1-9 to relabel the hovered box). */
  onHoverChange?: (boxId: string | null) => void;
  /** Draw a subtle pulsing glow around the "Click to detect" toolbar
      button to draw the eye to it (used by the public demo viewer). */
  glowDetect?: boolean;
};

const REVIEW_GREEN = { hue: 145, sat: 65, light: 50 };
const SIZE_RED = { hue: 0, sat: 78, light: 56 };
const SIZE_AMBER = { hue: 38, sat: 92, light: 58 };
const SIZE_GREEN = REVIEW_GREEN;

function sizeColour(status: "ok" | "warn" | "fail" | undefined): { hue: number; sat: number; light: number } | null {
  if (status === "fail") return SIZE_RED;
  if (status === "warn") return SIZE_AMBER;
  if (status === "ok") return SIZE_GREEN;
  return null;
}

type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

// Curated palette tuned for dark backgrounds, bright but not neon, evenly
// spaced around the wheel with the muddy yellow-green band skipped.
const PALETTE: { hue: number; sat: number; light: number }[] = [
  { hue: 350, sat: 82, light: 62 }, // rose
  { hue: 18, sat: 88, light: 60 },  // tangerine
  { hue: 38, sat: 92, light: 58 },  // amber
  { hue: 62, sat: 78, light: 58 },  // citron
  { hue: 145, sat: 65, light: 50 }, // emerald
  { hue: 170, sat: 70, light: 48 }, // teal
  { hue: 195, sat: 80, light: 55 }, // sky
  { hue: 218, sat: 82, light: 62 }, // azure
  { hue: 248, sat: 72, light: 68 }, // periwinkle
  { hue: 275, sat: 70, light: 66 }, // violet
  { hue: 305, sat: 72, light: 62 }, // orchid
  { hue: 328, sat: 78, light: 60 }, // magenta
];

// Tighten a freshly-drawn box to the actual the segmentation model segmentation,
// then snap any side that's hugging the image edge to the boundary.
//
// Why edge-snap: the segmentation model's mask decoder rounds off corners when a
// foreground object actually fills the frame, the corner pixels of
// the mask sit a few px inside the image bounds, leaving a visible
// gap. Anything within `snap` of an edge gets pushed to the edge
// so the resulting box covers the full extent of the object.
export function refineBoxFromMask(
  fallback: { x0: number; y0: number; x1: number; y1: number },
  mask: MaskShape | null | undefined,
  imageWidth: number,
  imageHeight: number,
): { x0: number; y0: number; x1: number; y1: number } {
  if (!mask?.polygons?.length) {
    return { x0: fallback.x0, y0: fallback.y0, x1: fallback.x1, y1: fallback.y1 };
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const poly of mask.polygons) {
    for (const pt of poly) {
      if (!pt || pt.length < 2) continue;
      const x = pt[0];
      const y = pt[1];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!isFinite(minX) || !isFinite(maxX) || maxX <= minX || maxY <= minY) {
    return { x0: fallback.x0, y0: fallback.y0, x1: fallback.x1, y1: fallback.y1 };
  }
  // 2% of the smaller image dim, with an 8 px floor, enough to
  // catch the segmentation model's rounded corners on near-full-frame objects without
  // accidentally swallowing a real 10 px gap.
  const snap = Math.max(8, Math.min(imageWidth, imageHeight) * 0.02);
  if (minX <= snap) minX = 0;
  if (minY <= snap) minY = 0;
  if (maxX >= imageWidth - snap) maxX = imageWidth;
  if (maxY >= imageHeight - snap) maxY = imageHeight;
  return {
    x0: Math.max(0, Math.floor(minX)),
    y0: Math.max(0, Math.floor(minY)),
    x1: Math.min(imageWidth, Math.ceil(maxX)),
    y1: Math.min(imageHeight, Math.ceil(maxY)),
  };
}


// Stable per-id colour so overlapping boxes/masks/labels are easy to tell
// apart. Hash the id into a palette index. Used as the fallback for
// boxes that don't have a label yet (e.g. mid-draw).
function colorFor(id: string): { hue: number; sat: number; light: number } {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return PALETTE[(h >>> 0) % PALETTE.length];
}

// Parse a #rrggbb (or 3-digit shorthand) into an HSL triple that
// matches the shape `tint()` consumes for the canvas overlay.
// Used to bridge the project-level LABEL_COLOURS palette (hex) into
// BoxEditor's HSL-based rendering pipeline.
function hexToHsl(hex: string): { hue: number; sat: number; light: number } {
  const m6 = /^#([0-9a-f]{6})$/i.exec(hex);
  let hex6: string;
  if (m6) {
    hex6 = m6[1];
  } else {
    const m3 = /^#([0-9a-f]{3})$/i.exec(hex);
    if (!m3) return { hue: 220, sat: 50, light: 60 };
    const c = m3[1];
    hex6 = c[0] + c[0] + c[1] + c[1] + c[2] + c[2];
  }
  const num = parseInt(hex6, 16);
  const r = ((num >> 16) & 255) / 255;
  const g = ((num >> 8) & 255) / 255;
  const b = (num & 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0));
    else if (max === g) h = ((b - r) / d + 2);
    else h = ((r - g) / d + 4);
    h *= 60;
  }
  return { hue: Math.round(h), sat: Math.round(s * 100), light: Math.round(l * 100) };
}

export function BoxEditor({ imageUrl, imageWidth, imageHeight, previewUrl = null, blurhash = null, boxes, onChange, onBoxDrawn, onClassifyBox, onAddBoxDetect, onPointDetect, projectTags = [], labelColours = null, colorMode = "palette", sizeStatuses, muteSizeWarnings = false, sizeFilter, onSizeFilterChange, readOnly = false, onLabelRenamed, emphasizedBoxId = null, onHoverChange, displayLabel, sizeColoringOn, onSizeColoringToggle, loadingLabels = false, glowDetect = false }: Props) {
  const displayLabelFn = displayLabel ?? ((s: string) => s);

  // Project-wide label palette, same map the workspace + tile chips
  // use, so a box's segmentation/outline tint matches its label
  // colour. Resolves canonical → hex; fallback to per-id colour for
  // boxes that don't have a project label yet (mid-draw, unsaved).
  const labelColourMap = useMemo(
    () => buildProjectLabelColourMap(projectTags, labelColours),
    [projectTags, labelColours],
  );
  const colourForBox = useCallback(
    (b: EditableBox): { hue: number; sat: number; light: number } => {
      const hex = labelColourMap.get((b.label || "").trim().toLowerCase());
      if (hex) return hexToHsl(hex);
      // Label is set but the project tag list hasn't caught it yet
      // (race: detections back from the resolver before /api/projects
      // returns its `tags` array). Fall back to a label-keyed palette
      // pick rather than the per-id colour so the box still reads as
      // its label rather than as a random hue.
      const lbl = (b.label || "").trim();
      if (lbl) return hexToHsl(colourForLabelStable(lbl));
      return colorFor(b.id);
    },
    [labelColourMap],
  );
  const svgRef = useRef<SVGSVGElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState<{ w: number; h: number } | null>(null);
  // Container box size — lets us absolutely-center the canvas wrapper so it can
  // grow with zoom (explicit width/height) instead of CSS scale(), which keeps
  // the SVG overlay (masks / boxes / labels) rendering as crisp vector at any
  // zoom rather than a bitmap-scaled, pixelated layer.
  const [containerSize, setContainerSize] = useState<{ w: number; h: number } | null>(null);
  // Full-resolution image load state — drives the blurhash preview + spinner so
  // a large/4K original never shows a blank canvas while it streams in. Reset on
  // every imageUrl change; an already-cached image reports complete at once.
  const [imgLoaded, setImgLoaded] = useState(false);
  // Progressive load: when a previewUrl is given we show it first (fast), then
  // swap to the full imageUrl once the hidden preloader below has decoded it, so
  // zoom is pixel-sharp without the 4K original blocking the first paint.
  const [fullLoaded, setFullLoaded] = useState(false);
  useEffect(() => {
    const im = imgRef.current;
    setImgLoaded(!!(im && im.complete && im.naturalWidth > 0));
    setFullLoaded(false);
  }, [imageUrl, previewUrl]);
  const displaySrc = (!previewUrl || fullLoaded) ? imageUrl : previewUrl;

  // Measure the flex container via getBoundingClientRect (reliable across all
  // CSS contexts), subtract p-2 padding (16px), then fit the image inside
  // while maintaining its exact aspect ratio.
  const computeSize = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    setContainerSize({ w: width, h: height });
    const aw = width - 16, ah = height - 16; // subtract p-2 padding each side
    if (aw <= 0 || ah <= 0) return;
    const ratio = imageWidth / imageHeight;
    let w = aw, h = aw / ratio;
    if (h > ah) { h = ah; w = ah * ratio; }
    setCanvasSize({ w: Math.floor(w), h: Math.floor(h) });
  }, [imageWidth, imageHeight]);

  // Measure before first paint so there's no single-frame gap.
  useLayoutEffect(() => { computeSize(); }, [computeSize]);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(computeSize);
    ro.observe(el);
    return () => ro.disconnect();
  }, [computeSize]);

  // ---- zoom + pan ----
  // Wheel zooms anchored at the cursor; alt-drag or middle-mouse drag pans.
  // Transform is applied to the canvas wrapper so the img + svg scale together
  // and `toLocal` (which reads img.getBoundingClientRect) keeps mapping clicks
  // to image-pixel coords correctly.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  // Declared up here (rather than below alongside the other editor state) so
  // the wheel-zoom effect can take a real dependency on it and disable itself
  // while painting, instead of racing a ref-based check.
  const [paintingId, setPaintingId] = useState<string | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const panState = useRef<{ x: number; y: number; panX: number; panY: number; pointerId: number } | null>(null);
  // Live count of fingers on the canvas. ≥2 ⇒ a pinch/two-finger-pan is
  // in progress, which must suppress (and abort) any box move/resize so
  // gestures never drag a bounding box around.
  const touchCountRef = useRef(0);
  // Refs so the stable wheel listener can read the latest values without
  // nesting setState calls (React 18 StrictMode would re-run a nested updater
  // and double the pan delta, throwing the cursor anchor off).
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;
  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);
  useEffect(() => {
    // Read-only viewers don't get to zoom/pan at all, same handler skip.
    if (readOnly) return;
    // While painting, the wheel resizes the brush in MaskPainter, don't
    // attach the zoom listener at all so it can never compete.
    if (paintingId !== null) return;
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const wrap = canvasWrapperRef.current;
      if (!wrap) return;
      // ctrl/cmd+wheel = zoom; plain wheel = pan.
      // Trackpad pinch synthesises a wheel event with ctrlKey=true, so
      // this branch covers both "mouse user holding Cmd" and "trackpad
      // user pinching" cleanly. The original wheel-always-zooms
      // behaviour made two-finger trackpad scroll feel jumpy — every
      // scroll re-anchored zoom instead of panning the image.
      if (e.ctrlKey || e.metaKey) {
        // Anchor on the canvas's *current* rendered rect so the point under the
        // cursor stays under the cursor regardless of pan or flex centering.
        const rect = wrap.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const z = zoomRef.current;
        const next = Math.max(1, Math.min(8, z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
        if (next === z) return;
        const k = next / z;
        setZoom(next);
        setPan({
          x: panRef.current.x + (1 - k) * cx,
          y: panRef.current.y + (1 - k) * cy,
        });
        return;
      }
      // Plain wheel = pan. deltaX comes from trackpad horizontal scrolls
      // and shift+wheel on a mouse; deltaY is the vertical axis. Subtract
      // because the visible image moves opposite to the scroll direction.
      setPan({
        x: panRef.current.x - e.deltaX,
        y: panRef.current.y - e.deltaY,
      });
    };
    // Native listener so we can preventDefault (React's onWheel is passive).
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [paintingId, readOnly]);

  // Touch pinch-zoom. With touch-action:none on the canvas (below) the
  // browser won't pinch-zoom the whole PAGE over the image, so we handle
  // a two-finger pinch here to zoom the viewport instead. Single-finger
  // touches fall through to the SVG (draw / click-to-detect) and never
  // pan the image. Non-passive so we can preventDefault the gesture.
  useEffect(() => {
    if (readOnly) return;
    if (paintingId !== null) return;
    const el = containerRef.current;
    if (!el) return;
    // pinch holds the gesture's start state: finger spread (dist), zoom,
    // the zoom anchor (ax/ay = start midpoint relative to the canvas), the
    // start midpoint in client space (for two-finger pan), and start pan.
    let pinch:
      | { dist: number; zoom: number; ax: number; ay: number; midX: number; midY: number; panX: number; panY: number }
      | null = null;
    const dist = (t: TouchList) =>
      Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const midpoint = (t: TouchList) => ({
      x: (t[0].clientX + t[1].clientX) / 2,
      y: (t[0].clientY + t[1].clientY) / 2,
    });
    const onStart = (e: TouchEvent) => {
      touchCountRef.current = e.touches.length;
      if (e.touches.length >= 2) {
        // A second finger landed → this is a zoom/pan, not box editing.
        // The first finger's pointerdown may have already selected a box
        // (startMove's tap-to-select), so clear everything: nothing stays
        // selected while zooming or panning.
        setSelectedId(null);
        setEditingId(null);
        setFocusedId(null);
        const rect = canvasWrapperRef.current?.getBoundingClientRect();
        const m = midpoint(e.touches);
        pinch = {
          dist: dist(e.touches), zoom: zoomRef.current,
          ax: m.x - (rect?.left ?? 0), ay: m.y - (rect?.top ?? 0),
          midX: m.x, midY: m.y,
          panX: panRef.current.x, panY: panRef.current.y,
        };
        e.preventDefault();
      }
    };
    const onMove = (e: TouchEvent) => {
      if (!pinch || e.touches.length !== 2 || pinch.dist <= 0) return;
      e.preventDefault();
      // Zoom from the finger-spread ratio, anchored on the start midpoint.
      const next = Math.max(1, Math.min(8, pinch.zoom * (dist(e.touches) / pinch.dist)));
      const k = next / pinch.zoom;
      // Two-finger pan: follow how far the midpoint has travelled.
      const m = midpoint(e.touches);
      const dx = m.x - pinch.midX;
      const dy = m.y - pinch.midY;
      setZoom(next);
      setPan({
        x: pinch.panX + (1 - k) * pinch.ax + dx,
        y: pinch.panY + (1 - k) * pinch.ay + dy,
      });
    };
    const onEnd = (e: TouchEvent) => {
      touchCountRef.current = e.touches.length;
      if (e.touches.length < 2) pinch = null;
    };
    el.addEventListener("touchstart", onStart, { passive: false });
    el.addEventListener("touchmove", onMove, { passive: false });
    el.addEventListener("touchend", onEnd);
    el.addEventListener("touchcancel", onEnd);
    return () => {
      el.removeEventListener("touchstart", onStart);
      el.removeEventListener("touchmove", onMove);
      el.removeEventListener("touchend", onEnd);
      el.removeEventListener("touchcancel", onEnd);
    };
  }, [paintingId, readOnly]);
  const onContainerPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (readOnly) return;
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      // Capture-phase + stopPropagation prevents MaskPainter and box handlers
      // from ever seeing the gesture as a paint/draw stroke.
      e.stopPropagation();
      panState.current = {
        x: e.clientX,
        y: e.clientY,
        panX: pan.x,
        panY: pan.y,
        pointerId: e.pointerId,
      };
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };
  const onContainerPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!panState.current) return;
    e.preventDefault();
    setPan({
      x: panState.current.panX + (e.clientX - panState.current.x),
      y: panState.current.panY + (e.clientY - panState.current.y),
    });
  };
  const onContainerPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (panState.current && panState.current.pointerId === e.pointerId) {
      e.currentTarget.releasePointerCapture(e.pointerId);
      panState.current = null;
    }
  };

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  // Surface real-mouse hover changes to the parent so it can wire
  // keyboard shortcuts to "the box under the cursor", e.g. press 1
  // to relabel the hovered detection. emphasizedBoxId stays an
  // input-only contract: the parent sets it for visual emphasis,
  // we report back what the user is actually pointing at.
  const onHoverChangeRef = useRef(onHoverChange);
  useEffect(() => { onHoverChangeRef.current = onHoverChange; });
  useEffect(() => { onHoverChangeRef.current?.(hoveredId); }, [hoveredId]);
  // Visual hover state == real mouse hover when no external emphasis
  // is set, falling through to the parent-controlled `emphasizedBoxId`
  // when the user isn't actively hovering. State updates (mouseEnter/
  // mouseLeave handlers) keep using `setHoveredId` directly, only the
  // visual reads below switch to this derived value.
  const effectiveHoveredId: string | null = hoveredId ?? emphasizedBoxId;
  const [drawMode, setDrawMode] = useState(false);
  const [pointMode, setPointMode] = useState(false);
  // Mobile-only: the Labels list is a dismissible bottom-sheet overlay
  // (a fixed sidebar would crush the image on a phone). Auto-hides 5s
  // after opening / last interaction; on md+ it's the static column.
  const [mobileLabelsOpen, setMobileLabelsOpen] = useState(false);
  const labelsHideTimer = useRef<number | null>(null);
  const armLabelsAutoHide = useCallback(() => {
    if (labelsHideTimer.current) window.clearTimeout(labelsHideTimer.current);
    labelsHideTimer.current = window.setTimeout(() => setMobileLabelsOpen(false), 5000);
  }, []);
  useEffect(() => {
    if (mobileLabelsOpen) armLabelsAutoHide();
    return () => { if (labelsHideTimer.current) window.clearTimeout(labelsHideTimer.current); };
  }, [mobileLabelsOpen, armLabelsAutoHide]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  // Click a row in the label list to "focus" that box, every other box
  // dims so the highlighted one is easy to read against busy detections.
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [paintMode, setPaintMode] = useState<PaintMode>("brush");
  const [brushSize, setBrushSize] = useState(40);
  const hasMasks = boxes.some((b) => (b.mask?.polygons?.length ?? 0) > 0);
  const paintingBox = paintingId ? boxes.find((b) => b.id === paintingId) ?? null : null;
  const painterHandle = useRef<MaskPainterHandle | null>(null);

  // Picker shows project tags first, then any extra labels users have typed
  // on existing boxes, gives one place to converge on a small vocabulary.
  const labelSuggestions = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of projectTags) {
      const tl = t.trim().toLowerCase();
      if (tl && !seen.has(tl)) { seen.add(tl); out.push(tl); }
    }
    for (const b of boxes) {
      const bl = (b.label || "").trim().toLowerCase();
      if (bl && bl !== "new" && !seen.has(bl)) { seen.add(bl); out.push(bl); }
    }
    return out;
  }, [projectTags, boxes]);

  // Single-step undo: snapshot the current `boxes` just before any mutation
  // and restore it on Ctrl/Cmd-Z. One level deep, enough to fix an accidental
  // drag, rename, or delete without a full undo stack.
  //
  // Snapshots are *gesture-scoped*: a drag fires onChange dozens of times per
  // second, and we only want the FIRST frame's pre-state. `canSnapshot` opens
  // a snapshot window on each pointerdown (or after an undo) and closes it
  // once the first onChange consumes it; subsequent same-gesture changes
  // just propagate without clobbering the saved pre-state.
  const undoRef = useRef<EditableBox[] | null>(null);
  const canSnapshotRef = useRef(true);
  const onChangeUndoable = useCallback((next: EditableBox[]) => {
    if (canSnapshotRef.current) {
      undoRef.current = stateRef.current.boxes;
      canSnapshotRef.current = false;
    }
    onChange(next);
  }, [onChange]);

  const stateRef = useRef({ boxes, onChange: onChangeUndoable });
  stateRef.current = { boxes, onChange: onChangeUndoable };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      const isUndo = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey;
      if (isUndo && undoRef.current) {
        e.preventDefault();
        // Apply via raw onChange so the undo doesn't itself become a snapshot.
        onChange(undoRef.current);
        undoRef.current = null;
        canSnapshotRef.current = true;
      }
    };
    window.addEventListener("keydown", onKey);
    // Each new pointerdown opens a fresh snapshot window, the next onChange
    // will save the pre-state, then close the window so the rest of the
    // gesture's updates don't overwrite it.
    const onDown = () => { canSnapshotRef.current = true; };
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [onChange]);

  // Overlay sizing is SCREEN-RELATIVE, not image-relative. SVG user-
  // space coordinates are image-space (the viewBox is set to the
  // image dimensions), so to keep stroke widths / fonts / handles a
  // constant size on screen we divide the desired screen-pixel
  // target by the current image→screen scale factor. This makes
  // overlays look identical on a 4K image and a 512 px crop, and
  // keeps them constant during pinch-zoom rather than fattening up
  // when the user zooms in.
  //   displayScale: image fitted into the container at zoom=1
  //                 (canvasSize.w / imageWidth)
  //   scaleFactor:  displayScale × user zoom, total image-px-per-
  //                 screen-px ratio currently in effect
  // Falls back to a reasonable default before the container has
  // been measured (one frame on mount); the layout effect below
  // updates canvasSize before paint so the fallback rarely renders.
  const displayScale = canvasSize ? canvasSize.w / imageWidth : 1;
  const scaleFactor = Math.max(0.001, displayScale * zoom);
  // Target screen-pixel sizes, what the user actually perceives.
  // Tuned to feel close to the previous image-relative values at a
  // typical 1080p layout while being visually consistent across
  // image sizes and zoom levels.
  const handleSize = 14 / scaleFactor;
  const strokeWidth = 2 / scaleFactor;
  const fontSize = 14 / scaleFactor;
  // Subtle rounding on boxes / chips so the overlays feel native
  // rather than engineering-tool. Same screen-relative model.
  const boxRadius = 4 / scaleFactor;
  const labelFont = '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif';

  // Use the img element's bounding rect, this is the ground truth for where
  // pixels actually render, regardless of any CSS wrapper sizing quirks.
  const toLocal = (clientX: number, clientY: number) => {
    const img = imgRef.current;
    if (!img) return { x: 0, y: 0 };
    const rect = img.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * imageWidth,
      y: ((clientY - rect.top) / rect.height) * imageHeight,
    };
  };

  const clamp = (b: EditableBox): EditableBox => ({
    ...b,
    x0: Math.max(0, Math.min(imageWidth, b.x0)),
    y0: Math.max(0, Math.min(imageHeight, b.y0)),
    x1: Math.max(0, Math.min(imageWidth, b.x1)),
    y1: Math.max(0, Math.min(imageHeight, b.y1)),
  });

  // ---- draw new box ----
  // When drawMode is on, start a new box wherever the user clicks, including
  // on top of an existing rect/polygon. Without this guard the existing-box
  // pointer handlers would steal the gesture and block drawing.
  const onSvgPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    // Read-only mode, no drawing, no point-detect, no resizing.
    if (readOnly) return;
    // Pan gesture (alt-drag or middle-mouse) → let the container handler take it.
    if (e.button === 1 || (e.button === 0 && e.altKey)) return;
    // Click-to-detect mode: one click → single detect_point request
    // → drop a finished box with geometry, mask AND label in one shot.
    // The backend's detect_point already runs segmentation + the label
    // resolver in the same call, so the FE no longer chases a separate
    // classify step (which used to race the auto-PUT and leave boxes stranded
    // mid-flight with persisted `detecting:true` flags).
    if (pointMode && onPointDetect && !editingId) {
      e.stopPropagation();
      e.preventDefault();
      const { x, y } = toLocal(e.clientX, e.clientY);
      const id = `n${Date.now()}`;
      const placeholder: EditableBox = {
        id,
        label: "detecting",
        x0: x - 1,
        y0: y - 1,
        x1: x + 1,
        y1: y + 1,
        score: null,
        detecting: { cx: x, cy: y },
      };
      onChangeUndoable([...stateRef.current.boxes, placeholder]);
      setSelectedId(id);
      setPointMode(false);
      (async () => {
        // Outer try/catch so a malformed response (missing box_xyxy
        // etc.) or any other unexpected throw drops the placeholder
        // visually AND surfaces the cause in DevTools — silent
        // failures here were what made the "second click vanishes"
        // bug so hard to track.
        try {
          let res: Awaited<ReturnType<NonNullable<typeof onPointDetect>>> = null;
          try {
            res = await onPointDetect({ x, y });
          } catch (err) {
            console.error("[click-to-detect] onPointDetect threw:", err);
            res = null;
          }
          if (!res) {
            console.warn("[click-to-detect] no result — removing placeholder", { id });
            stateRef.current.onChange(
              stateRef.current.boxes.filter((b) => b.id !== id),
            );
            return;
          }
          if (!Array.isArray(res.box_xyxy) || res.box_xyxy.length !== 4) {
            console.error("[click-to-detect] malformed response (missing box_xyxy):", res);
            stateRef.current.onChange(
              stateRef.current.boxes.filter((b) => b.id !== id),
            );
            return;
          }
          const [bx0, by0, bx1, by1] = res.box_xyxy;
          const label = res.label?.trim() || null;
          // One atomic commit. All transient flags cleared in the
          // same update that lands the final geometry + label, so
          // there's no window where the auto-PUT could snapshot an
          // in-flight box.
          stateRef.current.onChange(
            stateRef.current.boxes.map((b) => {
              if (b.id !== id) return b;
              return {
                ...b,
                x0: bx0,
                y0: by0,
                x1: bx1,
                y1: by1,
                mask: res!.mask ?? null,
                score: res!.score ?? null,
                label: label || "label",
                detecting: false,
                classifying: false,
                segmenting: false,
              };
            }),
          );
          // Resolver couldn't pick a tag — open the picker so the
          // user types one immediately.
          if (!label) setEditingId(id);
        } catch (err) {
          console.error("[click-to-detect] commit failed:", err);
          stateRef.current.onChange(
            stateRef.current.boxes.filter((b) => b.id !== id),
          );
        }
      })();
      return;
    }
    if (!drawMode || editingId) return;
    e.stopPropagation();
    e.preventDefault();
    const { x, y } = toLocal(e.clientX, e.clientY);
    const id = `n${Date.now()}`;
    const newBox: EditableBox = { id, label: "new", x0: x, y0: y, x1: x, y1: y, score: null };
    onChangeUndoable([...boxes, newBox]);
    setSelectedId(id);

    const onMove = (ev: PointerEvent) => {
      const { x: cx, y: cy } = toLocal(ev.clientX, ev.clientY);
      const next = stateRef.current.boxes.map((b) =>
        b.id === id
          ? clamp({
              ...b,
              x0: Math.min(x, cx),
              y0: Math.min(y, cy),
              x1: Math.max(x, cx),
              y1: Math.max(y, cy),
            })
          : b,
      );
      stateRef.current.onChange(next);
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      const last = stateRef.current.boxes.find((b) => b.id === id);
      if (!last || last.x1 - last.x0 < 6 || last.y1 - last.y0 < 6) {
        stateRef.current.onChange(stateRef.current.boxes.filter((b) => b.id !== id));
        setSelectedId(null);
      } else {
        const wantsCombined = Boolean(onAddBoxDetect);
        const wantsSegment = !wantsCombined && Boolean(onBoxDrawn);
        const wantsClassify = !wantsCombined && Boolean(onClassifyBox);
        // No backend handler at all: keep the user's drawn rect and
        // pop the rename picker so they can label it themselves.
        if (!wantsCombined && !wantsSegment && !wantsClassify) {
          setEditingId(id);
        } else {
          // Spinner state until the segment + classify result lands.
          // Whether we go through the combined one-shot endpoint
          // (preferred — one RTT) or the parallel pair (legacy
          // callers without a combined route), the result merges
          // into a SINGLE state commit below so the canvas never
          // reveals a half-finished box and the auto-PUT can't
          // catch transient flags mid-flight.
          stateRef.current.onChange(
            stateRef.current.boxes.map((b) =>
              b.id === id
                ? {
                    ...b,
                    segmenting: wantsCombined || wantsSegment,
                    classifying: wantsCombined || wantsClassify,
                  }
                : b,
            ),
          );
          (async () => {
            const finalBox = stateRef.current.boxes.find((b) => b.id === id);
            if (!finalBox) return;
            let mask: MaskShape | null = null;
            let label: string | null = null;
            if (wantsCombined) {
              let res: Awaited<ReturnType<NonNullable<typeof onAddBoxDetect>>> = null;
              try {
                res = await onAddBoxDetect!(finalBox);
              } catch {
                res = null;
              }
              mask = res?.mask ?? null;
              label = res?.label?.trim() || null;
            } else {
              const [segResult, clsResult] = await Promise.allSettled([
                wantsSegment ? onBoxDrawn!(finalBox) : Promise.resolve(null as MaskShape | null),
                wantsClassify ? onClassifyBox!(finalBox) : Promise.resolve(null as { label: string | null; score: number | null } | null),
              ]);
              mask = segResult.status === "fulfilled" ? (segResult.value ?? null) : null;
              const clsRes = clsResult.status === "fulfilled" ? clsResult.value : null;
              label = clsRes?.label?.trim() || null;
            }
            // Snap to mask if the segmentation model returned one; otherwise keep the
            // user's rect.
            const refined = mask
              ? refineBoxFromMask(finalBox, mask, imageWidth, imageHeight)
              : { x0: finalBox.x0, y0: finalBox.y0, x1: finalBox.x1, y1: finalBox.y1 };
            stateRef.current.onChange(
              stateRef.current.boxes.map((b) => {
                if (b.id !== id) return b;
                const next: EditableBox = {
                  ...b,
                  x0: refined.x0,
                  y0: refined.y0,
                  x1: refined.x1,
                  y1: refined.y1,
                  mask: mask ?? b.mask ?? null,
                  segmenting: false,
                  classifying: false,
                  detecting: false,
                };
                // Only patch the label when classify produced one AND
                // the user hasn't already renamed it in the meantime.
                if (label && b.label === "new") next.label = label;
                return next;
              }),
            );
            // Classifier couldn't pick a tag — open the picker.
            if (!label) setEditingId(id);
          })();
        }
      }
      setDrawMode(false);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };

  const onSvgClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      setSelectedId(null);
      setEditingId(null);
      setFocusedId(null);
    }
  };

  // ---- move existing ----
  const startMove = (e: React.PointerEvent, boxId: string) => {
    if (readOnly) return;
    if (drawMode) return;
    // A pinch / two-finger pan is in progress — never drag a box.
    if (touchCountRef.current >= 2) return;
    e.stopPropagation();
    const initial = stateRef.current.boxes.find((b) => b.id === boxId);
    if (!initial) return;
    // Touch: require the box to be SELECTED first. A press on an
    // unselected box only selects it — you then press the (now selected)
    // box and drag to move it. This stops a pan/zoom (or a stray tap)
    // from accidentally nudging a box. Mouse keeps press-and-drag.
    if (e.pointerType === "touch" && selectedId !== boxId) {
      setSelectedId(boxId);
      setFocusedId(null);
      return;
    }
    setSelectedId(boxId);
    // Clicking back on the canvas (whether on a box or background) brings
    // every other label out of the dimmed focus state.
    setFocusedId(null);
    const start = toLocal(e.clientX, e.clientY);
    const w = initial.x1 - initial.x0;
    const h = initial.y1 - initial.y0;
    // Snapshot a hand-painted mask so it can be translated to follow the
    // box during the drag (auto masks get re-segmented on drop instead).
    const initialMaskPolys =
      initial.maskEdited && initial.mask?.polygons ? initial.mask.polygons : null;

    const onMove = (ev: PointerEvent) => {
      const cur = toLocal(ev.clientX, ev.clientY);
      const dx = cur.x - start.x;
      const dy = cur.y - start.y;
      const nx0 = Math.max(0, Math.min(imageWidth - w, initial.x0 + dx));
      const ny0 = Math.max(0, Math.min(imageHeight - h, initial.y0 + dy));
      const mdx = nx0 - initial.x0;
      const mdy = ny0 - initial.y0;
      stateRef.current.onChange(
        stateRef.current.boxes.map((b) => {
          if (b.id !== boxId) return b;
          const moved = { ...b, x0: nx0, y0: ny0, x1: nx0 + w, y1: ny0 + h };
          if (initialMaskPolys && b.mask) {
            moved.mask = {
              ...b.mask,
              polygons: initialMaskPolys.map((poly) =>
                poly.map(([px, py]) => [px + mdx, py + mdy] as [number, number]),
              ),
            };
          }
          return moved;
        }),
      );
    };
    const cleanup = () => {
      document.removeEventListener("pointermove", onMoveTrack);
      document.removeEventListener("pointerup", onUp);
    };
    let moved = false;
    const onMoveTrack = (ev: PointerEvent) => {
      // A second finger landed mid-drag → it's a pinch/pan, abort the move.
      if (touchCountRef.current >= 2) { cleanup(); return; }
      moved = true;
      onMove(ev);
    };
    const onUp = () => {
      cleanup();
      if (moved) refreshMask(boxId);
    };
    document.addEventListener("pointermove", onMoveTrack);
    document.addEventListener("pointerup", onUp);
  };

  // Re-run the segmentation model for a box whose geometry just changed (drag-move or
  // handle-resize). No-op if the parent didn't supply a segmenter.
  const refreshMask = (boxId: string) => {
    if (!onBoxDrawn) return;
    const box = stateRef.current.boxes.find((b) => b.id === boxId);
    if (!box) return;
    // Never auto-overwrite a hand-painted mask: the user's silhouette is
    // the source of truth, and re-segmenting on a move/resize would
    // silently destroy it (or wipe it to null if the segmenter returns
    // nothing). On a move the polygons are translated to follow the box;
    // on a resize the painted mask is left as-is rather than discarded.
    if (box.maskEdited) return;
    stateRef.current.onChange(
      stateRef.current.boxes.map((b) => (b.id === boxId ? { ...b, segmenting: true } : b)),
    );
    (async () => {
      try {
        const mask = await onBoxDrawn(box);
        stateRef.current.onChange(
          stateRef.current.boxes.map((b) =>
            b.id === boxId ? { ...b, segmenting: false, mask: mask ?? null } : b,
          ),
        );
      } catch {
        stateRef.current.onChange(
          stateRef.current.boxes.map((b) => (b.id === boxId ? { ...b, segmenting: false } : b)),
        );
      }
    })();
  };

  // ---- resize via handle ----
  const startResize = (e: React.PointerEvent, boxId: string, handle: Handle) => {
    if (readOnly) return;
    if (drawMode) return;
    // Don't resize while a pinch / two-finger pan is in progress.
    if (touchCountRef.current >= 2) return;
    e.stopPropagation();
    const initial = stateRef.current.boxes.find((b) => b.id === boxId);
    if (!initial) return;
    setSelectedId(boxId);

    let moved = false;
    const onMove = (ev: PointerEvent) => {
      moved = true;
      const { x, y } = toLocal(ev.clientX, ev.clientY);
      let { x0, y0, x1, y1 } = initial;
      if (handle.includes("w")) x0 = Math.min(x, x1 - 6);
      if (handle.includes("e")) x1 = Math.max(x, x0 + 6);
      if (handle.includes("n")) y0 = Math.min(y, y1 - 6);
      if (handle.includes("s")) y1 = Math.max(y, y0 + 6);
      stateRef.current.onChange(
        stateRef.current.boxes.map((b) => (b.id === boxId ? clamp({ ...b, x0, y0, x1, y1 }) : b)),
      );
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      if (moved) refreshMask(boxId);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };

  const removeBox = (id: string) => {
    onChangeUndoable(boxes.filter((b) => b.id !== id));
    if (selectedId === id) setSelectedId(null);
    if (editingId === id) setEditingId(null);
    // Deleting the row that was driving focus would otherwise leave
    // the canvas in "everything dimmed except a now-deleted box"
    // mode. Drop focus so the remaining labels light back up.
    if (focusedId === id) setFocusedId(null);
    // Same problem for hover: if the user clicked the row's delete
    // button, the box is removed from the DOM before pointerLeave
    // fires, so `hoveredId` stays pointing at a now-stale id and
    // `dimForHover` keeps every surviving box at 15% opacity. Clear
    // it explicitly here.
    if (hoveredId === id) setHoveredId(null);
    if (paintingId === id) setPaintingId(null);
  };

  const renameBox = (id: string, label: string) => {
    const prev = boxes.find((b) => b.id === id);
    // Block profane labels client-side. The backend has the same
    // gate (`assert_clean` in `update_project`), but rejecting at
    // the source means the user gets immediate feedback instead of
    // an opaque 400 from the next debounced PUT.
    if (containsProfanity(label)) {
      console.warn(`[label] "${label}" contains a banned term, rename ignored.`);
      return;
    }
    onChangeUndoable(boxes.map((b) => (b.id === id ? { ...b, label } : b)));
    // Fire the rename callback so the parent can run a similarity
    // similar-label search. Skip placeholders ("new", "label",
    // "detecting") and no-op renames; we only care about a real
    // user-driven label change on an established box.
    if (
      onLabelRenamed
      && prev
      && prev.label !== label
      && label.trim() !== ""
      && !["new", "label", "detecting"].includes((prev.label || "").trim().toLowerCase())
    ) {
      onLabelRenamed(id, prev.label, label);
    }
  };

  // User overrides a validator rejection, flips the box to manually verified.
  const verifyBox = (id: string) => {
    onChangeUndoable(
      boxes.map((b) =>
        b.id === id
          ? {
              ...b,
              validation: {
                match: true,
                confidence: 1,
                reason: "manually verified",
                source: "manual",
              },
            }
          : b,
      ),
    );
  };


  // delete with backspace when a box is selected (and not editing label)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (editingId) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if ((e.key === "Backspace" || e.key === "Delete") && selectedId) {
        e.preventDefault();
        removeBox(selectedId);
      }
      if (e.key === "Escape") {
        setSelectedId(null);
        setEditingId(null);
        setDrawMode(false);
      }
      // Mode hotkeys, only fire on plain keys (no modifiers, no
      // input focus) so they don't fight with browser / OS shortcuts.
      if (!readOnly && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const k = e.key.toLowerCase();
        // M = toggle mask edit on the highlighted box.
        // Hovered wins over selected so cursor + M is enough, no
        // need to click first. Pressing M while painting closes
        // the painter (matches Esc).
        if (k === "m") {
          if (paintingId) {
            e.preventDefault();
            setPaintingId(null);
            return;
          }
          const target = hoveredId || selectedId;
          if (target) {
            e.preventDefault();
            setSelectedId(target);
            setPaintingId(target);
          }
          return;
        }
        // B = toggle draw-box mode. Mirrors the "+ Add box" button.
        // Cancels any other mode + clears selection so the canvas is
        // ready for the next click-drag.
        if (k === "b") {
          e.preventDefault();
          setDrawMode((v) => !v);
          setPointMode(false);
          setPaintingId(null);
          setSelectedId(null);
          setEditingId(null);
          return;
        }
        // X = toggle click-to-detect mode. Only when the parent
        // wired up onPointDetect, otherwise X is a no-op.
        if (k === "x" && onPointDetect) {
          e.preventDefault();
          setPointMode((v) => !v);
          setDrawMode(false);
          setPaintingId(null);
          setSelectedId(null);
          setEditingId(null);
          return;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const handlesFor = (b: EditableBox): { handle: Handle; cx: number; cy: number; cursor: string }[] => {
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    return [
      { handle: "nw", cx: b.x0, cy: b.y0, cursor: "nwse-resize" },
      { handle: "n", cx, cy: b.y0, cursor: "ns-resize" },
      { handle: "ne", cx: b.x1, cy: b.y0, cursor: "nesw-resize" },
      { handle: "e", cx: b.x1, cy, cursor: "ew-resize" },
      { handle: "se", cx: b.x1, cy: b.y1, cursor: "nwse-resize" },
      { handle: "s", cx, cy: b.y1, cursor: "ns-resize" },
      { handle: "sw", cx: b.x0, cy: b.y1, cursor: "nesw-resize" },
      { handle: "w", cx: b.x0, cy, cursor: "ew-resize" },
    ];
  };

  // Any in-flight per-box work (mask or label classification) keeps
  // the indeterminate progress bar visible. Counts feed the tooltip.
  const pendingClassify = boxes.filter((b) => b.classifying).length;
  const pendingSegment = boxes.filter((b) => b.segmenting).length;
  const busy = pendingClassify > 0 || pendingSegment > 0;
  const busyParts: string[] = [];
  if (pendingClassify > 0) busyParts.push(`labelling ${pendingClassify}`);
  if (pendingSegment > 0) busyParts.push(`segmenting ${pendingSegment}`);

  // When hovering one box, related boxes (a parent containing the hover or a
  // child contained by it) keep more presence than an unrelated box does ,
  // collapsing them all to the same dim level loses the relationship that
  // the user is trying to see. 16px slack absorbs detector wobble so a
  // helmet whose box pokes 5px outside the head is still treated as "inside".
  const HOVER_RELATION_TOL = 16;
  const hoveredBox = effectiveHoveredId ? boxes.find((b) => b.id === effectiveHoveredId) ?? null : null;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Toolbar wrapper: holds the scrollable button strip plus the
          mobile edge fades that cue horizontal scrolling. */}
      <div className="relative shrink-0">
      <div
        className="relative flex items-center justify-between gap-2 sm:gap-3 px-2 sm:px-4 py-2 sm:py-2.5 shrink-0 overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        // Themable toolbar: flat panel tint + hairline divider so the
        // strip reads against either page colour. Previously hard-coded
        // white-tint stripes that disappeared on a white surface in
        // light mode.
        style={{
          background: "var(--panel)",
          borderBottom: "1px solid var(--line)",
        }}
      >
        {busy && (
          <div
            className="absolute left-0 right-0 bottom-0 h-0.5 overflow-hidden"
            title={busyParts.join(" · ")}
          >
            <div className="indeterminate-bar h-full w-1/3 bg-[var(--accent)]" />
          </div>
        )}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          {!readOnly && (
            <button
              onClick={() => {
                setDrawMode((v) => !v);
                setPointMode(false);
                setSelectedId(null);
                setEditingId(null);
              }}
              className={[
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border",
                drawMode
                  ? "bg-[var(--accent-dim)] border-[var(--accent)] text-[var(--foreground)]"
                  : "border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]",
              ].join(" ")}
            >
              {drawMode ? "Drawing… (drag on image)" : "+ Add box"}
            </button>
          )}
          {!readOnly && onPointDetect && (
            <button
              onClick={() => {
                setPointMode((v) => !v);
                setDrawMode(false);
                setSelectedId(null);
                setEditingId(null);
              }}
              className={[
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border",
                pointMode
                  ? "bg-[var(--accent-dim)] border-[var(--accent)] text-[var(--foreground)]"
                  : "border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]",
                // Subtle pulsing glow to draw the eye to click-to-detect
                // in the public demo (off everywhere else). Suppressed
                // once the mode is active so it doesn't compete with the
                // active-state styling.
                glowDetect && !pointMode ? "demo-detect-glow" : "",
              ].join(" ")}
              title="Click anywhere on an object to auto-detect it"
            >
              {pointMode ? "Click an object…" : "✨ Click to detect"}
            </button>
          )}
          <OverlayToggle label="Boxes" on={showBoxes} setOn={setShowBoxes} />
          <OverlayToggle label="Labels" on={showLabels} setOn={setShowLabels} />
          <OverlayToggle label="Masks" on={showMasks} setOn={setShowMasks} disabled={!hasMasks} />
          {sizeFilter !== undefined && onSizeFilterChange && Object.values(sizeStatuses ?? {}).some((s) => s !== "ok") && (
            <div className="inline-flex rounded-md border border-[var(--line)] p-0.5">
              {([
                ["all", "All", "Show every box, regardless of size."],
                ["hide", "Hide warned", "Hide boxes too small or borderline at the current input shape."],
                ["only", "Only warned", "Show only boxes flagged as too small or borderline."],
              ] as const).map(([val, lbl, ttl]) => (
                <button
                  key={val}
                  onClick={() => onSizeFilterChange(val)}
                  title={ttl}
                  className={[
                    "rounded-[4px] px-3 py-1 text-[11px] font-medium uppercase tracking-wider transition-colors duration-150",
                    sizeFilter === val
                      ? "bg-[var(--surface-2)] text-[var(--foreground)]"
                      : "text-[var(--fg-dim)] hover:text-[var(--foreground)]",
                  ].join(" ")}
                >
                  {lbl}
                </button>
              ))}
            </div>
          )}
          {!readOnly && selectedId && !paintingId && (
            <button
              onClick={() => setPaintingId(selectedId)}
              className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]"
              title="Paint or erase pixels on this box's mask"
            >
              ✎ Edit mask
            </button>
          )}
          {!readOnly && paintingId && (
            // Desktop keeps the inline paint controls in the toolbar.
            // Mobile uses the dedicated bottom paint bar over the canvas
            // (below) so brush / size / done are thumb-reachable instead
            // of buried in the horizontally-scrolling toolbar.
            <div className="hidden md:flex items-center">
              <PaintControls
                mode={paintMode}
                setMode={setPaintMode}
                size={brushSize}
                setSize={setBrushSize}
              />
            </div>
          )}
        </div>
        <div className="text-xs text-[var(--muted)] flex items-center gap-2 sm:gap-3 shrink-0">
          {/* Size-warning toggle, explicit "Size warnings" pill so
              the user can tell what it does without hover. Sits at
              the right end of the toolbar, same row as Add box /
              Click to detect. Defaults ON in the parent so the
              traffic-light tint shows on first paint. */}
          {onSizeColoringToggle && (
            <button
              type="button"
              onClick={onSizeColoringToggle}
              aria-pressed={!!sizeColoringOn}
              title={
                sizeColoringOn
                  ? "Hide red/amber size warnings on small boxes"
                  : "Highlight boxes that are too small to detect once the image is downscaled"
              }
              className={[
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border inline-flex items-center gap-1.5",
                sizeColoringOn
                  ? "bg-[var(--surface-2)] border-[var(--line-strong)] text-[var(--foreground)]"
                  : "border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]",
              ].join(" ")}
            >
              {/* Three-stripe traffic-light glyph */}
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <rect x="9" y="3" width="6" height="18" rx="2" />
                <circle cx="12" cy="7.5" r="1.2" fill="currentColor" stroke="none" />
                <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
                <circle cx="12" cy="16.5" r="1.2" fill="currentColor" stroke="none" />
              </svg>
              Size warnings
            </button>
          )}
          {busy && (
            <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--accent)]">
              {busyParts.join(" · ")}
            </span>
          )}
          {/* Hotkey legend, only the BoxEditor-owned keys. The
              parent (DatasetViewer / ReferenceImageEditor) renders
              its own legend for nav + relabel digits in its header
              bar, so we don't double up here. */}
          {!readOnly && (
            <div className="hidden md:flex items-center gap-1.5">
              <HotkeyChip k="B" label="Add box" />
              {onPointDetect && <HotkeyChip k="X" label="Click-detect" />}
              <HotkeyChip k="M" label="Edit mask" />
            </div>
          )}
          <span className="tabular-nums">
            {boxes.length} box{boxes.length === 1 ? "" : "es"}
          </span>
        </div>
      </div>
      {/* Edge fades (mobile only) — a smooth gradient from the page
          background to transparent so it's obvious the button strip
          scrolls. Wide + no backdrop-blur so there's no hard edge. */}
      <div aria-hidden className="md:hidden pointer-events-none absolute inset-y-0 left-0 w-14 z-10 bg-gradient-to-r from-[var(--background)] via-[var(--background)]/70 to-transparent" />
      <div aria-hidden className="md:hidden pointer-events-none absolute inset-y-0 right-0 w-14 z-10 bg-gradient-to-l from-[var(--background)] via-[var(--background)]/70 to-transparent" />
      </div>

      <div className="relative flex-1 min-h-0 flex flex-row">
        <div
          ref={containerRef}
          className="relative select-none flex items-center justify-center min-h-0 p-2 flex-1 min-w-0 overflow-hidden"
          onPointerDownCapture={onContainerPointerDown}
          onPointerMove={onContainerPointerMove}
          onPointerUp={onContainerPointerUp}
          onPointerCancel={onContainerPointerUp}
          onDoubleClick={() => {
            if (paintingId || drawMode || pointMode) return;
            resetView();
          }}
          // touch-action:none → the browser won't pan/zoom the PAGE on
          // touch over the canvas (so the image can't be dragged and a
          // pinch zooms the viewport via our handler, not the whole page).
          style={{ cursor: panState.current ? "grabbing" : undefined, touchAction: "none" }}
        >
          {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
            <button
              onClick={resetView}
              // Themable reset-zoom chip: solid modal surface instead
              // of a translucent black tint so the chip is visible on
              // a light canvas. Border + text use tokens so contrast
              // is maintained across themes.
              className="absolute top-3 right-3 z-10 px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider tabular-nums rounded-md bg-[var(--modal-surface)] border border-[var(--line)] text-[var(--fg-muted)] shadow-[var(--shadow-soft)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)] transition-colors duration-150"
              title="Reset zoom / pan"
            >
              {Math.round(zoom * 100)}% · reset
            </button>
          )}
          {/* Loading wheel while the full-resolution image decodes. Sits in the
              container (not the zoomed wrapper) so it stays a fixed size + centred
              regardless of zoom, over the blurhash preview. */}
          {!imgLoaded && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center" aria-hidden>
              <span className="h-9 w-9 animate-spin rounded-full border-[3px] border-white/30 border-t-white drop-shadow-[0_1px_3px_rgba(0,0,0,0.5)]" />
            </div>
          )}
          <div
            ref={canvasWrapperRef}
            className="relative"
            style={{
              // Grow the wrapper to the zoomed size (explicit width/height) and
              // absolutely-center it, instead of CSS scale(zoom). Both the <img>
              // and the SVG fill it, so they stay perfectly aligned — but the
              // SVG now rasterizes its vector content (masks / boxes / labels)
              // at the TRUE zoomed resolution every frame, so the overlay never
              // pixelates and label text holds its on-screen size. Only the
              // raster <img> behind it scales (and may pixelate, which is fine).
              // The cursor-anchored zoom math (pan += (1-k)*cx) is unchanged:
              // the wrapper still scales from its top-left, just via size not
              // transform. Pan stays a translate so it never rasterizes.
              ...(canvasSize && containerSize
                ? {
                    position: "absolute" as const,
                    left: (containerSize.w - canvasSize.w) / 2,
                    top: (containerSize.h - canvasSize.h) / 2,
                    width: canvasSize.w * zoom,
                    height: canvasSize.h * zoom,
                  }
                : canvasSize
                  ? { width: canvasSize.w, height: canvasSize.h, flexShrink: 0 }
                  : { width: "100%", aspectRatio: `${imageWidth} / ${imageHeight}` }),
              transform: `translate(${pan.x}px, ${pan.y}px)`,
              transformOrigin: "0 0",
            }}
          >
            {/* Instant low-quality preview behind the image: the BlurHash paints
                immediately so a large/4K original streams in over a colour
                approximation instead of a blank grey canvas. Drops out once the
                full image has decoded. */}
            {blurhash && !imgLoaded && (
              <BlurhashCanvas
                hash={blurhash}
                width={48}
                height={Math.max(1, Math.round(48 * (imageHeight / Math.max(1, imageWidth))))}
                className="absolute inset-0 h-full w-full"
              />
            )}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imgRef}
              src={displaySrc}
              alt=""
              width={imageWidth}
              height={imageHeight}
              decoding="async"
              onLoad={() => setImgLoaded(true)}
              className={`absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-200 ${imgLoaded ? "opacity-100" : "opacity-0"}`}
            />
            {/* Hidden preloader for the full-resolution original. Once it decodes
                we flip displaySrc to it (already cached, so the swap is seamless)
                and zoom becomes pixel-sharp. */}
            {previewUrl && !fullLoaded && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageUrl}
                alt=""
                aria-hidden
                decoding="async"
                className="hidden"
                onLoad={() => setFullLoaded(true)}
              />
            )}
            <svg
          ref={svgRef}
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full"
          style={{ cursor: drawMode || pointMode ? "crosshair" : "default" }}
          onPointerDown={onSvgPointerDown}
          onClick={onSvgClick}
        >
          {/* Spotlight dim: when hovering a box, render a translucent
              black rect over the image with a hole punched out for that
              box's polygon(s). Z-stack stays the same, this sits at the
              very bottom of the SVG, above the <img> but below every
              existing box / mask / label, so brightness on the hovered
              segmentation is preserved while everything else fades. */}
          {(() => {
            if (!effectiveHoveredId) return null;
            const hovered = boxes.find((b) => b.id === effectiveHoveredId);
            if (!hovered) return null;
            const polys = hovered.mask?.polygons ?? [];
            const maskId = `dim-spotlight-${effectiveHoveredId}`;
            return (
              <>
                <defs>
                  <mask id={maskId} maskUnits="userSpaceOnUse" x={0} y={0} width={imageWidth} height={imageHeight}>
                    {/* white = dim visible; black = dim hidden (=
                        bright). Outside the hovered polygon stays
                        dimmed; inside the hovered polygon goes bright. */}
                    <rect x={0} y={0} width={imageWidth} height={imageHeight} fill="white" />
                    {polys.length > 0 ? (
                      polys.map((poly, pi) => (
                        <polygon
                          key={pi}
                          points={poly.map((p) => `${p[0]},${p[1]}`).join(" ")}
                          fill="black"
                        />
                      ))
                    ) : (
                      // Fallback to the bbox if no mask polygons exist on
                      // this box (manually-drawn boxes pre-segmentation).
                      <rect
                        x={hovered.x0}
                        y={hovered.y0}
                        width={Math.max(0, hovered.x1 - hovered.x0)}
                        height={Math.max(0, hovered.y1 - hovered.y0)}
                        fill="black"
                      />
                    )}
                  </mask>
                </defs>
                <rect
                  x={0}
                  y={0}
                  width={imageWidth}
                  height={imageHeight}
                  fill="black"
                  opacity={0.6}
                  mask={`url(#${maskId})`}
                  pointerEvents="none"
                />
              </>
            );
          })()}
          <defs>
            {/* Soft drop-shadow under label chips so they read as floating UI
                rather than a stamp on the image. userSpaceOnUse keeps the
                blur radius constant in image-space regardless of where the
                chip is. */}
            <filter id="bx-chip-shadow" x="-20%" y="-20%" width="140%" height="160%">
              <feDropShadow dx={0} dy={fontSize * 0.08} stdDeviation={fontSize * 0.10} floodColor="#000" floodOpacity={0.4} />
            </filter>
            {/* Selection glow so the active box "lifts" without slamming a
                stark white outline on top of the colour palette. */}
            <filter id="bx-select-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation={strokeWidth * 1.6} result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {[...boxes]
            // Largest area first → painted underneath, so smaller boxes layer
            // on top and stay hoverable. Hover/selection only changes styling,
            // never z-order, promoting a big box to the top would make any
            // smaller boxes inside it un-hoverable for the rest of the session.
            .sort((a, b) => {
              const aArea = (a.x1 - a.x0) * (a.y1 - a.y0);
              const bArea = (b.x1 - b.x0) * (b.y1 - b.y0);
              return bArea - aArea;
            })
            .map((b) => {
            const isSelected = b.id === selectedId;
            const isHovered = b.id === effectiveHoveredId;
            const isActive = isSelected || isHovered;
            const w = b.x1 - b.x0;
            const h = b.y1 - b.y0;

            const sizeC = sizeStatuses ? sizeColour(sizeStatuses[b.id]) : null;
            const c = sizeC ?? (colorMode === "review" ? REVIEW_GREEN : colourForBox(b));
            const rejected = b.validation && b.validation.match === false;
            // Muted box = small/too-small while the user has the "hide small"
            // toggle on. Render very faint and stop hover/click so it's there
            // but inert.
            const sizeStatus = sizeStatuses?.[b.id];
            const isMuted = muteSizeWarnings && (sizeStatus === "warn" || sizeStatus === "fail");
            const tint = (l: number, a: number) => `hsla(${c.hue}, ${c.sat}%, ${l}%, ${a})`;
            // Default stroke a touch firmer than before so non-selected
            // boxes still read clearly at rest. Hover/select bring it
            // up further but the at-rest state is now obvious without
            // shouting over the photo.
            const baseStroke = tint(c.light, 0.92);
            const hoverStroke = tint(Math.min(c.light + 14, 80), 0.97);
            const selectStroke = tint(Math.min(c.light + 18, 82), 1);
            const stroke = isSelected ? selectStroke : isHovered ? hoverStroke : baseStroke;
            const fillRect = isSelected
              ? tint(Math.min(c.light + 18, 80), 0.12)
              : isHovered
              ? tint(Math.min(c.light + 18, 80), 0.08)
              // At rest: a barely-there wash so the box reads as a
              // tinted region instead of just a wireframe, easier
              // to see at a glance, still well below hover/select.
              : tint(Math.min(c.light + 12, 78), 0.04);
            const sw = isActive ? strokeWidth * 1.3 : strokeWidth;
            // Cap radius so a thin/short box doesn't render as a near-circle.
            const cornerRadius = Math.min(boxRadius, w * 0.4, h * 0.4);

            // Hide the saved polygons for the box that's currently being
            // painted, the painter's canvas overlay is the truth in-flight,
            // and showing both makes the live edits hard to read.
            const isPainting = paintingId === b.id;
            const polys = !isPainting && showMasks ? (b.mask?.polygons ?? []) : [];
            const maskFill = isSelected
              ? "rgb(var(--foreground-rgb) / 0.28)"
              : isHovered
              ? tint(c.light, 0.45)
              : tint(c.light, 0.30);
            const maskStroke = isSelected ? "#fff" : tint(c.light, 0.9);

            const labelText = (rejected ? "⚠ " : "") + displayLabelFn(b.label || "label");
            // Approx 0.6em per char for ui-monospace; tighter pad than before so
            // the chip wraps the text cleanly. Cap very long labels.
            const charW = fontSize * 0.6;
            const padX = fontSize * 0.35;
            const maxChars = Math.max(8, Math.floor((w + fontSize * 4) / charW));
            const display = labelText.length > maxChars ? labelText.slice(0, maxChars - 1) + "…" : labelText;
            const labelW = display.length * charW + padX * 2;
            const labelH = fontSize * 1.3;
            // Chip text colour. When the chip is SELECTED its fill becomes the
            // foreground colour, so the text must flip to the background colour
            // or it's dark-on-dark (the "black label rectangle, can't read it"
            // bug). Rejected chips are red, so white reads on them.
            const chipTextFill = rejected
              ? "#fff"
              : isSelected
              ? "rgb(var(--background-rgb))"
              : c.light < 50 ? "#fff" : "#0a0a0a";
            // Three placement options for the chip:
            //   above , the default, when there's room above the box.
            //   below , when the box hugs the top of the image.
            //   inside, full-screen / very tall boxes where neither
            //            edge has room. We tuck the chip into the
            //            top-left interior of the box instead of
            //            letting it run off-canvas.
            const labelFitsAbove = b.y0 >= labelH;
            const labelFitsBelow = (imageHeight - b.y1) >= labelH;
            const labelInside = !labelFitsAbove && !labelFitsBelow;
            const labelBelow = !labelFitsAbove && labelFitsBelow;
            // Inside case: chip top sits ON the box's top edge so the
            // chip colour bleeds into the top border, it reads as a
            // thickened section of the border, not a floating pill.
            const labelY = labelInside
              ? b.y0
              : labelBelow
              ? b.y1
              : b.y0 - labelH;
            // "tabbed" = render box + chip as one continuous outer
            // silhouette. Only when the chip attaches to an external
            // edge (above / below). The inside case has its own render
            // path: full rounded box + a chip drawn against the box's
            // top-left interior so it looks like part of the border.
            const tabbed = showBoxes && showLabels && !labelInside;
            const insideChip = showBoxes && showLabels && labelInside;
            // Chip corner radius, match the box's so the silhouette looks
            // like a single rounded shape with a tab cut out, not two
            // shapes glued together. Capped to fit the chip dimensions.
            const chipR = Math.min(cornerRadius, labelH / 2, labelW / 2);

            const onEnter = () => setHoveredId(b.id);
            const onLeave = () => setHoveredId((h) => (h === b.id ? null : h));

            const rejectStroke = "rgba(255,80,80,1)";
            const rejectFill = "rgba(248,113,113,0.18)";
            const rejectMaskFill = "rgba(248,113,113,0.30)";

            // Dim every other box while a single one is in the spotlight ,
            // applies in five cases: drawing a fresh box, focusing a row
            // from the sidebar list, editing a mask in the painter, the
            // click-to-detect mode, or hovering one specific box.
            // The hovered/active box stays bright; everything else fades
            // back so the focused segmentation is unambiguous.
            const dimForDraw = drawMode && b.id !== selectedId;
            const dimForFocus = focusedId !== null && b.id !== focusedId;
            const dimForPaint = paintingId !== null && b.id !== paintingId;
            const dimForPoint = pointMode;
            const dimForHover = effectiveHoveredId !== null && b.id !== effectiveHoveredId;
            // Related = b fully contains the hovered box, or is fully
            // contained by it (with HOVER_RELATION_TOL slack on each side).
            // Those boxes stay much more visible so a parent/child pair
            // still reads as related rather than the parent vanishing.
            const isRelatedToHover = !!hoveredBox && b.id !== effectiveHoveredId && (
              (b.x0 >= hoveredBox.x0 - HOVER_RELATION_TOL &&
               b.y0 >= hoveredBox.y0 - HOVER_RELATION_TOL &&
               b.x1 <= hoveredBox.x1 + HOVER_RELATION_TOL &&
               b.y1 <= hoveredBox.y1 + HOVER_RELATION_TOL)
              ||
              (hoveredBox.x0 >= b.x0 - HOVER_RELATION_TOL &&
               hoveredBox.y0 >= b.y0 - HOVER_RELATION_TOL &&
               hoveredBox.x1 <= b.x1 + HOVER_RELATION_TOL &&
               hoveredBox.y1 <= b.y1 + HOVER_RELATION_TOL)
            );
            // Muted boxes (small/won't-detect under "hide small") are
            // normally rendered very faintly so they're still in the
            // scene without distracting from the viable labels. But
            // hovering or selecting one, usually via the right-hand
            // list, bumps it up to fully visible so the user can
            // actually see what they're pointing at.
            const groupOpacity = isMuted
              ? (isHovered || isSelected ? 0.9 : 0.12)
              : dimForDraw || dimForFocus || dimForPaint || dimForPoint
              ? 0.08
              : dimForHover
              ? (isRelatedToHover ? 0.55 : 0.15)
              : 1;

            // Click-to-detect in flight. Mask renders below with a soft
            // halo once SAM finishes; the chip uses the exact same sizing
            // and colours as a normal label chip so it never feels like a
            // separate UI from the rest of the editor.
            if (b.detecting) {
              const cx = typeof b.detecting === "object" ? b.detecting.cx : (b.x0 + b.x1) / 2;
              const cy = typeof b.detecting === "object" ? b.detecting.cy : (b.y0 + b.y1) / 2;
              const labelText = b.classifying ? "Labelling…" : "Detecting…";
              const dispW = labelText.length * charW + padX * 2;
              const polysAvailable = (b.mask?.polygons?.length ?? 0) > 0;
              // Screen-relative so the pulsing detect dot stays a constant size
              // on screen instead of ballooning as you zoom in.
              const dotR = 5 / scaleFactor;
              // Anchor the chip just above the bbox once we have one
              // (matches normal label placement); fall back to the click
              // point while we're still waiting on SAM.
              const chipX = polysAvailable ? b.x0 : cx;
              const chipY = polysAvailable
                ? b.y0 < labelH
                  ? b.y0 + strokeWidth
                  : b.y0 - labelH
                : cy - labelH / 2;
              return (
                <g key={b.id} opacity={groupOpacity} pointerEvents="none">
                  {polysAvailable && showMasks &&
                    (b.mask?.polygons ?? []).map((pts, pi) => (
                      <polygon
                        key={pi}
                        points={pts.map((p) => `${p[0]},${p[1]}`).join(" ")}
                        fill={tint(c.light, 0.22)}
                        stroke={tint(c.light, 0.95)}
                        strokeWidth={strokeWidth * 1.1}
                        // Matched-colour halo so the freshly-found mask pops.
                        // Blur radius is screen-relative (÷ scaleFactor) so the
                        // glow doesn't thicken into a fat border when zoomed in.
                        style={{ filter: `drop-shadow(0 0 ${(5 / scaleFactor).toFixed(2)}px ${tint(c.light, 0.7)})` }}
                      >
                        <animate attributeName="stroke-opacity" values="0.6;1;0.6" dur="1.2s" repeatCount="indefinite" />
                      </polygon>
                    ))}
                  {!polysAvailable && (
                    <circle cx={cx} cy={cy} r={dotR} fill={tint(c.light, 1)}>
                      <animate attributeName="r" values={`${dotR};${dotR * 2.6};${dotR}`} dur="1s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
                    </circle>
                  )}
                  {/* Same shape, padding, font and palette as a normal
                      label chip, only the text content differs. */}
                  <g filter="url(#bx-chip-shadow)">
                    <rect
                      x={chipX}
                      y={chipY}
                      width={dispW}
                      height={labelH}
                      rx={labelH * 0.5}
                      ry={labelH * 0.5}
                      fill={tint(c.light, 0.92)}
                    />
                    <text
                      x={chipX + padX + labelH * 0.15}
                      y={chipY + labelH * 0.72}
                      fontSize={fontSize * 0.92}
                      fontFamily={labelFont}
                      fontWeight={500}
                      letterSpacing={fontSize * -0.005}
                      fill={chipTextFill}
                    >
                      {labelText}
                    </text>
                  </g>
                </g>
              );
            }

            return (
              <g key={b.id} opacity={groupOpacity} pointerEvents={isMuted ? "none" : undefined}>
                {rejected && (
                  // Soft outer glow so the rejected box pops even when small.
                  <rect
                    x={b.x0 - sw * 2}
                    y={b.y0 - sw * 2}
                    width={w + sw * 4}
                    height={h + sw * 4}
                    fill="none"
                    stroke="rgba(255,80,80,0.35)"
                    strokeWidth={sw * 3}
                    pointerEvents="none"
                  />
                )}
                {polys.map((pts, pi) => (
                  <polygon
                    key={pi}
                    points={pts.map((p) => `${p[0]},${p[1]}`).join(" ")}
                    fill={rejected ? rejectMaskFill : maskFill}
                    stroke={rejected ? rejectStroke : maskStroke}
                    strokeWidth={sw * 0.6}
                    style={{ cursor: readOnly ? "default" : drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                    pointerEvents={pointMode ? "none" : undefined}
                    onPointerEnter={onEnter}
                    onPointerLeave={onLeave}
                    onPointerDown={(e) => startMove(e, b.id)}
                  />
                ))}
                {/* Unified box + chip silhouette. The chip cuts a tab out
                    of the box's top edge (or bottom, when there's no room
                    above), one path traces the entire combined outline
                    in a single closed loop, so there's never a doubled
                    border or visible seam. Three layers, in render order:

                    1. Box interior (transparent or hover/select wash)
                      , also the pointer-event surface for hover/drag.
                    2. Chip interior (opaque colour), non-interactive.
                    3. Combined outline + dash + select glow on top.

                    Text renders last so it always sits above the chip
                    fill regardless of how the silhouette is built. */}
                {tabbed ? (
                  <>
                    <path
                      d={
                        labelBelow
                          ? `M ${b.x0 + cornerRadius},${b.y0} L ${b.x1 - cornerRadius},${b.y0} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x1},${b.y0 + cornerRadius} L ${b.x1},${b.y1} L ${b.x0},${b.y1} L ${b.x0},${b.y0 + cornerRadius} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x0 + cornerRadius},${b.y0} Z`
                          : `M ${b.x0},${b.y0} L ${b.x1},${b.y0} L ${b.x1},${b.y1 - cornerRadius} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x1 - cornerRadius},${b.y1} L ${b.x0 + cornerRadius},${b.y1} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x0},${b.y1 - cornerRadius} Z`
                      }
                      fill={rejected ? rejectFill : fillRect === "transparent" ? "transparent" : fillRect}
                      stroke="none"
                      style={{ cursor: readOnly ? "default" : drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                      pointerEvents={pointMode || isMuted ? "none" : "all"}
                      onPointerEnter={onEnter}
                      onPointerLeave={onLeave}
                      onPointerDown={(e) => startMove(e, b.id)}
                    />
                    <path
                      d={
                        labelBelow
                          ? `M ${b.x0},${labelY} L ${b.x0 + labelW},${labelY} L ${b.x0 + labelW},${labelY + labelH - chipR} A ${chipR},${chipR} 0 0 1 ${b.x0 + labelW - chipR},${labelY + labelH} L ${b.x0 + chipR},${labelY + labelH} A ${chipR},${chipR} 0 0 1 ${b.x0},${labelY + labelH - chipR} Z`
                          : `M ${b.x0},${labelY + chipR} A ${chipR},${chipR} 0 0 1 ${b.x0 + chipR},${labelY} L ${b.x0 + labelW - chipR},${labelY} A ${chipR},${chipR} 0 0 1 ${b.x0 + labelW},${labelY + chipR} L ${b.x0 + labelW},${labelY + labelH} L ${b.x0},${labelY + labelH} Z`
                      }
                      fill={
                        rejected
                          ? "rgba(248,113,113,0.94)"
                          : isSelected
                          ? "rgb(var(--foreground-rgb) / 0.96)"
                          : isHovered
                          ? tint(Math.min(c.light + 8, 78), 0.96)
                          : tint(c.light, 0.92)
                      }
                      stroke="none"
                      pointerEvents="none"
                    />
                    <path
                      d={
                        labelBelow
                          ? `M ${b.x0 + cornerRadius},${b.y0} L ${b.x1 - cornerRadius},${b.y0} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x1},${b.y0 + cornerRadius} L ${b.x1},${b.y1} L ${b.x0 + labelW},${b.y1} L ${b.x0 + labelW},${labelY + labelH - chipR} A ${chipR},${chipR} 0 0 1 ${b.x0 + labelW - chipR},${labelY + labelH} L ${b.x0 + chipR},${labelY + labelH} A ${chipR},${chipR} 0 0 1 ${b.x0},${labelY + labelH - chipR} L ${b.x0},${b.y0 + cornerRadius} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x0 + cornerRadius},${b.y0} Z`
                          : `M ${b.x0 + chipR},${labelY} L ${b.x0 + labelW - chipR},${labelY} A ${chipR},${chipR} 0 0 1 ${b.x0 + labelW},${labelY + chipR} L ${b.x0 + labelW},${b.y0} L ${b.x1},${b.y0} L ${b.x1},${b.y1 - cornerRadius} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x1 - cornerRadius},${b.y1} L ${b.x0 + cornerRadius},${b.y1} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x0},${b.y1 - cornerRadius} L ${b.x0},${labelY + chipR} A ${chipR},${chipR} 0 0 1 ${b.x0 + chipR},${labelY} Z`
                      }
                      fill="none"
                      stroke={rejected ? rejectStroke : stroke}
                      strokeWidth={rejected ? sw * 1.5 : sw}
                      strokeLinejoin="round"
                      strokeDasharray={b.segmenting ? `${sw * 3} ${sw * 2}` : undefined}
                      filter={isSelected ? "url(#bx-select-glow)" : undefined}
                      pointerEvents="none"
                    />
                    <text
                      x={b.x0 + padX + labelH * 0.15}
                      y={labelY + labelH * 0.72}
                      fontSize={fontSize * 0.92}
                      fontFamily={labelFont}
                      fontWeight={500}
                      letterSpacing={fontSize * -0.005}
                      fill={chipTextFill}
                      pointerEvents="none"
                    >
                      {display}
                    </text>
                  </>
                ) : insideChip ? (
                  /* Inside case, full-screen / very tall box where
                     neither edge has room for the chip. We render the
                     box as a normal rounded rect, then drop a chip
                     path against the top-left interior whose top and
                     left edges sit ON the box's stroke. Same fill
                     colour as the tabbed chips, no separate stroke or
                     drop shadow, so the chip reads as a thicker patch
                     of the border (label baked into the frame) rather
                     than a floating pill. */
                  <>
                    <rect
                      x={b.x0}
                      y={b.y0}
                      width={w}
                      height={h}
                      rx={cornerRadius}
                      ry={cornerRadius}
                      fill={rejected ? rejectFill : fillRect}
                      stroke={rejected ? rejectStroke : stroke}
                      strokeWidth={rejected ? sw * 1.5 : sw}
                      strokeDasharray={b.segmenting ? `${sw * 3} ${sw * 2}` : undefined}
                      filter={isSelected ? "url(#bx-select-glow)" : undefined}
                      style={{ cursor: readOnly ? "default" : drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                      pointerEvents={pointMode || isMuted ? "none" : "all"}
                      onPointerEnter={onEnter}
                      onPointerLeave={onLeave}
                      onPointerDown={(e) => startMove(e, b.id)}
                    />
                    <path
                      // Top-left follows the box's corner curve, top
                      // and left edges sit on the box border, top-right
                      // is square (the colour just stops, blending into
                      // the top stroke), bottom corners are rounded ,
                      // the only "free" edges inside the box.
                      d={`M ${b.x0},${b.y0 + cornerRadius} A ${cornerRadius},${cornerRadius} 0 0 1 ${b.x0 + cornerRadius},${b.y0} L ${b.x0 + labelW},${b.y0} L ${b.x0 + labelW},${b.y0 + labelH - chipR} A ${chipR},${chipR} 0 0 1 ${b.x0 + labelW - chipR},${b.y0 + labelH} L ${b.x0 + chipR},${b.y0 + labelH} A ${chipR},${chipR} 0 0 1 ${b.x0},${b.y0 + labelH - chipR} Z`}
                      fill={
                        rejected
                          ? "rgba(248,113,113,0.94)"
                          : isSelected
                          ? "rgb(var(--foreground-rgb) / 0.96)"
                          : isHovered
                          ? tint(Math.min(c.light + 8, 78), 0.96)
                          : tint(c.light, 0.92)
                      }
                      stroke="none"
                      pointerEvents="none"
                    />
                    <text
                      x={b.x0 + padX + labelH * 0.15}
                      y={b.y0 + labelH * 0.72}
                      fontSize={fontSize * 0.92}
                      fontFamily={labelFont}
                      fontWeight={500}
                      letterSpacing={fontSize * -0.005}
                      fill={chipTextFill}
                      pointerEvents="none"
                    >
                      {display}
                    </text>
                  </>
                ) : (
                  <>
                    {showBoxes && (
                      <rect
                        x={b.x0}
                        y={b.y0}
                        width={w}
                        height={h}
                        rx={cornerRadius}
                        ry={cornerRadius}
                        fill={rejected ? rejectFill : fillRect}
                        stroke={rejected ? rejectStroke : stroke}
                        strokeWidth={rejected ? sw * 1.5 : sw}
                        strokeDasharray={b.segmenting ? `${sw * 3} ${sw * 2}` : undefined}
                        filter={isSelected ? "url(#bx-select-glow)" : undefined}
                        style={{ cursor: readOnly ? "default" : drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                        pointerEvents={pointMode ? "none" : undefined}
                        onPointerEnter={onEnter}
                        onPointerLeave={onLeave}
                        onPointerDown={(e) => startMove(e, b.id)}
                      />
                    )}
                    {showLabels && (
                      <g style={{ pointerEvents: "none" }} filter="url(#bx-chip-shadow)">
                        <rect
                          x={b.x0}
                          y={labelY}
                          width={labelW}
                          height={labelH}
                          rx={labelH * 0.5}
                          ry={labelH * 0.5}
                          fill={
                            rejected
                              ? "rgba(248,113,113,0.94)"
                              : isSelected
                              ? "rgb(var(--foreground-rgb) / 0.96)"
                              : isHovered
                              ? tint(Math.min(c.light + 8, 78), 0.96)
                              : tint(c.light, 0.92)
                          }
                        />
                        <text
                          x={b.x0 + padX + labelH * 0.15}
                          y={labelY + labelH * 0.72}
                          fontSize={fontSize * 0.92}
                          fontFamily={labelFont}
                          fontWeight={500}
                          letterSpacing={fontSize * -0.005}
                          fill={chipTextFill}
                        >
                          {display}
                        </text>
                      </g>
                    )}
                  </>
                )}
                {!readOnly && isSelected &&
                  handlesFor(b).map((h) => (
                    <rect
                      key={h.handle}
                      x={h.cx - handleSize / 2}
                      y={h.cy - handleSize / 2}
                      width={handleSize}
                      height={handleSize}
                      rx={handleSize * 0.3}
                      ry={handleSize * 0.3}
                      fill="rgb(var(--foreground-rgb) / 0.97)"
                      stroke={tint(c.light, 0.9)}
                      strokeWidth={strokeWidth * 0.6}
                      filter="url(#bx-chip-shadow)"
                      style={{ cursor: h.cursor }}
                      onPointerDown={(e) => startResize(e, b.id, h.handle)}
                    />
                  ))}
              </g>
            );
          })}
        </svg>
          {paintingBox && (
              <MaskPainter
                ref={painterHandle}
                // Re-mount per box so the canvas seeds fresh with that box's
                // polygons; without this, switching boxes would keep the
                // previous box's painted bitmap.
                key={paintingBox.id}
                imageUrl={imageUrl}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                initialPolygons={paintingBox.mask?.polygons ?? []}
                // Paint anywhere on the image, the box snaps to the
                // mask's bounds on save, so there's no point boxing
                // the user in. Used to be a 20% padding around the
                // existing box, which was too tight when the
                // detector under-shot.
                clip={{ x0: 0, y0: 0, x1: imageWidth, y1: imageHeight }}
                brushSize={brushSize}
                setBrushSize={setBrushSize}
                mode={paintMode}
                onSave={(polygons) => {
                  // Recompute the bounding box from the mask itself
                  //, start from infinity so erasing a chunk shrinks
                  // the box, and adding outside the old box grows
                  // it. Falls back to the original bounds only when
                  // every polygon got erased (so the box doesn't
                  // collapse to a point).
                  const pb = paintingBox;
                  let nx0 = Infinity, ny0 = Infinity, nx1 = -Infinity, ny1 = -Infinity;
                  for (const poly of polygons) {
                    for (const [px, py] of poly) {
                      if (px < nx0) nx0 = px;
                      if (py < ny0) ny0 = py;
                      if (px > nx1) nx1 = px;
                      if (py > ny1) ny1 = py;
                    }
                  }
                  if (!isFinite(nx0) || !isFinite(nx1)) {
                    nx0 = pb.x0; ny0 = pb.y0; nx1 = pb.x1; ny1 = pb.y1;
                  }
                  onChangeUndoable(
                    boxes.map((b) =>
                      b.id === pb.id
                        ? { ...b, mask: { polygons }, maskEdited: true, x0: nx0, y0: ny0, x1: nx1, y1: ny1 }
                        : b,
                    ),
                  );
                  setPaintingId(null);
                }}
                onCancel={() => setPaintingId(null)}
              />
          )}
          </div>
          {paintingBox && (
            // Desktop paint pill (Done/Cancel + keyboard hint). Lives in
            // the un-transformed container so zoom/pan don't move it.
            // Mobile gets the bottom paint bar instead (keyboard hints are
            // meaningless on touch), so this is hidden below md.
            <div
              className="absolute top-3 left-1/2 -translate-x-1/2 z-20 hidden md:flex items-center gap-2 rounded-lg px-2 py-1 text-xs"
              style={{
                // Themable: flat modal surface + hairline border so the
                // paint toolbar floats as a quiet flat panel in either
                // theme instead of a frosted gradient pill.
                background: "var(--modal-surface)",
                border: "1px solid var(--line)",
                boxShadow: "var(--shadow-strong)",
              }}
            >
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--fg-dim)] px-2">
                Painting · enter to save · esc to cancel
              </span>
              <button
                onClick={() => painterHandle.current?.commit()}
                className="rounded-md bg-[var(--accent)] text-[var(--accent-contrast)] px-3 py-1 font-medium transition-opacity duration-150 hover:opacity-90"
              >
                Done
              </button>
              <button
                onClick={() => setPaintingId(null)}
                className="rounded-md border border-[var(--line)] px-3 py-1 text-[var(--fg-muted)] transition-colors duration-150 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]"
              >
                Cancel
              </button>
            </div>
          )}

          {/* Mobile paint bar — thumb-reachable Brush/Eraser + Size +
              Done/Cancel pinned to the bottom of the canvas. Replaces the
              desktop top-pill + toolbar controls, which are hard to reach
              one-handed and reference keyboard shortcuts that don't exist
              on touch. */}
          {!readOnly && paintingBox && (
            <div className="md:hidden absolute inset-x-2 bottom-2 z-40 rounded-lg border border-[var(--line)] bg-[var(--modal-surface)] shadow-[var(--shadow-strong)] px-3 py-2.5 flex flex-col gap-2.5">
              <div className="flex items-center gap-2">
                <div className="flex flex-1 rounded-md border border-[var(--line)] p-0.5">
                  <button
                    type="button"
                    onClick={() => setPaintMode("brush")}
                    className={[
                      "flex-1 rounded-[4px] py-2 text-sm font-medium transition-colors touch-manipulation",
                      paintMode === "brush" ? "bg-[var(--accent-dim)] text-[var(--foreground)]" : "text-[var(--fg-muted)]",
                    ].join(" ")}
                  >
                    Brush
                  </button>
                  <button
                    type="button"
                    onClick={() => setPaintMode("eraser")}
                    className={[
                      "flex-1 rounded-[4px] py-2 text-sm font-medium transition-colors touch-manipulation",
                      paintMode === "eraser" ? "bg-[var(--accent-dim)] text-[var(--foreground)]" : "text-[var(--fg-muted)]",
                    ].join(" ")}
                  >
                    Eraser
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => setPaintingId(null)}
                  className="rounded-md border border-[var(--line)] px-4 py-2 text-sm text-[var(--fg-muted)] active:scale-95 transition-transform touch-manipulation shrink-0"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => painterHandle.current?.commit()}
                  className="rounded-md bg-[var(--accent)] text-[var(--accent-contrast)] px-4 py-2 text-sm font-semibold active:scale-95 transition-transform touch-manipulation shrink-0"
                >
                  Done
                </button>
              </div>
              <div className="flex items-center gap-3">
                <span className="pk-micro shrink-0">Size</span>
                <input
                  type="range"
                  min={4}
                  max={200}
                  step={1}
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="flex-1 h-6 accent-[var(--accent)] touch-manipulation"
                  aria-label="Brush size"
                />
                <span className="font-mono text-xs text-[var(--fg-muted)] tabular-nums w-8 text-right shrink-0">{brushSize}</span>
              </div>
            </div>
          )}
        </div>

        <div
          onPointerDown={armLabelsAutoHide}
          className={[
            // Mobile: dismissible bottom-sheet overlay. Sits at the bottom
            // with side margins so it covers the image as little as
            // possible, blurred + translucent so the image reads through.
            "flex flex-col min-h-0 absolute inset-x-2 bottom-2 z-30 max-h-[42%] rounded-lg border border-[var(--line)] bg-[rgb(var(--surface-rgb)/0.94)] backdrop-blur-md shadow-[var(--shadow-strong)] overflow-hidden",
            mobileLabelsOpen ? "flex" : "hidden",
            // Desktop (md+): the original static right-hand column.
            "md:flex md:static md:inset-auto md:bottom-auto md:z-auto md:max-h-none md:w-72 md:shrink-0 md:rounded-none md:border-y-0 md:border-r-0 md:border-l md:border-[var(--line)] md:bg-[var(--panel)] md:backdrop-blur-none md:shadow-none md:overflow-visible",
          ].join(" ")}
        >
          <div
            className="px-4 py-2.5 pk-micro shrink-0 flex items-center justify-between gap-2"
            style={{ borderBottom: "1px solid var(--line-soft)" }}
          >
            <span>Labels · {boxes.length}</span>
            <div className="flex items-center gap-2">
              {/* Small circular spinner while /annotations is in
                  flight for this image. Stays visible even when boxes
                  land before masks do, so the user sees the
                  "geometry still arriving" cue rather than a static
                  box count next to a half-rendered image. */}
              {loadingLabels && (
                <span
                  aria-label="Loading labels"
                  className="h-3 w-3 rounded-full border-2 border-foreground/15 border-t-foreground/60 animate-spin"
                />
              )}
              {/* Mobile-only close — desktop has no dismiss (always shown). */}
              <button
                type="button"
                onClick={() => setMobileLabelsOpen(false)}
                aria-label="Hide labels"
                className="md:hidden -mr-1 h-6 w-6 grid place-items-center rounded-full text-foreground/50 hover:text-foreground hover:bg-foreground/10 transition-colors"
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                </svg>
              </button>
            </div>
          </div>
          <ul className="flex-1 min-h-0 overflow-y-auto divide-y divide-foreground/[0.04]">
        {boxes.length === 0 && loadingLabels && (
          <li className="px-4 py-6 flex items-center gap-2 text-xs text-[var(--muted)]">
            <span
              aria-hidden
              className="h-3 w-3 rounded-full border-2 border-foreground/15 border-t-foreground/60 animate-spin"
            />
            Loading labels…
          </li>
        )}
        {boxes.length === 0 && !loadingLabels && (
          <li className="px-4 py-3 text-xs text-[var(--muted)]">No boxes. Click + Add box to draw one.</li>
        )}
        {boxes.map((b) => {
          const sizeC = sizeStatuses ? sizeColour(sizeStatuses[b.id]) : null;
          const c = sizeC ?? (colorMode === "review" ? REVIEW_GREEN : colourForBox(b));
          const sizeStatus = sizeStatuses?.[b.id];
          const isMuted = muteSizeWarnings && (sizeStatus === "warn" || sizeStatus === "fail");
          // Hover still works on muted rows, that's the whole point
          // of "hover the row to peek at the hidden box on canvas".
          // Selection stays gated, since the click handlers are
          // disabled for muted rows.
          const isHovered = effectiveHoveredId === b.id;
          const isSelected = selectedId === b.id && !isMuted;
          // Use the box colour for the row tint so the link between canvas
          // and list is unmistakable. Selected wins over hover.
          const rowBg = isSelected
            ? `hsla(${c.hue}, ${c.sat}%, ${c.light}%, 0.18)`
            : isHovered
            ? `hsla(${c.hue}, ${c.sat}%, ${c.light}%, 0.10)`
            : undefined;
          const rowBorder = isSelected
            ? `inset 3px 0 0 hsla(${c.hue}, ${c.sat}%, ${c.light}%, 0.95)`
            : isHovered
            ? `inset 3px 0 0 hsla(${c.hue}, ${c.sat}%, ${c.light}%, 0.6)`
            : undefined;
          return (
          <li
            key={b.id}
            ref={(el) => {
              if (el && isHovered) {
                el.scrollIntoView({ block: "nearest", behavior: "smooth" });
              }
            }}
            className={[
              "flex items-center gap-3 px-4 py-2 text-xs transition-colors",
              isMuted ? "opacity-40" : "",
            ].join(" ")}
            style={{ background: rowBg, boxShadow: rowBorder }}
            onMouseEnter={() => setHoveredId(b.id)}
            onMouseLeave={() => setHoveredId((h) => (h === b.id ? null : h))}
          >
            <span
              className="h-3 w-3 rounded-sm shrink-0 border border-foreground/10"
              style={{ backgroundColor: `hsl(${c.hue}, ${c.sat}%, ${c.light}%)` }}
            />
            {editingId === b.id && !readOnly && !isMuted ? (
              <LabelPicker
                initial={b.label}
                suggestions={labelSuggestions}
                displayLabel={displayLabelFn}
                onCommit={(label) => {
                  renameBox(b.id, label || "label");
                  setEditingId(null);
                }}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <>
                <button
                  onClick={() => { if (!isMuted) setFocusedId((cur) => (cur === b.id ? null : b.id)); }}
                  onDoubleClick={() => { if (!readOnly && !isMuted) setEditingId(b.id); }}
                  disabled={isMuted}
                  className={[
                    "flex-1 text-left font-mono",
                    isMuted
                      ? "cursor-not-allowed text-foreground/55"
                      : readOnly ? "cursor-default" : "hover:underline underline-offset-2",
                    focusedId === b.id ? "text-[var(--foreground)]" : "",
                  ].join(" ")}
                  title={isMuted
                    ? `${sizeStatus === "fail" ? "Won't detect" : "Small"} at the current input shape, toggle off "Hide small" to edit.`
                    : readOnly ? undefined : "click to focus · double-click to rename"}
                >
                  {displayLabelFn(b.label)}
                </button>
                {!readOnly && !isMuted && <button
                  onClick={() => setEditingId(b.id)}
                  className="text-[var(--muted)] hover:text-foreground px-1"
                  aria-label="rename"
                  title="rename"
                >
                  ✎
                </button>}
              </>
            )}
            {b.validation && b.validation.match === false && (() => {
              const reasonTitle = `Rejected${b.validation.reason ? `: ${b.validation.reason}` : ""}.`;
              // Stacked chip: bold verdict on top, tiny AI footer
              // below. Footer is small enough that the chip stays
              // compact next to the label name without crowding.
              const Chip = (
                <span className="flex flex-col items-end leading-[1] gap-[1px]">
                  <span>Rejected</span>
                  <span className="text-[6.5px] font-medium opacity-60 uppercase tracking-[0.18em]">
                    AI
                  </span>
                </span>
              );
              return readOnly ? (
                <span
                  className="inline-flex items-center rounded bg-[color-mix(in_srgb,var(--bad)_12%,transparent)] border border-[var(--bad)] text-[var(--bad)] px-2 py-1 text-[10px] uppercase tracking-wider"
                  title={reasonTitle}
                >
                  {Chip}
                </span>
              ) : (
                <button
                  onClick={() => verifyBox(b.id)}
                  className="group inline-flex items-center gap-1 rounded bg-[color-mix(in_srgb,var(--bad)_12%,transparent)] border border-[var(--bad)] text-[var(--bad)] hover:bg-[color-mix(in_srgb,var(--ok)_12%,transparent)] hover:border-[var(--ok)] hover:text-[var(--ok)] px-2 py-1 text-[10px] uppercase tracking-wider transition-colors"
                  title={`${reasonTitle} Click to mark as verified.`}
                >
                  <span className="group-hover:hidden">{Chip}</span>
                  <span className="hidden group-hover:inline">Verify ✓</span>
                </button>
              );
            })()}
            {b.validation && b.validation.match && b.validation.kind === "unsure" && (
              <span
                className="inline-flex items-center rounded bg-[color-mix(in_srgb,var(--warn)_12%,transparent)] border border-[var(--warn)] text-[var(--warn)] px-2 py-1 text-[10px] uppercase tracking-wider"
                title={b.validation.reason || "Resolver flagged this detection as ambiguous, worth eyeballing."}
              >
                Unsure
              </span>
            )}
            {b.validation && b.validation.match && b.validation.kind !== "unsure" && b.validation.confidence > 0 && (
              b.validation.source === "cascade" ? (
                // Label Cascade provenance, accent chip, distinct
                // from the green Verified/Manual chip so it's obvious
                // at a glance which boxes were promoted via the
                // Cascade modal vs reviewed by AI Review or
                // hand-verified.
                <span
                  className="inline-flex items-center rounded border border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--accent)] px-2 py-1 text-[10px] uppercase tracking-wider"
                  title={`Label applied via Label Cascade${b.validation.reason ? `, ${b.validation.reason}` : ""}.`}
                >
                  <span className="flex flex-col items-end leading-[1] gap-[1px]">
                    <span>Cascade</span>
                    <span className="text-[6.5px] font-medium opacity-70 uppercase tracking-[0.18em]">
                      Label
                    </span>
                  </span>
                </span>
              ) : (
                <span
                  className="inline-flex items-center rounded bg-[color-mix(in_srgb,var(--ok)_12%,transparent)] border border-[var(--ok)] text-[var(--ok)] px-2 py-1 text-[10px] uppercase tracking-wider"
                  title={`AI ${(b.validation.confidence * 100).toFixed(0)}%${b.validation.reason ? `, ${b.validation.reason}` : ""}`}
                >
                  {/* Same stacked-chip shape as the rejection so
                      verified and rejected boxes line up vertically
                      in the list. The "AI" footer disappears for
                      manual verifications, those didn't go through
                      the model. */}
                  <span className="flex flex-col items-end leading-[1] gap-[1px]">
                    <span>{b.validation.source === "manual" ? "Manual" : "Verified"}</span>
                    {b.validation.source !== "manual" && (
                      <span className="text-[6.5px] font-medium opacity-60 uppercase tracking-[0.18em]">
                        AI
                      </span>
                    )}
                  </span>
                </span>
              )
            )}
            {(b.segmenting || b.classifying) && (
              <span className="font-mono text-[var(--muted)] tabular-nums">…</span>
            )}
            {!readOnly && !isMuted && (
              <button
                onClick={() => removeBox(b.id)}
                className="rounded-full h-6 w-6 grid place-items-center hover:bg-foreground/10"
                aria-label="delete"
                title="delete"
              >
                ×
              </button>
            )}
          </li>
          );
        })}
      </ul>
        </div>

        {/* Mobile-only floating toggle to reveal the Labels sheet.
            Hidden on desktop (the column is always shown), while the
            sheet is open, and while painting (the bottom paint bar
            occupies that spot). */}
        {!mobileLabelsOpen && !paintingId && (
          <button
            type="button"
            onClick={() => setMobileLabelsOpen(true)}
            className="md:hidden absolute bottom-3 left-1/2 -translate-x-1/2 z-20 inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-[var(--modal-surface)] px-3.5 py-1.5 text-xs font-medium text-[var(--fg-soft)] shadow-[var(--shadow-strong)] active:scale-95 transition-transform touch-manipulation"
          >
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" />
              <line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" />
            </svg>
            Labels · {boxes.length}
          </button>
        )}
      </div>
    </div>
  );
}

function HotkeyChip({ k, label }: { k: string; label: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded border border-[var(--line)] px-1.5 py-0.5 text-[10px] tabular-nums leading-none"
      title={`${label} (press ${k})`}
    >
      <kbd className="font-mono text-[var(--fg-soft)] px-1 py-[1px] rounded-[3px] bg-[var(--surface-2)] leading-none">
        {k}
      </kbd>
      <span className="text-[var(--fg-dim)]">{label}</span>
    </span>
  );
}


function OverlayToggle({
  label,
  on,
  setOn,
  disabled,
}: {
  label: string;
  on: boolean;
  setOn: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={() => setOn(!on)}
      className={[
        "rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border",
        disabled
          ? "border-[var(--line)] text-[var(--fg-faint)] opacity-50 cursor-not-allowed"
          : on
          // Active state: quiet filled chip, surface fill + stronger
          // hairline, foreground text. Reads as "on" in both themes
          // without spending accent on a view toggle.
          ? "bg-[var(--surface-2)] border-[var(--line-strong)] text-[var(--foreground)]"
          : "border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]",
      ].join(" ")}
      title={`${label}: ${on ? "on" : "off"}`}
    >
      {label}: {on ? "on" : "off"}
    </button>
  );
}

function PaintControls({
  mode,
  setMode,
  size,
  setSize,
}: {
  mode: PaintMode;
  setMode: (m: PaintMode) => void;
  size: number;
  setSize: (n: number) => void;
}) {
  return (
    <div className="flex items-center gap-1 rounded-md border border-[var(--line)] px-1.5 py-1">
      <button
        onClick={() => setMode("brush")}
        className={[
          "rounded-[4px] px-2.5 py-1 text-xs font-medium transition-colors duration-150 ease-out",
          mode === "brush" ? "bg-[var(--accent-dim)] text-[var(--foreground)]" : "text-[var(--fg-dim)] hover:text-[var(--foreground)]",
        ].join(" ")}
        title="Brush, paint into the mask"
      >
        Brush
      </button>
      <button
        onClick={() => setMode("eraser")}
        className={[
          "rounded-[4px] px-2.5 py-1 text-xs font-medium transition-colors duration-150 ease-out",
          mode === "eraser" ? "bg-[var(--accent-dim)] text-[var(--foreground)]" : "text-[var(--fg-dim)] hover:text-[var(--foreground)]",
        ].join(" ")}
        title="Eraser, remove pixels from the mask"
      >
        Eraser
      </button>
      <span className="pk-micro pl-1">Size</span>
      <input
        type="range"
        min={4}
        max={200}
        step={1}
        value={size}
        onChange={(e) => setSize(Number(e.target.value))}
        className="w-24 accent-[var(--accent)]"
      />
      <span className="font-mono text-[10px] text-[var(--fg-dim)] tabular-nums w-8 text-right">{size}</span>
    </div>
  );
}

function LabelPicker({
  initial,
  suggestions,
  displayLabel,
  onCommit,
  onCancel,
}: {
  initial: string;
  suggestions: string[];
  displayLabel?: (label: string) => string;
  onCommit: (label: string) => void;
  onCancel: () => void;
}) {
  const displayLabelFn = displayLabel ?? ((s: string) => s);
  // Editing happens in display-space, the user sees and types
  // renamed labels, but suggestions are still keyed on canonical
  // so a matched pick commits the canonical (preserving the
  // backend-search invariant). Display the canonical's alias as
  // the initial input value.
  const [query, setQuery] = useState(displayLabelFn(initial));
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const committed = useRef(false);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const blocked = useMemo(() => containsProfanity(query), [query]);

  // Reverse map display → canonical: any suggestion whose display
  // form (case-insensitive) matches the input picks the canonical.
  // Falls through to the raw query for brand-new labels.
  const canonicalForDisplay = (input: string): string => {
    const lower = input.trim().toLowerCase();
    for (const s of suggestions) {
      if (displayLabelFn(s).trim().toLowerCase() === lower) return s;
      if (s.trim().toLowerCase() === lower) return s;
    }
    return input.trim();
  };

  // Click outside the picker commits whatever's typed (unless it's
  // a banned label, then the click-out is treated as a cancel so
  // the box keeps its old label and the user gets visible feedback
  // before the next keystroke).
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target as Node)) {
        if (!committed.current) {
          committed.current = true;
          if (blocked) onCancel();
          else onCommit(canonicalForDisplay(query));
        }
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, blocked, onCommit, onCancel]);

  const q = query.trim().toLowerCase();
  // Filter on EITHER display or canonical, keystrokes match both
  // representations so the suggestion list feels natural after a
  // rename.
  const filtered = q
    ? suggestions.filter((s) =>
        s.toLowerCase().includes(q)
        || displayLabelFn(s).toLowerCase().includes(q),
      )
    : suggestions;
  const showCreate = q.length > 0 && !suggestions.some(
    (s) => s.toLowerCase() === q || displayLabelFn(s).toLowerCase() === q,
  );

  const items = [...filtered, ...(showCreate ? [`__create:${q}`] : [])];

  const commit = (raw: string, isCreate = false) => {
    if (committed.current) return;
    if (containsProfanity(raw)) return;
    committed.current = true;
    // Brand-new label: commit verbatim (the parent's onLabelRenamed
    // / onAddProjectLabel path adds it to project.tags). Otherwise
    // map back to canonical so the underlying detection record
    // stays consistent with what the resolver / labelling job sees.
    const out = isCreate ? raw.trim() : canonicalForDisplay(raw);
    onCommit(out);
  };

  return (
    <div ref={wrapRef} className="relative flex-1">
      <input
        ref={inputRef}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setHighlight(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setHighlight((h) => Math.min(items.length - 1, h + 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setHighlight((h) => Math.max(0, h - 1));
          } else if (e.key === "Enter") {
            e.preventDefault();
            const picked = items[highlight];
            if (picked && picked.startsWith("__create:")) {
              commit(picked.slice("__create:".length), true);
            } else if (picked) {
              commit(picked, false);
            } else {
              commit(query, false);
            }
          } else if (e.key === "Escape") {
            e.preventDefault();
            committed.current = true;
            onCancel();
          } else if (e.key === "Tab") {
            commit(query, false);
          }
        }}
        placeholder="pick a label or type a new one"
        className={[
          // Solid theme-aware surface + explicit text colour: previously
          // bg-transparent with no colour, so over a dark image the typed
          // label was unreadable.
          "w-full rounded border px-2 py-1 outline-none font-mono text-xs bg-[var(--background)] text-[var(--foreground)] placeholder:text-[var(--fg-faint)]",
          blocked
            ? "border-[var(--bad)] focus:border-[var(--bad)]"
            : "border-[var(--line-strong)] focus:border-[var(--accent)]",
        ].join(" ")}
        aria-invalid={blocked ? true : undefined}
      />
      {blocked && (
        <p className="absolute left-0 right-0 top-full mt-1 text-[10px] text-[var(--bad)] z-30">
          That label can&rsquo;t be used.
        </p>
      )}
      {!blocked && items.length > 0 && (
        <ul
          role="listbox"
          className="absolute left-0 right-0 top-full mt-1 max-h-56 overflow-y-auto rounded-md border border-[var(--line)] bg-[var(--panel-solid)] shadow-[var(--shadow-strong)] z-20 py-1"
        >
          {items.map((it, i) => {
            const isCreate = it.startsWith("__create:");
            const rawValue = isCreate ? it.slice("__create:".length) : it;
            // List rows show the renamed display label even for
            // canonical suggestions; the create row shows whatever
            // the user typed.
            const text = isCreate ? rawValue : displayLabelFn(rawValue);
            return (
              <li
                key={it}
                role="option"
                aria-selected={i === highlight}
                onMouseEnter={() => setHighlight(i)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  commit(rawValue, isCreate);
                }}
                className={[
                  "px-2 py-1 text-xs font-mono cursor-pointer flex items-center gap-2",
                  i === highlight ? "bg-[var(--surface-hover)] text-[var(--foreground)]" : "text-[var(--fg-muted)]",
                ].join(" ")}
              >
                {isCreate ? (
                  <>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--ok)]">New</span>
                    <span>{text}</span>
                  </>
                ) : (
                  <span>{text}</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function cleanLabel(raw: string, tags: string[]): string {
  const lower = raw.toLowerCase().trim();
  const matched = tags.filter((t) =>
    new RegExp(`\\b${escapeRegExp(t.toLowerCase())}\\b`).test(lower),
  );
  if (matched.length > 0) return matched.join(" + ");
  const stripped = lower
    .split(/\s+/)
    .filter((tok) => tok !== "a" && tok !== "an")
    .join(" ")
    .trim();
  return stripped || raw;
}

export function detectionsToBoxes(
  detections: {
    label: string;
    score: number | null;
    box_xyxy: number[];
    mask?: MaskShape | null;
    validation?: Validation | null;
  }[],
  tags: string[] = [],
): EditableBox[] {
  return detections.map((d, i) => ({
    id: `d${i}`,
    label: cleanLabel(d.label, tags),
    x0: d.box_xyxy[0],
    y0: d.box_xyxy[1],
    x1: d.box_xyxy[2],
    y1: d.box_xyxy[3],
    score: d.score,
    mask: d.mask ?? null,
    validation: d.validation ?? null,
  }));
}
