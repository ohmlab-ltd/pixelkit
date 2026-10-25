"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { BlurhashCanvas } from "react-blurhash";
import { MaskPainter, type MaskPainterHandle, type PaintMode } from "./MaskPainter";
import { buildProjectLabelColourMap, colourForLabelStable } from "./v2/labelColours";

export type MaskShape = { polygons: number[][][] };

export type Validation = {
  match: boolean;
  confidence: number;
  reason: string;
  model?: string | null;
  // auto: the validator pass. manual: verified by hand.
  // cascade: applied by Label Cascade.
  source?: "auto" | "manual" | "cascade";
  // Stage tag for the rejection chip. "unsure" marks a kept but ambiguous label.
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
  /** Mask edited by hand in MaskPainter; a box move or resize keeps it. */
  maskEdited?: boolean;
  segmenting?: boolean;
  classifying?: boolean;
  /** One-click detection in flight, with the clicked point. */
  detecting?: { cx: number; cy: number } | boolean;
  validation?: Validation | null;
};

// Copy of a box without the UI-only flags, which must not reach the manifest.
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

/** `boxes` with masks from `source` filled in by id, for boxes that have
    none and have not moved. */
export function withMasksFrom(
  boxes: EditableBox[],
  source: EditableBox[] | null | undefined,
): EditableBox[] {
  if (!source?.length) return boxes;
  const byId = new Map(source.map((b) => [b.id, b]));
  let changed = false;
  const out = boxes.map((b) => {
    if (b.mask) return b;
    const s = byId.get(b.id);
    if (!s?.mask || s.x0 !== b.x0 || s.y0 !== b.y0 || s.x1 !== b.x1 || s.y1 !== b.y1) return b;
    changed = true;
    return { ...b, mask: s.mask, ...(s.maskEdited !== undefined ? { maskEdited: s.maskEdited } : {}) };
  });
  return changed ? out : boxes;
}

type Props = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  /** Downscaled URL shown until the full `imageUrl` has decoded. */
  previewUrl?: string | null;
  /** BlurHash painted behind the image while it loads. */
  blurhash?: string | null;
  boxes: EditableBox[];
  onChange: (next: EditableBox[]) => void;
  onBoxDrawn?: (box: EditableBox) => Promise<MaskShape | null>;
  /** Classifies a drawn box against the project tags. Label is null when no tag matches. */
  onClassifyBox?: (box: EditableBox) => Promise<{ label: string | null; score: number | null } | null>;
  /** Segments and classifies a drawn box in one call. Takes precedence over
      onBoxDrawn and onClassifyBox. */
  onAddBoxDetect?: (box: EditableBox) => Promise<{
    mask?: MaskShape | null;
    label?: string | null;
    score?: number | null;
  } | null>;
  /** Detects the object at a clicked point; null when nothing is found.
      Enables the "Click to detect" mode. */
  onPointDetect?:(point: { x: number; y: number }) => Promise<{
    box_xyxy: number[];
    mask?: MaskShape | null;
    label?: string | null;
    score?: number | null;
  } | null>;
  /** Offered in the label picker. */
  projectTags?: string[];
  /** Per-label colour overrides, lowercase canonical label -> #rrggbb. */
  labelColours?: Record<string, string> | null;
  /** Per-box size verdict against the project's input shape. Overrides the label colour. */
  sizeStatuses?: Record<string, "ok" | "warn" | "fail">;
  /** Fires when a box is renamed from one non-empty label to another. */
  onLabelRenamed?: (boxId: string, oldLabel: string, newLabel: string) => void;
  /** Canonical label -> displayed label. box.label stays canonical. */
  displayLabel?: (label: string) => string;
  /** The toolbar's "Size warnings" toggle, shown when the handler is set. */
  sizeColoringOn?: boolean;
  onSizeColoringToggle?: () => void;
  /** Shows a spinner in the Labels sidebar while /annotations loads. */
  loadingLabels?: boolean;
  /** Box highlighted as if hovered. A mouse hover takes precedence. */
  emphasizedBoxId?: string | null;
  /** Fires when the cursor enters or leaves a box on the canvas. */
  onHoverChange?: (boxId: string | null) => void;
};

const SIZE_RED = { hue: 0, sat: 78, light: 56 };
const SIZE_AMBER = { hue: 38, sat: 92, light: 58 };
const SIZE_GREEN = { hue: 145, sat: 65, light: 50 };

function sizeColour(status: "ok" | "warn" | "fail" | undefined): { hue: number; sat: number; light: number } | null {
  if (status === "fail") return SIZE_RED;
  if (status === "warn") return SIZE_AMBER;
  if (status === "ok") return SIZE_GREEN;
  return null;
}

type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

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

// Bounding box of the mask. A side within `snap` px of an image edge moves onto it.
function refineBoxFromMask(
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


// Palette colour hashed from the box id, for boxes without a label.
function colorFor(id: string): { hue: number; sat: number; light: number } {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return PALETTE[(h >>> 0) % PALETTE.length];
}

// #rrggbb or #rgb to the HSL triple that tint() takes.
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

export function BoxEditor({ imageUrl, imageWidth, imageHeight, previewUrl = null, blurhash = null, boxes, onChange, onBoxDrawn, onClassifyBox, onAddBoxDetect, onPointDetect, projectTags = [], labelColours = null, sizeStatuses, onLabelRenamed, emphasizedBoxId = null, onHoverChange, displayLabel, sizeColoringOn, onSizeColoringToggle, loadingLabels = false }: Props) {
  const displayLabelFn = displayLabel ?? ((s: string) => s);

  const labelColourMap = useMemo(
    () => buildProjectLabelColourMap(projectTags, labelColours),
    [projectTags, labelColours],
  );
  const colourForBox = useCallback(
    (b: EditableBox): { hue: number; sat: number; light: number } => {
      const hex = labelColourMap.get((b.label || "").trim().toLowerCase());
      if (hex) return hexToHsl(hex);
      // A label not in the project tags yet.
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
  // For centring the canvas wrapper.
  const [containerSize, setContainerSize] = useState<{ w: number; h: number } | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  // True once the hidden preloader has decoded the full imageUrl.
  const [fullLoaded, setFullLoaded] = useState(false);
  useEffect(() => {
    const im = imgRef.current;
    setImgLoaded(!!(im && im.complete && im.naturalWidth > 0));
    setFullLoaded(false);
  }, [imageUrl, previewUrl]);
  const displaySrc = (!previewUrl || fullLoaded) ? imageUrl : previewUrl;

  // Fits the image inside the container, keeping its aspect ratio.
  const computeSize = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    setContainerSize({ w: width, h: height });
    const aw = width - 16, ah = height - 16; // p-2 padding on both sides
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

  // Zoom and pan. Ctrl or Cmd + wheel zooms at the cursor; wheel, alt-drag
  // and middle-drag pan.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  // Declared before the wheel effect, which depends on it.
  const [paintingId, setPaintingId] = useState<string | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const panState = useRef<{ x: number; y: number; panX: number; panY: number; pointerId: number } | null>(null);
  // Fingers on the canvas. Two or more suppress box moves and resizes.
  const touchCountRef = useRef(0);
  // For the wheel listener: nested setState updaters run twice under StrictMode.
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;
  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);
  useEffect(() => {
    // While painting, the wheel resizes the MaskPainter brush.
    if (paintingId !== null) return;
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const wrap = canvasWrapperRef.current;
      if (!wrap) return;
      // A trackpad pinch arrives as a wheel event with ctrlKey set.
      if (e.ctrlKey || e.metaKey) {
        // Keeps the point under the cursor in place.
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
      setPan({
        x: panRef.current.x - e.deltaX,
        y: panRef.current.y - e.deltaY,
      });
    };
    // Native listener: React's onWheel is passive and cannot preventDefault.
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [paintingId]);

  // Two-finger pinch zooms and pans. One finger goes to the SVG.
  useEffect(() => {
    if (paintingId !== null) return;
    const el = containerRef.current;
    if (!el) return;
    // Gesture start: finger spread, zoom, anchor relative to the canvas,
    // client midpoint, pan.
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
        // The first finger may have selected a box; a pinch clears it.
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
      const next = Math.max(1, Math.min(8, pinch.zoom * (dist(e.touches) / pinch.dist)));
      const k = next / pinch.zoom;
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
  }, [paintingId]);
  const onContainerPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      // Capture phase: MaskPainter and the box handlers never see the pan.
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
  // Reports the mouse hover only, not emphasizedBoxId.
  const onHoverChangeRef = useRef(onHoverChange);
  useEffect(() => { onHoverChangeRef.current = onHoverChange; });
  useEffect(() => { onHoverChangeRef.current?.(hoveredId); }, [hoveredId]);
  // For rendering only. The handlers set hoveredId.
  const effectiveHoveredId: string | null = hoveredId ?? emphasizedBoxId;
  const [drawMode, setDrawMode] = useState(false);
  const [pointMode, setPointMode] = useState(false);
  // Below md the Labels list is a bottom sheet that hides 5 s after the last interaction.
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
  // Box focused from the label list. The others dim.
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [paintMode, setPaintMode] = useState<PaintMode>("brush");
  const [brushSize, setBrushSize] = useState(40);
  const hasMasks = boxes.some((b) => (b.mask?.polygons?.length ?? 0) > 0);
  const paintingBox = paintingId ? boxes.find((b) => b.id === paintingId) ?? null : null;
  const painterHandle = useRef<MaskPainterHandle | null>(null);

  // Project tags first, then labels typed on existing boxes.
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

  // One-level undo on Ctrl or Cmd + Z. Each pointer gesture snapshots the
  // boxes before its first change.
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
        // Raw onChange, so the undo is not itself snapshotted.
        onChange(undoRef.current);
        undoRef.current = null;
        canSnapshotRef.current = true;
      }
    };
    window.addEventListener("keydown", onKey);
    const onDown = () => { canSnapshotRef.current = true; };
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [onChange]);

  // The viewBox is in image pixels. Overlay sizes are screen pixels divided
  // by scaleFactor, screen px per image px.
  const displayScale = canvasSize ? canvasSize.w / imageWidth : 1;
  const scaleFactor = Math.max(0.001, displayScale * zoom);
  const handleSize = 14 / scaleFactor;
  const strokeWidth = 2 / scaleFactor;
  const fontSize = 14 / scaleFactor;
  const boxRadius = 4 / scaleFactor;
  const labelFont = '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif';

  // Client coordinates to image pixels, from the img element's rendered rect.
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

  // Draw a new box. In draw mode a press over an existing box starts a new one.
  const onSvgPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    // Pan, handled by the container.
    if (e.button === 1 || (e.button === 0 && e.altKey)) return;
    // Click to detect: one request returns the box, mask and label.
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
        // A miss or a failure removes the placeholder. A failure is logged.
        try {
          let res: Awaited<ReturnType<NonNullable<typeof onPointDetect>>> = null;
          try {
            res = await onPointDetect({ x, y });
          } catch (err) {
            console.error("[click-to-detect] onPointDetect threw:", err);
            res = null;
          }
          if (!res) {
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
          // One update, so the auto-PUT never sees an in-flight box.
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
          // No tag resolved: open the label picker.
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
        // No handlers: keep the drawn box and open the label picker.
        if (!wantsCombined && !wantsSegment && !wantsClassify) {
          setEditingId(id);
        } else {
          // Spinners until the result lands, in one update below.
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
                // Keeps a rename made while the request ran.
                if (label && b.label === "new") next.label = label;
                return next;
              }),
            );
            // No tag resolved: open the label picker.
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

  const startMove =(e: React.PointerEvent, boxId: string) => {
    if (drawMode) return;
    if (touchCountRef.current >= 2) return;
    e.stopPropagation();
    const initial = stateRef.current.boxes.find((b) => b.id === boxId);
    if (!initial) return;
    // A touch on an unselected box selects it without moving it.
    if (e.pointerType === "touch" && selectedId !== boxId) {
      setSelectedId(boxId);
      setFocusedId(null);
      return;
    }
    setSelectedId(boxId);
    setFocusedId(null);
    const start = toLocal(e.clientX, e.clientY);
    const w = initial.x1 - initial.x0;
    const h = initial.y1 - initial.y0;
    // A hand-painted mask moves with the box. Others are re-segmented on drop.
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
      // A second finger aborts the move.
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

  // Re-segments a box after a move or resize.
  const refreshMask = (boxId: string) => {
    if (!onBoxDrawn) return;
    const box = stateRef.current.boxes.find((b) => b.id === boxId);
    if (!box) return;
    // A hand-painted mask is never re-segmented.
    if (box.maskEdited) return;
    stateRef.current.onChange(
      stateRef.current.boxes.map((b) => (b.id === boxId ? { ...b, segmenting: true } : b)),
    );
    (async () => {
      try {
        const mask = await onBoxDrawn(box);
        stateRef.current.onChange(
          stateRef.current.boxes.map((b) => {
            if (b.id !== boxId) return b;
            // Stale: the box has moved since the request.
            if (b.x0 !== box.x0 || b.y0 !== box.y0 || b.x1 !== box.x1 || b.y1 !== box.y1) {
              return { ...b, segmenting: false };
            }
            // No mask: keep the box's current one.
            return { ...b, segmenting: false, mask: mask ?? b.mask ?? null };
          }),
        );
      } catch {
        stateRef.current.onChange(
          stateRef.current.boxes.map((b) => (b.id === boxId ? { ...b, segmenting: false } : b)),
        );
      }
    })();
  };

  const startResize =(e: React.PointerEvent, boxId: string, handle: Handle) => {
    if (drawMode) return;
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
    if (focusedId === id) setFocusedId(null);
    // The box leaves the DOM before pointerLeave can fire.
    if (hoveredId === id) setHoveredId(null);
    if (paintingId === id) setPaintingId(null);
  };

  const renameBox = (id: string, label: string) => {
    const prev = boxes.find((b) => b.id === id);
    onChangeUndoable(boxes.map((b) => (b.id === id ? { ...b, label } : b)));
    // "new", "label" and "detecting" are placeholder labels.
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

  // Overrides a validator rejection.
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


  // Delete, Esc and the M, B, X mode keys.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (editingId) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if ((e.key === "Backspace" || e.key === "Delete") && selectedId) {
        e.preventDefault();
        removeBox(selectedId);
      }
      if (e.key === "Escape" && !e.defaultPrevented && (selectedId || editingId || drawMode)) {
        // Used here, so the viewer behind does not close as well.
        e.preventDefault();
        setSelectedId(null);
        setEditingId(null);
        setDrawMode(false);
      }
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const k = e.key.toLowerCase();
        // M: toggle mask painting on the hovered, else the selected, box.
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
        // B: toggle draw mode.
        if (k === "b") {
          e.preventDefault();
          setDrawMode((v) => !v);
          setPointMode(false);
          setPaintingId(null);
          setSelectedId(null);
          setEditingId(null);
          return;
        }
        // X: toggle click to detect.
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
    // Capture phase, so Esc is handled here before the viewer behind sees it.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
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

  const pendingClassify = boxes.filter((b) => b.classifying).length;
  const pendingSegment = boxes.filter((b) => b.segmenting).length;
  const busy = pendingClassify > 0 || pendingSegment > 0;
  const busyParts: string[] = [];
  if (pendingClassify > 0) busyParts.push(`labelling ${pendingClassify}`);
  if (pendingSegment > 0) busyParts.push(`segmenting ${pendingSegment}`);

  // Hover dims a parent or child of the hovered box less than other boxes.
  // Px of slack in the containment test.
  const HOVER_RELATION_TOL = 16;
  const hoveredBox = effectiveHoveredId ? boxes.find((b) => b.id === effectiveHoveredId) ?? null : null;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Toolbar */}
      <div className="relative shrink-0">
      <div
        className="relative flex items-center justify-between gap-2 sm:gap-3 px-2 sm:px-4 py-2 sm:py-2.5 shrink-0 overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
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
          {onPointDetect && (
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
              ].join(" ")}
              title="Click anywhere on an object to auto-detect it"
            >
              {pointMode ? "Click an object…" : "Click to detect"}
            </button>
          )}
          <OverlayToggle label="Boxes" on={showBoxes} setOn={setShowBoxes} />
          <OverlayToggle label="Labels" on={showLabels} setOn={setShowLabels} />
          <OverlayToggle label="Masks" on={showMasks} setOn={setShowMasks} disabled={!hasMasks} />
          {selectedId && !paintingId && (
            <button
              onClick={() => setPaintingId(selectedId)}
              className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out border border-[var(--line)] text-[var(--fg-muted)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)]"
              title="Paint or erase pixels on this box's mask"
            >
              ✎ Edit mask
            </button>
          )}
          {paintingId && (
            // Below md the paint controls are in the bottom paint bar.
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
          {/* Legend for the keys BoxEditor handles, where the toolbar has room. */}
          <div className="hidden 2xl:flex items-center gap-1.5">
            <HotkeyChip k="B" label="Add box" />
            {onPointDetect && <HotkeyChip k="X" label="Click-detect" />}
            <HotkeyChip k="M" label="Edit mask" />
          </div>
          <span className="tabular-nums">
            {boxes.length} box{boxes.length === 1 ? "" : "es"}
          </span>
        </div>
      </div>
      {/* Edge fades showing that the toolbar scrolls, below md. */}
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
          // touch-action none: a pinch zooms the canvas, not the page.
          style={{ cursor: panState.current ? "grabbing" : undefined, touchAction: "none" }}
        >
          {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
            <button
              onClick={resetView}
              className="absolute top-3 right-3 z-10 px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider tabular-nums rounded-md bg-[var(--modal-surface)] border border-[var(--line)] text-[var(--fg-muted)] shadow-[var(--shadow-soft)] hover:border-[var(--line-strong)] hover:text-[var(--foreground)] transition-colors duration-150"
              title="Reset zoom / pan"
            >
              {Math.round(zoom * 100)}% · reset
            </button>
          )}
          {/* Spinner while the image loads, outside the zoomed wrapper. */}
          {!imgLoaded && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center" aria-hidden>
              <span className="h-9 w-9 animate-spin rounded-full border-[3px] border-white/30 border-t-white drop-shadow-[0_1px_3px_rgba(0,0,0,0.5)]" />
            </div>
          )}
          <div
            ref={canvasWrapperRef}
            className="relative"
            style={{
              // Zoom sets the size rather than a scale() so the SVG stays sharp.
              // It grows from the top-left, as the zoom anchor math assumes.
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
            {/* Hidden preloader for the full image. */}
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
          {/* Hover spotlight: dims the image outside the hovered box's mask,
              below every box. */}
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
                    {/* White: dimmed. Black: not dimmed. */}
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
                      // No mask: the box itself.
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
            <filter id="bx-chip-shadow" x="-20%" y="-20%" width="140%" height="160%">
              <feDropShadow dx={0} dy={fontSize * 0.08} stdDeviation={fontSize * 0.10} floodColor="#000" floodOpacity={0.4} />
            </filter>
            <filter id="bx-select-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation={strokeWidth * 1.6} result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {[...boxes]
            // Largest first, so smaller boxes stay on top and hoverable. Hover
            // and selection never change the order.
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
            const c = sizeC ?? colourForBox(b);
            const rejected = b.validation && b.validation.match === false;
            const tint = (l: number, a: number) => `hsla(${c.hue}, ${c.sat}%, ${l}%, ${a})`;
            const baseStroke = tint(c.light, 0.92);
            const hoverStroke = tint(Math.min(c.light + 14, 80), 0.97);
            const selectStroke = tint(Math.min(c.light + 18, 82), 1);
            const stroke = isSelected ? selectStroke : isHovered ? hoverStroke : baseStroke;
            const fillRect = isSelected
              ? tint(Math.min(c.light + 18, 80), 0.12)
              : isHovered
              ? tint(Math.min(c.light + 18, 80), 0.08)
              : tint(Math.min(c.light + 12, 78), 0.04);
            const sw = isActive ? strokeWidth * 1.3 : strokeWidth;
            const cornerRadius = Math.min(boxRadius, w * 0.4, h * 0.4);

            // MaskPainter draws the mask of the box being painted.
            const isPainting = paintingId === b.id;
            const polys = !isPainting && showMasks ? (b.mask?.polygons ?? []) : [];
            const maskFill = isSelected
              ? "rgb(var(--foreground-rgb) / 0.28)"
              : isHovered
              ? tint(c.light, 0.45)
              : tint(c.light, 0.30);
            const maskStroke = isSelected ? "#fff" : tint(c.light, 0.9);

            const labelText = (rejected ? "⚠ " : "") + displayLabelFn(b.label || "label");
            // Estimated at 0.6 em per character.
            const charW = fontSize * 0.6;
            const padX = fontSize * 0.35;
            const maxChars = Math.max(8, Math.floor((w + fontSize * 4) / charW));
            const display = labelText.length > maxChars ? labelText.slice(0, maxChars - 1) + "…" : labelText;
            const labelW = display.length * charW + padX * 2;
            const labelH = fontSize * 1.3;
            // A selected chip is filled with the foreground colour, so its text
            // takes the background colour.
            const chipTextFill = rejected
              ? "#fff"
              : isSelected
              ? "rgb(var(--background-rgb))"
              : c.light < 50 ? "#fff" : "#0a0a0a";
            // The chip goes above the box, else below, else inside at the top-left.
            const labelFitsAbove = b.y0 >= labelH;
            const labelFitsBelow = (imageHeight - b.y1) >= labelH;
            const labelInside = !labelFitsAbove && !labelFitsBelow;
            const labelBelow = !labelFitsAbove && labelFitsBelow;
            const labelY = labelInside
              ? b.y0
              : labelBelow
              ? b.y1
              : b.y0 - labelH;
            // tabbed: box and chip drawn as one outline, chip above or below.
            const tabbed = showBoxes && showLabels && !labelInside;
            const insideChip = showBoxes && showLabels && labelInside;
            const chipR = Math.min(cornerRadius, labelH / 2, labelW / 2);

            const onEnter = () => setHoveredId(b.id);
            const onLeave = () => setHoveredId((h) => (h === b.id ? null : h));

            const rejectStroke = "rgba(255,80,80,1)";
            const rejectFill = "rgba(248,113,113,0.18)";
            const rejectMaskFill = "rgba(248,113,113,0.30)";

            const dimForDraw = drawMode && b.id !== selectedId;
            const dimForFocus = focusedId !== null && b.id !== focusedId;
            const dimForPaint = paintingId !== null && b.id !== paintingId;
            const dimForPoint = pointMode;
            const dimForHover = effectiveHoveredId !== null && b.id !== effectiveHoveredId;
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
            const groupOpacity = dimForDraw || dimForFocus || dimForPaint || dimForPoint
              ? 0.08
              : dimForHover
              ? (isRelatedToHover ? 0.55 : 0.15)
              : 1;

            // Click to detect in flight.
            if (b.detecting) {
              const cx = typeof b.detecting === "object" ? b.detecting.cx : (b.x0 + b.x1) / 2;
              const cy = typeof b.detecting === "object" ? b.detecting.cy : (b.y0 + b.y1) / 2;
              const labelText = b.classifying ? "Labelling…" : "Detecting…";
              const dispW = labelText.length * charW + padX * 2;
              const polysAvailable = (b.mask?.polygons?.length ?? 0) > 0;
              const dotR = 5 / scaleFactor;
              // Above the box once a mask exists, else at the click point.
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
              <g key={b.id} opacity={groupOpacity}>
                {rejected && (
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
                    style={{ cursor: drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                    pointerEvents={pointMode ? "none" : undefined}
                    onPointerEnter={onEnter}
                    onPointerLeave={onLeave}
                    onPointerDown={(e) => startMove(e, b.id)}
                  />
                ))}
                {/* Box and chip as one outline. Paths: box interior, which takes
                    the pointer events; chip fill; combined outline. */}
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
                      style={{ cursor: drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                      pointerEvents={pointMode ? "none" : "all"}
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
                  /* Chip inside the box, against its top-left corner. */
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
                      style={{ cursor: drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
                      pointerEvents={pointMode ? "none" : "all"}
                      onPointerEnter={onEnter}
                      onPointerLeave={onLeave}
                      onPointerDown={(e) => startMove(e, b.id)}
                    />
                    <path
                      // Rounded top-left and bottom corners, square top-right.
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
                        style={{ cursor: drawMode ? "crosshair" : pointMode ? "crosshair" : "move" }}
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
                {isSelected &&
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
                // Remounts per box so the canvas starts from that box's polygons.
                key={paintingBox.id}
                imageUrl={imageUrl}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                initialPolygons={paintingBox.mask?.polygons ?? []}
                // The whole image: the box snaps to the mask's bounds on save.
                clip={{ x0: 0, y0: 0, x1: imageWidth, y1: imageHeight }}
                brushSize={brushSize}
                setBrushSize={setBrushSize}
                mode={paintMode}
                onSave={(polygons) => {
                  // Box from the mask's bounds, or the old box if the mask is empty.
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
                    stateRef.current.boxes.map((b) =>
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
            // Paint controls from md up, outside the zoomed wrapper.
            <div
              className="absolute top-3 left-1/2 -translate-x-1/2 z-20 hidden md:flex items-center gap-2 rounded-lg px-2 py-1 text-xs"
              style={{
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

          {/* Paint bar below md. */}
          {paintingBox && (
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
            // Below md: bottom sheet.
            "flex flex-col min-h-0 absolute inset-x-2 bottom-2 z-30 max-h-[42%] rounded-lg border border-[var(--line)] bg-[rgb(var(--surface-rgb)/0.94)] backdrop-blur-md shadow-[var(--shadow-strong)] overflow-hidden",
            mobileLabelsOpen ? "flex" : "hidden",
            // md and up: right-hand column.
            "md:flex md:static md:inset-auto md:bottom-auto md:z-auto md:max-h-none md:w-72 md:shrink-0 md:rounded-none md:border-y-0 md:border-r-0 md:border-l md:border-[var(--line)] md:bg-[var(--panel)] md:backdrop-blur-none md:shadow-none md:overflow-visible",
          ].join(" ")}
        >
          <div
            className="px-4 py-2.5 pk-micro shrink-0 flex items-center justify-between gap-2"
            style={{ borderBottom: "1px solid var(--line-soft)" }}
          >
            <span>Labels · {boxes.length}</span>
            <div className="flex items-center gap-2">
              {loadingLabels && (
                <span
                  aria-label="Loading labels"
                  className="h-3 w-3 rounded-full border-2 border-foreground/15 border-t-foreground/60 animate-spin"
                />
              )}
              {/* Close, below md only. */}
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
          const c = sizeC ?? colourForBox(b);
          const isHovered = effectiveHoveredId === b.id;
          const isSelected = selectedId === b.id;
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
            className="flex items-center gap-3 px-4 py-2 text-xs transition-colors"
            style={{ background: rowBg, boxShadow: rowBorder }}
            onMouseEnter={() => setHoveredId(b.id)}
            onMouseLeave={() => setHoveredId((h) => (h === b.id ? null : h))}
          >
            <span
              className="h-3 w-3 rounded-sm shrink-0 border border-foreground/10"
              style={{ backgroundColor: `hsl(${c.hue}, ${c.sat}%, ${c.light}%)` }}
            />
            {editingId === b.id ? (
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
                  onClick={() => setFocusedId((cur) => (cur === b.id ? null : b.id))}
                  onDoubleClick={() => setEditingId(b.id)}
                  className={[
                    "flex-1 text-left font-mono hover:underline underline-offset-2",
                    focusedId === b.id ? "text-[var(--foreground)]" : "",
                  ].join(" ")}
                  title="click to focus · double-click to rename"
                >
                  {displayLabelFn(b.label)}
                </button>
                <button
                  onClick={() => setEditingId(b.id)}
                  className="text-[var(--muted)] hover:text-foreground px-1"
                  aria-label="Rename"
                  title="Rename"
                >
                  ✎
                </button>
              </>
            )}
            {b.validation && b.validation.match === false && (() => {
              const reasonTitle = `Rejected${b.validation.reason ? `: ${b.validation.reason}` : ""}.`;
              const Chip = <span>Rejected</span>;
              return (
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
                title={b.validation.reason || "PixelKit flagged this detection as ambiguous. Confirm or delete it."}
              >
                Unsure
              </span>
            )}
            {b.validation && b.validation.match && b.validation.kind !== "unsure" && b.validation.confidence > 0 && (
              b.validation.source === "cascade" ? (
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
                  title={`Verified ${(b.validation.confidence * 100).toFixed(0)}%${b.validation.reason ? `, ${b.validation.reason}` : ""}`}
                >
                  <span>{b.validation.source === "manual" ? "Manual" : "Verified"}</span>
                </span>
              )
            )}
            {(b.segmenting || b.classifying) && (
              <span className="font-mono text-[var(--muted)] tabular-nums">…</span>
            )}
            <button
              onClick={() => removeBox(b.id)}
              className="rounded-full h-6 w-6 grid place-items-center hover:bg-foreground/10"
              aria-label="Delete"
              title="Delete"
            >
              ×
            </button>
          </li>
          );
        })}
      </ul>
        </div>

        {/* Opens the Labels sheet, below md. */}
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
        title="Brush: paint into the mask"
      >
        Brush
      </button>
      <button
        onClick={() => setMode("eraser")}
        className={[
          "rounded-[4px] px-2.5 py-1 text-xs font-medium transition-colors duration-150 ease-out",
          mode === "eraser" ? "bg-[var(--accent-dim)] text-[var(--foreground)]" : "text-[var(--fg-dim)] hover:text-[var(--foreground)]",
        ].join(" ")}
        title="Eraser: remove pixels from the mask"
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
  // The input shows display labels. A matched pick commits the canonical label.
  const [query, setQuery] = useState(displayLabelFn(initial));
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const committed = useRef(false);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  // Display label -> canonical label, or the input itself for a new label.
  const canonicalForDisplay = (input: string): string => {
    const lower = input.trim().toLowerCase();
    for (const s of suggestions) {
      if (displayLabelFn(s).trim().toLowerCase() === lower) return s;
      if (s.trim().toLowerCase() === lower) return s;
    }
    return input.trim();
  };

  // A click outside commits the input.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target as Node)) {
        if (!committed.current) {
          committed.current = true;
          onCommit(canonicalForDisplay(query));
        }
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, onCommit]);

  const q = query.trim().toLowerCase();
  const filtered = q
    ? suggestions.filter((s) =>
        s.toLowerCase().includes(q)
        || displayLabelFn(s).toLowerCase().includes(q),
      )
    : suggestions;
  const showCreate = q.length > 0 && !suggestions.some(
    (s) => s.toLowerCase() === q || displayLabelFn(s).toLowerCase() === q,
  );

  // Enter takes a label that starts with the input, else a new label as typed.
  const startsQ = (s: string) =>
    s.toLowerCase().startsWith(q) || displayLabelFn(s).toLowerCase().startsWith(q);
  const prefixed = q ? filtered.filter(startsQ) : filtered;
  const inside = q ? filtered.filter((s) => !startsQ(s)) : [];
  const items = showCreate && prefixed.length === 0
    ? [`__create:${q}`, ...inside]
    : [...prefixed, ...inside, ...(showCreate ? [`__create:${q}`] : [])];

  const commit = (raw: string, isCreate = false) => {
    if (committed.current) return;
    committed.current = true;
    // A new label commits as typed. Others map back to the canonical label.
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
        className="w-full rounded border border-[var(--line-strong)] px-2 py-1 outline-none font-mono text-xs bg-[var(--background)] text-[var(--foreground)] placeholder:text-[var(--fg-faint)] focus:border-[var(--accent)]"
      />
      {items.length > 0 && (
        <ul
          role="listbox"
          className="absolute left-0 right-0 top-full mt-1 max-h-56 overflow-y-auto rounded-md border border-[var(--line)] bg-[var(--panel-solid)] shadow-[var(--shadow-strong)] z-20 py-1"
        >
          {items.map((it, i) => {
            const isCreate = it.startsWith("__create:");
            const rawValue = isCreate ? it.slice("__create:".length) : it;
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

function cleanLabel(raw: string, tags: string[]): string {
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
