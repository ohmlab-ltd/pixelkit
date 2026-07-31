"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export type PaintMode = "brush" | "eraser";

export type MaskPainterHandle = { commit: () => void };

type Props = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  /** Polygons to seed the canvas with, usually the box's existing mask. */
  initialPolygons: number[][][];
  /** Where the brush is constrained, only pixels inside this clip-box can
      be painted, so a stroke can't escape the bounding box that owns the
      mask. Coordinates are image-space. */
  clip?: { x0: number; y0: number; x1: number; y1: number } | null;
  brushSize: number;
  /** Mouse wheel over the canvas adjusts brush size, needs to flow back
      out so the toolbar slider stays in sync. */
  setBrushSize?: (n: number) => void;
  mode: PaintMode;
  onSave: (polygons: number[][][]) => void;
  onCancel: () => void;
};

/** Drop-in painter that sits on top of the BoxEditor canvas. Owns its own
    <canvas> and keeps the raster in-state until the user commits, the
    extracted polygons are only synthesised on save, so we don't pay marching-
    squares cost on every brush stroke. */
export const MaskPainter = forwardRef<MaskPainterHandle, Props>(function MaskPainter({
  imageUrl,
  imageWidth,
  imageHeight,
  initialPolygons,
  clip,
  brushSize,
  setBrushSize,
  mode,
  onSave,
  onCancel,
}, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const drawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);

  // Bright cyan for the in-progress mask, chosen because it survives on
  // grass, sky, road, skin, etc. The CSS `filter` below stamps a crisp
  // white outline around it so the painted edge is unambiguous.
  const PAINT_COLOR = "rgba(34, 211, 238, 1)"; // tailwind cyan-400

  // Seed the canvas with the existing polygons exactly once. We deliberately
  // don't react to `initialPolygons` changes, the parent passes a snapshot
  // taken when paint mode opened; subsequent edits live in the canvas.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, imageWidth, imageHeight);
    ctx.fillStyle = PAINT_COLOR;
    for (const poly of initialPolygons) {
      if (poly.length < 3) continue;
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.fill();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageWidth, imageHeight]);

  const toLocal = (clientX: number, clientY: number) => {
    const wrap = wrapRef.current;
    if (!wrap) return { x: 0, y: 0 };
    const rect = wrap.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * imageWidth,
      y: ((clientY - rect.top) / rect.height) * imageHeight,
    };
  };

  const stroke = (from: { x: number; y: number } | null, to: { x: number; y: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.save();
    if (clip) {
      ctx.beginPath();
      ctx.rect(clip.x0, clip.y0, clip.x1 - clip.x0, clip.y1 - clip.y0);
      ctx.clip();
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = mode === "eraser" ? "rgba(0,0,0,1)" : PAINT_COLOR;
    ctx.fillStyle = mode === "eraser" ? "rgba(0,0,0,1)" : PAINT_COLOR;
    ctx.globalCompositeOperation = mode === "eraser" ? "destination-out" : "source-over";
    if (from) {
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(to.x, to.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    drawingRef.current = true;
    const p = toLocal(e.clientX, e.clientY);
    lastPosRef.current = p;
    stroke(null, p);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const p = toLocal(e.clientX, e.clientY);
    setCursor(p);
    if (!drawingRef.current) return;
    e.preventDefault();
    stroke(lastPosRef.current, p);
    lastPosRef.current = p;
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    drawingRef.current = false;
    lastPosRef.current = null;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  };

  const onPointerLeave = () => {
    setCursor(null);
  };

  const commit = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      onSave([]);
      return;
    }
    const polys = canvasToPolygons(canvas);
    onSave(polys);
  };

  useImperativeHandle(ref, () => ({ commit }));

  // Wheel resizes the brush. React's onWheel attaches as passive so
  // preventDefault is a no-op there, bind manually with passive: false.
  // brushSize is read via a ref so the listener stays stable.
  const sizeRef = useRef(brushSize);
  sizeRef.current = brushSize;
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap || !setBrushSize) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // ~5% of the current size per notch, clamped to the slider range.
      const step = Math.max(1, Math.round(sizeRef.current * 0.1));
      const delta = e.deltaY < 0 ? step : -step;
      const next = Math.max(4, Math.min(200, sizeRef.current + delta));
      setBrushSize(next);
    };
    wrap.addEventListener("wheel", onWheel, { passive: false });
    return () => wrap.removeEventListener("wheel", onWheel);
  }, [setBrushSize]);

  // Save with Enter, cancel with Escape, matches the rename picker's chord.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The wrapper sits in the same coordinate space as the BoxEditor SVG
  // (absolute inset-0 inside the image-aspect div). The canvas paints at
  // image-pixel resolution; CSS scales it to the displayed size.
  return (
    <div
      ref={wrapRef}
      className="absolute inset-0"
      style={{ touchAction: "none", cursor: "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerLeave}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt=""
        className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-0"
        aria-hidden
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{
          opacity: 0.7,
          // 4-direction drop-shadow trick: stamps the canvas alpha 1px in
          // each axis to draw a crisp 1px white outline around the painted
          // region, regardless of background colour.
          filter:
            "drop-shadow(1px 0 0 white) drop-shadow(-1px 0 0 white) drop-shadow(0 1px 0 white) drop-shadow(0 -1px 0 white)",
        }}
      />
      {cursor && (
        <svg
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          <circle
            cx={cursor.x}
            cy={cursor.y}
            r={brushSize / 2}
            fill="none"
            stroke={mode === "eraser" ? "rgba(248,113,113,0.95)" : "rgb(var(--foreground-rgb) / 0.95)"}
            // non-scaling-stroke keeps the brush outline a constant ~1.5px on
            // screen instead of thickening with zoom (the ring radius still
            // scales, only the outline weight stays fixed).
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      )}
    </div>
  );
});

/** Marching-squares contour extraction. Reads the canvas alpha channel,
    emits a flat list of polygons that approximate every connected blob.
    Saddle cases are resolved consistently (always pick the connection that
    keeps each loop closed). */
export function canvasToPolygons(canvas: HTMLCanvasElement, threshold = 64): number[][][] {
  const W = canvas.width;
  const H = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  const data = ctx.getImageData(0, 0, W, H).data;
  const inside = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) inside[i] = data[i * 4 + 3] > threshold ? 1 : 0;

  // Fill enclosed holes: flood from every canvas-edge pixel that's
  // currently "outside", marking them as background. Anything still
  // unflagged after the flood is enclosed by mask pixels, that's a
  // hole. Flip those to "inside" so the marching-squares walk doesn't
  // emit an inner contour and SVG even-odd fill doesn't render the
  // segmentation as a donut. Stack-based DFS (push/pop) for O(N) on
  // up to ~1M pixels without the array-shift cost of BFS.
  const reached = new Uint8Array(W * H);
  const stack: number[] = [];
  const seedEdge = (x: number, y: number) => {
    const i = y * W + x;
    if (inside[i] || reached[i]) return;
    reached[i] = 1;
    stack.push(i);
  };
  for (let x = 0; x < W; x++) {
    seedEdge(x, 0);
    seedEdge(x, H - 1);
  }
  for (let y = 0; y < H; y++) {
    seedEdge(0, y);
    seedEdge(W - 1, y);
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % W;
    const y = (i - x) / W;
    if (x > 0) {
      const j = i - 1;
      if (!inside[j] && !reached[j]) {
        reached[j] = 1;
        stack.push(j);
      }
    }
    if (x < W - 1) {
      const j = i + 1;
      if (!inside[j] && !reached[j]) {
        reached[j] = 1;
        stack.push(j);
      }
    }
    if (y > 0) {
      const j = i - W;
      if (!inside[j] && !reached[j]) {
        reached[j] = 1;
        stack.push(j);
      }
    }
    if (y < H - 1) {
      const j = i + W;
      if (!inside[j] && !reached[j]) {
        reached[j] = 1;
        stack.push(j);
      }
    }
  }
  for (let i = 0; i < W * H; i++) {
    if (!inside[i] && !reached[i]) inside[i] = 1;
  }

  // Edge midpoints, indexed clockwise from the top edge.
  // 0 = top, 1 = right, 2 = bottom, 3 = left.
  const edge = (cx: number, cy: number, e: number): [number, number] => {
    if (e === 0) return [cx + 0.5, cy];
    if (e === 1) return [cx + 1, cy + 0.5];
    if (e === 2) return [cx + 0.5, cy + 1];
    return [cx, cy + 0.5]; // e === 3
  };

  // For each of the 16 marching-squares codes, list the edge pairs that
  // form line segments (each pair = one segment).
  const SEGS: number[][][] = [
    [], // 0
    [[3, 0]], // 1: TL only
    [[0, 1]], // 2: TR only
    [[3, 1]], // 3: TL+TR
    [[1, 2]], // 4: BR only
    [[3, 0], [1, 2]], // 5: saddle (TL+BR)
    [[0, 2]], // 6: TR+BR
    [[3, 2]], // 7: !BL
    [[2, 3]], // 8: BL only
    [[0, 2]], // 9: TL+BL
    [[0, 1], [2, 3]], // 10: saddle (TR+BL)
    [[1, 2]], // 11: !BR
    [[1, 3]], // 12: BL+BR
    [[0, 1]], // 13: !TR
    [[3, 0]], // 14: !TL
    [], // 15: all in
  ];

  // Map each segment endpoint to a list of segments touching it. Use string
  // keys so floating-point endpoints (always halves) hash deterministically.
  const key = (p: [number, number]) => `${p[0]},${p[1]}`;
  const adj = new Map<string, [number, number][]>();
  const addSeg = (a: [number, number], b: [number, number]) => {
    const ka = key(a);
    const kb = key(b);
    if (!adj.has(ka)) adj.set(ka, []);
    if (!adj.has(kb)) adj.set(kb, []);
    adj.get(ka)!.push(b);
    adj.get(kb)!.push(a);
  };

  // Walk one extra cell past each edge with virtual "outside" corners so
  // mask pixels touching the canvas boundary still emit edge segments.
  // Without this, anything in the rightmost column or bottom row gets
  // silently clipped off when we extract polygons.
  const cell = (px: number, py: number): number =>
    px < 0 || py < 0 || px >= W || py >= H ? 0 : inside[py * W + px];
  const clamp01 = (n: number, max: number) => Math.max(0, Math.min(max, n));
  for (let y = -1; y < H; y++) {
    for (let x = -1; x < W; x++) {
      const tl = cell(x, y);
      const tr = cell(x + 1, y);
      const br = cell(x + 1, y + 1);
      const bl = cell(x, y + 1);
      const code = tl | (tr << 1) | (br << 2) | (bl << 3);
      for (const seg of SEGS[code]) {
        const a = edge(x, y, seg[0]);
        const b = edge(x, y, seg[1]);
        addSeg(
          [clamp01(a[0], W), clamp01(a[1], H)],
          [clamp01(b[0], W), clamp01(b[1], H)],
        );
      }
    }
  }

  // Walk the adjacency map to assemble closed polygons. Pop endpoints as we
  // visit them so each segment is consumed once.
  const polygons: number[][][] = [];
  while (adj.size > 0) {
    const startKey = adj.keys().next().value!;
    const startNeighbours = adj.get(startKey);
    if (!startNeighbours || startNeighbours.length === 0) {
      adj.delete(startKey);
      continue;
    }
    const start = startKey.split(",").map(Number) as [number, number];
    const polygon: number[][] = [start];
    let curKey = startKey;
    let cur: [number, number] = start;
    let safety = W * H * 4;
    while (safety-- > 0) {
      const neighbours = adj.get(curKey);
      if (!neighbours || neighbours.length === 0) {
        adj.delete(curKey);
        break;
      }
      const next = neighbours.pop()!;
      // Mirror-pop on the neighbour side so the segment is fully consumed.
      const nextKey = key(next);
      const nextNeighbours = adj.get(nextKey);
      if (nextNeighbours) {
        const idx = nextNeighbours.findIndex((p) => p[0] === cur[0] && p[1] === cur[1]);
        if (idx >= 0) nextNeighbours.splice(idx, 1);
        if (nextNeighbours.length === 0) adj.delete(nextKey);
      }
      if (!neighbours.length) adj.delete(curKey);
      if (next[0] === start[0] && next[1] === start[1]) {
        polygon.push(start);
        break;
      }
      polygon.push(next);
      cur = next;
      curKey = nextKey;
    }
    if (polygon.length >= 4) polygons.push(simplify(polygon, 1.0));
  }
  return polygons;
}

/** Douglas-Peucker simplification. Pixel-aligned marching-squares output is
    visually noisy, this brings the vertex count down ~10x without losing
    obvious detail. */
function simplify(points: number[][], tolerance: number): number[][] {
  if (points.length < 3) return points;
  const sqTol = tolerance * tolerance;
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [first, last] = stack.pop()!;
    let maxSqDist = 0;
    let index = first;
    for (let i = first + 1; i < last; i++) {
      const d = sqSegDist(points[i], points[first], points[last]);
      if (d > maxSqDist) {
        maxSqDist = d;
        index = i;
      }
    }
    if (maxSqDist > sqTol) {
      keep[index] = 1;
      stack.push([first, index]);
      stack.push([index, last]);
    }
  }
  const out: number[][] = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}

function sqSegDist(p: number[], a: number[], b: number[]): number {
  let x = a[0];
  let y = a[1];
  let dx = b[0] - x;
  let dy = b[1] - y;
  if (dx !== 0 || dy !== 0) {
    const t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy);
    if (t > 1) {
      x = b[0];
      y = b[1];
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
    }
  }
  dx = p[0] - x;
  dy = p[1] - y;
  return dx * dx + dy * dy;
}
