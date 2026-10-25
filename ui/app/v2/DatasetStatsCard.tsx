"use client";

// Variation plot: the images and their augmentations as 2-D points, coloured by label.

import { useRef, useState } from "react";

import { readableTextForBg } from "./labelColours";

export type StatsPoint = {
  id: string;
  filename: string | null;
  x: number;
  y: number;
  label: string | null;
  n_detections: number;
};

export type AugmentPoint = {
  id: string;
  source_id: string;
  filename: string | null;
  x: number;
  y: number;
  label: string | null;
};

export function VariationPlot({
  points,
  augmentations,
  colourFor,
  nearDuplicateIds,
  onJumpToImport,
  computing = false,
  totalImports = 0,
  embeddingsReady = 0,
}: {
  points: StatsPoint[];
  augmentations: AugmentPoint[];
  colourFor: (label: string | null) => string;
  nearDuplicateIds: Set<string>;
  onJumpToImport?: (importId: string, kind: "image" | "augmentation") => void;
  /** Some images lack embeddings. With no points, shows a progress ring. */
  computing?: boolean;
  totalImports?: number;
  embeddingsReady?: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<StatsPoint | null>(null);

  if (points.length === 0) {
    if (computing) {
      const pct = totalImports > 0
        ? Math.min(100, Math.round((embeddingsReady / totalImports) * 100))
        : 0;
      const ringSize = 56;
      return (
        <div
          ref={wrapRef}
          className="relative aspect-square rounded-md border border-[var(--line)] bg-[var(--panel)] grid place-items-center"
        >
          <style>{`
            @keyframes pk-stats-spin { to { transform: rotate(360deg); } }
            .pk-stats-spin { animation: pk-stats-spin 1.1s linear infinite; transform-origin: center; }
          `}</style>
          <div className="flex flex-col items-center gap-3 text-center px-6">
            <div className="relative grid place-items-center" style={{ width: ringSize, height: ringSize }}>
              {/* Progress ring: embeddings_ready / imports. */}
              <div
                aria-hidden
                className="absolute inset-0 rounded-full"
                style={{
                  background: `conic-gradient(rgb(var(--foreground-rgb)) ${pct * 3.6}deg, rgb(var(--foreground-rgb) / 0.10) ${pct * 3.6}deg)`,
                  // Hollow centre.
                  mask: "radial-gradient(circle, transparent 0 18px, #000 19px)",
                  WebkitMask: "radial-gradient(circle, transparent 0 18px, #000 19px)",
                }}
              />
              {/* Spinner on CSS keyframes: SMIL animateTransform does not
                  apply to a bare SVG path in every browser. */}
              <svg
                viewBox="0 0 24 24"
                className="pk-stats-spin h-7 w-7 text-foreground/55"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              >
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            </div>
            <span className="text-xs text-foreground/65">
              Computing embeddings
            </span>
          </div>
        </div>
      );
    }
    return (
      <div
        ref={wrapRef}
        className="relative aspect-square grid place-items-center rounded-md border border-[var(--line)] bg-[var(--panel)]"
      >
        <span className="h-7 w-7 animate-spin rounded-full border-2 border-foreground/15 border-t-[var(--accent)]" aria-label="Loading image variation" role="status" />
      </div>
    );
  }

  const inset = 8;
  const project = (v: number): number => inset + ((v + 1) / 2) * (100 - inset * 2);

  return (
    <div
      ref={wrapRef}
      className="relative aspect-square rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden"
    >
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        {[25, 50, 75].map((g) => (
          <g key={g} stroke="rgb(var(--foreground-rgb) / 0.06)" strokeWidth="0.4">
            <line x1={g} y1="0" x2={g} y2="100" />
            <line x1="0" y1={g} x2="100" y2={g} />
          </g>
        ))}
        {/* Augmentations first so the originals paint on top. */}
        {augmentations.map((p) => {
          const cx = project(p.x);
          const cy = project(p.y);
          const colour = colourFor(p.label);
          return (
            <circle
              key={p.id}
              cx={cx}
              cy={cy}
              r={1}
              fill={colour}
              fillOpacity={0.55}
              stroke={readableTextForBg(colour)}
              strokeOpacity={0.55}
              strokeWidth={0.35}
              style={{ cursor: onJumpToImport ? "pointer" : "default" }}
              onClick={(e) => {
                if (!onJumpToImport) return;
                e.stopPropagation();
                onJumpToImport(p.source_id, "augmentation");
              }}
              onPointerEnter={() => setHover({
                id: p.id,
                filename: p.filename,
                x: p.x,
                y: p.y,
                label: p.label,
                n_detections: 0,
              })}
              onPointerLeave={() => setHover((cur) => (cur?.id === p.id ? null : cur))}
            />
          );
        })}
        {points.map((p) => {
          const cx = project(p.x);
          const cy = project(p.y);
          const r = 1.0 + Math.min(2.0, p.n_detections * 0.35);
          const isDup = nearDuplicateIds.has(p.id);
          const colour = colourFor(p.label);
          return (
            <g
              key={p.id}
              onPointerEnter={() => setHover(p)}
              onPointerLeave={() => setHover((cur) => (cur?.id === p.id ? null : cur))}
              onClick={(e) => {
                if (!onJumpToImport) return;
                e.stopPropagation();
                onJumpToImport(p.id, "image");
              }}
              style={{ cursor: onJumpToImport ? "pointer" : "default" }}
            >
              {isDup && (
                <circle cx={cx} cy={cy} r={r + 1.6} fill="none" stroke="var(--warn)" strokeOpacity="0.75" strokeWidth="0.5" />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={colour}
                // Contrasts with the dot's own fill.
                stroke={readableTextForBg(colour)}
                strokeWidth="0.45"
                strokeOpacity="0.6"
              />
            </g>
          );
        })}
      </svg>
      {hover && (
        <div
          className="absolute bottom-2 left-2 right-2 px-2 py-1.5 rounded-md text-[11px] font-mono pointer-events-none truncate text-[var(--foreground)]"
          style={{
            background: "rgb(var(--background-rgb) / 0.55)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            border: "1px solid rgb(var(--foreground-rgb) / 0.08)",
          }}
        >
          <span className="truncate">{hover.filename ?? hover.id}</span>
          {hover.label && <span className="opacity-70 ml-2">· {hover.label}</span>}
          {hover.n_detections > 0 && <span className="opacity-70 ml-2">· {hover.n_detections} det.</span>}
          {nearDuplicateIds.has(hover.id) && <span className="opacity-80 ml-2 text-[var(--warn)]">· near-dup</span>}
        </div>
      )}
    </div>
  );
}
