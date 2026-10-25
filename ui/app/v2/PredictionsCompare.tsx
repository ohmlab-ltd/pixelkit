"use client";

// A run's val images, each as ground truth and as the exported int8 model's
// output at the shipped threshold, worst first.

import { useEffect, useMemo, useState } from "react";

import { API_BASE, apiFetch } from "@/lib/apiFetch";
import { errorCounts, worstFirst, type RunPredictions, type ValPrediction } from "@/lib/trainRuns";

import { GlassDialog } from "./GlassDialog";

const COLOUR = {
  truth: "#28c85a",
  missed: "#fac828",
  tp: "#4682f0",
  wrong: "#fa821e",
  fp: "#e63c3c",
} as const;

const LEGEND: [string, string][] = [
  [COLOUR.truth, "truth"],
  [COLOUR.missed, "missed"],
  [COLOUR.tp, "correct"],
  [COLOUR.wrong, "wrong class"],
  [COLOUR.fp, "spurious"],
];

type Drawn = { box: number[]; colour: string; label: string };

export function PredictionsCompare({
  projectId,
  run,
  onClose,
}: {
  projectId: string;
  run: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<RunPredictions | null>(null);
  const [failed, setFailed] = useState(false);
  const [errorsOnly, setErrorsOnly] = useState(true);
  const [at, setAt] = useState(0);

  useEffect(() => {
    let alive = true;
    apiFetch(`/api/v2/projects/${projectId}/models/${run}/predictions`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: RunPredictions) => { if (alive) setData(d); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [projectId, run]);

  const ranked = useMemo(() => worstFirst(data?.images ?? []), [data]);
  const withErrors = useMemo(
    () => ranked.filter((p) => Object.values(errorCounts(p)).some((n) => n > 0)),
    [ranked],
  );
  const images = errorsOnly && withErrors.length > 0 ? withErrors : ranked;
  const i = Math.min(at, Math.max(0, images.length - 1));
  const cur: ValPrediction | undefined = images[i];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setAt((a) => Math.min(a + 1, images.length - 1));
      else if (e.key === "ArrowLeft") setAt((a) => Math.max(a - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [images.length]);

  const name = (c: number) => data?.classes[c] ?? `class ${c}`;
  const src = (p: ValPrediction) =>
    `${API_BASE}/api/v2/projects/${p.project}/imports/${encodeURIComponent(p.image)}`;
  const errors = cur ? errorCounts(cur) : null;
  const summary = errors
    ? [
        errors.missed ? `${errors.missed} missed` : "",
        errors.wrong ? `${errors.wrong} wrong class` : "",
        errors.spurious ? `${errors.spurious} spurious` : "",
      ].filter(Boolean).join(" · ") || "all correct"
    : "";

  return (
    <GlassDialog open onClose={onClose} title="Val images: truth and model" maxWidth="max-w-5xl">
      {failed ? (
        <p className="text-[12px] text-[var(--bad)]">Couldn&rsquo;t load this run&rsquo;s predictions.</p>
      ) : !data ? (
        <p className="text-[12px] text-foreground/55">Loading…</p>
      ) : !cur ? (
        <p className="text-[12px] text-foreground/55">This run has no val images.</p>
      ) : (
        <div className="grid min-w-0 gap-3">
          <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[12px]">
            <span className="min-w-0 truncate text-foreground/75" title={cur.image}>{cur.image}</span>
            <span className="tabular-nums text-foreground/55">{summary}</span>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Panel
              title="Ground truth"
              src={src(cur)}
              w={cur.width}
              h={cur.height}
              boxes={cur.truth.map((t) => ({
                box: t.box,
                colour: t.missed ? COLOUR.missed : COLOUR.truth,
                label: t.missed ? `${name(t.class)} (missed)` : name(t.class),
              }))}
            />
            <Panel
              title={`Model at ${data.conf.toFixed(2)}`}
              src={src(cur)}
              w={cur.width}
              h={cur.height}
              boxes={cur.predicted.map((p) => ({
                box: p.box,
                colour: COLOUR[p.kind],
                label: `${name(p.class)} ${p.score.toFixed(2)}`,
              }))}
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 text-[11.5px] text-foreground/55">
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              {LEGEND.map(([colour, label]) => (
                <span key={label} className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colour }} />
                  {label}
                </span>
              ))}
            </span>
            <span className="flex items-center gap-2">
              {withErrors.length > 0 && withErrors.length < ranked.length && (
                <label className="inline-flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={errorsOnly}
                    onChange={(e) => { setErrorsOnly(e.target.checked); setAt(0); }}
                  />
                  only the {withErrors.length} with errors
                </label>
              )}
              <button type="button" className="pk-btn" onClick={() => setAt(Math.max(i - 1, 0))} disabled={i === 0}>
                ‹ Previous
              </button>
              <span className="tabular-nums">{i + 1} / {images.length}</span>
              <button
                type="button"
                className="pk-btn"
                onClick={() => setAt(Math.min(i + 1, images.length - 1))}
                disabled={i >= images.length - 1}
              >
                Next ›
              </button>
            </span>
          </div>
        </div>
      )}
    </GlassDialog>
  );
}

// Boxes are in the image's own pixel coordinates.
function Panel({ title, src, w, h, boxes }: { title: string; src: string; w: number; h: number; boxes: Drawn[] }) {
  const side = Math.max(w, h);
  const stroke = Math.max(1.5, side / 320);
  const font = Math.max(10, side / 40);
  return (
    <figure className="min-w-0">
      <figcaption className="pk-micro mb-1 text-foreground/45">{title}</figcaption>
      <div className="relative overflow-hidden rounded-md border border-[var(--line)] bg-black/40">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt="" className="block w-full" />
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          {boxes.map((b, k) => {
            const [x0, y0, x1, y1] = b.box;
            return (
              <g key={k}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={b.colour} strokeWidth={stroke} />
                <text
                  x={x0 + stroke}
                  y={y0 > font * 1.2 ? y0 - stroke * 2 : y1 + font}
                  fontSize={font}
                  fill={b.colour}
                  stroke="rgba(0,0,0,0.8)"
                  strokeWidth={font / 6}
                  paintOrder="stroke"
                >
                  {b.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </figure>
  );
}
