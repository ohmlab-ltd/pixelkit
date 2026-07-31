"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  analyseImage,
  getQuantiseSourceModels,
  type AnalyseDetection,
  type AnalyseResult,
  type SourceModel,
} from "../../lib/mlJobs";

// Drag-drop an image → the backend runs the labelling pipeline, the trained
// (float) model and the quantised (int8) model. One preview overlays them all,
// colour-coded, with a confidence slider, hover-highlight, zoom/pan and a
// click-to-expand in-app lightbox. "Missed" = reference labels no shown model
// detection matches (recomputed live as the confidence cutoff moves).
type Layer = "float" | "int8" | "missed";

const LAYER_META: Record<Layer, { label: string; dot: string; box: string }> = {
  float: { label: "Trained", dot: "bg-amber-500", box: "border-amber-500" },
  int8: { label: "Quantised", dot: "bg-sky-500", box: "border-sky-500" },
  missed: { label: "Missed", dot: "bg-red-500", box: "border-red-500 border-dashed" },
};

const area = (b: number[]) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
function iou(a: number[], b: number[]): number {
  const ix0 = Math.max(a[0], b[0]), iy0 = Math.max(a[1], b[1]);
  const ix1 = Math.min(a[2], b[2]), iy1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
  const u = area(a) + area(b) - inter;
  return u > 0 ? inter / u : 0;
}

type Item = { key: string; layer: Layer; d: AnalyseDetection };

export function AnalysePanel({ projectId }: { projectId: string }) {
  const [models, setModels] = useState<SourceModel[] | null>(null);
  const [sourceJobId, setSourceJobId] = useState<string>("");
  const sourceRef = useRef<string>("");          // synchronous mirror for re-runs
  const fileRef = useRef<File | null>(null);     // last image, for model re-runs
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyseResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [show, setShow] = useState<Record<Layer, boolean>>({ float: true, int8: false, missed: true });
  const [conf, setConf] = useState(0.4);
  const [hovered, setHovered] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── zoom & pan (image + overlays share one transform layer) ──
  const vpRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);

  const clampView = useCallback((scale: number, tx: number, ty: number) => {
    scale = Math.min(8, Math.max(1, scale));
    if (scale <= 1) return { scale: 1, tx: 0, ty: 0 };
    const vp = vpRef.current;
    const w = vp?.clientWidth ?? 0, h = vp?.clientHeight ?? 0;
    return { scale, tx: Math.min(0, Math.max(w * (1 - scale), tx)), ty: Math.min(0, Math.max(h * (1 - scale), ty)) };
  }, []);
  const zoomAt = useCallback((mx: number, my: number, factor: number) => {
    setView((v) => {
      const ns = Math.min(8, Math.max(1, v.scale * factor));
      return clampView(ns, mx - ((mx - v.tx) / v.scale) * ns, my - ((my - v.ty) / v.scale) * ns);
    });
  }, [clampView]);

  useEffect(() => {
    const vp = vpRef.current;
    if (!vp) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    };
    vp.addEventListener("wheel", onWheel, { passive: false });
    return () => vp.removeEventListener("wheel", onWheel);
  }, [zoomAt, previewUrl, expanded]);

  // Esc closes the lightbox.
  useEffect(() => {
    if (!expanded) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [expanded]);

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (!d.moved && Math.abs(dx) + Math.abs(dy) > 5) d.moved = true;
    if (view.scale > 1) setView((v) => clampView(v.scale, d.tx + dx, d.ty + dy));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) setExpanded((x) => !x);   // a click (not a pan) → expand/collapse
  };
  const resetView = () => setView({ scale: 1, tx: 0, ty: 0 });

  useEffect(() => {
    let cancelled = false;
    getQuantiseSourceModels(projectId)
      .then((d) => { if (!cancelled) setModels(d.models); })
      .catch(() => { if (!cancelled) setModels([]); });
    return () => { cancelled = true; };
  }, [projectId]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const run = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) { setError("Please drop an image file."); return; }
    setError(null);
    setResult(null);
    setView({ scale: 1, tx: 0, ty: 0 });
    fileRef.current = file;
    const url = URL.createObjectURL(file);
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return url; });
    setBusy(true);
    try {
      setResult(await analyseImage(projectId, file, sourceRef.current || undefined));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [projectId]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void run(f);
  };

  const items = useMemo<Item[]>(() => {
    if (!result) return [];
    const vFloat = result.float.filter((d) => d.score >= conf);
    const vInt8 = (result.int8 ?? []).filter((d) => d.score >= conf);
    const vRef = result.reference.filter((d) => d.score >= conf);
    const missed = vRef.filter((r) => !vFloat.some((d) => d.label === r.label && iou(r.box_xyxy, d.box_xyxy) >= 0.5));
    const arr: Item[] = [];
    if (show.float) vFloat.forEach((d, i) => arr.push({ key: `f${i}`, layer: "float", d }));
    if (show.int8) vInt8.forEach((d, i) => arr.push({ key: `i${i}`, layer: "int8", d }));
    if (show.missed) missed.forEach((d, i) => arr.push({ key: `m${i}`, layer: "missed", d }));
    arr.sort((a, b) => area(b.d.box_xyxy) - area(a.d.box_xyxy));   // big first → small on top
    return arr;
  }, [result, conf, show]);

  const counts = useMemo(() => {
    if (!result) return { float: 0, int8: 0, missed: 0 };
    const vFloat = result.float.filter((d) => d.score >= conf);
    const vRef = result.reference.filter((d) => d.score >= conf);
    return {
      float: vFloat.length,
      int8: (result.int8 ?? []).filter((d) => d.score >= conf).length,
      missed: vRef.filter((r) => !vFloat.some((d) => d.label === r.label && iou(r.box_xyxy, d.box_xyxy) >= 0.5)).length,
    };
  }, [result, conf]);

  const missedList = useMemo(() => {
    if (!result) return [];
    const vFloat = result.float.filter((d) => d.score >= conf);
    return result.reference.filter((r) => r.score >= conf && !vFloat.some((d) => d.label === r.label && iou(r.box_xyxy, d.box_xyxy) >= 0.5));
  }, [result, conf]);

  const CARD = "rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02]";

  if (models !== null && models.length === 0) {
    return (
      <div className="px-6 lg:px-10 py-10">
        <Header />
        <div className={`pk-up mt-6 flex flex-col items-center gap-3 px-6 py-16 text-center ${CARD}`}>
          <div className="grid h-12 w-12 place-items-center rounded-full bg-foreground/[0.06] text-foreground/40">
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 21h16" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
          <div className="text-[17px] font-semibold tracking-tight">Train a model first</div>
          <p className="max-w-md text-[14px] leading-relaxed text-foreground/50">
            Analyse compares your trained model (and its quantised version) against the
            labelling pipeline on any image. Train a model on the Train tab to unlock it.
          </p>
        </div>
      </div>
    );
  }

  // The interactive stage (image + overlays + controls). Rendered EITHER inline
  // or inside the lightbox — never both at once, so vpRef binds to one element.
  const stage = (
    <>
      <div
        ref={vpRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => { drag.current = null; }}
        className={`relative w-full touch-none select-none overflow-hidden ${view.scale > 1 ? "cursor-grab active:cursor-grabbing" : expanded ? "cursor-zoom-out" : "cursor-zoom-in"}`}
      >
        {/* No will-change: keeps DOM text re-rasterised crisply under scale; each
            label also counter-scales to stay constant-size + sharp. */}
        <div className="relative origin-top-left" style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})` }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl ?? ""} alt="analysed" draggable={false}
            className={expanded ? "block h-auto max-h-[88vh] w-auto max-w-[92vw]" : "block h-auto w-full"} />
          {result && (
            <div className="pointer-events-none absolute inset-0">
              {items.map((it, idx) => {
                const W = result.size.width || 1, H = result.size.height || 1;
                const [x0, y0, x1, y1] = it.d.box_xyxy;
                const meta = LAYER_META[it.layer];
                const hl = hovered === it.key;
                const nearRight = x1 / W > 0.72;   // anchor label inward near the edge
                return (
                  <div
                    key={it.key}
                    onMouseEnter={() => setHovered(it.key)}
                    onMouseLeave={() => setHovered((h) => (h === it.key ? null : h))}
                    className={`pointer-events-auto absolute rounded-[2px] border ${meta.box} ${hl ? "ring-2 ring-white/80" : ""}`}
                    style={{
                      left: `${(x0 / W) * 100}%`, top: `${(y0 / H) * 100}%`,
                      width: `${((x1 - x0) / W) * 100}%`, height: `${((y1 - y0) / H) * 100}%`,
                      zIndex: hl ? 1000 : idx,
                      opacity: hovered && !hl ? 0.4 : 1,
                    }}
                  >
                    <span
                      className={`absolute top-0 max-w-[60vw] truncate whitespace-nowrap px-1 text-[10px] font-semibold leading-[1.45] text-white ${meta.dot} ${nearRight ? "right-0 origin-top-right rounded-bl-[3px] text-right" : "left-0 origin-top-left rounded-br-[3px]"}`}
                      style={{ transform: `scale(${1 / view.scale})` }}
                    >
                      {it.d.label} {Math.round(it.d.score * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* zoom controls */}
      <div className="absolute right-2 top-2 z-20 flex flex-col overflow-hidden rounded-lg border border-white/10 bg-black/50 text-lg text-white backdrop-blur" onPointerDown={(e) => e.stopPropagation()}>
        <button type="button" aria-label="Zoom in" className="grid h-8 w-8 place-items-center leading-none transition-colors hover:bg-white/15"
          onClick={() => { const vp = vpRef.current; if (vp) zoomAt(vp.clientWidth / 2, vp.clientHeight / 2, 1.3); }}>+</button>
        <button type="button" aria-label="Zoom out" className="grid h-8 w-8 place-items-center border-t border-white/10 leading-none transition-colors hover:bg-white/15"
          onClick={() => { const vp = vpRef.current; if (vp) zoomAt(vp.clientWidth / 2, vp.clientHeight / 2, 1 / 1.3); }}>−</button>
        <button type="button" aria-label="Reset zoom" className="grid h-8 w-8 place-items-center border-t border-white/10 transition-colors hover:bg-white/15"
          onClick={resetView}><svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="none"><path d="M4 8V5a1 1 0 011-1h3M13 4h2a1 1 0 011 1v2M16 12v3a1 1 0 01-1 1h-3M7 16H5a1 1 0 01-1-1v-2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></button>
      </div>

      {/* confidence slider + layer toggles */}
      <div className="absolute inset-x-0 bottom-0 z-20 flex flex-wrap items-center gap-x-4 gap-y-2 bg-gradient-to-t from-black/75 to-transparent px-3 pb-2.5 pt-7"
        onPointerDown={(e) => e.stopPropagation()}>
        <label className="flex items-center gap-2 text-[11px] font-medium text-white/85">
          <span className="tabular-nums">Conf ≥ {Math.round(conf * 100)}%</span>
          <input type="range" min={5} max={95} step={5} value={Math.round(conf * 100)} onChange={(e) => setConf(Number(e.target.value) / 100)} className="w-28 accent-amber-500" />
        </label>
        <div className="flex flex-wrap items-center gap-1.5">
          {(["float", "int8", "missed"] as Layer[]).map((l) => {
            if (l === "int8" && !result?.int8) return null;
            return (
              <button key={l} type="button" onClick={() => setShow((s) => ({ ...s, [l]: !s[l] }))}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors ${show[l] ? "bg-white/15 text-white" : "bg-white/5 text-white/40"}`}>
                <span className={`h-2 w-2 rounded-full ${LAYER_META[l].dot}`} />
                {LAYER_META[l].label} <span className="tabular-nums opacity-70">{counts[l]}</span>
              </button>
            );
          })}
        </div>
      </div>

      {!expanded && view.scale <= 1 && (
        <div className="pointer-events-none absolute left-2 top-2 z-20 rounded-full bg-black/50 px-2 py-0.5 text-[10px] text-white/70 backdrop-blur">click to expand · scroll to zoom</div>
      )}

      {busy && (
        <div className="absolute inset-0 z-30 grid place-items-center bg-background/50 backdrop-blur-[1px]">
          <div className="flex items-center gap-2 text-sm text-foreground/70">
            <svg viewBox="0 0 24 24" className="h-5 w-5 animate-spin" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.6" /><path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" /></svg>
            Analysing…
          </div>
        </div>
      )}
    </>
  );

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <Header />

      {/* model selector */}
      {models && models.length > 0 && (
        <div className="pk-up mt-5 flex items-center gap-3">
          <span className="text-[13px] text-foreground/55">Model</span>
          <div className="relative">
            <select
              value={sourceJobId}
              onChange={(e) => {
                const v = e.target.value;
                setSourceJobId(v); sourceRef.current = v;
                if (fileRef.current) void run(fileRef.current);
              }}
              className="cursor-pointer appearance-none rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2 pr-9 text-sm font-medium outline-none transition-colors hover:border-foreground/25"
            >
              <option value="">Latest trained</option>
              {models.map((m) => (
                <option key={m.source_job_id} value={m.source_job_id}>{m.name || m.source_job_id.slice(0, 8)}</option>
              ))}
            </select>
            <svg className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" viewBox="0 0 20 20" fill="none"><path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
        </div>
      )}

      {/* drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`pk-up mt-5 cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${dragOver ? "border-amber-500 bg-amber-500/[0.04]" : "border-foreground/15 hover:border-foreground/30 bg-foreground/[0.02]"}`}
        style={{ animationDelay: "40ms" }}
      >
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void run(f); }} />
        <div className="text-[15px] font-medium text-foreground/70">{previewUrl ? "Drop another image to re-analyse" : "Drag & drop an image to analyse"}</div>
        <div className="mt-1 text-[12px] text-foreground/40">runs the labelling pipeline, your trained model, and the quantised model</div>
      </div>

      {error && <div className="pk-up mt-4 rounded-2xl bg-red-500/10 px-3.5 py-2.5 text-sm text-red-500 ring-1 ring-inset ring-red-500/20">{error}</div>}

      {previewUrl && !expanded && (
        <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1fr)_280px]">
          <div className={`pk-up relative overflow-hidden ${CARD}`}>{stage}</div>

          <div className="flex flex-col gap-4">
            <div className={`pk-up p-4 ${CARD}`}>
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-wider text-foreground/40">Missed by trained model</span>
                <span className="text-[13px] font-semibold tabular-nums text-foreground/70">{counts.missed}</span>
              </div>
              {missedList.length > 0 ? (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {missedList.map((d, i) => (
                    <span key={i} className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2.5 py-1 text-[12px] font-medium text-red-500 ring-1 ring-inset ring-red-500/20">{d.label} {Math.round(d.score * 100)}%</span>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-[12px] text-foreground/40">{result ? "Nothing missed at this cutoff." : "—"}</p>
              )}
              {result && !result.quantised && (
                <p className="mt-3 text-[11px] leading-relaxed text-foreground/40">No quantised model yet — quantise on the Quantise tab to compare it here.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* in-app lightbox: theme-aware blurred backdrop filling the window, image
          floated on top with rounded edges. NOT browser fullscreen. */}
      {previewUrl && expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 p-4 backdrop-blur-xl sm:p-8"
          onClick={() => setExpanded(false)}
        >
          <div className="relative overflow-hidden rounded-2xl shadow-2xl ring-1 ring-foreground/10" onClick={(e) => e.stopPropagation()}>
            {stage}
            <button type="button" aria-label="Close" onClick={() => setExpanded(false)}
              className="absolute left-2 top-2 z-30 grid h-8 w-8 place-items-center rounded-full bg-black/50 text-white transition-colors hover:bg-black/70" onPointerDown={(e) => e.stopPropagation()}>
              <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Header() {
  return (
    <header className="pk-up">
      <style dangerouslySetInnerHTML={{ __html:
        "@keyframes pkUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}"
        + ".pk-up{animation:pkUp .55s cubic-bezier(.16,1,.3,1) both}"
      }} />
      <h2 className="text-[28px] font-semibold tracking-tight">Analyse</h2>
      <p className="mt-1.5 text-[15px] text-foreground/50">
        Compare your trained model and its quantised version against the labelling pipeline on any image.
      </p>
    </header>
  );
}
