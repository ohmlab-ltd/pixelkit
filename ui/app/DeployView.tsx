"use client";

import { useState } from "react";

const QUANT_FORMATS = [
  { id: "int8-pc", label: "INT8 per-channel", blurb: "Best accuracy. Recommended for STM32N6 NPU." },
  { id: "int8-pt", label: "INT8 per-tensor", blurb: "Slightly smaller, slightly less accurate." },
  { id: "int16", label: "INT16", blurb: "Higher precision, ~2× memory and latency." },
  { id: "fp16", label: "FP16", blurb: "Float fallback for layers the NPU can't INT-quantise." },
] as const;
type QuantId = (typeof QUANT_FORMATS)[number]["id"];

const OPT_LEVELS = [
  { id: "balanced", label: "Balanced", blurb: "Default. Good size/latency trade-off." },
  { id: "latency", label: "Latency", blurb: "Optimise for fewer cycles." },
  { id: "ram", label: "RAM", blurb: "Minimise activation memory." },
  { id: "rom", label: "ROM", blurb: "Minimise weights footprint." },
] as const;
type OptId = (typeof OPT_LEVELS)[number]["id"];

const ARTEFACTS = [
  { id: "onnx", label: "ONNX", ext: ".onnx", blurb: "Float reference for sanity-check on PC." },
  { id: "tflite", label: "TFLite (quantised)", ext: ".tflite", blurb: "Quantised graph for cross-checking." },
  { id: "stedge", label: "ST Edge AI C source", ext: ".zip", blurb: "network.c / .h / weights for STM32CubeIDE." },
  { id: "report", label: "Validation report", ext: ".html", blurb: "Per-layer error vs FP32 reference." },
] as const;

export function DeployView({ projectName, hasModel }: { projectName: string; hasModel: boolean }) {
  const [quant, setQuant] = useState<QuantId>("int8-pc");
  const [optLevel, setOptLevel] = useState<OptId>("balanced");
  const [calibrationPct, setCalibrationPct] = useState(0.2);
  const [includeNonNpu, setIncludeNonNpu] = useState(true);
  const [calibrating, setCalibrating] = useState(false);
  const [converting, setConverting] = useState(false);

  const runCalibration = async () => {
    setCalibrating(true);
    await new Promise((r) => setTimeout(r, 900));
    setCalibrating(false);
    alert(
      `Quantisation preview only.\n\n` +
        `Format: ${quant}\n` +
        `Calibration: ${(calibrationPct * 100).toFixed(0)}% of labelled images\n` +
        `Include non-NPU layers: ${includeNonNpu}`,
    );
  };

  const runConversion = async () => {
    setConverting(true);
    await new Promise((r) => setTimeout(r, 900));
    setConverting(false);
    alert(`ST Edge AI conversion preview.\n\nOptimisation: ${optLevel}\nQuantisation: ${quant}`);
  };

  const download = (id: string, ext: string) => {
    alert(`Download is a preview. Would emit:\n\n${projectName}_${id}${ext}`);
  };

  return (
    <section className="mx-auto max-w-6xl px-6 pt-12 pb-24 grid gap-12">
      <div>
        <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Deploy</div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mt-1">
          Quantise, convert, ship.
        </h2>
        <p className="mt-3 max-w-xl text-[var(--muted)]">
          Context-aware quantisation calibrated on your project images, then ST Edge AI emits a
          C model ready for STM32CubeIDE. Placeholder, nothing is built yet.
        </p>
      </div>

      <div
        className="rounded-2xl border border-amber-300/30 bg-amber-300/[0.05] px-5 py-3 flex items-start gap-3"
        style={{ boxShadow: "0 0 24px rgba(251, 146, 60, 0.08), 0 0 48px rgba(251, 146, 60, 0.04)" }}
      >
        <span className="rounded-full bg-amber-300/20 border border-amber-300/40 text-amber-200 px-2 py-0.5 text-[10px] uppercase tracking-wider mt-0.5">Beta</span>
        <p className="text-sm text-foreground/80 leading-relaxed">
          Models deployed with Pixel Kit must be validated by you. Pixel Kit does not
          guarantee model accuracy, safety, compliance or suitability for any use case
         , including production, regulated or safety-critical use.
        </p>
      </div>

      {!hasModel && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/[0.05] px-4 py-3 text-sm text-amber-200">
          Train a model on the Train tab before deploying.
        </div>
      )}

      {/* Quantisation */}
      <div className="rounded-xl border border-[var(--border)] p-5 grid gap-5">
        <div className="flex items-center justify-between">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)]">
            Context-aware quantisation
          </div>
          <button
            onClick={runCalibration}
            disabled={!hasModel || calibrating}
            className="rounded-full bg-foreground text-background px-4 py-2 text-xs font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {calibrating ? "Calibrating…" : "Calibrate & quantise"}
          </button>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          {QUANT_FORMATS.map((q) => {
            const active = quant === q.id;
            return (
              <button
                key={q.id}
                onClick={() => setQuant(q.id)}
                className={[
                  "text-left rounded-lg border p-4 transition-colors",
                  active
                    ? "border-[var(--foreground)] bg-foreground/[0.06]"
                    : "border-[var(--border)] hover:border-zinc-500",
                ].join(" ")}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={[
                      "h-3 w-3 rounded-full border transition-colors",
                      active ? "bg-white border-[var(--foreground)]" : "border-zinc-600",
                    ].join(" ")}
                  />
                  <span className="text-sm font-medium">{q.label}</span>
                </div>
                <div className="mt-1 ml-5 text-xs text-[var(--muted)]">{q.blurb}</div>
              </button>
            );
          })}
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <div>
            <div className="text-xs text-[var(--muted)] mb-2">Calibration sample size</div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={calibrationPct}
                onChange={(e) => setCalibrationPct(parseFloat(e.target.value))}
                className="flex-1 accent-white"
              />
              <span className="font-mono text-sm tabular-nums w-16 text-right">
                {(calibrationPct * 100).toFixed(0)}%
              </span>
            </div>
            <p className="mt-2 text-xs text-[var(--muted)]">
              Fraction of labelled images used to set per-layer activation ranges.
            </p>
          </div>
          <label className="flex items-center justify-between gap-4 cursor-pointer self-end">
            <span className="text-sm">Include non-NPU layers (FP fallback)</span>
            <button
              type="button"
              onClick={() => setIncludeNonNpu(!includeNonNpu)}
              className={[
                "relative h-6 w-11 rounded-full transition-colors",
                includeNonNpu ? "bg-white" : "bg-[var(--border)]",
              ].join(" ")}
              aria-pressed={includeNonNpu}
            >
              <span
                className={[
                  "absolute top-0.5 h-5 w-5 rounded-full transition-transform",
                  includeNonNpu ? "left-[1.5rem] bg-[var(--background)]" : "left-0.5 bg-zinc-400",
                ].join(" ")}
              />
            </button>
          </label>
        </div>
      </div>

      {/* ST Edge AI conversion */}
      <div className="rounded-xl border border-[var(--border)] p-5 grid gap-5">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <div className="text-xs uppercase tracking-wider text-[var(--muted)]">
              ST Edge AI conversion
            </div>
            <div className="mt-1 text-sm">
              Target: <span className="font-mono">STM32N6</span>
              <span className="text-[var(--muted)]"> · Neuro N6 NPU</span>
            </div>
          </div>
          <button
            onClick={runConversion}
            disabled={!hasModel || converting}
            className="rounded-full bg-foreground text-background px-4 py-2 text-xs font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {converting ? "Converting…" : "Generate C model"}
          </button>
        </div>

        <div>
          <div className="text-xs text-[var(--muted)] mb-2">Optimisation</div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {OPT_LEVELS.map((o) => {
              const active = optLevel === o.id;
              return (
                <button
                  key={o.id}
                  onClick={() => setOptLevel(o.id)}
                  className={[
                    "text-left rounded-lg border p-3 transition-colors",
                    active
                      ? "border-[var(--foreground)] bg-foreground/[0.06]"
                      : "border-[var(--border)] hover:border-zinc-500",
                  ].join(" ")}
                >
                  <div className="text-sm font-medium">{o.label}</div>
                  <div className="mt-1 text-xs text-[var(--muted)]">{o.blurb}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Downloads */}
      <div className="rounded-xl border border-[var(--border)] p-5 grid gap-4">
        <div className="text-xs uppercase tracking-wider text-[var(--muted)]">Downloads</div>
        <div className="grid gap-2 sm:grid-cols-2">
          {ARTEFACTS.map((a) => (
            <button
              key={a.id}
              onClick={() => download(a.id, a.ext)}
              disabled={!hasModel}
              className={[
                "text-left rounded-lg border p-4 transition-colors flex items-center justify-between gap-4",
                hasModel
                  ? "border-[var(--border)] hover:border-zinc-500"
                  : "border-[var(--border)] opacity-50 cursor-not-allowed",
              ].join(" ")}
            >
              <div>
                <div className="text-sm font-medium">{a.label}</div>
                <div className="mt-1 text-xs text-[var(--muted)]">{a.blurb}</div>
              </div>
              <span className="rounded-full bg-foreground/10 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider shrink-0">
                {a.ext}
              </span>
            </button>
          ))}
        </div>
        <p className="text-xs text-[var(--muted)]">
          Bundle artefacts will appear once a model has been trained, quantised, and converted.
        </p>
      </div>
    </section>
  );
}
