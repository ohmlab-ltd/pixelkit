"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// Models: SSDLite + MobileNetV3-Large is the only one wired to the backend
// for now (BSD-3 licence via torchvision, ~3.4M params, exports to ONNX
// for STM32CubeAI). The other entries remain in the picker behind a
// "Coming soon" overlay so the UI shape doesn't have to change when we
// add them.
const MODELS = [
  {
    id: "ssdlite-mobilenetv3",
    name: "SSDLite + MobileNetV3-L",
    family: "Detection",
    params: "3.4M",
    blurb: "Compact detector. BSD-3, exports to ONNX for STM32CubeAI.",
    defaultInputSize: 320,
  },
  {
    id: "yolov8n",
    name: "YOLOv8n",
    family: "Detection",
    params: "3.2M",
    blurb: "Small detector. Fast on edge, decent accuracy.",
    defaultInputSize: 320,
  },
  {
    id: "ssd-mobilenet",
    name: "SSD-MobileNet",
    family: "Detection",
    params: "6.8M",
    blurb: "Single-shot detector with MobileNet backbone.",
    defaultInputSize: 300,
  },
  {
    id: "mobilenetv2",
    name: "MobileNetV2",
    family: "Classification",
    params: "3.5M",
    blurb: "Inverted residuals, depthwise separable convs.",
    defaultInputSize: 224,
  },
  {
    id: "efficientnet-lite",
    name: "EfficientNet-Lite",
    family: "Classification",
    params: "4.7M",
    blurb: "Quantization-friendly EfficientNet variant.",
    defaultInputSize: 224,
  },
] as const;

const INPUT_SIZES = [96, 128, 160, 192, 224, 256, 300, 320, 416, 512, 640];

const AUGMENTATIONS = [
  { id: "hflip", label: "Horizontal flip", blurb: "Mirror left↔right" },
  { id: "vflip", label: "Vertical flip", blurb: "Mirror top↕bottom" },
  { id: "rotate", label: "Random rotation", blurb: "±15°" },
  { id: "scale", label: "Random scale", blurb: "0.5×–1.5× zoom" },
  { id: "crop", label: "Random crop", blurb: "Crop & pad" },
  { id: "color", label: "Colour jitter", blurb: "Brightness, contrast, saturation" },
  { id: "hsv", label: "HSV shift", blurb: "Hue and saturation" },
  { id: "blur", label: "Gaussian blur", blurb: "Light defocus" },
  { id: "noise", label: "Gaussian noise", blurb: "Sensor-style grain" },
  { id: "cutout", label: "Cutout / erase", blurb: "Random rectangle erase" },
  { id: "mosaic", label: "Mosaic", blurb: "4-image collage (YOLO)" },
  { id: "mixup", label: "Mixup", blurb: "Blend two samples" },
] as const;

type SizeFilter = "all" | "exclude_fail" | "exclude_warn";

type ProgressEvt = {
  index: number;
  total: number;
  train_loss?: number;
  val_loss?: number;
};

type ProjectModelInfo = {
  kind?: string;
  classes?: string[];
  imgsz?: number;
  trained_at?: string;
  summary?: { n_train?: number; n_val?: number; epochs?: number; best_val_loss?: number };
} | null;

export function TrainView({ projectName, n_labeled }: { projectName: string; n_labeled: number }) {
  const [inputSize, setInputSize] = useState(320);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [includeVlmRejected, setIncludeVlmRejected] = useState(false);
  const [sizeFilter, setSizeFilter] = useState<SizeFilter>("exclude_fail");

  const [jobId, setJobId] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "waiting" | "running" | "done" | "error">("idle");
  const [progress, setProgress] = useState<ProgressEvt>({ index: 0, total: epochs });
  const [error, setError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ProjectModelInfo>(null);
  const esRef = useRef<EventSource | null>(null);

  // Pull current model info on mount, and after each completed run, so we
  // know whether to show the download button (and what to label it).
  // Wrapped in useCallback so the SSE useEffect can list it as a
  // dependency without re-subscribing on every render.
  const refreshModelInfo = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/projects/${projectName}`, { cache: "no-store" });
      if (!r.ok) return;
      const m = await r.json();
      setModelInfo((m?.model as ProjectModelInfo) ?? null);
    } catch {
      // Silent: no big deal if a status refresh fails; the button just
      // won't appear until the next reload.
    }
  }, [projectName]);
  useEffect(() => {
    refreshModelInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectName]);

  const downloadModel = () => {
    window.location.href = `${API}/api/projects/${projectName}/model/download`;
  };

  const start = async () => {
    if (jobId) return;
    setError(null);
    setPhase("waiting");
    setProgress({ index: 0, total: epochs });
    try {
      const minBoxPx = sizeFilter === "exclude_warn" ? 24 : sizeFilter === "exclude_fail" ? 12 : 0;
      const r = await fetch(`${API}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: projectName,
          kind: "train",
          params: {
            epochs,
            batch: batchSize,
            imgsz: inputSize,
            learning_rate: learningRate,
            include_vlm_rejected: includeVlmRejected,
            min_box_px: minBoxPx,
          },
        }),
      });
      if (!r.ok) throw new Error(await r.text() || `http ${r.status}`);
      const d = await r.json();
      setJobId(d.jobId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    try {
      await fetch(`${API}/api/projects/${projectName}/jobs/${jobId}`, { method: "DELETE" });
    } catch {
      // SSE cancelled handler clears local state
    }
  };

  // SSE listener for the active train job.
  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`${API}/api/jobs/${jobId}/events`);
    esRef.current = es;
    es.addEventListener("status", (ev: MessageEvent) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.phase === "waiting") setPhase("waiting");
        else if (d.phase === "preparing" || d.phase === "running") setPhase("running");
      } catch {
        // ignore malformed status frames
      }
    });
    es.addEventListener("progress", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data) as ProgressEvt;
      setPhase("running");
      setProgress(d);
    });
    const close = () => {
      es.close();
      esRef.current = null;
      setJobId(null);
    };
    es.addEventListener("complete", () => {
      setPhase("done");
      close();
      refreshModelInfo();
    });
    es.addEventListener("done", () => {
      setPhase("done");
      close();
      refreshModelInfo();
    });
    es.addEventListener("failed", (ev: MessageEvent) => {
      let msg = "training failed";
      try { msg = JSON.parse(ev.data).error || msg; } catch {
        // backend sometimes emits failures without a payload
      }
      setError(msg);
      setPhase("error");
      close();
    });
    es.addEventListener("cancelled", () => {
      setPhase("idle");
      close();
    });
    es.onerror = () => {
      close();
      setPhase((cur) => (cur === "done" || cur === "error" ? cur : "error"));
    };
    return () => es.close();
  }, [jobId, projectName, refreshModelInfo]);

  const pct = progress.total > 0 ? Math.min(100, (progress.index / progress.total) * 100) : 0;
  const busy = !!jobId;

  return (
    <section className="mx-auto max-w-6xl px-6 pt-12 pb-24 grid gap-10">
      <WipBanner />

      <div>
        <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Train</div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mt-1">
          Train a baseline model
        </h2>
        <p className="mt-3 max-w-xl text-[var(--muted)]">
          Fine-tunes <span className="font-mono text-foreground/80">SSDLite + MobileNetV3-L</span> on
          your labelled data, no augmentations. STM32N6-friendly and ONNX-exportable for STM32CubeAI.
        </p>
      </div>

      <ActiveModelCard />

      <div className="rounded-xl border border-[var(--border)] p-5 grid gap-5">
        <div className="text-xs uppercase tracking-wider text-[var(--muted)]">Training settings</div>

        <Field label="Input size">
          <div className="flex gap-1 flex-wrap">
            {INPUT_SIZES.map((s) => (
              <button
                key={s}
                onClick={() => setInputSize(s)}
                disabled={busy}
                className={[
                  "rounded-full px-3 py-1 text-xs font-mono border transition-colors disabled:opacity-40",
                  s === inputSize
                    ? "bg-foreground text-background border-[var(--foreground)]"
                    : "border-[var(--border)] text-[var(--muted)] hover:border-zinc-500 hover:text-foreground",
                ].join(" ")}
              >
                {s}
              </button>
            ))}
          </div>
        </Field>

        <div className="grid gap-5 md:grid-cols-3">
          <NumberField label="Epochs" value={epochs} onChange={(v) => setEpochs(Math.min(100, Math.max(1, v)))} min={1} max={100} step={1} disabled={busy} hint="max 100 for now" />
          <NumberField label="Batch size" value={batchSize} onChange={(v) => setBatchSize(Math.min(16, Math.max(1, v)))} min={1} max={16} step={1} disabled={busy} hint="max 16 for now" />
          <Field label="Learning rate">
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={-5}
                max={-1}
                step={0.25}
                value={Math.log10(learningRate)}
                onChange={(e) => setLearningRate(parseFloat(Math.pow(10, parseFloat(e.target.value)).toPrecision(2)))}
                disabled={busy}
                className="flex-1 accent-white"
              />
              <span className="font-mono text-sm tabular-nums w-20 text-right">{learningRate.toExponential(1)}</span>
            </div>
          </Field>
        </div>

        <Field label="Rejected boxes">
          <ToggleRow
            value={includeVlmRejected}
            onChange={setIncludeVlmRejected}
            disabled={busy}
            offLabel="Excluded (default)"
            onLabel="Included as positive samples"
          />
        </Field>

        <Field label="Box-size filter">
          <div className="flex flex-col gap-1">
            {([
              { id: "exclude_fail", label: "Exclude won’t-detect (too small)", note: "default" },
              { id: "exclude_warn", label: "Also exclude small (borderline)", note: "stricter" },
              { id: "all", label: "Include every box", note: "no filter" },
            ] as const).map((opt) => (
              <label
                key={opt.id}
                className={[
                  "flex items-center gap-3 rounded-lg border px-3 py-2 cursor-pointer transition-colors",
                  sizeFilter === opt.id
                    ? "border-foreground/40 bg-foreground/[0.04]"
                    : "border-[var(--border)] hover:border-zinc-500",
                  busy ? "opacity-60 cursor-not-allowed" : "",
                ].join(" ")}
              >
                <input
                  type="radio"
                  name="size-filter"
                  checked={sizeFilter === opt.id}
                  onChange={() => setSizeFilter(opt.id)}
                  disabled={busy}
                  className="accent-white"
                />
                <span className="text-sm flex-1">{opt.label}</span>
                <span className="text-[10px] text-[var(--muted)] uppercase tracking-wider">{opt.note}</span>
              </label>
            ))}
          </div>
        </Field>
      </div>

      <ComingSoonBlock title="Model picker">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {MODELS.map((m) => (
            <div key={m.id} className="rounded-xl border border-[var(--border)] p-4">
              <div className="flex items-baseline justify-between gap-2">
                <div className="font-medium">{m.name}</div>
                <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{m.family}</div>
              </div>
              <div className="mt-2 text-xs text-[var(--muted)]">~<span className="font-mono">{m.params}</span> params</div>
              <div className="mt-3 text-sm text-[var(--muted)]">{m.blurb}</div>
            </div>
          ))}
        </div>
      </ComingSoonBlock>

      <ComingSoonBlock title="Augmentations">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {AUGMENTATIONS.map((a) => (
            <div key={a.id} className="rounded-lg border border-[var(--border)] px-3 py-2.5">
              <div className="flex items-center gap-2">
                <span className="h-3.5 w-3.5 rounded-sm border border-zinc-600" />
                <span className="text-sm font-medium">{a.label}</span>
              </div>
              <div className="mt-1 ml-5 text-xs text-[var(--muted)]">{a.blurb}</div>
            </div>
          ))}
        </div>
      </ComingSoonBlock>

      <div className="rounded-xl border border-[var(--border)] p-5 flex items-center justify-between gap-6 flex-wrap">
        <div className="flex items-center gap-8 flex-wrap">
          <div className="text-sm">
            <div className="text-[var(--muted)] text-xs uppercase tracking-wider">Dataset</div>
            <div className="mt-1">
              <span className="font-mono">{n_labeled}</span> labelled image{n_labeled === 1 ? "" : "s"}
            </div>
          </div>
          <div className="text-sm">
            <div className="text-[var(--muted)] text-xs uppercase tracking-wider">Phase</div>
            <div className="mt-1 flex items-center gap-2">
              <PhaseDot phase={phase} />
              <span className="font-mono">{phaseLabel(phase)}</span>
              {phase === "running" && (
                <span className="text-[var(--muted)] text-xs">
                  · epoch <span className="font-mono">{progress.index}</span>/<span className="font-mono">{progress.total}</span>
                  {progress.val_loss != null && <> · val <span className="font-mono">{progress.val_loss.toFixed(3)}</span></>}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {modelInfo && !busy && (
            <button
              onClick={downloadModel}
              className="inline-flex items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-500/10 hover:bg-emerald-500/15 hover:border-emerald-400/60 text-emerald-100 px-4 py-2 text-sm transition-colors"
              title={
                modelInfo.trained_at
                  ? `Trained ${new Date(modelInfo.trained_at).toLocaleString()}`
                  : "Download trained model"
              }
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download model
            </button>
          )}
          {busy && (
            <button
              onClick={cancel}
              className="rounded-full border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 hover:border-foreground/30 px-5 py-2 text-sm transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            onClick={start}
            disabled={busy || n_labeled === 0}
            className="rounded-full bg-foreground text-background px-6 py-3 text-sm font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Training…" : modelInfo ? "Re-train" : "Start training"}
          </button>
        </div>
      </div>

      {phase === "running" && progress.total > 0 && (
        <div className="rounded-xl border border-[var(--border)] p-5 grid gap-3">
          <div className="flex items-baseline justify-between">
            <div className="text-xs uppercase tracking-wider text-[var(--muted)]">Progress</div>
            <div className="text-sm tabular-nums">{Math.round(pct)}%</div>
          </div>
          <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-foreground/[0.06]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                background: "linear-gradient(90deg, #fb923c, #f97316)",
                boxShadow: "0 0 14px rgba(249,115,22,0.5)",
              }}
            />
          </div>
          {(progress.train_loss != null || progress.val_loss != null) && (
            <div className="flex gap-6 text-xs text-[var(--muted)]">
              {progress.train_loss != null && <span>train loss <span className="font-mono text-foreground/80">{progress.train_loss.toFixed(4)}</span></span>}
              {progress.val_loss != null && <span>val loss <span className="font-mono text-foreground/80">{progress.val_loss.toFixed(4)}</span></span>}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/[0.06] px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}
    </section>
  );
}

function phaseLabel(p: "idle" | "waiting" | "running" | "done" | "error"): string {
  switch (p) {
    case "idle": return "ready";
    case "waiting": return "waiting for queue to drain…";
    case "running": return "training";
    case "done": return "done";
    case "error": return "error";
  }
}

function PhaseDot({ phase }: { phase: "idle" | "waiting" | "running" | "done" | "error" }) {
  const colour = phase === "running" || phase === "waiting"
    ? "bg-amber-300"
    : phase === "done"
    ? "bg-emerald-400"
    : phase === "error"
    ? "bg-red-400"
    : "bg-foreground/30";
  return (
    <span className="relative flex h-2.5 w-2.5">
      {(phase === "running" || phase === "waiting") && (
        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${colour} opacity-70`} />
      )}
      <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${colour}`} />
    </span>
  );
}

function ActiveModelCard() {
  return (
    <div
      className="rounded-2xl border border-foreground/10 px-5 py-4"
      style={{
        background:
          "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 45%, rgba(255,255,255,0) 100%), #141416",
        boxShadow: "0 1px 0 rgb(var(--foreground-rgb) / 0.05) inset",
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-[var(--muted)]">Model</div>
          <div className="mt-0.5 text-lg font-medium">PixelKit Edge Detector</div>
        </div>
        <span className="rounded-full bg-emerald-500/15 border border-emerald-400/40 text-emerald-200 px-2.5 py-0.5 text-[10px] uppercase tracking-wider">Active</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs text-[var(--muted)]">
        <span>Lightweight</span>
        <span>Exportable to STM32</span>
        <span>No augmentations</span>
      </div>
    </div>
  );
}

function WipBanner() {
  return (
    <div
      className="rounded-2xl border border-amber-300/30 bg-amber-300/[0.05] px-5 py-3 flex items-start gap-3"
      style={{ boxShadow: "0 0 24px rgba(251, 146, 60, 0.08), 0 0 48px rgba(251, 146, 60, 0.04)" }}
    >
      <span className="rounded-full bg-amber-300/20 border border-amber-300/40 text-amber-200 px-2 py-0.5 text-[10px] uppercase tracking-wider mt-0.5">Beta</span>
      <p className="text-sm text-foreground/80 leading-relaxed">
        Models trained or deployed with Pixel Kit must be validated by you. Pixel Kit
        does not guarantee model accuracy, safety, compliance or suitability for any use case.
      </p>
    </div>
  );
}

function ComingSoonBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="relative rounded-xl border border-[var(--border)] p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-[var(--muted)]">{title}</div>
        <span className="rounded-full border border-amber-300/30 bg-amber-300/[0.06] text-amber-200 px-2 py-0.5 text-[10px] uppercase tracking-wider">Coming soon</span>
      </div>
      <div aria-hidden className="pointer-events-none select-none [filter:blur(8px)] opacity-50">
        {children}
      </div>
      <div className="absolute inset-0 grid place-items-center">
        <div className="rounded-2xl border border-foreground/10 bg-black/40 backdrop-blur-md px-5 py-3 text-center">
          <div className="text-sm text-foreground/85">Not yet available</div>
          <div className="text-[11px] text-foreground/45 mt-0.5">Coming in a future release.</div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-[var(--muted)] mb-2">{label}</div>
      {children}
    </div>
  );
}

function NumberField({
  label, value, onChange, min, max, step, disabled, hint,
}: {
  label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; disabled?: boolean;
  hint?: string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <div className="text-xs text-[var(--muted)] mb-2">{label}</div>
        {hint && <div className="text-[10px] text-[var(--muted)] uppercase tracking-wider mb-2">{hint}</div>}
      </div>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!Number.isNaN(v)) onChange(v);
        }}
        className="w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm font-mono focus:outline-none focus:border-[var(--foreground)]/40 disabled:opacity-50"
      />
    </label>
  );
}

function ToggleRow({
  value, onChange, disabled, offLabel, onLabel,
}: {
  value: boolean; onChange: (v: boolean) => void; disabled?: boolean;
  offLabel: string; onLabel: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      disabled={disabled}
      onClick={() => onChange(!value)}
      className={[
        "inline-flex items-center gap-3 self-start rounded-full border px-3 py-1.5 text-sm transition-colors",
        value
          ? "border-amber-300/40 bg-amber-300/[0.08] text-amber-100 hover:bg-amber-300/[0.12]"
          : "border-foreground/15 bg-foreground/5 text-foreground/80 hover:bg-foreground/10 hover:text-foreground",
        disabled ? "opacity-60 cursor-not-allowed" : "",
      ].join(" ")}
    >
      <span
        aria-hidden
        className={[
          "h-4 w-7 rounded-full p-0.5 transition-colors flex",
          value ? "bg-amber-300/70 justify-end" : "bg-foreground/15",
        ].join(" ")}
      >
        <span className="h-3 w-3 rounded-full bg-[#141416]" />
      </span>
      {value ? onLabel : offLabel}
    </button>
  );
}
