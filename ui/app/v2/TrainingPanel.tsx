"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  cancelTrainingJob,
  createTrainingJob,
  estimateTrainingTokens,
  fmtDuration,
  getCachedTrainingJobs,
  getCachedTrainingModels,
  getTrainingArtifactUrl,
  getTrainingLogs,
  getTrainingModels,
  listTrainingJobs,
  mlJobIsActive,
  type MLJob,
  type MLJobStatus,
  type TrainingModel,
  type TrainingModelField,
} from "../../lib/mlJobs";

type ConfigValue = string | number | boolean;

const STATUS_LABEL: Record<MLJobStatus, string> = {
  queued: "Queued",
  preparing: "Preparing",
  running: "Training",
  cancelling: "Stopping",
  cancelled: "Cancelled",
  failed: "Failed",
  completed: "Completed",
};

function StatusPill({ status }: { status: MLJobStatus }) {
  const tone =
    status === "completed" ? "bg-emerald-500/10 text-emerald-500 ring-emerald-500/20"
    : status === "failed" ? "bg-red-500/10 text-red-500 ring-red-500/20"
    : status === "cancelled" ? "bg-foreground/[0.06] text-foreground/45 ring-foreground/10"
    : status === "running" ? "bg-amber-500/10 text-amber-500 ring-amber-500/20"
    : "bg-foreground/[0.06] text-foreground/55 ring-foreground/10";
  const dot =
    status === "completed" ? "bg-emerald-500"
    : status === "failed" ? "bg-red-500"
    : status === "cancelled" ? "bg-foreground/40"
    : status === "running" ? "bg-amber-500 pk-pulse"
    : "bg-foreground/50";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 h-6 text-[11px] font-medium tracking-wide ring-1 ring-inset ${tone}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {STATUS_LABEL[status]}
    </span>
  );
}

// Small inline loading wheel — shown in the model/config/estimate areas
// while the (static, cached) model registry loads on a first visit.
function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={`animate-spin ${className}`} fill="none" aria-label="Loading" role="status">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.6" />
      <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

// Generic control for one model hyperparameter, driven by the field
// schema the backend returns — so new models/params need no FE changes.
function Field({
  field, value, onChange,
}: {
  field: TrainingModelField;
  value: ConfigValue;
  onChange: (v: ConfigValue) => void;
}) {
  const base =
    "rounded-lg bg-foreground/[0.04] border border-foreground/15 focus:border-foreground/35 outline-none px-3 py-1.5 text-sm text-[var(--foreground)]";
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-foreground/60">{field.label}</span>
      {field.kind === "bool" ? (
        <span className="inline-flex items-center gap-2">
          <button
            type="button"
            role="switch"
            aria-checked={!!value}
            aria-label={field.label}
            onClick={() => onChange(!value)}
            // Toggle slider. White knob reads on both the amber (on) and
            // neutral (off) tracks in light AND dark mode; the track uses
            // theme-aware foreground/20 so it's visible on a light page.
            className={[
              "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
              value ? "bg-amber-500" : "bg-foreground/20",
            ].join(" ")}
          >
            <span
              className={[
                "inline-block h-5 w-5 rounded-full bg-white shadow-sm transition-transform",
                value ? "translate-x-[22px]" : "translate-x-[2px]",
              ].join(" ")}
            />
          </button>
          <span className="text-xs text-foreground/55">{value ? "On" : "Off"}</span>
        </span>
      ) : field.kind === "enum" ? (
        <select className={base} value={String(value)} onChange={(e) => onChange(numericLike(field) ? Number(e.target.value) : e.target.value)}>
          {(field.choices ?? []).map((c) => (
            <option key={String(c)} value={String(c)}>{String(c)}</option>
          ))}
        </select>
      ) : field.kind === "int" || field.kind === "float" ? (
        <input
          type="number"
          className={base}
          value={String(value)}
          min={field.min ?? undefined}
          max={field.max ?? undefined}
          step={field.kind === "float" ? "any" : 1}
          onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
        />
      ) : (
        <input type="text" className={base} value={String(value)} onChange={(e) => onChange(e.target.value)} />
      )}
      {field.help ? <span className="text-[11px] text-foreground/40">{field.help}</span> : null}
    </label>
  );
}

function numericLike(field: TrainingModelField): boolean {
  return (field.choices ?? []).every((c) => typeof c === "number");
}

// Compact stacked bar: original images vs augmented copies that will go
// into training. Amber = augmented (matches the augment accent); neutral
// = originals. Light/dark friendly via foreground tokens.
function DatasetBar({ nImages, nAugmentations }: { nImages: number; nAugmentations: number }) {
  const total = nImages + nAugmentations;
  const denom = Math.max(1, total);
  return (
    <div className="rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02] p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-foreground/40">Training data</span>
        <span className="text-sm tabular-nums text-foreground/70">{total} total</span>
      </div>
      <div className="mt-2 flex h-3 w-full overflow-hidden rounded-full bg-foreground/10">
        <div className="h-full bg-foreground/45" style={{ width: `${(nImages / denom) * 100}%` }} />
        <div className="h-full bg-amber-500" style={{ width: `${(nAugmentations / denom) * 100}%` }} />
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-foreground/55">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-foreground/45" />
          {nImages} image{nImages === 1 ? "" : "s"}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-amber-500" />
          {nAugmentations} augmented
        </span>
      </div>
    </div>
  );
}

export function TrainingPanel({
  projectId,
  nImages,
  nAugmentations,
  onEditAugmentations,
  inputSize,
  onInputSizeChange,
}: {
  projectId: string;
  projectName: string;
  nImages: number;
  nAugmentations: number;
  onEditAugmentations: () => void;
  // The project's target input shape ("NxN"), shown at the top of the project.
  // The training "Image size" field is LINKED to it: seeded from this value and
  // writing back through onInputSizeChange so changing it here changes it
  // project-wide (box-size warnings, export, …).
  inputSize: string;
  onInputSizeChange: (next: string) => void;
}) {
  // Seed from the synchronous cache so the dropdown + config fields paint on
  // the FIRST frame when we've loaded the (static) registry before. A wheel
  // shows only on the genuinely-first visit, while the network load runs.
  const cached = getCachedTrainingModels();
  const cachedFirst = cached ? (cached.default || cached.models[0]?.id || "") : "";
  const [models, setModels] = useState<TrainingModel[]>(cached?.models ?? []);
  const [modelId, setModelId] = useState<string>(cachedFirst);
  const [config, setConfig] = useState<Record<string, ConfigValue>>(() => {
    const spec = cached?.models.find((m) => m.id === cachedFirst);
    return spec ? seedConfig(spec) : {};
  });
  // Seed from the cache (warmed while the project loaded) so the already-trained
  // models paint on the first frame; refreshJobs() then revalidates.
  const [jobs, setJobs] = useState<MLJob[]>(() => getCachedTrainingJobs(projectId) ?? []);
  const [logs, setLogs] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [augSummary, setAugSummary] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(!cached);

  const model = useMemo(() => models.find((m) => m.id === modelId) ?? null, [models, modelId]);
  const modelLabelById = useMemo(
    () => Object.fromEntries(models.map((m) => [m.id, m.label])) as Record<string, string>,
    [models],
  );

  // Image size is locked to the project's target input shape ("NxN" → N). Keep
  // config.imgsz pinned to it whenever the project size OR the model changes
  // (selecting a model reseeds config.imgsz to the model default — re-pin it).
  const linkedImgsz = useMemo(() => {
    const n = parseInt(String(inputSize), 10);
    return Number.isFinite(n) ? n : null;
  }, [inputSize]);
  useEffect(() => {
    if (linkedImgsz == null || !model?.fields.some((f) => f.key === "imgsz")) return;
    setConfig((c) => (c.imgsz === linkedImgsz ? c : { ...c, imgsz: linkedImgsz }));
  }, [linkedImgsz, model]);

  // Revalidate the registry in the background. Keeps the current selection if
  // it still exists; only picks (and seeds) a default when nothing is chosen
  // yet — so a background refresh never clobbers the user's in-progress edits.
  useEffect(() => {
    let cancelled = false;
    getTrainingModels()
      .then((data) => {
        if (cancelled) return;
        setModels(data.models);
        setModelId((prev) => (prev && data.models.some((m) => m.id === prev))
          ? prev
          : (data.default || data.models[0]?.id || ""));
      })
      // Only surface a load error when we have nothing cached to show; a
      // failed background revalidation shouldn't blow away a working UI.
      .catch((e) => { if (!cancelled && !cached) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (!cancelled) setModelsLoading(false); });
    return () => { cancelled = true; };
    // Runs once on mount; `cached` is the snapshot captured at first render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Seed config defaults once a model is selected but config is still empty
  // (covers the first-ever load, where the cache was cold). Guarded so it
  // never overwrites config that's already been seeded or edited; switching
  // models in the dropdown seeds explicitly in its onChange handler.
  useEffect(() => {
    if (!modelId) return;
    setConfig((prev) => {
      if (Object.keys(prev).length > 0) return prev;
      const spec = models.find((m) => m.id === modelId);
      return spec ? seedConfig(spec) : prev;
    });
  }, [modelId, models]);

  // Best-effort augmentation-config reference (controlled on the
  // Augmentations tab; training does not duplicate the editor).
  useEffect(() => {
    let cancelled = false;
    import("../../lib/apiFetch").then(({ apiFetch }) =>
      apiFetch(`/api/v2/projects/${projectId}/augment/config`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((cfg) => {
          if (cancelled) return;
          setAugSummary(summariseAugConfig(cfg));
        })
        .catch(() => { if (!cancelled) setAugSummary(null); }),
    );
    return () => { cancelled = true; };
  }, [projectId]);

  const refreshJobs = useCallback(async () => {
    try {
      const data = await listTrainingJobs(projectId);
      setJobs(data.jobs);
    } catch {
      /* transient; next tick retries */
    }
  }, [projectId]);

  useEffect(() => { void refreshJobs(); }, [refreshJobs]);

  const activeJob = useMemo(
    () => jobs.find((j) => mlJobIsActive(j.status)) ?? jobs[0] ?? null,
    [jobs],
  );

  // Poll while there's an active job (status + logs); stop when idle.
  useEffect(() => {
    if (!activeJob || !mlJobIsActive(activeJob.status)) return;
    const id = window.setInterval(() => {
      void refreshJobs();
      getTrainingLogs(activeJob.id, 200).then((d) => setLogs(d.logs)).catch(() => {});
    }, 4000);
    return () => window.clearInterval(id);
  }, [activeJob, refreshJobs]);

  // Recompute on ANY config change (input size, augmentations toggle,
  // epochs, …) — `config` is replaced on every edit so this memo re-runs.
  const estimate = useMemo(
    () => estimateTrainingTokens(model, config, nImages, nAugmentations),
    [model, config, nImages, nAugmentations],
  );

  const start = async () => {
    if (!model || busy) return;
    setBusy(true);
    setError(null);
    try {
      // Guarantee the submitted size matches the linked project input shape,
      // even if the sync effect hasn't flushed config.imgsz yet.
      const cfg = linkedImgsz != null && model.fields.some((f) => f.key === "imgsz")
        ? { ...config, imgsz: linkedImgsz }
        : config;
      await createTrainingJob(projectId, { model_id: model.id, config: cfg });
      await refreshJobs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (jobId: string) => {
    try {
      await cancelTrainingJob(jobId);
      await refreshJobs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const download = async (jobId: string) => {
    try {
      const { url } = await getTrainingArtifactUrl(jobId);
      window.open(url, "_blank", "noopener");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const canStart = !!model && nImages > 0 && !busy;

  // Job card title = the user's chosen OUTPUT model name (not the architecture).
  const jobName = (j: MLJob) =>
    ((j.config?.output_name as string | undefined)?.trim())
    || (j.model_id ? (modelLabelById[j.model_id] ?? j.model_id) : "Model");

  // One radius + one gap shared by every card and both axes (per design).
  const CARD = "rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02]";

  return (
    <div className="px-6 lg:px-10 py-10">
      <style dangerouslySetInnerHTML={{ __html:
        "@keyframes pkUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}"
        + "@keyframes pkPulse{0%,100%{opacity:1}50%{opacity:.3}}"
        + "@keyframes pkBlink{0%,49%{opacity:1}50%,100%{opacity:0}}"
        + ".pk-up{animation:pkUp .55s cubic-bezier(.16,1,.3,1) both}"
        + ".pk-pulse{animation:pkPulse 1.5s ease-in-out infinite}"
        + ".pk-blink{animation:pkBlink 1.1s step-end infinite}"
        + ".pk-card{transition:border-color .3s ease,box-shadow .3s ease,transform .3s ease}"
      }} />

      <header className="pk-up mb-5">
        <h2 className="text-[28px] font-semibold tracking-tight">Train a model</h2>
        <p className="mt-1.5 text-[15px] text-foreground/50">
          Fine-tune a detector on this project&apos;s labelled images.
        </p>
      </header>

      {/* Equal gap on both axes; columns stretch so the jobs panel's top
          aligns with Training data and its bottom with Estimated cost. */}
      <div className="grid items-stretch gap-5 lg:grid-cols-2">
        {/* ── configure (narrower column) ── */}
        <div className="flex flex-col gap-5">
          <div className="pk-up" style={{ animationDelay: "40ms" }}>
            <DatasetBar nImages={nImages} nAugmentations={nAugmentations} />
          </div>

          {/* Model + parameters — grouped "settings" card */}
          <div className={`pk-up pk-card flex flex-col gap-5 p-5 ${CARD}`} style={{ animationDelay: "80ms" }}>
            <label className="flex flex-col gap-2">
              <span className="text-[13px] font-medium text-foreground/70">Model</span>
              {modelsLoading && models.length === 0 ? (
                <span className="inline-flex items-center gap-2 rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 text-sm text-foreground/45">
                  <Spinner className="h-4 w-4" /> Loading models…
                </span>
              ) : (
                <div className="relative">
                  <select
                    className="w-full cursor-pointer appearance-none rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 pr-9 text-sm font-medium text-[var(--foreground)] outline-none transition-colors hover:border-foreground/25 focus:border-foreground/40"
                    value={modelId}
                    onChange={(e) => {
                      const id = e.target.value;
                      setModelId(id);
                      const spec = models.find((m) => m.id === id);
                      if (spec) setConfig(seedConfig(spec));
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                  <svg className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" viewBox="0 0 20 20" fill="none">
                    <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
              {model?.recommended && (
                <span className="inline-flex w-fit items-center gap-1.5 rounded-full bg-amber-500/[0.12] px-2.5 h-6 text-[11px] font-medium text-amber-500 ring-1 ring-inset ring-amber-500/20">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                  {model.recommended_note || "Recommended for STM32N6"}
                </span>
              )}
              {model?.description ? <span className="text-[12px] leading-relaxed text-foreground/45">{model.description}</span> : null}
            </label>

            {model ? (
              <>
                <div className="h-px bg-foreground/[0.07]" />
                <div className="grid grid-cols-2 gap-x-5 gap-y-4">
                  {model.fields.map((f) => (
                    <Field
                      key={f.key}
                      field={f}
                      value={f.key === "imgsz" && linkedImgsz != null ? linkedImgsz : (config[f.key] ?? (f.default as ConfigValue))}
                      onChange={(v) => {
                        setConfig((c) => ({ ...c, [f.key]: v }));
                        // Image size is linked to the project's target input shape.
                        if (f.key === "imgsz") {
                          const n = Number(v);
                          if (Number.isFinite(n)) onInputSizeChange(`${n}x${n}`);
                        }
                      }}
                    />
                  ))}
                </div>
              </>
            ) : modelsLoading ? (
              <div className="flex items-center gap-2 text-sm text-foreground/45">
                <Spinner className="h-4 w-4" /> Loading configuration…
              </div>
            ) : null}
          </div>

          {/* Augmentation reference — edited on the Augmentations tab. */}
          <div className={`pk-up pk-card flex items-center justify-between gap-3 p-4 ${CARD}`} style={{ animationDelay: "120ms" }}>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-foreground/40">Augmentations</div>
              <div className="mt-0.5 text-sm text-foreground/70">{augSummary ?? "Configured on the Augmentations tab"}</div>
            </div>
            <button
              type="button"
              onClick={onEditAugmentations}
              className="h-8 shrink-0 rounded-full bg-foreground/[0.06] px-3.5 text-xs font-medium text-foreground/70 transition-all hover:bg-foreground/[0.12] active:scale-95"
            >
              Edit
            </button>
          </div>

          {/* Estimate + billing note */}
          <div className="pk-up rounded-2xl border border-amber-500/20 bg-amber-500/[0.05] p-4" style={{ animationDelay: "160ms" }}>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-foreground/65">Estimated cost</span>
              {modelsLoading && !model ? (
                <Spinner className="h-4 w-4 text-amber-500" />
              ) : (
                <span className="text-sm font-semibold text-amber-500">
                  ~{estimate.blocks} {estimate.blocks === 1 ? "token" : "tokens"}
                  <span className="font-normal text-foreground/40"> · {fmtDuration(estimate.seconds)}</span>
                </span>
              )}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-foreground/45">
              Estimate only — based on {nImages} image{nImages === 1 ? "" : "s"}. Tokens are charged for actual active
              training time (1 token per 15 minutes). Queued time is not charged.
            </p>
          </div>
        </div>

        {/* ── jobs (wider column). Absolutely-filled so the grid row height is
            driven by the config column → top/bottom align; list scrolls. ── */}
        <div className="pk-up relative min-h-[440px] lg:min-h-0" style={{ animationDelay: "120ms" }}>
          <div className="absolute inset-0 flex flex-col rounded-2xl border border-foreground/[0.07] bg-foreground/[0.015] p-5">
            <h3 className="px-1 text-[13px] font-semibold tracking-tight text-foreground/70">Training jobs</h3>
            <div className="relative mt-4 min-h-0 flex-1">
              <div className="flex h-full flex-col gap-3 overflow-y-auto overflow-x-hidden pb-12 pr-1">
                {jobs.length === 0 && (
                  <div className="rounded-2xl border border-dashed border-foreground/10 px-4 py-10 text-center text-sm text-foreground/40">
                    No training jobs yet.
                  </div>
                )}
                {activeJob && (
                  <ActiveJobCard job={activeJob} modelName={jobName(activeJob)} logs={logs} onCancel={cancel} onDownload={download} />
                )}
                {jobs.filter((j) => j !== activeJob).slice(0, 50).map((j, i) => (
                  <div key={j.id} className="pk-up" style={{ animationDelay: `${160 + i * 40}ms` }}>
                    <HistoryRow job={j} modelName={jobName(j)} onDownload={download} />
                  </div>
                ))}
              </div>
              {/* Graduated blur at the bottom — signals more jobs to scroll to. */}
              <div
                className="pointer-events-none absolute inset-x-0 bottom-0 h-14 rounded-b-2xl bg-gradient-to-t from-background to-transparent"
                style={{ backdropFilter: "blur(2px)", WebkitMaskImage: "linear-gradient(to top, #000, transparent)", maskImage: "linear-gradient(to top, #000, transparent)" }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Actions sit BELOW the grid so the jobs panel bottom lines up with the Estimated cost card. */}
      <div className="mt-5 flex flex-col gap-3">
        {error && <div className="pk-up rounded-2xl bg-red-500/10 px-3.5 py-2.5 text-sm text-red-500 ring-1 ring-inset ring-red-500/20">{error}</div>}
        <button
          type="button"
          disabled={!canStart}
          onClick={start}
          className="pk-up h-11 w-full rounded-full bg-foreground px-6 text-sm font-semibold text-background transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-40 sm:w-fit"
        >
          {busy ? "Starting…" : "Start training"}
        </button>
        {nImages <= 0 && <div className="text-xs text-foreground/45">Add labelled images before training.</div>}
      </div>
    </div>
  );
}

function ActiveJobCard({
  job, modelName, logs, onCancel, onDownload,
}: {
  job: MLJob;
  modelName: string;
  logs: string[];
  onCancel: (id: string) => void;
  onDownload: (id: string) => void;
}) {
  const progress = job.progress as {
    epoch?: number; total?: number; train_loss?: number; val_loss?: number; val_map?: number; history?: LossPoint[];
  };
  const history = Array.isArray(progress?.history) ? progress.history : [];
  const pct = progress?.epoch != null && progress?.total
    ? Math.min(100, Math.round((progress.epoch / progress.total) * 100))
    : null;
  // Live, per-second runtime for a running job (the backend only reports it
  // every ~30s). Ticks off started_at; falls back to the server value when
  // the job isn't actively running.
  const running = mlJobIsActive(job.status);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running]);
  const runtimeSec = running && job.started_at
    ? Math.max(job.active_runtime_seconds, (now - Date.parse(job.started_at)) / 1000)
    : job.active_runtime_seconds;
  return (
    <div className="pk-up pk-card flex flex-col gap-3.5 rounded-2xl border border-foreground/10 bg-background/40 p-4 shadow-sm shadow-black/[0.03]">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold tracking-tight">{modelName}</div>
          <div className="mt-0.5 font-mono text-[10px] text-foreground/35">{job.id.slice(0, 8)}</div>
        </div>
        <StatusPill status={job.status} />
      </div>

      {pct != null && (
        <div>
          <div className="mb-1 flex justify-between text-[11px] text-foreground/45">
            <span>Epoch {progress.epoch}{progress.total ? ` / ${progress.total}` : ""}</span>
            <span className="tabular-nums">{pct}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-foreground/10">
            <div className="h-full rounded-full bg-amber-500 transition-all duration-500 ease-out" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3 text-sm">
        {job.status === "queued" && (
          <Stat label="Queue" value={job.queue_position != null ? `#${job.queue_position + 1}` : "—"} />
        )}
        <Stat label="Runtime" value={fmtDuration(runtimeSec)} />
        <Stat label="Accuracy" value={fmtAccuracy(progress?.val_map)} />
        <Stat label="Tokens" value={String(job.charged_credits)} />
        {progress?.train_loss != null && <Stat label="Train loss" value={Number(progress.train_loss).toFixed(3)} />}
      </div>

      {/* Live training graph: train loss, val loss + accuracy over epochs. */}
      {history.length >= 2 && (
        <div className="rounded-xl border border-foreground/[0.08] bg-foreground/[0.02] p-3">
          <GraphLegend />
          <TrainingGraph history={history} />
        </div>
      )}

      {job.error && (
        <div className="rounded-lg bg-red-500/10 px-3 py-2 text-[13px] leading-relaxed text-red-500 ring-1 ring-inset ring-red-500/20">
          {job.error}
        </div>
      )}

      {logs.length > 0 && <TrainingLog logs={logs} running={mlJobIsActive(job.status)} />}

      <div className="flex gap-2">
        {mlJobIsActive(job.status) && (
          <button type="button" onClick={() => onCancel(job.id)} title="Stop training and keep the current checkpoint"
            className="h-8 rounded-full bg-foreground/[0.06] px-3.5 text-xs font-medium text-foreground/70 transition-all hover:bg-foreground/[0.12] active:scale-95">
            Stop &amp; save
          </button>
        )}
        {job.status === "completed" && (
          <button type="button" onClick={() => onDownload(job.id)} className="h-8 rounded-full bg-foreground px-3.5 text-xs font-semibold text-background transition-all hover:opacity-90 active:scale-95">
            Download model
          </button>
        )}
      </div>
    </div>
  );
}

type LossPoint = { epoch?: number; train_loss?: number | null; val_loss?: number | null; val_map?: number | null };

// val mAP@0.5 is stored 0..1 → show as a 0–100% accuracy.
function fmtAccuracy(v?: number | null): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

function HistoryRow({ job, modelName, onDownload }: { job: MLJob; modelName: string; onDownload: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const progress = job.progress as {
    epoch?: number; total?: number; train_loss?: number; val_loss?: number; val_map?: number; history?: LossPoint[];
  };
  const history = Array.isArray(progress?.history) ? progress.history : [];
  const cfg = (job.config ?? {}) as Record<string, unknown>;
  const imgsz = Number(cfg.imgsz);
  const lastTrain = history.length ? history[history.length - 1].train_loss : progress?.train_loss;
  // Best (highest) val mAP across the run — the headline accuracy.
  const bestMap = history.reduce<number | null>(
    (b, h) => (typeof h.val_map === "number" ? (b == null ? h.val_map : Math.max(b, h.val_map)) : b),
    typeof progress?.val_map === "number" ? progress.val_map : null,
  );

  return (
    <div className="pk-card overflow-hidden rounded-2xl border border-foreground/[0.07] bg-foreground/[0.02]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left transition-colors hover:bg-foreground/[0.03]"
      >
        <div className="flex min-w-0 items-center gap-2.5">
          <StatusPill status={job.status} />
          <div className="min-w-0">
            <div className="truncate text-[13px] font-medium text-foreground/80">{modelName}</div>
            <div className="truncate text-[11px] text-foreground/40">
              {fmtDuration(job.active_runtime_seconds)} · {job.charged_credits} tok
            </div>
          </div>
        </div>
        <svg
          className={`h-4 w-4 shrink-0 text-foreground/35 transition-transform duration-300 ${open ? "rotate-180" : ""}`}
          viewBox="0 0 20 20" fill="none" aria-hidden
        >
          <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="border-t border-foreground/[0.07] px-3.5 pb-3.5 pt-3">
          {/* Created first + on its own row so the date reads cleanly. */}
          <div className="mb-3 flex items-center justify-between rounded-lg bg-foreground/[0.03] px-3 py-2">
            <span className="text-[10px] uppercase tracking-wider text-foreground/40">Created</span>
            <span className="text-[13px] font-medium text-foreground/80">{fmtDateTime(job.created_at)}</span>
          </div>
          <div className="grid grid-cols-3 gap-x-3 gap-y-3">
            <Stat label="Accuracy" value={fmtAccuracy(bestMap)} />
            <Stat label="Training time" value={fmtDuration(job.active_runtime_seconds)} />
            <Stat label="Tokens" value={String(job.charged_credits)} />
            <Stat label="Epochs" value={progress?.epoch != null ? `${progress.epoch}${progress.total ? ` / ${progress.total}` : ""}` : (cfg.epochs != null ? String(cfg.epochs) : "—")} />
            <Stat label="Image size" value={Number.isFinite(imgsz) ? `${imgsz}×${imgsz}` : "—"} />
            <Stat label="Learning rate" value={cfg.learning_rate != null ? String(cfg.learning_rate) : "—"} />
            <Stat label="Final train loss" value={typeof lastTrain === "number" ? lastTrain.toFixed(3) : "—"} />
          </div>

          {history.length >= 2 && (
            <div className="mt-3.5">
              <GraphLegend />
              <TrainingGraph history={history} />
            </div>
          )}

          {job.error && (
            <div className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-[12px] leading-relaxed text-red-500 ring-1 ring-inset ring-red-500/20">
              {job.error}
            </div>
          )}

          {job.status === "completed" && (
            <button
              type="button"
              onClick={() => onDownload(job.id)}
              className="mt-3 h-8 rounded-full bg-foreground px-3.5 text-xs font-semibold text-background transition-all hover:opacity-90 active:scale-95"
            >
              Download model
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Training graph over epochs: train loss + val loss (shared scale — same
// units, so directly comparable) and accuracy (val mAP@0.5) on its own scale
// so its trend stays visible despite the very different magnitude. Each series
// fills the plot height for its range; epochs are evenly spaced along x.
function TrainingGraph({ history }: { history: LossPoint[] }) {
  const pts = history.filter((h) =>
    typeof h.train_loss === "number" || typeof h.val_loss === "number" || typeof h.val_map === "number");
  if (pts.length < 2) return null;
  const n = pts.length;
  const W = 280, H = 70, pad = 6;
  const x = (i: number) => pad + (i / (n - 1)) * (W - 2 * pad);
  const scaleY = (nums: number[]) => {
    const mn = Math.min(...nums), mx = Math.max(...nums), sp = (mx - mn) || 1;
    return (v: number) => pad + (1 - (v - mn) / sp) * (H - 2 * pad);
  };
  const path = (vals: (number | null)[], yfn: (v: number) => number) =>
    vals.map((v, i) => (v == null ? "" : `${i === 0 || vals[i - 1] == null ? "M" : "L"}${x(i).toFixed(1)},${yfn(v).toFixed(1)}`)).join(" ");

  const train = pts.map((p) => (typeof p.train_loss === "number" ? p.train_loss : null));
  const val = pts.map((p) => (typeof p.val_loss === "number" ? p.val_loss : null));
  const acc = pts.map((p) => (typeof p.val_map === "number" ? (p.val_map as number) * 100 : null));

  const lossNums = [...train, ...val].filter((v): v is number => v != null);
  const accNums = acc.filter((v): v is number => v != null);
  const yLoss = lossNums.length ? scaleY(lossNums) : null;   // train + val share this
  const yAcc = accNums.length ? scaleY(accNums) : null;      // accuracy own scale

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[200px] w-full sm:h-[260px] xl:h-[340px]" role="img" aria-label="Training metrics over epochs">
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="currentColor" className="text-foreground/10" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      {yLoss && <path d={path(train, yLoss)} fill="none" stroke="currentColor" className="text-amber-500" strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
      {yLoss && <path d={path(val, yLoss)} fill="none" stroke="currentColor" className="text-sky-500" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
      {yAcc && <path d={path(acc, yAcc)} fill="none" stroke="currentColor" className="text-emerald-500" strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

// Shared legend for the training graph.
function GraphLegend() {
  return (
    <div className="mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] uppercase tracking-wider text-foreground/40">
      <span className="inline-flex items-center gap-1"><span className="h-1.5 w-3 rounded-full bg-amber-500" /> train loss</span>
      <span className="inline-flex items-center gap-1"><span className="h-1.5 w-3 rounded-full bg-sky-500" /> val loss</span>
      <span className="inline-flex items-center gap-1"><span className="h-1.5 w-3 rounded-full bg-emerald-500" /> accuracy</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-foreground/40">{label}</div>
      <div className="mt-0.5 text-[15px] font-medium tabular-nums text-foreground/85">{value}</div>
    </div>
  );
}

// Clean, minimal training log that matches the rest of the app (theme tokens,
// light + dark friendly — no dark terminal chrome). A soft card with a small
// labelled header, an auto-scrolling monospace body that lightly accents
// epoch / status / error lines, and a blinking caret while the job runs.
function TrainingLog({ logs, running }: { logs: string[]; running: boolean }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);
  const shown = logs.slice(-200);
  return (
    <div className="overflow-hidden rounded-xl border border-foreground/[0.08] bg-foreground/[0.025]">
      <div className="flex items-center gap-2 border-b border-foreground/[0.06] px-3.5 py-2">
        <span className={`h-1.5 w-1.5 rounded-full ${running ? "bg-amber-500 pk-pulse" : "bg-foreground/30"}`} />
        <span className="text-[10px] font-medium uppercase tracking-wider text-foreground/45">Training log</span>
      </div>
      <div ref={bodyRef} className="max-h-44 overflow-y-auto overflow-x-hidden px-3.5 py-2.5 font-mono text-[11px] leading-[1.7]">
        {shown.map((ln, i) => {
          const { time, msg } = splitLogLine(ln);
          return (
            <div key={i} className="flex gap-2.5 whitespace-pre-wrap break-words">
              {time && <span className="shrink-0 select-none tabular-nums text-foreground/30">{time}</span>}
              <span className={logLineClass(msg)}>{msg}</span>
            </div>
          );
        })}
        {running && <span className="pk-blink text-foreground/40">▋</span>}
      </div>
    </div>
  );
}

// Backend log lines are prefixed with a UTC ISO timestamp
// ("2026-06-01T19:17:32+00:00  message"). Split it off and render just a clean
// local HH:MM:SS so the log reads naturally; leave un-timestamped lines as-is.
function splitLogLine(ln: string): { time: string; msg: string } {
  const m = ln.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)\s+([\s\S]*)$/);
  if (!m) return { time: "", msg: ln };
  const d = new Date(m[1]);
  if (Number.isNaN(d.getTime())) return { time: "", msg: m[2] };
  return {
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
    msg: m[2],
  };
}

function logLineClass(ln: string): string {
  const l = ln.toLowerCase();
  if (/(error|fail|traceback|oom|exception)/.test(l)) return "text-red-500";
  if (/(epoch|loss|train=|val=)/.test(l)) return "text-amber-600 dark:text-amber-400";
  if (/(start|complet|done|saved|upload|queued)/.test(l)) return "text-emerald-600 dark:text-emerald-400";
  return "text-foreground/55";
}

// ── helpers ──────────────────────────────────────────────────────────
// Readable, locale-aware date-time: "1 Jun 2026, 14:32".
function fmtDateTime(v?: string | null): string {
  if (!v) return "—";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function seedConfig(model: TrainingModel): Record<string, ConfigValue> {
  const out: Record<string, ConfigValue> = {};
  for (const f of model.fields) out[f.key] = f.default as ConfigValue;
  return out;
}

function summariseAugConfig(cfg: unknown): string {
  if (!cfg || typeof cfg !== "object") return "No augmentations configured";
  const obj = cfg as Record<string, unknown>;
  const transforms = obj.transforms;
  if (Array.isArray(transforms)) {
    const enabled = transforms.filter((t) => !t || typeof t !== "object" || (t as { enabled?: boolean }).enabled !== false).length;
    return enabled > 0 ? `${enabled} transform${enabled === 1 ? "" : "s"} configured` : "No augmentations configured";
  }
  if (obj.enabled === false) return "Augmentations off";
  return "Configured on the Augmentations tab";
}
