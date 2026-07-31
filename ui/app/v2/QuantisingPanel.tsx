"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  cancelQuantiseJob,
  createQuantiseJob,
  getCachedQuantiseOptions,
  getQuantiseArtifactUrl,
  getQuantiseLogs,
  getQuantiseOptions,
  getQuantiseSourceModels,
  listQuantiseJobs,
  mlJobIsActive,
  type MLJob,
  type MLJobStatus,
  type QuantiseOptions,
  type SourceModel,
} from "../../lib/mlJobs";

const STATUS_LABEL: Record<MLJobStatus, string> = {
  queued: "Queued",
  preparing: "Preparing",
  running: "Quantising",
  cancelling: "Cancelling",
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

function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={`animate-spin ${className}`} fill="none" aria-label="Loading" role="status">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.6" />
      <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

export function QuantisingPanel({ projectId }: { projectId: string; projectName: string }) {
  // Options are static per deploy — seed from the cache for an instant first
  // paint, then revalidate. The source-model list is per-project, so it shows
  // a wheel until it loads.
  const cachedOptions = getCachedQuantiseOptions();
  const [options, setOptions] = useState<QuantiseOptions | null>(cachedOptions);
  const [sources, setSources] = useState<SourceModel[]>([]);
  const [sourcesLoading, setSourcesLoading] = useState(true);
  const [sourceJobId, setSourceJobId] = useState<string>("");
  const [samples, setSamples] = useState<number>(cachedOptions?.calibration.default_samples ?? 100);
  const [jobs, setJobs] = useState<MLJob[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getQuantiseOptions()
      .then((o) => {
        if (cancelled) return;
        setOptions(o);
        // Only adopt the default sample count if the user hasn't been
        // offered a cached value to start from.
        if (!cachedOptions) setSamples(o.calibration.default_samples);
      })
      .catch((e) => !cancelled && !cachedOptions && setError(e instanceof Error ? e.message : String(e)));
    getQuantiseSourceModels(projectId)
      .then((d) => {
        if (cancelled) return;
        setSources(d.models);
        if (d.models[0]) setSourceJobId(d.models[0].source_job_id);
      })
      .catch(() => { /* none yet */ })
      .finally(() => { if (!cancelled) setSourcesLoading(false); });
    return () => { cancelled = true; };
    // cachedOptions is read once on mount; projectId drives the source list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const refreshJobs = useCallback(async () => {
    try {
      const d = await listQuantiseJobs(projectId);
      setJobs(d.jobs);
    } catch {
      /* transient */
    }
  }, [projectId]);

  useEffect(() => { void refreshJobs(); }, [refreshJobs]);

  const activeJob = useMemo(
    () => jobs.find((j) => mlJobIsActive(j.status)) ?? jobs[0] ?? null,
    [jobs],
  );

  useEffect(() => {
    if (!activeJob || !mlJobIsActive(activeJob.status)) return;
    const id = window.setInterval(() => {
      void refreshJobs();
      getQuantiseLogs(activeJob.id, 200).then((d) => setLogs(d.logs)).catch(() => {});
    }, 4000);
    return () => window.clearInterval(id);
  }, [activeJob, refreshJobs]);

  const hasSource = sources.length > 0;

  const start = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await createQuantiseJob(projectId, {
        source_job_id: sourceJobId || null,
        mode: options?.default_mode,
        calibration_samples: samples,
      });
      await refreshJobs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (jobId: string) => {
    try { await cancelQuantiseJob(jobId); await refreshJobs(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  const download = async (jobId: string, which: "int8_onnx" | "float_onnx") => {
    try {
      const { url } = await getQuantiseArtifactUrl(jobId, which);
      window.open(url, "_blank", "noopener");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Quantise job title = the source model's name (mirrors the train page's
  // output-name titles).
  const jobName = (j: MLJob) => {
    const sid = (j.config?.source_job_id as string | undefined) || "";
    const src = sources.find((s) => s.source_job_id === sid);
    return src?.name || src?.model_id || "INT8 model";
  };

  // One radius + one gap shared by every card and both axes (matches Train).
  const CARD = "rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02]";

  return (
    <div className="px-6 lg:px-10 py-10">
      <style dangerouslySetInnerHTML={{ __html:
        "@keyframes pkUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}"
        + "@keyframes pkPulse{0%,100%{opacity:1}50%{opacity:.3}}"
        + ".pk-up{animation:pkUp .55s cubic-bezier(.16,1,.3,1) both}"
        + ".pk-pulse{animation:pkPulse 1.5s ease-in-out infinite}"
        + ".pk-card{transition:border-color .3s ease,box-shadow .3s ease,transform .3s ease}"
      }} />

      <header className="pk-up mb-5">
        <h2 className="text-[28px] font-semibold tracking-tight">Quantise a model</h2>
        <p className="mt-1.5 text-[15px] text-foreground/50">
          Export a trained model to INT8 ONNX, calibrated on your images — smaller and faster for edge inference.
        </p>
      </header>

      <div className="grid items-stretch gap-5 lg:grid-cols-2">
        {/* ── configure ── */}
        <div className="flex flex-col gap-5">
          {/* Source + quantisation + calibration — grouped "settings" card */}
          <div className={`pk-up pk-card flex flex-col gap-5 p-5 ${CARD}`} style={{ animationDelay: "40ms" }}>
            <label className="flex flex-col gap-2">
              <span className="text-[13px] font-medium text-foreground/70">Source model</span>
              {sourcesLoading && !hasSource ? (
                <span className="inline-flex items-center gap-2 rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 text-sm text-foreground/45">
                  <Spinner className="h-4 w-4" /> Loading trained models…
                </span>
              ) : (
                <div className="relative">
                  <select
                    className="w-full cursor-pointer appearance-none rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 pr-9 text-sm font-medium text-[var(--foreground)] outline-none transition-colors hover:border-foreground/25 focus:border-foreground/40 disabled:opacity-50"
                    value={sourceJobId}
                    disabled={!hasSource}
                    onChange={(e) => setSourceJobId(e.target.value)}
                  >
                    {!hasSource && <option value="">No trained models yet</option>}
                    {sources.map((s) => (
                      <option key={s.source_job_id} value={s.source_job_id}>
                        {(s.name || s.model_id || "model")}{s.trained_at ? ` · ${new Date(s.trained_at).toLocaleDateString()}` : ""}
                      </option>
                    ))}
                  </select>
                  <svg className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" viewBox="0 0 20 20" fill="none">
                    <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </label>

            {hasSource && <div className="h-px bg-foreground/[0.07]" />}

            {hasSource && (
              <div className="grid grid-cols-2 gap-x-5 gap-y-4">
                <label className="flex flex-col gap-2">
                  <span className="text-[13px] font-medium text-foreground/70">Quantisation</span>
                  <div className="relative">
                    <select
                      className="w-full cursor-not-allowed appearance-none rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 pr-9 text-sm text-[var(--foreground)] outline-none"
                      value={options?.default_mode ?? "int8_static"}
                      disabled
                    >
                      {(options?.modes ?? [{ id: "int8_static", label: "INT8 (static PTQ)", help: "" }]).map((m) => (
                        <option key={m.id} value={m.id}>{m.label}</option>
                      ))}
                    </select>
                    <svg className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/40" viewBox="0 0 20 20" fill="none">
                      <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                </label>

                <label className="flex flex-col gap-2">
                  <span className="text-[13px] font-medium text-foreground/70">Calibration images</span>
                  <input
                    type="number"
                    className="w-full rounded-xl border border-foreground/15 bg-foreground/[0.04] px-3.5 py-2.5 text-sm text-[var(--foreground)] outline-none transition-colors hover:border-foreground/25 focus:border-foreground/40"
                    value={samples}
                    min={options?.calibration.min_samples ?? 1}
                    max={options?.calibration.max_samples ?? 1000}
                    onChange={(e) => setSamples(Math.max(1, Number(e.target.value) || 1))}
                  />
                </label>
              </div>
            )}
            <span className="text-[12px] leading-relaxed text-foreground/45">
              {hasSource
                ? (options?.calibration.help ?? "Representative images used to calibrate INT8 ranges.")
                : "Train a model first, then quantise it here."}
            </span>
          </div>

          {/* Cost note (mirrors the Estimated cost card on the Train page) */}
          <div className="pk-up rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.05] p-4" style={{ animationDelay: "80ms" }}>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-foreground/65">Cost</span>
              <span className="text-sm font-semibold text-emerald-500">Free</span>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-foreground/45">
              Quantising runs on the rig and isn&apos;t charged — it exports your trained model to INT8 ONNX (QDQ),
              calibrated on a sample of your project images.
            </p>
          </div>
        </div>

        {/* ── jobs (separated, equal-height, scroll + graduated blur) ── */}
        <div className="pk-up relative min-h-[440px] lg:min-h-0" style={{ animationDelay: "120ms" }}>
          <div className="absolute inset-0 flex flex-col rounded-2xl border border-foreground/[0.07] bg-foreground/[0.015] p-5">
            <h3 className="px-1 text-[13px] font-semibold tracking-tight text-foreground/70">Quantising jobs</h3>
            <div className="relative mt-4 min-h-0 flex-1">
              <div className="flex h-full flex-col gap-3 overflow-y-auto overflow-x-hidden pb-12 pr-1">
                {jobs.length === 0 && (
                  <div className="rounded-2xl border border-dashed border-foreground/10 px-4 py-10 text-center text-sm text-foreground/40">
                    No quantising jobs yet.
                  </div>
                )}
                {activeJob && (
                  <div className="pk-up pk-card flex flex-col gap-3.5 rounded-2xl border border-foreground/10 bg-background/40 p-4 shadow-sm shadow-black/[0.03]">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold tracking-tight">{jobName(activeJob)}</div>
                        <div className="mt-0.5 font-mono text-[10px] text-foreground/35">{activeJob.id.slice(0, 8)}</div>
                      </div>
                      <StatusPill status={activeJob.status} />
                    </div>
                    {activeJob.status === "queued" && activeJob.queue_position != null && (
                      <div className="text-[13px] text-foreground/55">Queue position #{activeJob.queue_position + 1}</div>
                    )}
                    {activeJob.error && (
                      <div className="rounded-lg bg-red-500/10 px-3 py-2 text-[13px] leading-relaxed text-red-500 ring-1 ring-inset ring-red-500/20">{activeJob.error}</div>
                    )}
                    {logs.length > 0 && (
                      <pre className="max-h-40 overflow-y-auto overflow-x-hidden rounded-lg bg-foreground/[0.04] p-2.5 text-[11px] leading-relaxed text-foreground/55 whitespace-pre-wrap break-words">
                        {logs.slice(-40).join("\n")}
                      </pre>
                    )}
                    <div className="flex flex-wrap gap-2">
                      {mlJobIsActive(activeJob.status) && (
                        <button type="button" onClick={() => cancel(activeJob.id)} className="h-8 rounded-full bg-foreground/[0.06] px-3.5 text-xs font-medium text-foreground/70 transition-all hover:bg-foreground/[0.12] active:scale-95">
                          Cancel
                        </button>
                      )}
                      {activeJob.status === "completed" && (
                        <>
                          <button type="button" onClick={() => download(activeJob.id, "int8_onnx")} className="h-8 rounded-full bg-foreground px-3.5 text-xs font-semibold text-background transition-all hover:opacity-90 active:scale-95">
                            Download INT8 ONNX
                          </button>
                          <button type="button" onClick={() => download(activeJob.id, "float_onnx")} className="h-8 rounded-full bg-foreground/[0.06] px-3.5 text-xs font-medium text-foreground/70 transition-all hover:bg-foreground/[0.12] active:scale-95">
                            Float ONNX
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
                {jobs.filter((j) => j !== activeJob).slice(0, 50).map((j, i) => (
                  <div key={j.id} className="pk-up" style={{ animationDelay: `${160 + i * 40}ms` }}>
                    <div className="pk-card flex items-center justify-between gap-3 rounded-2xl border border-foreground/[0.07] bg-foreground/[0.02] px-3.5 py-2.5">
                      <div className="flex min-w-0 items-center gap-2.5">
                        <StatusPill status={j.status} />
                        <div className="truncate text-[13px] font-medium text-foreground/80">{jobName(j)}</div>
                      </div>
                      {j.status === "completed" && (
                        <button type="button" onClick={() => download(j.id, "int8_onnx")} className="shrink-0 text-xs font-medium text-amber-500 hover:underline">
                          INT8 ONNX
                        </button>
                      )}
                    </div>
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

      {/* Actions below the grid so the jobs panel aligns with the Cost card. */}
      <div className="mt-5 flex flex-col gap-3">
        {error && <div className="pk-up rounded-2xl bg-red-500/10 px-3.5 py-2.5 text-sm text-red-500 ring-1 ring-inset ring-red-500/20">{error}</div>}
        <button
          type="button"
          disabled={!hasSource || busy}
          onClick={start}
          className="pk-up h-11 w-full rounded-full bg-foreground px-6 text-sm font-semibold text-background transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-40 sm:w-fit"
        >
          {busy ? "Starting…" : "Start quantising"}
        </button>
      </div>
    </div>
  );
}
