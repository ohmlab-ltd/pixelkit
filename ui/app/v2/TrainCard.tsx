"use client";

// Train a YOLOX-Nano on this dataset and export it for the Neuro N6.
// Phase 1 surface: epochs + input-size pickers, a Start button that
// schedules the engine's train_yolox job (progress rides the same
// status-bar job segment as labelling), and the list of completed
// runs with their fp32/int8 AP@0.5 and a one-click N6 export zip.

import { useCallback, useEffect, useRef, useState } from "react";

import { apiFetch } from "@/lib/apiFetch";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

type TrainedRun = {
  run: string;
  imgsz: number | null;
  epochs: number | null;
  images: { train: number; val: number } | null;
  ap50Fp32: number | null;
  ap50Int8: number | null;
  minutes: number | null;
  hasExport: boolean;
};

type ActiveJob = { id: string; kind: string; project: string; status: string };

const IMG_SIZES = [128, 192, 256, 320, 416] as const;

function fmtRunDate(run: string): string {
  // 20260820T155545Z
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(run);
  if (!m) return run;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]));
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

const pct = (v: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "-");

export function TrainCard({ projectId }: { projectId: string }) {
  const [runs, setRuns] = useState<TrainedRun[] | null>(null);
  const [epochs, setEpochs] = useState(50);
  const [imgsz, setImgsz] = useState<number>(256);
  const [training, setTraining] = useState(false); // an active train job for THIS dataset
  const [error, setError] = useState<string | null>(null);
  const wasTraining = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/models`);
      if (r.ok) setRuns(((await r.json()) as { models: TrainedRun[] }).models);
    } catch {
      /* engine down; card just shows nothing new */
    }
  }, [projectId]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Watch the global jobs poll for a train job on this dataset so the
  // button disables while one runs and the list refreshes when it ends.
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const tick = async () => {
      let active = false;
      try {
        const r = await apiFetch("/api/jobs/active", { cache: "no-store" });
        if (r.ok) {
          const jobs = ((await r.json()) as { jobs?: ActiveJob[] }).jobs ?? [];
          active = jobs.some((j) => j.kind === "train_yolox" && j.project === projectId);
        }
      } catch {
        active = false;
      }
      if (cancelled) return;
      if (wasTraining.current && !active) void refresh(); // just finished
      wasTraining.current = active;
      setTraining(active);
      timer = window.setTimeout(tick, active ? 3000 : 8000);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [projectId, refresh]);

  const start = async () => {
    setError(null);
    try {
      const r = await apiFetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: "train_yolox",
          project: projectId,
          params: { epochs, imgsz },
        }),
      });
      if (!r.ok) {
        const body = (await r.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? `could not start training (${r.status})`);
      }
      wasTraining.current = true;
      setTraining(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="pk-card rounded-md p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="pk-micro">Train for Neuro N6</h3>
        <span className="text-[11px] text-foreground/40">YOLOX-Nano · Apache-2.0</span>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-foreground/55">
        Trains on this dataset&rsquo;s labelled images, quantizes to int8, and
        packs a model + labels + sketch snippet ready for{" "}
        <span className="font-mono text-[11.5px]">#pragma neuron6</span>.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60">
          Epochs
          <input
            type="number"
            min={1}
            max={500}
            value={epochs}
            onChange={(e) => setEpochs(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
            className="w-16 rounded-md border border-[var(--line)] bg-transparent px-2 py-1 text-[12px] tabular-nums outline-none focus:border-[var(--line-strong)]"
          />
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60">
          Input
          <select
            value={imgsz}
            onChange={(e) => setImgsz(Number(e.target.value))}
            className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[12px] tabular-nums outline-none focus:border-[var(--line-strong)]"
          >
            {IMG_SIZES.map((s) => (
              <option key={s} value={s}>{s}×{s}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => void start()}
          disabled={training}
          className="pk-btn-primary ml-auto rounded-md px-3.5 py-1.5 text-[12px] font-medium disabled:opacity-40"
        >
          {training ? "Training…" : "Start training"}
        </button>
      </div>
      {training && (
        <p className="mt-2 text-[12px] text-foreground/50">
          Running - progress is in the status bar. You can keep working; the
          runs list updates when it finishes.
        </p>
      )}
      {error && <p className="mt-2 text-[12px] text-[var(--bad)]">{error}</p>}

      {runs && runs.length > 0 && (
        <ul className="mt-3 grid gap-1.5">
          {runs.map((r) => (
            <li
              key={r.run}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-[var(--line)] px-3 py-2"
            >
              <span className="min-w-0 text-[12px] text-foreground/75 tabular-nums">
                {fmtRunDate(r.run)}
                <span className="ml-2 text-foreground/40">
                  {r.imgsz ?? "?"}px · {r.epochs ?? "?"} epochs
                </span>
              </span>
              <span className="flex shrink-0 items-center gap-3 text-[11.5px] tabular-nums">
                <span className="text-foreground/60" title="AP@0.5, float model / int8 export">
                  AP50 {pct(r.ap50Fp32)}
                  <span className="text-foreground/35"> → int8 {pct(r.ap50Int8)}</span>
                </span>
                {r.hasExport && (
                  <a
                    href={`${API}/api/v2/projects/${projectId}/models/${r.run}/export`}
                    className="rounded-md border border-[var(--line)] px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-foreground/60 transition-colors hover:border-[var(--line-strong)] hover:text-foreground"
                  >
                    N6 export
                  </a>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
