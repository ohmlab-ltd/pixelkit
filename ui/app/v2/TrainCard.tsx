"use client";

// Train a YOLOX model on this dataset and export it for the Neuro N6.
//
// The card is the whole "dataset in, deployable model out" surface:
// the run options (epochs, input, Nano/Tiny, split, augmentations,
// warm start, held-out dataset), a pre-flight check of what the run
// would use and what would go wrong, the runs list with live curves
// while one trains, and per run the training curves, the calibrated
// threshold, per-class precision/recall, the confusion matrix, the
// worst val images, plus the actions: N6 export, label images with
// the model, resume an interrupted run, cancel, delete.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { API_BASE, apiFetch } from "@/lib/apiFetch";
import {
  IMG_SIZES,
  SPLITS,
  confusionShade,
  curveGeometry,
  etaFromProgressText,
  fmtRunDate,
  kb,
  pct,
  sizeLabel,
  statusLabel,
  thresholdNote,
  type ClassReportRow,
  type HistoryRow,
  type Neuron6Summary,
  type Preflight,
  type RunDetail,
  type RunStatus,
  type TrainedRun,
} from "@/lib/trainRuns";

import { GlassDialog } from "./GlassDialog";

type ActiveJob = {
  id: string;
  kind: string;
  project: string;
  status: string;
  progress?: { index?: number | null; total?: number | null; image?: string | null } | null;
};

type FinishedJob = { status?: string; error?: string | null };

type DatasetOption = { id: string; name: string };

// Tailwind's opacity modifier cannot apply to a bare CSS-variable colour
// (it needs RGB channels), so tinted backgrounds use color-mix inline.
const tint = (colour: string, pctOf = 15) => `color-mix(in srgb, ${colour} ${pctOf}%, transparent)`;

async function readDetail(r: Response, fallback: string): Promise<string> {
  const body = (await r.json().catch(() => null)) as { detail?: string } | null;
  return body?.detail ?? fallback;
}

// One line of NPU truth per run: did ST Edge AI build it, how much ran
// in software, and what it costs against the board's weights pool.
function Neuron6Line({ n6, rejected }: { n6: Neuron6Summary | null | undefined; rejected?: boolean }) {
  if (rejected || n6?.error) {
    return (
      <span className="text-[11px] text-[var(--bad)]" title={n6?.error ?? "ST Edge AI rejected this export"}>
        Rejected by ST Edge AI{n6?.error ? `: ${n6.error.slice(0, 80)}` : ""}
      </span>
    );
  }
  if (!n6 || !n6.verified) {
    return (
      <span className="text-[11px] text-foreground/40" title={n6?.reason ?? undefined}>
        NPU build not verified{n6?.available === false ? " (ST Edge AI not installed)" : ""}
      </span>
    );
  }
  const sw = n6.epochsSoftware ?? 0;
  return (
    <span className="text-[11px] tabular-nums text-foreground/55" title="Compiled with ST Edge AI against the Neuro N6 memory map">
      <span className={sw === 0 ? "text-[var(--ok)]" : "text-[var(--warn,#c9a227)]"}>
        {sw === 0 ? "Fully on NPU" : `${sw} software epoch${sw === 1 ? "" : "s"}`}
      </span>
      {" · "}weights {kb(n6.weightsBytes)}
      {typeof n6.poolUsedPct === "number" && ` (${n6.poolUsedPct.toFixed(1)}% of pool)`}
      {" · "}activations {kb(n6.activationsBytes)}
    </span>
  );
}

// Loss (line) and AP@0.5 (dots) per epoch; the dashed line marks where
// quantisation-aware fine-tuning starts.
function Curves({ history, total }: { history: HistoryRow[] | null | undefined; total: number | null | undefined }) {
  const g = useMemo(() => curveGeometry(history, total, 320, 96), [history, total]);
  if (!g || !history || history.length === 0) return null;
  const last = history[history.length - 1];
  const lastAp = [...history].reverse().find((h) => typeof h.ap50 === "number");
  return (
    <div className="flex flex-wrap items-end gap-3">
      <svg
        viewBox={`0 0 ${g.width} ${g.height}`}
        className="h-24 w-full max-w-[320px] rounded-md border border-[var(--line)]"
        role="img"
        aria-label="training curves"
      >
        {g.qatStartX != null && (
          <line x1={g.qatStartX} x2={g.qatStartX} y1={2} y2={g.height - 2} stroke="currentColor" strokeOpacity={0.25} strokeDasharray="3 3" />
        )}
        <path d={g.lossPath} fill="none" stroke="currentColor" strokeOpacity={0.45} strokeWidth={1.25} />
        {g.apPath && <path d={g.apPath} fill="none" stroke="var(--ok)" strokeWidth={1.25} />}
        {g.apPoints.map((p) => (
          <circle key={p.epoch} cx={p.x} cy={p.y} r={2} fill="var(--ok)">
            <title>{`epoch ${p.epoch}: AP@0.5 ${pct(p.ap)}`}</title>
          </circle>
        ))}
      </svg>
      <div className="text-[11px] tabular-nums text-foreground/55">
        <div>
          <span className="inline-block h-[2px] w-3 bg-foreground/45 align-middle" /> loss {last.loss.toFixed(2)}
          <span className="text-foreground/35"> (max {g.maxLoss.toFixed(1)})</span>
        </div>
        <div>
          <span className="inline-block h-[2px] w-3 bg-[var(--ok)] align-middle" /> AP@0.5{" "}
          {lastAp ? pct(lastAp.ap50) : "-"}
        </div>
        <div className="text-foreground/40">
          epoch {last.epoch}/{g.epochs}
          {g.qatStartX != null ? " · dashed: int8 fine-tuning" : ""}
        </div>
      </div>
    </div>
  );
}

function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  if (labels.length === 0 || matrix.length !== labels.length) return null;
  return (
    <div className="min-w-0 overflow-x-auto">
      <table className="text-[10.5px] tabular-nums">
        <thead>
          <tr>
            <th className="pr-2 text-left font-normal text-foreground/40">truth ↓ · predicted →</th>
            {labels.map((l) => (
              <th key={l} className="px-1 pb-1 text-right font-normal text-foreground/50">
                <span className="inline-block max-w-[64px] truncate align-bottom" title={l}>{l}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <td className="pr-2 text-foreground/60">
                <span className="inline-block max-w-[96px] truncate align-bottom" title={labels[i]}>{labels[i]}</span>
              </td>
              {row.map((v, j) => {
                const shade = confusionShade(row, v);
                const diag = i === j && i < labels.length - 1;
                return (
                  <td
                    key={j}
                    className="px-1 text-right"
                    style={{
                      background: shade > 0 ? tint(diag ? "var(--ok)" : "var(--bad)", Math.round(shade * 55)) : undefined,
                    }}
                    title={`${labels[i]} → ${labels[j]}: ${v}`}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type ClassRow = Partial<ClassReportRow> & { name: string };

function RunDetails({
  projectId,
  detail,
  liveHistory,
  status,
}: {
  projectId: string;
  detail: RunDetail;
  liveHistory: HistoryRow[] | null | undefined;
  status: RunStatus;
}) {
  const rep = detail.report;
  // A run in progress: the list's live history is fresher than the
  // detail fetched when the panel opened.
  const history =
    status === "training" && liveHistory && liveHistory.length > 0
      ? liveHistory
      : rep?.history ?? detail.live?.history ?? detail.history ?? null;
  const thr = rep?.threshold;
  // Reports from the previous engine version have per-class AP but no
  // class report: still show what they have.
  const rows: ClassRow[] =
    rep?.classReport && rep.classReport.length > 0
      ? rep.classReport
      : Object.keys(rep?.per_class_int8 ?? rep?.per_class_fp32 ?? {}).map((name) => ({ name }));
  const [showAllFailures, setShowAllFailures] = useState(false);
  const [zoomed, setZoomed] = useState<string | null>(null);
  const failures = rep?.failures ?? [];
  const shown = showAllFailures ? failures : failures.slice(0, 8);
  const failureUrl = (file: string) => `${API_BASE}/api/v2/projects/${projectId}/models/${detail.run}/failures/${file}`;
  const error = detail.error ?? detail.live?.error ?? null;
  return (
    <div className="mt-2 grid min-w-0 gap-3 border-t border-[var(--line)] pt-3">
      {error && (status === "failed" || status === "interrupted") && (
        <p className="text-[12px] text-[var(--bad)]">{error}</p>
      )}
      <Curves history={history} total={detail.totalEpochs} />

      {thr && (
        <p className="text-[12px] text-foreground/65">
          <span className="font-medium text-foreground/80">Confidence threshold {thr.conf.toFixed(2)}</span>
          {" · "}
          {thresholdNote(thr)}
        </p>
      )}

      {rows.length > 0 && (
        <div className="min-w-0 overflow-x-auto">
          <table className="w-full text-[11px] tabular-nums">
            <thead className="text-foreground/40">
              <tr>
                <th className="text-left font-normal">class</th>
                <th className="text-right font-normal">AP float</th>
                <th className="text-right font-normal">AP int8</th>
                <th className="text-right font-normal" title="at the threshold above">precision</th>
                <th className="text-right font-normal" title="at the threshold above">recall</th>
                <th className="text-right font-normal">val boxes</th>
              </tr>
            </thead>
            <tbody className="text-foreground/70">
              {rows.map((c) => (
                <tr key={c.name} className="border-t border-[var(--line)]">
                  <td className="py-0.5 pr-2">{c.name}</td>
                  <td className="text-right">{pct(rep?.per_class_fp32?.[c.name])}</td>
                  <td className="text-right">{pct(rep?.per_class_int8?.[c.name])}</td>
                  <td className="text-right">{pct(c.precision)}</td>
                  <td className="text-right">{pct(c.recall)}</td>
                  <td className="text-right text-foreground/45">{typeof c.support === "number" ? c.support : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {rep?.confusion && rep.confusionLabels && (
        <div className="min-w-0">
          <div className="pk-micro mb-1 text-foreground/45">Confusion matrix (val, at the threshold)</div>
          <ConfusionMatrix labels={rep.confusionLabels} matrix={rep.confusion} />
        </div>
      )}

      {failures.length > 0 && (
        <div className="min-w-0">
          <div className="pk-micro mb-1 text-foreground/45">
            Worst val images · green truth, blue correct,{" "}
            <span className="text-[var(--warn,#c9a227)]">yellow</span> missed,{" "}
            <span className="text-[var(--bad)]">red</span> spurious, orange wrong class. Click to enlarge.
          </div>
          {zoomed && (
            <figure className="mb-2 min-w-0">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={failureUrl(zoomed)}
                alt={zoomed}
                className="w-full cursor-zoom-out rounded-md border border-[var(--line)]"
                onClick={() => setZoomed(null)}
              />
            </figure>
          )}
          <div className="grid grid-cols-2 gap-2">
            {shown.map((f) => (
              <figure key={f.file} className="min-w-0">
                <button
                  type="button"
                  onClick={() => setZoomed(zoomed === f.file ? null : f.file)}
                  className="block w-full cursor-zoom-in"
                  aria-label={`Enlarge ${f.image}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={failureUrl(f.file)}
                    alt={f.image}
                    loading="lazy"
                    className="w-full rounded-md border border-[var(--line)]"
                  />
                </button>
                <figcaption className="mt-0.5 truncate text-[10.5px] text-foreground/50" title={f.image}>
                  {[
                    f.fn ? `${f.fn} missed` : null,
                    f.fp ? `${f.fp} spurious` : null,
                    f.wrong ? `${f.wrong} wrong class` : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                  {" · "}
                  {f.image}
                </figcaption>
              </figure>
            ))}
          </div>
          {failures.length > 8 && (
            <button
              type="button"
              onClick={() => setShowAllFailures((v) => !v)}
              className="mt-1 text-[11px] text-foreground/50 underline-offset-2 hover:underline"
            >
              {showAllFailures ? "Show fewer" : `Show all ${failures.length}`}
            </button>
          )}
        </div>
      )}
      {rep && rows.length > 0 && rep.failures && rep.failures.length === 0 && (
        <p className="text-[11.5px] text-foreground/45">No errors on the val split at this threshold.</p>
      )}

      {(rep?.warnings?.length || rep?.notes?.length || detail.live?.notes?.length) ? (
        <ul className="grid gap-0.5 text-[11px] text-foreground/50">
          {[...(rep?.warnings ?? []), ...(rep?.notes ?? []), ...(detail.live?.notes ?? [])].map((w, i) => (
            <li key={i}>· {w}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function TrainCard({ projectId }: { projectId: string }) {
  const [runs, setRuns] = useState<TrainedRun[] | null>(null);
  // Raw string so clearing the field to retype doesn't snap to 1 and
  // turn "20" into "120"; clamped when read.
  const [epochsText, setEpochsText] = useState("50");
  const epochs = Math.max(1, Math.min(500, Number(epochsText) || 50));
  const [imgsz, setImgsz] = useState<number>(256);
  const [size, setSize] = useState<"nano" | "tiny">("nano");
  const [split, setSplit] = useState<number>(0.8);
  const [useAug, setUseAug] = useState(true);
  const [initRun, setInitRun] = useState<string>("");
  const [evalProject, setEvalProject] = useState<string>("");
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [preflight, setPreflight] = useState<Preflight | null>(null);
  // Bumped when a labelling job on this dataset finishes: the label
  // counts the pre-flight reports have changed.
  const [labelsVersion, setLabelsVersion] = useState(0);
  const [training, setTraining] = useState(false); // an active train job for THIS dataset
  const [queued, setQueued] = useState(false); // ...still waiting behind another job
  const [labelling, setLabelling] = useState(false); // an active label_model job for THIS dataset
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const wasTraining = useRef(false);
  const wasLabelling = useRef(false);
  // Job ids this card started, so a job that fails before it writes a
  // run folder (no GPU, labels changed) still gets its error shown.
  const trainJobId = useRef<string | null>(null);
  const labelJobId = useRef<string | null>(null);
  const openRef = useRef<string | null>(null);
  openRef.current = open;
  // The card is re-rendered, not remounted, on a project switch: a late
  // response for the previous project must not overwrite this one's list.
  const fetchSeq = useRef(0);
  const preflightSeq = useRef(0);

  const refresh = useCallback(async () => {
    const seq = ++fetchSeq.current;
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/models`, { cache: "no-store" });
      if (r.ok) {
        const models = ((await r.json()) as { models: TrainedRun[] }).models;
        if (seq === fetchSeq.current) setRuns(models);
      }
    } catch {
      /* engine down; card just shows nothing new */
    }
  }, [projectId]);

  const loadDetail = useCallback(
    async (run: string) => {
      const seq = fetchSeq.current;
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/models/${run}`, { cache: "no-store" });
        if (r.ok) {
          const d = (await r.json()) as RunDetail;
          if (seq === fetchSeq.current) setDetails((prev) => ({ ...prev, [run]: d }));
        }
      } catch {
        /* the row still shows its summary */
      }
    },
    [projectId],
  );

  useEffect(() => {
    setRuns(null);
    setError(null);
    setNotice(null);
    setOpen(null);
    setDetails({});
    setPreflight(null);
    setInitRun("");
    setEvalProject("");
    setDatasets([]);
    setActiveJobId(null);
    setProgressText(null);
    setConfirmDelete(null);
    wasTraining.current = false;
    wasLabelling.current = false;
    trainJobId.current = null;
    labelJobId.current = null;
    setTraining(false);
    setQueued(false);
    setLabelling(false);
    void refresh();
    // Other datasets in the workspace, for the held-out picker.
    const seq = fetchSeq.current;
    (async () => {
      try {
        const r = await apiFetch("/api/projects");
        if (!r.ok) return;
        const body = (await r.json()) as unknown;
        const items = (Array.isArray(body) ? body : ((body as { items?: unknown[] }).items ?? [])) as {
          id?: string;
          project_id?: string;
          name?: string;
        }[];
        const opts = items
          .map((it) => ({ id: String(it.id ?? it.project_id ?? ""), name: String(it.name ?? "") }))
          .filter((it) => it.id && it.id !== projectId);
        if (seq === fetchSeq.current) setDatasets(opts);
      } catch {
        /* optional */
      }
    })();
  }, [projectId, refresh]);

  // Watch the global jobs poll for train / label_model jobs on this
  // dataset so the buttons disable while one runs, the list refreshes
  // when one ends (and every few seconds during training, for the live
  // curves), and a job that failed says why.
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    let ticks = 0;
    const finished = async (jobId: string | null, what: "Training" | "Labelling") => {
      if (!jobId) return;
      try {
        const r = await apiFetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        if (!r.ok) return;
        const j = (await r.json()) as FinishedJob;
        if (cancelled) return;
        if (j.status === "failed") setError(`${what} failed: ${j.error ?? "no detail"}`);
        else if (j.status === "cancelled") {
          setNotice(what === "Training"
            ? "Training cancelled. The run keeps its last checkpoint and can be resumed."
            : "Labelling cancelled; the images done so far keep their boxes.");
        } else if (j.status === "done" && what === "Labelling") {
          setNotice("Labelling finished. Review the new boxes in the gallery; hand-edited images were not touched.");
        }
      } catch {
        /* nothing to add */
      }
    };
    const tick = async () => {
      let active = false;
      let isQueued = false;
      let lab = false;
      let text: string | null = null;
      let jobId: string | null = null;
      try {
        const r = await apiFetch("/api/jobs/active", { cache: "no-store" });
        if (r.ok) {
          const jobs = ((await r.json()) as { jobs?: ActiveJob[] }).jobs ?? [];
          const mine = jobs.find((j) => j.kind === "train_yolox" && j.project === projectId);
          active = !!mine;
          isQueued = mine?.status === "queued";
          jobId = mine?.id ?? null;
          if (mine) trainJobId.current = mine.id;
          const labJob = jobs.find((j) => j.kind === "label_model" && j.project === projectId);
          lab = !!labJob;
          if (labJob) labelJobId.current = labJob.id;
          if (mine?.progress) {
            const p = mine.progress;
            text = p.index != null && p.total ? `epoch ${p.index}/${p.total}${p.image ? ` · ${p.image}` : ""}` : (p.image ?? null);
          }
        }
      } catch {
        active = false;
      }
      if (cancelled) return;
      if (wasTraining.current && !active) {
        // Just finished: the list, the open panel (live history became
        // a report), and the job's own verdict.
        void refresh();
        const id = trainJobId.current;
        trainJobId.current = null;
        setDetails({});
        if (openRef.current) void loadDetail(openRef.current);
        void finished(id, "Training");
      } else if (active && ticks++ % 3 === 0) {
        void refresh(); // live history
      }
      if (wasLabelling.current && !lab) {
        const id = labelJobId.current;
        labelJobId.current = null;
        setLabelsVersion((v) => v + 1);
        void finished(id, "Labelling");
        window.dispatchEvent(new CustomEvent("pixelkit-labels-changed", { detail: { projectId } }));
      }
      wasTraining.current = active;
      wasLabelling.current = lab;
      setTraining(active);
      setQueued(isQueued);
      setLabelling(lab);
      setActiveJobId(jobId);
      setProgressText(text);
      timer = window.setTimeout(tick, active || lab ? 3000 : 8000);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [projectId, refresh, loadDetail]);

  // Pre-flight: what the run would use, debounced on option changes and
  // re-run when a labelling job changed the counts. Not on every list
  // refresh: the check reads every image header.
  useEffect(() => {
    const seq = ++preflightSeq.current;
    const t = window.setTimeout(async () => {
      const failed = (msg: string) => {
        if (seq === preflightSeq.current) {
          setPreflight({ ok: false, blockers: [msg], warnings: [], classes: [], images: null });
        }
      };
      try {
        const q = new URLSearchParams({
          imgsz: String(imgsz),
          split: String(split),
          size,
          augmentations: useAug ? "1" : "0",
          evalProject: evalProject,
        });
        const r = await apiFetch(`/api/v2/projects/${projectId}/train/preflight?${q}`, { cache: "no-store" });
        if (!r.ok) {
          failed(`pre-flight check failed: ${await readDetail(r, `engine returned ${r.status}`)}`);
          return;
        }
        const body = (await r.json()) as Preflight;
        if (seq === preflightSeq.current) setPreflight(body);
      } catch {
        failed("pre-flight check failed: the engine is not reachable");
      }
    }, 350);
    return () => window.clearTimeout(t);
  }, [projectId, imgsz, split, size, useAug, evalProject, labelsVersion]);

  const postJob = async (kind: "train_yolox" | "label_model", params: Record<string, unknown>) => {
    const r = await apiFetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, project: projectId, params }),
    });
    if (!r.ok) throw new Error(await readDetail(r, `could not start (${r.status})`));
    return ((await r.json()) as { jobId?: string }).jobId ?? null;
  };

  const start = async () => {
    setError(null);
    setNotice(null);
    try {
      const params: Record<string, unknown> = { epochs, imgsz, size, split, augmentations: useAug };
      if (initRun) params.initRun = initRun;
      if (evalProject) params.evalProject = evalProject;
      trainJobId.current = await postJob("train_yolox", params);
      wasTraining.current = true;
      setTraining(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const resume = async (run: string) => {
    setError(null);
    setNotice(null);
    setBusy(run);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/models/${run}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!r.ok) throw new Error(await readDetail(r, `could not resume (${r.status})`));
      trainJobId.current = ((await r.json()) as { jobId?: string }).jobId ?? null;
      wasTraining.current = true;
      setTraining(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const cancelTraining = async () => {
    const id = activeJobId ?? trainJobId.current;
    if (!id) return;
    setError(null);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/jobs/${id}`, { method: "DELETE" });
      if (!r.ok) throw new Error(await readDetail(r, `could not cancel (${r.status})`));
      setNotice("Cancelling after the current epoch. The run keeps its last checkpoint and can be resumed.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const labelWith = async (run: string) => {
    setError(null);
    setNotice(null);
    setBusy(run);
    try {
      labelJobId.current = await postJob("label_model", { run });
      wasLabelling.current = true;
      setLabelling(true);
      setNotice("Labelling the unlabelled images with this model. Progress is in the status bar; review the boxes in the gallery when it finishes.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const remove = async (run: string) => {
    setConfirmDelete(null);
    setError(null);
    setBusy(run);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/models/${run}`, { method: "DELETE" });
      if (!r.ok) throw new Error(await readDetail(r, `could not delete (${r.status})`));
      if (open === run) setOpen(null);
      if (initRun === run) setInitRun("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const toggle = async (run: string) => {
    if (open === run) {
      setOpen(null);
      return;
    }
    setOpen(run);
    await loadDetail(run);
  };

  const doneRuns = (runs ?? []).filter((r) => r.status === "done");
  const blockers = preflight?.blockers ?? [];
  const warnings = preflight?.warnings ?? [];
  const eta = etaFromProgressText(progressText);
  const canStart = !training && blockers.length === 0 && preflight !== null;
  const pillColour = (s: RunStatus) =>
    s === "training" ? "var(--ok)" : s === "interrupted" ? "var(--warn,#c9a227)" : "var(--bad)";

  return (
    <div className="pk-card min-w-0 rounded-md p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="pk-micro">Train for Neuro N6</h3>
        <span className="text-[11px] text-foreground/40">YOLOX · Apache-2.0</span>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-foreground/55">
        Trains on this dataset&rsquo;s labelled images and their augmentations, quantises to int8,
        calibrates the confidence threshold, and packs a Neuro N6 sketch ready to build and upload.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60">
          Epochs
          <input
            type="number"
            min={1}
            max={500}
            value={epochsText}
            onChange={(e) => setEpochsText(e.target.value)}
            onBlur={() => setEpochsText(String(epochs))}
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
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60" title="Nano: 0.9 M parameters, under 1 MB of weights. Tiny: 5 M parameters, about 5 MB, more accurate, about six times the compute.">
          Model
          <select
            value={size}
            onChange={(e) => setSize(e.target.value === "tiny" ? "tiny" : "nano")}
            className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[12px] outline-none focus:border-[var(--line-strong)]"
          >
            <option value="nano">Nano (smallest)</option>
            <option value="tiny">Tiny (more accurate)</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60" title="Share of images that train; the rest is held out per class to measure accuracy.">
          Split
          <select
            value={split}
            onChange={(e) => setSplit(Number(e.target.value))}
            disabled={!!evalProject}
            className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[12px] tabular-nums outline-none focus:border-[var(--line-strong)] disabled:opacity-40"
          >
            {SPLITS.map((s) => (
              <option key={s} value={s}>{Math.round(s * 100)}% train</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-foreground/60" title="Generated augmentations of training images join the training set (never the val split).">
          <input type="checkbox" checked={useAug} onChange={(e) => setUseAug(e.target.checked)} />
          Augmentations
          {preflight?.images && useAug && preflight.images.augmented > 0 && (
            <span className="text-foreground/40 tabular-nums">({preflight.images.augmented})</span>
          )}
        </label>
        {doneRuns.length > 0 && (
          <label className="flex items-center gap-1.5 text-[12px] text-foreground/60" title="Continue from a previous run's weights instead of the COCO checkpoint.">
            Start from
            <select
              value={initRun}
              onChange={(e) => setInitRun(e.target.value)}
              className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[12px] outline-none focus:border-[var(--line-strong)]"
            >
              <option value="">COCO checkpoint</option>
              {doneRuns.map((r) => (
                <option key={r.run} value={r.run}>
                  run {fmtRunDate(r.run)} ({sizeLabel(r.size).replace("YOLOX-", "")}, int8 {pct(r.ap50Int8)})
                </option>
              ))}
            </select>
          </label>
        )}
        {datasets.length > 0 && (
          <label className="flex items-center gap-1.5 text-[12px] text-foreground/60" title="Score on a separate dataset instead of a held-out split; everything here trains.">
            Evaluate on
            <select
              value={evalProject}
              onChange={(e) => setEvalProject(e.target.value)}
              className="max-w-[200px] rounded-md border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-[12px] outline-none focus:border-[var(--line-strong)]"
            >
              <option value="">held-out split</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name || d.id}</option>
              ))}
            </select>
          </label>
        )}
        {training ? (
          <button
            type="button"
            onClick={() => void cancelTraining()}
            disabled={!activeJobId && !trainJobId.current}
            className="ml-auto rounded-md border border-[var(--line)] px-3.5 py-1.5 text-[12px] text-foreground/70 transition-colors hover:border-[var(--line-strong)] hover:text-foreground disabled:opacity-40"
          >
            Cancel training
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void start()}
            disabled={!canStart}
            className="pk-btn-primary ml-auto rounded-md px-3.5 py-1.5 text-[12px] font-medium disabled:opacity-40"
            title={blockers.length ? blockers[0] : undefined}
          >
            Start training
          </button>
        )}
      </div>

      {preflight && (
        <div className="mt-2 text-[11.5px] leading-relaxed text-foreground/55">
          {preflight.images && (
            <span className="tabular-nums">
              {preflight.images.train} train · {preflight.images.val} val
              {preflight.images.background ? `, ${preflight.images.background} of them background` : ""}
              {useAug && preflight.images.augmented > 0 ? ` · ${preflight.images.augmented} augmented copies` : ""}
              {preflight.images.unlabelled > 0 ? ` · ${preflight.images.unlabelled} unlabelled (skipped)` : ""}
              {preflight.classes.length > 0 ? ` · ${preflight.classes.length} classes` : ""}
            </span>
          )}
          {blockers.map((b) => (
            <div key={b} className="text-[var(--bad)]">{b}</div>
          ))}
          {warnings.slice(0, 5).map((w) => (
            <div key={w} className="text-[var(--warn,#c9a227)]">{w}</div>
          ))}
          {warnings.length > 5 && <div className="text-foreground/40">and {warnings.length - 5} more warnings</div>}
        </div>
      )}
      {training && (
        <p className="mt-2 text-[12px] text-foreground/50">
          {queued
            ? "Queued behind another job; it starts when that one finishes."
            : `Running${progressText ? ` · ${progressText}` : ""}${eta ? "" : " · progress is in the status bar"}. You can keep working; the run below updates as it trains and can be resumed if the app closes.`}
        </p>
      )}
      {error && <p className="mt-2 text-[12px] text-[var(--bad)]">{error}</p>}
      {notice && <p className="mt-2 text-[12px] text-foreground/60">{notice}</p>}

      {runs && runs.length > 0 && (
        <ul className="mt-3 grid min-w-0 gap-1.5">
          {runs.map((r) => {
            const isOpen = open === r.run;
            const detail = details[r.run];
            const showPill = r.status !== "done" && r.status !== "rejected";
            return (
              <li key={r.run} className="min-w-0 rounded-md border border-[var(--line)] px-3 py-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="min-w-0 text-[12px] text-foreground/75 tabular-nums">
                    {fmtRunDate(r.run)}
                    <span className="ml-2 text-foreground/40">
                      {sizeLabel(r.size).replace("YOLOX-", "")} · {r.imgsz ?? "?"}px · {r.epochs ?? "?"} epochs
                      {typeof r.qatEpochs === "number" && r.qatEpochs > 0 ? ` + ${r.qatEpochs} QAT` : ""}
                      {r.initRun ? ` · from ${fmtRunDate(r.initRun)}` : ""}
                      {r.split?.evalProject ? " · held-out dataset" : ""}
                    </span>
                    {showPill && (
                      <span
                        className="ml-2 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider"
                        style={{ background: tint(pillColour(r.status)), color: pillColour(r.status) }}
                        title={r.error ?? undefined}
                      >
                        {statusLabel(r)}
                      </span>
                    )}
                  </span>
                  <span className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] tabular-nums">
                    {r.status === "done" && (
                      <span className="text-foreground/60" title="AP@0.5, float model / int8 export">
                        AP50 {pct(r.ap50Fp32)}
                        <span className="text-foreground/35"> → int8 {pct(r.ap50Int8)}</span>
                        {r.images?.scoredOnTrain ? <span className="text-foreground/35"> (on train)</span> : null}
                      </span>
                    )}
                    {r.status === "done" && r.threshold && (
                      <span className="text-foreground/50" title="confidence threshold written into the labels header">
                        thr {r.threshold.conf.toFixed(2)}
                        {typeof r.threshold.f1 === "number" && r.threshold.calibrated ? ` · F1 ${r.threshold.f1.toFixed(2)}` : ""}
                      </span>
                    )}
                    {(r.status === "done" || r.status === "rejected") && (
                      <Neuron6Line n6={r.neuron6} rejected={r.exportRejected} />
                    )}
                    {r.resumable && !training && (
                      <button
                        type="button"
                        onClick={() => void resume(r.run)}
                        disabled={busy === r.run}
                        className="rounded-md border border-[var(--line)] px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-foreground/60 transition-colors hover:border-[var(--line-strong)] hover:text-foreground disabled:opacity-40"
                      >
                        Resume
                      </button>
                    )}
                    {r.status === "done" && r.hasExport && (
                      <button
                        type="button"
                        onClick={() => void labelWith(r.run)}
                        disabled={labelling || busy === r.run}
                        title="Pre-label the unlabelled images with this model, at its calibrated threshold"
                        className="rounded-md border border-[var(--line)] px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-foreground/60 transition-colors hover:border-[var(--line-strong)] hover:text-foreground disabled:opacity-40"
                      >
                        {labelling ? "Labelling…" : "Label images"}
                      </button>
                    )}
                    {r.hasExport && (
                      <a
                        href={`${API_BASE}/api/v2/projects/${projectId}/models/${r.run}/export`}
                        className="rounded-md border border-[var(--line)] px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-foreground/60 transition-colors hover:border-[var(--line-strong)] hover:text-foreground"
                      >
                        N6 export
                      </a>
                    )}
                    {(r.status === "done" || r.status === "rejected" || r.history || r.error) && (
                      <button
                        type="button"
                        onClick={() => void toggle(r.run)}
                        className="rounded-md border border-[var(--line)] px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-foreground/60 transition-colors hover:border-[var(--line-strong)] hover:text-foreground"
                      >
                        {isOpen ? "Hide" : "Details"}
                      </button>
                    )}
                    {r.status !== "training" && (
                      <button
                        type="button"
                        onClick={() => setConfirmDelete(r.run)}
                        disabled={busy === r.run || training}
                        title="Delete this run"
                        aria-label="Delete this run"
                        className="rounded-md px-1.5 py-1 text-[12px] text-foreground/35 transition-colors hover:text-[var(--bad)] disabled:opacity-40"
                      >
                        ×
                      </button>
                    )}
                  </span>
                </div>
                {r.error && (r.status === "failed" || r.status === "interrupted") && !isOpen && (
                  <p className="mt-1 text-[11px] text-[var(--bad)]">{r.error}</p>
                )}
                {r.status === "training" && r.history && r.history.length > 0 && !isOpen && (
                  <div className="mt-2">
                    <Curves history={r.history} total={r.totalEpochs} />
                  </div>
                )}
                {isOpen && (detail ? (
                  <RunDetails projectId={projectId} detail={detail} liveHistory={r.history} status={r.status} />
                ) : (
                  <p className="mt-2 text-[11.5px] text-foreground/40">Loading…</p>
                ))}
              </li>
            );
          })}
        </ul>
      )}

      <GlassDialog open={confirmDelete !== null} onClose={() => setConfirmDelete(null)} title="Delete this run?" maxWidth="max-w-md">
        <p className="pt-3 text-[13px] text-foreground/70">
          {confirmDelete ? `The run from ${fmtRunDate(confirmDelete)}` : "This run"}, its weights, its export and its
          gallery are removed from the dataset folder. This cannot be undone.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button type="button" onClick={() => setConfirmDelete(null)} className="pk-btn px-3.5">
            Keep it
          </button>
          <button
            type="button"
            onClick={() => confirmDelete && void remove(confirmDelete)}
            className="pk-btn-primary rounded-md px-3.5 py-1.5 text-[12px] font-medium"
          >
            Delete run
          </button>
        </div>
      </GlassDialog>
    </div>
  );
}
