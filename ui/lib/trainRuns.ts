// Types and pure helpers for the Train card: the engine's trained-run
// records (GET /api/v2/projects/:id/models and .../models/:run), the
// pre-flight report, and the small formatting + charting functions the
// card renders with. Kept free of React so they can be unit-tested.

export type Neuron6Summary = {
  verified?: boolean | null;
  available?: boolean | null;
  error?: string | null;
  reason?: string | null;
  epochsSoftware?: number | null;
  epochsHardware?: number | null;
  epochsTotal?: number | null;
  weightsBytes?: number | null;
  activationsBytes?: number | null;
  poolRemainingBytes?: number | null;
  poolUsedPct?: number | null;
};

export type HistoryRow = {
  epoch: number;
  phase: "float" | "qat";
  loss: number;
  lr?: number;
  ap50: number | null;
  seconds?: number;
  batch?: number;
};

export type ThresholdSummary = {
  conf: number;
  f1?: number | null;
  precision?: number | null;
  recall?: number | null;
  calibrated?: boolean;
  fallback?: boolean | null; // the model found nothing on val: the default was written
};

// Reports written before the flag existed carry calibrated=false with an
// all-zero sweep when the model found nothing: read that as the fallback.
export function isThresholdFallback(thr: ThresholdSummary): boolean {
  if (thr.fallback) return true;
  return !thr.calibrated && (thr.f1 ?? 0) === 0 && (thr.precision ?? 0) === 0 && (thr.recall ?? 0) === 0;
}

export function thresholdNote(thr: ThresholdSummary): string {
  if (thr.calibrated) {
    return `chosen on the val split for the best F1 (${(thr.f1 ?? 0).toFixed(2)}): precision ${pct(thr.precision)}, recall ${pct(thr.recall)}. Written into the labels header.`;
  }
  if (isThresholdFallback(thr)) return "the default: the model found nothing on the val split at any threshold.";
  return "as configured for this run.";
}

export type RunStatus = "done" | "rejected" | "training" | "interrupted" | "failed";

export type TrainedRun = {
  run: string;
  model: string;
  size: "nano" | "tiny" | string;
  status: RunStatus;
  resumable: boolean;
  error?: string | null;
  imgsz: number | null;
  epochs: number | null;
  qatEpochs?: number | null;
  epochsDone?: number;
  totalEpochs?: number | null;
  images: { train: number; val: number; augmented?: number; scoredOnTrain?: boolean } | null;
  split?: { ratio: number; stratified: boolean; evalProject: string | null } | null;
  batch?: number | null;
  ap50Fp32: number | null;
  ap50Int8: number | null;
  minutes: number | null;
  hasExport: boolean;
  exportRejected?: boolean;
  initRun?: string | null;
  resumed?: boolean;
  threshold?: ThresholdSummary | null;
  failuresCount?: number;
  warningsCount?: number;
  neuron6?: Neuron6Summary | null;
  history?: HistoryRow[] | null;
};

export type ClassReportRow = {
  name: string;
  support: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

export type FailureItem = {
  kind: "fp" | "fn" | "wrong";
  class: string;
  predicted?: string;
  score: number | null;
  box: number[];
};

export type FailureEntry = {
  file: string;
  image: string;
  index: number;
  fn: number;
  fp: number;
  wrong: number;
  items: FailureItem[];
  width: number;
  height: number;
};

export type RunReport = {
  classes?: string[];
  history?: HistoryRow[];
  threshold?: ThresholdSummary & { sweep?: { conf: number; precision: number; recall: number; f1: number }[] };
  classReport?: ClassReportRow[];
  confusion?: number[][];
  confusionLabels?: string[];
  failures?: FailureEntry[];
  per_class_fp32?: Record<string, number | null>;
  per_class_int8?: Record<string, number | null>;
  warnings?: string[];
  notes?: string[];
  neuron6?: Neuron6Summary;
  exportRejected?: boolean;
};

export type RunDetail = TrainedRun & {
  report: RunReport | null;
  live: { history?: HistoryRow[]; notes?: string[]; error?: string | null; status?: string } | null;
};

export type PreflightClass = {
  name: string;
  trainImages: number;
  valImages: number;
  trainBoxes: number;
  valBoxes: number;
  augmentedBoxes: number;
  smallBoxesPct?: number;
};

export type Preflight = {
  ok: boolean;
  blockers: string[];
  warnings: string[];
  classes: PreflightClass[];
  images: { train: number; val: number; augmented: number; unlabelled: number } | null;
};

export const IMG_SIZES = [128, 192, 256, 320, 416] as const;
export const SPLITS = [0.6, 0.7, 0.8, 0.9] as const;

export function fmtRunDate(run: string): string {
  // 20260820T155545Z
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(run);
  if (!m) return run;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]));
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export const pct = (v: number | null | undefined): string =>
  typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "-";

export const kb = (b: number | null | undefined): string =>
  typeof b === "number"
    ? b >= 1024 * 1024
      ? `${(b / 1024 / 1024).toFixed(2)} MB`
      : `${Math.round(b / 1024)} kB`
    : "-";

export function sizeLabel(size: string | null | undefined): string {
  return size === "tiny" ? "YOLOX-Tiny" : "YOLOX-Nano";
}

export function statusLabel(r: Pick<TrainedRun, "status" | "epochsDone" | "totalEpochs">): string {
  switch (r.status) {
    case "training":
      return r.totalEpochs ? `Training · epoch ${r.epochsDone ?? 0}/${r.totalEpochs}` : "Training";
    case "interrupted":
      return r.totalEpochs
        ? `Interrupted at epoch ${r.epochsDone ?? 0}/${r.totalEpochs}`
        : "Interrupted";
    case "failed":
      return "Failed";
    case "rejected":
      return "Rejected by ST Edge AI";
    default:
      return "Done";
  }
}

// A polyline in a fixed viewBox for the training curves: loss on the
// left axis (normalised to its own max), AP@0.5 as points on a 0..1
// axis. Returns null when there is nothing to draw.
export type CurveGeometry = {
  width: number;
  height: number;
  lossPath: string;
  apPath: string;
  apPoints: { x: number; y: number; epoch: number; ap: number }[];
  qatStartX: number | null;
  maxLoss: number;
  epochs: number;
};

export function curveGeometry(
  history: HistoryRow[] | null | undefined,
  totalEpochs: number | null | undefined,
  width = 320,
  height = 96,
): CurveGeometry | null {
  const rows = (history ?? []).filter((h) => typeof h.epoch === "number" && Number.isFinite(h.loss));
  if (rows.length === 0) return null;
  const epochs = Math.max(totalEpochs ?? 0, rows[rows.length - 1].epoch, 1);
  const pad = { l: 4, r: 4, t: 4, b: 4 };
  const w = width - pad.l - pad.r;
  const h = height - pad.t - pad.b;
  const x = (e: number) => pad.l + (epochs <= 1 ? w / 2 : ((e - 1) / (epochs - 1)) * w);
  const maxLoss = Math.max(...rows.map((r) => r.loss), 1e-6);
  const yLoss = (v: number) => pad.t + h - (v / maxLoss) * h;
  const yAp = (v: number) => pad.t + h - Math.max(0, Math.min(1, v)) * h;
  const lossPath = rows
    .map((r, i) => `${i === 0 ? "M" : "L"}${x(r.epoch).toFixed(1)},${yLoss(r.loss).toFixed(1)}`)
    .join(" ");
  const apRows = rows.filter((r) => typeof r.ap50 === "number");
  const apPoints = apRows.map((r) => ({ x: x(r.epoch), y: yAp(r.ap50 as number), epoch: r.epoch, ap: r.ap50 as number }));
  const apPath = apPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const firstQat = rows.find((r) => r.phase === "qat");
  return {
    width,
    height,
    lossPath,
    apPath,
    apPoints,
    qatStartX: firstQat ? x(firstQat.epoch) : null,
    maxLoss,
    epochs,
  };
}

// Confusion-matrix cell shade: 0 for an empty cell, up to 1 for the
// largest cell in its row (the truth), so a row reads as "where did
// this class go".
export function confusionShade(row: number[], value: number): number {
  const max = Math.max(...row, 0);
  if (max <= 0 || value <= 0) return 0;
  return 0.15 + 0.85 * (value / max);
}

export function etaFromProgressText(text: string | null | undefined): string | null {
  // The engine appends "· 12 min left" to the phase text.
  if (!text) return null;
  const m = /([0-9]+(?: h [0-9]{2} min| min| s)) left/.exec(text);
  return m ? m[1] : null;
}
