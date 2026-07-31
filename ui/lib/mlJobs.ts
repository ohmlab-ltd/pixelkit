// Typed client for the standalone training + quantising backend
// (/api/training/* and /api/quantising/*). Thin wrappers over apiFetch
// so the bearer token is attached and errors surface as readable
// messages. No dependency on ST Model Zoo — this talks only to the
// app's own native endpoints.

import { apiFetch } from "./apiFetch";

export type MLJobStatus =
  | "queued" | "preparing" | "running" | "cancelling"
  | "cancelled" | "failed" | "completed";

export const ML_TERMINAL: ReadonlySet<MLJobStatus> = new Set([
  "completed", "failed", "cancelled",
]);

export function mlJobIsActive(status: MLJobStatus): boolean {
  return !ML_TERMINAL.has(status);
}

export type MLJob = {
  id: string;
  job_type: "training" | "quantising";
  status: MLJobStatus;
  user_id: string;
  project_id: string;
  model_id?: string | null;
  config: Record<string, unknown>;
  queue_seq: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  active_runtime_seconds: number;
  charged_blocks: number;
  charged_credits: number;
  error?: string | null;
  progress: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  log_path?: string | null;
  queue_position?: number | null;
};

export type TrainingModelField = {
  key: string;
  kind: "int" | "float" | "bool" | "enum" | "str";
  default: unknown;
  label: string;
  help: string;
  min: number | null;
  max: number | null;
  choices: unknown[] | null;
  required: boolean;
};

export type TrainingModel = {
  id: string;
  label: string;
  description: string;
  task: string;
  quantisable: boolean;
  export_formats: string[];
  fields: TrainingModelField[];
  defaults: Record<string, unknown>;
  est_seconds_per_epoch_per_100img: number;
  est_base_imgsz: number;
  recommended: boolean;
  recommended_note: string;
};

export type TokenEstimate = {
  isEstimate: boolean;
  seconds: number;
  blocks: number;
  credits: number;
  note: string;
};

export type QuantiseOptions = {
  modes: { id: string; label: string; help: string }[];
  default_mode: string;
  output_format: string;
  calibration: { default_samples: number; min_samples: number; max_samples: number; help: string };
};

export type SourceModel = {
  source_job_id: string;
  model_id?: string | null;
  trained_at?: string | null;
  classes: string[];
  name?: string | null;
};

const BLOCK_SECONDS = 15 * 60;

async function errMessage(r: Response): Promise<string> {
  try {
    const d = await r.json();
    return (d && (d.detail || d.error)) || `http ${r.status}`;
  } catch {
    return `http ${r.status}`;
  }
}

async function jget<T>(path: string): Promise<T> {
  const r = await apiFetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await errMessage(r));
  return r.json() as Promise<T>;
}

async function jpost<T>(path: string, body?: unknown): Promise<T> {
  const r = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(await errMessage(r));
  return r.json() as Promise<T>;
}

// ── training ──────────────────────────────────────────────────────────
// The model registry is static per deploy, so cache the response in memory
// + localStorage. The panel reads the cache synchronously for an instant
// first paint (dropdown + config fields appear immediately on revisit), then
// revalidates in the background. The version suffix in the key invalidates
// any stale cached shape after a registry change (e.g. removing a model).
export type TrainingModelsResponse = { models: TrainingModel[]; default: string };
const MODELS_CACHE_KEY = "pk.trainingModels.v2";
let _modelsCache: TrainingModelsResponse | null = null;

export function getCachedTrainingModels(): TrainingModelsResponse | null {
  if (_modelsCache) return _modelsCache;
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(MODELS_CACHE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as TrainingModelsResponse;
      if (parsed && Array.isArray(parsed.models) && parsed.models.length) {
        _modelsCache = parsed;
        return parsed;
      }
    }
  } catch {
    /* private mode / corrupt entry — fall through to a network load */
  }
  return null;
}

export const getTrainingModels = async (): Promise<TrainingModelsResponse> => {
  const data = await jget<TrainingModelsResponse>(`/api/training/models`);
  _modelsCache = data;
  if (typeof window !== "undefined") {
    try { window.localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify(data)); }
    catch { /* quota / private mode — in-memory cache still applies */ }
  }
  return data;
};

export const createTrainingJob = (
  projectId: string,
  body: { model_id?: string; config: Record<string, unknown> },
) => jpost<{ job: MLJob; estimate: TokenEstimate }>(
  `/api/training/projects/${encodeURIComponent(projectId)}/jobs`, body,
);

// Per-project training-jobs cache so the Train tab's "Training jobs" column
// (the user's already-trained models) paints instantly instead of waiting on a
// fetch. Warmed while the project loads; the panel still revalidates in the
// background. Memory is the source of truth within a session; localStorage
// gives an instant paint on the very first click / a fresh reload.
const _trainingJobsCache = new Map<string, MLJob[]>();
const trainingJobsKey = (pid: string) => `pk.trainingJobs.${pid}`;

export function getCachedTrainingJobs(projectId: string): MLJob[] | null {
  const mem = _trainingJobsCache.get(projectId);
  if (mem) return mem;
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(trainingJobsKey(projectId));
    if (raw) {
      const parsed = JSON.parse(raw) as MLJob[];
      if (Array.isArray(parsed)) { _trainingJobsCache.set(projectId, parsed); return parsed; }
    }
  } catch {
    /* private mode / corrupt entry — fall through */
  }
  return null;
}

export const listTrainingJobs = async (projectId: string): Promise<{ jobs: MLJob[] }> => {
  const data = await jget<{ jobs: MLJob[] }>(`/api/training/jobs?project_id=${encodeURIComponent(projectId)}`);
  _trainingJobsCache.set(projectId, data.jobs);
  if (typeof window !== "undefined") {
    try { window.localStorage.setItem(trainingJobsKey(projectId), JSON.stringify(data.jobs)); }
    catch { /* quota / private mode — in-memory cache still applies */ }
  }
  return data;
};

export const getTrainingJob = (jobId: string) =>
  jget<{ job: MLJob }>(`/api/training/jobs/${encodeURIComponent(jobId)}`);

export const cancelTrainingJob = (jobId: string) =>
  jpost<{ ok: boolean; job: MLJob }>(`/api/training/jobs/${encodeURIComponent(jobId)}/cancel`);

export const getTrainingLogs = (jobId: string, tail = 200) =>
  jget<{ status: MLJobStatus; progress: Record<string, unknown>; logs: string[] }>(
    `/api/training/jobs/${encodeURIComponent(jobId)}/logs?tail=${tail}`,
  );

export const getTrainingArtifactUrl = (jobId: string) =>
  jget<{ url: string }>(`/api/training/jobs/${encodeURIComponent(jobId)}/artifact`);

// ── quantising ──────────────────────────────────────────────────────────
// Quantise options are static per deploy — cache like the training models so
// the panel renders its controls instantly on revisit, then revalidates.
const QUANT_OPTS_CACHE_KEY = "pk.quantiseOptions.v1";
let _quantOptsCache: QuantiseOptions | null = null;

export function getCachedQuantiseOptions(): QuantiseOptions | null {
  if (_quantOptsCache) return _quantOptsCache;
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(QUANT_OPTS_CACHE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as QuantiseOptions;
      if (parsed && Array.isArray(parsed.modes)) { _quantOptsCache = parsed; return parsed; }
    }
  } catch {
    /* fall through to a network load */
  }
  return null;
}

export const getQuantiseOptions = async (): Promise<QuantiseOptions> => {
  const o = await jget<QuantiseOptions>(`/api/quantising/options`);
  _quantOptsCache = o;
  if (typeof window !== "undefined") {
    try { window.localStorage.setItem(QUANT_OPTS_CACHE_KEY, JSON.stringify(o)); }
    catch { /* quota / private mode */ }
  }
  return o;
};

export const getQuantiseSourceModels = (projectId: string) =>
  jget<{ models: SourceModel[] }>(
    `/api/quantising/projects/${encodeURIComponent(projectId)}/source-models`,
  );

export const createQuantiseJob = (
  projectId: string,
  body: { source_job_id?: string | null; mode?: string; calibration_samples?: number; output_name?: string },
) => jpost<{ job: MLJob }>(
  `/api/quantising/projects/${encodeURIComponent(projectId)}/jobs`, body,
);

export const listQuantiseJobs = (projectId: string) =>
  jget<{ jobs: MLJob[] }>(`/api/quantising/jobs?project_id=${encodeURIComponent(projectId)}`);

export const getQuantiseJob = (jobId: string) =>
  jget<{ job: MLJob }>(`/api/quantising/jobs/${encodeURIComponent(jobId)}`);

export const cancelQuantiseJob = (jobId: string) =>
  jpost<{ ok: boolean; job: MLJob }>(`/api/quantising/jobs/${encodeURIComponent(jobId)}/cancel`);

export const getQuantiseLogs = (jobId: string, tail = 200) =>
  jget<{ status: MLJobStatus; progress: Record<string, unknown>; logs: string[] }>(
    `/api/quantising/jobs/${encodeURIComponent(jobId)}/logs?tail=${tail}`,
  );

// Returns a short-lived presigned URL; the caller opens it (the endpoint
// is bearer-gated, so we can't just point an <a href> at it).
export const getQuantiseArtifactUrl = (jobId: string, which: "float_onnx" | "int8_onnx") =>
  jget<{ url: string; which: string }>(
    `/api/quantising/jobs/${encodeURIComponent(jobId)}/artifact/${which}`,
  );

// ── analyse ───────────────────────────────────────────────────────────
export type AnalyseDetection = { label: string; score: number; box_xyxy: number[] };
export type AnalyseResult = {
  size: { width: number; height: number };
  classes: string[];
  source_job_id: string;
  quantised: boolean;
  reference: AnalyseDetection[];
  float: AnalyseDetection[];
  int8: AnalyseDetection[] | null;
  missed_by_float: AnalyseDetection[];
  missed_by_int8: AnalyseDetection[] | null;
};

// Upload one image; the backend runs the labelling pipeline + the trained
// (float) model + the quantised (int8) model and returns all detections + the
// labels each model missed. Multipart, so we let the browser set the boundary.
export const analyseImage = async (
  projectId: string, file: File, sourceJobId?: string,
): Promise<AnalyseResult> => {
  const fd = new FormData();
  fd.append("image", file);
  if (sourceJobId) fd.append("source_job_id", sourceJobId);
  const r = await apiFetch(`/api/training/projects/${encodeURIComponent(projectId)}/analyse`, {
    method: "POST", body: fd,
  });
  if (!r.ok) throw new Error(await errMessage(r));
  return r.json() as Promise<AnalyseResult>;
};

// ── client-side helpers ─────────────────────────────────────────────────

// Rough pre-start token estimate. Mirrors model_registry.estimate_training_
// seconds — clearly labelled an estimate in the UI; the authoritative value
// is also returned by createTrainingJob. Scales with epochs, the effective
// sample count (images + augmentations when the toggle is on), AND input
// size (compute ~ pixels → (imgsz/base)^2), so it moves when the user
// touches any of those controls.
export function estimateTrainingTokens(
  model: TrainingModel | null | undefined,
  config: Record<string, unknown>,
  nImages: number,
  nAugmentations: number,
): { seconds: number; blocks: number } {
  const epochs = Number(config?.epochs ?? 0) || 0;
  if (!model || epochs <= 0) return { seconds: 0, blocks: 0 };
  let samples = Math.max(1, nImages);
  if (config?.use_augmentations) samples += Math.max(0, nAugmentations || 0);
  const base = model.est_base_imgsz || 320;
  const imgsz = Number(config?.imgsz ?? base) || base;
  const sizeScale = base > 0 ? Math.pow(imgsz / base, 2) : 1;
  const perEpoch = Math.max(
    3,
    model.est_seconds_per_epoch_per_100img * (samples / 100) * sizeScale,
  );
  const seconds = Math.round(epochs * perEpoch);
  const blocks = seconds > 0 ? Math.ceil(seconds / BLOCK_SECONDS) : 0;
  return { seconds, blocks };
}

export function fmtDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
