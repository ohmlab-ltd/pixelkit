// Client for the engine's model manager + settings (portable build).
import { apiFetch } from "@/lib/apiFetch";

export type ModelName = "sam3" | "dinov2";

export type DownloadRec = {
  status: "downloading" | "done" | "error";
  downloaded_bytes: number;
  total_bytes: number;
  error: string | null;
} | null;

export type ModelInfo = {
  repo: string;
  label: string;
  gated: boolean;
  required: boolean;
  approxGb: number;
  downloaded: boolean;
  loaded: boolean;
  download: DownloadRec;
};

export type ModelsStatus = {
  models: Record<ModelName, ModelInfo>;
  weightsDir: string;
  freeDiskGb: number;
  hfTokenConfigured: boolean;
};

export type TokenStatus = {
  configured: boolean;
  valid: boolean | null;
  username: string | null;
  sam3Access: boolean | null;
  detail: string | null;
};

export type DevicePreference = "auto" | "cuda" | "mps" | "cpu" | `cuda:${number}`;

export type GpuInfo = { index: number; name: string; vramGb: number };

export type EngineSettings = {
  workspace: string;
  device: string; // "cuda" | "cuda:<n>" | "mps" | "cpu"
  devicePreference: DevicePreference;
  deviceEnvOverride: string | null;
  gpuAvailable: boolean;
  gpus: GpuInfo[];
  hfTokenConfigured: boolean;
  sam3Repo: string;
};

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export const fetchModelsStatus = () =>
  apiFetch("/api/models/status").then((r) => json<ModelsStatus>(r));

export const fetchEngineSettings = () =>
  apiFetch("/api/settings").then((r) => json<EngineSettings>(r));

export const fetchTokenStatus = () =>
  apiFetch("/api/settings/hf-token").then((r) => json<TokenStatus>(r));

export const setHfToken = (token: string) =>
  apiFetch("/api/settings/hf-token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  }).then((r) => json<TokenStatus>(r));

export const clearHfToken = () =>
  apiFetch("/api/settings/hf-token", { method: "DELETE" }).then((r) => json<{ ok: boolean }>(r));

// NOTE: the per-model download/load/unload client calls are gone —
// models are engine-managed plumbing (auto-download + auto-load once
// the HF token exists); the UI only ever READS /api/models/status.

export const setWorkspacePath = (path: string) =>
  apiFetch("/api/settings/workspace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).then((r) => json<{ ok: boolean; workspace: string; restartRequired: boolean }>(r));

export const setDevicePreference = (device: DevicePreference) =>
  apiFetch("/api/settings/device", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device }),
  }).then((r) =>
    json<{ ok: boolean; devicePreference: DevicePreference; restartRequired: boolean }>(r),
  );

export function downloadPct(d: DownloadRec): number | null {
  if (!d || d.status !== "downloading" || !d.total_bytes) return null;
  return Math.min(99, Math.round((d.downloaded_bytes / d.total_bytes) * 100));
}
