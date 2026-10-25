// Client for /api/containers. A container is what the UI calls a Project;
// /api/projects/{id} is a single dataset.

import { API_BASE, apiFetch } from "./apiFetch";

export interface Container {
  id: string;
  name: string;
  cover: string | null;
  /** Average luminance, 0-255, of the cover's bottom band. */
  coverLuma?: number;
  /** Changes only when the cover image changes; busts the cover cache. */
  cover_updated?: string;
  dataset_ids: string[];
  created: string;
  updated: string;
}

export interface ContainerCard {
  id: string;
  name: string;
  cover: string | null;
  cover_updated?: string;
  n_datasets: number;
  updated: string;
  created: string;
}

export interface ContainerDataset {
  id: string;
  name: string;
  cover: string | null;
  n_images: number;
  hasModel: boolean;
  updated?: string;
  /** True for a cropped child of another dataset. */
  derived?: boolean;
}

export interface ContainerDetail extends Container {
  datasets: ContainerDataset[];
}

export interface ActivityItem {
  ts: string;
  kind: string;
  job_kind?: string | null;
  container?: string | null;
  dataset?: string | null;
  status?: string | null;
  n_images?: number | null;
  name?: string | null;
}

const JSON_HEADERS = { "Content-Type": "application/json" } as const;

export function containerCoverUrl(id: string, updated?: string): string {
  const v = updated ? `?v=${encodeURIComponent(updated)}` : "";
  return `${API_BASE}/api/containers/${encodeURIComponent(id)}/cover${v}`;
}

export function datasetCoverUrl(datasetId: string, updated?: string): string {
  const v = updated ? `?v=${encodeURIComponent(updated)}` : "";
  return `${API_BASE}/api/projects/${encodeURIComponent(datasetId)}/cover_thumb${v}`;
}

/** Project cards, or null when the request failed; callers keep their last list on null. */
export async function listContainers(): Promise<ContainerCard[] | null> {
  try {
    const r = await apiFetch("/api/containers");
    if (!r.ok) return null;
    return ((await r.json()).containers as ContainerCard[]) ?? [];
  } catch {
    return null;
  }
}

export async function getContainer(id: string): Promise<ContainerDetail | null> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}`);
  if (!r.ok) return null;
  return (await r.json()) as ContainerDetail;
}

export async function createContainer(name: string): Promise<Container | null> {
  const r = await apiFetch("/api/containers", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name }),
  });
  if (!r.ok) return null;
  return (await r.json()) as Container;
}

export async function patchContainer(
  id: string,
  patch: { name?: string; cover?: string | null },
): Promise<Container | null> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify(patch),
  });
  if (!r.ok) return null;
  return (await r.json()) as Container;
}

export async function deleteContainer(id: string): Promise<boolean> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}`, { method: "DELETE" });
  return r.ok;
}

export async function addDataset(id: string, datasetId: string): Promise<boolean> {
  const r = await apiFetch(
    `/api/containers/${encodeURIComponent(id)}/datasets/${encodeURIComponent(datasetId)}`,
    { method: "POST" },
  );
  return r.ok;
}

export async function removeDataset(id: string, datasetId: string): Promise<boolean> {
  const r = await apiFetch(
    `/api/containers/${encodeURIComponent(id)}/datasets/${encodeURIComponent(datasetId)}`,
    { method: "DELETE" },
  );
  return r.ok;
}

/** Deletes the dataset itself, not only its link to the Project. */
export async function deleteDataset(datasetId: string): Promise<boolean> {
  const r = await apiFetch(`/api/projects/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
  return r.ok;
}

/** Returns the stored cover ref, or null on failure. */
export async function uploadCover(id: string, file: File): Promise<string | null> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}/cover`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) return null;
  return ((await r.json()).cover as string) ?? null;
}

export async function getActivity(id: string, limit = 100): Promise<ActivityItem[]> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}/activity?limit=${limit}`);
  if (!r.ok) return [];
  return ((await r.json()).activity as ActivityItem[]) ?? [];
}
