// Frontend client for the Project-container backend (gd/containers.py +
// /api/containers/*). A "Project" (user-facing) is a CONTAINER of datasets +
// members; the legacy code-level "project" is a single Dataset. All calls go
// through apiFetch so the backend's membership-aware guards see the bearer.

import { API_BASE as API, apiFetch } from "./apiFetch";

export type Role = "owner" | "editor" | "viewer";

export interface Member {
  username: string;
  role: Role;
}

export interface Container {
  id: string;
  name: string;
  owner: string;
  private: boolean;
  cover: string | null;
  /** Average luminance (0-255) of the cover's bottom band, so the page can
   *  pick a black/white title that stands out. Absent when there's no cover. */
  coverLuma?: number;
  /** Bumped only when the cover image itself changes (the R2 key is fixed, so
   *  this is how the page busts its cover cache without reloading the hero on
   *  every unrelated settings save). */
  cover_updated?: string;
  /** Max longest-edge (px) uploads are resized to; datasets inherit it. */
  max_input_size?: number;
  members: Member[];
  dataset_ids: string[];
  created: string;
  updated: string;
}

export interface ContainerCard {
  id: string;
  name: string;
  owner: string;
  private: boolean;
  cover: string | null;
  max_input_size?: number;
  n_datasets: number;
  n_members: number;
  updated: string;
  created: string;
}

export interface ContainerDataset {
  id: string;
  name: string;
  cover: string | null;
  private: boolean;
  n_images: number;
  hasModel: boolean;
  updated?: string;
  /** Creator handle. Drives delete permissions: only the creator can destroy a
   *  dataset; the Project owner can detach (remove) any from the Project. */
  owner?: string;
  /** True when this dataset is a derived (cropped child) of another dataset. */
  derived?: boolean;
}

export interface ContainerDetail extends Container {
  my_role: Role | null;
  datasets: ContainerDataset[];
}

export interface ActivityItem {
  ts: string;
  kind: string;
  job_kind?: string | null;
  actor?: string | null;
  container?: string | null;
  dataset?: string | null;
  status?: string | null;
  n_images?: number | null;
  member?: string | null;
  role?: Role | string | null;
  name?: string | null;
}

const JSON_HEADERS = { "Content-Type": "application/json" } as const;

/** Public URL for a Project cover (an <img src> — authenticates via the pk_auth
 *  cookie for private Projects). */
export function containerCoverUrl(id: string): string {
  return `${API}/api/containers/${encodeURIComponent(id)}/cover`;
}

/** Workspace-style cover thumbnail for a dataset (the same image the workspace
 *  card shows). Cache-busted by `updated` so a cover swap refreshes. */
export function datasetCoverUrl(datasetId: string, updated?: string): string {
  const v = updated ? `?v=${encodeURIComponent(updated)}` : "";
  return `${API}/api/projects/${encodeURIComponent(datasetId)}/cover_thumb${v}`;
}

export interface UserHit {
  username: string;
  image: string | null;
  name: string | null;
}

/** Username typeahead for the add-member field. Best-effort: [] on failure. */
export async function searchUsers(q: string): Promise<UserHit[]> {
  const query = (q || "").trim();
  if (!query) return [];
  try {
    const r = await fetch(`/api/users/search?q=${encodeURIComponent(query)}`);
    if (!r.ok) return [];
    return ((await r.json()).users as UserHit[]) ?? [];
  } catch {
    return [];
  }
}

/** Returns the user's Project cards, or `null` when the request FAILED (e.g.
 *  a 401 from an expired bearer, or a network/5xx blip). The null vs [] split
 *  matters: a failure must NOT be shown as "you have no projects" — callers
 *  keep their last-known row instead of clobbering it with an empty state. */
export async function listContainers(): Promise<ContainerCard[] | null> {
  try {
    const r = await apiFetch("/api/containers");
    if (!r.ok) return null;
    return ((await r.json()).containers as ContainerCard[]) ?? [];
  } catch {
    return null;
  }
}

// Public (non-private) Projects for the Community carousel. Plain fetch (no
// auth needed); returns [] on any failure so the carousel just hides.
export async function listPublicContainers(): Promise<ContainerCard[]> {
  try {
    const r = await fetch(`${API}/api/containers/public`, { cache: "no-store" });
    if (!r.ok) return [];
    return ((await r.json()).containers as ContainerCard[]) ?? [];
  } catch {
    return [];
  }
}

export async function getContainer(id: string): Promise<ContainerDetail | null> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}`);
  if (!r.ok) return null;
  return (await r.json()) as ContainerDetail;
}

export async function createContainer(name: string, isPrivate = true): Promise<Container | null> {
  const r = await apiFetch("/api/containers", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name, private: isPrivate }),
  });
  if (!r.ok) return null;
  return (await r.json()) as Container;
}

export async function patchContainer(
  id: string,
  patch: { name?: string; private?: boolean; cover?: string | null; max_input_size?: number },
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

export async function addMember(id: string, username: string, role: Role = "editor"): Promise<Container | null> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}/members`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ username, role }),
  });
  if (!r.ok) return null;
  return (await r.json()) as Container;
}

export async function removeMember(id: string, username: string): Promise<boolean> {
  const r = await apiFetch(
    `/api/containers/${encodeURIComponent(id)}/members/${encodeURIComponent(username)}`,
    { method: "DELETE" },
  );
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

/** Permanently delete a dataset (destroy — not just detach from the Project).
 *  The backend allows this only for the dataset's own creator. */
export async function deleteDataset(datasetId: string): Promise<boolean> {
  const r = await apiFetch(`/api/projects/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
  return r.ok;
}

/** Upload/replace the Project cover. Returns the stored cover ref or null. */
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

/** Resolve usernames -> real avatar URLs via our Next server (which can read
 *  the NextAuth user store). Same-origin, cookie-authed. Best-effort: returns
 *  {} on any failure, so callers fall back to monograms. */
export async function fetchAvatars(usernames: string[]): Promise<Record<string, string>> {
  const list = Array.from(new Set(usernames.map((u) => (u || "").trim()).filter(Boolean)));
  if (list.length === 0) return {};
  try {
    const r = await fetch("/api/users/avatars", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ usernames: list }),
    });
    if (!r.ok) return {};
    return ((await r.json()).avatars as Record<string, string>) ?? {};
  } catch {
    return {};
  }
}

export async function getActivity(id: string, limit = 100): Promise<ActivityItem[]> {
  const r = await apiFetch(`/api/containers/${encodeURIComponent(id)}/activity?limit=${limit}`);
  if (!r.ok) return [];
  return ((await r.json()).activity as ActivityItem[]) ?? [];
}

/** After adding a member, ask our Next server to email them (it has the user
 *  email store; the backend does not). Best-effort. */
export async function notifyMemberAdded(
  containerId: string,
  containerName: string,
  username: string,
  role: Role,
): Promise<void> {
  try {
    await fetch("/api/projects/notify-member", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ containerId, containerName, username, role }),
    });
  } catch {
    /* best-effort */
  }
}
