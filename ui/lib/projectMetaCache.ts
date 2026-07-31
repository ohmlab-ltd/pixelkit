// Persistent cache of small, render-critical project metadata so
// opening a project from the workspace card paints with labels +
// reference count + dataset type already populated, instead of
// flashing empty placeholders for ~200-500 ms while
// /api/projects/{id} and /api/v2/projects/{id}/dataset-type round-
// trip. The workspace list endpoint already returns most of these
// fields; the dataset-type fetch happens once per session and is
// memoised here so subsequent opens of the same project skip the
// extra call entirely.
//
// Cache is global (not per-username), project meta isn't sensitive
// (every field is also visible on the public workspace card), and a
// single global key is simpler for ProjectViewV2Stub to read without
// having to thread a username prop through the V2 mount path.

const KEY = "pixelkit_project_meta_v1";

export type ProjectMetaEntry = {
  name?: string;
  // Project owner's username. Cached so /app/<id> deep-links can
  // resolve readOnly state on the first paint instead of waiting on
  // the /api/projects/{id} manifest fetch, without this the public
  // read-only project view briefly shows the owner chrome to its own
  // viewer (and conversely, an owner's project briefly renders in
  // read-only mode for them on refresh).
  owner?: string;
  labels?: string[];
  // Display-rename map (canonical_lower → renamed). Cached so the
  // workspace + public cards can render a label rename instantly
  //, without it, cards waited for the next /api/projects poll
  // (~4s) before the new name appeared.
  labelAliases?: Record<string, string>;
  // Per-label colour overrides ({canonical_lower: "#rrggbb"}). Cached
  // so the next paint shows the user's colour choices instantly,
  // matching the labelAliases pattern above.
  labelColours?: Record<string, string>;
  // Privacy flag, cached so the padlock badge can paint on the
  // first frame of the project page (workspace already knows whether
  // the card is private; the project view would otherwise wait on
  // /api/projects/{id} and pop the padlock in late).
  private?: boolean;
  v2?: boolean;
  nImages?: number;
  nReferences?: number;
  datasetType?: { type: "general" | "specific"; reason?: string | null; source?: string | null };
  cachedAt?: number;
  // Per-tile blurhash placeholders cached so the next project-open
  // paints gradient tiles instantly, the network round-trip to
  // /api/projects/{id} no longer gates the placeholder grid. Sized
  // small (just id + hash strings) so the localStorage 5 MB budget
  // is plenty for hundreds of projects.
  refTiles?: { id: string; filename: string; blurhash?: string | null; width?: number; height?: number }[];
  importTiles?: { id: string; filename: string; blurhash?: string | null; width?: number; height?: number; createdAt?: number | string | null }[];
};

export type ProjectMetaCache = Record<string, ProjectMetaEntry>;

export function readProjectMetaCache(): ProjectMetaCache {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as ProjectMetaCache) : {};
  } catch {
    return {};
  }
}

export function writeProjectMetaCache(next: ProjectMetaCache): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* quota exceeded, keep working in-memory */
  }
}

export function readProjectMeta(projectId: string): ProjectMetaEntry | null {
  const all = readProjectMetaCache();
  return all[projectId] ?? null;
}

export function patchProjectMeta(
  projectId: string,
  patch: Partial<ProjectMetaEntry>,
): void {
  if (typeof window === "undefined") return;
  const all = readProjectMetaCache();
  all[projectId] = { ...(all[projectId] ?? {}), ...patch, cachedAt: Date.now() };
  writeProjectMetaCache(all);
}
