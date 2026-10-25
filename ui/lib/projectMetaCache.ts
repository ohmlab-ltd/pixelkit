// localStorage cache of small project metadata, so an opened project paints before its fetches return.

const KEY = "pixelkit_project_meta_v1";

type ProjectMetaEntry = {
  name?: string;
  labels?: string[];
  // canonical_lower -> display name
  labelAliases?: Record<string, string>;
  // canonical_lower -> "#rrggbb"
  labelColours?: Record<string, string>;
  nImages?: number;
  nReferences?: number;
  datasetType?: { type: "general" | "specific"; reason?: string | null; source?: string | null };
  cachedAt?: number;
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
    /* quota exceeded */
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
