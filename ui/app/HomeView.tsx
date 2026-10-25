"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ProjectTagsRow } from "./components/ProjectTagsRow";
import { BlurhashCanvas } from "react-blurhash";
import { LABEL_COLOURS } from "./v2/labelColours";
import type { ReferenceImage } from "./v2/ReferenceImageEditor";
import { WORD_CLOUD_EXIT_MS, WordCloud } from "./v2/WordCloud";
import { PixelKitLoader } from "./v2/PixelKitLoader";
import { ProjectsSection } from "./ProjectsSection";
import { ProjectPage } from "./ProjectPage";
import { CreateDatasetModal } from "./CreateDatasetModal";
import { API_BASE, apiFetch } from "@/lib/apiFetch";
import { onNewDatasetRequest, requestExplorerRefresh } from "@/lib/appNav";
import { addDataset } from "@/lib/containers";
import { splitLabels } from "@/lib/labelInput";
import { readProjectMetaCache, writeProjectMetaCache, type ProjectMetaCache } from "@/lib/projectMetaCache";
import {
  collectInput,
  parseDataset,
  uploadDataset,
  ZIP_SOFT_LIMIT_BYTES,
  type ParsedDataset,
} from "@/lib/datasetImport";

type ProjectSummary = {
  id: string;
  name: string;
  createdAt: string | null;
  updatedAt: string | null;
  n_images: number;
  n_labelled: number;
  n_unlabelled: number;
  tags: string[];
  thumbnail: string | null;
  hasModel: boolean;
  favouritedByMe: boolean;
  // Derived projects: cropped, one-label-per-image copies of a parent.
  derived?: { parentProjectId?: string; parentName?: string } | null;
  running?: boolean;
  n_references?: number;
  // BlurHash of the cover image, or null.
  cover_blurhash?: string | null;
  // Lowercase canonical label -> display label.
  label_aliases?: Record<string, string>;
  // Per-label colour overrides, lowercase canonical label -> #rrggbb.
  labelColours?: Record<string, string>;
  // The Project this dataset belongs to, or null.
  container?: { id: string; name: string } | null;
};


// Compares only the fields the cards render, to skip re-renders on a poll.
function projectsEqual(a: ProjectSummary[] | null, b: ProjectSummary[]): boolean {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (
      x.id !== y.id ||
      x.name !== y.name ||
      x.thumbnail !== y.thumbnail ||
      x.updatedAt !== y.updatedAt ||
      x.n_images !== y.n_images ||
      x.n_labelled !== y.n_labelled ||
      x.n_unlabelled !== y.n_unlabelled ||
      x.hasModel !== y.hasModel ||
      x.favouritedByMe !== y.favouritedByMe ||
      (x.container?.id ?? null) !== (y.container?.id ?? null) ||
      (x.container?.name ?? null) !== (y.container?.name ?? null) ||
      x.running !== y.running
    ) return false;
    const at = x.tags ?? [];
    const bt = y.tags ?? [];
    if (at.length !== bt.length) return false;
    for (let j = 0; j < at.length; j++) if (at[j] !== bt[j]) return false;
  }
  return true;
}


// Branch glyph for a derived project.
function DerivedBadge({ parentName }: { parentName?: string | null }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-[var(--fg-dim)] shrink-0"
      aria-label="Derived dataset"
      role="img"
    >
      <title>{parentName ? `Derived from ${parentName}` : "Derived dataset"}</title>
      <circle cx="4" cy="3.2" r="1.7" />
      <circle cx="4" cy="12.8" r="1.7" />
      <circle cx="12" cy="12.8" r="1.7" />
      <path d="M4 4.9V11.1M5.7 12.8H10.3" />
    </svg>
  );
}

// Sorting is client-side. The server handles only the search `q`.
type SortMode = "updated" | "created" | "name" | "images";
const SORT_OPTIONS: { key: SortMode; label: string }[] = [
  { key: "updated", label: "Recently updated" },
  { key: "created", label: "Recently created" },
  { key: "name", label: "Name" },
  { key: "images", label: "Most images" },
];

// Status chips for a project card.
function projectStatusBadges(
  project: ProjectSummary,
): { label: string; dot: string }[] {
  const nImages = project.n_images ?? 0;
  const nLabelled = project.n_labelled ?? 0;
  const nUnlabelled = project.n_unlabelled ?? 0;
  const allLabelled = nImages > 0 && nUnlabelled === 0;
  const someLabelled = nLabelled > 0 && nUnlabelled > 0;
  const allUnlabelled = nImages > 0 && !nLabelled;
  const badges: { label: string; dot: string }[] = [];
  if (project.running) {
    badges.push({ label: "In progress", dot: "bg-[var(--accent)]" });
  } else if (allLabelled) {
    badges.push({ label: "Labelled", dot: "bg-[var(--ok)]" });
  } else if (someLabelled) {
    badges.push({ label: "Partial", dot: "bg-[var(--warn)]" });
  } else if (allUnlabelled) {
    badges.push({ label: "Unlabelled", dot: "bg-[var(--bad)]" });
  }
  if (project.hasModel) {
    badges.push({ label: "Model", dot: "bg-[var(--fg-dim)]" });
  }
  return badges;
}

function SortMenu({
  value,
  onChange,
}: {
  value: SortMode;
  onChange: (m: SortMode) => void;
}) {
  const [open, setOpen] = useState(false);
  const active = SORT_OPTIONS.find((o) => o.key === value) ?? SORT_OPTIONS[0];
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="inline-flex h-9 items-center gap-1.5 rounded-md border border-[var(--line)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] px-3 text-[13px] text-foreground/75 transition-colors"
      >
        {active.label}
        <svg
          viewBox="0 0 24 24"
          className="h-3.5 w-3.5 text-foreground/40"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} aria-hidden />
          <div
            role="menu"
            className="pk-glass absolute right-0 top-[calc(100%+8px)] z-50 w-48 rounded-md p-1.5 shadow-[var(--shadow-strong)]"
          >
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                role="menuitem"
                onClick={() => {
                  setOpen(false);
                  onChange(opt.key);
                }}
                className={[
                  "flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm transition-colors",
                  opt.key === value
                    ? "bg-foreground/[0.08] text-[var(--foreground)]"
                    : "text-foreground/75 hover:bg-foreground/[0.05]",
                ].join(" ")}
              >
                {opt.label}
                {opt.key === value && (
                  <svg
                    viewBox="0 0 24 24"
                    className="h-3.5 w-3.5 text-[var(--accent)]"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden
                  >
                    <path d="M20 6L9 17l-5-5" />
                  </svg>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </span>
  );
}

export function HomeView({
  onOpen,
  onV2Begin,
}: {
  onOpen: (id: string, displayName?: string, fromProjectId?: string) => void;
  /** Fires when onboarding finishes. */
  onV2Begin: (
    name: string,
    labels: string[],
    references: ReferenceImage[],
    projectId: string | null,
    /** "onboarding": straight from the create flow, so the project view skips its own loader. */
    firstLoad?: "onboarding" | null,
  ) => void;
}) {
  // Null on the first render, as in the static prerender; then the
  // localStorage copy until refresh() returns.
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  // Infinite scroll. `total` is the server's project count, for placeholders.
  const WORKSPACE_PAGE_SIZE = 12;
  const [total, setTotal] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  // projects.length for the 4 s poll, whose closure would see a stale value.
  const projectsLenRef = useRef<number>(0);
  // Deleted projects, hidden from refreshes until the server stops listing them.
  const pendingDeleteIdsRef = useRef<Set<string>>(new Set());
  // Favourite toggles in flight, kept over a poll that lands before the POST returns.
  const pendingFavouriteIdsRef = useRef<Map<string, boolean>>(new Map());
  const [searchQuery, setSearchQuery] = useState("");
  // Debounced search query. The ref is read by refresh(), which a setInterval also runs.
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const queryRef = useRef("");
  // Client-side filter and sort.
  const [tab, setTab] = useState<"all" | "starred">("all");
  const [sortMode, setSortMode] = useState<SortMode>("updated");
  const [error, setError] = useState<string | null>(null);
  // Set: the Project view shows in place of the workspace.
  const [projectViewId, setProjectViewId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectSummary | null>(null);
  // Dataset being duplicated. Its Duplicate action is disabled meanwhile.
  const [duplicatingId, setDuplicatingId] = useState<string | null>(null);

  // ?project=<id> opens that Project.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const fromUrl = () => new URLSearchParams(window.location.search).get("project");
    const initial = fromUrl();
    if (initial) setProjectViewId(initial);
    const onPop = () => setProjectViewId(fromUrl() || null);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // Keeps ?project= in step with the open Project.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const current = params.get("project");
    if (projectViewId) {
      if (current !== projectViewId) {
        params.set("project", projectViewId);
        window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
      }
    } else if (current) {
      params.delete("project");
      const qs = params.toString();
      window.history.replaceState(null, "", qs ? `${window.location.pathname}?${qs}` : window.location.pathname);
    }
  }, [projectViewId]);

  // Onboarding. "creating" shows a loader while the server creates the dataset.
  type V2Stage = "idle" | "name" | "labels" | "creating" | "import";
  const [v2Stage, setV2Stage] = useState<V2Stage>("idle");
  const [v2Name, setV2Name] = useState("");
  const [v2Labels, setV2Labels] = useState<string[]>([]);
  // Random colour per label, saved with the project by v2EnsureProject.
  const [v2LabelColours, setV2LabelColours] = useState<Record<string, string>>({});
  const [v2Input, setV2Input] = useState("");
  // Set once v2EnsureProject has created the project.
  const [v2ProjectId, setV2ProjectId] = useState<string | null>(null);
  const v2ProjectIdRef = useRef<string | null>(null);
  // Import of a labelled dataset. Its classes replace the typed labels.
  const [v2ImportParsing, setV2ImportParsing] = useState(false);
  const [v2ImportParseProgress, setV2ImportParseProgress] = useState<{ done: number; total: number } | null>(null);
  const [v2ImportParsed, setV2ImportParsed] = useState<ParsedDataset | null>(null);
  const [v2ImportBusy, setV2ImportBusy] = useState(false);
  const [v2ImportProgress, setV2ImportProgress] = useState<{ done: number; total: number } | null>(null);
  // v2ImportBusy for the v2Reset and Esc closures, which see a stale value.
  const v2ImportBusyRef = useRef(false);
  // Set while v2HandOff creates the dataset and opens it.
  const v2HandOffBusyRef = useRef(false);
  // Project that the new dataset joins, when onboarding started from a Project page.
  const pendingContainerRef = useRef<string | null>(null);
  // State copy of pendingContainerRef, for the labels stage's Back button.
  const [v2ReturnProjectId, setV2ReturnProjectId] = useState<string | null>(null);
  // Workspace "+ Add Dataset" asks for the name in CreateDatasetModal.
  const [showWorkspaceCreate, setShowWorkspaceCreate] = useState(false);
  useEffect(() => { v2ProjectIdRef.current = v2ProjectId; }, [v2ProjectId]);

  // The desktop Explorer's "+" acts as "+ Add Dataset". Ignored during onboarding.
  useEffect(() => {
    return onNewDatasetRequest(() => {
      if (v2Stage !== "idle") return;
      pendingContainerRef.current = null;
      setV2ReturnProjectId(null);
      setProjectViewId(null);
      setShowWorkspaceCreate(true);
    });
  }, [v2Stage]);

  const v2InputRef = useRef<HTMLInputElement | null>(null);

  // Stages that replace the projects grid.
  const v2Active = v2Stage === "labels" || v2Stage === "creating" || v2Stage === "import";

  // The word cloud stays mounted after the labels stage for its fade-out.
  const v2WordCloudShouldShow = v2Stage === "labels";
  const [v2WordCloudMounted, setV2WordCloudMounted] = useState(false);
  useEffect(() => {
    if (v2WordCloudShouldShow) {
      setV2WordCloudMounted(true);
      return;
    }
    if (!v2WordCloudMounted) return;
    // Once the last word has faded out.
    const t = window.setTimeout(() => setV2WordCloudMounted(false), WORD_CLOUD_EXIT_MS + 100);
    return () => window.clearTimeout(t);
  }, [v2WordCloudShouldShow, v2WordCloudMounted]);

  // Today's date, set as onboarding starts: the project is not created until
  // it finishes, and the first render must match the static prerender.
  const [v2DateLabel, setV2DateLabel] = useState("");
  useEffect(() => {
    if (!v2Active) return;
    setV2DateLabel(
      new Intl.DateTimeFormat("en-GB", {
        day: "numeric",
        month: "long",
        year: "numeric",
      }).format(new Date()),
    );
  }, [v2Active]);

  // Esc cancels onboarding.
  useEffect(() => {
    if (v2Stage === "idle") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") v2Reset();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Stage]);

  // Ctrl/Cmd+Enter triggers Done on the labels stage.
  useEffect(() => {
    if (v2Stage !== "labels") return;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
      e.preventDefault();
      if (v2Labels.length > 0) v2DoneLabels();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Stage, v2Labels.length]);

  useEffect(() => {
    if (v2Stage !== "labels") return;
    const t = window.setTimeout(() => v2InputRef.current?.focus(), 320);
    return () => window.clearTimeout(t);
  }, [v2Stage]);

  const v2Reset = () => {
    // No cancel during an import upload, which would delete its project, or
    // during a hand-off, which would open the dataset anyway.
    if (v2ImportBusyRef.current || v2HandOffBusyRef.current) return;
    pendingContainerRef.current = null;
    setV2ReturnProjectId(null);
    // A cancelled onboarding deletes its project.
    const cancelledProjectId = v2ProjectIdRef.current;
    setV2Stage("idle");
    setV2Name("");
    setV2Labels([]);
    setV2LabelColours({});
    setV2Input("");
    setV2ProjectId(null);
    v2ProjectIdRef.current = null;
    setV2ImportParsed(null);
    setV2ImportParsing(false);
    setV2ImportParseProgress(null);
    setV2ImportProgress(null);
    if (cancelledProjectId) {
      // Fire and forget. A failed delete leaves the project listed.
      apiFetch(`/api/projects/${cancelledProjectId}`, { method: "DELETE" })
        .catch((e) => console.warn("[v2 cancel] project delete failed:", e));
    }
  };

  const v2BeginLabels = (nameArg?: string) => {
    const nm = (nameArg ?? v2Name).trim();
    if (!nm) return;
    setError(null);
    // A name passed in goes into state, where v2HandOff reads it.
    if (nameArg !== undefined) setV2Name(nm);
    setV2Stage("labels");
  };

  // Pasted text arrives without key events, so commas and newlines split it here.
  const v2AddLabel = (raw: string) => {
    const parts = splitLabels(raw);
    if (parts.length === 0) return;
    const labels = [...v2Labels];
    const colours = { ...v2LabelColours };
    for (const next of parts) {
      if (labels.some((l) => l.toLowerCase() === next.toLowerCase())) continue;
      // A random unused colour, or any colour once all are used.
      const taken = new Set(Object.values(colours));
      const free = LABEL_COLOURS.filter((c) => !taken.has(c));
      const pool = free.length > 0 ? free : LABEL_COLOURS;
      colours[next.toLowerCase()] = pool[Math.floor(Math.random() * pool.length)];
      labels.push(next);
    }
    setError(null);
    setV2Labels(labels);
    setV2LabelColours(colours);
    setV2Input("");
  };

  const v2RemoveLabel = (idx: number) => {
    const list = v2Labels.slice();
    const removed = list[idx];
    list.splice(idx, 1);
    setV2Labels(list);
    if (removed) {
      setV2LabelColours((m) => {
        const next = { ...m };
        delete next[removed.toLowerCase()];
        return next;
      });
    }
  };

  // Creates the project once per onboarding. Returns its id, or null on failure.
  const v2EnsureProject = async (
    name: string,
    labels: string[],
  ): Promise<string | null> => {
    if (v2ProjectIdRef.current) return v2ProjectIdRef.current;
    try {
      const fd = new FormData();
      fd.append("name", name.trim() || "Untitled");
      fd.append("labels", JSON.stringify(labels));
      fd.append("label_colours", JSON.stringify(v2LabelColours));
      const r = await apiFetch(`/api/v2/projects`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = (await r.json()) as { project_id: string };
      const pid = data.project_id;
      setV2ProjectId(pid);
      v2ProjectIdRef.current = pid;
      // The Explorer otherwise polls every 10 s.
      requestExplorerRefresh();
      return pid;
    } catch (e) {
      console.error("[v2 ensure-project]", e);
      return null;
    }
  };

  // Import a labelled dataset
  const v2BeginImport = () => {
    setError(null);
    setV2ImportParsed(null);
    setV2ImportParseProgress(null);
    setV2ImportProgress(null);
    setV2Stage("import");
  };

  // Parses a picked folder or zip for preview, without uploading.
  const v2PickDataset = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setError(null);
    setV2ImportParsed(null);
    setV2ImportParsing(true);
    setV2ImportParseProgress(null);
    try {
      const arr = Array.from(files);
      const zip = arr.length === 1 && /\.zip$/i.test(arr[0].name) ? arr[0] : null;
      if (zip && zip.size > ZIP_SOFT_LIMIT_BYTES) {
        throw new Error(
          "That .zip is very large. Unzipping it here may run out of memory, so use Choose folder for big datasets.",
        );
      }
      const collected = await collectInput(arr);
      const parsed = await parseDataset(collected, (done, total) =>
        setV2ImportParseProgress({ done, total }),
      );
      setV2ImportParsed(parsed);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't read that dataset.");
    } finally {
      setV2ImportParsing(false);
      setV2ImportParseProgress(null);
    }
  };

  // Creates the project, uploads the images and boxes in batches, then opens it.
  const v2RunImport = async () => {
    const parsed = v2ImportParsed;
    if (!parsed || parsed.items.length === 0 || v2ImportBusyRef.current) return;
    setError(null);
    v2ImportBusyRef.current = true;
    setV2ImportBusy(true);
    setV2ImportProgress({ done: 0, total: parsed.items.length });
    try {
      const projectId = await v2EnsureProject(v2Name, parsed.classes);
      if (!projectId) throw new Error("Couldn't create the dataset.");
      const result = await uploadDataset(
        parsed.items,
        (form) =>
          apiFetch(`/api/v2/projects/${projectId}/imports/raw_batch`, { method: "POST", body: form }),
        (p) => setV2ImportProgress({ done: p.done, total: p.total }),
      );
      if (result.done === 0) {
        throw new Error(
          result.errors[0] ? `Import failed: ${result.errors[0]}` : "No images were imported.",
        );
      }
      if (result.failed > 0) {
        // A partial import still opens the project.
        const first = result.errors.slice(0, 2).join("; ");
        window.alert(
          `${result.failed} of ${result.total} image(s) could not be imported${first ? `: ${first}` : "."}`,
        );
      }
      onV2Begin(v2Name.trim(), parsed.classes, [], projectId, null);
      v2ImportBusyRef.current = false;
      setV2ImportBusy(false);
      setV2Stage("idle");
      setV2Name("");
      setV2Labels([]);
      setV2LabelColours({});
      setV2Input("");
      setV2ProjectId(null);
      v2ProjectIdRef.current = null;
      setV2ImportParsed(null);
      setV2ImportProgress(null);
      } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed.");
      // Deletes the half-filled project so a retry starts fresh. The parsed dataset is kept.
      const orphan = v2ProjectIdRef.current;
      if (orphan) {
        apiFetch(`/api/projects/${orphan}`, { method: "DELETE" })
          .catch((err) => console.warn("[v2 import] orphan project delete failed:", err));
      }
      setV2ProjectId(null);
      v2ProjectIdRef.current = null;
      v2ImportBusyRef.current = false;
      setV2ImportBusy(false);
      setV2ImportProgress(null);
    }
  };

  // Creates the project if needed and hands it to the parent.
  const v2HandOff = async (
    name: string,
    labels: string[],
    firstLoad: "onboarding" | null = null,
  ) => {
    // One at a time: a second Skip would create a second dataset.
    if (v2HandOffBusyRef.current) return;
    v2HandOffBusyRef.current = true;
    try {
      let projectId = v2ProjectIdRef.current;
      if (!projectId) projectId = await v2EnsureProject(name, labels);
      if (projectId && pendingContainerRef.current) {
        try {
          await addDataset(pendingContainerRef.current, projectId);
          // The dataset moves under its Project node in the tree.
          requestExplorerRefresh();
        } catch (e) {
          console.error("[v2 handoff] add-to-container failed:", e);
        }
        pendingContainerRef.current = null;
      }
      onV2Begin(name.trim(), labels, [], projectId, firstLoad);
      setV2Stage("idle");
      setV2Name("");
      setV2Labels([]);
      setV2LabelColours({});
      setV2Input("");
      setV2ProjectId(null);
      v2ProjectIdRef.current = null;
    } finally {
      v2HandOffBusyRef.current = false;
    }
  };

  // Labels confirmed: create the dataset and open it.
  const v2DoneLabels = () => {
    if (v2Labels.length === 0) { v2HandOff(v2Name, []); return; }
    setV2Stage("creating");
    void v2HandOff(v2Name, v2Labels, "onboarding");
  };

  const v2SkipFromLabels = () => v2HandOff(v2Name, []);

  // Applies the project view's meta-changed and deleted events before the next poll.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        projectId?: string;
        tags?: string[];
        label_aliases?: Record<string, string>;
        labelColours?: Record<string, string>;
        name?: string;
        cover?: string | null;
      } | null | undefined;
      if (!detail?.projectId) return;
      setProjects((prev) =>
        prev?.map((p) =>
          p.id === detail.projectId
            ? {
                ...p,
                tags: detail.tags ?? p.tags,
                label_aliases: detail.label_aliases ?? p.label_aliases,
                labelColours: detail.labelColours ?? p.labelColours,
                name: detail.name ?? p.name,
              }
            : p,
        ) ?? null,
      );
    };
    const deletedHandler = (e: Event) => {
      const detail = (e as CustomEvent).detail as { projectId?: string } | null | undefined;
      if (!detail?.projectId) return;
      setProjects((prev) => prev?.filter((p) => p.id !== detail.projectId) ?? null);
    };
    window.addEventListener("pixelkit-project-meta-changed", handler);
    window.addEventListener("pixelkit-project-deleted", deletedHandler);
    return () => {
      window.removeEventListener("pixelkit-project-meta-changed", handler);
      window.removeEventListener("pixelkit-project-deleted", deletedHandler);
    };
  }, []);

  // Favourites are stored in the manifest's `favouritedBy` list.
  const toggleFavourite = async (project_id: string) => {
    let optimistic = false;
    setProjects((prev) => {
      if (!prev) return prev;
      return prev.map((p) => {
        if (p.id !== project_id) return p;
        optimistic = !p.favouritedByMe;
        return { ...p, favouritedByMe: optimistic };
      });
    });
    pendingFavouriteIdsRef.current.set(project_id, optimistic);
    try {
      const r = await apiFetch(`/api/projects/${project_id}/favourite`, { method: "POST" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data: { favouritedByMe: boolean } = await r.json();
      setProjects((prev) =>
        prev?.map((p) =>
          p.id === project_id ? { ...p, favouritedByMe: data.favouritedByMe } : p,
        ) ?? null,
      );
    } catch {
      // The refresh reverts the optimistic toggle.
      refresh();
    } finally {
      pendingFavouriteIdsRef.current.delete(project_id);
    }
  };

  const refresh = async () => {
    try {
      // Refetches every loaded project, and at least one page.
      const currentCount = projectsLenRef.current;
      const limit = Math.max(WORKSPACE_PAGE_SIZE, currentCount);
      const qParam = queryRef.current ? `&q=${encodeURIComponent(queryRef.current)}` : "";
      const url = `${API_BASE}/api/projects?offset=0&limit=${limit}${qParam}`;
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const payload = await r.json() as
        | ProjectSummary[]
        | { total: number; items: ProjectSummary[]; offset?: number; limit?: number };
      // The response is {total, items} or a bare array.
      const data: ProjectSummary[] = Array.isArray(payload)
        ? payload
        : (payload.items ?? []);
      const totalCount: number = Array.isArray(payload)
        ? data.length
        : (payload.total ?? data.length);
      setTotal(totalCount);
      let visible = data;
      if (pendingDeleteIdsRef.current.size > 0) {
        const pending = pendingDeleteIdsRef.current;
        const stillListed = new Set(data.map((p) => p.id));
        for (const id of Array.from(pending)) {
          if (!stillListed.has(id)) pending.delete(id);
        }
        if (pending.size > 0) {
          visible = data.filter((p) => !pending.has(p.id));
        }
      }
      if (pendingFavouriteIdsRef.current.size > 0) {
        const map = pendingFavouriteIdsRef.current;
        visible = visible.map((p) =>
          map.has(p.id)
            ? { ...p, favouritedByMe: map.get(p.id)! }
            : p,
        );
      }
      setProjects((prev) => (projectsEqual(prev, visible) ? prev : visible));
      // Search results are a subset, so only an unfiltered list is cached.
      const searching = queryRef.current !== "";
      if (!searching) {
        try {
          window.localStorage.setItem("pixelkit_workspace_projects_v1", JSON.stringify(data));
        } catch {
          /* Quota exceeded. */
        }
        const metaCache: ProjectMetaCache = readProjectMetaCache();
        const seen = new Set<string>();
        for (const p of data) {
          seen.add(p.id);
          const prevEntry = metaCache[p.id] ?? {};
          metaCache[p.id] = {
            ...prevEntry,
            name: p.name,
            labels: p.tags ?? [],
            nImages: p.n_images,
            nReferences: p.n_references ?? 0,
            labelAliases: p.label_aliases ?? prevEntry.labelAliases,
            labelColours: p.labelColours ?? prevEntry.labelColours,
            cachedAt: Date.now(),
          };
        }
        for (const id of Object.keys(metaCache)) {
          if (!seen.has(id)) delete metaCache[id];
        }
        writeProjectMetaCache(metaCache);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const t = window.setTimeout(() => {
      setDebouncedQuery(searchQuery.trim());
    }, 200);
    return () => window.clearTimeout(t);
  }, [searchQuery]);

  // Refetches on a new query. Skipped on mount, which keeps the cached list.
  const queryEffectRanRef = useRef(false);
  useEffect(() => {
    queryRef.current = debouncedQuery;
    if (!queryEffectRanRef.current) {
      queryEffectRanRef.current = true;
      return;
    }
    setProjects(null);
    setTotal(null);
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQuery]);

  // Appends the next workspace page.
  const loadMoreWorkspacePage = useCallback(async (
    currentLen: number,
    currentTotal: number,
  ) => {
    if (loadingMore) return;
    setLoadingMore(true);
    try {
      const offset = currentLen;
      const qParam = queryRef.current ? `&q=${encodeURIComponent(queryRef.current)}` : "";
      const url = `${API_BASE}/api/projects?offset=${offset}&limit=${WORKSPACE_PAGE_SIZE}${qParam}`;
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const payload = await r.json() as
        | ProjectSummary[]
        | { total: number; items: ProjectSummary[]; offset?: number; limit?: number };
      const items: ProjectSummary[] = Array.isArray(payload)
        ? payload
        : (payload.items ?? []);
      const totalCount: number = Array.isArray(payload)
        ? items.length
        : (payload.total ?? currentTotal);
      setProjects((prev) => {
        const seen = new Set((prev ?? []).map((p) => p.id));
        const fresh = items.filter((p) => !seen.has(p.id));
        return [...(prev ?? []), ...fresh];
      });
      setTotal(totalCount);
    } catch (e) {
      console.warn("[workspace] paginate failed:", e);
    } finally {
      setLoadingMore(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadingMore]);

  useEffect(() => {
    if (!projects || total === null) return;
    if (projects.length >= total) return;
    const node = sentinelRef.current;
    if (!node) return;
    const obs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        void loadMoreWorkspacePage(projects.length, total);
      }
    }, { rootMargin: "600px" });
    obs.observe(node);
    return () => obs.disconnect();
  }, [projects, total, loadingMore, loadMoreWorkspacePage]);

  // Scroll fallback: after a back-navigation the observer can be left on a
  // detached sentinel node.
  useEffect(() => {
    if (!projects || total === null) return;
    if (projects.length >= total) return;
    const onScroll = () => {
      if (loadingMore) return;
      const scrolled = window.scrollY + window.innerHeight;
      const total_h = document.documentElement.scrollHeight;
      if (total_h - scrolled < 600) {
        void loadMoreWorkspacePage(projects.length, total);
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [projects, total, loadingMore, loadMoreWorkspacePage]);

  useEffect(() => {
    projectsLenRef.current = projects?.length ?? 0;
  }, [projects]);

  // Polls so the status badges follow job state.
  useEffect(() => {
    const id = window.setInterval(() => {
      refresh();
    }, 4000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sortedProjects = useMemo(() => {
    if (!projects) return null;
    // Search is server-side. Favourites come first in the default sort.
    const base =
      tab === "starred" ? projects.filter((p) => p.favouritedByMe) : projects;
    const byUpdated = (a: ProjectSummary, b: ProjectSummary) =>
      (b.updatedAt ?? b.createdAt ?? "").localeCompare(a.updatedAt ?? a.createdAt ?? "");
    return [...base].sort((a, b) => {
      switch (sortMode) {
        case "created":
          return (b.createdAt ?? "").localeCompare(a.createdAt ?? "");
        case "name":
          return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
        case "images":
          return (b.n_images ?? 0) - (a.n_images ?? 0);
        case "updated":
        default: {
          const fa = a.favouritedByMe ? 1 : 0;
          const fb = b.favouritedByMe ? 1 : 0;
          if (fa !== fb) return fb - fa;
          return byUpdated(a, b);
        }
      }
    });
  }, [projects, tab, sortMode]);

  // Counts loaded projects only.
  const starredCount = useMemo(
    () => (projects ?? []).filter((p) => p.favouritedByMe).length,
    [projects],
  );

  // On a cold load cards mount one per 70 ms, spreading the cover image
  // requests over the browser's connection limit. A cached list mounts at once.
  const [visibleCount, setVisibleCount] = useState(0);
  useEffect(() => {
    let cached: unknown = null;
    try {
      cached = JSON.parse(window.localStorage.getItem("pixelkit_workspace_projects_v1") ?? "null");
    } catch {
      /* no usable copy */
    }
    if (!Array.isArray(cached)) return;
    const list = cached as ProjectSummary[];
    setProjects((cur) => cur ?? list);
    setVisibleCount((v) => Math.max(v, list.length));
  }, []);
  const projectCount = sortedProjects?.length ?? 0;
  useEffect(() => {
    if (visibleCount >= projectCount) return;
    const t = window.setTimeout(() => {
      setVisibleCount((v) => Math.min(v + 1, projectCount));
    }, 70);
    return () => window.clearTimeout(t);
  }, [visibleCount, projectCount]);
  // Search results render at once.
  const isSearching = searchQuery.trim().length > 0;
  const renderCount = isSearching ? projectCount : visibleCount;

  const confirmDelete = async (project_id: string) => {
    // A delete already in flight.
    if (pendingDeleteIdsRef.current.has(project_id)) {
      setDeleteTarget(null);
      return;
    }
    // Removed from the list at once, and restored if the delete fails.
    pendingDeleteIdsRef.current.add(project_id);
    const previous = projects;
    setProjects((prev) => (prev ? prev.filter((p) => p.id !== project_id) : prev));
    setDeleteTarget(null);
    try {
      const r = await apiFetch(`/api/projects/${project_id}`, { method: "DELETE" });
      // 404: already deleted.
      if (!r.ok && r.status !== 404) throw new Error(`http ${r.status}`);
      requestExplorerRefresh();
      await refresh();
    } catch (e) {
      pendingDeleteIdsRef.current.delete(project_id);
      setProjects(previous);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // The server copies the dataset's images and labels into a new project.
  const duplicateProject = async (p: ProjectSummary) => {
    if (duplicatingId) return;
    setDuplicatingId(p.id);
    setError(null);
    try {
      const r = await apiFetch(`/api/projects/${p.id}/duplicate`, { method: "POST" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      requestExplorerRefresh();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDuplicatingId(null);
    }
  };

  // An open Project shows in place of the workspace.
  if (projectViewId) {
    return (
      <ProjectPage
        containerId={projectViewId}
        onBack={() => setProjectViewId(null)}
        onOpenDataset={(dsId) => {
          // The dataset view's back button returns to this Project.
          const fromProject = projectViewId ?? undefined;
          setProjectViewId(null);
          onOpen(dsId, "", fromProject);
        }}
        onNewDataset={(name) => {
          // Named in the Project: onboarding starts at labels and the dataset joins it.
          pendingContainerRef.current = projectViewId;
          setV2ReturnProjectId(projectViewId);
          setV2Labels([]);
          setV2Input("");
          setProjectViewId(null);
          v2BeginLabels(name);
        }}
      />
    );
  }

  return (
    <main className="min-h-full bg-[var(--background)] overflow-x-hidden">
      {/* Title band. z-20 keeps it above the onboarding body, which
          overlaps it with a negative top margin. */}
      <section className="relative z-20 mx-auto max-w-[1400px] px-6 pt-8 pb-6 flex items-end justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          {/* "Workspace", or the new project's name. The arriving heading
              fades in after the leaving one has gone. */}
          {/* pb-2 and leading-[1.3] keep descenders clear of the truncate clip. */}
          <div className="relative pb-2">
            <h1
              className={[
                "text-2xl font-medium tracking-tight leading-[1.3] transition-all duration-150 ease-out",
                v2Active
                  ? "opacity-0 -translate-y-5 pointer-events-none select-none"
                  : "opacity-100 translate-y-0 delay-150",
              ].join(" ")}
            >
              Workspace
            </h1>
            <h1
              aria-hidden={!v2Active}
              className={[
                "absolute inset-0 text-2xl font-medium tracking-tight leading-[1.3] transition-all duration-150 ease-out truncate",
                v2Active
                  ? "opacity-100 translate-y-0 delay-150"
                  : "opacity-0 translate-y-4 pointer-events-none select-none",
              ].join(" ")}
            >
              {v2Name || " "}
            </h1>
          </div>
          {/* Tagline, or the creation date during onboarding. */}
          <div className="relative mt-2 max-w-2xl">
            <p
              className={[
                "text-foreground/50 text-sm transition-all duration-150 ease-out",
                v2Active
                  ? "opacity-0 -translate-y-3 pointer-events-none select-none"
                  : "opacity-100 translate-y-0 delay-150",
              ].join(" ")}
            >
              Label data. Augment it. Export training-ready datasets. All on this machine.
            </p>
            <div
              aria-hidden={!v2Active}
              className={[
                "absolute inset-0 text-sm text-foreground/55 transition-all duration-150 ease-out flex items-center gap-2",
                v2Active
                  ? "opacity-100 translate-y-0 delay-150"
                  : "opacity-0 translate-y-3 pointer-events-none select-none",
              ].join(" ")}
            >
              <span className="tabular-nums">Created {v2DateLabel}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Name popup for "+ Add Dataset", then the labels stage. */}
      <CreateDatasetModal
        open={showWorkspaceCreate}
        onClose={() => setShowWorkspaceCreate(false)}
        onContinue={(name) => {
          setShowWorkspaceCreate(false);
          pendingContainerRef.current = null;
          setV2ReturnProjectId(null);
          setV2Labels([]);
          setV2Input("");
          v2BeginLabels(name);
        }}
      />

      {/* Onboarding body. Labels and creating are centred; import flows from the top. */}
      {v2Active && (
        <div
          className={[
            "relative px-6",
            v2Stage === "labels" || v2Stage === "creating"
              ? "-mt-[6rem] min-h-[calc(100vh-10rem)] flex items-center justify-center pointer-events-none"
              : "pt-4 pb-16 pointer-events-none",
          ].join(" ")}
        >
          {v2WordCloudMounted && <WordCloud show={v2WordCloudShouldShow} />}
          {(v2Stage === "labels" || v2Stage === "creating") && (
            <div
              key="v2-labels"
              className={[
                // The creating loader overlays the form inside this wrapper.
                "relative w-full max-w-2xl pointer-events-auto",
                v2Stage === "labels" ? "animate-[fadeIn_280ms_ease-out]" : "",
              ].join(" ")}
            >
            <div
              key="v2-labels-form"
              className="transition-all duration-300 ease-out"
              style={{
                opacity: v2Stage === "labels" ? 1 : 0,
                transform: v2Stage === "labels" ? "translateY(0)" : "translateY(-6px)",
                pointerEvents: v2Stage === "labels" ? "auto" : "none",
              }}
            >
              {/* Back to the Project page that started onboarding. */}
              {v2ReturnProjectId && (
                <button
                  type="button"
                  onClick={() => {
                    const back = v2ReturnProjectId;
                    v2Reset();
                    if (back) setProjectViewId(back);
                  }}
                  className="mb-5 -ml-1 inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[13px] font-medium text-foreground/55 transition-colors hover:bg-[var(--surface-hover)] hover:text-foreground/90"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M15 18l-6-6 6-6" />
                  </svg>
                  Back to project
                </button>
              )}
              <h2 className="text-xl font-medium tracking-tight text-[var(--foreground)] leading-tight">
                What do you want to detect?
              </h2>
              <p className="mt-3 text-sm text-foreground/50 leading-relaxed">
                Enter the objects you want to detect. You can add more later.
              </p>
              <button
                type="button"
                onClick={v2BeginImport}
                className="mt-2.5 inline-flex items-center gap-1.5 text-sm text-foreground/55 hover:text-foreground transition-colors"
              >
                Already have a labelled dataset?
                <span className="font-medium text-[var(--foreground)]">Import it →</span>
              </button>

              <div className="mt-8 rounded-md border border-[var(--line)] bg-[var(--panel)] focus-within:border-[var(--accent)] transition-colors px-4 py-3.5 flex flex-wrap items-center gap-2">
                {v2Labels.map((lab, i) => {
                  const bg = v2LabelColours[lab.toLowerCase()] ?? LABEL_COLOURS[i % LABEL_COLOURS.length];
                  return (
                    <span
                      key={`${lab}-${i}`}
                      className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent pl-2.5 pr-1 h-7 font-mono text-[12px] text-[var(--foreground)] animate-[fadeIn_180ms_ease-out]"
                    >
                      <span aria-hidden className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: bg }} />
                      <span className="select-none">{lab}</span>
                      <button
                        type="button"
                        onClick={() => v2RemoveLabel(i)}
                        aria-label={`Remove ${lab}`}
                        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-foreground/40 transition-colors hover:bg-[var(--surface-hover)] hover:text-foreground"
                      >
                        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden>
                          <path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                        </svg>
                      </button>
                    </span>
                  );
                })}
                <input
                  ref={v2InputRef}
                  value={v2Input}
                  onChange={(e) => setV2Input(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === "," || e.key === ".") {
                      e.preventDefault();
                      v2AddLabel(v2Input);
                    } else if (e.key === "Backspace" && v2Input === "" && v2Labels.length) {
                      e.preventDefault();
                      v2RemoveLabel(v2Labels.length - 1);
                    }
                  }}
                  onBlur={() => {
                    if (v2Input.trim()) v2AddLabel(v2Input);
                  }}
                  placeholder={v2Labels.length === 0 ? "e.g. pothole, crack, manhole" : ""}
                  className="flex-1 min-w-[8rem] bg-transparent py-1 text-base text-[var(--foreground)] placeholder:text-foreground/35"
                />
              </div>

              {/* The workspace error line is hidden during onboarding. */}
              {error && <p className="mt-3 text-sm text-[var(--bad)]">{error}</p>}

              <div className="mt-6 flex items-center gap-3">
                <button
                  type="button"
                  onClick={v2Reset}
                  className="ml-auto inline-flex items-center justify-center px-4 py-2.5 text-sm leading-none text-foreground/45 hover:text-foreground/80 transition-colors"
                  title="Cancel and discard this new dataset."
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={v2SkipFromLabels}
                  className="inline-flex items-center justify-center px-4 py-2.5 text-sm leading-none text-foreground/55 hover:text-foreground/90 transition-colors"
                  title="Skip: open the dataset without setting labels."
                >
                  Skip
                </button>
                <div className="relative">
                  <button
                    type="button"
                    onClick={v2DoneLabels}
                    disabled={v2Labels.length === 0 || v2Stage === "creating"}
                    className="pk-btn pk-btn-primary"
                  >
                    Done
                  </button>
                  {/* Absolute, so the hint does not shift the Done button. */}
                  <span className="pointer-events-none absolute right-0 top-full mt-1 text-[10px] text-foreground/30 font-mono select-none">
                    Ctrl ↵
                  </span>
                </div>
              </div>
            </div>

            {v2Stage === "creating" && (
              <div
                key="v2-creating"
                className="absolute inset-0 grid place-items-center pointer-events-auto animate-[fadeIn_320ms_ease-out]"
              >
                <PixelKitLoader size={128} message="Opening dataset…" />
              </div>
            )}
            </div>
          )}

          {v2Stage === "import" && (
            <div
              key="v2-import"
              className="w-full max-w-2xl mx-auto animate-[fadeIn_280ms_ease-out] pointer-events-auto"
            >
              <button
                type="button"
                onClick={() => { if (!v2ImportBusy) setV2Stage("labels"); }}
                disabled={v2ImportBusy}
                className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--fg-dim)] hover:text-foreground transition-colors mb-4 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <span aria-hidden>←</span>
                Back to labels
              </button>
              <h2 className="text-xl font-medium tracking-tight text-[var(--foreground)] leading-tight">
                Import a labelled dataset
              </h2>
              <p className="mt-3 text-sm text-foreground/50 leading-relaxed">
                Pascal VOC (<span className="font-mono">Annotations/*.xml</span>), COCO (a JSON
                with images and annotations) or YOLO (<span className="font-mono">labels/*.txt</span>{" "}
                with <span className="font-mono">classes.txt</span> or{" "}
                <span className="font-mono">data.yaml</span>), as a folder or a{" "}
                <span className="font-mono">.zip</span>. Every image and box comes in fully
                editable; polygon masks are kept.
              </p>

              {error && <p className="mt-4 text-sm text-[var(--bad)]">{error}</p>}

              {/* Folder or zip picker, until a dataset is parsed. */}
              {!v2ImportParsed && (
                <div className="mt-8 flex flex-wrap items-center gap-3">
                  <label className="pk-btn cursor-pointer">
                    Choose folder
                    <input
                      type="file"
                      multiple
                      className="hidden"
                      ref={(el) => {
                        if (el) {
                          el.setAttribute("webkitdirectory", "");
                          el.setAttribute("directory", "");
                        }
                      }}
                      onChange={(e) => v2PickDataset(e.target.files)}
                    />
                  </label>
                  <label className="pk-btn cursor-pointer">
                    Choose .zip
                    <input
                      type="file"
                      accept=".zip,application/zip"
                      className="hidden"
                      onChange={(e) => v2PickDataset(e.target.files)}
                    />
                  </label>
                  {v2ImportParsing && (
                    <span className="text-sm text-foreground/55 inline-flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-full border-2 border-foreground/50 border-t-transparent animate-spin" />
                      Reading dataset
                      {v2ImportParseProgress
                        ? `… ${v2ImportParseProgress.done}/${v2ImportParseProgress.total}`
                        : "…"}
                    </span>
                  )}
                </div>
              )}

              {/* Preview once parsed. */}
              {v2ImportParsed && (
                <div className="mt-8 animate-[fadeIn_220ms_ease-out]">
                  <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] p-5">
                    <div className="text-sm text-[var(--foreground)] tabular-nums">
                      <span className="font-medium uppercase">{v2ImportParsed.format}</span>
                      <span className="text-foreground/45"> · </span>
                      {v2ImportParsed.stats.images.toLocaleString()} images
                      <span className="text-foreground/45"> · </span>
                      {v2ImportParsed.stats.boxes.toLocaleString()} boxes
                      {v2ImportParsed.stats.background > 0 && (
                        <>
                          <span className="text-foreground/45"> · </span>
                          {v2ImportParsed.stats.background.toLocaleString()} background
                        </>
                      )}
                    </div>
                    {v2ImportParsed.classes.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {v2ImportParsed.classes.map((c, i) => {
                          const bg = LABEL_COLOURS[i % LABEL_COLOURS.length];
                          return (
                            <span
                              key={c}
                              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent px-2 h-6 font-mono text-[12px] text-[var(--foreground)]"
                            >
                              <span aria-hidden className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: bg }} />
                              {c}
                            </span>
                          );
                        })}
                      </div>
                    )}
                    {v2ImportParsed.warnings.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {v2ImportParsed.warnings.map((w, i) => (
                          <li key={i} className="text-[12px] text-[var(--warn)]">
                            {w}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  <div className="mt-6 flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => { if (!v2ImportBusy) setV2ImportParsed(null); }}
                      disabled={v2ImportBusy}
                      className="inline-flex items-center justify-center px-4 py-2.5 text-sm leading-none text-foreground/55 hover:text-foreground/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Choose another
                    </button>
                    <button
                      type="button"
                      onClick={v2Reset}
                      disabled={v2ImportBusy}
                      className="ml-auto inline-flex items-center justify-center px-4 py-2.5 text-sm leading-none text-foreground/45 hover:text-foreground/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={v2RunImport}
                      disabled={v2ImportBusy || v2ImportParsed.items.length === 0}
                      className="pk-btn pk-btn-primary tabular-nums"
                    >
                      {v2ImportBusy && (
                        <span className="inline-block h-3 w-3 rounded-full border-2 border-current border-t-transparent opacity-60 animate-spin" />
                      )}
                      {v2ImportBusy
                        ? `Importing… ${v2ImportProgress ? `${v2ImportProgress.done}/${v2ImportProgress.total}` : ""}`
                        : `Import ${v2ImportParsed.stats.images.toLocaleString()} images`}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      )}

      {/* Projects grid, hidden during onboarding. */}
      {!v2Active && (
        <section className="mx-auto max-w-[1400px] px-6 pb-24 grid gap-8">
          {error && <div className="text-sm text-[var(--bad)]">{error}</div>}

          {/* Projects: containers of datasets. */}
          <ProjectsSection onOpenProject={(id) => setProjectViewId(id)} />

          {/* Datasets toolbar. Tabs, search and sort appear once there are
              datasets; the create button always shows. */}
          {sortedProjects !== null && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* Tabs and sort */}
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-1">
                  {(projects?.length ?? 0) > 0 && ([
                    { key: "all" as const, label: "All datasets", count: total ?? projects?.length ?? 0 },
                    { key: "starred" as const, label: "Starred", count: starredCount },
                  ]).map((t) => {
                    const isActive = tab === t.key;
                    return (
                      <button
                        key={t.key}
                        type="button"
                        onClick={() => setTab(t.key)}
                        className={[
                          "inline-flex items-center gap-2 rounded-md px-3 h-9 text-[13px] transition-colors",
                          isActive
                            ? "bg-accent/10 text-accent font-medium"
                            : "text-foreground/55 hover:text-foreground/90 hover:bg-[var(--surface-hover)]",
                        ].join(" ")}
                      >
                        {t.label}
                        <span
                          className={[
                            "tabular-nums rounded px-1.5 text-[11px] leading-5",
                            isActive ? "bg-accent/15 text-accent" : "bg-foreground/[0.06] text-foreground/45",
                          ].join(" ")}
                        >
                          {t.count}
                        </span>
                      </button>
                    );
                  })}
                </div>
                {(projects?.length ?? 0) > 0 && <SortMenu value={sortMode} onChange={setSortMode} />}
              </div>
              {/* Search and create */}
              <div className="flex items-center gap-2 flex-wrap">
                {(projects?.length ?? 0) > 0 && (
                  <label className="relative">
                    <span className="sr-only">Search datasets</span>
                    <svg
                      aria-hidden
                      viewBox="0 0 24 24"
                      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/40"
                      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    >
                      <circle cx="11" cy="11" r="7" />
                      <path d="m21 21-4.3-4.3" />
                    </svg>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search datasets"
                      className="h-9 rounded-md bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--accent)] outline-none pl-9 pr-3 text-sm text-[var(--foreground)] placeholder:text-foreground/40 transition-colors w-44 sm:w-56"
                    />
                  </label>
                )}
                <button
                  type="button"
                  onClick={() => {
                    // Workspace create is never scoped to a Project.
                    pendingContainerRef.current = null;
                    if (v2Stage === "idle") setShowWorkspaceCreate(true);
                    else v2Reset();
                  }}
                  className={[
                    "h-9 rounded-md px-4 text-[13px] font-medium transition",
                    v2Stage !== "idle"
                      ? "border border-[var(--line)] text-foreground/75 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
                      : "bg-[var(--accent)] text-[var(--accent-contrast)] hover:brightness-105",
                  ].join(" ")}
                >
                  {v2Stage !== "idle" ? "Cancel" : "+ Add Dataset"}
                </button>
              </div>
            </div>
          )}

          {sortedProjects === null ? (
            <div className="text-[var(--muted)] text-sm">Loading…</div>
          ) : sortedProjects.length === 0 ? (
            <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] py-10 text-center text-foreground/50">
              {isSearching ? (
                <>No datasets match <span className="text-foreground/90">&ldquo;{searchQuery.trim()}&rdquo;</span>.</>
              ) : tab === "starred" ? (
                <>No starred datasets yet. Click the star on a card to pin it here.</>
              ) : (
                <>No datasets yet. Click <span className="text-foreground/90">+ Add Dataset</span> to start one.</>
              )}
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {sortedProjects.slice(0, renderCount).map((p, i) => (
                  <ProjectCard
                    key={p.id}
                    project={p}
                    favourited={p.favouritedByMe}
                    onToggleFavourite={() => toggleFavourite(p.id)}
                    onOpen={() => onOpen(p.id, p.name)}
                    onOpenProject={(id) => setProjectViewId(id)}
                    onDelete={() => setDeleteTarget(p)}
                    onDuplicate={() => duplicateProject(p)}
                    duplicating={duplicatingId === p.id}
                    eagerCover={i < WORKSPACE_PAGE_SIZE}
                  />
                ))}
                {/* Placeholders for up to one page not yet loaded. */}
                {tab === "all"
                  && !isSearching
                  && total !== null
                  && projects !== null
                  && projects.length < total
                  && Array.from({ length: Math.min(WORKSPACE_PAGE_SIZE, total - projects.length) }).map((_, i) => (
                    <div
                      key={`ws-skel-${projects.length + i}`}
                      className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden"
                    >
                      <div className="aspect-video bg-foreground/[0.04]" />
                      <div className="p-4 space-y-2">
                        <div className="h-4 w-2/3 rounded bg-foreground/[0.06]" />
                        <div className="h-3 w-1/3 rounded bg-foreground/[0.04]" />
                      </div>
                    </div>
                  ))}
              </div>
              {/* Infinite scroll sentinel, on the All tab without a search. */}
              {tab === "all" && !isSearching && total !== null && projects !== null && projects.length < total && (
                <div ref={sentinelRef} aria-hidden className="h-1 w-full mt-2" />
              )}
            </>
          )}
        </section>
      )}

      {deleteTarget && (
        <DeleteProjectDialog
          project={deleteTarget}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => confirmDelete(deleteTarget.id)}
        />
      )}
    </main>
  );
}

// Card cover: the server's /cover_thumb, which finds the cover among the images
// and the references. The blurhash stays when it does not load. `updatedAt`
// busts the browser cache.
function ProjectCover({
  projectId,
  updatedAt,
  blurhash = null,
  eager = false,
}: {
  projectId: string;
  updatedAt: string | null;
  /** Placeholder behind the cover image. */
  blurhash?: string | null;
  /** Loads the image at once instead of near the viewport. */
  eager?: boolean;
}) {
  const cacheBuster = updatedAt ? `?v=${encodeURIComponent(updatedAt)}` : "";
  const thumbUrl = `${API_BASE}/api/projects/${projectId}/cover_thumb${cacheBuster}`;
  // A new URL preloads hidden as `pending`, so the painted `src` never blanks.
  const [src, setSrc] = useState(thumbUrl);
  const [pending, setPending] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (thumbUrl === src) return;
    setPending(thumbUrl);
  }, [thumbUrl, src]);

  // A non-eager cover loads once it is within 600 px of the viewport.
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [showImage, setShowImage] = useState<boolean>(eager);
  useEffect(() => {
    if (showImage) return;
    const node = wrapRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") { setShowImage(true); return; }
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setShowImage(true);
            obs.disconnect();
            return;
          }
        }
      },
      { rootMargin: "600px" },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, [showImage]);

  return (
    <div ref={wrapRef} className="relative w-full h-full">
      {blurhash && (
        <BlurhashCanvas
          hash={blurhash}
          width={32}
          height={18}
          punch={1}
          className="absolute inset-0 w-full h-full"
        />
      )}
      {showImage && (
        <>
          {!failed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={src}
              alt=""
              loading="lazy"
              decoding="async"
              onError={() => setFailed(true)}
              className="absolute inset-0 w-full h-full object-cover"
            />
          )}
          {pending && pending !== src && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={pending}
              alt=""
              decoding="async"
              onLoad={() => { setSrc(pending); setPending(null); setFailed(false); }}
              onError={() => setPending(null)}
              style={{ display: "none" }}
            />
          )}
        </>
      )}
    </div>
  );
}


function ProjectCard({
  project,
  favourited,
  onToggleFavourite,
  onOpen,
  onOpenProject,
  onDelete,
  onDuplicate,
  duplicating = false,
  eagerCover = false,
}: {
  project: ProjectSummary;
  favourited: boolean;
  onToggleFavourite: () => void;
  onOpen: () => void;
  onOpenProject?: (id: string) => void;
  onDelete: () => void;
  onDuplicate?: () => void;
  duplicating?: boolean;
  /** The cover image loads at once. */
  eagerCover?: boolean;
}) {
  const statusBadges = projectStatusBadges(project);
  const [tagsOverflowOpen, setTagsOverflowOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const coverBtn =
    "h-8 w-8 rounded-md grid place-items-center backdrop-blur-md bg-white/85 hover:bg-white text-zinc-700 transition-colors";
  const coverBtnShadow = { boxShadow: "var(--shadow-soft)" };
  return (
    <div
      className="pk-card pk-card-hover group overflow-hidden rounded-md"
      style={{
        // Above the dim overlay, z-1000, while a popup is open.
        position: tagsOverflowOpen || menuOpen ? "relative" : undefined,
        zIndex: tagsOverflowOpen || menuOpen ? 1001 : undefined,
      }}
    >
      <div className="relative">
        <button onClick={onOpen} className="block w-full text-left">
          <div className="pk-cover aspect-video flex items-center justify-center">
            {project.thumbnail ? (
              <ProjectCover
                projectId={project.id}
                updatedAt={project.updatedAt}
                blurhash={project.cover_blurhash ?? null}
                eager={eagerCover}
              />
            ) : (
              <span className="text-foreground/30 text-xs">No images yet</span>
            )}
          </div>
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavourite();
          }}
          aria-pressed={favourited}
          aria-label={favourited ? "Unstar" : "Star"}
          className={[coverBtn, "absolute top-3 right-3", favourited ? "text-[var(--accent)]" : ""].join(" ")}
          style={coverBtnShadow}
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill={favourited ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          >
            <path d="M12 17.3l-6.18 3.7 1.64-7.03L2 9.24l7.19-.61L12 2l2.81 6.63L22 9.24l-5.46 4.73L18.18 21z" />
          </svg>
        </button>
        <div className="absolute bottom-3 right-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((v) => !v);
            }}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label="Dataset actions"
            className={coverBtn}
            style={coverBtnShadow}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden>
              <circle cx="5" cy="12" r="1.6" />
              <circle cx="12" cy="12" r="1.6" />
              <circle cx="19" cy="12" r="1.6" />
            </svg>
          </button>
          {menuOpen && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                }}
                aria-hidden
              />
              <div
                role="menu"
                className="pk-glass absolute bottom-[calc(100%+8px)] right-0 z-50 w-44 rounded-md p-1.5 shadow-[var(--shadow-strong)]"
              >
                {onDuplicate && (
                  <button
                    type="button"
                    role="menuitem"
                    disabled={duplicating}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (duplicating) return;
                      setMenuOpen(false);
                      onDuplicate();
                    }}
                    className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-foreground/80 hover:bg-foreground/[0.06] transition-colors disabled:opacity-50 disabled:cursor-default"
                  >
                    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                      <rect x="9" y="9" width="11" height="11" rx="2" />
                      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
                    </svg>
                    {duplicating ? "Duplicating…" : "Duplicate"}
                  </button>
                )}
                <button
                  type="button"
                  role="menuitem"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpen(false);
                    onDelete();
                  }}
                  className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-[var(--bad)] hover:bg-[var(--surface-hover)] transition-colors"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                  </svg>
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <button onClick={onOpen} className="block w-full text-left">
        <div className="p-4 grid gap-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-lg font-semibold tracking-tight truncate flex items-center gap-2">
                <span className="truncate">{project.name}</span>
                {project.derived && <DerivedBadge parentName={project.derived.parentName} />}
              </div>
            </div>
            {statusBadges.length > 0 && (
              <div className="flex flex-wrap gap-1.5 justify-end shrink-0 pt-0.5">
                {statusBadges.map((b) => (
                  <span
                    key={b.label}
                    className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent px-2 py-0.5 font-mono text-[12px] leading-normal text-[var(--fg-soft)]"
                  >
                    <span aria-hidden className={["h-2 w-2 rounded-full shrink-0", b.dot].join(" ")} />
                    {b.label}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <div className="flex items-baseline gap-1.5">
              <span className="pk-micro">Images</span>
              <span className="tabular-nums text-sm font-semibold text-[var(--foreground)]">{project.n_images}</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="pk-micro">Labelled</span>
              <span className="tabular-nums text-sm font-semibold text-[var(--foreground)]">
                {project.n_labelled}
                <span className="text-foreground/35 font-normal"> / {project.n_images}</span>
              </span>
            </div>
          </div>

          {project.tags.length > 0 && (
            <ProjectTagsRow
              tags={project.tags}
              labelAliases={project.label_aliases}
              colourOverrides={project.labelColours}
              onOverflowOpenChange={setTagsOverflowOpen}
            />
          )}
        </div>
      </button>
      {/* The empty span keeps the Project chip right-aligned. */}
      <div className="px-4 pb-3.5 flex items-center justify-between text-xs">
        <span />
        {project.container && onOpenProject && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onOpenProject(project.container!.id);
            }}
            className="inline-flex items-center gap-1 rounded-md border border-[var(--line)] px-2.5 py-1 text-[11px] font-medium text-foreground/70 transition-colors hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
            title={`Open project: ${project.container.name}`}
          >
            <svg
              viewBox="0 0 24 24"
              className="h-3 w-3"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            </svg>
            <span className="max-w-[120px] truncate">{project.container.name}</span>
          </button>
        )}
      </div>
    </div>
  );
}

function DeleteProjectDialog({
  project,
  onCancel,
  onConfirm,
}: {
  project: ProjectSummary;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const matches = typed === project.name;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const submit = async () => {
    if (!matches || busy) return;
    setBusy(true);
    try {
      await onConfirm();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="pk-backdrop fixed inset-0 z-50 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel();
      }}
    >
      <div className="pk-glass pk-pop max-w-md w-full rounded-md overflow-hidden">
        <header className="px-6 pt-6 pb-3">
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--bad)]">Permanent action</div>
          <h2 className="mt-2 text-xl font-semibold tracking-tight">Delete dataset?</h2>
        </header>

        <div className="px-6 pb-5 grid gap-4">
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            This removes{" "}
            <span className="font-mono text-[var(--foreground)]">{project.name}</span> along with{" "}
            <span className="font-mono text-[var(--foreground)]">{project.n_images}</span> image
            {project.n_images === 1 ? "" : "s"}, every label, every verdict, and any model state.{" "}
            <span className="text-[var(--bad)]">This cannot be recovered.</span>
          </p>

          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2.5 text-xs text-[var(--bad)]">
            Type the dataset name{" "}
            <span className="font-mono text-[var(--foreground)]">{project.name}</span> below to confirm.
          </div>

          <input
            autoFocus
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            placeholder={project.name}
            className="w-full rounded-md border border-[var(--line)] bg-transparent px-3 py-2.5 font-mono text-sm focus:outline-none focus:border-[var(--line-strong)]"
            spellCheck={false}
            autoComplete="off"
          />
        </div>

        <footer className="px-6 py-4 border-t border-[var(--line)] flex items-center justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-md border border-[var(--line)] px-4 py-2 text-[13px] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!matches || busy}
            className="rounded-md bg-[var(--bad)] px-4 py-2 text-[13px] font-medium text-[var(--background)] transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Deleting…" : "Delete forever"}
          </button>
        </footer>
      </div>
    </div>
  );
}
