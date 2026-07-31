"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Footer } from "./Footer";
import { ProjectTagsRow } from "./components/ProjectTagsRow";
import { containsProfanity } from "./profanity";
import { BlurhashCanvas } from "react-blurhash";
import { usePlan } from "./PlanPill";
import { LABEL_COLOURS, readableTextForBg } from "./v2/OnboardLabelsV2";
import type { ReferenceImage } from "./v2/OnboardReferencesV2";
import { detectionsToBoxes, type MaskShape, type EditableBox } from "./BoxEditor";
import { ReferenceImageEditor } from "./v2/ReferenceImageEditor";
import { WordCloud } from "./v2/WordCloud";
import { PixelKitLoader } from "./v2/PixelKitLoader";
import { Tooltip } from "./Tooltip";
import { capture } from "./lib/analytics";
import { ProjectsSection } from "./ProjectsSection";
import { ProjectPage } from "./ProjectPage";
import { CreateDatasetModal } from "./CreateDatasetModal";
import { apiFetch } from "@/lib/apiFetch";
import { addDataset } from "@/lib/containers";
import { isProPlan } from "@/lib/plans";
import { patchProjectMeta, readProjectMetaCache, writeProjectMetaCache, type ProjectMetaCache } from "@/lib/projectMetaCache";
import { isImageFile, resizeForUpload } from "@/lib/resize";
import {
  collectInput,
  parseDataset,
  uploadDataset,
  IMPORT_MAX_SIZE_OPTIONS,
  ZIP_SOFT_LIMIT_BYTES,
  type ParsedDataset,
} from "@/lib/datasetImport";

// Total reference images uploaded as a shared pool (not per-label).
const V2_MAX_REFS = 20;
// Annotations required per label (shown as a target, not enforced in the UI).
const V2_ANNOTS_PER_LABEL = 5;

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type ProjectSummary = {
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
  createdBy: string;
  likes: number;
  likedByMe: boolean;
  favourites: number;
  favouritedByMe: boolean;
  certified: boolean;
  private?: boolean;
  // Set on derived ("child") projects — cropped, one-label-per-image clones of
  // a parent. Drives the derived badge on the card.
  derived?: { parentProjectId?: string; parentName?: string } | null;
  running?: boolean;
  // True for projects created via the V2 onboarding flow, drives the
  // workspace's V1-vs-V2 dispatch when the user clicks a project card.
  v2?: boolean;
  // V2-only count of stored reference images (the list endpoint
  // returns this for V2 projects; for V1 projects it's always 0).
  n_references?: number;
  // BlurHash-encoded thumbnail (string ~30 chars) for the cover
  // image. Decoded client-side via react-blurhash into a colour
  // gradient placeholder that renders before the real image
  // streams in. Null when the cover is missing or encode failed.
  cover_blurhash?: string | null;
  // Display-rename map (canonical_lower → renamed). Applied at
  // chip render so a label rename inside the project view shows
  // up on the workspace + public cards immediately instead of
  // waiting for the user to reopen the project.
  label_aliases?: Record<string, string>;
  // Per-label colour overrides ({canonical_lower: "#rrggbb"}). Used
  // by the card's chip row so a colour change in Settings repaints
  // on every workspace + public surface without a refetch.
  labelColours?: Record<string, string>;
  // The Project (container) this dataset belongs to, or null. Drives
  // the clickable Project chip on the dataset card that jumps to the
  // Project page.
  container?: { id: string; name: string } | null;
};


// Shallow-equality check on the fields the Workspace cards actually
// render. We don't deep-compare the whole structure, bytes-level
// equality is too strict (e.g. server reorders the tag list) and
// JSON.stringify on every poll is wasteful. Comparing the visible
// columns is enough to skip needless re-renders.
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
      x.likes !== y.likes ||
      x.likedByMe !== y.likedByMe ||
      x.favourites !== y.favourites ||
      x.favouritedByMe !== y.favouritedByMe ||
      x.certified !== y.certified ||
      x.private !== y.private ||
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


// Stable hash → hue, so each project gets a consistent colour palette for
// its label chips. Same function as ProjectsView so chips look identical
// across both pages.
// Inline padlock used by both Workspace + Public cards to flag projects
// the owner has marked private. Same orange-glow accent as elsewhere in
// the privacy UI.
export function PrivateLockIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-amber-600 dark:text-amber-300/80 shrink-0"
      aria-label="Private project"
      role="img"
    >
      <title>Private project</title>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </svg>
  );
}

// Branch glyph flagging a derived (cropped child) project on the cards — sky
// accent so it reads distinctly from the amber privacy padlock.
export function DerivedBadge({ parentName }: { parentName?: string | null }) {
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
      className="text-sky-600 dark:text-sky-400/80 shrink-0"
      aria-label="Derived project"
      role="img"
    >
      <title>{parentName ? `Derived from ${parentName}` : "Derived project"}</title>
      <circle cx="4" cy="3.2" r="1.7" />
      <circle cx="4" cy="12.8" r="1.7" />
      <circle cx="12" cy="12.8" r="1.7" />
      <path d="M4 4.9V11.1M5.7 12.8H10.3" />
    </svg>
  );
}

function hueFor(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

// Workspace sort modes for the toolbar dropdown. Sorting is purely
// client-side over the already-loaded list (the server only handles
// the search `q`), so adding a mode here never needs a backend change.
type SortMode = "updated" | "created" | "name" | "images";
const SORT_OPTIONS: { key: SortMode; label: string }[] = [
  { key: "updated", label: "Recently updated" },
  { key: "created", label: "Recently created" },
  { key: "name", label: "Name" },
  { key: "images", label: "Most images" },
];

// Status pill set for a project card (Labelled / Partial / Unlabelled
// / In progress, plus Model). Outline style so it reads as a quiet
// badge on the new card surface. Shared with the public feed so the
// two card surfaces stay visually identical.
export function projectStatusBadges(
  project: ProjectSummary,
): { label: string; classes: string }[] {
  const nImages = project.n_images ?? 0;
  const nLabelled = project.n_labelled ?? 0;
  const nUnlabelled = project.n_unlabelled ?? 0;
  const allLabelled = nImages > 0 && nUnlabelled === 0;
  const someLabelled = nLabelled > 0 && nUnlabelled > 0;
  const allUnlabelled = nImages > 0 && !nLabelled;
  const badges: { label: string; classes: string }[] = [];
  if (project.running) {
    badges.push({ label: "In progress", classes: "border-orange-500/45 text-orange-700 dark:border-orange-400/45 dark:text-orange-300" });
  } else if (allLabelled) {
    badges.push({ label: "Labelled", classes: "border-emerald-500/45 text-emerald-700 dark:border-emerald-400/45 dark:text-emerald-300" });
  } else if (someLabelled) {
    badges.push({ label: "Partial", classes: "border-amber-500/50 text-amber-700 dark:border-amber-400/45 dark:text-amber-300" });
  } else if (allUnlabelled) {
    badges.push({ label: "Unlabelled", classes: "border-red-500/45 text-red-600 dark:border-red-400/45 dark:text-red-300" });
  }
  if (project.hasModel) {
    badges.push({ label: "Model", classes: "border-sky-500/45 text-sky-700 dark:border-sky-400/45 dark:text-sky-300" });
  }
  return badges;
}

// Compact "2h ago" / "3d ago" relative timestamp for the card footer.
// Falls back to an empty string for a missing / unparseable date so
// the footer just omits the line rather than printing "NaN".
export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.round(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.round(days / 365)}y ago`;
}

// Small custom dropdown for the workspace sort control. Mirrors the
// chrome of DatasetTypePill (button + click-away backdrop + menu) so
// it reads consistently with the rest of the app and themes cleanly.
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
        className="inline-flex h-9 items-center gap-1.5 rounded-full border border-foreground/10 bg-foreground/[0.03] hover:border-foreground/20 hover:bg-foreground/[0.06] px-3.5 text-sm text-foreground/75 transition-colors"
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
            className="absolute right-0 top-[calc(100%+8px)] z-50 w-48 rounded-xl border border-foreground/10 bg-[var(--background)] p-1.5 shadow-xl"
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
                  "flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-sm transition-colors",
                  opt.key === value
                    ? "bg-foreground/[0.08] text-[var(--foreground)]"
                    : "text-foreground/75 hover:bg-foreground/[0.05]",
                ].join(" ")}
              >
                {opt.label}
                {opt.key === value && (
                  <svg
                    viewBox="0 0 24 24"
                    className="h-3.5 w-3.5 text-[var(--accent-orange)]"
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
  username,
  userImage = null,
  loggedIn = true,
}: {
  onOpen: (name: string, owner?: string, displayName?: string, v2?: boolean, fromProjectId?: string) => void;
  /** Optional V2 entry point. Fired once the user has finished the
      whole inline V2 flow (name + labels + references). The parent
      mounts the post-onboarding view from this callback. When the
      prop is undefined the V2 button stays hidden and only the V1
      path is reachable. */
  onV2Begin?: (
    name: string,
    labels: string[],
    references: ReferenceImage[],
    projectId: string | null,
    /** "general" → the project view should suppress its own
     *  full-screen mount loader (the HomeView "Reading between the
     *  labels…" overlay is still on screen and carries through as
     *  "Loading project…").
     *  "specific" → render the smaller in-page "Loading project…"
     *  card while /initial is in flight, the user just left a busy
     *  references screen and a full-screen takeover would feel
     *  abrupt.
     *  null → normal load, project view uses its default loader. */
    firstLoad?: "general" | "specific" | null,
  ) => void;
  username: string;
  /** Avatar URL for the signed-in user. Shown beside the username
      under the project title during V2 onboarding so the metadata
      line matches how project cards display owners. */
  userImage?: string | null;
  loggedIn?: boolean;
}) {
  // Hydrate from localStorage on mount so coming back from a
  // project view paints the workspace cards (cover photos, names,
  // counts) on the first frame instead of flashing the loading
  // spinner. The /api/projects fetch still runs in refresh() and
  // overwrites with fresh data when it resolves, but the cached
  // copy carries the user through the fetch latency.
  const [projects, setProjects] = useState<ProjectSummary[] | null>(() => {
    if (typeof window === "undefined") return null;
    try {
      const raw = window.localStorage.getItem("pixelkit_workspace_projects_v1");
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as ProjectSummary[]) : null;
    } catch {
      return null;
    }
  });
  // Pagination state, matches the public projects page pattern.
  // Initial mount fetches one page; sentinel-driven infinite scroll
  // pulls more. `total` is the canonical project count from the
  // backend so we can render blurred placeholders for the slots
  // that haven't paginated in yet.
  const WORKSPACE_PAGE_SIZE = 12;
  const [total, setTotal] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  // Live mirror of projects.length so the 4-second poll's refresh
  // closure always queries the CURRENT pagination cursor instead of
  // the stale one it captured at creation. Without this the poll
  // refetched limit=WORKSPACE_PAGE_SIZE forever and clobbered any
  // paginated entries, which made the workspace flicker every 4 s
  // when the user had scrolled past the first page.
  const projectsLenRef = useRef<number>(0);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  // Projects the user has just deleted. We strip these from any
  // /api/projects refresh response until the backend stops listing
  // them, without this, a refresh that races the delete (R2 cleanup,
  // shutil.rmtree latency) re-introduces the project card and lets
  // the user click delete again, which then 404s.
  const pendingDeleteIdsRef = useRef<Set<string>>(new Set());
  // Track projects whose favourite toggle is mid-flight. The poll
  // would otherwise clobber the optimistic state with the (still
  // un-favourited) server response if the 4 s refresh happens to land
  // inside the favourite POST's request/response window.
  const pendingFavouriteIdsRef = useRef<Map<string, boolean>>(new Map());
  const [searchQuery, setSearchQuery] = useState("");
  // Debounced + ref'd copy of the search query, read by refresh()
  // and loadMoreWorkspacePage() so they send the latest `q` without
  // re-binding via deps (refresh is also driven by a setInterval).
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const queryRef = useRef("");
  // Workspace toolbar: tab filter (all vs starred) + sort mode. Both
  // are client-side over the already-fetched list, so they never
  // touch the server search (`q`) path or the pagination logic.
  const [tab, setTab] = useState<"all" | "starred">("all");
  const [sortMode, setSortMode] = useState<SortMode>("updated");
  const [error, setError] = useState<string | null>(null);
  // When set, the workspace is replaced by the one-page Project (container) view.
  const [projectViewId, setProjectViewId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectSummary | null>(null);
  // Id of the dataset currently being duplicated (a few seconds for big
  // datasets) — disables the card's Duplicate action so it can't double-fire.
  const [duplicatingId, setDuplicatingId] = useState<string | null>(null);

  // Deep-link Projects via `?project=<id>` so each Project has a unique,
  // shareable URL. Read it on mount + on browser back/forward; ProjectPage's
  // getContainer 404s for non-members, so private Projects stay protected
  // exactly like private datasets.
  useEffect(() => {
    if (!loggedIn || typeof window === "undefined") return;
    const fromUrl = () => new URLSearchParams(window.location.search).get("project");
    const initial = fromUrl();
    if (initial) setProjectViewId(initial);
    const onPop = () => setProjectViewId(fromUrl() || null);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loggedIn]);

  // Keep the URL in sync when a Project is opened/closed so it's copy-shareable.
  useEffect(() => {
    if (!loggedIn || typeof window === "undefined") return;
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
  }, [projectViewId, loggedIn]);

  // V2 inline onboarding. The V2 button opens its own naming form
  // (visually identical to V1) so V1's `creating` flow stays untouched
  // and either path is a one-click revert. After Enter on the name
  // field the workspace title swaps to the project name in-place, the
  // subtitle becomes the V2 prompt, search + add buttons fade out, the
  // page locks scroll, and the labels / references stages render in
  // the same vertical band underneath.
  type V2Stage = "idle" | "name" | "labels" | "classifying" | "references" | "import";
  const [v2Stage, setV2Stage] = useState<V2Stage>("idle");
  // Message shown on the in-place PixelKit loader during classifying.
  // Starts as "Reading between the labels…" the moment the user
  // clicks Done; flips to "Loading project…" the instant the
  // classifier returns "general" so the overlay carries straight
  // through the v2HandOff → ProjectViewV2 mount without a copy gap.
  const [v2ClassifyMsg, setV2ClassifyMsg] = useState("Reading between the labels…");
  const [v2Name, setV2Name] = useState("");
  const [v2Labels, setV2Labels] = useState<string[]>([]);
  // Random colour chosen for each label the moment it's added in the
  // onboarding chip strip. Persisted to the project's manifest at
  // creation time (see v2EnsureProject) so the chips stay the colours
  // the user previewed during onboarding.
  const [v2LabelColours, setV2LabelColours] = useState<Record<string, string>>({});
  const [v2Input, setV2Input] = useState("");
  // ID of the V2 project, created eagerly when the user enters the
  // references stage. Lets each upload POST directly to
  // /api/v2/projects/{id}/references, single round-trip, server
  // runs GD+SAM + saves bytes + embeds in one shot. Without this
  // we'd hit /references/process (no save) then /references (save
  // + embed) per image, GPU twice for the same bytes.
  const [v2ProjectId, setV2ProjectId] = useState<string | null>(null);
  const v2ProjectIdRef = useRef<string | null>(null);
  // ── Import-a-labelled-dataset flow ──────────────────────────────────
  // The "import" stage reuses v2Name + v2EnsureProject + onV2Begin, but
  // derives its labels from the dataset instead of the chip input. State:
  // the parsed dataset (folder/zip → VOC items + classes), the chosen max
  // image size (null = preserve full resolution), and parse/upload progress.
  const [v2ImportParsing, setV2ImportParsing] = useState(false);
  const [v2ImportParseProgress, setV2ImportParseProgress] = useState<{ done: number; total: number } | null>(null);
  const [v2ImportParsed, setV2ImportParsed] = useState<ParsedDataset | null>(null);
  const [v2ImportMaxSide, setV2ImportMaxSide] = useState<number | null>(null);
  const [v2ImportBusy, setV2ImportBusy] = useState(false);
  const [v2ImportProgress, setV2ImportProgress] = useState<{ done: number; total: number } | null>(null);
  // Mirrors v2ImportBusy for the v2Reset/ESC closures (which capture a stale
  // state value) so cancel can't delete the project mid-upload.
  const v2ImportBusyRef = useRef(false);
  // When the V2 onboarding was started from a Project page, the container id is
  // parked here so v2HandOff can add the freshly-created dataset to it.
  const pendingContainerRef = useRef<string | null>(null);
  // Reactive mirror of the parked container so the labels stage can offer a
  // Back/Cancel that returns to the project it was launched from (a ref alone
  // can't drive the render).
  const [v2ReturnProjectId, setV2ReturnProjectId] = useState<string | null>(null);
  // Workspace "+ Add Dataset" now collects the name via the same popup the
  // Project page uses (CreateDatasetModal) instead of an inline bar, then jumps
  // straight to the labels stage.
  const [showWorkspaceCreate, setShowWorkspaceCreate] = useState(false);
  // Reason string from the onboarding dataset-type classification,
  // stashed so v2HandOff can seed it into the project-meta cache
  // alongside the type (the type travels via firstLoad).
  const v2DatasetReasonRef = useRef<string>("");
  useEffect(() => { v2ProjectIdRef.current = v2ProjectId; }, [v2ProjectId]);

  // Persist reference box/label edits made in the onboarding editor to
  // the backend so they survive into the project. The reference was
  // already POSTed during upload (it carries a referenceId); this PUTs
  // the edited detections to /references/{referenceId}, which re-embeds
  // changed boxes server-side. Without this the edits lived only in
  // local state and the project re-hydrated the ORIGINAL auto-detected
  // boxes from the manifest, so the user's corrections were lost.
  const v2RefEditTimers = useRef<Map<string, number>>(new Map());
  const v2RefEditFlights = useRef<Set<Promise<void>>>(new Set());
  const v2FlushRefEdit = useCallback((referenceId: string, boxes: EditableBox[]) => {
    const projectId = v2ProjectIdRef.current;
    if (!projectId || !referenceId) return;
    const detections = (boxes ?? []).map((b) => ({
      label: b.label,
      score: b.score,
      box: [b.x0, b.y0, b.x1, b.y1],
      mask: b.mask ?? null,
    }));
    const p = apiFetch(
      `/api/v2/projects/${projectId}/references/${referenceId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detections }),
      },
    ).then(() => undefined).catch((e) => {
      console.warn("[v2 ref edit flush] PUT failed:", e);
    });
    v2RefEditFlights.current.add(p);
    void p.finally(() => v2RefEditFlights.current.delete(p));
  }, []);
  const v2ScheduleRefEdit = useCallback(
    (referenceId: string | undefined, boxes: EditableBox[]) => {
      if (!referenceId) return;
      const timers = v2RefEditTimers.current;
      const prev = timers.get(referenceId);
      if (prev) window.clearTimeout(prev);
      timers.set(referenceId, window.setTimeout(() => {
        timers.delete(referenceId);
        v2FlushRefEdit(referenceId, boxes);
      }, 500));
    },
    [v2FlushRefEdit],
  );
  // Whether the new project should be created as private. Toggle is
  // only rendered when the user's plan allows private projects (the
  // same gate ProjectSettingsV2 uses). Persisted into the manifest at
  // create time via the is_private form field on POST /api/v2/projects.
  const [v2IsPrivate, setV2IsPrivate] = useState(false);
  // Flat pool of reference images, up to V2_MAX_REFS total, shared
  // across all labels. Annotations per label happen in the project view.
  const [v2RefImages, setV2RefImages] = useState<ReferenceImage[]>([]);
  // URLs of reference images currently being processed by the pipeline.
  const [v2RefProcessing, setV2RefProcessing] = useState<Set<string>>(new Set());
  // Reference image currently open in the full-screen viewer (by
  // index so prev/next nav can move through the pool).
  const [v2ViewingIdx, setV2ViewingIdx] = useState<number | null>(null);
  const v2InputRef = useRef<HTMLInputElement | null>(null);
  const v2FileRef = useRef<HTMLInputElement | null>(null);

  // Stages that take over the workspace surface. While active the
  // search/add buttons fade out and the projects grid is hidden; the
  // page stays scrollable and the footer remains in flow so the
  // surface doesn't feel isolated.
  const v2Active = v2Stage === "labels" || v2Stage === "classifying" || v2Stage === "references" || v2Stage === "import";

  // Ambient word-cloud overlay is only shown during the labels stage.
  // We keep it mounted briefly after the user leaves the stage so the
  // exit fade-out can play before the DOM nodes are removed.
  const v2WordCloudShouldShow = v2Stage === "labels";
  const [v2WordCloudMounted, setV2WordCloudMounted] = useState(false);
  useEffect(() => {
    if (v2WordCloudShouldShow) {
      setV2WordCloudMounted(true);
      return;
    }
    if (!v2WordCloudMounted) return;
    // Match the longest transition in <WordCloud> (640 ms) plus a
    // little headroom so every word has cleared before we unmount.
    const t = window.setTimeout(() => setV2WordCloudMounted(false), 750);
    return () => window.clearTimeout(t);
  }, [v2WordCloudShouldShow, v2WordCloudMounted]);

  // "Created today" stamp shown beside the creator under the project
  // title. The project doesn't exist on the backend yet (we only
  // create it once onboarding finishes) so the display date is
  // simply the moment the user opened the V2 flow.
  const v2DateLabel = useMemo(
    () =>
      new Intl.DateTimeFormat("en-GB", {
        day: "numeric",
        month: "long",
        year: "numeric",
      }).format(new Date()),
    [],
  );

  // Esc closes whichever V2 stage is active and returns the page to
  // its normal layout. Mirrors how the V1 Cancel button feels.
  // Suppressed while the per-reference editor is open: ESC there
  // should just close the editor (handled by ReferenceImageEditor's
  // own onClose), not nuke the whole onboarding session.
  useEffect(() => {
    if (v2Stage === "idle") return;
    if (v2ViewingIdx !== null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") v2Reset();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Stage, v2ViewingIdx]);

  // Ctrl/Cmd+Enter advances the active V2 stage, triggers Done on
  // the labels stage and Open project on the references stage.
  // Deps include v2RefImages.length so the onKey closure picks up
  // the latest v2DoneRefs after the user uploads images, otherwise
  // Ctrl+Enter would fire with the empty snapshot from when the
  // stage first opened.
  useEffect(() => {
    if (v2Stage !== "labels" && v2Stage !== "references") return;
    if (v2ViewingIdx !== null) return;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
      e.preventDefault();
      if (v2Stage === "labels" && v2Labels.length > 0) v2DoneLabels();
      else if (v2Stage === "references") v2DoneRefs();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Stage, v2Labels.length, v2RefImages.length, v2ViewingIdx]);

  // Auto-focus the chip input when the labels stage opens.
  useEffect(() => {
    if (v2Stage !== "labels") return;
    const t = window.setTimeout(() => v2InputRef.current?.focus(), 320);
    return () => window.clearTimeout(t);
  }, [v2Stage]);

  // Revoke object URLs for any references the user picked but didn't
  // ship through to the project view. Runs only on full unmount.
  useEffect(() => {
    return () => {
      for (const r of v2RefImages) {
        try { URL.revokeObjectURL(r.preview); } catch { /* already revoked */ }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const v2Reset = () => {
    // An in-flight import is mid-upload to a real project; ignore cancel/ESC
    // until it finishes (the Cancel button is disabled too) so we never
    // delete the project out from under the upload.
    if (v2ImportBusyRef.current) return;
    // Cancelling clears any parked Project so the next normal create doesn't
    // get attached to it.
    pendingContainerRef.current = null;
    setV2ReturnProjectId(null);
    for (const r of v2RefImages) {
      try { URL.revokeObjectURL(r.preview); } catch { /* already revoked */ }
    }
    // Capture the eagerly-created project ID before we wipe local
    // state so we can DELETE it on the backend. Cancelling out of
    // the onboarding flow should NOT leave an orphaned project
    // sitting in the workspace, that was the previous behaviour
    // (it relied on a fictitious "orphan projects are cheap" claim).
    const cancelledProjectId = v2ProjectIdRef.current;
    setV2Stage("idle");
    setV2Name("");
    setV2Labels([]);
    setV2LabelColours({});
    setV2Input("");
    setV2RefImages([]);
    setV2ProjectId(null);
    v2ProjectIdRef.current = null;
    setV2ImportParsed(null);
    setV2ImportParsing(false);
    setV2ImportParseProgress(null);
    setV2ImportProgress(null);
    setV2ImportMaxSide(null);
    if (cancelledProjectId) {
      // Fire-and-forget. The user's already on the next screen by
      // the time the DELETE round-trip lands; surfacing an error
      // here would be more confusing than the silent failure of
      // leaving an extra row in the workspace list (which the next
      // poll picks up anyway if the delete didn't go through).
      apiFetch(`/api/projects/${cancelledProjectId}`, { method: "DELETE" })
        .catch((e) => console.warn("[v2 cancel] project delete failed:", e));
    }
  };

  const v2BeginLabels = (nameArg?: string) => {
    const nm = (nameArg ?? v2Name).trim();
    if (!nm) return;
    setError(null);
    if (containsProfanity(nm)) {
      setError(`"${nm}" can't be used as a project name.`);
      return;
    }
    // When the name came from elsewhere (e.g. the in-project create flow), make
    // sure it's in state for the rest of the onboarding (v2HandOff reads it).
    if (nameArg !== undefined) setV2Name(nm);
    setV2Stage("labels");
  };

  const v2AddLabel = (raw: string) => {
    const next = raw.trim().replace(/[,.]+$/g, "").trim();
    if (!next) return;
    if (v2Labels.some((l) => l.toLowerCase() === next.toLowerCase())) {
      setV2Input("");
      return;
    }
    // Client-side profanity guard. Mirrors the backend assert_clean
    // gate so the user sees an inline error instead of a 400 from
    // /api/v2/projects when they click Continue.
    const bad = containsProfanity(next);
    if (bad) {
      setError(`"${next}" can't be used as a label.`);
      return;
    }
    setError(null);
    // Random palette pick without repeating any currently-used colour.
    // Falls back to a random pick once the palette is saturated.
    const taken = new Set(Object.values(v2LabelColours));
    const free = LABEL_COLOURS.filter((c) => !taken.has(c));
    const pool = free.length > 0 ? free : LABEL_COLOURS;
    const colour = pool[Math.floor(Math.random() * pool.length)];
    setV2Labels([...v2Labels, next]);
    setV2LabelColours((m) => ({ ...m, [next.toLowerCase()]: colour }));
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

  // Eagerly create the V2 project on the backend and stash its ID so
  // each reference upload during onboarding can POST directly to
  // /api/v2/projects/{id}/references, single round-trip per image
  // instead of the old /process → /references two-hit flow.
  // Idempotent: returns the existing ID if we've already created
  // one in this onboarding session.
  const v2EnsureProject = async (
    name: string,
    labels: string[],
  ): Promise<string | null> => {
    if (v2ProjectIdRef.current) return v2ProjectIdRef.current;
    try {
      const fd = new FormData();
      fd.append("name", name.trim() || "Untitled");
      fd.append("labels", JSON.stringify(labels));
      // Colours captured during the labels stage so the project view
      // keeps the swatches the user previewed in onboarding.
      fd.append("label_colours", JSON.stringify(v2LabelColours));
      // Visibility, only sent when the user actually toggled the
      // private switch (which is itself gated on a Pro / Mega plan).
      // Backend defaults to public when the field is missing.
      if (v2IsPrivate) fd.append("is_private", "true");
      fd.append("owner", username || "anonymous");
      const r = await apiFetch(`/api/v2/projects`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = (await r.json()) as { project_id: string };
      const pid = data.project_id;
      setV2ProjectId(pid);
      v2ProjectIdRef.current = pid;
      capture("project_create", { labels: labels.length });
      // eslint-disable-next-line no-console
      console.log("[v2 ensure-project] created:", pid);
      return pid;
    } catch (e) {
      console.error("[v2 ensure-project]", e);
      return null;
    }
  };

  // ── Import a labelled dataset ────────────────────────────────────────
  // Switch from the labels stage into the import stage. The project name is
  // already set (create modal → v2BeginLabels); the dataset's class list
  // replaces the typed labels.
  const v2BeginImport = () => {
    setError(null);
    setV2ImportParsed(null);
    setV2ImportParseProgress(null);
    setV2ImportProgress(null);
    setV2Stage("import");
  };

  // Parse a picked folder/zip into a dataset preview (no upload yet).
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
          "That .zip is very large — unzipping it in the browser may run out of memory. Use the folder picker for big datasets.",
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

  // Create the project (labels = the dataset's classes) and stream the
  // images + their boxes up via the batched ingest endpoint, then open it.
  const v2RunImport = async () => {
    const parsed = v2ImportParsed;
    if (!parsed || parsed.items.length === 0 || v2ImportBusyRef.current) return;
    setError(null);
    v2ImportBusyRef.current = true;
    setV2ImportBusy(true);
    setV2ImportProgress({ done: 0, total: parsed.items.length });
    try {
      const projectId = await v2EnsureProject(v2Name, parsed.classes);
      if (!projectId) throw new Error("Couldn't create the project.");
      const result = await uploadDataset(
        parsed.items,
        { maxSide: v2ImportMaxSide },
        (form) =>
          apiFetch(`/api/v2/projects/${projectId}/imports/raw_batch`, { method: "POST", body: form }),
        (p) => setV2ImportProgress({ done: p.done, total: p.total }),
      );
      capture("dataset_import", {
        images: result.done,
        failed: result.failed,
        classes: parsed.classes.length,
        format: parsed.format,
      });
      if (result.done === 0) {
        throw new Error(
          result.errors[0] ? `Import failed: ${result.errors[0]}` : "No images were imported.",
        );
      }
      // Open the new project. Labels = the dataset's classes; no references
      // and no dataset-type seed (imports skip onboarding classification).
      if (onV2Begin) onV2Begin(v2Name.trim(), parsed.classes, [], projectId, null);
      // Reset onboarding state (mirrors v2HandOff's tail).
      v2ImportBusyRef.current = false;
      setV2ImportBusy(false);
      setV2Stage("idle");
      setV2Name("");
      setV2Labels([]);
      setV2LabelColours({});
      setV2Input("");
      setV2IsPrivate(false);
      setV2ProjectId(null);
      v2ProjectIdRef.current = null;
      setV2ImportParsed(null);
      setV2ImportProgress(null);
      setV2ImportMaxSide(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed.");
      // Clean up the eagerly-created project so a failed import never leaves an
      // orphan, and clear the project ref so a retry creates a FRESH project
      // instead of re-uploading into the half-filled one. Mirrors v2Reset's
      // orphan delete; the parsed dataset is kept so the user can just retry.
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

  // Hand the gathered name + labels + reference images off to the
  // parent. The V2 project was created eagerly when the user
  // entered the references stage, so this just transitions UI ,
  // every accepted reference is already saved on the server with
  // its detections + embeddings persisted to the manifest.
  // For projects where the user skipped the references stage we
  // still need a backend project record, so v2EnsureProject runs
  // here as a fallback when v2ProjectId is null.
  const v2HandOff = async (
    name: string,
    labels: string[],
    images: ReferenceImage[],
    firstLoad: "general" | "specific" | null = null,
  ) => {
    if (!onV2Begin) return;
    // eslint-disable-next-line no-console
    console.log("[v2 handoff] start ,", { name, labels, imageCount: images.length, firstLoad });
    let projectId = v2ProjectIdRef.current;
    if (!projectId) projectId = await v2EnsureProject(name, labels);
    // If this onboarding was launched from a Project page, add the new dataset
    // to that Project (container) so it lands inside it, not standalone.
    if (projectId && pendingContainerRef.current) {
      try {
        await addDataset(pendingContainerRef.current, projectId);
      } catch (e) {
        console.error("[v2 handoff] add-to-container failed:", e);
      }
      pendingContainerRef.current = null;
    }
    // Flush any pending reference-edit PUTs NOW (clear debounce timers
    // and fire immediately) and wait for them to land, so the project
    // view hydrates from a manifest that already reflects the user's
    // box/label corrections instead of the original auto-detections.
    for (const [refId, t] of Array.from(v2RefEditTimers.current.entries())) {
      window.clearTimeout(t);
      v2RefEditTimers.current.delete(refId);
      const ref = images.find((r) => r.referenceId === refId);
      if (ref) v2FlushRefEdit(refId, ref.boxes ?? []);
    }
    if (v2RefEditFlights.current.size > 0) {
      await Promise.allSettled(Array.from(v2RefEditFlights.current));
    }
    // Seed the dataset-type classification we already computed during
    // onboarding into the project-meta cache, keyed by the new project
    // id. ProjectViewV2Stub's state initialiser reads this on mount, so
    // the general/specific pill and the references-section mode paint on
    // the first frame instead of waiting on a fresh (and redundant)
    // classification round-trip to /dataset-type. firstLoad carries the
    // type; null means the user skipped labels, so there's nothing to seed.
    if (projectId && firstLoad) {
      patchProjectMeta(projectId, {
        datasetType: {
          type: firstLoad,
          reason: v2DatasetReasonRef.current || null,
          source: "auto",
        },
      });
    }
    // eslint-disable-next-line no-console
    console.log("[v2 handoff] calling onV2Begin with", images.length, "images, projectId =", projectId);
    onV2Begin(name.trim(), labels, images, projectId, firstLoad);
    setV2Stage("idle");
    setV2Name("");
    setV2Labels([]);
    setV2LabelColours({});
    setV2Input("");
    setV2RefImages([]);
    setV2IsPrivate(false);
    setV2ProjectId(null);
    v2ProjectIdRef.current = null;
    v2DatasetReasonRef.current = "";
  };

  const v2DoneLabels = () => {
    if (v2Labels.length === 0) { v2HandOff(v2Name, [], []); return; }
    // Reset the loader copy back to the entry message every time
    // we re-enter classifying (e.g. user backed out of references
    // and pressed Done again).
    setV2ClassifyMsg("Reading between the labels…");
    // Step into the classifying stage; the useEffect below pings
    // /api/v2/dataset-type/preview and routes to either references
    // (specific) or straight to the project (general).
    setV2Stage("classifying");
    // Fire-and-forget: backend project creation runs in parallel
    // with classification so the project record is ready by the
    // time we land on either references or the project view.
    v2EnsureProject(v2Name, v2Labels);
  };

  // Dataset-type classifier, runs once when the stage flips into
  // "classifying". General label sets skip the references stage and
  // go straight to the project view. Specific ones go through
  // references as before. Network errors fall back to "specific" so
  // the user still gets a chance to upload references rather than
  // landing on a project page they didn't expect.
  useEffect(() => {
    if (v2Stage !== "classifying") return;
    let cancelled = false;
    (async () => {
      let datasetType: "general" | "specific" = "specific";
      v2DatasetReasonRef.current = "";
      try {
        const r = await fetch(`${API}/api/v2/dataset-type/preview`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ labels: v2Labels }),
        });
        if (r.ok) {
          const d = await r.json();
          if (d?.type === "general") datasetType = "general";
          if (typeof d?.reason === "string") v2DatasetReasonRef.current = d.reason;
        }
      } catch {
        // Keep default ("specific") so user lands on references.
      }
      if (cancelled) return;
      if (datasetType === "general") {
        // Skip references entirely, hand off with no images. Switch
        // the loader copy first so the same PixelKit animation
        // continues straight through to the project mount with
        // "Opening project…", the ProjectViewV2Stub then renders
        // its in-page first-load loader (same animation, project
        // chrome visible behind) until /initial lands.
        setV2ClassifyMsg("Opening project…");
        v2HandOff(v2Name, v2Labels, [], "general");
      } else {
        setV2Stage("references");
      }
    })();
    return () => {
      cancelled = true;
    };
    // v2HandOff and v2Name/v2Labels are read at fire time; we only
    // want this effect to run when the stage flips to classifying.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Stage]);

  const v2SkipFromLabels = () => v2HandOff(v2Name, [], []);
  const v2SkipFromRefs   = () => v2HandOff(v2Name, v2Labels, [], "specific");
  // Refs Done = "specific" path. Pass "specific" so the project
  // view renders the smaller in-page "Loading project…" loader card
  // instead of the full-screen mount loader (per design intent: the
  // user is moving from a busy refs page to their new project, and
  // the smaller loader keeps the page chrome in view).
  const v2DoneRefs       = () => v2HandOff(v2Name, v2Labels, v2RefImages, "specific");

  const v2Remaining = Math.max(0, V2_MAX_REFS - v2RefImages.length);
  const [v2RefDragOver, setV2RefDragOver] = useState(false);
  // Active filter for the references grid during onboarding ,
  // mirrors the same chip row pattern on the V2 project page.
  // null = show all.
  const [v2FilterLabel, setV2FilterLabel] = useState<string | null>(null);

  // Single-shot reference upload: POST directly to
  // /api/v2/projects/{id}/references with the bytes + labels and let
  // the backend run GD+SAM, save the file to disk, embed each box
  // with the embedding model, and stamp everything into the manifest in one
  // round-trip. The endpoint echoes back detections so we can render
  // boxes immediately without a follow-up manifest GET.
  //
  // Falls back to the legacy stateless /process endpoint if the
  // backend project hasn't been created yet (shouldn't happen, we
  // call v2EnsureProject on stage entry, but the network race
  // between "user dragged 20 files" and "POST /api/v2/projects
  // returned" is real, and stateless /process is a fine cushion).
  const v2TriggerPipeline = async (ref: ReferenceImage) => {
    setV2RefProcessing((prev) => new Set([...prev, ref.preview]));
    try {
      const projectId = v2ProjectIdRef.current ?? (await v2EnsureProject(v2Name, v2Labels));
      if (projectId) {
        const fd = new FormData();
        fd.append("image", ref.file);
        fd.append("labels", JSON.stringify(v2Labels));
        const r = await apiFetch(`/api/v2/projects/${projectId}/references`, {
          method: "POST",
          body: fd,
        });
        if (!r.ok) {
          const body = await r.text().catch(() => "");
          throw new Error(`http ${r.status}, ${body || "no body"}`);
        }
        const data = (await r.json()) as {
          reference_id: string;
          filename: string;
          width: number;
          height: number;
          detections: { label: string; score: number; box: number[]; mask: MaskShape | null }[];
        };
        const boxes = detectionsToBoxes(
          (data.detections ?? []).map((d) => ({
            label: d.label,
            score: d.score,
            box_xyxy: d.box,
            mask: d.mask,
          })),
          v2Labels,
        );
        setV2RefImages((cur) => {
          const stillExists = cur.some((it) => it.preview === ref.preview);
          if (!stillExists) {
            // User removed this ref while the pipeline was in-flight —
            // the POST already landed on the server, so delete it to
            // avoid an orphaned manifest entry.
            void apiFetch(
              `/api/v2/projects/${projectId}/references/${data.reference_id}`,
              { method: "DELETE" },
            ).catch(() => {});
            return cur;
          }
          return cur.map((it) => {
            if (it.preview !== ref.preview) return it;
            // Tag the ref with its server-side ID + filename so the
            // post-onboarding background uploader skips it (no
            // duplicate POST) and the editor's edit-flush PUT can
            // target /api/v2/projects/{id}/references/{referenceId}.
            const tagged = {
              ...it,
              width: data.width,
              height: data.height,
              referenceId: data.reference_id,
              filename: data.filename,
            };
            // Don't clobber boxes if the user already drew one before
            // the pipeline returned (same guard as before).
            return it.boxes !== undefined ? tagged : { ...tagged, boxes };
          });
        });
        return;
      }

      // Fallback path, project creation hasn't completed. Hit the
      // stateless /process so the user at least sees boxes; the
      // background uploader will persist the ref later.
      const fd = new FormData();
      fd.append("image", ref.file);
      fd.append("labels", JSON.stringify(v2Labels));
      const r = await fetch(`${API}/api/v2/references/process`, { method: "POST", body: fd });
      if (!r.ok) {
        const body = await r.text().catch(() => "");
        throw new Error(`http ${r.status}, ${body || "no body"}`);
      }
      const data = (await r.json()) as {
        width: number;
        height: number;
        detections: { label: string; score: number; box: number[]; mask: MaskShape | null }[];
      };
      const boxes = detectionsToBoxes(
        data.detections.map((d) => ({
          label: d.label,
          score: d.score,
          box_xyxy: d.box,
          mask: d.mask,
        })),
        v2Labels,
      );
      setV2RefImages((cur) =>
        cur.map((it) => {
          if (it.preview !== ref.preview) return it;
          if (it.boxes !== undefined) {
            return { ...it, width: data.width, height: data.height };
          }
          return { ...it, width: data.width, height: data.height, boxes };
        }),
      );
    } catch (e) {
      console.error("[v2 ref pipeline]", e);
    } finally {
      setV2RefProcessing((prev) => {
        const next = new Set(prev);
        next.delete(ref.preview);
        return next;
      });
    }
  };

  const v2OnUploadFiles = async (files: FileList | null) => {
    // eslint-disable-next-line no-console
    console.log("[v2 upload] called with", files?.length ?? 0, "file(s), remaining slots:", v2Remaining);
    if (!files || files.length === 0) return;
    // Match V1's upload pipeline: every accepted image is downsized
    // to 1500px on its longest edge and recompressed under 500 KB
    // JPEG before it leaves the browser. Saves home-network upload
    // time, R2 storage, and tunnel bandwidth, and keeps reference
    // sizes consistent with the import dataset (so the embedder sees the
    // same scale of crop bytes either way).
    // Accept files by extension when `file.type` is empty, Safari
    // hands drag-source files without a MIME in several flows.
    const candidates = Array.from(files).filter(isImageFile).slice(0, v2Remaining);
    if (candidates.length === 0) {
      if (v2FileRef.current) v2FileRef.current.value = "";
      return;
    }
    // Phase 1: spawn placeholder tiles IMMEDIATELY from the original
    // files (createObjectURL is instant, no decode/resize) so the grid
    // fills the moment the user drops — previously we awaited every
    // resize before showing anything, so 20 images took 5-10s to appear.
    const placeholders: ReferenceImage[] = candidates.map((f) => ({
      file: f,
      preview: URL.createObjectURL(f),
    }));
    setV2RefImages((cur) => [...cur, ...placeholders]);
    if (v2FileRef.current) v2FileRef.current.value = "";
    // Phase 2: resize each in the background and swap in the downsized
    // File (preview/key stay stable so tiles don't remount/flicker),
    // then kick the per-ref pipeline on the resized bytes.
    placeholders.forEach((ph) => {
      void (async () => {
        const resized = await resizeForUpload(ph.file).catch(() => ph.file);
        const ref: ReferenceImage = resized === ph.file ? ph : { ...ph, file: resized };
        if (ref !== ph) {
          setV2RefImages((cur) => cur.map((it) => (it.preview === ph.preview ? ref : it)));
        }
        v2TriggerPipeline(ref);
      })();
    });
  };

  const v2RemoveRef = (idx: number) => {
    const list = v2RefImages.slice();
    const [gone] = list.splice(idx, 1);
    if (gone) URL.revokeObjectURL(gone.preview);
    setV2RefImages(list);
    // If the ref was already uploaded to the server (has a referenceId),
    // delete it so it doesn't resurface on the next /initial hydration.
    if (gone?.referenceId && v2ProjectIdRef.current) {
      void apiFetch(
        `/api/v2/projects/${v2ProjectIdRef.current}/references/${gone.referenceId}`,
        { method: "DELETE" },
      ).catch(() => {});
    }
  };

  const planData = usePlan();
  // Free-tier projects are always public, surface the consequence
  // right where the user is about to create one, before they upload
  // anything sensitive.
  const isFreePlan = planData?.plan === "free";
  // Pro, Mega, and Beta get the private-on-create toggle in the
  // labels stage. Same gate ProjectSettingsV2 applies to its
  // visibility switch, so the two surfaces stay consistent. Beta
  // users keep parity with Pro for the duration of their window.
  const v2CanPrivate =
    (planData?.plan ? isProPlan(planData.plan) : false)
    || planData?.plan === "mega"
    || planData?.plan === "beta"
    || planData?.plan === "enterprise";

  // A labelled dataset is the user's own (often sensitive) data, so default the
  // import to private for anyone whose plan allows it the moment they enter the
  // import stage (free users stay public — the backend forces it). Kept as an
  // effect rather than inline in v2BeginImport so it can read v2CanPrivate,
  // which is defined here, after that handler.
  useEffect(() => {
    if (v2Stage === "import") setV2IsPrivate(v2CanPrivate);
  }, [v2Stage, v2CanPrivate]);

  // Optimistic patch from a `pixelkit-project-meta-changed` event the
  // project view dispatches the moment a label save / rename PUT
  // returns OK. Without this the workspace card waited for the next
  // /api/projects poll (~4 s) before the rename showed up.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        projectId?: string;
        tags?: string[];
        label_aliases?: Record<string, string>;
        labelColours?: Record<string, string>;
        name?: string;
        private?: boolean;
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
                private: typeof detail.private === "boolean" ? detail.private : p.private,
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

  // Single source of truth for favourites: the backend manifest's
  // `favouritedBy` list. Both Workspace and Projects pages use the same
  // toggle endpoint so a project favourited in one is favourited in both.
  const toggleFavourite = async (project_id: string) => {
    if (!username) return;
    let optimistic = false;
    setProjects((prev) => {
      if (!prev) return prev;
      return prev.map((p) => {
        if (p.id !== project_id) return p;
        optimistic = !p.favouritedByMe;
        return { ...p, favouritedByMe: optimistic };
      });
    });
    // Park the optimistic target so the next /api/projects poll
    // doesn't undo it before our POST returns. Map key is the project
    // id; value is the target favouritedByMe we just applied.
    pendingFavouriteIdsRef.current.set(project_id, optimistic);
    try {
      const r = await fetch(`${API}/api/projects/${project_id}/favourite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: username }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data: { favourites: number; favouritedByMe: boolean } = await r.json();
      setProjects((prev) =>
        prev?.map((p) =>
          p.id === project_id
            ? { ...p, favouritedByMe: data.favouritedByMe, favourites: data.favourites }
            : p,
        ) ?? null,
      );
    } catch {
      // Revert by next refresh.
      refresh();
    } finally {
      pendingFavouriteIdsRef.current.delete(project_id);
    }
  };

  const refresh = async () => {
    try {
      // Paginated to match the public projects page. Limit covers
      // currently-loaded projects so the periodic poll keeps every
      // loaded card in sync without re-pulling the user's whole
      // workspace on every tick.
      // projectsLenRef is the LIVE pagination cursor, reading
      // projects?.length here directly would close over the value
      // at this function's creation time, and the setInterval below
      // keeps invoking that stale closure forever, so paginated
      // pages got clobbered on every poll.
      const currentCount = projectsLenRef.current;
      const limit = Math.max(WORKSPACE_PAGE_SIZE, currentCount);
      const qParam = queryRef.current ? `&q=${encodeURIComponent(queryRef.current)}` : "";
      const url = `${API}/api/projects?owner=${encodeURIComponent(username)}&viewer=${encodeURIComponent(username)}&offset=0&limit=${limit}${qParam}`;
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const payload = await r.json() as
        | ProjectSummary[]
        | { total: number; items: ProjectSummary[]; offset?: number; limit?: number };
      // Pagination form returns {total, items}; back-compat shape is
      // a bare array. Either way, normalise to (data, totalCount).
      const data: ProjectSummary[] = Array.isArray(payload)
        ? payload
        : (payload.items ?? []);
      const totalCount: number = Array.isArray(payload)
        ? data.length
        : (payload.total ?? data.length);
      setTotal(totalCount);
      // Strip any projects the user has just deleted but the backend
      // still lists (in-flight delete). Once the backend stops listing
      // them, drop them from the pending set.
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
      // Respect in-flight favourite toggles. If the user just
      // optimistically favourited a project but the POST hasn't
      // returned yet, the server's listing still reports the old
      // state, patch it back to the user's target so the row
      // doesn't flicker (favourite → top → un-favourite → bottom
      // → favourite-again) while the toggle round-trips.
      if (pendingFavouriteIdsRef.current.size > 0) {
        const map = pendingFavouriteIdsRef.current;
        visible = visible.map((p) =>
          map.has(p.id)
            ? { ...p, favouritedByMe: map.get(p.id)! }
            : p,
        );
      }
      // Skip the state update when nothing the user can see has
      // actually changed. Without this, every 4-second poll forces
      // a re-render even when the response is byte-identical to the
      // last one, which (a) flickers the progressive-reveal counter
      // and (b) makes the search filter feel like it's "reloading"
      // the project list on every tick.
      setProjects((prev) => (projectsEqual(prev, visible) ? prev : visible));
      // localStorage + metaCache writes are gated on an empty query ,
      // when search is active the response is a filtered subset, not
      // the full workspace, so persisting it would corrupt the cold-
      // load cache and evict meta for every non-matching project.
      const searching = queryRef.current !== "";
      if (!searching) {
        try {
          window.localStorage.setItem("pixelkit_workspace_projects_v1", JSON.stringify(data));
        } catch {
          /* quota, keeps working from in-memory state */
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
            v2: !!p.v2,
            nImages: p.n_images,
            nReferences: p.n_references ?? 0,
            labelAliases: p.label_aliases ?? prevEntry.labelAliases,
            labelColours: p.labelColours ?? prevEntry.labelColours,
            private: typeof p.private === "boolean" ? p.private : prevEntry.private,
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
  }, [username]);

  // Debounce the search input so we don't fire a /api/projects fetch
  // on every keystroke. 200 ms feels instant but coalesces fast typing.
  useEffect(() => {
    const t = window.setTimeout(() => {
      setDebouncedQuery(searchQuery.trim());
    }, 200);
    return () => window.clearTimeout(t);
  }, [searchQuery]);

  // Refetch with the new query when it lands. Resets the loaded slice
  // so old results don't linger underneath the new ones. The poll
  // (4 s) picks up the same `q` via queryRef on subsequent ticks.
  // Skipped on first mount so the localStorage-warmed projects state
  // isn't clobbered before the initial refresh fills it in.
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

  // Sentinel-driven infinite scroll. Mirrors the public projects
  // page pattern: when the bottom-of-list anchor scrolls into view
  // (with a 300 px rootMargin), pull the next page of the user's
  // workspace projects and append. Skips when there's nothing left
  // to load or a fetch is already in flight, and pauses while the
  // search input is non-empty so the filter applies to "loaded so
  // far" rather than triggering more network round-trips per
  // keystroke.
  // Pull the next workspace page. Extracted so both the
  // IntersectionObserver below AND the scroll-near-bottom fallback can
  // call it without duplicating the fetch logic.
  const loadMoreWorkspacePage = useCallback(async (
    currentLen: number,
    currentTotal: number,
  ) => {
    if (loadingMore) return;
    setLoadingMore(true);
    try {
      const offset = currentLen;
      const qParam = queryRef.current ? `&q=${encodeURIComponent(queryRef.current)}` : "";
      const url = `${API}/api/projects?owner=${encodeURIComponent(username)}&viewer=${encodeURIComponent(username)}&offset=${offset}&limit=${WORKSPACE_PAGE_SIZE}${qParam}`;
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
  }, [loadingMore, username]);

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
  }, [projects, total, loadingMore, username, loadMoreWorkspacePage]);

  // Scroll-event fallback. The IntersectionObserver gets occasionally
  // stuck after a back-navigation from a project view, the sentinel
  // ref repaints to a fresh DOM node but the effect's deps haven't
  // changed, so the OLD observer is left observing a detached node
  // and never fires. Listening to scroll lets the user keep
  // paginating by reaching for the bottom of the list even when the
  // observer hasn't re-armed. Cheap: a single math check per scroll
  // tick. Same gates as the observer (search empty, more to load,
  // not already loading).
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

  // Keep projectsLenRef in sync with the current projects state so
  // the polling refresh closure below reads the live pagination
  // cursor instead of the stale one captured at first render.
  useEffect(() => {
    projectsLenRef.current = projects?.length ?? 0;
  }, [projects]);

  // Light poll so the workspace badges (In progress / Partial /
  // Unlabelled / Labelled) reflect job state without forcing the user
  // to reload. Kept on the slow side, the badges aren't latency-
  // critical and the manifest list is the heaviest endpoint.
  useEffect(() => {
    const id = window.setInterval(() => {
      refresh();
    }, 4000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username]);

  const sortedProjects = useMemo(() => {
    if (!projects) return null;
    // Text filtering is server-side via the `q` param; the list we get
    // back is already the matching set across every project in the
    // user's workspace, not just the loaded slice. The Starred tab
    // narrows that further client-side, then we sort by the chosen
    // mode (favourites still float to the top on the default mode so
    // the user's pinned projects stay where they expect them).
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

  // Starred count for the tab chip. Counts the loaded list (favourite
  // state is always present on every loaded card) so the number tracks
  // the user's toggles immediately.
  const starredCount = useMemo(
    () => (projects ?? []).filter((p) => p.favouritedByMe).length,
    [projects],
  );

  // Progressive card reveal: instead of mounting every card at once
  // (which makes every cover-image GET fire in the same tick and
  // chokes on the browser's ~6 concurrent-connection limit), we mount
  // them one at a time on a short timer. Each card's image starts
  // fetching as it mounts, so the loads stagger naturally, the
  // user sees cards filling in instead of a long blank wait.
  //
  // The stagger ONLY applies on the initial cold load. Once it's
  // ramped up to the full project count we never reset it on a
  // search-query change, search results should appear instantly,
  // not re-stagger from zero on every keystroke (which is what was
  // making search feel laggy). When search is active we also
  // bypass the slice entirely so the full filtered set renders the
  // moment the user types.
  // Seed visibleCount from the cached project count on first
  // render: when the user comes back to workspace from a project
  // page, the localStorage hydration in the `projects` initialiser
  // has already filled `sortedProjects` synchronously, so every
  // card should mount in the same tick instead of one-per-70ms.
  // Cold loads (no cache) start at 0 and ramp up via the stagger
  // below, which still serves its original purpose of spreading
  // cover-image GETs across the browser's connection limit.
  const [visibleCount, setVisibleCount] = useState<number>(
    () => projects?.length ?? 0,
  );
  const projectCount = sortedProjects?.length ?? 0;
  useEffect(() => {
    if (visibleCount >= projectCount) return;
    const t = window.setTimeout(() => {
      setVisibleCount((v) => Math.min(v + 1, projectCount));
    }, 70);
    return () => window.clearTimeout(t);
  }, [visibleCount, projectCount]);
  // While the user is typing in the search box, ignore the staggered
  // reveal entirely, render the full match set right now. Cleared
  // search reverts to whatever the cold-load reveal had reached.
  const isSearching = searchQuery.trim().length > 0;
  const renderCount = isSearching ? projectCount : visibleCount;

  const create = async () => {
    if (!newName.trim()) return;
    setError(null);
    const bad = containsProfanity(newName);
    if (bad) {
      // Block client-side too so the user gets instant feedback
      // instead of a round-trip and a generic backend 400.
      setError(`"${newName.trim()}" can't be used as a project name.`);
      return;
    }
    // Plan-quota gate. If the user is already at their project limit,
    // refuse before hitting the backend so they get an instant,
    // specific error message instead of an opaque 200 + soft cap.
    try {
      const usageR = await fetch("/api/users/usage", { cache: "no-store" });
      if (usageR.ok) {
        const usage = (await usageR.json()) as {
          plan: string;
          planName: string;
          limits: { projects: number };
          usage: { projects: number };
        };
        if (usage.usage.projects >= usage.limits.projects) {
          setError(
            `You've hit your ${usage.planName} plan limit of ${usage.limits.projects} projects. Upgrade or delete a project to create another.`,
          );
          return;
        }
      }
    } catch {
      // Usage endpoint down, fall through and let the backend create
      // the project. Better to over-permit than wedge the user out.
    }
    try {
      const r = await apiFetch(`/api/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName, owner: username }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = await r.json();
      onOpen(data.id, username, newName);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const toggleLike = async (project_id: string) => {
    if (!username) return;
    // Optimistic flip so the heart feels instant.
    setProjects((prev) =>
      prev?.map((p) =>
        p.id === project_id
          ? { ...p, likedByMe: !p.likedByMe, likes: p.likes + (p.likedByMe ? -1 : 1) }
          : p,
      ) ?? null,
    );
    try {
      const r = await fetch(`${API}/api/projects/${project_id}/like`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: username }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data: { likes: number; likedByMe: boolean } = await r.json();
      setProjects((prev) =>
        prev?.map((p) => (p.id === project_id ? { ...p, likes: data.likes, likedByMe: data.likedByMe } : p)) ?? null,
      );
    } catch {
      // Snap back to truth on failure.
      refresh();
    }
  };

  const confirmDelete = async (project_id: string) => {
    // Guard against re-entry: if a delete is already in flight for
    // this project (the dialog can sometimes be re-opened against
    // a project that hasn't been pruned from a stale refresh yet),
    // ignore the second click.
    if (pendingDeleteIdsRef.current.has(project_id)) {
      setDeleteTarget(null);
      return;
    }
    // Optimistic strip so the card disappears immediately. Without
    // this, a slow R2 delete or shutil.rmtree on Windows leaves the
    // dialog "Deleting…" while the list still shows the project, and
    // a subsequent refresh that lands before the backend finishes
    // can re-list the very project we're trying to delete.
    pendingDeleteIdsRef.current.add(project_id);
    const previous = projects;
    setProjects((prev) => (prev ? prev.filter((p) => p.id !== project_id) : prev));
    setDeleteTarget(null);
    try {
      const r = await apiFetch(`/api/projects/${project_id}`, { method: "DELETE" });
      // 404 = already gone (e.g. another tab deleted it). Treat as
      // success so a double-click doesn't surface a scary error.
      if (!r.ok && r.status !== 404) throw new Error(`http ${r.status}`);
      // Resync, server is the source of truth once the delete has
      // landed, and the optimistic strip might have raced an unrelated
      // upload or rename in another tab.
      await refresh();
    } catch (e) {
      // Restore the card on failure so the user sees the project is
      // still there and gets a clear error message.
      pendingDeleteIdsRef.current.delete(project_id);
      setProjects(previous);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Duplicate a dataset into a new project owned by the user. The backend
  // byte-copies the folder (images + labels) and returns the new id; we then
  // resync from the server so the copy appears in the workspace.
  const duplicateProject = async (p: ProjectSummary) => {
    if (duplicatingId) return;
    setDuplicatingId(p.id);
    setError(null);
    try {
      const r = await apiFetch(`/api/projects/${p.id}/duplicate`, { method: "POST" });
      if (!r.ok) throw new Error(`http ${r.status}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDuplicatingId(null);
    }
  };

  // One-page Project (container) view replaces the workspace when a Project card
  // is opened. TopNav lives in page.tsx, so this still renders under the nav.
  if (loggedIn && projectViewId) {
    return (
      <ProjectPage
        containerId={projectViewId}
        username={username}
        onBack={() => setProjectViewId(null)}
        onOpenDataset={(dsId) => {
          // Remember which Project this dataset was opened from so the dataset
          // view's back button reads "Back to project" and returns here.
          const fromProject = projectViewId ?? undefined;
          setProjectViewId(null);
          onOpen(dsId, "", "", true, fromProject);
        }}
        onNewDataset={(name) => {
          // The name was entered INSIDE the Project; park the container, leave
          // the project view, and jump straight into the onboarding LABELS
          // stage (skipping the workspace name form) so the new dataset lands
          // in this Project.
          pendingContainerRef.current = projectViewId;
          setV2ReturnProjectId(projectViewId);
          setV2Labels([]);
          setV2Input("");
          setV2RefImages([]);
          setProjectViewId(null);
          v2BeginLabels(name);
        }}
      />
    );
  }

  // (Logged-out marketing view removed — the portable build has no accounts.)

  return (
    <main className="min-h-screen bg-[var(--background)] overflow-x-hidden">
      {/* Title + controls strip. Stays mounted across all stages ,
          the H1 / subtitle / search-add buttons crossfade rather
          than reflow so the page never visibly jumps. `relative
          z-20` puts the title above the V2 wrapper, which uses a
          negative top margin to overlap with this band so the
          centred V2 body lands at exactly 50vh of the viewport. */}
      <section className="relative z-20 mx-auto max-w-6xl px-6 pt-16 pb-10 flex items-end justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          {/* Heading stack: "Workspace" → project name. The two H1s'
              transitions are sequenced, the leaving one fades out
              first, the arriving one fades in only after a matching
              delay, so they never appear on screen at the same time
              the way a simultaneous crossfade did. */}
          {/* The wrapper's `pb-2` gives the H1s an extra 0.5rem of
              box height below their line-box so descenders (g, y, p)
              don't get clipped by the absolute project-name H1's
              `truncate` (overflow: hidden). Both H1s also get
              `leading-[1.3]`, slightly looser than `leading-tight`
              (1.25), so the line-box itself accommodates the tail
              even on fonts with deeper descenders. */}
          <div className="relative pb-2">
            <h1
              className={[
                "text-5xl md:text-6xl font-medium tracking-tight leading-[1.3] transition-all duration-150 ease-out",
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
                "absolute inset-0 text-5xl md:text-6xl font-medium tracking-tight leading-[1.3] transition-all duration-150 ease-out truncate",
                v2Active
                  ? "opacity-100 translate-y-0 delay-150"
                  : "opacity-0 translate-y-4 pointer-events-none select-none",
              ].join(" ")}
            >
              {v2Name || " "}
            </h1>
          </div>
          {/* Subtitle stack, same sequencing rule. While V2 is active
              the workspace tagline is replaced by the project's
              creator + creation date so the user can read off the
              metadata for the project they just named. */}
          <div className="relative mt-4 max-w-2xl">
            <p
              className={[
                "text-foreground/50 text-lg transition-all duration-150 ease-out",
                v2Active
                  ? "opacity-0 -translate-y-3 pointer-events-none select-none"
                  : "opacity-100 translate-y-0 delay-150",
              ].join(" ")}
            >
              Label data. Train models. Deploy and optimise - everything in one place.
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
              {/* Avatar, same h-5 w-5 as the public project cards so
                  this line reads consistently with how owners appear
                  in the rest of the app. Falls back to a hue-based
                  monogram when the user has no profile picture. */}
              {userImage ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={userImage}
                  alt=""
                  className="h-5 w-5 rounded-full object-cover shrink-0"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span
                  className="h-5 w-5 rounded-full grid place-items-center text-[9px] font-semibold text-[var(--foreground)] shrink-0"
                  style={{
                    backgroundImage: `linear-gradient(135deg, hsl(${hueFor(username || "user")},70%,55%), hsl(${(hueFor(username || "user") + 60) % 360},70%,55%))`,
                  }}
                >
                  {(username || "?").charAt(0).toUpperCase()}
                </span>
              )}
              <span className="text-foreground/80">@{username || "you"}</span>
              <span aria-hidden className="text-foreground/25">·</span>
              <span className="tabular-nums">{v2DateLabel}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Workspace "+ Add Dataset" name popup — the exact same modal the
          Project page uses. On continue we jump straight to the labels stage
          (no Project to attach to, so the parked container + return id are
          cleared). */}
      <CreateDatasetModal
        open={showWorkspaceCreate}
        onClose={() => setShowWorkspaceCreate(false)}
        onContinue={(name) => {
          setShowWorkspaceCreate(false);
          pendingContainerRef.current = null;
          setV2ReturnProjectId(null);
          setV2Labels([]);
          setV2Input("");
          setV2RefImages([]);
          v2BeginLabels(name);
        }}
      />

      {/* V2 stage body. Labels stage uses the centred-prompt layout
          (negative top margin + flex items-center) so the question
          sits a touch above the viewport's mid-line. References
          stage drops the centring entirely, its content grows
          arbitrarily as the user adds reference images, and a
          flex-centered container would yank the heading above the
          viewport as the centroid shifts up with growing content
          (the page itself doesn't gain enough height to scroll
          because the wrapper's min-h is fixed at one viewport).
          Normal flow + a small top padding keeps the heading
          anchored and lets the page grow naturally. */}
      {v2Active && (
        <div
          className={[
            "relative px-6",
            // The classifying stage shares the same vertically-centred
            // layout as the labels stage so the swap is in-place, no
            // wrapper-level layout shift, no flicker between forms.
            v2Stage === "labels" || v2Stage === "classifying"
              ? "-mt-[13rem] min-h-[calc(100vh-6rem)] flex items-center justify-center pointer-events-none"
              : "pt-4 pb-16 pointer-events-none",
          ].join(" ")}
        >
          {/* Ambient word backdrop, only on the labels stage. The
              parent gate keeps it mounted briefly after the user
              leaves so the fade-out animation plays in full. */}
          {v2Stage === "labels" && v2WordCloudMounted && <WordCloud show={v2WordCloudShouldShow} />}
          {(v2Stage === "labels" || v2Stage === "classifying") && (
            <div
              key="v2-labels"
              className={[
                // Both the form and the classifying loader live in
                // this relative wrapper so the loader can absolute-
                // overlay the form and the two crossfade in place
                // when the user clicks Done.
                "relative w-full max-w-2xl pointer-events-auto",
                v2Stage === "labels" ? "animate-[fadeIn_280ms_ease-out]" : "",
              ].join(" ")}
            >
            {/* Labels form crossfades out as the loader fades in.
                Done is disabled while the dataset-type call is in
                flight so the user can't double-fire. */}
            <div
              key="v2-labels-form"
              className="transition-all duration-300 ease-out"
              style={{
                opacity: v2Stage === "labels" ? 1 : 0,
                transform: v2Stage === "labels" ? "translateY(0)" : "translateY(-6px)",
                pointerEvents: v2Stage === "labels" ? "auto" : "none",
              }}
            >
              {/* Back/cancel, only when this onboarding was launched from a
                  Project page. Abandons the in-progress dataset and returns to
                  the project the user came from. */}
              {v2ReturnProjectId && (
                <button
                  type="button"
                  onClick={() => {
                    const back = v2ReturnProjectId;
                    v2Reset();
                    if (back) setProjectViewId(back);
                  }}
                  className="mb-5 -ml-1 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-sm font-medium text-foreground/55 transition-colors hover:bg-foreground/[0.06] hover:text-foreground/90"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M15 18l-6-6 6-6" />
                  </svg>
                  Back to project
                </button>
              )}
              <h2 className="text-4xl md:text-5xl font-light tracking-tight text-[var(--foreground)] leading-tight">
                What do you want to detect?
              </h2>
              <p className="mt-3 text-sm text-foreground/50 leading-relaxed">
                Enter the objects you want to detect, you can add more later.
              </p>
              <button
                type="button"
                onClick={v2BeginImport}
                className="mt-2.5 inline-flex items-center gap-1.5 text-sm text-foreground/55 hover:text-foreground transition-colors"
              >
                Already have a labelled dataset?
                <span className="font-medium text-[#fb923c]">Import it →</span>
              </button>

              <div className="mt-10 rounded-2xl border border-foreground/10 bg-foreground/[0.03] focus-within:border-foreground/30 focus-within:bg-foreground/[0.05] transition-colors px-4 py-3.5 flex flex-wrap items-center gap-2">
                {v2Labels.map((lab, i) => {
                  const bg = v2LabelColours[lab.toLowerCase()] ?? LABEL_COLOURS[i % LABEL_COLOURS.length];
                  return (
                    <span
                      key={`${lab}-${i}`}
                      // Shrunk by default (just the label); on hover the chip
                      // grows and the remove (×) button expands in from zero
                      // width, so the chip visibly enlarges under the cursor.
                      className="group inline-flex items-center rounded-full pl-3 pr-3 group-hover:pr-1.5 h-7 text-sm font-medium animate-[fadeIn_180ms_ease-out] transition-[padding] duration-150 ease-out motion-reduce:transition-none"
                      style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                    >
                      <span className="select-none">{lab}</span>
                      <button
                        type="button"
                        onClick={() => v2RemoveLabel(i)}
                        aria-label={`Remove ${lab}`}
                        className="inline-flex h-5 w-0 shrink-0 items-center justify-center overflow-hidden rounded-full opacity-0 transition-all duration-150 ease-out group-hover:ml-1 group-hover:w-5 group-hover:opacity-100 hover:bg-black/20 motion-reduce:transition-none"
                        style={{ color: readableTextForBg(bg) }}
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
                  className="flex-1 min-w-[8rem] bg-transparent outline-none py-1 text-base text-[var(--foreground)] placeholder:text-foreground/35"
                />
              </div>

              <div className="mt-6 flex items-center gap-3">
                {/* Visibility toggle, Pro/Mega only. Its left edge
                    sits flush with the label input above (both live
                    inside the same w-full max-w-2xl container) and
                    its centreline lines up with Skip + Done thanks
                    to items-center on the row. */}
                {v2CanPrivate && (
                  <button
                    type="button"
                    role="switch"
                    aria-checked={v2IsPrivate}
                    onClick={() => setV2IsPrivate((p) => !p)}
                    className={[
                      "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-colors",
                      v2IsPrivate
                        ? "border-amber-500/50 bg-amber-300/[0.12] text-amber-800 dark:text-amber-100 hover:bg-amber-300/[0.18]"
                        : "border-foreground/20 bg-foreground/5 text-[var(--foreground)] hover:bg-foreground/10",
                    ].join(" ")}
                    title={v2IsPrivate ? "Private, only you will see this project" : "Public, visible in the community feed"}
                  >
                    <span
                      aria-hidden
                      className={[
                        "h-4 w-7 rounded-full p-0.5 transition-colors flex",
                        v2IsPrivate ? "bg-amber-500/80 justify-end" : "bg-foreground/25",
                      ].join(" ")}
                    >
                      <span className="h-3 w-3 rounded-full bg-background" />
                    </span>
                    {v2IsPrivate ? "Private" : "Public"}
                  </button>
                )}

                {/* Buttons sit on the right, spaced from the toggle by
                    an auto-margin so the toggle (or its empty slot)
                    stays pinned to the left while Skip + Done hold
                    the right edge. */}
                {/* Cancel discards the in-progress dataset and returns to the
                    workspace, matching the Cancel on the Project page's create
                    flow. (The toolbar's morphed Cancel sits behind this overlay,
                    so the labels stage needs its own.) */}
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
                  title="Skip, open the project without setting labels."
                >
                  Skip
                </button>
                <div className="relative">
                  <button
                    type="button"
                    onClick={v2DoneLabels}
                    disabled={v2Labels.length === 0 || v2Stage === "classifying"}
                    className="inline-flex items-center justify-center rounded-full px-7 py-2.5 text-sm font-semibold leading-none text-black transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{ backgroundColor: "#fb923c" }}
                  >
                    Done
                  </button>
                  {/* Ctrl ↵ hint sits OUTSIDE the flex flow so it
                      doesn't shift the Done button's centreline away
                      from Skip's. */}
                  <span className="pointer-events-none absolute right-0 top-full mt-1 text-[10px] text-foreground/30 font-mono select-none">
                    Ctrl ↵
                  </span>
                </div>
              </div>
            </div>

            {/* Classifying loader, absolute over the form's space.
                Crossfades with the form (form opacity 0 / loader
                fades in) so the swap reads as a single in-place
                transition rather than a layout shift. The message
                flips to "Loading project…" the moment the classifier
                lands on "general" so the same animation carries
                straight through to the project mount. */}
            {v2Stage === "classifying" && (
              <div
                key="v2-classifying"
                className="absolute inset-0 grid place-items-center pointer-events-auto animate-[fadeIn_320ms_ease-out]"
              >
                <PixelKitLoader size={128} message={v2ClassifyMsg} />
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
                className="inline-flex items-center gap-1.5 text-xs uppercase tracking-[0.18em] font-mono text-foreground/45 hover:text-foreground transition-colors mb-4 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <span aria-hidden>←</span>
                Back to labels
              </button>
              <h2 className="text-4xl md:text-5xl font-light tracking-tight text-[var(--foreground)] leading-tight">
                Import a labelled dataset
              </h2>
              <p className="mt-3 text-sm text-foreground/50 leading-relaxed">
                Pascal VOC — an <span className="font-mono">Annotations/</span> folder of XML next
                to your images — as a folder or a <span className="font-mono">.zip</span>. Every
                image and box comes in fully editable.
              </p>

              {error && <p className="mt-4 text-sm text-red-600 dark:text-red-400">{error}</p>}

              {/* Picker — folder or zip. Hidden once a dataset is parsed. */}
              {!v2ImportParsed && (
                <div className="mt-8 flex flex-wrap items-center gap-3">
                  <label className="cursor-pointer inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 hover:border-foreground/25 px-4 py-2.5 text-sm transition-colors">
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
                  <label className="cursor-pointer inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 hover:border-foreground/25 px-4 py-2.5 text-sm transition-colors">
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
                        ? ` — ${v2ImportParseProgress.done}/${v2ImportParseProgress.total}`
                        : "…"}
                    </span>
                  )}
                </div>
              )}

              {/* Preview + options once parsed. */}
              {v2ImportParsed && (
                <div className="mt-8 animate-[fadeIn_220ms_ease-out]">
                  <div className="rounded-2xl border border-foreground/10 bg-foreground/[0.03] p-5">
                    <div className="text-sm text-[var(--foreground)]">
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
                              className="inline-flex items-center rounded-full px-2.5 h-6 text-xs font-medium"
                              style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                            >
                              {c}
                            </span>
                          );
                        })}
                      </div>
                    )}
                    {v2ImportParsed.warnings.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {v2ImportParsed.warnings.map((w, i) => (
                          <li key={i} className="text-[12px] text-amber-700 dark:text-amber-300/85">
                            {w}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {/* Max image size — mirrors the project size dropdown; default
                      preserves full resolution (right for small-object data). */}
                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <label className="text-sm text-foreground/60" htmlFor="v2-import-size">
                      Image size
                    </label>
                    <select
                      id="v2-import-size"
                      value={v2ImportMaxSide ?? ""}
                      onChange={(e) =>
                        setV2ImportMaxSide(e.target.value === "" ? null : Number(e.target.value))
                      }
                      disabled={v2ImportBusy}
                      className="rounded-lg border border-foreground/15 bg-foreground/5 px-3 py-1.5 text-sm text-[var(--foreground)] outline-none focus:border-foreground/30 disabled:opacity-50"
                    >
                      {IMPORT_MAX_SIZE_OPTIONS.map((o) => (
                        <option key={o.label} value={o.value ?? ""}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                    <span className="text-[11px] text-foreground/40">
                      Full resolution preserves small objects; boxes are rescaled if you downsize.
                    </span>
                  </div>

                  {v2CanPrivate && (
                    <div className="mt-4 flex items-center gap-3">
                      <button
                        type="button"
                        role="switch"
                        aria-checked={v2IsPrivate}
                        onClick={() => setV2IsPrivate((p) => !p)}
                        disabled={v2ImportBusy}
                        className={[
                          "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-colors disabled:opacity-50",
                          v2IsPrivate
                            ? "border-amber-500/50 bg-amber-300/[0.12] text-amber-800 dark:text-amber-100 hover:bg-amber-300/[0.18]"
                            : "border-foreground/20 bg-foreground/5 text-[var(--foreground)] hover:bg-foreground/10",
                        ].join(" ")}
                        title={
                          v2IsPrivate
                            ? "Private — only you can see this project"
                            : "Public — visible in the community feed"
                        }
                      >
                        <span
                          aria-hidden
                          className={[
                            "h-4 w-7 rounded-full p-0.5 transition-colors flex",
                            v2IsPrivate ? "bg-amber-500/80 justify-end" : "bg-foreground/25",
                          ].join(" ")}
                        >
                          <span className="h-3 w-3 rounded-full bg-background" />
                        </span>
                        {v2IsPrivate ? "Private" : "Public"}
                      </button>
                      <span className="text-[11px] text-foreground/40">
                        Imported datasets default to private.
                      </span>
                    </div>
                  )}

                  <div className="mt-6 flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => { if (!v2ImportBusy) setV2ImportParsed(null); }}
                      disabled={v2ImportBusy}
                      className="inline-flex items-center justify-center px-4 py-2.5 text-sm leading-none text-foreground/55 hover:text-foreground/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Choose different
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
                      className="inline-flex items-center justify-center gap-2 rounded-full px-7 py-2.5 text-sm font-semibold leading-none text-black transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                      style={{ backgroundColor: "#fb923c" }}
                    >
                      {v2ImportBusy && (
                        <span className="inline-block h-3 w-3 rounded-full border-2 border-black/40 border-t-transparent animate-spin" />
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

          {v2Stage === "references" && (
            <div
              key="v2-refs"
              // mx-auto centres horizontally now that the parent
              // wrapper is using normal-flow layout instead of
              // flex justify-center.
              className="w-full max-w-2xl mx-auto animate-[fadeIn_280ms_ease-out] pointer-events-auto"
            >
              <button
                type="button"
                onClick={() => setV2Stage("labels")}
                className="inline-flex items-center gap-1.5 text-xs uppercase tracking-[0.18em] font-mono text-foreground/45 hover:text-foreground transition-colors mb-4"
              >
                <span aria-hidden>←</span>
                Back to labels
              </button>
              <h2 className="text-4xl md:text-5xl font-light tracking-tight text-[var(--foreground)] leading-tight">
                Reference images
              </h2>
              <p className="mt-3 text-sm text-foreground/50 leading-relaxed">
                Upload up to {V2_MAX_REFS} images containing your target objects.
              </p>
              <p className="mt-1.5 text-sm text-foreground/50 leading-relaxed">
                You&rsquo;ll annotate at least {V2_ANNOTS_PER_LABEL} examples per label directly in the project.{" "}
                <span className="text-foreground/70 italic">This step isn&rsquo;t required but is recommended.</span>
              </p>

              {/* Label chips, both the per-label progress display AND
                  the filter for the references grid below. Clicking a
                  chip filters the grid to images containing at least
                  one box of that label; clicking the active chip (or
                  "Show all") clears the filter. The chip's coloured
                  fill stays the visual anchor, with an extra ring
                  appearing when the chip is the active filter. */}
              {v2Labels.length > 0 && (
                <div className="mt-6 flex flex-wrap items-center gap-2">
                  {v2Labels.map((lab, i) => {
                    const annotCount = v2RefImages.reduce(
                      (sum, ri) => sum + (ri.boxes ?? []).filter((b) => b.label === lab).length,
                      0,
                    );
                    const annotDone = annotCount >= V2_ANNOTS_PER_LABEL;
                    const active = v2FilterLabel?.toLowerCase() === lab.toLowerCase();
                    const filterDisabled = v2RefImages.length === 0;
                    const bg = v2LabelColours[lab.toLowerCase()] ?? LABEL_COLOURS[i % LABEL_COLOURS.length];
                    return (
                      <button
                        key={lab}
                        type="button"
                        disabled={filterDisabled}
                        onClick={() => {
                          if (filterDisabled) return;
                          setV2FilterLabel((cur) =>
                            cur?.toLowerCase() === lab.toLowerCase() ? null : lab,
                          );
                        }}
                        aria-pressed={active}
                        className={[
                          "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium transition-all",
                          filterDisabled ? "cursor-default" : "cursor-pointer",
                          active ? "ring-2 ring-foreground/85 ring-offset-2 ring-offset-[var(--background)]" : "",
                        ].join(" ")}
                        style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                        title={
                          filterDisabled
                            ? `${annotCount}/${V2_ANNOTS_PER_LABEL} ${lab} annotations, upload references to enable filtering`
                            : active
                              ? `Showing only references containing "${lab}", click again to show all`
                              : `Show only references containing "${lab}"`
                        }
                      >
                        {lab}
                        {annotDone ? (
                          <svg viewBox="0 0 24 24" className="h-3 w-3 opacity-50 shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-label="Complete">
                            <polyline points="5 12 10 17 19 7" />
                          </svg>
                        ) : (
                          <span className="text-[10px] opacity-60 tabular-nums">{annotCount}/{V2_ANNOTS_PER_LABEL}</span>
                        )}
                      </button>
                    );
                  })}
                  {/* Show all sits after the label chips, only useful
                      once a filter is active or refs exist. Dim
                      border-only style so it never competes visually
                      with the coloured progress chips. */}
                  {v2RefImages.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setV2FilterLabel(null)}
                      aria-pressed={v2FilterLabel === null}
                      className={[
                        "rounded-full bg-[var(--background)] px-3 py-1 text-[11px] font-medium tracking-wide transition-all border",
                        v2FilterLabel === null
                          ? "border-foreground/15 text-foreground/40"
                          : "border-foreground/[0.06] text-foreground/25 hover:border-foreground/15 hover:text-foreground/45",
                      ].join(" ")}
                      title="Show every reference image"
                    >
                      Show all
                      <span className="ml-1.5 text-[10px] tabular-nums opacity-60">{v2RefImages.length}</span>
                    </button>
                  )}
                </div>
              )}

              {/* Shared image pool, single drop zone for all labels. */}
              <div className="mt-8">{/* (filter row removed, merged into the label-chip row above) */}
                {v2RefImages.length > 0 && (
                  <div className="mb-4 grid grid-cols-4 sm:grid-cols-5 gap-2">
                    {v2RefImages
                      .map((r, i) => ({ r, i }))
                      .filter(({ r }) => {
                        if (v2FilterLabel === null) return true;
                        const k = v2FilterLabel.toLowerCase();
                        return (r.boxes ?? []).some((b) => (b.label ?? "").toLowerCase() === k);
                      })
                      .map(({ r, i }) => {
                      const processing = v2RefProcessing.has(r.preview);
                      return (
                        <div
                          key={i}
                          className="group relative aspect-square rounded-lg overflow-hidden bg-foreground/5 border border-foreground/10 animate-[fadeIn_180ms_ease-out]"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={r.preview}
                            alt=""
                            className={[
                              "w-full h-full object-cover transition-all duration-300",
                              processing ? "opacity-35 scale-105 blur-[1px]" : "cursor-pointer",
                            ].join(" ")}
                            onClick={() => !processing && setV2ViewingIdx(i)}
                          />
                          {processing ? (
                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                              <svg className="h-7 w-7 animate-spin text-[var(--foreground)]" viewBox="0 0 24 24" fill="none">
                                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                              </svg>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={(e) => { e.stopPropagation(); v2RemoveRef(i); }}
                              className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-white hover:bg-black/90 grid place-items-center text-sm opacity-0 group-hover:opacity-100 transition-opacity"
                              aria-label="Remove reference"
                            >
                              ×
                            </button>
                          )}
                        </div>
                      );
                    })}
                    {v2FilterLabel !== null && v2RefImages.every((r) =>
                      !(r.boxes ?? []).some((b) => (b.label ?? "").toLowerCase() === v2FilterLabel.toLowerCase()),
                    ) && (
                      <div className="col-span-full rounded-lg border border-dashed border-foreground/10 px-4 py-6 text-center text-[12px] text-foreground/45">
                        No references contain &ldquo;{v2FilterLabel}&rdquo; yet.
                      </div>
                    )}
                  </div>
                )}
                <div
                  onClick={() => v2Remaining > 0 && v2FileRef.current?.click()}
                  // Safari refuses drops unless dragover both
                  // preventDefaults AND sets dataTransfer.dropEffect.
                  // stopPropagation defends against parent drag
                  // handlers swallowing the event.
                  onDragEnter={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (v2Remaining > 0) setV2RefDragOver(true);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setV2RefDragOver(false);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (v2Remaining > 0 && e.dataTransfer) e.dataTransfer.dropEffect = "copy";
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setV2RefDragOver(false);
                    if (v2Remaining > 0) v2OnUploadFiles(e.dataTransfer.files);
                  }}
                  className={[
                    "w-full rounded-2xl border-2 border-dashed transition-all p-6 text-center",
                    v2Remaining <= 0
                      ? "border-foreground/5 opacity-50 cursor-not-allowed"
                      : v2RefDragOver
                      ? "border-foreground/50 bg-foreground/[0.04] cursor-copy"
                      : "border-foreground/15 hover:border-foreground/30 hover:bg-foreground/[0.02] cursor-pointer",
                  ].join(" ")}
                >
                  <div className="text-sm text-foreground/85">
                    {v2Remaining > 0 ? "Click or drop images here" : `Maximum ${V2_MAX_REFS} images reached`}
                  </div>
                  {v2Remaining > 0 && (
                    <div className="mt-1 text-xs text-foreground/45">
                      {v2RefImages.length} of {V2_MAX_REFS} used &nbsp;·&nbsp; jpg · png · webp
                    </div>
                  )}
                </div>
                <input
                  ref={v2FileRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={(e) => v2OnUploadFiles(e.target.files)}
                />
              </div>

              <div className="mt-8 flex items-center justify-end gap-3">
                <Tooltip side="top" align="center" label="Skip, open project without uploading references">
                  <button
                    type="button"
                    onClick={v2SkipFromRefs}
                    className="px-4 py-2 text-sm text-foreground/55 hover:text-foreground/90 transition-colors"
                  >
                    Skip
                  </button>
                </Tooltip>
                <div className="flex flex-col items-end gap-1">
                  <button
                    type="button"
                    onClick={v2DoneRefs}
                    className="rounded-full px-7 py-2.5 text-sm font-semibold text-black transition-all"
                    style={{ backgroundColor: "#fb923c" }}
                  >
                    Open project
                  </button>
                  <span className="text-[10px] text-foreground/30 font-mono select-none">Ctrl ↵</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Reference image editor, opened on thumbnail click. Esc /
          Close button + Left/Right arrows + on-screen prev/next
          buttons all flow through setV2ViewingIdx. */}
      {v2ViewingIdx !== null && v2RefImages[v2ViewingIdx] && (
        <ReferenceImageEditor
          refImage={v2RefImages[v2ViewingIdx]}
          labels={v2Labels}
          projectId={v2ProjectId}
          onChange={(nextBoxes) => {
            const editing = v2RefImages[v2ViewingIdx];
            const previewKey = editing?.preview;
            if (!previewKey) return;
            setV2RefImages((cur) =>
              cur.map((it) =>
                it.preview === previewKey ? { ...it, boxes: nextBoxes } : it,
              ),
            );
            // Persist the edit so it carries through to the project.
            v2ScheduleRefEdit(editing?.referenceId, nextBoxes);
          }}
          onClose={() => setV2ViewingIdx(null)}
          onPrev={() => setV2ViewingIdx((i) => (i === null ? null : Math.max(0, i - 1)))}
          onNext={() =>
            setV2ViewingIdx((i) =>
              i === null ? null : Math.min(v2RefImages.length - 1, i + 1),
            )
          }
          hasPrev={v2ViewingIdx > 0}
          hasNext={v2ViewingIdx < v2RefImages.length - 1}
          index={v2ViewingIdx}
          total={v2RefImages.length}
        />
      )}

      {/* V1 inline create form + projects grid. Hidden when V2 is
          fully active, the V2 stages take over the centre while the
          projects collapse out. Footer stays mounted (below) so the
          page still feels grounded. */}
      {!v2Active && (
        <section className="mx-auto max-w-6xl px-6 pb-24 grid gap-8">
          {creating && (
            <div className="rounded-xl border border-[var(--border)] p-5 grid gap-4 animate-[fadeIn_180ms_ease-out]">
              <div className="flex flex-wrap items-center gap-3">
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") create();
                    if (e.key === "Escape") setCreating(false);
                  }}
                  placeholder="Project name (e.g. potholes)"
                  className="flex-1 min-w-[12rem] bg-transparent border-b border-[var(--border)] focus:border-zinc-400 outline-none py-2 text-base"
                />
                <button
                  onClick={create}
                  disabled={!newName.trim()}
                  className="rounded-full bg-foreground text-background px-5 py-2.5 text-sm font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Create
                </button>
              </div>
              {isFreePlan && (
                <div className="rounded-lg border border-amber-300/25 bg-amber-300/[0.04] px-3.5 py-2.5 flex items-start gap-3">
                  <span aria-hidden className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-300/80 shrink-0" />
                  <div className="text-[12px] text-foreground/75 leading-relaxed">
                    <span className="text-amber-200/90 font-mono uppercase tracking-wider text-[10px]">Public project</span>
                    <span className="text-foreground/45"> · </span>
                    Free projects are public, anyone can see the images, labels and exports.
                    Don&rsquo;t upload anything personal, sensitive, controlled, restricted or
                    confidential. Upgrade to Pro for private projects.
                  </div>
                </div>
              )}
            </div>
          )}

          {error && <div className="text-sm text-red-400">{error}</div>}

          {/* Projects (team containers of datasets). Sits above the Datasets
              grid. Only for signed-in users (lists the user's own Projects). */}
          {loggedIn && (
            <ProjectsSection onOpenProject={(id) => setProjectViewId(id)} />
          )}

          {/* Datasets toolbar: tab filters (All / Starred) on the left; search,
              sort + the create button on the right, all on one line. The create
              button always renders so a brand-new account can make its first
              dataset; tabs / search / sort only appear once there's data. */}
          {sortedProjects !== null && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* Left group: tab filters + sort */}
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
                          "inline-flex items-center gap-2 rounded-full px-3.5 h-9 text-sm transition-colors",
                          isActive
                            ? "bg-orange-500/[0.12] text-orange-700 dark:text-orange-300 font-semibold"
                            : "text-foreground/55 hover:text-foreground/90 hover:bg-foreground/[0.04]",
                        ].join(" ")}
                      >
                        {t.label}
                        <span
                          className={[
                            "tabular-nums rounded-full px-1.5 text-[11px] leading-5",
                            isActive ? "bg-orange-500/20 text-orange-700 dark:text-orange-200" : "bg-foreground/[0.06] text-foreground/45",
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
              {/* Right group: search + create */}
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
                      className="h-9 rounded-full bg-foreground/[0.04] border border-foreground/10 hover:border-foreground/20 focus:border-foreground/40 focus:bg-foreground/[0.06] outline-none pl-9 pr-3 text-sm text-[var(--foreground)] placeholder:text-foreground/40 transition-colors w-44 sm:w-56"
                    />
                  </label>
                )}
                <button
                  type="button"
                  onClick={() => {
                    // Workspace create is never scoped to a Project.
                    pendingContainerRef.current = null;
                    if (creating) setCreating(false);
                    if (v2Stage === "idle") {
                      if (onV2Begin) {
                        // Same name popup as the Project page, instead of the
                        // old inline name bar at the top of the workspace.
                        setShowWorkspaceCreate(true);
                      } else {
                        setCreating((v) => !v);
                      }
                    } else {
                      v2Reset();
                    }
                  }}
                  className="h-9 rounded-full bg-foreground text-background px-4 text-sm font-medium hover:opacity-90 transition-opacity"
                >
                  {v2Stage !== "idle" || creating ? "Cancel" : "+ Add Dataset"}
                </button>
              </div>
            </div>
          )}

          {sortedProjects === null ? (
            <div className="text-[var(--muted)] text-sm">Loading…</div>
          ) : sortedProjects.length === 0 ? (
            <div className="rounded-3xl border border-foreground/10 bg-foreground/[0.02] py-16 text-center text-foreground/50">
              {isSearching ? (
                <>No projects match <span className="text-foreground/90">&ldquo;{searchQuery.trim()}&rdquo;</span>.</>
              ) : tab === "starred" ? (
                <>No starred projects yet. Tap the bookmark on a card to pin it here.</>
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
                    onOpen={() => onOpen(p.id, p.createdBy, p.name, p.v2)}
                    onOpenProject={(id) => setProjectViewId(id)}
                    onDelete={() => setDeleteTarget(p)}
                    onDuplicate={() => duplicateProject(p)}
                    duplicating={duplicatingId === p.id}
                    onLike={() => toggleLike(p.id)}
                    /* First WORKSPACE_PAGE_SIZE cards load real bytes
                       immediately; the rest blurhash-only until the
                       cover scrolls into the near-viewport. */
                    eagerCover={i < WORKSPACE_PAGE_SIZE}
                  />
                ))}
                {/* Skeleton placeholders for slots that haven't been
                    paginated in yet. Renders the full final grid
                    shape on first paint so the user gets a sense of
                    the workspace size before everything streams in.
                    Hidden while the user has an active search query
                    (the filter applies to "loaded so far", adding
                    placeholders below the matches would be noise). */}
                {tab === "all"
                  && !isSearching
                  && total !== null
                  && projects !== null
                  && projects.length < total
                  && Array.from({ length: Math.min(WORKSPACE_PAGE_SIZE, total - projects.length) }).map((_, i) => (
                    <div
                      key={`ws-skel-${projects.length + i}`}
                      className="rounded-2xl border border-foreground/10 bg-foreground/[0.02] overflow-hidden"
                    >
                      <div className="aspect-video bg-foreground/[0.04]" />
                      <div className="p-4 space-y-2">
                        <div className="h-4 w-2/3 rounded bg-foreground/[0.06]" />
                        <div className="h-3 w-1/3 rounded bg-foreground/[0.04]" />
                      </div>
                    </div>
                  ))}
              </div>
              {/* Sentinel for the IntersectionObserver, sits 300 px
                  above the actual bottom so the next page kicks off
                  before the user reaches the end of what's loaded.
                  Only armed on the unfiltered All tab, the Starred
                  view + active search both work over the loaded set. */}
              {tab === "all" && !isSearching && total !== null && projects !== null && projects.length < total && (
                <div ref={sentinelRef} aria-hidden className="h-1 w-full mt-2" />
              )}
            </>
          )}
        </section>
      )}

      {/* Footer is always mounted, full multi-column footer in
          every state. During V2 the wrapper above sizes itself to
          (100vh - title height) so the centred body sits at the
          viewport's vertical middle when scrolled to the top, and
          the footer lives in normal flow below, visible by
          scrolling, and counted in the page height. */}
      <Footer />

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

// Cover image that prefers the server-rendered annotated preview
// (boxes + masks baked in) and falls back to the raw original when
// the annotated version doesn't exist for that filename. The
// `updatedAt` query string busts the browser cache whenever the
// manifest is rewritten, manual edits in the editor trigger a
// re-bake on the backend, so the cover refreshes the next time the
// list is polled without us having to invalidate anything by hand.
function ProjectCover({
  projectId,
  thumbnail,
  updatedAt,
  v2 = false,
  blurhash = null,
  eager = false,
}: {
  projectId: string;
  thumbnail: string;
  updatedAt: string | null;
  v2?: boolean;
  /** BlurHash placeholder rendered until the card scrolls into the
      near-viewport. Without it, off-screen cards drop to a flat dark
      square, fine, just less informative. */
  blurhash?: string | null;
  /** When true, skip the IntersectionObserver gate and load the cover
      image immediately. Used for the first row of cards above the
      fold so they paint with real images on initial load instead of
      waiting for the observer to tick. */
  eager?: boolean;
}) {
  // Workspace card uses the server-rendered ~480 px /cover_thumb so
  // the grid paints at a fraction of the bytes the full-resolution
  // cover would cost. Falls back to the original when the thumb
  // endpoint isn't available (older deployments) so the card never
  // shows a broken image. Cache-buster keeps a renamed-cover swap
  // responsive, backend resolves source from references/ then
  // imports/ regardless of v2 flag.
  const cacheBuster = updatedAt ? `?v=${encodeURIComponent(updatedAt)}` : "";
  const thumbUrl = `${API}/api/projects/${projectId}/cover_thumb${cacheBuster}`;
  const fallbackUrl = v2
    ? `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(thumbnail)}${cacheBuster}`
    : `${API}/api/projects/${projectId}/originals/${encodeURIComponent(thumbnail)}${cacheBuster}`;
  // `src` is what's actually painted; `pending` is the next URL we're
  // preloading off-screen. Without this dance the browser blanks the
  // <img> for one frame whenever the cache-buster (?v=updatedAt)
  // changes, which happens every time the user opens a project,
  // because the blurhash-backfill async job writes the manifest and
  // bumps updatedAt by the time they navigate back to the workspace.
  // That single-frame blank flashed the white card background.
  const [src, setSrc] = useState(thumbUrl);
  const [pending, setPending] = useState<string | null>(null);
  useEffect(() => {
    if (thumbUrl === src) return;
    setPending(thumbUrl);
  }, [thumbUrl, src]);

  // IntersectionObserver gate. The first WORKSPACE_PAGE_SIZE (= 12)
  // cards opt-in to `eager` so they paint with real bytes on the
  // initial render. Everything below the fold renders the blurhash
  // placeholder until it scrolls within 600 px of the viewport, then
  // upgrades to the actual <img>. Without this, every card the user
  // had scrolled past was sitting on the GPU as a decoded JPEG,
  // making theme toggles + card-open animations stutter on a wide
  // workspace.
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
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt=""
            loading="lazy"
            decoding="async"
            onError={() => { if (src !== fallbackUrl) setSrc(fallbackUrl); }}
            className="absolute inset-0 w-full h-full object-cover"
          />
          {pending && pending !== src && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={pending}
              alt=""
              decoding="async"
              onLoad={() => { setSrc(pending); setPending(null); }}
              onError={() => {
                if (pending !== fallbackUrl) setPending(fallbackUrl);
                else setPending(null);
              }}
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
  onLike,
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
  onLike: () => void;
  /** First-row hint, when true, the cover skips its
      IntersectionObserver gate and loads bytes immediately on
      mount. */
  eagerCover?: boolean;
}) {
  // Tag-chip colours come from colourForLabelStable so the same label
  // keeps its colour across the workspace card, public-feed card, and
  // the project page (see ProjectTagsRow). Status pills are shared with
  // the public feed via projectStatusBadges so both surfaces match.
  const statusBadges = projectStatusBadges(project);
  const [tagsOverflowOpen, setTagsOverflowOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  // Shared chrome for the two floating cover buttons: a frosted white
  // pill that stays legible on any thumbnail in both themes.
  const coverBtn =
    "h-8 w-8 rounded-full grid place-items-center backdrop-blur-md bg-white/85 hover:bg-white text-zinc-700 transition-colors";
  const coverBtnShadow = { boxShadow: "0 1px 6px rgb(var(--shadow-rgb) / 0.45), 0 0 0 1px rgb(var(--shadow-rgb) / 0.08)" };
  return (
    <div
      className="pk-card pk-card-hover group overflow-hidden rounded-2xl"
      style={{
        // Lift the card above the dim overlay (z-1000) while either
        // popup (tag overflow / overflow menu) is open so the card
        // chrome stays bright.
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
                thumbnail={project.thumbnail}
                updatedAt={project.updatedAt}
                v2={project.v2}
                blurhash={project.cover_blurhash ?? null}
                eager={eagerCover}
              />
            ) : (
              <span className="text-foreground/30 text-xs">No images yet</span>
            )}
          </div>
        </button>
        {project.certified && (
          <div className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 backdrop-blur-md border border-amber-300/40 text-amber-50 px-2.5 py-1 text-[10px] font-medium shadow-lg pointer-events-none">
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="currentColor">
              <path d="M12 1.6l2.6 5.6 6.1.6-4.6 4.2 1.4 6.1L12 14.9 6.5 18.1l1.4-6.1L3.3 7.8l6.1-.6z" />
            </svg>
            Certified
          </div>
        )}
        {/* Favourite (star), top-right, mirrors the Starred tab. */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavourite();
          }}
          aria-pressed={favourited}
          aria-label={favourited ? "unfavourite" : "favourite"}
          className={[coverBtn, "absolute top-3 right-3", favourited ? "text-amber-500" : ""].join(" ")}
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
        {/* Overflow menu, bottom-right, holds the destructive Delete
            action so it's out of the primary click path but still one
            tap away. */}
        <div className="absolute bottom-3 right-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((v) => !v);
            }}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label="Project actions"
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
                className="absolute bottom-[calc(100%+8px)] right-0 z-50 w-44 rounded-xl border border-foreground/10 bg-[var(--background)] p-1.5 shadow-xl"
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
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm text-foreground/80 hover:bg-foreground/[0.06] transition-colors disabled:opacity-50 disabled:cursor-default"
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
                  className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm text-red-600 hover:bg-red-500/10 dark:text-red-300 transition-colors"
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
                {project.private && <PrivateLockIcon />}
                {project.derived && <DerivedBadge parentName={project.derived.parentName} />}
              </div>
              <div className="mt-0.5 text-xs text-foreground/40">
                by <span className="text-foreground/65">@{project.createdBy || "you"}</span>
              </div>
            </div>
            {statusBadges.length > 0 && (
              <div className="flex flex-wrap gap-1.5 justify-end shrink-0 pt-0.5">
                {statusBadges.map((b) => (
                  <span
                    key={b.label}
                    className={[
                      "rounded-full px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider border",
                      b.classes,
                    ].join(" ")}
                  >
                    {b.label}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Compact inline stats: small uppercase label + value, so
              the card reads at a glance without the old oversized
              numbers dominating the body. */}
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <div className="flex items-baseline gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Images</span>
              <span className="tabular-nums text-sm font-semibold text-[var(--foreground)]">{project.n_images}</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Labelled</span>
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
      <div className="px-4 pb-3.5 flex items-center justify-between text-xs">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onLike();
          }}
          aria-pressed={project.likedByMe}
          className={[
            "inline-flex items-center gap-1.5 rounded-full px-2 py-1 -ml-2 transition-colors",
            project.likedByMe
              ? "text-pink-500 hover:text-pink-400"
              : "text-foreground/45 hover:text-foreground",
          ].join(" ")}
          title={project.likedByMe ? "Unlike" : "Like"}
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill={project.likedByMe ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 000-7.78z" />
          </svg>
          <span className="tabular-nums">{project.likes}</span>
        </button>
        {project.container && onOpenProject && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onOpenProject(project.container!.id);
            }}
            className="inline-flex items-center gap-1 rounded-full bg-orange-500/10 px-2.5 py-1 text-[11px] font-medium text-orange-600 transition-colors hover:bg-orange-500/20 dark:text-orange-300"
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
      <div className="pk-glass pk-pop max-w-md w-full rounded-2xl overflow-hidden">
        <header className="px-6 pt-6 pb-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-red-700 dark:text-red-400">Permanent action</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Delete project?</h2>
        </header>

        <div className="px-6 pb-5 grid gap-4">
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            This removes{" "}
            <span className="font-mono text-[var(--foreground)]">{project.name}</span> along with{" "}
            <span className="font-mono text-[var(--foreground)]">{project.n_images}</span> image
            {project.n_images === 1 ? "" : "s"}, every label, every verdict, and any model state.{" "}
            <span className="text-red-700 dark:text-red-300">This cannot be recovered.</span>
          </p>

          <div className="rounded-lg border border-red-500/40 dark:border-red-500/30 bg-red-500/[0.08] dark:bg-red-500/[0.05] px-3 py-2.5 text-xs text-red-800 dark:text-red-200">
            Type the project name{" "}
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
            className="w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2.5 font-mono text-sm focus:outline-none focus:border-[var(--foreground)]/40"
            spellCheck={false}
            autoComplete="off"
          />
        </div>

        <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!matches || busy}
            className={[
              "rounded-full px-5 py-2 text-sm font-medium transition-colors",
              matches && !busy
                ? "bg-red-600 dark:bg-red-500 text-white hover:bg-red-700 dark:hover:bg-red-400"
                : "bg-red-500/20 text-red-700/60 dark:text-red-300/60 cursor-not-allowed",
            ].join(" ")}
          >
            {busy ? "Deleting…" : "Delete forever"}
          </button>
        </footer>
      </div>
    </div>
  );
}
