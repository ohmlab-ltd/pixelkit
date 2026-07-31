"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ProjectTagsRow } from "./components/ProjectTagsRow";
import { containsProfanity } from "./profanity";
import { BlurhashCanvas } from "react-blurhash";
import { LABEL_COLOURS } from "./v2/OnboardLabelsV2";
import type { ReferenceImage } from "./v2/OnboardReferencesV2";
import { WordCloud } from "./v2/WordCloud";
import { PixelKitLoader } from "./v2/PixelKitLoader";
import { capture } from "./lib/analytics";
import { ProjectsSection } from "./ProjectsSection";
import { ProjectPage } from "./ProjectPage";
import { CreateDatasetModal } from "./CreateDatasetModal";
import { apiFetch } from "@/lib/apiFetch";
import { onNewDatasetRequest, requestExplorerRefresh } from "@/lib/appNav";
import { addDataset } from "@/lib/containers";
import { readProjectMetaCache, writeProjectMetaCache, type ProjectMetaCache } from "@/lib/projectMetaCache";
import {
  collectInput,
  parseDataset,
  uploadDataset,
  ZIP_SOFT_LIMIT_BYTES,
  type ParsedDataset,
} from "@/lib/datasetImport";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

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


// Branch glyph flagging a derived (cropped child) project on the cards —
// quiet neutral icon tone so it doesn't compete with the status chips.
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
      className="text-[var(--fg-dim)] shrink-0"
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

// Status chip set for a project card (Labelled / Partial / Unlabelled
// / In progress, plus Model). Rendered as neutral hairline chips with
// a small colour dot — tone comes from the dot only, never the chip
// surface, so the cards stay flat in both themes.
export function projectStatusBadges(
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

// Small custom dropdown for the workspace sort control (button +
// click-away backdrop + menu), consistent with the rest of the app's
// pill menus and themes cleanly.
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
  username,
  loggedIn = true,
}: {
  onOpen: (name: string, owner?: string, displayName?: string, v2?: boolean, fromProjectId?: string) => void;
  /** Optional V2 entry point. Fired once the user has finished the
      inline V2 flow (name + labels). The parent mounts the
      post-onboarding view from this callback. When the prop is
      undefined the V2 button stays hidden and only the V1 path is
      reachable. References are always empty at creation time — they
      can be added later from the dataset view. */
  onV2Begin?: (
    name: string,
    labels: string[],
    references: ReferenceImage[],
    projectId: string | null,
    /** "onboarding" → the user is arriving straight out of the
     *  create flow; HomeView's "Opening project…" overlay is still
     *  on screen, so the project view should suppress its own
     *  full-screen mount loader.
     *  null → normal load, project view uses its default loader. */
    firstLoad?: "onboarding" | null,
  ) => void;
  username: string;
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
  // page locks scroll, and the labels stage renders in the same
  // vertical band underneath. "creating" is the brief loader shown
  // while the backend creates the dataset before it opens.
  type V2Stage = "idle" | "name" | "labels" | "creating" | "import";
  const [v2Stage, setV2Stage] = useState<V2Stage>("idle");
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
  useEffect(() => { v2ProjectIdRef.current = v2ProjectId; }, [v2ProjectId]);

  // Desktop shell: the Explorer side bar's "+" button fires a
  // new-dataset request on the appNav bus. React exactly like the
  // workspace toolbar's "+ Add Dataset" button — workspace-scoped
  // (never attached to a Project), same CreateDatasetModal entry.
  // If onboarding is already underway the request is a no-op rather
  // than a cancel, "+" should never destroy in-progress work.
  useEffect(() => {
    return onNewDatasetRequest(() => {
      if (v2Stage !== "idle") return;
      pendingContainerRef.current = null;
      setV2ReturnProjectId(null);
      setProjectViewId(null);
      if (creating) setCreating(false);
      if (onV2Begin) setShowWorkspaceCreate(true);
      else setCreating(true);
    });
  }, [creating, v2Stage, onV2Begin]);

  const v2InputRef = useRef<HTMLInputElement | null>(null);

  // Stages that take over the workspace surface. While active the
  // search/add buttons fade out and the projects grid is hidden; the
  // page stays scrollable and the footer remains in flow so the
  // surface doesn't feel isolated.
  const v2Active = v2Stage === "labels" || v2Stage === "creating" || v2Stage === "import";

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

  // Auto-focus the chip input when the labels stage opens.
  useEffect(() => {
    if (v2Stage !== "labels") return;
    const t = window.setTimeout(() => v2InputRef.current?.focus(), 320);
    return () => window.clearTimeout(t);
  }, [v2Stage]);

  const v2Reset = () => {
    // An in-flight import is mid-upload to a real project; ignore cancel/ESC
    // until it finishes (the Cancel button is disabled too) so we never
    // delete the project out from under the upload.
    if (v2ImportBusyRef.current) return;
    // Cancelling clears any parked Project so the next normal create doesn't
    // get attached to it.
    pendingContainerRef.current = null;
    setV2ReturnProjectId(null);
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
    setV2ProjectId(null);
    v2ProjectIdRef.current = null;
    setV2ImportParsed(null);
    setV2ImportParsing(false);
    setV2ImportParseProgress(null);
    setV2ImportProgress(null);
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

  // Create the V2 project on the backend and stash its ID.
  // Idempotent: returns the existing ID if we've already created
  // one in this onboarding session (e.g. the import path).
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
      fd.append("owner", username || "anonymous");
      const r = await apiFetch(`/api/v2/projects`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = (await r.json()) as { project_id: string };
      const pid = data.project_id;
      setV2ProjectId(pid);
      v2ProjectIdRef.current = pid;
      // The Explorer tree polls its listing on a 10 s cycle — nudge it
      // now so the fresh dataset appears immediately.
      requestExplorerRefresh();
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
        // Full resolution always — no client-side downscale on import.
        { maxSide: null },
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
      // Open the new project. Labels = the dataset's classes.
      if (onV2Begin) onV2Begin(v2Name.trim(), parsed.classes, [], projectId, null);
      // Reset onboarding state (mirrors v2HandOff's tail).
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

  // Hand the gathered name + labels off to the parent. The backend
  // project record is created here if it doesn't exist yet (the
  // import path creates it earlier).
  const v2HandOff = async (
    name: string,
    labels: string[],
    firstLoad: "onboarding" | null = null,
  ) => {
    if (!onV2Begin) return;
    let projectId = v2ProjectIdRef.current;
    if (!projectId) projectId = await v2EnsureProject(name, labels);
    // If this onboarding was launched from a Project page, add the new dataset
    // to that Project (container) so it lands inside it, not standalone.
    if (projectId && pendingContainerRef.current) {
      try {
        await addDataset(pendingContainerRef.current, projectId);
        // Re-fetch the tree again: the dataset now lives under its
        // Project node, not at the root where the create landed it.
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
  };

  // Labels confirmed → create the dataset and open it. The "creating"
  // stage shows the in-place PixelKit loader while the create POST is
  // in flight; reference images can be added later from the dataset
  // view's References tab.
  const v2DoneLabels = () => {
    if (v2Labels.length === 0) { v2HandOff(v2Name, []); return; }
    setV2Stage("creating");
    void v2HandOff(v2Name, v2Labels, "onboarding");
  };

  const v2SkipFromLabels = () => v2HandOff(v2Name, []);

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
            `You've hit the limit of ${usage.limits.projects} projects. Delete a project to create another.`,
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
      requestExplorerRefresh();
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
      requestExplorerRefresh();
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
      requestExplorerRefresh();
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
          setProjectViewId(null);
          v2BeginLabels(name);
        }}
      />
    );
  }

  // (Logged-out marketing view removed — the portable build has no accounts.)

  return (
    <main className="min-h-full bg-[var(--background)] overflow-x-hidden">
      {/* Title + controls strip. Stays mounted across all stages ,
          the H1 / subtitle / search-add buttons crossfade rather
          than reflow so the page never visibly jumps. `relative
          z-20` puts the title above the V2 wrapper, which uses a
          negative top margin to overlap with this band so the
          centred V2 body lands at exactly 50vh of the viewport. */}
      <section className="relative z-20 mx-auto max-w-[1400px] px-6 pt-8 pb-6 flex items-end justify-between gap-6 flex-wrap">
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
          {/* Subtitle stack, same sequencing rule. While V2 is active
              the workspace tagline is replaced by the project's
              creation date so the user can read off the metadata for
              the project they just named. */}
          <div className="relative mt-2 max-w-2xl">
            <p
              className={[
                "text-foreground/50 text-sm transition-all duration-150 ease-out",
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
              {/* No identity in this build — just the creation-date
                  stamp. */}
              <span className="tabular-nums">Created {v2DateLabel}</span>
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
          v2BeginLabels(name);
        }}
      />

      {/* V2 stage body. Labels stage uses the centred-prompt layout
          (negative top margin + flex items-center) so the question
          sits a touch above the viewport's mid-line. The import
          stage drops the centring, its content grows as the user
          picks a dataset, so normal flow + a small top padding
          keeps the heading anchored and lets the page grow. */}
      {v2Active && (
        <div
          className={[
            "relative px-6",
            // The creating stage shares the same vertically-centred
            // layout as the labels stage so the swap is in-place, no
            // wrapper-level layout shift, no flicker between forms.
            // Pull-up tuned against the denser title band (was -13rem
            // when the workspace heading was text-6xl marketing scale).
            v2Stage === "labels" || v2Stage === "creating"
              ? "-mt-[6rem] min-h-[calc(100vh-10rem)] flex items-center justify-center pointer-events-none"
              : "pt-4 pb-16 pointer-events-none",
          ].join(" ")}
        >
          {/* Ambient word backdrop, only on the labels stage. The
              parent gate keeps it mounted briefly after the user
              leaves so the fade-out animation plays in full. */}
          {v2Stage === "labels" && v2WordCloudMounted && <WordCloud show={v2WordCloudShouldShow} />}
          {(v2Stage === "labels" || v2Stage === "creating") && (
            <div
              key="v2-labels"
              className={[
                // Both the form and the creating loader live in
                // this relative wrapper so the loader can absolute-
                // overlay the form and the two crossfade in place
                // when the user clicks Done.
                "relative w-full max-w-2xl pointer-events-auto",
                v2Stage === "labels" ? "animate-[fadeIn_280ms_ease-out]" : "",
              ].join(" ")}
            >
            {/* Labels form crossfades out as the loader fades in.
                Done is disabled while the create call is in flight
                so the user can't double-fire. */}
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
                Enter the objects you want to detect, you can add more later.
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
                      // Neutral chip: hairline border, transparent surface,
                      // the label's colour carried by the dot only.
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

              {/* Inline validation (e.g. the profanity guard) — the V1
                  error strip below is hidden while this stage is up. */}
              {error && <p className="mt-3 text-sm text-[var(--bad)]">{error}</p>}

              <div className="mt-6 flex items-center gap-3">
                {/* Buttons sit on the right via an auto-margin so
                    Skip + Done hold the right edge. */}
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
                    disabled={v2Labels.length === 0 || v2Stage === "creating"}
                    className="pk-btn pk-btn-primary"
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

            {/* Creating loader, absolute over the form's space.
                Crossfades with the form (form opacity 0 / loader
                fades in) so the swap reads as a single in-place
                transition while the dataset is created and opened. */}
            {v2Stage === "creating" && (
              <div
                key="v2-creating"
                className="absolute inset-0 grid place-items-center pointer-events-auto animate-[fadeIn_320ms_ease-out]"
              >
                <PixelKitLoader size={128} message="Opening project…" />
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
                Pascal VOC — an <span className="font-mono">Annotations/</span> folder of XML next
                to your images — as a folder or a <span className="font-mono">.zip</span>. Every
                image and box comes in fully editable.
              </p>

              {error && <p className="mt-4 text-sm text-[var(--bad)]">{error}</p>}

              {/* Picker — folder or zip. Hidden once a dataset is parsed. */}
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
                        ? ` — ${v2ImportParseProgress.done}/${v2ImportParseProgress.total}`
                        : "…"}
                    </span>
                  )}
                </div>
              )}

              {/* Preview + options once parsed. */}
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

      {/* V1 inline create form + projects grid. Hidden when V2 is
          fully active, the V2 stages take over the centre while the
          projects collapse out. */}
      {!v2Active && (
        <section className="mx-auto max-w-[1400px] px-6 pb-24 grid gap-8">
          {creating && (
            <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] p-5 grid gap-4 animate-[fadeIn_180ms_ease-out]">
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
                  className="flex-1 min-w-[12rem] rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-base transition-colors"
                />
                <button
                  onClick={create}
                  disabled={!newName.trim()}
                  className="pk-btn pk-btn-primary"
                >
                  Create
                </button>
              </div>
            </div>
          )}

          {error && <div className="text-sm text-[var(--bad)]">{error}</div>}

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
                      className="h-9 rounded-md bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--accent)] outline-none pl-9 pr-3 text-sm text-[var(--foreground)] placeholder:text-foreground/40 transition-colors w-44 sm:w-56"
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
                  className={[
                    "h-9 rounded-md px-4 text-[13px] font-medium transition",
                    // Accent is reserved for the single primary action; the
                    // morphed Cancel state drops to the flat neutral recipe.
                    v2Stage !== "idle" || creating
                      ? "border border-[var(--line)] text-foreground/75 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
                      : "bg-[var(--accent)] text-[var(--accent-contrast)] hover:brightness-105",
                  ].join(" ")}
                >
                  {v2Stage !== "idle" || creating ? "Cancel" : "+ Add Dataset"}
                </button>
              </div>
            </div>
          )}

          {sortedProjects === null ? (
            <div className="text-[var(--muted)] text-sm">Loading…</div>
          ) : sortedProjects.length === 0 ? (
            <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] py-10 text-center text-foreground/50">
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
    "h-8 w-8 rounded-md grid place-items-center backdrop-blur-md bg-white/85 hover:bg-white text-zinc-700 transition-colors";
  const coverBtnShadow = { boxShadow: "var(--shadow-soft)" };
  return (
    <div
      className="pk-card pk-card-hover group overflow-hidden rounded-md"
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
          <div
            className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-md bg-white/85 backdrop-blur-md px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-zinc-700 pointer-events-none"
            style={{ boxShadow: "var(--shadow-soft)" }}
          >
            <span aria-hidden className="h-2 w-2 rounded-full bg-[var(--warn)]" />
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

          {/* Compact inline stats: small uppercase label + value, so
              the card reads at a glance without the old oversized
              numbers dominating the body. */}
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
      <div className="px-4 pb-3.5 flex items-center justify-between text-xs">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onLike();
          }}
          aria-pressed={project.likedByMe}
          className={[
            "inline-flex items-center gap-1.5 rounded-md px-2 py-1 -ml-2 transition-colors",
            project.likedByMe
              ? "text-accent hover:text-accent/80"
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
          <h2 className="mt-2 text-xl font-semibold tracking-tight">Delete project?</h2>
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
