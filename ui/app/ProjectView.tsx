"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { navigateAppTo } from "@/lib/appNav";
import { BoxEditor, EditableBox, detectionsToBoxes } from "./BoxEditor";
import { SimilarLabelsModal, type SimilarMatch } from "./SimilarLabelsModal";
import { LabelCascadeReviewModal, type CascadeGroup } from "./LabelCascadeReviewModal";
import { isImageFile, resizeForUpload } from "@/lib/resize";
import { ReviewMode, type ReviewScope } from "./ReviewMode";
import { TrainView } from "./TrainView";
import { DeployView } from "./DeployView";
import { OptimiseView } from "./OptimiseView";
import { ProjectSettings } from "./ProjectSettings";
import { ExportModal } from "./ExportModal";
import { Footer } from "./Footer";
import { lookupUsers } from "@/lib/userCache";
import { apiFetch } from "@/lib/apiFetch";
import { isProPlan } from "@/lib/plans";
import { containsProfanity } from "./profanity";
import { usePlan } from "./PlanPill";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// Block a labelling job if it would push the user over their plan's
// monthly quota. Returns null when the run is allowed, or a
// user-facing error string when it must be refused. Falls open if the
// usage endpoint can't be reached, better to over-permit than to
// wedge users when the API is having a moment.
async function checkLabelQuota(plannedImages: number): Promise<string | null> {
  try {
    const r = await fetch("/api/users/usage", { cache: "no-store" });
    if (!r.ok) return null;
    const usage = (await r.json()) as {
      planName: string;
      limits: { imagesLabelledPerMonth: number };
      usage: { imagesLabelledThisPeriod: number };
    };
    const used = usage.usage.imagesLabelledThisPeriod;
    const limit = usage.limits.imagesLabelledPerMonth;
    if (used >= limit) {
      return `You've used ${used.toLocaleString()} of ${limit.toLocaleString()} auto-labelled images this period on the ${usage.planName} plan. Switch to manual labelling or upgrade to keep auto-labelling.`;
    }
    if (used + plannedImages > limit) {
      const remaining = limit - used;
      return `This run would label ${plannedImages.toLocaleString()} images but only ${remaining.toLocaleString()} remain on your ${usage.planName} plan this period.`;
    }
    return null;
  } catch {
    return null;
  }
}

type Detection = {
  label: string;
  score: number | null;
  box_xyxy: number[];
  mask?: { polygons: number[][][] } | null;
  validation?: {
    match: boolean;
    confidence: number;
    reason: string;
    model?: string | null;
    source?: "auto" | "manual";
  } | null;
};

type Result = {
  image: string;
  annotated: string | null;
  size: { width: number; height: number };
  detections: Detection[];
  pending?: boolean;
  // Origin metadata stamped by the URL-import flow. Used to filter
  // future Openverse searches so already-imported images don't show
  // up again on the same project.
  source?: { kind: string; url?: string } | null;
};

type Phase = "idle" | "uploading" | "running" | "done" | "error";
type Verdict = "good" | "bad";
type Filter = "all" | "good" | "bad" | "unrated" | "unlabeled" | "vlm";
type Tab = "label" | "train" | "deploy" | "optimise";

type Manifest = {
  name: string;
  prompt: string;
  tags: string[];
  thresholds: { box: number; text: number; nms: number };
  // Resolution choice for the labelling run: tile large images at native
  // resolution (small objects survive) vs the classic full-frame downscale.
  tiling?: { native?: boolean; tileSize?: number };
  vlm_action?: "manual" | "auto_reject";
  synonyms_enabled?: boolean;
  results: Result[];
  verdicts: Record<string, Verdict>;
  editedBoxes: Record<string, EditableBox[]>;
  cover?: string | null;
  hasModel?: boolean;
  owner?: string;
  createdAt?: string;
};

// Two named modes for the auto-label thresholds. "Normal" is the
// permissive default, picks up borderline detections so the user has
// more candidates to review. "Stricter" raises both score gates so
// only confident matches survive, useful when the project tag is
// generic ("road") and the model would otherwise over-detect.
const LABEL_MODES = {
  normal: { box: 0.05, text: 0.15, nms: 0.7 },
  stricter: { box: 0.15, text: 0.20, nms: 0.7 },
} as const;

export function ProjectView({
  name,
  initialDisplayName = "",
  username,
  readOnly = false,
  onClose,
  onRename,
}: {
  name: string;
  initialDisplayName?: string;
  username: string;
  readOnly?: boolean;
  onClose: () => void;
  onRename: (newName: string) => void;
}) {
  const [loaded, setLoaded] = useState(false);
  // Seed with the name we already know from the project list so the heading
  // never flashes the UUID before the manifest fetch lands.
  const [displayName, setDisplayName] = useState<string>(initialDisplayName);
  const [tab, setTab] = useState<Tab>("label");
  const [cover, setCover] = useState<string | null>(null);
  const [hasModel, setHasModel] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [inputSize, setInputSize] = useState<string>("256x256");
  // When on: small/too-small boxes are dimmed (not removed), thumbnail size
  // chips disappear, the tip line is suppressed, and the editor's right-side
  // list shows them greyed + non-clickable. Lets the user focus on viable
  // labels without losing track of which ones won't survive the resize.
  const [hideSmall, setHideSmall] = useState<boolean>(false);
  const [owner, setOwner] = useState<string>("");
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  // Manifest's `updatedAt`, appended as ?v= to the annotated
  // thumbnail URL so a backend re-bake (lite refresh, manual edit)
  // is reflected in the grid without a hard reload.
  const [manifestUpdatedAt, setManifestUpdatedAt] = useState<string | null>(null);
  const [ownerInfo, setOwnerInfo] = useState<{ name: string | null; image: string | null } | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [draftTag, setDraftTag] = useState("");
  const [boxThr, setBoxThr] = useState<number>(LABEL_MODES.normal.box);
  const [textThr, setTextThr] = useState<number>(LABEL_MODES.normal.text);
  const [nmsIou, setNmsIou] = useState<number>(LABEL_MODES.normal.nms);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Native-resolution tiling for large images. Off = classic single-pass
  // (the detector's preprocessor downscales big frames, which can shrink
  // small objects below detectability). On = the backend slices the image
  // into overlapping native-res tiles and merges — slower, more compute.
  const [tileNative, setTileNative] = useState(false);
  const [tileSize, setTileSize] = useState<number>(1024);
  // Last custom threshold set the user has dialled in. Persists in
  // memory across Normal / Stricter swaps so the third "Custom" pill
  // stays available and clicking it restores their saved values.
  const [customThresholds, setCustomThresholds] = useState<
    { box: number; text: number; nms: number } | null
  >(null);
  const [vlmAction, setVlmAction] = useState<"off" | "manual" | "auto_reject">("manual");
  const [synonymsEnabled, setSynonymsEnabled] = useState<boolean>(true);

  // AI Review is a Pro-only feature, Free users see the segmented
  // control greyed out with an upgrade hint. Whenever they aren't Pro
  // we coerce the value sent to the backend to "off" too, so a stale
  // localStorage / pre-downgrade value can't leak through.
  const planData = usePlan();
  const isProTier = planData?.plan ? isProPlan(planData.plan) : false;
  const effectiveVlmAction: "off" | "manual" | "auto_reject" = isProTier ? vlmAction : "off";

  const [phase, setPhase] = useState<Phase>("idle");
  // Hover state for the Target Input Shape pill / picker swap. Uses
  // a width-animated wrapper so the Hide-Small toggle + info icon
  // ride along on the layout reflow with constant `gap-4` spacing.
  // The intent timer (150 ms) gates expansion so a cursor that's
  // just dragging across the row to reach something else doesn't
  // accidentally trigger the animation.
  const [shapeHovered, setShapeHovered] = useState(false);
  const shapeHoverTimerRef = useRef<number | null>(null);
  const onShapeEnter = () => {
    if (shapeHoverTimerRef.current !== null) window.clearTimeout(shapeHoverTimerRef.current);
    shapeHoverTimerRef.current = window.setTimeout(() => {
      setShapeHovered(true);
      shapeHoverTimerRef.current = null;
    }, 150);
  };
  const onShapeLeave = () => {
    if (shapeHoverTimerRef.current !== null) {
      window.clearTimeout(shapeHoverTimerRef.current);
      shapeHoverTimerRef.current = null;
    }
    setShapeHovered(false);
  };
  const [progressIndex, setProgressIndex] = useState(0);
  const [progressTotal, setProgressTotal] = useState(0);
  const [progressImage, setProgressImage] = useState<string>("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Tracks an in-flight lite re-run per tag so the chip can show a spinner
  // and the SSE listener knows what to refetch on completion.
  const [liteJob, setLiteJob] = useState<{ id: string; tag: string } | null>(null);
  const [liteStartedAt, setLiteStartedAt] = useState<number | null>(null);
  const [liteProgress, setLiteProgress] = useState<{ index: number; total: number; image: string; added: number }>(
    { index: 0, total: 0, image: "", added: 0 },
  );
  const [cancellingLite, setCancellingLite] = useState(false);
  // Wall-clock anchor for elapsed/ETA. Set when we attach to a job, either
  // because we just scheduled it or because we found one already running.
  const [jobStartedAt, setJobStartedAt] = useState<number | null>(null);
  const [cancellingJob, setCancellingJob] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [editedBoxes, setEditedBoxes] = useState<Record<string, EditableBox[]>>({});
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [reviewing, setReviewing] = useState(false);
  const [reviewList, setReviewList] = useState<Result[] | null>(null);
  const [reviewScope, setReviewScope] = useState<ReviewScope>("unrated");
  // Label Cascade, project-wide review surfaces visually-similar
  // box clusters after auto-labelling completes. State holds the
  // groups returned by the scan; modal opens when non-empty.
  const [cascadeGroups, setCascadeGroups] = useState<CascadeGroup[] | null>(null);
  const [cascadeScanning, setCascadeScanning] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  // Progressive rendering: start with the first chunk and load more
  // as the user scrolls. Mounting hundreds of ImageThumbs upfront is
  // both a paint hit and an `<img>` request flood, even with native
  // lazy loading.
  const GRID_PAGE_SIZE = 15;
  const [displayLimit, setDisplayLimit] = useState(GRID_PAGE_SIZE);
  const gridSentinelRef = useRef<HTMLDivElement | null>(null);
  const [openImage, setOpenImage] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deletingImage, setDeletingImage] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  // Duplicate-upload notice: yellow, distinct from the red NSFW block so the
  // user can tell at a glance which kind of rejection happened.
  const [dupeMsg, setDupeMsg] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{
    phase: "preparing" | "extracting" | "uploading" | "processing";
    fileCount: number;
    bytesLoaded: number;
    bytesTotal: number;
    // Frame extraction progress, only set during the "extracting" phase.
    extractedFrames?: number;
    totalFrames?: number;
    sourceName?: string;
  } | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  const [dragOver, setDragOver] = useState(false);
  // Queue of video files awaiting the trim/sample-rate modal. Only the
  // first one is shown at a time; on confirm or cancel we shift it off
  // and the next one (if any) takes its place.
  const [videoQueue, setVideoQueue] = useState<File[]>([]);
  // Live frame-extraction progress driven from inside the modal so the
  // popup can show a determinate bar (and stay open) while the canvas
  // loop runs through the timeline.
  const [videoExtracting, setVideoExtracting] = useState<{ done: number; total: number } | null>(null);

  // Online-import flow: expanded panel + Openverse preview. Type a
  // word, the backend hits Openverse for 5 CC-licensed images, the
  // user confirms whether they're happy with the look. (The actual
  // import step plugs in later.)
  type ImportImage = {
    id?: string;
    url?: string;
    thumbnail?: string;
    title?: string | null;
    creator?: string | null;
    license?: string | null;
    license_version?: string | null;
    source?: string | null;
    foreign_landing_url?: string | null;
    width?: number | null;
    height?: number | null;
  };
  const [importOpen, setImportOpen] = useState(false);
  const [importDesc, setImportDesc] = useState("");
  const [importResults, setImportResults] = useState<ImportImage[]>([]);
  const [importLoading, setImportLoading] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  // Three-stage flow:
  //   search  , input + 5 preview thumbnails + Yes / Try again
  //   settings, slider for how many to pull + Pull button
  //   review  , full pulled set, click to mark bad, Add good ones
  const [importStage, setImportStage] = useState<"search" | "settings" | "review">("search");
  const [importCount, setImportCount] = useState<number>(50);
  // Tracks thumbnails that failed to load (404, hotlink-blocked, host
  // down). Broken thumbnails get filtered out of the visible preview
  // entirely, empty slots fill the row when there aren't 5 good ones.
  const [importBroken, setImportBroken] = useState<Set<string>>(new Set());
  // Preview window: search returns up to 25, we step through them 5
  // at a time with prev/next arrows.
  const [importPreviewPage, setImportPreviewPage] = useState(0);
  // True once a search request has completed at least once for the
  // current panel session, gates the "No images found" message so
  // it doesn't flash on first open before anything's been searched.
  const [importSearched, setImportSearched] = useState(false);
  const PREVIEW_PAGE_SIZE = 5;
  const PREVIEW_FETCH_COUNT = 25;
  // Larger pulled set (up to 250) shown in the review stage. Separate
  // from the 5-image preview so going back to "search" keeps the
  // preview thumbnails intact.
  const [importPulled, setImportPulled] = useState<ImportImage[]>([]);
  const [importPulling, setImportPulling] = useState(false);
  // URLs of images the user has clicked to mark as bad, those get
  // skipped when we add the rest to the dataset.
  const [importBadUrls, setImportBadUrls] = useState<Set<string>>(new Set());
  // Persistent rejected-URL list (per project, in localStorage). When
  // the user marks an image bad we add it here, and every subsequent
  // search filters those URLs out so the same rejects don't keep
  // resurfacing. The imported URLs come straight from the manifest
  // via results[].source.url, so no separate cache is needed for those.
  const REJECTED_KEY = `openverse_rejected:${name}`;
  const [rejectedUrls, setRejectedUrls] = useState<Set<string>>(() => {
    if (typeof window === "undefined") return new Set();
    try {
      const raw = window.localStorage.getItem(REJECTED_KEY);
      if (!raw) return new Set();
      const arr = JSON.parse(raw);
      return new Set(Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : []);
    } catch {
      return new Set();
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(REJECTED_KEY, JSON.stringify(Array.from(rejectedUrls)));
    } catch {
      /* private mode / storage full, fall back to in-memory only */
    }
  }, [REJECTED_KEY, rejectedUrls]);
  // Combined exclusion set used to filter Openverse responses: the
  // saved rejects plus everything already imported into this project.
  const importedUrls = useMemo(() => {
    const s = new Set<string>();
    for (const r of results) {
      const src = r.source;
      if (src && src.kind === "openverse" && typeof src.url === "string") s.add(src.url);
    }
    return s;
  }, [results]);
  const excludedUrls = useMemo(() => {
    const s = new Set<string>(rejectedUrls);
    for (const u of importedUrls) s.add(u);
    return s;
  }, [rejectedUrls, importedUrls]);
  const pullImportImages = useCallback(async () => {
    const q = importDesc.trim();
    if (!q) return;
    setImportStage("review");
    setImportPulling(true);
    setImportPulled([]);
    setImportBadUrls(new Set());
    try {
      // Over-fetch a little so the post-filter still leaves us close
      // to the count the user asked for. We never go above the
      // backend's hard ceiling (250) so the request stays cheap.
      const overCount = Math.min(250, Math.max(importCount, importCount + Math.min(importCount, 25)));
      const params = new URLSearchParams({ q, count: String(overCount), commercial: "true" });
      const r = await fetch(`${API}/api/openverse/search?${params.toString()}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      const all: ImportImage[] = Array.isArray(d.results) ? d.results : [];
      // Drop URLs already imported into this project or rejected in
      // a prior session, same as the preview filter.
      const filtered = all.filter((c) => !c.url || !excludedUrls.has(c.url));
      // Final dedupe by URL inside the batch, Openverse pagination
      // can surface the same URL across pages and we don't want
      // duplicates in the review grid.
      const seen = new Set<string>();
      const deduped: ImportImage[] = [];
      for (const c of filtered) {
        const key = c.url || c.thumbnail || "";
        if (!key || seen.has(key)) continue;
        seen.add(key);
        deduped.push(c);
      }
      setImportPulled(deduped.slice(0, importCount));
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportPulling(false);
    }
  }, [importDesc, importCount, excludedUrls]);
  const togglePulledBad = useCallback((url: string) => {
    setImportBadUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
    // Mirror into the persistent rejected set so future searches in
    // this project skip URLs the user has already turned down.
    setRejectedUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
  }, []);
  const closeImport = useCallback(() => {
    setImportOpen(false);
    setImportStage("search");
    setImportPulled([]);
    setImportBadUrls(new Set());
    setImportPulling(false);
    setImportSearched(false);
  }, []);
  const [importAdding, setImportAdding] = useState(false);
  const addPulledToDataset = useCallback(async () => {
    const goodUrls = importPulled
      .map((p) => p.url)
      .filter((u): u is string => !!u && !importBadUrls.has(u));
    if (goodUrls.length === 0) return;
    setImportAdding(true);
    setImportError(null);
    try {
      const r = await apiFetch(
        `/api/projects/${name}/images_from_urls?user=${encodeURIComponent(username)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls: goodUrls, query: importDesc }),
        },
      );
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      // Pull the freshly-grown manifest in one go, same shape as the
      // upload-from-disk flow so the project list updates immediately.
      const m: Manifest = await fetch(`${API}/api/projects/${name}`, { cache: "no-store" }).then((x) => x.json());
      setResults(m.results ?? []);
      closeImport();
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportAdding(false);
    }
  }, [importPulled, importBadUrls, name, username, closeImport, importDesc]);
  // Always commercial-only, no opt-out. Users importing images into a
  // labelling project will mostly be training models that ship; better
  // to scope the corpus up front than to surface a licence question
  // after they've already labelled.
  const searchImportImages = useCallback(async () => {
    const q = importDesc.trim();
    if (!q) return;
    setImportLoading(true);
    setImportError(null);
    setImportResults([]);
    setImportStage("search");
    setImportBroken(new Set());
    setImportPreviewPage(0);
    setImportSearched(false);
    try {
      const params = new URLSearchParams({ q, count: String(PREVIEW_FETCH_COUNT), commercial: "true" });
      const r = await fetch(`${API}/api/openverse/search?${params.toString()}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      const all: ImportImage[] = Array.isArray(d.results) ? d.results : [];
      // Drop anything already imported into this project or marked
      // bad in a previous session, same query won't keep surfacing
      // the same rejects.
      const seenInBatch = new Set<string>();
      const candidates: ImportImage[] = [];
      for (const c of all) {
        const key = c.url || c.thumbnail || "";
        if (!key || seenInBatch.has(key)) continue;
        if (excludedUrls.has(c.url ?? "")) continue;
        seenInBatch.add(key);
        candidates.push(c);
      }
      // Probe thumbnails in parallel but render PROGRESSIVELY: as
      // soon as PREVIEW_PAGE_SIZE valid images have come back we
      // drop the skeleton and show the first page. Slower probes
      // continue and stream into pages 2+ as they land.
      //
      // The probe is a two-stage check:
      //   1. Plain load, confirms the URL is reachable AND the
      //      image is bigger than a placeholder (some hosts return
      //      a 200 OK with a 1×1 stub or "image unavailable" tile).
      //   2. CORS-enabled load + pixel-variance sample, catches
      //      uniformly black / white / transparent thumbnails that
      //      pass the dimension check but render as blank in the
      //      grid. If CORS blocks the canvas read we trust stage 1
      //      and let the image through (better to surface a maybe-
      //      blank image than to drop a real one because its host
      //      didn't send Access-Control-Allow-Origin).
      const PROBE_TIMEOUT_MS = 5000;
      const CORS_TIMEOUT_MS = 2500;
      const MIN_DIM = 96;          // anything smaller is almost certainly a placeholder
      const MIN_VARIANCE = 8;      // RGB stdev floor, JPEG noise alone clears this
      const probe = (img: ImportImage): Promise<ImportImage | null> =>
        new Promise((resolve) => {
          const url = img.thumbnail || img.url;
          if (!url) {
            resolve(null);
            return;
          }
          let settled = false;
          const finish = (ok: boolean, reason?: string) => {
            if (settled) return;
            settled = true;
            window.clearTimeout(stageOneTimer);
            if (!ok && reason) {
              // Helpful when debugging which images are getting
              // filtered out, kept at info level, not error.
              console.debug(`[openverse] dropped ${url}: ${reason}`);
            }
            resolve(ok ? img : null);
          };
          const stage1 = new window.Image();
          const stageOneTimer = window.setTimeout(
            () => finish(false, "load timeout"),
            PROBE_TIMEOUT_MS,
          );
          stage1.onerror = () => finish(false, "load error");
          stage1.onload = () => {
            if (stage1.naturalWidth < MIN_DIM || stage1.naturalHeight < MIN_DIM) {
              finish(false, `tiny ${stage1.naturalWidth}x${stage1.naturalHeight}`);
              return;
            }
            // Stage 2, sample pixel variance via canvas. The image
            // has to be re-loaded with crossOrigin set; the previous
            // taint-free load isn't reusable.
            const stage2 = new window.Image();
            stage2.crossOrigin = "anonymous";
            const corsTimer = window.setTimeout(
              () => finish(true), // CORS too slow → trust stage 1
              CORS_TIMEOUT_MS,
            );
            stage2.onerror = () => {
              window.clearTimeout(corsTimer);
              finish(true); // CORS blocked → trust stage 1
            };
            stage2.onload = () => {
              window.clearTimeout(corsTimer);
              try {
                const canvas = document.createElement("canvas");
                canvas.width = 32;
                canvas.height = 32;
                const ctx = canvas.getContext("2d");
                if (!ctx) {
                  finish(true);
                  return;
                }
                ctx.drawImage(stage2, 0, 0, 32, 32);
                const { data } = ctx.getImageData(0, 0, 32, 32);
                // Mean across R/G/B (alpha ignored, fully
                // transparent thumbs collapse to channel zero and
                // get caught by variance below).
                const n = data.length / 4;
                let mR = 0, mG = 0, mB = 0;
                for (let i = 0; i < data.length; i += 4) {
                  mR += data[i]; mG += data[i + 1]; mB += data[i + 2];
                }
                mR /= n; mG /= n; mB /= n;
                let v = 0;
                for (let i = 0; i < data.length; i += 4) {
                  const dr = data[i] - mR;
                  const dg = data[i + 1] - mG;
                  const db = data[i + 2] - mB;
                  v += dr * dr + dg * dg + db * db;
                }
                const stdev = Math.sqrt(v / (n * 3));
                if (stdev < MIN_VARIANCE) {
                  finish(false, `flat thumbnail (stdev ${stdev.toFixed(2)})`);
                  return;
                }
                finish(true);
              } catch {
                // SecurityError on tainted canvas, host didn't
                // honour our crossOrigin request. Trust stage 1.
                finish(true);
              }
            };
            stage2.src = url;
          };
          stage1.src = url;
        });
      // Slot per candidate so we keep original Openverse ordering as
      // probes resolve out of order, no reshuffling on each tick.
      const slot: (ImportImage | null | undefined)[] = new Array(candidates.length).fill(undefined);
      let firstPageDone = false;
      const flush = () => {
        const validNow: ImportImage[] = [];
        for (const x of slot) if (x) validNow.push(x);
        setImportResults(validNow);
        if (!firstPageDone && validNow.length >= PREVIEW_PAGE_SIZE) {
          firstPageDone = true;
          setImportLoading(false);
        }
      };
      await Promise.all(
        candidates.map(async (img, idx) => {
          const result = await probe(img);
          slot[idx] = result; // null = broken, ImportImage = good
          flush();
        }),
      );
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportLoading(false);
      setImportSearched(true);
    }
  }, [importDesc, excludedUrls]);

  // ---- load manifest ----
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API}/api/projects/${name}`);
        if (!r.ok) throw new Error(`http ${r.status}`);
        const m: Manifest = await r.json();
        if (cancelled) return;
        setTags(m.tags ?? []);
        const loadedBox = m.thresholds?.box ?? LABEL_MODES.normal.box;
        const loadedText = m.thresholds?.text ?? LABEL_MODES.normal.text;
        const loadedNms = m.thresholds?.nms ?? LABEL_MODES.normal.nms;
        setBoxThr(loadedBox);
        setTextThr(loadedText);
        setNmsIou(loadedNms);
        // If the loaded values don't match either preset, the project
        // was already on custom thresholds, surface them as a saved
        // Custom pill so the user can flip back after trying a preset.
        const matchesNormal =
          loadedBox === LABEL_MODES.normal.box &&
          loadedText === LABEL_MODES.normal.text &&
          loadedNms === LABEL_MODES.normal.nms;
        const matchesStricter =
          loadedBox === LABEL_MODES.stricter.box &&
          loadedText === LABEL_MODES.stricter.text &&
          loadedNms === LABEL_MODES.stricter.nms;
        if (!matchesNormal && !matchesStricter) {
          setCustomThresholds({ box: loadedBox, text: loadedText, nms: loadedNms });
        }
        setDisplayName(m.name ?? name);
        setOwner(m.owner ?? "");
        setCreatedAt(m.createdAt ?? null);
        setManifestUpdatedAt((m as { updatedAt?: string | null }).updatedAt ?? null);
        setVlmAction(m.vlm_action === "auto_reject" ? "auto_reject" : "manual");
        // Re-hydrate the Downscale/Tile choice persisted by the last run.
        // Fallback resets like the sibling fields so a manifest without the
        // key can't leak a previous project's pick if this effect ever
        // re-runs across projects.
        setTileNative(!!m.tiling?.native);
        setTileSize(m.tiling?.tileSize ?? 1024);
        // Synonyms default to ON when the manifest doesn't specify
        // (older projects predate the toggle).
        setSynonymsEnabled(m.synonyms_enabled !== false);
        const rs = m.results ?? [];
        setResults(rs);
        setVerdicts(m.verdicts ?? {});
        setCover(m.cover ?? null);
        setHasModel(!!m.hasModel);
        setIsPrivate(!!(m as { private?: boolean }).private);
        // editedBoxes is the BoxEditor's source of truth. The SSE
        // `result` events keep it in sync during a live run, but if we
        // missed any (page wasn't focused, browser dropped the
        // connection, etc.) any newly-labelled image would have its
        // detections in results.detections but nothing in editedBoxes ,
        // it'd render with no boxes. Per-image fallback: for any
        // labelled image without an editedBoxes entry, derive one from
        // detections at load time. Existing entries are preserved.
        const eb = m.editedBoxes ?? {};
        const merged: Record<string, EditableBox[]> = { ...eb };
        for (const r of rs) {
          if (!r.pending && !merged[r.image]) {
            merged[r.image] = detectionsToBoxes(r.detections, m.tags ?? []);
          }
        }
        setEditedBoxes(merged);
        setPhase(rs.some((r) => !r.pending) ? "done" : "idle");
      } catch (e) {
        if (!cancelled) setErrorMsg(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [name]);

  // Look up the creator's display name + avatar once we know the username
  // from the manifest. lookupUsers is cached locally for 24h so revisits to
  // the same project cost nothing.
  useEffect(() => {
    if (!owner) {
      setOwnerInfo(null);
      return;
    }
    let cancelled = false;
    (async () => {
      const map = await lookupUsers([owner]);
      if (!cancelled) setOwnerInfo(map[owner.toLowerCase()] ?? null);
    })();
    return () => {
      cancelled = true;
    };
  }, [owner]);

  // ---- project tags == union of labels actually present on any box ----
  // Adding a label in the editor / Review auto-promotes it to a project
  // tag; deleting the last box that used a label drops the tag. Keeps the
  // project vocabulary in lockstep with what's actually labelled, so stale
  // tags don't accumulate.
  useEffect(() => {
    if (!loaded) return;
    const PLACEHOLDERS = new Set(["new", "label", "detecting", "detecting…", "labelling", "labelling…"]);
    const used = new Set<string>();
    for (const arr of Object.values(editedBoxes)) {
      for (const b of arr) {
        const lab = (b.label || "").trim();
        if (!lab || PLACEHOLDERS.has(lab.toLowerCase())) continue;
        used.add(lab);
      }
    }
    const next = Array.from(used).sort();
    setTags((cur) => {
      if (cur.length === next.length && cur.every((t, i) => t === next[i])) return cur;
      return next;
    });
  }, [loaded, editedBoxes]);

  // ---- debounced persistence of metadata edits ----
  useEffect(() => {
    // Read-only viewers never write back to someone else's manifest.
    if (!loaded || readOnly) return;
    // Don't auto-save while a label/lite job is running on the backend.
    // The job mutates manifest.editedBoxes asynchronously; if our debounced
    // PUT lands on top of it with our (stale) view of the state, the new
    // boxes the job is adding get clobbered. The job's completion handler
    // refetches the manifest and we resume auto-saving from fresh state.
    if (activeJobId || liteJob) return;
    const t = window.setTimeout(() => {
      apiFetch(`/api/projects/${name}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tags,
          thresholds: { box: boxThr, text: textThr, nms: nmsIou },
          // Persist the user's stored preference (`vlmAction`), not
          // the effective one, otherwise a downgrade would overwrite
          // their saved choice with "off". Job execution still uses
          // `effectiveVlmAction` so Free users can't actually run AI Review.
          vlm_action: vlmAction,
          synonyms_enabled: synonymsEnabled,
          verdicts,
          editedBoxes,
          cover,
        }),
      }).catch((e) => console.error("save failed", e));
    }, 500);
    return () => window.clearTimeout(t);
  }, [loaded, readOnly, name, tags, boxThr, textThr, nmsIou, vlmAction, synonymsEnabled, verdicts, editedBoxes, cover, activeJobId, liteJob]);

  const compiledPrompt = useMemo(() => tags.map((t) => `a ${t}.`).join(" "), [tags]);

  const pendingCount = useMemo(
    () => results.filter((r) => r.pending).length,
    [results],
  );
  const labeledCount = results.length - pendingCount;

  // Inline warning shown under the labels input when the user types
  // a banned term. Cleared automatically after a few seconds so the
  // row doesn't carry a stale message forever.
  const [labelWarning, setLabelWarning] = useState<string | null>(null);
  useEffect(() => {
    if (!labelWarning) return;
    const t = window.setTimeout(() => setLabelWarning(null), 4000);
    return () => window.clearTimeout(t);
  }, [labelWarning]);

  const addTag = (raw: string) => {
    const t = raw.trim().toLowerCase();
    if (!t) return;
    if (tags.includes(t)) return;
    const bad = containsProfanity(t);
    if (bad) {
      // Don't store anything banned. Show a brief warning and clear
      // the draft so the input is ready for a new attempt.
      setLabelWarning(`"${t}" can't be used as a label.`);
      setDraftTag("");
      return;
    }
    setTags([...tags, t]);
    setDraftTag("");
    setLabelWarning(null);
  };
  const removeTag = (tag: string) => setTags(tags.filter((t) => t !== tag));

  // Extract frames from a video file via a hidden <video> + <canvas>.
  // Seeks to each target time, draws to canvas, encodes to jpeg, and
  // returns File objects with the project's `[name]_frame####.jpg`
  // naming so they slot into the existing image-upload flow without
  // any backend changes.
  const extractVideoFrames = async (
    file: File,
    params: { start: number; end: number; fps: number },
    onProgress: (i: number, total: number) => void,
  ): Promise<File[]> => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    try {
      await new Promise<void>((resolve, reject) => {
        const onLoaded = () => {
          video.removeEventListener("loadeddata", onLoaded);
          resolve();
        };
        const onErr = () => {
          video.removeEventListener("error", onErr);
          reject(new Error("Failed to load video"));
        };
        video.addEventListener("loadeddata", onLoaded);
        video.addEventListener("error", onErr);
      });

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 1920;
      canvas.height = video.videoHeight || 1080;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas 2D context unavailable");

      const baseName = file.name.replace(/\.[^.]+$/, "").replace(/\s+/g, "_");
      const interval = 1 / Math.max(0.1, params.fps);
      const span = Math.max(0, params.end - params.start);
      const totalFrames = Math.max(1, Math.floor(span * params.fps) + 1);
      const out: File[] = [];

      for (let i = 0; i < totalFrames; i++) {
        const t = Math.min(params.end, params.start + i * interval);
        await new Promise<void>((resolve) => {
          let done = false;
          const onSeeked = () => {
            if (done) return;
            done = true;
            video.removeEventListener("seeked", onSeeked);
            resolve();
          };
          video.addEventListener("seeked", onSeeked);
          // Some browsers ignore tiny seeks if the time matches the
          // current frame exactly, so always nudge by a hair.
          video.currentTime = t === video.currentTime ? t + 0.001 : t;
          // Belt-and-braces timeout, if the browser fails to fire
          // `seeked` (rare, but happens on some codecs), don't hang.
          setTimeout(() => {
            if (!done) {
              done = true;
              video.removeEventListener("seeked", onSeeked);
              resolve();
            }
          }, 3000);
        });
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob: Blob | null = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", 0.92),
        );
        if (blob) {
          const fname = `${baseName}_frame${String(i + 1).padStart(4, "0")}.jpg`;
          out.push(new File([blob], fname, { type: "image/jpeg" }));
        }
        onProgress(i + 1, totalFrames);
      }

      return out;
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  const onVideoModalCancel = () => {
    if (videoExtracting) return; // can't cancel mid-extract; keep modal up
    setVideoQueue((prev) => prev.slice(1));
  };

  const onVideoModalConfirm = async (params: { start: number; end: number; fps: number }) => {
    const video = videoQueue[0];
    if (!video) return;
    const expected = Math.max(1, Math.floor((params.end - params.start) * params.fps) + 1);
    setErrorMsg(null);
    // Modal stays mounted (videoQueue is unchanged) so the user sees
    // the progress bar inline. The modal switches to its extracting
    // view via the `extracting` prop.
    setVideoExtracting({ done: 0, total: expected });
    let frames: File[] = [];
    try {
      frames = await extractVideoFrames(video, params, (i, total) => {
        setVideoExtracting({ done: i, total });
      });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      setVideoExtracting(null);
      setVideoQueue((prev) => prev.slice(1));
      return;
    }
    // Extraction done, close the modal, then run the upload using
    // the page-level progress card.
    setVideoExtracting(null);
    setVideoQueue((prev) => prev.slice(1));
    if (frames.length > 0) {
      try {
        await uploadImageFiles(frames);
      } catch (e) {
        setErrorMsg(e instanceof Error ? e.message : String(e));
      }
    }
  };

  // Hard cap on individual video files. Browser-side decode + seek of
  // anything larger gets sluggish or OOMs on smaller machines, and
  // we'd rather refuse early than crash the tab mid-extraction.
  const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

  // Top-level handler for the drop/file-picker. Splits images (which
  // upload directly) from videos (which open the trim/sample modal so
  // the user can pick a range and frame rate before extraction).
  const handleMediaFiles = async (list: FileList | File[] | null) => {
    if (readOnly) return;
    if (!list || (list as FileList).length === 0) return;
    const arr = Array.from(list as ArrayLike<File>);
    const images = arr.filter(isImageFile);
    const videos = arr.filter((f) => f.type.startsWith("video/"));
    const tooBig = videos.filter((v) => v.size > MAX_VIDEO_BYTES);
    const okVideos = videos.filter((v) => v.size <= MAX_VIDEO_BYTES);
    if (tooBig.length > 0) {
      const names = tooBig.map((v) => `${v.name} (${(v.size / (1024 * 1024)).toFixed(1)} MB)`).join(", ");
      setErrorMsg(`Video too large, 100 MB limit: ${names}`);
    }
    if (okVideos.length > 0) {
      setVideoQueue((prev) => [...prev, ...okVideos]);
    }
    if (images.length > 0) {
      await uploadImageFiles(images);
    }
  };

  const uploadImageFiles = async (input: FileList | File[]) => {
    if (readOnly) return;
    const files = Array.from(input as ArrayLike<File>).filter(isImageFile);
    if (files.length === 0) return;
    setErrorMsg(null);
    setDupeMsg(null);
    setUploading(true);
    setUploadProgress({ phase: "preparing", fileCount: files.length, bytesLoaded: 0, bytesTotal: 0 });
    const nsfwRejected: string[] = [];
    const dupeRejected: { file: string; duplicate_of?: string }[] = [];
    try {
      // Resize every file client-side first so the upload is the only
      // network round-trip. Sequential, Promise.all'ing big canvas
      // resizes can stutter the main thread.
      const resized: File[] = [];
      for (const raw of files) {
        resized.push(await resizeForUpload(raw));
      }

      // Single multipart POST containing every file. Backend stages
      // them in a temp area, then runs the per-image NSFW + R2 + manifest
      // pipeline sequentially while we wait on the response. We use XHR
      // here because `fetch` doesn't expose upload-byte progress.
      const fd = new FormData();
      for (const f of resized) fd.append("images", f);

      setUploadProgress({ phase: "uploading", fileCount: files.length, bytesLoaded: 0, bytesTotal: 0 });
      // XHR (not fetch) for upload progress events. Portable build:
      // no accounts, so no bearer to attach.
      const body = await new Promise<{
        added?: string[];
        skipped?: string[];
        rejected?: { file: string; reason: string; duplicate_of?: string }[];
      }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API}/api/projects/${name}/images?user=${encodeURIComponent(username)}`);
        xhr.upload.addEventListener("progress", (ev) => {
          if (!ev.lengthComputable) return;
          setUploadProgress((prev) => prev && {
            ...prev,
            bytesLoaded: ev.loaded,
            bytesTotal: ev.total,
          });
        });
        xhr.upload.addEventListener("load", () => {
          // Bytes are all on the server; backend now starts processing.
          setUploadProgress((prev) => prev && { ...prev, phase: "processing" });
        });
        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText || "{}"));
            } catch (e) {
              reject(e);
            }
          } else {
            reject(new Error(`upload failed: http ${xhr.status}`));
          }
        });
        xhr.addEventListener("error", () => reject(new Error("upload failed: network error")));
        xhr.addEventListener("abort", () => reject(new Error("upload aborted")));
        xhr.send(fd);
      });

      const rej = body.rejected ?? [];
      for (const x of rej) {
        if (x.reason === "duplicate") dupeRejected.push({ file: x.file, duplicate_of: x.duplicate_of });
        else nsfwRejected.push(x.file);
      }
      // Pull the freshly-grown manifest in one go.
      const m: Manifest = await fetch(`${API}/api/projects/${name}`, { cache: "no-store" }).then((x) => x.json());
      // Preserve the user's scroll position. New images insert at
      // the top of the grid (newest first) so the document grows
      // upward; without restoring scrollY the user's view jumps
      // down by however many tiles were added.
      const prevScrollY = window.scrollY;
      setResults(m.results ?? []);
      // The DOM update happens on the next paint, restore in two
      // animation frames to absorb both React's reconciliation and
      // any layout shift from images decoding.
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          window.scrollTo({ top: prevScrollY, behavior: "instant" as ScrollBehavior });
        });
      });

      if (nsfwRejected.length > 0) {
        setErrorMsg(`Blocked by NSFW filter: ${nsfwRejected.join(", ")}`);
      }
      if (dupeRejected.length > 0) {
        // Just the count, listing every filename grew unwieldy on
        // bulk imports and the user already knows which images they
        // tried to upload.
        const n = dupeRejected.length;
        setDupeMsg(`${n} duplicate${n === 1 ? "" : "s"} detected`);
      }
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  // Browser tab-close guard while bytes are flowing, uploads can be
  // tens of MB and dropping the tab mid-flight loses everything.
  useEffect(() => {
    if (!uploading) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [uploading]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setDragOver(false);
      handleMediaFiles(e.dataTransfer.files);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [name],
  );

  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current += 1;
    setDragOver(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) setDragOver(false);
  };

  const autoLabel = async () => {
    if (!compiledPrompt) {
      setErrorMsg("Add at least one label tag");
      return;
    }
    if (pendingCount === 0) return;
    setErrorMsg(null);

    // Plan-quota gate. Block the run if the user has already used up
    // this month's auto-labelling allowance, or if the pending count
    // would push them over. Better to refuse with a clear message
    // than to start a job and have it half-complete.
    const blocked = await checkLabelQuota(pendingCount);
    if (blocked) {
      setErrorMsg(blocked);
      return;
    }

    setPhase("running");
    setProgressIndex(0);
    setProgressTotal(pendingCount);
    setProgressImage("");

    const promptTags = [...tags];

    try {
      const r = await apiFetch(`/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: name,
          kind: "label",
          user: username,
          params: {
            prompt: compiledPrompt,
            tags: promptTags,
            box_threshold: boxThr,
            text_threshold: textThr,
            nms_iou: nmsIou,
            vlm_action: effectiveVlmAction,
            tile_native: tileNative,
            tile_size: tileSize,
          },
        }),
      });
      if (!r.ok) {
        throw new Error(`schedule failed: ${r.status}`);
      }
      const d = await r.json();
      setActiveJobId(d.jobId);
      setJobStartedAt(Date.now());
    } catch (e) {
      setPhase("error");
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  // Kick off a lightweight re-run for a single label. The backend runs GD
  // with only this tag in the prompt, segments + validates the new boxes,
  // and APPENDS them to existing detections (skipping anything that
  // overlaps via IoU). Existing manual edits and verdicts are untouched.
  const runLiteForTag = async (tag: string) => {
    if (readOnly || liteJob || activeJobId) return;
    // Lite re-runs touch every image in the project, so use the full
    // result set as the upper bound for the quota check.
    const liteImageCount = (results || []).length;
    const blocked = await checkLabelQuota(liteImageCount);
    if (blocked) {
      setErrorMsg(blocked);
      return;
    }
    try {
      const r = await apiFetch(`/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: name,
          kind: "label_lite",
          user: username,
          params: {
            tags: [tag],
            box_threshold: boxThr,
            text_threshold: textThr,
            nms_iou: nmsIou,
            vlm_action: effectiveVlmAction,
            tile_native: tileNative,
            tile_size: tileSize,
          },
        }),
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(body || `http ${r.status}`);
      }
      const d = await r.json();
      setLiteJob({ id: d.jobId, tag });
      setLiteStartedAt(Date.now());
      setLiteProgress({ index: 0, total: 0, image: "", added: 0 });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  // Lite-run SSE listener. Tracks progress for the banner, then refetches
  // the manifest on completion so all the new boxes appear.
  useEffect(() => {
    if (!liteJob) return;
    const es = new EventSource(`${API}/api/jobs/${liteJob.id}/events`);
    es.addEventListener("status", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      setLiteProgress((p) => ({ ...p, total: d.total ?? p.total }));
    });
    es.addEventListener("progress", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      setLiteProgress((p) => ({ ...p, index: d.index, total: d.total ?? p.total, image: d.image }));
    });
    es.addEventListener("result", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      setLiteProgress((p) => ({ ...p, added: p.added + (d.added ?? 0) }));
    });
    const finish = async () => {
      es.close();
      try {
        const r = await fetch(`${API}/api/projects/${name}`, { cache: "no-store" });
        if (r.ok) {
          const m: Manifest = await r.json();
          const rs = m.results ?? [];
          setResults(rs);
          setEditedBoxes(m.editedBoxes ?? {});
        }
      } finally {
        setLiteJob(null);
        setLiteStartedAt(null);
      }
    };
    es.addEventListener("done", finish);
    es.addEventListener("complete", finish);
    es.addEventListener("failed", () => {
      es.close();
      setErrorMsg("Lite re-run failed");
      setLiteJob(null);
      setLiteStartedAt(null);
    });
    es.addEventListener("cancelled", () => {
      es.close();
      setLiteJob(null);
      setLiteStartedAt(null);
    });
    es.onerror = () => {
      es.close();
      setLiteJob(null);
      setLiteStartedAt(null);
    };
    return () => es.close();
  }, [liteJob, name]);

  const cancelLiteJob = async () => {
    if (!liteJob || cancellingLite) return;
    setCancellingLite(true);
    try {
      await apiFetch(`/api/projects/${name}/jobs/${liteJob.id}`, { method: "DELETE" });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setCancellingLite(false);
    }
  };

  // Subscribe to the active job's SSE, runs on every job change so that
  // navigating away/back (which re-mounts ProjectView and re-fetches the
  // active job) re-attaches to the live event stream.
  const promptTagsRef = useRef<string[]>([]);
  useEffect(() => {
    promptTagsRef.current = tags;
  }, [tags]);

  useEffect(() => {
    if (!activeJobId) return;
    const es = new EventSource(`${API}/api/jobs/${activeJobId}/events`);
    es.addEventListener("progress", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      setProgressIndex(d.index);
      setProgressTotal(d.total);
      setProgressImage(d.image);
    });
    es.addEventListener("result", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      const r = d.result as Result;
      setResults((prev) => prev.map((x) => (x.image === r.image ? r : x)));
      setEditedBoxes((prev) => ({
        ...prev,
        [r.image]: detectionsToBoxes(r.detections, promptTagsRef.current),
      }));
    });
    const clear = () => {
      setActiveJobId(null);
      setJobStartedAt(null);
    };
    const finish = () => {
      setPhase("done");
      clear();
      es.close();
      // Label Cascade is disabled, see _EMBEDDINGS_ENABLED on the
      // backend. The modal infrastructure is still wired up; flip
      // the backend flag and uncomment the call below to bring it
      // back.
      // void runLabelCascadeScan();
    };
    es.addEventListener("done", finish);
    es.addEventListener("complete", finish);
    es.addEventListener("failed", () => {
      setPhase("error");
      setErrorMsg("Job failed");
      clear();
      es.close();
    });
    es.addEventListener("cancelled", () => {
      setPhase("idle");
      clear();
      es.close();
    });
    es.onerror = () => {
      es.close();
      setPhase((cur) => (cur === "done" ? cur : "error"));
      setErrorMsg((m) => m ?? "connection lost");
      clear();
    };
    return () => es.close();
  }, [activeJobId]);

  // On mount (or project switch), check whether a label/segment job is
  // already running for this project, re-attaches to it after navigation.
  useEffect(() => {
    if (!loaded) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API}/api/projects/${name}/jobs/active`, { cache: "no-store" });
        if (!r.ok) return;
        const j = await r.json();
        if (cancelled || !j) return;
        setActiveJobId(j.id);
        setPhase("running");
        setProgressIndex(j.progress?.index ?? 0);
        setProgressTotal(j.progress?.total ?? j.n_images ?? 0);
        setProgressImage(j.progress?.image ?? "");
        // Reconstruct the wall-clock anchor: startedAt is an ISO string;
        // fall back to "now minus elapsedS" if startedAt is absent.
        const started = j.startedAt ? Date.parse(j.startedAt) : null;
        if (started && !Number.isNaN(started)) setJobStartedAt(started);
        else if (typeof j.elapsedS === "number") setJobStartedAt(Date.now() - j.elapsedS * 1000);
        else setJobStartedAt(Date.now());
      } catch {
        // Ignore, page just won't show a resumed job.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loaded, name]);

  const cancelActiveJob = async () => {
    if (!activeJobId || cancellingJob) return;
    const id = activeJobId;
    setCancellingJob(true);
    // Clear UI state immediately, the user clicked cancel, the card
    // should disappear right away. The backend SSE will catch up later
    // (and harmlessly try to update state we've already reset).
    setPhase("idle");
    setActiveJobId(null);
    setJobStartedAt(null);
    setProgressIndex(0);
    setProgressTotal(0);
    setProgressImage("");
    try {
      await apiFetch(`/api/projects/${name}/jobs/${id}`, { method: "DELETE" });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setCancellingJob(false);
    }
  };

  const deleteImage = async (image: string) => {
    if (readOnly) return;
    try {
      const r = await apiFetch(`/api/projects/${name}/images/${encodeURIComponent(image)}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error(`delete failed: ${r.status}`);
      // Local strip first for an instant UI response.
      setResults((prev) => prev.filter((x) => x.image !== image));
      setEditedBoxes((prev) => {
        const next = { ...prev };
        delete next[image];
        return next;
      });
      setVerdicts((prev) => {
        const next = { ...prev };
        delete next[image];
        return next;
      });
      // Then resync with the backend so any race with the auto-save
      // PUT (which would otherwise echo the stale results array back
      // when the next manifest read landed) doesn't resurrect the card.
      // This is what was forcing the user to reload to see the delete.
      try {
        const m: Manifest = await fetch(`${API}/api/projects/${name}`, { cache: "no-store" }).then((x) => x.json());
        setResults(m.results ?? []);
        setEditedBoxes(m.editedBoxes ?? {});
        setVerdicts(m.verdicts ?? {});
      } catch {
        /* keep the optimistic state if the resync fails */
      }
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const renameLabel = async (oldLabel: string, newLabel: string) => {
    if (readOnly) return;
    const old = oldLabel.trim();
    const next = newLabel.trim();
    if (!old || !next || old.toLowerCase() === next.toLowerCase()) return;
    const bad = containsProfanity(next);
    if (bad) {
      setLabelWarning(`"${next}" can't be used as a label.`);
      // Throw so the TagChip's `save()` keeps the input in edit
      // mode, the user gets to fix the value instead of the chip
      // collapsing back as if nothing happened.
      throw new Error("profanity blocked");
    }
    try {
      const r = await apiFetch(`/api/projects/${name}/labels/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_label: old, new_label: next }),
      });
      if (!r.ok) throw new Error(`rename failed: ${r.status}`);
      // Refetch the manifest so tags, detections, and editedBoxes
      // all land in lockstep, the rename changes all three and
      // refetching is simpler than mirroring the rewrite logic in
      // three useState calls.
      const m: Manifest = await fetch(`${API}/api/projects/${name}`, { cache: "no-store" }).then((x) => x.json());
      setTags(m.tags ?? []);
      setResults(m.results ?? []);
      setEditedBoxes(m.editedBoxes ?? {});
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const clearImageLabels = async (image: string) => {
    if (readOnly) return;
    try {
      const r = await apiFetch(
        `/api/projects/${name}/images/${encodeURIComponent(image)}/clear_labels`,
        { method: "POST" },
      );
      if (!r.ok) throw new Error(`clear failed: ${r.status}`);
      // Reset the image to unlabelled in local state, drop edits,
      // verdict, detections, and the annotated preview reference. The
      // image stays in the manifest but the grid renders it as
      // "Unlabelled" again and it rejoins the auto-label queue.
      setResults((prev) =>
        prev.map((x) =>
          x.image === image
            ? { ...x, detections: [], annotated: null, pending: true }
            : x,
        ),
      );
      setEditedBoxes((prev) => {
        const next = { ...prev };
        delete next[image];
        return next;
      });
      setVerdicts((prev) => {
        const next = { ...prev };
        delete next[image];
        return next;
      });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      throw e;
    }
  };

  // Defined inline so filterCounts can use it without depending on the
  // useCallback below being declared first.
  const isVlmRejectedFor = useCallback(
    (image: string) => (editedBoxes[image] ?? []).some((b) => b.validation?.match === false),
    [editedBoxes],
  );

  const filterCounts = useMemo(
    () => ({
      all: results.length,
      unlabeled: results.filter((r) => r.pending).length,
      good: results.filter((r) => verdicts[r.image] === "good").length,
      bad: results.filter((r) => verdicts[r.image] === "bad").length,
      unrated: results.filter((r) => !r.pending && !verdicts[r.image]).length,
      vlm: results.filter((r) => !r.pending && isVlmRejectedFor(r.image)).length,
    }),
    [results, verdicts, isVlmRejectedFor],
  );

  const filteredResults = useMemo(
    () =>
      // Newest uploads first. The manifest stores results in upload order, so
      // reversing gives most-recent-at-top.
      [...results].reverse().filter((r) => {
        if (filter === "all") return true;
        if (filter === "unlabeled") return !!r.pending;
        if (filter === "unrated") return !r.pending && !verdicts[r.image];
        if (filter === "vlm") return !r.pending && isVlmRejectedFor(r.image);
        return verdicts[r.image] === filter;
      }),
    [results, filter, verdicts, isVlmRejectedFor],
  );

  // Reset progressive rendering when the visible set changes shape
  // (filter flip, fresh project load). Without this you'd land on a
  // small filter group already showing 200+ ghost slots from the
  // previous page's scroll.
  useEffect(() => {
    setDisplayLimit(GRID_PAGE_SIZE);
  }, [filter, name]);

  // Bump displayLimit whenever the sentinel below the grid scrolls
  // into view. IntersectionObserver fires once per crossing, so the
  // load-more is discrete (no thrashing) and uses no scroll listener.
  useEffect(() => {
    const node = gridSentinelRef.current;
    if (!node) return;
    if (displayLimit >= filteredResults.length) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setDisplayLimit((cur) => Math.min(cur + GRID_PAGE_SIZE, filteredResults.length));
          }
        }
      },
      { rootMargin: "400px" },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, [displayLimit, filteredResults.length]);

  const visibleResults = useMemo(
    () => filteredResults.slice(0, displayLimit),
    [filteredResults, displayLimit],
  );

  const openResult = useMemo(
    () => (openImage ? results.find((r) => r.image === openImage) : null),
    [openImage, results],
  );

  const reviewCounts = useMemo(
    () => ({
      unrated: results.filter((r) => !r.pending && !verdicts[r.image]).length,
      vlm: results.filter((r) => !r.pending && isVlmRejectedFor(r.image)).length,
      good: results.filter((r) => verdicts[r.image] === "good").length,
      bad: results.filter((r) => verdicts[r.image] === "bad").length,
      all: results.filter((r) => !r.pending).length,
    }),
    [results, verdicts, isVlmRejectedFor],
  );

  const startReview = (scope: ReviewScope = "unrated") => {
    const list = results.filter((r) => {
      if (r.pending) return false;
      if (scope === "unrated") return !verdicts[r.image];
      if (scope === "vlm") return isVlmRejectedFor(r.image);
      if (scope === "good") return verdicts[r.image] === "good";
      if (scope === "bad") return verdicts[r.image] === "bad";
      return true;
    });
    if (list.length === 0) return;
    setReviewScope(scope);
    setReviewList(list);
    setReviewing(true);
  };
  // Kept for re-enabling when _EMBEDDINGS_ENABLED flips back on.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const runLabelCascadeScan = async () => {
    if (cascadeScanning) return;
    setCascadeScanning(true);
    try {
      const r = await fetch(`${API}/api/projects/${name}/embeddings/scan`, { method: "POST" });
      if (!r.ok) return;
      const d = await r.json();
      const groups: CascadeGroup[] = Array.isArray(d?.groups) ? d.groups : [];
      // Only surface the modal when there's something to look at.
      // The scan itself indexes new boxes into the embedding store
      // either way, so it's not wasted work even when there's
      // nothing visually duplicated.
      if (groups.length > 0) {
        setCascadeGroups(groups);
      }
    } catch (e) {
      console.debug("[label-cascade] scan failed", e);
    } finally {
      setCascadeScanning(false);
    }
  };

  const applyCascadeRelabel = (targets: { image: string; box_id: string }[], newLabel: string) => {
    const cascadeValidation = {
      match: true,
      confidence: 1,
      reason: "Label Cascade",
      source: "cascade" as const,
      kind: "cascade" as const,
    };
    setEditedBoxes((prev) => {
      const next = { ...prev };
      for (const t of targets) {
        const list = next[t.image];
        if (!list) continue;
        let touched = false;
        const updated = list.map((b) => {
          if (b.id !== t.box_id) return b;
          touched = true;
          return { ...b, label: newLabel, validation: cascadeValidation };
        });
        if (touched) next[t.image] = updated;
      }
      return next;
    });
  };

  const closeReview = () => {
    setReviewing(false);
    setReviewList(null);
  };

  // Tally boxes that would be too small after scaling to the chosen input
  // shape. Keyed off both the boxes and inputSize so it recomputes when
  // either changes, no manual recompute needed.
  const sizeByImage = useMemo(() => {
    const m: Record<string, { width: number; height: number }> = {};
    for (const r of results) m[r.image] = r.size;
    return m;
  }, [results]);

  const boxSummary = useMemo(() => {
    let warn = 0;
    let fail = 0;
    for (const [img, arr] of Object.entries(editedBoxes)) {
      const sz = sizeByImage[img];
      if (!sz) continue;
      for (const b of arr) {
        const s = statusFor(scaledMinSide(b, sz.width, sz.height, inputSize));
        if (s === "warn") warn += 1;
        else if (s === "fail") fail += 1;
      }
    }
    return { warn, fail };
  }, [editedBoxes, sizeByImage, inputSize]);

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <section className="mx-auto max-w-6xl px-6 pt-12 pb-2">
        <button
          onClick={onClose}
          className="text-xs uppercase tracking-wider text-[var(--muted)] hover:text-foreground"
        >
          ← Projects
        </button>
        <div className="mt-3 flex items-end justify-between gap-4 flex-wrap">
          <div>
            <div className="text-xs text-foreground/40">Project</div>
            <h1 className="text-4xl md:text-5xl font-light tracking-tight mt-1 leading-snug pb-0.5 flex items-center gap-3">
              <span>{displayName}</span>
              {isPrivate && (
                <svg
                  viewBox="0 0 24 24"
                  width="22"
                  height="22"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-amber-300/80 shrink-0"
                  aria-label="Private project"
                  role="img"
                >
                  <title>Private project</title>
                  <rect x="4" y="11" width="16" height="10" rx="2" />
                  <path d="M8 11V7a4 4 0 0 1 8 0v4" />
                </svg>
              )}
            </h1>
          </div>
          {readOnly ? (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setExportOpen(true)}
                className="text-xs text-foreground/70 hover:text-foreground border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 rounded-full px-4 py-2 transition-colors"
              >
                Export
              </button>
              <span className="text-xs text-foreground/60 border border-foreground/15 bg-foreground/5 rounded-full px-3 py-1.5">
                Read-only
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setExportOpen(true)}
                className="text-xs text-foreground/70 hover:text-foreground border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 rounded-full px-4 py-2 transition-colors"
              >
                Export
              </button>
              <button
                onClick={() => setSettingsOpen(true)}
                className="text-xs text-foreground/70 hover:text-foreground border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 rounded-full px-4 py-2 transition-colors"
              >
                Settings
              </button>
            </div>
          )}
        </div>

        <ProjectMeta
          owner={owner}
          ownerInfo={ownerInfo}
          createdAt={createdAt}
          tags={tags}
          paletteSeed={displayName}
        />
      </section>

      <section className="mx-auto max-w-6xl px-6 mt-8">
        <div className="flex items-center gap-4 flex-wrap">
          {/* Target Input Shape: pill ↔ picker swap. The wrapper's
              max-width animates between the pill width (collapsed)
              and the picker width (expanded), this changes the
              wrapper's actual layout box, so flex siblings (Hide-
              Small + info) shift rightward by exactly the picker's
              growth on every animation frame. Pill and picker are
              CSS-grid-stacked in the same cell with crossfading
              opacity. Asymmetric easing: zippy + slight overshoot
              on expand, longer & gentler on close. */}
          <div
            className="flex items-center gap-3"
            onMouseEnter={onShapeEnter}
            onMouseLeave={onShapeLeave}
          >
            <span className="text-sm text-foreground/55 shrink-0">Target input shape</span>
            <div
              className="overflow-hidden grid items-center"
              style={{
                gridTemplateAreas: '"stack"',
                // Collapsed width fits the longest pill text
                // ("640x640") with a small ring of breathing room.
                // Expanded width covers the full segmented row of
                // INPUT_SHAPES (~680 px on desktop). The flex
                // parent's `flex-wrap` handles narrower viewports.
                maxWidth: shapeHovered ? "720px" : "92px",
                transition: shapeHovered
                  // Expand, quick with a touch of overshoot.
                  ? "max-width 360ms cubic-bezier(0.34, 1.30, 0.64, 1)"
                  // Close, longer, smooth ease-out, no overshoot.
                  : "max-width 600ms cubic-bezier(0.40, 0.00, 0.20, 1)",
              }}
            >
              {/* Compact pill, visible while not hovered. Clickable
                  so impatient users can pop the picker open without
                  waiting for the hover-intent timer. */}
              <button
                type="button"
                onClick={() => setShapeHovered(true)}
                style={{
                  gridArea: "stack",
                  opacity: shapeHovered ? 0 : 1,
                  // Invisible elements must opt out of pointer events,
                  // otherwise the picker (which is on top of the
                  // pill in source order) captures clicks meant for
                  // the pill, and vice versa. Without this the user
                  // would click the visible pill but the click would
                  // land on whichever picker option happened to be
                  // at that pixel, silently setting inputSize.
                  pointerEvents: shapeHovered ? "none" : "auto",
                  transition: shapeHovered
                    ? "opacity 150ms cubic-bezier(0.40, 0.00, 0.20, 1)"
                    : "opacity 600ms cubic-bezier(0.40, 0.00, 0.20, 1)",
                }}
                className="appearance-none bg-transparent border-0 p-0 text-left cursor-pointer"
                aria-label="Expand input-shape picker"
              >
                <span className="inline-flex items-center font-mono text-xs text-foreground/85 px-3 py-1 rounded-full border border-foreground/15 bg-foreground/[0.04] hover:bg-foreground/[0.06] hover:border-foreground/25 whitespace-nowrap transition-colors">
                  {inputSize}
                </span>
              </button>
              {/* Full picker, visible on hover. Pointer events gated
                  by shapeHovered so the invisible pill underneath
                  doesn't get bypassed. */}
              <div
                style={{
                  gridArea: "stack",
                  opacity: shapeHovered ? 1 : 0,
                  pointerEvents: shapeHovered ? "auto" : "none",
                  transition: shapeHovered
                    ? "opacity 300ms cubic-bezier(0.34, 1.30, 0.64, 1)"
                    : "opacity 600ms cubic-bezier(0.40, 0.00, 0.20, 1)",
                }}
              >
                <div className="whitespace-nowrap">
                  <SegmentedControl
                    value={inputSize}
                    onChange={setInputSize}
                    options={INPUT_SHAPES.map((s) => ({ value: s, label: s }))}
                  />
                </div>
              </div>
            </div>
          </div>
          {/* Hide-Small toggle + info popover. Wrapped in a flex
              container that's a flex sibling of the picker; as the
              picker expands inline, the gap-3 spacing stays
              consistent and these two slide rightward as the
              browser re-flows the row. */}
          <HideSmallToggle
            value={hideSmall}
            onChange={setHideSmall}
            warn={boxSummary.warn}
            fail={boxSummary.fail}
          />
          <HideSmallHelpPopover />
        </div>
        {!hideSmall && (
          <div className="mt-2">
            <BoxSizeTip warn={boxSummary.warn} fail={boxSummary.fail} />
          </div>
        )}
      </section>

      {!readOnly && (
        <div className="border-b border-foreground/10 mt-6">
          <div className="mx-auto max-w-6xl px-6 flex gap-1">
            <TabButton active={tab === "label"} onClick={() => setTab("label")}>
              Label
            </TabButton>
            <TabButton active={tab === "train"} onClick={() => setTab("train")}>
              Train
            </TabButton>
            <TabButton active={tab === "deploy"} onClick={() => setTab("deploy")}>
              Deploy
            </TabButton>
            <TabButton active={tab === "optimise"} onClick={() => setTab("optimise")}>
              Optimise
            </TabButton>
          </div>
        </div>
      )}

      {tab === "label" && (
        <section className="mx-auto max-w-6xl px-6 pt-10 pb-24 grid gap-10">
          {!readOnly && (<>
          {uploadProgress && <UploadProgressCard progress={uploadProgress} />}
          {/* Visual order: dropzone first, then the "Don't have images?"
              card. We keep the JSX in its existing order and flip the
              column with flex-col-reverse so the import card's expanding
              animation logic and refs don't have to move. */}
          <div className="flex flex-col-reverse gap-10">
            {/* Online-import card. Click anywhere on the row to expand a
                panel below where the LLM generates dataset-style search
                queries. Animation uses the grid-template-rows 0fr→1fr
                trick so the panel expands to its real height and the
                surrounding layout reflows smoothly. */}
            <div
              className="rounded-3xl border border-foreground/10 overflow-hidden"
              style={{
                background:
                  "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 45%, rgba(255,255,255,0) 100%), #141416",
                boxShadow:
                  "0 1px 0 rgb(var(--foreground-rgb) / 0.05) inset, 0 30px 60px -30px rgb(var(--shadow-rgb) / 0.6)",
              }}
            >
              <button
                type="button"
                onClick={() => {
                  if (importOpen) {
                    closeImport();
                  } else {
                    setImportOpen(true);
                  }
                }}
                aria-expanded={importOpen}
                className="group flex w-full items-center justify-between gap-4 px-7 py-6 text-left transition-colors duration-150 hover:bg-foreground/[0.02]"
              >
                <span className="flex items-center gap-4">
                  <span className="grid h-10 w-10 place-items-center rounded-full bg-foreground/[0.05] border border-foreground/10 text-foreground/75 shrink-0 group-hover:text-foreground transition-colors">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="12" cy="12" r="9" />
                      <path d="M3 12h18" />
                      <path d="M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" />
                    </svg>
                  </span>
                  <span className="flex flex-col">
                    <span className="text-base font-semibold text-[var(--foreground)] tracking-tight">Don&rsquo;t have images?</span>
                    <span className="text-sm text-foreground/55">Import from online sources</span>
                  </span>
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-foreground/15 bg-foreground/[0.05] px-3.5 py-1.5 text-xs font-medium text-foreground/80 transition-all duration-150 group-hover:bg-foreground/[0.10] group-hover:border-foreground/25 group-hover:text-foreground">
                  {importOpen ? "Close" : "Get started"}
                  <svg
                    viewBox="0 0 24 24"
                    width="13"
                    height="13"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                    className="transition-transform duration-300"
                    style={{ transform: importOpen ? "rotate(90deg)" : "rotate(0deg)" }}
                  >
                    <path d="M9 6l6 6-6 6" />
                  </svg>
                </span>
              </button>
              <div
                className="grid transition-[grid-template-rows] duration-300 ease-out"
                style={{ gridTemplateRows: importOpen ? "1fr" : "0fr" }}
              >
                <div className="min-h-0 overflow-hidden">
                  <div className="px-6 pb-6 pt-1 flex flex-col border-t border-foreground/[0.06]">
                    <div className="grid gap-2 pt-4">
                      <label className="text-xs uppercase tracking-wider text-foreground/55" htmlFor="import-desc">
                        What images are you looking for?
                      </label>
                      <div className="flex items-stretch gap-2">
                        <input
                          id="import-desc"
                          type="text"
                          value={importDesc}
                          onChange={(e) => setImportDesc(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !importLoading && importDesc.trim()) {
                              e.preventDefault();
                              searchImportImages();
                            }
                          }}
                          placeholder="e.g. potholes, hard hats, ripe strawberries"
                          className="flex-1 rounded-xl border border-foreground/10 bg-foreground/[0.03] px-4 py-3 text-sm text-[var(--foreground)] placeholder:text-foreground/40 outline-none transition-colors hover:border-foreground/20 focus:border-foreground/30 focus:bg-foreground/[0.05]"
                        />
                        <button
                          type="button"
                          onClick={searchImportImages}
                          disabled={!importDesc.trim() || importLoading}
                          className="rounded-xl border border-foreground/15 bg-foreground/[0.05] px-5 text-sm font-medium text-foreground/90 transition-all duration-150 hover:bg-foreground/[0.10] hover:border-foreground/25 disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          {importLoading ? "Searching…" : "Search"}
                        </button>
                      </div>
                      <p className="flex items-center flex-wrap gap-x-3 gap-y-1 text-xs text-foreground/55 mt-1">
                        <span className="inline-flex items-center gap-1.5">
                          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="text-emerald-300/80">
                            <path d="M5 13l4 4L19 7" />
                          </svg>
                          All results are Creative Commons licensed and free for commercial use.
                        </span>
                        <span className="text-[11px] text-foreground/40">
                          By using this you agree to the{" "}
                          <a
                            href="/openverse"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-foreground/55 hover:text-foreground underline underline-offset-2"
                          >
                            Openverse policy
                          </a>
                          .
                        </span>
                      </p>
                    </div>
                    {/* Animated reveal for the error chip, same
                        grid-template-rows trick as the outer panel so
                        the layout never jumps when an error appears.
                        Margin lives inside the overflow-hidden so a
                        collapsed wrapper takes zero space. */}
                    <div
                      className="grid transition-[grid-template-rows] duration-300 ease-out"
                      style={{ gridTemplateRows: importError ? "1fr" : "0fr" }}
                    >
                      <div className="min-h-0 overflow-hidden">
                        <div className="mt-4 rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-xs text-red-200">
                          {importError}
                        </div>
                      </div>
                    </div>
                    {/* Animated reveal for the results section. Wraps
                        the thumbnails grid + the stage-specific row
                        below, so search → settings transitions all
                        live inside a single height-animated container. */}
                    <div
                      className="grid transition-[grid-template-rows] duration-300 ease-out"
                      style={{ gridTemplateRows: importResults.length > 0 || importLoading || importSearched ? "1fr" : "0fr" }}
                    >
                      <div className="min-h-0 overflow-hidden">
                        <div className="mt-4 grid gap-3">
                          {(() => {
                            // Drop unavailable images entirely, the
                            // user wants no "Unavailable" placeholder
                            // showing through. Window the rest into
                            // pages of 5 with prev/next arrows; pages
                            // with fewer than 5 valid images are
                            // padded with empty glass slots so the row
                            // stays tidy.
                            const valid = importResults.filter((r) => {
                              const k = r.url || r.thumbnail || "";
                              return !!(r.thumbnail || r.url) && !importBroken.has(k);
                            });
                            // Empty state, search ran, nothing to
                            // show. Either Openverse returned zero
                            // results or every URL turned out to be
                            // broken / hotlink-blocked.
                            if (!importLoading && importSearched && valid.length === 0) {
                              return (
                                <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
                                  <span className="grid h-10 w-10 place-items-center rounded-full bg-foreground/[0.04] border border-foreground/10 text-foreground/45">
                                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                      <circle cx="11" cy="11" r="7" />
                                      <path d="m20 20-3.5-3.5" />
                                    </svg>
                                  </span>
                                  <span className="text-sm text-foreground/70">No images found</span>
                                  <span className="text-xs text-foreground/40">Try a different search term.</span>
                                </div>
                              );
                            }
                            const totalPages = Math.max(1, Math.ceil(valid.length / PREVIEW_PAGE_SIZE));
                            const safePage = Math.min(importPreviewPage, totalPages - 1);
                            const start = safePage * PREVIEW_PAGE_SIZE;
                            const visible = valid.slice(start, start + PREVIEW_PAGE_SIZE);
                            const slots: (ImportImage | null)[] = Array.from({ length: PREVIEW_PAGE_SIZE }).map(
                              (_, i) => visible[i] ?? null,
                            );
                            const hasPrev = safePage > 0;
                            const hasNext = safePage < totalPages - 1;
                            return (
                              <div className="relative pb-5">
                                <ul className="grid gap-2 grid-cols-2 sm:grid-cols-5">
                                  {importLoading
                                    ? Array.from({ length: 5 }).map((_, i) => (
                                        <li
                                          key={`skel-${i}`}
                                          className="aspect-square rounded-lg border border-foreground/[0.06] bg-foreground/[0.03] animate-pulse"
                                          style={{ animationDelay: `${i * 80}ms` }}
                                        />
                                      ))
                                    : slots.map((img, i) => {
                                        if (!img) {
                                          return (
                                            <li
                                              key={`empty-${safePage}-${i}`}
                                              className="aspect-square rounded-lg border border-foreground/[0.06] bg-foreground/[0.02]"
                                            />
                                          );
                                        }
                                        const key = img.url || img.thumbnail || `idx-${start + i}`;
                                        return (
                                          <li
                                            key={`${safePage}-${i}-${img.id ?? img.url}`}
                                            className="group relative aspect-square overflow-hidden rounded-lg border border-foreground/[0.06] bg-foreground/[0.02]"
                                            title={[img.title, img.creator ? `by ${img.creator}` : null, img.license ? `${img.license}${img.license_version ? ` ${img.license_version}` : ""}` : null]
                                              .filter(Boolean)
                                              .join(" · ")}
                                          >
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img
                                              src={img.thumbnail || img.url}
                                              alt={img.title || ""}
                                              className="absolute inset-0 h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
                                              loading="lazy"
                                              onError={() => {
                                                setImportBroken((prev) => {
                                                  const next = new Set(prev);
                                                  next.add(key);
                                                  return next;
                                                });
                                              }}
                                            />
                                            {(img.creator || img.license) && (
                                              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/80 via-black/30 to-transparent px-2 py-1.5 text-[10px] text-foreground/85 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                                                <span className="truncate">{img.creator || ""}</span>
                                                {img.license && (
                                                  <span className="uppercase tracking-wider text-foreground/65">{img.license}</span>
                                                )}
                                              </div>
                                            )}
                                          </li>
                                        );
                                      })}
                                </ul>
                                {!importLoading && totalPages > 1 && (
                                  <>
                                    {/* Prev / Next: glass pills sitting
                                        inside the grid margin, vertically
                                        centered. Larger hit area, subtle
                                        lift on hover, chevron nudges in
                                        its direction. Hidden entirely
                                        (not just disabled) at the ends so
                                        the row doesn't feel cluttered
                                        when there's nowhere to go. */}
                                    {hasPrev && (
                                      <button
                                        type="button"
                                        onClick={() => setImportPreviewPage((p) => Math.max(0, p - 1))}
                                        aria-label="Previous preview page"
                                        className="group/nav absolute left-1.5 top-[calc(50%-10px)] -translate-y-1/2 grid h-11 w-11 place-items-center rounded-full text-[var(--foreground)] transition-all duration-200 hover:scale-105 active:scale-95 backdrop-blur-2xl"
                                        style={{
                                          background:
                                            "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.18) 0%, rgb(var(--foreground-rgb) / 0.08) 100%)",
                                          border: "1px solid rgb(var(--foreground-rgb) / 0.22)",
                                          boxShadow:
                                            "0 1px 0 rgb(var(--foreground-rgb) / 0.18) inset, 0 6px 18px rgb(var(--shadow-rgb) / 0.45), 0 0 0 1px rgb(var(--shadow-rgb) / 0.05)",
                                        }}
                                      >
                                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="transition-transform duration-200 group-hover/nav:-translate-x-0.5">
                                          <path d="m15 18-6-6 6-6" />
                                        </svg>
                                      </button>
                                    )}
                                    {hasNext && (
                                      <button
                                        type="button"
                                        onClick={() => setImportPreviewPage((p) => Math.min(totalPages - 1, p + 1))}
                                        aria-label="Next preview page"
                                        className="group/nav absolute right-1.5 top-[calc(50%-10px)] -translate-y-1/2 grid h-11 w-11 place-items-center rounded-full text-[var(--foreground)] transition-all duration-200 hover:scale-105 active:scale-95 backdrop-blur-2xl"
                                        style={{
                                          background:
                                            "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.18) 0%, rgb(var(--foreground-rgb) / 0.08) 100%)",
                                          border: "1px solid rgb(var(--foreground-rgb) / 0.22)",
                                          boxShadow:
                                            "0 1px 0 rgb(var(--foreground-rgb) / 0.18) inset, 0 6px 18px rgb(var(--shadow-rgb) / 0.45), 0 0 0 1px rgb(var(--shadow-rgb) / 0.05)",
                                        }}
                                      >
                                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="transition-transform duration-200 group-hover/nav:translate-x-0.5">
                                          <path d="m9 6 6 6-6 6" />
                                        </svg>
                                      </button>
                                    )}
                                    {/* Page-dot indicator under the row.
                                        Filled dot for the current page,
                                        hollow for the others, quicker
                                        to scan than "2 / 5" text. */}
                                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex items-center gap-1.5">
                                      {Array.from({ length: totalPages }).map((_, p) => (
                                        <button
                                          key={`dot-${p}`}
                                          type="button"
                                          onClick={() => setImportPreviewPage(p)}
                                          aria-label={`Go to preview page ${p + 1}`}
                                          aria-current={p === safePage}
                                          className={[
                                            "h-1.5 rounded-full transition-all duration-200",
                                            p === safePage
                                              ? "w-4 bg-foreground/85"
                                              : "w-1.5 bg-foreground/25 hover:bg-foreground/45",
                                          ].join(" ")}
                                        />
                                      ))}
                                    </div>
                                  </>
                                )}
                              </div>
                            );
                          })()}
                          {/* Stage 1, Happy with these? Yes / Try again.
                              Animated together with the settings stage
                              below so the swap reads as one motion. */}
                          <div
                            className="grid transition-[grid-template-rows] duration-300 ease-out"
                            style={{ gridTemplateRows: !importLoading && importResults.length > 0 && importStage === "search" ? "1fr" : "0fr" }}
                          >
                            <div className="min-h-0 overflow-hidden">
                              <div className="flex items-center justify-between gap-3 pt-1">
                                <span className="text-sm text-foreground/70">Happy with these?</span>
                                <div className="flex items-center gap-2">
                                  <button
                                    type="button"
                                    onClick={closeImport}
                                    className="rounded-full border border-foreground/10 bg-foreground/[0.04] px-3.5 py-1.5 text-xs font-medium text-foreground/75 transition-all duration-150 hover:bg-foreground/[0.08] hover:border-foreground/20 hover:text-foreground"
                                  >
                                    Try again
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setImportStage("settings")}
                                    className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/40 bg-emerald-500/15 px-3.5 py-1.5 text-xs font-medium text-emerald-100 transition-all duration-150 hover:bg-emerald-500/25 hover:border-emerald-300/60"
                                  >
                                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                      <path d="M5 13l4 4L19 7" />
                                    </svg>
                                    Yes, this is what I&rsquo;m looking for
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                          {/* Stage 2, settings (count slider + Pull). */}
                          <div
                            className="grid transition-[grid-template-rows] duration-300 ease-out"
                            style={{ gridTemplateRows: !importLoading && importResults.length > 0 && importStage === "settings" ? "1fr" : "0fr" }}
                          >
                            <div className="min-h-0 overflow-hidden">
                              <div className="grid gap-4 pt-2 mt-1 border-t border-foreground/[0.06]">
                                <div className="grid gap-2 pt-3">
                                  <div className="flex items-center justify-between gap-3">
                                    <label className="text-xs uppercase tracking-wider text-foreground/55" htmlFor="import-count">
                                      Number of images to pull
                                    </label>
                                    <span className="font-mono tabular-nums text-sm text-foreground/90">{importCount}</span>
                                  </div>
                                  <input
                                    id="import-count"
                                    type="range"
                                    min={10}
                                    max={250}
                                    step={10}
                                    value={importCount}
                                    onChange={(e) => setImportCount(parseInt(e.target.value, 10))}
                                    className="w-full accent-white"
                                  />
                                  <div className="flex items-center justify-between text-[10px] text-foreground/40 font-mono tabular-nums">
                                    <span>10</span>
                                    <span>250</span>
                                  </div>
                                </div>
                                <div className="flex items-center justify-end gap-2 pt-1">
                                  <button
                                    type="button"
                                    onClick={closeImport}
                                    className="rounded-full border border-foreground/10 bg-foreground/[0.04] px-3.5 py-1.5 text-xs font-medium text-foreground/75 transition-all duration-150 hover:bg-foreground/[0.08] hover:border-foreground/20 hover:text-foreground"
                                  >
                                    Cancel
                                  </button>
                                  <button
                                    type="button"
                                    onClick={pullImportImages}
                                    className="inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-medium text-black bg-white transition-all duration-150 hover:bg-foreground/90 shadow-[var(--shadow-strong)]"
                                  >
                                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                      <polyline points="7 10 12 15 17 10" />
                                      <line x1="12" y1="15" x2="12" y2="3" />
                                    </svg>
                                    Pull {importCount} images
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                          {/* Stage 3, review pulled set, click to mark bad. */}
                          <div
                            className="grid transition-[grid-template-rows] duration-300 ease-out"
                            style={{ gridTemplateRows: importStage === "review" ? "1fr" : "0fr" }}
                          >
                            <div className="min-h-0 overflow-hidden">
                              <div className="grid gap-3 pt-3 mt-1 border-t border-foreground/[0.06]">
                                <div className="flex items-center justify-between gap-3">
                                  <span className="text-[10px] uppercase tracking-wider text-foreground/45">
                                    {importPulling
                                      ? `Pulling ${importCount}…`
                                      : `Click any image you don’t want, the rest get added.`}
                                  </span>
                                  {!importPulling && importPulled.length > 0 && (
                                    <span className="text-xs font-mono tabular-nums text-foreground/70">
                                      <span className="text-emerald-300">{importPulled.length - importBadUrls.size}</span>
                                      <span className="text-foreground/40"> good · </span>
                                      <span className="text-rose-300">{importBadUrls.size}</span>
                                      <span className="text-foreground/40"> excluded</span>
                                    </span>
                                  )}
                                </div>
                                <ul className="grid gap-1.5 grid-cols-4 sm:grid-cols-6 md:grid-cols-8 max-h-[60vh] overflow-y-auto pr-1 -mr-1">
                                  {importPulling
                                    ? Array.from({ length: Math.min(importCount, 24) }).map((_, i) => (
                                        <li
                                          key={`pull-skel-${i}`}
                                          className="aspect-square rounded-md border border-foreground/[0.06] bg-foreground/[0.03] animate-pulse"
                                          style={{ animationDelay: `${(i % 8) * 50}ms` }}
                                        />
                                      ))
                                    : importPulled.map((img, i) => {
                                        const key = img.url || img.thumbnail || `pull-${i}`;
                                        const broken = importBroken.has(key);
                                        const bad = img.url ? importBadUrls.has(img.url) : false;
                                        return (
                                          <li key={`pull-${i}-${img.id ?? img.url}`}>
                                            <button
                                              type="button"
                                              onClick={() => img.url && togglePulledBad(img.url)}
                                              title={
                                                bad
                                                  ? "Marked excluded, click to keep"
                                                  : "Click to exclude"
                                              }
                                              className={[
                                                "group relative w-full aspect-square overflow-hidden rounded-md border transition-all duration-150",
                                                bad
                                                  ? "border-rose-400/60 ring-2 ring-rose-400/30"
                                                  : "border-foreground/[0.06] hover:border-foreground/25",
                                              ].join(" ")}
                                            >
                                              {!broken && (img.thumbnail || img.url) ? (
                                                // eslint-disable-next-line @next/next/no-img-element
                                                <img
                                                  src={img.thumbnail || img.url}
                                                  alt={img.title || ""}
                                                  className={[
                                                    "absolute inset-0 h-full w-full object-cover transition-all duration-200",
                                                    bad ? "opacity-30 grayscale" : "group-hover:scale-[1.04]",
                                                  ].join(" ")}
                                                  loading="lazy"
                                                  onError={() => {
                                                    setImportBroken((prev) => {
                                                      const next = new Set(prev);
                                                      next.add(key);
                                                      return next;
                                                    });
                                                  }}
                                                />
                                              ) : null}
                                              {broken && (
                                                <div className="absolute inset-0 grid place-items-center bg-foreground/[0.02] text-foreground/40">
                                                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                                    <rect x="3" y="3" width="18" height="18" rx="2" />
                                                    <circle cx="9" cy="9" r="1.5" />
                                                    <path d="M21 15l-5-5L5 21" />
                                                  </svg>
                                                </div>
                                              )}
                                              {bad && (
                                                <div className="absolute inset-0 grid place-items-center bg-rose-500/15">
                                                  <span className="grid h-7 w-7 place-items-center rounded-full bg-rose-500/85 text-[var(--foreground)] shadow-[0_2px_6px_rgba(244,63,94,0.45)]">
                                                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                                      <path d="M18 6 6 18" />
                                                      <path d="m6 6 12 12" />
                                                    </svg>
                                                  </span>
                                                </div>
                                              )}
                                            </button>
                                          </li>
                                        );
                                      })}
                                </ul>
                                {!importPulling && importPulled.length > 0 && (
                                  <div className="flex items-center justify-end gap-2 pt-1">
                                    <button
                                      type="button"
                                      onClick={() => setImportStage("settings")}
                                      disabled={importAdding}
                                      className="rounded-full border border-foreground/10 bg-foreground/[0.04] px-3.5 py-1.5 text-xs font-medium text-foreground/75 transition-all duration-150 hover:bg-foreground/[0.08] hover:border-foreground/20 hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
                                    >
                                      Back
                                    </button>
                                    <button
                                      type="button"
                                      onClick={addPulledToDataset}
                                      disabled={importAdding || importPulled.length - importBadUrls.size === 0}
                                      className="inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-medium text-black bg-white transition-all duration-150 hover:bg-foreground/90 shadow-[var(--shadow-strong)] disabled:opacity-40 disabled:cursor-not-allowed"
                                    >
                                      {importAdding ? (
                                        <>
                                          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="animate-spin">
                                            <path d="M21 12a9 9 0 1 1-6.2-8.55" />
                                          </svg>
                                          Adding…
                                        </>
                                      ) : (
                                        <>
                                          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                            <path d="M5 13l4 4L19 7" />
                                          </svg>
                                          Add {importPulled.length - importBadUrls.size} to dataset
                                        </>
                                      )}
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            {/* dropzone */}
            <div
              onDragEnter={onDragEnter}
              onDragLeave={onDragLeave}
              // Safari needs dragover to set dropEffect, otherwise
              // the drop is silently rejected.
              onDragOver={(e) => {
                e.preventDefault();
                if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
              }}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={[
                "relative cursor-pointer rounded-3xl border border-dashed transition-colors px-8 py-10 text-center",
                dragOver
                  ? "border-[var(--foreground)] bg-foreground/[0.04]"
                  : "border-[var(--border)] hover:border-[var(--border)] hover:bg-foreground/[0.02]",
              ].join(" ")}
            >
            <input
              ref={inputRef}
              type="file"
              accept="image/*,video/*"
              multiple
              className="hidden"
              onChange={(e) => {
                handleMediaFiles(e.target.files);
                if (inputRef.current) inputRef.current.value = "";
              }}
            />
            <div className="text-base">
              {uploading ? "Uploading…" : "Drop media to add to this project"}
            </div>
            <div className="mt-2 grid gap-0.5 text-xs text-[var(--muted)]">
              <div>
                <span className="text-foreground/55">Images</span>
                <span className="text-foreground/30"> · </span>
                jpg · jpeg · png · webp · gif · bmp · tiff · heic · avif
              </div>
              <div>
                <span className="text-foreground/55">Videos</span>
                <span className="text-foreground/30"> · </span>
                mp4 · mov · webm · m4v · avi · mkv
              </div>
            </div>
            <p className="mt-3 text-[11px] text-foreground/40 leading-relaxed">
              By uploading media you agree to our{" "}
              <a
                href="/acceptable-use"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground/55 hover:text-foreground underline underline-offset-2"
              >
                Acceptable Use Policy
              </a>
              {" "}and{" "}
              <a
                href="/privacy"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground/55 hover:text-foreground underline underline-offset-2"
              >
                Privacy Policy
              </a>
              .
            </p>
            </div>
          </div>

          {/* Auto-label panel: title, primary action, labels, settings, AI Review
              review. One self-contained card so the relationship between
              all the auto-labelling controls is unambiguous. The negative
              top margin trims the outer section's gap-10 down to a tighter
              ~12px so the auto-label panel reads as paired with the
              import-card above it, while the dropzone keeps its bigger
              breathing room above. */}
          <div
            className="rounded-3xl border border-foreground/10 px-7 py-6 grid gap-8 -mt-7"
            style={{
              background:
                "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 45%, rgba(255,255,255,0) 100%), #141416",
              boxShadow: "0 1px 0 rgb(var(--foreground-rgb) / 0.05) inset, 0 30px 60px -30px rgb(var(--shadow-rgb) / 0.6)",
            }}
          >
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div>
                <h2 className="text-lg font-semibold tracking-tight">Auto-label</h2>
                <p className="text-xs text-foreground/45 mt-1">
                  Define labels, choose how strict the detector should be, then run.
                </p>
              </div>
              <button
                onClick={autoLabel}
                disabled={
                  !compiledPrompt ||
                  pendingCount === 0 ||
                  phase === "running" ||
                  phase === "uploading"
                }
                style={{ boxShadow: "0 0 14px 1px rgba(249, 115, 22, 0.45)" }}
                className="rounded-full bg-orange-500 text-black px-5 py-2 text-sm font-medium hover:bg-orange-400 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none transition-colors"
              >
                {phase === "running" ? "Auto-labelling…" : (
                  <>
                    Start{pendingCount > 0 && <span className="font-mono opacity-70 ml-1">({pendingCount})</span>}
                  </>
                )}
              </button>
            </div>

            {errorMsg && <div className="text-sm text-red-400">{errorMsg}</div>}
            {dupeMsg && (
              <div
                className="text-sm rounded-xl border border-amber-300/30 bg-amber-300/[0.06] text-amber-200 px-3 py-2"
                style={{ boxShadow: "0 0 18px rgba(251, 146, 60, 0.08)" }}
              >
                {dupeMsg}
              </div>
            )}

            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="text-xs uppercase text-[var(--foreground)]/90">Labels</div>
                <LabelsHelpPopover />
              </div>
              <div className="flex flex-wrap gap-2 rounded-lg border border-[var(--border)] p-2 min-h-[3.25rem] focus-within:border-zinc-500">
                {tags.map((tag) => (
                  <TagChip
                    key={tag}
                    tag={tag}
                    isRunning={liteJob?.tag === tag}
                    runDisabled={!!liteJob || !!activeJobId || (results?.length ?? 0) === 0}
                    activeJobId={activeJobId}
                    liteJobTag={liteJob?.tag}
                    hasResults={(results?.length ?? 0) > 0}
                    onRun={() => runLiteForTag(tag)}
                    onRemove={() => removeTag(tag)}
                    onRename={async (next) => {
                      await renameLabel(tag, next);
                    }}
                  />
                ))}
                <input
                  value={draftTag}
                  onChange={(e) => setDraftTag(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTag(draftTag);
                    }
                    if (e.key === "Backspace" && draftTag === "" && tags.length > 0) {
                      setTags(tags.slice(0, -1));
                    }
                  }}
                  placeholder={tags.length === 0 ? "type a label, then Enter" : ""}
                  className="flex-1 min-w-[10ch] bg-transparent outline-none text-sm py-1 px-1"
                />
              </div>
              {labelWarning && (
                <p className="mt-2 text-xs text-rose-300/90 flex items-center gap-1.5" role="alert">
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 8v4" />
                    <path d="M12 16h.01" />
                  </svg>
                  {labelWarning}
                </p>
              )}
            </div>

            <div
              className="grid gap-6 md:gap-8 md:grid-cols-2 md:divide-x md:divide-foreground/10"
            >
            <div className="md:pr-8 flex flex-col h-full">
              <div className="flex items-center gap-2 mb-3">
                <div className="text-xs uppercase text-[var(--foreground)]/90">Settings</div>
                <SettingsHelpPopover />
              </div>
              {(() => {
                // Mode picker, black-box the threshold values unless
                // the user opens the gear popup. "Custom" persists in
                // the picker once the user has dialled in their own
                // settings, so flipping to Normal / Stricter and back
                // is non-destructive.
                const isNormal =
                  boxThr === LABEL_MODES.normal.box &&
                  textThr === LABEL_MODES.normal.text &&
                  nmsIou === LABEL_MODES.normal.nms;
                const isStricter =
                  boxThr === LABEL_MODES.stricter.box &&
                  textThr === LABEL_MODES.stricter.text &&
                  nmsIou === LABEL_MODES.stricter.nms;
                const customMatches =
                  !!customThresholds &&
                  customThresholds.box === boxThr &&
                  customThresholds.text === textThr &&
                  customThresholds.nms === nmsIou;
                // Active = which pill highlights. Custom wins when
                // current values match the saved Custom AND don't
                // happen to coincide with a preset.
                const isCustomActive = !isNormal && !isStricter;
                const applyMode = (m: "normal" | "stricter") => {
                  const p = LABEL_MODES[m];
                  setBoxThr(p.box);
                  setTextThr(p.text);
                  setNmsIou(p.nms);
                };
                const applyCustom = () => {
                  if (!customThresholds) {
                    setAdvancedOpen(true);
                    return;
                  }
                  setBoxThr(customThresholds.box);
                  setTextThr(customThresholds.text);
                  setNmsIou(customThresholds.nms);
                };
                const opts: { id: "normal" | "stricter" | "custom"; label: string; active: boolean }[] = [
                  { id: "normal", label: "Normal", active: isNormal },
                  { id: "stricter", label: "Stricter", active: isStricter },
                ];
                if (customThresholds || isCustomActive) {
                  opts.push({ id: "custom", label: "Custom", active: isCustomActive && customMatches });
                }
                return (
                  <div className="flex items-center gap-2 pt-1">
                    <div className="inline-flex rounded-full border border-[var(--border)] p-0.5">
                      {opts.map((o) => (
                        <button
                          key={o.id}
                          onClick={() => {
                            if (o.id === "custom") applyCustom();
                            else applyMode(o.id);
                          }}
                          className={[
                            "rounded-full px-3.5 py-1 text-xs uppercase tracking-wide transition-colors",
                            o.active
                              ? "bg-foreground text-background"
                              : "text-[var(--muted)] hover:text-foreground",
                          ].join(" ")}
                          title={
                            o.id === "normal"
                              ? "Normal, picks up more borderline detections"
                              : o.id === "stricter"
                              ? "Stricter, only confident matches"
                              : "Restore your saved custom thresholds"
                          }
                        >
                          {o.label}
                        </button>
                      ))}
                    </div>
                    <button
                      onClick={() => setAdvancedOpen(true)}
                      title="Customise thresholds"
                      aria-label="Customise thresholds"
                      className="grid h-8 w-8 place-items-center rounded-full border border-[var(--border)] text-[var(--muted)] hover:text-foreground hover:border-zinc-500 transition-colors"
                    >
                      <span className="text-base leading-none" aria-hidden="true">⚙️</span>
                    </button>
                  </div>
                );
              })()}

              {/* Resolution: classic full-frame pass (the detector's
                  preprocessor downscales big frames) vs native-resolution
                  tiling, which slices large images into overlapping tiles
                  so small objects keep their pixels. One detector pass per
                  tile — slower and heavier, so it's an explicit opt-in. */}
              <div className="pt-4">
                <div className="text-xs uppercase text-[var(--foreground)]/90 mb-2">Resolution</div>
                <SegmentedControl
                  value={tileNative ? "tile" : "downscale"}
                  onChange={(v) => setTileNative(v === "tile")}
                  options={[
                    {
                      value: "downscale",
                      label: "Downscale",
                      title: "One detector pass per image (fast). Large images are downscaled, so very small objects can be missed.",
                    },
                    {
                      value: "tile",
                      label: "Tile · native",
                      title: "Slice large images into native-resolution tiles and detect on each. Best for small objects in big frames (e.g. aerial imagery) — slower and uses more compute.",
                    },
                  ]}
                />
                {tileNative && (
                  <>
                    <div className="mt-2.5 flex items-center gap-2">
                      <label className="text-[11px] text-foreground/50" htmlFor="tile-size-select">
                        Tile size
                      </label>
                      <select
                        id="tile-size-select"
                        value={tileSize}
                        onChange={(e) => setTileSize(Number(e.target.value))}
                        className="rounded-lg border border-[var(--border)] bg-transparent px-2.5 py-1 text-xs text-[var(--foreground)] outline-none focus:border-zinc-500"
                      >
                        <option value={640}>640 px — max small-object detail</option>
                        <option value={1024}>1024 px — balanced</option>
                        <option value={1280}>1280 px — faster</option>
                      </select>
                    </div>
                    <p className="mt-1.5 text-[11px] text-foreground/40">
                      A 4K image ≈ {tileSize === 640 ? 33 : tileSize === 1280 ? 9 : 16} detector
                      passes instead of 1. Small images run as a single pass automatically.
                    </p>
                  </>
                )}
              </div>

              {/* Read-only threshold readout. Sits at the bottom of
                  the Settings column (mt-auto) so its baseline lines
                  up with the bottom of the right column's Synonyms
                  section. Style mirrors the rest of the page ,
                  small grey label, mono numeric value. */}
              <div className="mt-auto pt-5 grid grid-cols-3 gap-3">
                <ThresholdReadout label="Box" value={boxThr} />
                <ThresholdReadout label="Text" value={textThr} />
                <ThresholdReadout label="NMS" value={nmsIou} />
              </div>
            </div>

            <div className="md:pl-8 grid gap-4">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="text-xs uppercase text-[var(--foreground)]/90">AI Review</div>
                  <VlmReviewHelpPopover />
                </div>
                <SegmentedControl
                  value={isProTier ? vlmAction : "off"}
                  onChange={setVlmAction}
                  disabled={!isProTier}
                  disabledTitle="AI Review is a Pro feature"
                  options={[
                    { value: "off", label: "Off", title: "Skip the AI review entirely. Faster runs." },
                    { value: "manual", label: "Manual", title: "Keep flagged boxes; you decide whether to delete or verify." },
                    { value: "auto_reject", label: "Auto-reject", title: "Drop any detection flagged as a mismatch before saving." },
                  ]}
                />
                {!isProTier && (
                  <p className="mt-2 text-[11px] text-foreground/40">
                    <button
                      type="button"
                      onClick={() => navigateAppTo("pricing")}
                      className="text-foreground/60 hover:text-foreground underline underline-offset-2"
                    >
                      Upgrade to Pro
                    </button>{" "}
                    to use AI Review.
                  </p>
                )}
              </div>
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="text-xs uppercase text-[var(--foreground)]/90">Synonyms</div>
                  <SynonymsHelpPopover />
                </div>
                <SegmentedControl
                  value={synonymsEnabled ? "on" : "off"}
                  onChange={(v) => setSynonymsEnabled(v === "on")}
                  options={[
                    { value: "on", label: "On", title: "Expand each label with related terms before sending to the detector." },
                    { value: "off", label: "Off", title: "Send only the labels you typed, no expansion." },
                  ]}
                />
              </div>
            </div>
            </div>
          </div>

          {/* Review pairs with the auto-label panel above (it's the
              "now do something with the labelled run" step). The
              negative top margin pulls it tight under the panel so
              they read as one cluster, while the bottom margin opens
              a clear gap before the filter / images block below. */}
          {phase !== "running" && reviewCounts.all > 0 && (
            <div className="flex items-center gap-3 flex-wrap -mt-4 mb-3">
              <span className="text-sm text-foreground/55 shrink-0">Review</span>
              <ReviewScopePicker counts={reviewCounts} onPick={startReview} />
            </div>
          )}

          {/* progress */}
          {phase === "running" && (
            <AutoLabelProgress
              jobId={activeJobId}
              startedAt={jobStartedAt}
              index={progressIndex}
              total={progressTotal}
              image={progressImage}
              onCancel={cancelActiveJob}
              cancelling={cancellingJob}
              vlmEnabled={effectiveVlmAction !== "off"}
            />
          )}
          {liteJob && phase !== "running" && (
            <AutoLabelProgress
              jobId={liteJob.id}
              startedAt={liteStartedAt}
              index={liteProgress.index}
              total={liteProgress.total}
              image={liteProgress.image}
              onCancel={cancelLiteJob}
              cancelling={cancellingLite}
              vlmEnabled={effectiveVlmAction !== "off"}
            />
          )}
          </>)}

          {results.length > 0 && (
            <div className="grid gap-6">
              {/* Only show filters that actually match images. `all` is the
                  default; the others appear once they're useful. Hide the
                  whole bar if everything is in the same bucket. */}
              {(() => {
                const visible = (["all", "unlabeled", "unrated", "good", "bad", "vlm"] as const)
                  .filter((f) => f === "all" || filterCounts[f] > 0);
                if (visible.length <= 1) return null;
                return (
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-sm text-foreground/55 shrink-0">Filter</span>
                    <div className="flex flex-wrap items-center gap-2">
                      {visible.map((f) => {
                        const active = filter === f;
                        const label = f === "vlm" ? "AI-rejected" : f;
                        return (
                          <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={[
                              "rounded-full px-3 py-1 text-xs uppercase tracking-wide border transition-colors",
                              active
                                ? "bg-foreground text-background border-[var(--foreground)] hover:bg-zinc-200"
                                : "border-foreground/15 text-[var(--muted)] hover:border-foreground/35 hover:text-foreground",
                            ].join(" ")}
                          >
                            {label}
                            <span className="font-mono opacity-70 ml-1">({filterCounts[f]})</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {filteredResults.length === 0 ? (
                <div className="text-center py-16 text-[var(--muted)] text-sm border border-dashed border-[var(--border)] rounded-xl">
                  No images match this filter
                </div>
              ) : (
                <>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {visibleResults.map((r) => (
                      <ImageThumb
                        key={r.image}
                        result={r}
                        boxes={editedBoxes[r.image] ?? []}
                        verdict={verdicts[r.image]}
                        projectName={name}
                        apiBase={API}
                        inputShape={inputSize}
                        isCover={cover === r.image}
                        hideSmall={hideSmall}
                        cacheKey={manifestUpdatedAt}
                        onClick={() => setOpenImage(r.image)}
                      />
                    ))}
                  </div>
                  {displayLimit < filteredResults.length && (
                    <>
                      <div ref={gridSentinelRef} aria-hidden="true" className="h-1" />
                      <div className="text-center py-4 text-xs text-foreground/40">
                        Loading more… ({displayLimit} of {filteredResults.length})
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      )}

      {tab === "train" && (
        <BetaLocked>
          <TrainView projectName={name} n_labeled={labeledCount} />
        </BetaLocked>
      )}

      {tab === "deploy" && (
        <BetaLocked>
          <DeployView projectName={name} hasModel={hasModel} />
        </BetaLocked>
      )}

      {tab === "optimise" && (
        <BetaLocked>
          <OptimiseView projectName={name} hasModel={hasModel} />
        </BetaLocked>
      )}

      <Footer />

      {/* Label Cascade modal disabled, see _EMBEDDINGS_ENABLED on
          the backend. Component import + state are kept so flipping
          the flag re-enables the surface without any other changes. */}
      {cascadeGroups && cascadeGroups.length > 0 && false && (
        <LabelCascadeReviewModal
          apiBase={API}
          projectId={name}
          groups={cascadeGroups!}
          projectTags={tags}
          onClose={() => setCascadeGroups(null)}
          onApplied={applyCascadeRelabel}
        />
      )}

      {reviewing && reviewList && (
        <ReviewMode
          results={reviewList}
          boxesByImage={editedBoxes}
          jobId={name}
          apiBase={API}
          urlMode="projects"
          verdicts={verdicts}
          onVerdict={(image, verdict) => setVerdicts((prev) => ({ ...prev, [image]: verdict }))}
          onClose={closeReview}
          scope={reviewScope}
          projectTags={tags}
          onBoxesChange={(image, next) =>
            setEditedBoxes((prev) => ({ ...prev, [image]: next }))
          }
          onSegmentBox={async (image, b) => {
            const r = await fetch(`${API}/api/projects/${name}/segment_box?user=${encodeURIComponent(username)}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image, box: [b.x0, b.y0, b.x1, b.y1] }),
            });
            if (!r.ok) return null;
            const d = await r.json();
            return d.mask ?? null;
          }}
          onClassifyBox={
            tags.length > 0
              ? async (image, b) => {
                  const r = await fetch(`${API}/api/projects/${name}/classify_box?user=${encodeURIComponent(username)}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image, box: [b.x0, b.y0, b.x1, b.y1] }),
                  });
                  if (!r.ok) return null;
                  const d = await r.json();
                  return { label: d.label ?? null, score: d.score ?? null };
                }
              : undefined
          }
          onPointDetect={async (image, point) => {
            const r = await fetch(`${API}/api/projects/${name}/detect_point?user=${encodeURIComponent(username)}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image, point: [point.x, point.y] }),
            });
            if (!r.ok) return null;
            return await r.json();
          }}
        />
      )}

      {exportOpen && (
        <ExportModal
          projectId={name}
          projectName={displayName}
          inputShape={inputSize}
          onClose={() => setExportOpen(false)}
        />
      )}

      {settingsOpen && (
        <ProjectSettings
          name={name}
          displayName={displayName}
          cover={cover}
          results={results.map((r) => ({ image: r.image, pending: r.pending }))}
          username={username}
          initialPrivate={isPrivate}
          onRenamed={(newName) => {
            setSettingsOpen(false);
            setDisplayName(newName);
            onRename(newName);
          }}
          onCoverChange={(c) => setCover(c)}
          onPrivateChange={(next) => setIsPrivate(next)}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      {openResult && (
        <EditModal
          result={openResult}
          boxes={editedBoxes[openResult.image] ?? []}
          inputShape={inputSize}
          hideSmall={hideSmall}
          onChange={(next) => {
            if (readOnly) return;
            setEditedBoxes((prev) => ({ ...prev, [openResult.image]: next }));
          }}
          onCascadeRelabel={(targets, newLabel) => {
            if (readOnly) return;
            // Apply the new label + Cascade validation to every
            // (image, box_id) the user accepted. Mirrors what the
            // backend writes in the relabel endpoint so the
            // BoxEditor chip swaps to the orange Label Cascade
            // pill immediately, instead of waiting for the 4s poll
            // to bring the validation field in.
            const cascadeValidation = {
              match: true,
              confidence: 1,
              reason: "Label Cascade",
              source: "cascade" as const,
              kind: "cascade" as const,
            };
            setEditedBoxes((prev) => {
              const next = { ...prev };
              for (const t of targets) {
                const list = next[t.image];
                if (!list) continue;
                let touched = false;
                const updated = list.map((b) => {
                  if (b.id !== t.box_id) return b;
                  touched = true;
                  return { ...b, label: newLabel, validation: cascadeValidation };
                });
                if (touched) next[t.image] = updated;
              }
              return next;
            });
          }}
          verdict={verdicts[openResult.image]}
          onVerdict={(verdict) => {
            if (readOnly) return;
            const image = openResult.image;
            setVerdicts((prev) => {
              const next = { ...prev };
              if (verdict === null) delete next[image];
              else next[image] = verdict;
              return next;
            });
          }}
          onDelete={() => setConfirmDelete(openResult.image)}
          onClearLabels={() => clearImageLabels(openResult.image)}
          projectName={name}
          apiBase={API}
          projectTags={tags}
          username={username}
          // Lock editing while a job is running, boxes and segmentations
          // are being updated server-side, the user shouldn't be able to
          // step on them mid-flight.
          readOnly={readOnly || phase === "running"}
          // Show the "auto-labelling in progress" banner inside the
          // editor so the user understands why they can't edit.
          labelling={!!(activeJobId || liteJob)}
          onClose={() => setOpenImage(null)}
          onPrev={() => {
            // Navigate within the currently filtered set; arrow keys hop
            // through whatever the user is reviewing right now.
            const i = filteredResults.findIndex((r) => r.image === openResult.image);
            if (i > 0) setOpenImage(filteredResults[i - 1].image);
          }}
          onNext={() => {
            const i = filteredResults.findIndex((r) => r.image === openResult.image);
            if (i >= 0 && i < filteredResults.length - 1) setOpenImage(filteredResults[i + 1].image);
          }}
        />
      )}

      {videoQueue[0] && (
        <VideoFrameModal
          file={videoQueue[0]}
          extracting={videoExtracting}
          onCancel={onVideoModalCancel}
          onConfirm={onVideoModalConfirm}
        />
      )}

      {advancedOpen && (() => {
        const isNormal =
          boxThr === LABEL_MODES.normal.box &&
          textThr === LABEL_MODES.normal.text &&
          nmsIou === LABEL_MODES.normal.nms;
        const isStricter =
          boxThr === LABEL_MODES.stricter.box &&
          textThr === LABEL_MODES.stricter.text &&
          nmsIou === LABEL_MODES.stricter.nms;
        return (
          <AdvancedSettingsModal
            customThresholds={customThresholds}
            setCustomThresholds={setCustomThresholds}
            activeBox={boxThr}
            activeText={textThr}
            activeNms={nmsIou}
            isCustomActive={!isNormal && !isStricter}
            setBoxThr={setBoxThr}
            setTextThr={setTextThr}
            setNmsIou={setNmsIou}
            onClose={() => setAdvancedOpen(false)}
          />
        );
      })()}

      {confirmDelete && (
        <DeleteImageModal
          imageName={confirmDelete}
          busy={deletingImage}
          onCancel={() => {
            if (!deletingImage) setConfirmDelete(null);
          }}
          onConfirm={async () => {
            setDeletingImage(true);
            try {
              await deleteImage(confirmDelete);
              setOpenImage((cur) => (cur === confirmDelete ? null : cur));
              setConfirmDelete(null);
            } finally {
              setDeletingImage(false);
            }
          }}
        />
      )}
    </main>
  );
}

function AdvancedSettingsModal({
  customThresholds,
  setCustomThresholds,
  activeBox,
  activeText,
  activeNms,
  isCustomActive,
  setBoxThr,
  setTextThr,
  setNmsIou,
  onClose,
}: {
  customThresholds: { box: number; text: number; nms: number } | null;
  setCustomThresholds: (v: { box: number; text: number; nms: number } | null) => void;
  activeBox: number;
  activeText: number;
  activeNms: number;
  isCustomActive: boolean;
  setBoxThr: (v: number) => void;
  setTextThr: (v: number) => void;
  setNmsIou: (v: number) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  // Sliders display the saved Custom values when present, falling
  // back to whatever's currently active so the user has a sensible
  // starting point. The fallback is read-only-ish, Custom isn't
  // committed (no Custom pill) until the user actually moves a slider.
  const displayBox = customThresholds?.box ?? activeBox;
  const displayText = customThresholds?.text ?? activeText;
  const displayNms = customThresholds?.nms ?? activeNms;
  const updateField = (field: "box" | "text" | "nms", value: number) => {
    const updated = {
      box: customThresholds?.box ?? activeBox,
      text: customThresholds?.text ?? activeText,
      nms: customThresholds?.nms ?? activeNms,
      [field]: value,
    };
    setCustomThresholds(updated);
    // If the user is already on Custom mode, mirror the change to
    // the live values so detections on the next run pick up the
    // edit immediately. Otherwise we just tee up the new Custom for
    // when they click the Custom pill.
    if (isCustomActive) {
      setBoxThr(updated.box);
      setTextThr(updated.text);
      setNmsIou(updated.nms);
    }
  };
  return (
    <div
      className="fixed inset-0 z-50 backdrop-blur-md bg-black/85 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[var(--background)] rounded-2xl border border-[var(--border)] max-w-lg w-full overflow-hidden shadow-2xl">
        <header className="px-6 pt-6 pb-3 flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-foreground/45">Advanced</div>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight">Custom thresholds</h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-full text-foreground/60 hover:text-foreground hover:bg-foreground/10 transition-colors"
          >
            <span className="text-lg leading-none" aria-hidden="true">×</span>
          </button>
        </header>
        <div className="px-6 pb-2">
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            Drag any slider to override the auto labeller thresholds.
          </p>
        </div>
        <div className="px-6 py-5 grid gap-7">
          <ThresholdControl label="Detection confidence" value={displayBox} onChange={(v) => updateField("box", v)} />
          <ThresholdControl label="Label match" value={displayText} onChange={(v) => updateField("text", v)} />
          <ThresholdControl label="Overlap tolerance" value={displayNms} onChange={(v) => updateField("nms", v)} />
        </div>
        <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-100 transition-colors"
          >
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}

function TagChip({
  tag,
  isRunning,
  runDisabled,
  activeJobId,
  liteJobTag,
  hasResults,
  onRun,
  onRemove,
  onRename,
}: {
  tag: string;
  isRunning: boolean;
  runDisabled: boolean;
  activeJobId: string | null;
  liteJobTag?: string;
  hasResults: boolean;
  onRun: () => void;
  onRemove: () => void;
  onRename: (next: string) => Promise<void> | void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(tag);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  // Keep the draft synced if the upstream tag value rotates while we
  // aren't editing (e.g. another rename completes).
  useEffect(() => {
    if (!editing) setDraft(tag);
  }, [tag, editing]);
  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const cancel = () => {
    setEditing(false);
    setDraft(tag);
  };
  const save = async () => {
    const next = draft.trim();
    if (!next || next.toLowerCase() === tag.toLowerCase()) {
      cancel();
      return;
    }
    setSaving(true);
    try {
      await onRename(next);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  if (editing) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-foreground/15 pl-3 pr-1.5 py-1 text-sm ring-1 ring-foreground/30">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              save();
            } else if (e.key === "Escape") {
              e.preventDefault();
              cancel();
            }
          }}
          onBlur={save}
          disabled={saving}
          aria-label={`rename ${tag}`}
          className="bg-transparent outline-none text-sm w-[14ch] disabled:opacity-60"
          autoFocus
        />
        <button
          onClick={cancel}
          disabled={saving}
          className="rounded-full hover:bg-foreground/20 h-5 w-5 grid place-items-center text-xs disabled:opacity-40"
          aria-label="cancel rename"
          title="cancel"
        >
          ×
        </button>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-foreground/10 pl-3 pr-1.5 py-1 text-sm">
      <button
        type="button"
        onClick={() => setEditing(true)}
        title="Click to rename across the whole project"
        className="cursor-text hover:underline underline-offset-2 decoration-white/30"
      >
        {tag}
      </button>
      <button
        onClick={onRun}
        disabled={runDisabled || isRunning}
        className="rounded-full hover:bg-foreground/20 h-5 w-5 grid place-items-center text-xs disabled:opacity-40 disabled:cursor-not-allowed"
        aria-label={`re-run detection for ${tag}`}
        title={
          !hasResults
            ? "Add images first"
            : isRunning
            ? `Detecting ${tag}…`
            : !!activeJobId
            ? "Wait for current job to finish"
            : !!liteJobTag
            ? `Detecting ${liteJobTag}…`
            : `Re-run detection for "${tag}" across all images (appends, won't overwrite)`
        }
      >
        {isRunning ? (
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-foreground/60 border-t-transparent animate-spin" />
        ) : "↻"}
      </button>
      <button
        onClick={onRemove}
        className="rounded-full hover:bg-foreground/20 h-5 w-5 grid place-items-center text-xs"
        aria-label={`remove ${tag}`}
      >
        ×
      </button>
    </span>
  );
}

function VideoFrameModal({
  file,
  extracting,
  onCancel,
  onConfirm,
}: {
  file: File;
  extracting: { done: number; total: number } | null;
  onCancel: () => void;
  onConfirm: (params: { start: number; end: number; fps: number }) => void;
}) {
  // Object URL lifecycle is managed inside the effect (not lazy-init
  // useState) so React 18's StrictMode double-mount doesn't revoke
  // the URL the live <video> element is still pointing at, that bug
  // surfaced as "couldn't decode this video" the instant the modal
  // opened, even on a clean 20 MB MP4.
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [fps, setFps] = useState(1);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Strip of evenly-spaced thumbnails rendered as the trim bar's
  // background, gives the user a visual cue of what's at each time
  // so they know what they're trimming, like the iOS Photos editor.
  const THUMB_COUNT = 16;
  const [thumbs, setThumbs] = useState<string[]>([]);
  const isExtracting = !!extracting;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isExtracting) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, isExtracting]);

  // Generate the thumbnail strip on a hidden video. Decoupled from the
  // visible <video> so scrubbing/playing the preview doesn't fight the
  // seeks needed for thumbs.
  useEffect(() => {
    if (!loaded || duration <= 0 || !src) return;
    let cancelled = false;
    (async () => {
      const v = document.createElement("video");
      v.src = src;
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      try {
        await new Promise<void>((resolve, reject) => {
          const ok = () => resolve();
          const err = () => reject(new Error("load failed"));
          v.addEventListener("loadeddata", ok, { once: true });
          v.addEventListener("error", err, { once: true });
        });
        const N = THUMB_COUNT;
        const TW = 120;
        const ar = (v.videoHeight || 1) / (v.videoWidth || 1);
        const TH = Math.max(40, Math.round(TW * ar));
        const canvas = document.createElement("canvas");
        canvas.width = TW;
        canvas.height = TH;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const out: string[] = [];
        for (let i = 0; i < N; i++) {
          if (cancelled) return;
          const t = (i / Math.max(1, N - 1)) * duration * 0.999;
          await new Promise<void>((resolve) => {
            let done = false;
            const onSeeked = () => {
              if (done) return;
              done = true;
              v.removeEventListener("seeked", onSeeked);
              resolve();
            };
            v.addEventListener("seeked", onSeeked);
            v.currentTime = t === v.currentTime ? t + 0.001 : t;
            setTimeout(() => {
              if (!done) {
                done = true;
                v.removeEventListener("seeked", onSeeked);
                resolve();
              }
            }, 1500);
          });
          ctx.drawImage(v, 0, 0, TW, TH);
          out.push(canvas.toDataURL("image/jpeg", 0.6));
          if (!cancelled) setThumbs([...out]);
        }
      } catch {
        // Non-fatal, modal still works without the strip.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loaded, duration, src]);

  const onLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const d = e.currentTarget.duration;
    if (Number.isFinite(d) && d > 0) {
      setDuration(d);
      setEnd(d);
      setLoaded(true);
    }
  };

  const seek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return `${m}:${r.toFixed(1).padStart(4, "0")}`;
  };

  const frameCount = duration > 0 ? Math.max(1, Math.floor((end - start) * fps) + 1) : 0;
  const span = end - start;

  return (
    <div
      className="fixed inset-0 z-50 backdrop-blur-md bg-black/85 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget && !isExtracting) onCancel();
      }}
    >
      <div className="bg-[var(--background)] rounded-2xl border border-[var(--border)] max-w-2xl w-full overflow-hidden shadow-2xl">
        <header className="px-6 pt-6 pb-3 flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-foreground/45">Video import</div>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight truncate">
              {file.name}
            </h2>
          </div>
          <button
            onClick={onCancel}
            disabled={isExtracting}
            aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-full text-foreground/60 hover:text-foreground hover:bg-foreground/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span className="text-lg leading-none" aria-hidden="true">×</span>
          </button>
        </header>

        <div className="px-6 pb-2">
          <div className="relative rounded-xl overflow-hidden bg-[var(--background)] border border-foreground/[0.06]">
            {src && (
            /* eslint-disable-next-line jsx-a11y/media-has-caption */
            <video
              ref={videoRef}
              src={src}
              controls
              playsInline
              preload="auto"
              className="w-full max-h-[50vh] block bg-[var(--background)]"
              onLoadedMetadata={onLoadedMetadata}
              onError={(e) => {
                // Only flag a real decode failure. The bare `onError`
                // signature on HTMLVideoElement fires from harmless
                // transient states too (seek-during-buffer, abort on
                // re-render, etc.); we only want to surface the
                // permanent codec-not-supported case, where
                // `mediaElement.error.code` is set.
                const err = (e.currentTarget as HTMLVideoElement).error;
                if (err && (err.code === err.MEDIA_ERR_SRC_NOT_SUPPORTED || err.code === err.MEDIA_ERR_DECODE)) {
                  setLoadError("Couldn't decode this video, try a different format (mp4 / webm / mov).");
                }
              }}
            />
            )}
            {/* "Loading video…" overlay shown until the metadata
                event fires. Important for big files where the video
                element appears black for a couple of seconds before
                the duration is known. */}
            {!loaded && !loadError && (
              <div className="absolute inset-0 grid place-items-center bg-black/70 backdrop-blur-sm">
                <div className="flex items-center gap-3 text-sm text-foreground/85">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="animate-spin">
                    <path d="M21 12a9 9 0 1 1-6.2-8.55" />
                  </svg>
                  Loading video…
                </div>
              </div>
            )}
            {loadError && (
              <div className="absolute inset-0 grid place-items-center bg-black/80 p-6 text-center">
                <p className="text-sm text-rose-200">{loadError}</p>
              </div>
            )}
          </div>
        </div>

        <div className="px-6 py-5 grid gap-5">
          {/* Trim slider, iOS-style two-handle range over a single
              track. Dragging either handle scrubs the video to that
              time so the user previews exactly where the cut sits. */}
          <div>
            <div className="flex items-center justify-between text-xs text-foreground/55 mb-2">
              <span className="flex items-center gap-2">
                Trim
                {loaded && thumbs.length > 0 && thumbs.length < THUMB_COUNT && (
                  <span className="inline-flex items-center gap-1.5 text-foreground/40">
                    <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="animate-spin">
                      <path d="M21 12a9 9 0 1 1-6.2-8.55" />
                    </svg>
                    Generating preview…
                  </span>
                )}
              </span>
              <span className="font-mono tabular-nums">
                {fmt(start)}  →  {fmt(end)}
                <span className="text-foreground/35 ml-2">({fmt(span)})</span>
              </span>
            </div>
            <TrimSlider
              duration={duration}
              start={start}
              end={end}
              thumbs={thumbs}
              disabled={!loaded || isExtracting}
              onChange={(s, e) => {
                setStart(s);
                setEnd(e);
              }}
              onScrub={(t) => seek(t)}
            />
          </div>

          {/* Frame rate */}
          <div>
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-foreground/55">Sample rate</span>
              <span className="font-mono tabular-nums text-sm text-foreground/90">{fps.toFixed(1)} fps</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={10}
              step={0.5}
              value={fps}
              onChange={(e) => setFps(parseFloat(e.target.value))}
              disabled={isExtracting}
              className="w-full accent-white disabled:opacity-50"
            />
            <div className="flex items-center justify-between text-[10px] text-foreground/35 font-mono tabular-nums mt-1">
              <span>0.5</span>
              <span>10</span>
            </div>
          </div>
        </div>

        {isExtracting ? (
          <footer className="px-6 py-4 border-t border-[var(--border)] grid gap-2">
            <div className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-sm text-foreground/85">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-orange-400 opacity-70" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-orange-400" />
                </span>
                Extracting frames
              </span>
              <span className="text-xs text-foreground/55 tabular-nums">
                {extracting!.done} / {extracting!.total} ·{" "}
                {Math.round((extracting!.done / Math.max(1, extracting!.total)) * 100)}%
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-foreground/[0.08]">
              <div
                className="h-full rounded-full transition-all duration-150"
                style={{
                  width: `${Math.min(100, (extracting!.done / Math.max(1, extracting!.total)) * 100)}%`,
                  background: "linear-gradient(90deg, #fb923c, #f97316)",
                  boxShadow: "0 0 14px rgba(249,115,22,0.5)",
                }}
              />
            </div>
            <p className="text-[11px] text-foreground/45">
              Don&rsquo;t close this window, frames are being decoded locally.
            </p>
          </footer>
        ) : (
          <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-between gap-3">
            <span className="text-xs text-foreground/55 font-mono tabular-nums">
              {loaded ? `${frameCount} frame${frameCount === 1 ? "" : "s"}` : "Loading…"}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={onCancel}
                className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onConfirm({ start, end, fps })}
                disabled={!loaded || frameCount === 0 || !!loadError}
                className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Extract &amp; upload
              </button>
            </div>
          </footer>
        )}
      </div>
    </div>
  );
}

function TrimSlider({
  duration,
  start,
  end,
  thumbs,
  disabled,
  onChange,
  onScrub,
}: {
  duration: number;
  start: number;
  end: number;
  thumbs: string[];
  disabled?: boolean;
  onChange: (start: number, end: number) => void;
  onScrub?: (t: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef<"start" | "end" | null>(null);

  const positionAt = (clientX: number) => {
    const track = trackRef.current;
    if (!track || duration <= 0) return 0;
    const rect = track.getBoundingClientRect();
    const t = ((clientX - rect.left) / rect.width) * duration;
    return Math.max(0, Math.min(duration, t));
  };

  const onPointerDown = (which: "start" | "end") => (e: React.PointerEvent) => {
    if (disabled) return;
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    draggingRef.current = which;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    const t = positionAt(e.clientX);
    if (draggingRef.current === "start") {
      const next = Math.min(t, end - 0.05);
      onChange(Math.max(0, next), end);
      onScrub?.(next);
    } else {
      const next = Math.max(t, start + 0.05);
      onChange(start, Math.min(duration, next));
      onScrub?.(next);
    }
  };

  const onPointerUp = () => {
    draggingRef.current = null;
  };

  const startPct = duration > 0 ? (start / duration) * 100 : 0;
  const endPct = duration > 0 ? (end / duration) * 100 : 100;

  // Reserved gutter width on each side so the orange handle bars sit
  // *inside* the strip rather than over the very first / last
  // thumbnail. Matches the iOS Photos trimmer.
  const HANDLE_W = 14;
  // Inset margin around the filmstrip itself so the thumbs visibly
  // stop where the trim region starts. This is purely visual, the
  // handle/percent math still spans 0-100% of the container.

  return (
    <div
      ref={trackRef}
      className={[
        "relative h-20 select-none rounded-xl overflow-hidden border border-foreground/10 bg-[var(--background)]",
        disabled ? "opacity-50" : "",
      ].join(" ")}
      style={{ touchAction: "none" }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {/* Filmstrip background, evenly-spaced thumbnails. While the
          strip is still generating we render placeholder cells so the
          bar doesn't reflow from empty to populated. */}
      <div className="absolute inset-0 flex">
        {(thumbs.length > 0 ? thumbs : Array.from({ length: 16 }).map(() => "")).map((url, i) => (
          <div
            key={i}
            className="flex-1 bg-cover bg-center"
            style={{
              backgroundImage: url ? `url(${url})` : undefined,
              backgroundColor: url ? undefined : "rgb(var(--foreground-rgb) / 0.04)",
              borderLeft: i === 0 ? undefined : "1px solid rgb(var(--shadow-rgb) / 0.25)",
            }}
          />
        ))}
      </div>

      {/* Dim overlay for the parts of the timeline OUTSIDE the
          selection. iOS-style, the unselected stretch is darker so
          the eye locks onto the selected range. */}
      <div
        className="absolute top-0 bottom-0 left-0 bg-black/60 pointer-events-none"
        style={{ width: `${startPct}%` }}
      />
      <div
        className="absolute top-0 bottom-0 right-0 bg-black/60 pointer-events-none"
        style={{ width: `${100 - endPct}%` }}
      />

      {/* Top + bottom orange bars connecting the two end handles ,
          visual frame around the selection. */}
      <div
        className="absolute top-0 h-1 bg-orange-500 pointer-events-none"
        style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}
      />
      <div
        className="absolute bottom-0 h-1 bg-orange-500 pointer-events-none"
        style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}
      />

      {/* Start handle, vertical orange grip bar on the left edge of
          the selection. */}
      <button
        type="button"
        aria-label="Trim start"
        onPointerDown={onPointerDown("start")}
        className="absolute top-0 bottom-0 cursor-ew-resize touch-none flex items-center justify-center"
        style={{
          left: `${startPct}%`,
          width: HANDLE_W,
          transform: "translateX(-50%)",
          background: "#f97316",
          boxShadow: "0 0 0 1px rgb(var(--shadow-rgb) / 0.25), 0 4px 14px rgba(249,115,22,0.45)",
        }}
      >
        <span className="block h-6 w-[2px] rounded-full bg-foreground/80" />
      </button>

      {/* End handle */}
      <button
        type="button"
        aria-label="Trim end"
        onPointerDown={onPointerDown("end")}
        className="absolute top-0 bottom-0 cursor-ew-resize touch-none flex items-center justify-center"
        style={{
          left: `${endPct}%`,
          width: HANDLE_W,
          transform: "translateX(-50%)",
          background: "#f97316",
          boxShadow: "0 0 0 1px rgb(var(--shadow-rgb) / 0.25), 0 4px 14px rgba(249,115,22,0.45)",
        }}
      >
        <span className="block h-6 w-[2px] rounded-full bg-foreground/80" />
      </button>
    </div>
  );
}

function ClearLabelsModal({
  imageName,
  count,
  busy,
  onCancel,
  onConfirm,
}: {
  imageName: string;
  count: number;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (busy) return;
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onCancel, onConfirm]);
  return (
    <div
      className="fixed inset-0 z-[60] backdrop-blur-md bg-black/85 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="bg-[var(--background)] rounded-2xl border border-[var(--border)] max-w-md w-full overflow-hidden shadow-2xl">
        <header className="px-6 pt-6 pb-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-amber-300/90">Reset labels</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Clear labels?</h2>
        </header>
        <div className="px-6 pb-5">
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            Removes all {count} label{count === 1 ? "" : "s"} on{" "}
            <span className="font-mono text-[var(--foreground)] break-all">{imageName}</span>, deletes the rendered preview, and marks the image as unlabelled. Other images aren&rsquo;t affected.
          </p>
        </div>
        <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2">
          <button
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500 transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            autoFocus
            className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-100 transition-colors disabled:opacity-60"
          >
            {busy ? "Clearing…" : "Clear labels"}
          </button>
        </footer>
      </div>
    </div>
  );
}

function DeleteImageModal({
  imageName,
  busy,
  onCancel,
  onConfirm,
}: {
  imageName: string;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (busy) return;
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onCancel, onConfirm]);

  return (
    <div
      className="fixed inset-0 z-50 backdrop-blur-md bg-black/85 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="bg-[var(--background)] rounded-2xl border border-[var(--border)] max-w-md w-full overflow-hidden shadow-2xl">
        <header className="px-6 pt-6 pb-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-red-400">Permanent action</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Delete image?</h2>
        </header>
        <div className="px-6 pb-5">
          <p className="text-sm text-[var(--muted)] leading-relaxed">
            Removes <span className="font-mono text-[var(--foreground)] break-all">{imageName}</span> along with its label, mask, and verdict.{" "}
            <span className="text-red-300">This cannot be undone.</span>
          </p>
        </div>
        <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2">
          <button
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500 transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            autoFocus
            className="rounded-full bg-red-500 text-[var(--foreground)] px-5 py-2 text-sm font-medium hover:bg-red-400 transition-colors disabled:opacity-60"
          >
            {busy ? "Deleting…" : "Delete image"}
          </button>
        </footer>
      </div>
    </div>
  );
}

// Friendly rotating status messages while a job is running. Two pools so the
// AI-Review-enabled path can swap to confidence/review phrases when the model's
// auditing rather than drawing.
const LABEL_PHRASES = [
  "Scanning for suspiciously object-shaped pixels…",
  "Asking the model “what vibes do you detect?”",
  "Proposing candidate regions like a very confident intern…",
  "Generating bounding boxes with questionable confidence…",
  "Looking for things that might be things…",
  "Detecting objects (or at least trying really hard to)",
  "Guessing where stuff might be hiding…",
  "Turning pixels into hypotheses…",
  "Finding blobs that could legally count as objects…",
  "Running zero-shot detection (and hoping for the best)",
  "Putting borders on things…",
  "Drawing lines where the model feels emotionally confident…",
  "Separating object from not-object…",
  "Carving reality into neat little masks…",
  "Turning blobs into shapes…",
  "Refining edges like a perfectionist…",
  "Segmenting with surgical optimism…",
  "Making masks that almost make sense…",
  "Naming things with varying degrees of confidence…",
  "Applying labels and mild overconfidence…",
  "Making educated guesses about reality…",
  "Translating pixels into words and words into pixels…",
  "Arguing internally…",
  "Assigning labels that seem defensible…",
  "Performing semantic guesswork…",
  "Choosing labels and standing by them (for now)",
  "Converting shapes into nouns…",
  "Comparing things to other things we’ve seen…",
  "Looking for familiar-looking objects…",
  "Finding visually similar troublemakers…",
  "Finding objects that pass the “same-ish” test…",
  "Recalling past visual experiences…",
  "Doing computer vision things…",
  "Making sense of chaos…",
  "Turning images into structured optimism…",
  "Working through pixels one decision at a time…",
  "Progress is happening, we promise…",
  "Applying questionable intelligence at scale…",
  "Optimising reality…",
  "Crunching pixels responsibly…",
  "Finding shapes…",
  "Figuring out what’s what…",
  "Comparing things to things…",
  "Fixing similar ones…",
  "Making better guesses…",
  "Inspecting pixels with cautious optimism…",
  "Highlighting regions of potential importance…",
  "Running object hypotheses through the pipeline…",
  "Estimating where boundaries might exist…",
  "Interpreting visual signals at scale…",
  "Locating candidate objects in the scene…",
  "Breaking images into meaningful parts…",
  "Approximating object locations…",
  "Evaluating visual regions for relevance…",
  "Filtering out obvious non-objects…",
  "Tracing edges with algorithmic confidence…",
  "Defining object boundaries (approximately)…",
  "Separating foreground from background (hopefully)…",
  "Building masks from educated guesses…",
  "Isolating regions of interest…",
  "Improving edge alignment slightly…",
  "Resolving where one thing ends and another begins…",
  "Applying mask generation routines…",
  "Reducing ambiguity one pixel at a time…",
  "Assigning semantic meaning to shapes…",
  "Matching regions to known concepts…",
  "Selecting the most plausible label…",
  "Evaluating classification candidates…",
  "Applying category predictions…",
  "Doing something…",
  "Resolving label ambiguity…",
  "Aligning detections with labels…",
  "Estimating object identity…",
  "Processing image batch sequentially…",
  "Evaluating dataset incrementally…",
  "Advancing through workload…",
  "Maintaining steady throughput…",
  "Continuing structured analysis…",
  "Iterating over image set…",
  "Progressing through pipeline stages…",
  "Updating task completion state…",
  "Tracking processing metrics…",
  "Advancing analysis pipeline…",
  "Balancing precision and recall…",
  "Analyzing visual input for object presence…",
  "Running inference on image data…",
  "Locating regions with high object likelihood…",
  "Generating candidate detections…",
  "Applying spatial reasoning to image features…",
  "Estimating object extents…",
  "Deriving region proposals…",
  "Scoring detected regions…",
  "Suppressing low-confidence predictions…",
  "Performing non-maximum suppression…",
  "Refining bounding box coordinates…",
  "Projecting segmentation masks onto image…",
  "Isolating object regions…",
  "Computing object-level predictions…",
  "Mapping detections to class labels…",
  "Evaluating classification probabilities…",
  "Resolving competing label assignments…",
  "Applying decision thresholds…",
  "Merging overlapping regions…",
  "Pruning redundant detections…",
  "Structuring detection outputs…",
  "Processing inputs sequentially…",
  "Iterating through dataset samples…",
  "Updating inference progress…",
  "Monitoring pipeline execution…",
  "Maintaining output consistency…",
  "Improving prediction stability…",
  "Normalizing detection results…",
  "Running object detection across the image…",
  "Identifying candidate regions of interest…",
  "Evaluating potential object locations…",
  "Applying detection model to input data…",
  "Extracting bounding box predictions…",
  "Filtering detections by confidence threshold…",
  "Refining detected regions…",
  "Generating segmentation masks…",
  "Estimating object boundaries…",
  "Separating foreground objects from background…",
  "Applying mask refinement…",
  "Aligning segmentation with detected regions…",
  "Processing object candidates…",
  "Assigning class labels to detections…",
  "Evaluating classification confidence…",
  "Resolving label predictions…",
  "Selecting the most probable class…",
  "Applying classification logic…",
  "Reconciling overlapping predictions…",
  "Filtering duplicate detections…",
  "Consolidating detection outputs…",
  "Processing batch of images…",
  "Advancing through dataset…",
  "Updating progress state…",
  "Tracking processing performance…",
  "Maintaining detection consistency…",
  "Balancing detection precision and recall…",
  "Optimizing inference results…",
];
const VLM_PHRASES = [
  "Not entirely convinced about this one…",
  "This feels like a “maybe”…",
  "Confidence is… negotiable…",
  "Sending this one to the thinking department…",
  "This could go either way…",
  "Flagging for human wisdom…",
  "Mild uncertainty detected…",
  "Escalating to higher intelligence…",
  "The model is hesitating…",
  "This one requires vibes + judgment…",
  "Detecting borderline classifications…",
  "Separating confident vs uncertain outputs…",
  "Highlighting items needing review…",
  "Pausing on unclear results…",
  "Deferring uncertain decisions…",
  "Tracking low-confidence outputs…",
  "Marking outputs requiring verification…",
  "Reviewing detection quality…",
  "Validating detection results…",
  "Screening for uncertain predictions…",
  "Reviewing low-confidence predictions…",
  "Flagging ambiguous detections for review…",
];

function LiveMessage({ vlmEnabled }: { vlmEnabled: boolean }) {
  const [msg, setMsg] = useState(() => LABEL_PHRASES[Math.floor(Math.random() * LABEL_PHRASES.length)]);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    let mounted = true;
    const tick = () => {
      if (!mounted) return;
      setVisible(false);
      // Wait for the fade-out before swapping text, then fade back in. Keeps
      // the change feeling intentional rather than flickery.
      window.setTimeout(() => {
        if (!mounted) return;
        // 30% chance to draw from the review pool when it's enabled, the rest
        // of the time stay on the labelling pool, since labelling is the
        // bulk of the work.
        const pool = vlmEnabled && Math.random() < 0.3 ? VLM_PHRASES : LABEL_PHRASES;
        let next = pool[Math.floor(Math.random() * pool.length)];
        // Avoid same message twice in a row.
        setMsg((cur) => {
          if (next === cur) {
            next = pool[(pool.indexOf(next) + 1) % pool.length];
          }
          return next;
        });
        setVisible(true);
      }, 250);
    };
    // Slightly randomised cadence (~3.2s) so it doesn't feel metronomic.
    const id = window.setInterval(tick, 3200);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, [vlmEnabled]);

  return (
    <span className={["transition-opacity duration-200", visible ? "opacity-100" : "opacity-0"].join(" ")}>
      {msg}
    </span>
  );
}

function AutoLabelProgress({
  jobId,
  startedAt,
  index,
  total,
  image,
  onCancel,
  cancelling,
  vlmEnabled,
}: {
  jobId: string | null;
  startedAt: number | null;
  index: number;
  total: number;
  image: string;
  onCancel: () => void;
  cancelling: boolean;
  vlmEnabled: boolean;
}) {
  // Refresh every second so elapsed/ETA stay live without other state churn.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const elapsedMs = startedAt ? Math.max(0, now - startedAt) : 0;
  const done = Math.max(0, index);
  const remaining = Math.max(0, total - done);
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  const etaMs = done > 0 && remaining > 0 ? (elapsedMs / done) * remaining : null;

  return (
    <div
      className="relative overflow-hidden rounded-3xl border border-foreground/10 px-8 py-7"
      style={{
        // Static layered surface, the same Apple-glass *look* but without
        // backdrop-filter, which combined with animated children makes the
        // browser re-composite the whole region every frame.
        background:
          "radial-gradient(120% 140% at 100% 100%, rgba(249,115,22,0.10) 0%, rgba(249,115,22,0) 55%), linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 45%, rgba(255,255,255,0) 100%), #141416",
        boxShadow:
          "0 1px 0 rgb(var(--foreground-rgb) / 0.05) inset, 0 30px 60px -30px rgba(249,115,22,0.20), 0 1px 30px -8px rgb(var(--shadow-rgb) / 0.6)",
      }}
    >
      <div className="relative grid gap-7">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="grid gap-2 min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium text-orange-300/90">
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-orange-400 opacity-75" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-orange-400" />
              </span>
              Working
            </div>
            <div className="text-2xl md:text-3xl font-light tracking-tight text-[var(--foreground)] truncate leading-snug pb-0.5">
              <LiveMessage vlmEnabled={vlmEnabled} />
            </div>
          </div>

          <div className="flex items-center gap-5">
            <div className="text-right leading-none">
              <div className="tabular-nums text-5xl md:text-6xl font-thin text-[var(--foreground)]">
                {Math.round(pct)}
                <span className="text-2xl text-foreground/30 ml-0.5 align-top">%</span>
              </div>
              <div className="mt-2 text-sm text-foreground/50 tabular-nums">
                {done} of {total} images
              </div>
            </div>
            <button
              onClick={onCancel}
              disabled={cancelling || !jobId}
              className="rounded-full border border-foreground/15 bg-foreground/5 text-foreground/80 hover:text-foreground hover:bg-foreground/10 px-4 py-2 text-xs transition-colors disabled:opacity-40"
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          </div>
        </div>

        {/* Single-tone progress: a thin pill with a soft glow, no banded
            gradient. Premium feel, lower visual noise. */}
        <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-foreground/[0.06]">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              background: "linear-gradient(90deg, #fb923c, #f97316)",
              boxShadow: "0 0 14px rgba(249,115,22,0.5)",
            }}
          />
          <div
            aria-hidden
            className="progress-sweep absolute inset-y-0 left-0 w-1/4"
            style={{
              background: "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgb(var(--foreground-rgb) / 0.18) 50%, rgba(255,255,255,0) 100%)",
            }}
          />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-4">
          <AppleStat label="Done" value={`${done}`} />
          <AppleStat label="Remaining" value={`${remaining}`} />
          <AppleStat label="Elapsed" value={formatDuration(elapsedMs)} />
          <AppleStat label="ETA" value={etaMs !== null ? `~${formatDuration(etaMs)}` : ","} />
        </div>

        {image && (
          <div className="flex items-center gap-3 text-sm text-foreground/40 min-w-0 pt-1 border-t border-foreground/[0.06]">
            <span className="text-xs">Now</span>
            <span className="font-mono text-foreground/60 truncate">{image}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function AppleStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-foreground/40">{label}</div>
      <div className="mt-1 tabular-nums text-xl font-light text-[var(--foreground)]">{value}</div>
    </div>
  );
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return ",";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm.toString().padStart(2, "0")}m`;
}

function ReviewScopePicker({
  counts,
  onPick,
}: {
  counts: { unrated: number; vlm: number; good: number; bad: number; all: number };
  onPick: (scope: ReviewScope) => void;
}) {
  const items: { scope: ReviewScope; label: string; primary?: boolean }[] = [
    { scope: "unrated", label: "Unrated", primary: true },
    { scope: "vlm", label: "AI-rejected" },
    { scope: "good", label: "Good" },
    { scope: "bad", label: "Bad" },
    { scope: "all", label: "All" },
  ];
  // Hide zero-count entries entirely so the row only shows what's actionable.
  const visible = items.filter(({ scope }) => counts[scope] > 0);
  if (visible.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      {visible.map(({ scope, label, primary }) => {
        const n = counts[scope];
        return (
          <button
            key={scope}
            onClick={() => onPick(scope)}
            className={[
              "rounded-full px-3 py-1 text-xs uppercase tracking-wide border transition-colors",
              primary
                ? "bg-foreground text-background border-[var(--foreground)] hover:bg-zinc-200"
                : "border-foreground/15 text-[var(--muted)] hover:border-foreground/35 hover:text-foreground",
            ].join(" ")}
          >
            {label} <span className="font-mono opacity-70 ml-0.5">({n})</span>
          </button>
        );
      })}
    </div>
  );
}

function InfoPopover({
  ariaLabel = "More info",
  children,
}: {
  ariaLabel?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const timerRef = useRef<number | null>(null);
  // Short enough to feel responsive, long enough to filter out a
  // mouse just sweeping across the page.
  const HOVER_DELAY_MS = 200;
  const onEnter = () => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    timerRef.current = window.setTimeout(() => {
      setOpen(true);
      timerRef.current = null;
    }, HOVER_DELAY_MS);
  };
  const onLeave = () => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setOpen(false);
  };
  useEffect(() => () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
  }, []);
  return (
    <div
      className="relative inline-flex"
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <span
        role="img"
        aria-label={ariaLabel}
        className="grid h-4 w-4 place-items-center rounded-full border border-foreground/20 text-[10px] font-semibold text-foreground/55 cursor-help select-none transition-colors hover:text-foreground hover:border-foreground/40"
      >
        i
      </span>
      {open && (
        <>
          <div
            className="fixed inset-0 z-40 backdrop-blur-[2px] bg-black/15 pointer-events-none"
            aria-hidden="true"
          />
          <div
            role="tooltip"
            className="absolute top-full left-0 mt-2 z-50 w-80 rounded-2xl border border-foreground/10 p-4 shadow-2xl"
            style={{
              background:
                "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 60%, rgba(255,255,255,0) 100%), #0c0c0e",
              boxShadow:
                "0 1px 0 rgb(var(--foreground-rgb) / 0.06) inset, 0 24px 60px -10px rgb(var(--shadow-rgb) / 0.7)",
            }}
          >
            {children}
          </div>
        </>
      )}
    </div>
  );
}

function LabelsHelpPopover() {
  return (
    <InfoPopover ariaLabel="Labels help">
      <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
        Tips for great labels
      </h3>
      <ul className="text-xs text-foreground/70 leading-relaxed space-y-2">
        <li>
          Use the name of a thing you can point to in the image.{" "}
          <span className="font-mono text-foreground/90">car</span>,{" "}
          <span className="font-mono text-foreground/90">pothole</span>, and{" "}
          <span className="font-mono text-foreground/90">hard hat</span> all work well.
        </li>
        <li>Write it in lowercase. No punctuation.</li>
        <li>
          Singular or plural is fine. Pick whichever sounds natural and stick with it.
        </li>
        <li>
          One thing per label. If your label combines two ideas (
          <span className="font-mono text-foreground/90">cars and trucks</span>),
          split it into two separate labels so detections stay clean.
        </li>
        <li>
          Add a describing word when it actually changes what you want to find.{" "}
          <span className="font-mono text-foreground/90">ripe strawberries</span> only
          finds the ripe ones; <span className="font-mono text-foreground/90">rusty fence</span>{" "}
          ignores fences in good condition.
        </li>
        <li>
          Skip abstract concepts. The model can detect a{" "}
          <span className="font-mono text-foreground/90">smile</span> but not{" "}
          <span className="font-mono text-foreground/90">happiness</span>; if you can&rsquo;t draw a box around it, it won&rsquo;t work.
        </li>
      </ul>
    </InfoPopover>
  );
}

function SettingsHelpPopover() {
  return (
    <InfoPopover ariaLabel="Settings help">
      <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
        Threshold modes
      </h3>
      <ul className="text-xs text-foreground/70 leading-relaxed space-y-2">
        <li>
          <span className="text-foreground/90 font-medium">Normal</span>: permissive. Picks up borderline detections so you have more candidates to review and prune.
        </li>
        <li>
          <span className="text-foreground/90 font-medium">Stricter</span>: only confident matches survive. Useful when a generic label (
          <span className="font-mono text-foreground/90">road</span>,{" "}
          <span className="font-mono text-foreground/90">person</span>) would otherwise over-fire.
        </li>
        <li>
          <span className="text-foreground/90 font-medium">Custom</span>: appears when you&rsquo;ve dialed in your own values in the parameters menu (gear icon).
        </li>
      </ul>
      <p className="text-xs text-foreground/45 mt-3 leading-relaxed">
        Start on Normal. It&rsquo;s easier to review extras than to chase missed ones. Move to Stricter if false positives outnumber real detections.
      </p>
    </InfoPopover>
  );
}

function SynonymsHelpPopover() {
  return (
    <InfoPopover ariaLabel="Synonyms help">
      <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
        Synonym expansion
      </h3>
      <p className="text-xs text-foreground/70 leading-relaxed mb-3">
        Before sending your labels to the detector, an LLM suggests a few alternative phrasings for each one (e.g. <span className="font-mono text-foreground/90">car</span> picks up <span className="font-mono text-foreground/90">automobile</span>, <span className="font-mono text-foreground/90">sedan</span>). Detected variants are folded back to your label in post.
      </p>
      <ul className="text-xs text-foreground/70 leading-relaxed space-y-2">
        <li>
          <span className="text-foreground/90 font-medium">On</span>: better recall on broad labels (car, dog, building) where the detector responds to common subtypes.
        </li>
        <li>
          <span className="text-foreground/90 font-medium">Off</span>: only the exact words you typed reach the detector. Pick this when your labels are specific or technical (pothole, stop sign) and any expansion would just add false positives.
        </li>
      </ul>
      <p className="text-xs text-foreground/45 mt-3 leading-relaxed">
        Synonyms are cached per project, so toggling off won&rsquo;t re-call the LLM on the next run.
      </p>
    </InfoPopover>
  );
}

function HideSmallHelpPopover() {
  return (
    <InfoPopover ariaLabel="Hide small help">
      <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
        Hide small boxes
      </h3>
      <p className="text-xs text-foreground/70 leading-relaxed mb-3">
        Boxes that would shrink below the model&rsquo;s detection floor when the image is resized to the chosen <span className="font-mono text-foreground/85">Target input shape</span> are flagged as <span className="text-amber-300">small</span> or <span className="text-rose-300">won&rsquo;t detect</span>. Toggling Hide Small dims those boxes in the editor and excludes them from the size warnings on each thumbnail.
      </p>
      <p className="text-xs text-foreground/70 leading-relaxed">
        Useful when you&rsquo;re mid-label and don&rsquo;t want the warning chips drawing attention away from the boxes that will actually train. The boxes themselves stay in the manifest either way.
      </p>
    </InfoPopover>
  );
}


function VlmReviewHelpPopover() {
  return (
    <InfoPopover ariaLabel="AI Review help">
      <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">
        AI Review
      </h3>
      <p className="text-xs text-foreground/70 leading-relaxed mb-3">
        After detection, PixelKit takes a second look at each box and decides
        whether the assigned label actually matches the image. Mismatches get
        flagged.
      </p>
      <ul className="text-xs text-foreground/70 leading-relaxed space-y-2">
        <li>
          <span className="text-foreground/90 font-medium">Off</span>: skip the check entirely. Fastest auto-label runs.
        </li>
        <li>
          <span className="text-foreground/90 font-medium">Manual</span>: flagged boxes stay in the manifest with a red badge in the editor. You decide whether to verify or delete.
        </li>
        <li>
          <span className="text-foreground/90 font-medium">Auto-reject</span>: flagged boxes are dropped before saving. Most aggressive, useful when you trust the reviewer&rsquo;s judgement and want a clean dataset out of the gate.
        </li>
      </ul>
    </InfoPopover>
  );
}

function ThresholdReadout({ label, value }: { label: string; value: number }) {
  // Read-only display for one of the three threshold parameters
  // (Box / Text / NMS). Small grey caption, mono tabular numeric so
  // values line up across the row regardless of digit width.
  return (
    <div className="rounded-lg border border-foreground/[0.06] bg-foreground/[0.015] px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wider text-foreground/40">
        {label}
      </div>
      <div className="mt-1 font-mono text-sm text-foreground/85 tabular-nums">
        {value.toFixed(2)}
      </div>
    </div>
  );
}


function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  disabled = false,
  disabledTitle,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; title?: string }[];
  disabled?: boolean;
  disabledTitle?: string;
}) {
  return (
    <div
      className={[
        "inline-flex rounded-full border border-[var(--border)] p-0.5",
        disabled ? "opacity-50 cursor-not-allowed" : "",
      ].join(" ")}
      title={disabled ? disabledTitle : undefined}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => !disabled && onChange(opt.value)}
          disabled={disabled}
          title={disabled ? disabledTitle : opt.title}
          className={[
            "rounded-full px-3 py-1 text-xs uppercase tracking-wide transition-colors",
            value === opt.value
              ? "bg-foreground text-background"
              : "text-[var(--muted)] hover:text-foreground",
            disabled ? "cursor-not-allowed" : "",
          ].join(" ")}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// Target input tensor H×W. STM32N6 Neural-ART friendly square shapes, match
// the strides used by FOMO/MobileNet-SSD/YOLO-tiny family detectors.
const INPUT_SHAPES = [
  "96x96", "128x128", "160x160", "192x192", "224x224", "256x256",
  "320x320", "512x512", "640x640",
];

// Detection backbones lose objects whose smallest side gets squeezed below
// roughly two grid cells. Hard floor 12 px (won't detect at all), warning
// band 12-24 px (detector becomes unreliable). Same thresholds whatever the
// input shape, they're a property of the network's stride, not the image.
const BOX_FAIL_PX = 12;
const BOX_WARN_PX = 24;
type BoxStatus = "ok" | "warn" | "fail";

function parseInputShape(s: string): { w: number; h: number } {
  const [w, h] = s.split("x").map((n) => parseInt(n, 10));
  return { w: Number.isFinite(w) ? w : 256, h: Number.isFinite(h) ? h : 256 };
}

// Letterbox scaling, image fits inside the target without distortion. Same
// scale factor on both axes, so the box's worst-side bottleneck is preserved.
function scaledMinSide(box: EditableBox, imgW: number, imgH: number, inputShape: string): number {
  if (!imgW || !imgH) return Infinity;
  const t = parseInputShape(inputShape);
  const s = Math.min(t.w / imgW, t.h / imgH);
  return Math.min(Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0)) * s;
}

function statusFor(minSide: number): BoxStatus {
  if (minSide < BOX_FAIL_PX) return "fail";
  if (minSide < BOX_WARN_PX) return "warn";
  return "ok";
}

// Same hash → hue used by the project cards in HomeView/ProjectsView, so
// chips on the project page colour-match the chips on the card you came from.
function hueForName(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function formatCreated(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// Phrase pools for the box-size hint. We pick one at random per
// (inputShape × counts) state so users see varied wording but the line
// stays stable while they're staring at the same screen.
const TIP_FAIL_PHRASES = [
  "Some images have bounding boxes that are too small to be detected.",
  "A few labels will fall below the detector's minimum size after scaling.",
  "At this input shape, some boxes shrink past what the model can pick up.",
  "Some boxes won't survive the resize, the detector will miss them.",
];
const TIP_WARN_PHRASES = [
  "Some images have bounding boxes which may be difficult to detect.",
  "A few labels are borderline small, detection might be unreliable.",
  "Some boxes are close to the minimum detectable size after scaling.",
  "Detection on the smallest boxes may flicker at this input shape.",
];
const TIP_MIXED_PHRASES = [
  "Some boxes are too small to detect, others are borderline at this input shape.",
  "Mixed picture: a few labels won't be picked up, more are borderline small.",
];

function UploadProgressCard({
  progress,
}: {
  progress: {
    phase: "preparing" | "extracting" | "uploading" | "processing";
    fileCount: number;
    bytesLoaded: number;
    bytesTotal: number;
    extractedFrames?: number;
    totalFrames?: number;
    sourceName?: string;
  };
}) {
  const pct = progress.bytesTotal > 0
    ? Math.min(100, (progress.bytesLoaded / progress.bytesTotal) * 100)
    : 0;
  const mbLoaded = (progress.bytesLoaded / (1024 * 1024)).toFixed(1);
  const mbTotal = (progress.bytesTotal / (1024 * 1024)).toFixed(1);
  const phaseLabel =
    progress.phase === "preparing"
      ? "Resizing locally…"
      : progress.phase === "extracting"
      ? `Extracting frames${progress.sourceName ? ` from ${progress.sourceName}` : ""}…`
      : progress.phase === "uploading"
      ? "Uploading"
      : "Processing on server…";
  const isUploading = progress.phase === "uploading" && progress.bytesTotal > 0;
  const isExtracting = progress.phase === "extracting" && (progress.totalFrames ?? 0) > 0;
  const extractPct = isExtracting && progress.totalFrames
    ? Math.min(100, ((progress.extractedFrames ?? 0) / progress.totalFrames) * 100)
    : 0;
  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-amber-300/30 px-6 py-4"
      style={{
        background:
          "radial-gradient(120% 140% at 100% 100%, rgba(249,115,22,0.10) 0%, rgba(249,115,22,0) 55%), linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 45%, rgba(255,255,255,0) 100%), #141416",
        boxShadow:
          "0 1px 0 rgb(var(--foreground-rgb) / 0.05) inset, 0 0 24px rgba(251, 146, 60, 0.10), 0 0 48px rgba(251, 146, 60, 0.06)",
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-foreground/85">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-300 opacity-70" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-amber-300" />
          </span>
          {phaseLabel}
          {progress.fileCount > 0 && (
            <span className="text-foreground/45">
              · {progress.fileCount} image{progress.fileCount === 1 ? "" : "s"}
            </span>
          )}
        </div>
        {isUploading && (
          <span className="text-xs text-foreground/55 tabular-nums">
            {mbLoaded} / {mbTotal} MB · {Math.round(pct)}%
          </span>
        )}
        {isExtracting && (
          <span className="text-xs text-foreground/55 tabular-nums">
            {progress.extractedFrames ?? 0} / {progress.totalFrames} frames · {Math.round(extractPct)}%
          </span>
        )}
      </div>
      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-foreground/[0.08]">
        {isUploading ? (
          <div
            className="h-full rounded-full transition-all duration-200"
            style={{
              width: `${pct}%`,
              background: "linear-gradient(90deg, #fb923c, #f97316)",
              boxShadow: "0 0 14px rgba(249,115,22,0.5)",
            }}
          />
        ) : isExtracting ? (
          <div
            className="h-full rounded-full transition-all duration-200"
            style={{
              width: `${extractPct}%`,
              background: "linear-gradient(90deg, #fb923c, #f97316)",
              boxShadow: "0 0 14px rgba(249,115,22,0.5)",
            }}
          />
        ) : (
          // Indeterminate sweep for preparing / processing phases, we
          // don't have a meaningful percentage to show there. Reuses the
          // existing global `progress-sweep` keyframe so we don't ship a
          // second one-off animation.
          <div
            className="h-full w-1/3 progress-sweep"
            style={{
              background: "linear-gradient(90deg, transparent, #f97316, transparent)",
            }}
          />
        )}
      </div>
      <p className="mt-2 text-[11px] text-amber-200/85">
        Don&rsquo;t close this window, your upload is in progress.
      </p>
    </div>
  );
}

function HideSmallToggle({
  value,
  onChange,
  warn,
  fail,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  warn: number;
  fail: number;
}) {
  const total = warn + fail;
  const disabled = total === 0;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      disabled={disabled}
      onClick={() => onChange(!value)}
      title={
        disabled
          ? "No small or too-small boxes at this input shape."
          : value
          ? `Showing all boxes. Click to dim ${total} small/too-small.`
          : `Dim ${total} small/too-small box${total === 1 ? "" : "es"}.`
      }
      className={[
        "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] uppercase tracking-wider transition-colors",
        disabled
          ? "border-foreground/10 bg-foreground/[0.02] text-foreground/30 cursor-not-allowed"
          : value
          ? "border-amber-300/45 bg-amber-300/[0.08] text-amber-100 hover:bg-amber-300/[0.12]"
          : "border-foreground/15 bg-foreground/[0.04] text-foreground/70 hover:bg-foreground/[0.08] hover:text-foreground",
      ].join(" ")}
    >
      <span
        aria-hidden
        className={[
          "h-3.5 w-6 rounded-full p-0.5 transition-colors flex",
          value ? "bg-amber-300/70 justify-end" : "bg-foreground/15",
        ].join(" ")}
      >
        <span className="h-2.5 w-2.5 rounded-full bg-[#141416]" />
      </span>
      Hide small
    </button>
  );
}

function BoxSizeTip({ warn, fail }: { warn: number; fail: number }) {
  // useMemo keyed on the summary signature so the phrase doesn't reroll on
  // every render but does change when the warning state changes.
  const message = useMemo(() => {
    const pool = fail > 0 && warn > 0
      ? TIP_MIXED_PHRASES
      : fail > 0
      ? TIP_FAIL_PHRASES
      : warn > 0
      ? TIP_WARN_PHRASES
      : null;
    if (!pool) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }, [warn, fail]);

  if (!message) return null;
  return (
    <p
      className="text-xs text-amber-200/85 max-w-md"
      style={{ textShadow: "0 0 10px rgba(251, 146, 60, 0.35), 0 0 18px rgba(251, 146, 60, 0.12)" }}
    >
      <span className="text-amber-300 font-medium mr-1">Tip:</span>
      {message}
    </p>
  );
}

function ProjectMeta({
  owner,
  ownerInfo,
  createdAt,
  tags,
  paletteSeed,
}: {
  owner: string;
  ownerInfo: { name: string | null; image: string | null } | null;
  createdAt: string | null;
  tags: string[];
  paletteSeed: string;
}) {
  const hue = hueForName(paletteSeed);
  const tagColour = (i: number) => {
    const hues = [hue, (hue + 40) % 360, (hue + 80) % 360];
    return `hsla(${hues[i % hues.length]}, 70%, 55%, 0.95)`;
  };
  const handle = owner || "";
  const display = ownerInfo?.name || handle;
  const initial = (display[0] ?? "?").toUpperCase();
  const created = formatCreated(createdAt);

  if (!owner && tags.length === 0) return null;

  return (
    <div className="mt-4 flex items-center gap-x-4 gap-y-2 flex-wrap text-xs text-foreground/55">
      {owner && (
        <div className="flex items-center gap-2 min-w-0">
          {ownerInfo?.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={ownerInfo.image}
              alt=""
              className="h-5 w-5 rounded-full object-cover shrink-0"
              referrerPolicy="no-referrer"
            />
          ) : (
            <span
              className="h-5 w-5 rounded-full grid place-items-center text-[9px] font-semibold text-[var(--foreground)] shrink-0"
              style={{ backgroundImage: `linear-gradient(135deg, hsl(${hue},70%,55%), hsl(${(hue + 60) % 360},70%,55%))` }}
            >
              {initial}
            </span>
          )}
          <span className="truncate text-foreground/75">@{handle}</span>
        </div>
      )}
      {created && (
        <span className="text-foreground/45">created {created}</span>
      )}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {tags.map((t, i) => (
            <span
              key={t}
              className="rounded-full px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider text-black truncate max-w-[10rem]"
              style={{ backgroundColor: tagColour(i) }}
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function BetaLocked({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative">
      <div aria-hidden className="pointer-events-none select-none [filter:blur(14px)] opacity-60">
        {children}
      </div>
      <div className="absolute inset-0 flex items-center justify-center bg-black/30">
        <div className="rounded-2xl border border-foreground/10 bg-foreground/[0.04] px-6 py-4 text-center max-w-sm">
          <div className="text-sm font-medium text-foreground/90">Not yet available in beta</div>
          <div className="mt-1 text-xs text-foreground/50">This section is coming soon.</div>
        </div>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={[
        "px-4 py-3 text-sm border-b-2 -mb-px transition-colors",
        active ? "border-[var(--foreground)] text-[var(--foreground)]" : "border-transparent text-foreground/40 hover:text-foreground",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function ThresholdControl({
  label,
  value,
  onChange,
  presets,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  presets?: { label: string; value: number }[];
}) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-2">
        <span className="text-[var(--muted)]">{label}</span>
        <span className="font-mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0.05}
        max={0.9}
        step={0.05}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-white"
      />
      {presets && presets.length > 0 && (
        <div className="mt-2 flex gap-1">
          {presets.map((p) => {
            const active = Math.abs(value - p.value) < 0.001;
            return (
              <button
                key={p.label}
                onClick={() => onChange(p.value)}
                className={[
                  "flex-1 rounded-full px-2 py-1 text-[10px] uppercase tracking-wide transition-colors border",
                  active
                    ? "bg-foreground text-background border-[var(--foreground)]"
                    : "border-[var(--border)] text-[var(--muted)] hover:border-zinc-500 hover:text-foreground",
                ].join(" ")}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ImageThumb({
  result,
  boxes,
  verdict,
  projectName,
  apiBase,
  inputShape,
  isCover = false,
  hideSmall = false,
  cacheKey,
  onClick,
}: {
  result: Result;
  boxes: EditableBox[];
  verdict?: Verdict;
  projectName: string;
  apiBase: string;
  inputShape: string;
  isCover?: boolean;
  hideSmall?: boolean;
  /** Cache-buster appended to the annotated thumbnail URL so a
      backend re-bake (lite refresh, manual edit, etc.) is reflected
      in the grid without a hard reload. Typically the manifest's
      `updatedAt`. */
  cacheKey?: string | null;
  onClick: () => void;
}) {
  const isPending = !!result.pending;
  // Use the backend-baked preview JPEG (downsized + translucent green masks
  // baked in) when one exists. Falls back to the raw original for pending
  // images. The preview is ~20-50 KB so the grid loads instantly.
  const v = cacheKey ? `?v=${encodeURIComponent(cacheKey)}` : "";
  const url = !isPending && result.annotated
    ? `${apiBase}/api/projects/${projectName}/files/${encodeURIComponent(result.annotated)}${v}`
    : `${apiBase}/api/projects/${projectName}/originals/${encodeURIComponent(result.image)}${v}`;

  // Worst-case box status on this image at the current target input shape ,
  // drives a small corner badge so the user can spot problematic frames.
  // Skipped on the project's cover image: that's a showcase thumbnail and
  // a "Too small" chip there competes for attention. Also skipped when the
  // user has toggled "hide small", they've explicitly opted out of the
  // chips already.
  let worst: BoxStatus = "ok";
  if (!hideSmall && !isCover && !isPending && result.size?.width && result.size?.height) {
    for (const b of boxes) {
      const s = statusFor(scaledMinSide(b, result.size.width, result.size.height, inputShape));
      if (s === "fail") { worst = "fail"; break; }
      if (s === "warn") worst = "warn";
    }
  }

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className="cursor-pointer rounded-xl border border-foreground/10 hover:border-foreground/25 transition-colors overflow-hidden bg-foreground/[0.02]"
    >
      <ThumbImage src={url} pending={isPending} verdict={verdict} sizeStatus={worst} />
      <div className="px-3 py-2 flex justify-between items-center text-xs">
        <span className="truncate text-foreground/80">{result.image}</span>
        <span className="text-foreground/40 tabular-nums shrink-0 ml-2">
          {isPending ? "," : `${boxes.length} ${boxes.length === 1 ? "box" : "boxes"}`}
        </span>
      </div>
    </div>
  );
}

function ThumbImage({
  src,
  pending,
  verdict,
  sizeStatus,
}: {
  src: string;
  pending: boolean;
  verdict?: Verdict;
  sizeStatus?: BoxStatus;
}) {
  // The image is rendered by the wrapper's CSS background, there is no
  // <img> tag in the DOM tree, no object-fit semantics, no aspect-ratio
  // negotiation. The wrapper has explicit width and height, so background-size:
  // cover paints across the entire box. This is the most robust way to render
  // a "filled" thumbnail in a grid.
  return (
    <div
      className="relative bg-[var(--background)]"
      style={{
        width: "100%",
        height: "200px",
        backgroundImage: `url("${src}")`,
        backgroundSize: "cover",
        backgroundPosition: "center center",
        backgroundRepeat: "no-repeat",
        opacity: pending ? 0.7 : 1,
      }}
    >
      {pending && (
        <div className="absolute top-2 left-2 rounded-full bg-red-600/90 border border-red-400/70 text-[var(--foreground)] px-2 py-0.5 text-[10px] leading-normal font-semibold uppercase tracking-wider shadow-md">
          Unlabelled
        </div>
      )}
      {!pending && verdict && (
        <div
          className={[
            "absolute top-2 left-2 rounded-full px-2 py-0.5 text-[10px] leading-normal font-semibold uppercase tracking-wider shadow-md",
            verdict === "good" ? "bg-emerald-600 text-[var(--foreground)]" : "bg-red-600 text-[var(--foreground)]",
          ].join(" ")}
        >
          {verdict}
        </div>
      )}
      {!pending && sizeStatus && sizeStatus !== "ok" && (
        <div
          title={sizeStatus === "fail" ? "At least one box is too small for this input shape, won't be detected" : "At least one box is borderline small for this input shape"}
          className={[
            "absolute bottom-2 left-2 rounded-full px-2 py-0.5 text-[10px] leading-normal font-semibold uppercase tracking-wider shadow-md",
            sizeStatus === "fail" ? "bg-red-500/90 text-[var(--foreground)]" : "bg-amber-400/90 text-black",
          ].join(" ")}
        >
          {sizeStatus === "fail" ? "Too small" : "Small"}
        </div>
      )}
    </div>
  );
}

function EditModal({
  result,
  boxes,
  inputShape,
  hideSmall = false,
  onChange,
  verdict,
  onVerdict,
  onDelete,
  onClearLabels,
  projectName,
  apiBase,
  projectTags,
  username,
  readOnly = false,
  labelling = false,
  onClose,
  onPrev,
  onNext,
  onCascadeRelabel,
}: {
  result: Result;
  boxes: EditableBox[];
  inputShape: string;
  hideSmall?: boolean;
  onChange: (next: EditableBox[]) => void;
  verdict?: Verdict;
  onVerdict: (v: Verdict | null) => void;
  onDelete: () => void;
  onClearLabels: () => Promise<void> | void;
  projectName: string;
  apiBase: string;
  projectTags: string[];
  username: string;
  readOnly?: boolean;
  /** True while a label/lite job is running on the backend. The
      editor stays mounted so the user can watch progress, but every
      edit is locked behind a banner, saving while a runner is
      mutating manifest.editedBoxes asynchronously would clobber
      the new boxes the job is adding. */
  labelling?: boolean;
  onClose: () => void;
  onPrev?: () => void;
  /** Apply a Label Cascade rename across project-wide editedBoxes
      so the parent's state reflects the change immediately, not on
      the next 4-second poll tick. */
  onCascadeRelabel?: (targets: { image: string; box_id: string }[], newLabel: string) => void;
  onNext?: () => void;
}) {
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [sizeFilter, setSizeFilter] = useState<"all" | "hide" | "only">("all");
  // Similar-label suggestion modal state. Populated by the BoxEditor's
  // onLabelRenamed callback after a similarity lookup succeeds.
  const [similarPrompt, setSimilarPrompt] = useState<{
    oldLabel: string;
    newLabel: string;
    matches: SimilarMatch[];
  } | null>(null);

  // Plan quota gate. When the user has burned through this period's
  // auto-labelling allowance we strip the AI handlers off BoxEditor:
  //   - onPointDetect undefined  -> "Click to detect" button hides
  //   - onBoxDrawn undefined     -> manual draws skip the segmentation round-trip
  //   - onClassifyBox undefined  -> manual draws skip the label guess
  // Result: the user can still draw boxes manually, paint masks via
  // the Edit-mask tool, and type labels, pure manual labelling.
  const planData = usePlan();
  const aiAllowed = !planData?.over.anyLabelLimit;

  // Kept for re-enabling when _EMBEDDINGS_ENABLED flips back on.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const triggerSimilarLabelLookup = useCallback(async (boxId: string, oldLabel: string, newLabel: string) => {
    if (readOnly) return;
    try {
      const r = await fetch(`${apiBase}/api/projects/${projectName}/embeddings/find_similar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: result.image,
          box_id: boxId,
          new_label: newLabel,
          old_label: oldLabel,
        }),
      });
      if (!r.ok) return;
      const d = await r.json();
      const matches: SimilarMatch[] = Array.isArray(d?.matches) ? d.matches : [];
      if (matches.length > 0) {
        setSimilarPrompt({ oldLabel, newLabel, matches });
      }
    } catch (e) {
      // Silent, the suggestion is optional, never blocks editing.
      console.debug("[similar-labels] lookup failed", e);
    }
  }, [readOnly, apiBase, projectName, result.image]);

  // Tag every box with its size status against the chosen input shape, then
  // apply the user's filter mode. Computed in the same memo so the filter
  // dropdown and the per-box chips stay in sync with the canvas.
  const boxesWithStatus = useMemo(() => {
    const W = result.size?.width ?? 0;
    const H = result.size?.height ?? 0;
    return boxes.map((b) => ({
      box: b,
      status: W && H ? statusFor(scaledMinSide(b, W, H, inputShape)) : ("ok" as BoxStatus),
    }));
  }, [boxes, result.size, inputShape]);

  const visibleBoxes = useMemo(() => {
    if (sizeFilter === "hide") return boxesWithStatus.filter((b) => b.status === "ok").map((b) => b.box);
    if (sizeFilter === "only") return boxesWithStatus.filter((b) => b.status !== "ok").map((b) => b.box);
    return boxes;
  }, [boxesWithStatus, sizeFilter, boxes]);

  // Per-box id → size status, fed to BoxEditor so warned boxes paint amber
  // and failed boxes paint red on the canvas (and in the side list). Lets
  // the user see at a glance which labels will survive the input scaling.
  const sizeStatusMap = useMemo(() => {
    const m: Record<string, "ok" | "warn" | "fail"> = {};
    for (const { box, status } of boxesWithStatus) m[box.id] = status;
    return m;
  }, [boxesWithStatus]);

  // BoxEditor onChange returns the boxes it currently sees, when we're
  // filtering, that's a subset. Merge edits back into the full list keyed
  // by box id so filtered-out boxes don't disappear from the manifest.
  // When the user edits a box that previously got the "Verified"
  // badge from the validator, downgrade the validation source to "manual"
  // so the chip reads "Manual", Verified is reserved for things the
  // validator checked at the boxes' current geometry/label. We only flip
  // when something actually changed: position, mask reference, or
  // label.
  const downgradeIfEdited = (oldList: EditableBox[], nextList: EditableBox[]): EditableBox[] => {
    const prev = new Map(oldList.map((b) => [b.id, b]));
    return nextList.map((b) => {
      const old = prev.get(b.id);
      if (!old) return b;
      const v = b.validation;
      if (!v || v.source !== "auto") return b;
      const moved =
        old.x0 !== b.x0 || old.y0 !== b.y0 || old.x1 !== b.x1 || old.y1 !== b.y1;
      const renamed = (old.label || "") !== (b.label || "");
      const maskChanged = old.mask !== b.mask;
      if (!(moved || renamed || maskChanged)) return b;
      return {
        ...b,
        validation: {
          ...v,
          source: "manual",
          confidence: 1.0,
          reason: "user-edited",
        },
      };
    });
  };

  const handleEditorChange = (next: EditableBox[]) => {
    const stamped = downgradeIfEdited(boxes, next);
    if (sizeFilter === "all") {
      onChange(stamped);
      return;
    }
    // BoxEditor only sees the filtered subset, so `next` is missing
    // the boxes that were filtered out. Anything in the visible set
    // that's gone from `next` was *deleted*, drop it. Anything
    // outside the visible set (filtered out) stays untouched.
    const visibleIds = new Set(visibleBoxes.map((b) => b.id));
    const byId = new Map(stamped.map((b) => [b.id, b]));
    const merged = boxes
      .filter((b) => !visibleIds.has(b.id) || byId.has(b.id))
      .map((b) => byId.get(b.id) ?? b)
      .concat(stamped.filter((b) => !boxes.some((x) => x.id === b.id)));
    onChange(merged);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      // Don't hijack arrows/escape while typing in a label rename, search,
      // mask-painter input, etc.
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "ArrowLeft" && onPrev) {
        e.preventDefault();
        onPrev();
        return;
      }
      if (e.key === "ArrowRight" && onNext) {
        e.preventDefault();
        onNext();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onPrev, onNext]);

  return (
    <div
      className="fixed inset-0 z-40 flex flex-col"
      role="dialog"
      aria-modal="true"
      style={{
        // Apple-style frosted backdrop: stronger blur with a touch of
        // colour-saturation lift so the dimmed page underneath has the
        // same depth as a macOS sheet rather than a flat black slab.
        background:
          "linear-gradient(180deg, rgba(8,8,10,0.78) 0%, rgba(8,8,10,0.86) 100%)",
        backdropFilter: "blur(28px) saturate(140%)",
        WebkitBackdropFilter: "blur(28px) saturate(140%)",
      }}
    >
      <header
        className="flex items-center justify-between gap-4 px-6 py-3"
        style={{
          background:
            "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.04) 0%, rgb(var(--foreground-rgb) / 0.015) 100%)",
          borderBottom: "1px solid rgb(var(--foreground-rgb) / 0.07)",
          boxShadow: "0 1px 0 rgb(var(--foreground-rgb) / 0.04) inset",
        }}
      >
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-[13px] font-medium text-foreground/85 tracking-tight truncate">{result.image}</span>
          {result.pending && (
            <span className="rounded-full bg-foreground/[0.06] border border-foreground/15 text-foreground/70 px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider">
              Unlabelled
            </span>
          )}
          {(onPrev || onNext) && (
            <span className="hidden md:inline-flex items-center gap-1.5 text-[11px] text-foreground/35 ml-1">
              <kbd className="rounded-md border border-foreground/10 bg-foreground/[0.04] px-1.5 py-0.5 text-[10px] font-mono">←</kbd>
              <kbd className="rounded-md border border-foreground/10 bg-foreground/[0.04] px-1.5 py-0.5 text-[10px] font-mono">→</kbd>
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!readOnly && (
            <div className="inline-flex rounded-full border border-foreground/10 bg-foreground/[0.03] p-0.5 backdrop-blur-md">
              <button
                onClick={() => onVerdict(verdict === "bad" ? null : "bad")}
                className={[
                  "rounded-full px-3.5 py-1 text-[11px] uppercase tracking-wider font-medium transition-all duration-150",
                  verdict === "bad"
                    ? "bg-red-500/25 text-red-100 shadow-inner"
                    : "text-foreground/55 hover:text-foreground/80",
                ].join(" ")}
              >
                Bad
              </button>
              <button
                onClick={() => onVerdict(verdict === "good" ? null : "good")}
                className={[
                  "rounded-full px-3.5 py-1 text-[11px] uppercase tracking-wider font-medium transition-all duration-150",
                  verdict === "good"
                    ? "bg-emerald-500/25 text-emerald-100 shadow-inner"
                    : "text-foreground/55 hover:text-foreground/80",
                ].join(" ")}
              >
                Good
              </button>
            </div>
          )}
          {!readOnly && boxes.length > 0 && (
            <button
              onClick={() => setConfirmClear(true)}
              title="Clear all labels on this image"
              aria-label="Clear labels"
              className="inline-flex items-center gap-2 rounded-full border border-foreground/10 bg-foreground/[0.03] px-3 py-1.5 text-[11px] uppercase tracking-wider font-medium text-foreground/70 hover:bg-foreground/[0.08] hover:border-foreground/20 hover:text-foreground transition-all duration-150"
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M3 6h18" />
                <path d="m19 6-1.5 13.5a2 2 0 0 1-2 1.5h-7a2 2 0 0 1-2-1.5L5 6" />
                <path d="M10 11v6" />
                <path d="M14 11v6" />
              </svg>
              Clear
            </button>
          )}
          {!readOnly && (
            <button
              onClick={onDelete}
              title="Delete image"
              aria-label="Delete image"
              className="inline-flex items-center gap-2 rounded-full border border-red-400/25 bg-red-500/[0.07] px-3 py-1.5 text-[11px] uppercase tracking-wider font-medium text-red-300/90 transition-all duration-150 hover:bg-red-500/15 hover:border-red-400/50 hover:text-red-100"
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M3 6h18" />
                <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                <path d="M10 11v6" />
                <path d="M14 11v6" />
              </svg>
              Delete
            </button>
          )}
          <button
            onClick={onClose}
            className="ml-1 grid h-7 w-7 place-items-center rounded-full bg-foreground/[0.04] border border-foreground/10 text-foreground/55 hover:bg-foreground/[0.10] hover:border-foreground/20 hover:text-foreground transition-all duration-150"
            aria-label="close"
            title="Close (Esc)"
          >
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      </header>
      <div className="flex-1 min-h-0 p-5">
        {labelling && (
          <div className="mb-3 rounded-xl border border-amber-300/25 bg-amber-300/[0.04] px-4 py-2.5 flex items-start gap-3">
            <span aria-hidden className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-300/80 shrink-0" />
            <div className="min-w-0 text-[12px] text-foreground/75 leading-relaxed">
              <span className="text-amber-200/90 font-mono uppercase tracking-wider text-[10px]">Auto-labelling in progress</span>
              <span className="text-foreground/45"> · </span>
              Edits are locked while the labelling job runs, once it&rsquo;s
              finished you can review boxes, masks, and labels.
            </div>
          </div>
        )}
        {!labelling && !aiAllowed && (
          <div className="mb-3 rounded-xl border border-amber-300/25 bg-amber-300/[0.04] px-4 py-2.5 flex items-start gap-3">
            <span aria-hidden className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-300/80 shrink-0" />
            <div className="min-w-0 text-[12px] text-foreground/75 leading-relaxed">
              <span className="text-amber-200/90 font-mono uppercase tracking-wider text-[10px]">Manual mode</span>
              <span className="text-foreground/45"> · </span>
              You&rsquo;ve used your auto-labelling allowance for this period. Draw boxes,
              paint masks with the <span className="text-foreground/85">Edit mask</span> tool,
              and type labels manually until your quota resets.
            </div>
          </div>
        )}
        <div
          className="h-full overflow-hidden rounded-2xl border border-foreground/[0.06]"
          style={{
            background: "linear-gradient(180deg, rgb(var(--foreground-rgb) / 0.02) 0%, rgba(255,255,255,0) 100%), #0a0a0c",
            boxShadow:
              "0 1px 0 rgb(var(--foreground-rgb) / 0.04) inset, 0 30px 80px -20px rgb(var(--shadow-rgb) / 0.6)",
          }}
        >
          <BoxEditor
            // Force a remount on image change. Without this, internal
            // state inside BoxEditor (selectedId, focusedId, hoveredId,
            // editingId, drawMode, the in-progress mask painter) leaks
            // across arrow-key navigation, the most visible symptom
            // is "a segmentation looks selected but it's not on the
            // image" because selectedId still points at a box id from
            // the previous image. Keying on the filename is safe, it
            // changes exactly when the image swaps and is stable for
            // tweaks within the same image.
            key={result.image}
            imageUrl={`${apiBase}/api/projects/${projectName}/originals/${encodeURIComponent(result.image)}`}
            imageWidth={result.size.width}
            imageHeight={result.size.height}
            boxes={visibleBoxes}
            sizeStatuses={sizeStatusMap}
            muteSizeWarnings={hideSmall}
            sizeFilter={sizeFilter}
            onSizeFilterChange={setSizeFilter}
            onChange={handleEditorChange}
            projectTags={projectTags}
            readOnly={readOnly || labelling}
            onBoxDrawn={
              aiAllowed
                ? async (b) => {
                    const r = await fetch(`${apiBase}/api/projects/${projectName}/segment_box?user=${encodeURIComponent(username)}`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ image: result.image, box: [b.x0, b.y0, b.x1, b.y1] }),
                    });
                    if (!r.ok) return null;
                    const d = await r.json();
                    return d.mask ?? null;
                  }
                : undefined
            }
            onClassifyBox={
              aiAllowed && projectTags.length > 0
                ? async (b) => {
                    const r = await fetch(`${apiBase}/api/projects/${projectName}/classify_box?user=${encodeURIComponent(username)}`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ image: result.image, box: [b.x0, b.y0, b.x1, b.y1] }),
                    });
                    if (!r.ok) return null;
                    const d = await r.json();
                    return { label: d.label ?? null, score: d.score ?? null };
                  }
                : undefined
            }
            onPointDetect={
              aiAllowed
                ? async (point) => {
                    const r = await fetch(`${apiBase}/api/projects/${projectName}/detect_point?user=${encodeURIComponent(username)}`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ image: result.image, point: [point.x, point.y] }),
                    });
                    if (!r.ok) return null;
                    return await r.json();
                  }
                : undefined
            }
            // Per-rename Label Cascade lookups are off, the
            // project-wide review now runs after auto-labelling
            // completes (LabelCascadeReviewModal) and surfaces
            // every cluster regardless of label.
          />
        </div>
      </div>
      {confirmClear && (
        <ClearLabelsModal
          imageName={result.image}
          count={boxes.length}
          busy={clearing}
          onCancel={() => {
            if (!clearing) setConfirmClear(false);
          }}
          onConfirm={async () => {
            setClearing(true);
            try {
              await onClearLabels();
              setConfirmClear(false);
            } finally {
              setClearing(false);
            }
          }}
        />
      )}
      {similarPrompt && (
        <SimilarLabelsModal
          apiBase={apiBase}
          projectId={projectName}
          oldLabel={similarPrompt.oldLabel}
          newLabel={similarPrompt.newLabel}
          matches={similarPrompt.matches}
          onClose={() => setSimilarPrompt(null)}
          onRelabelled={(targets, newLabel) => {
            // Update the project-level editedBoxes state so the
            // rename is visible the moment the user navigates to
            // any of the affected images, instead of waiting for
            // the next 4s poll.
            onCascadeRelabel?.(targets, newLabel);
          }}
        />
      )}
    </div>
  );
}
