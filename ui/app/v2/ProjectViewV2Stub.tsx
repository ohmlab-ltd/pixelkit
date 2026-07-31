"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { BlurhashCanvas } from "react-blurhash";

import { Footer } from "../Footer";
import { LABEL_COLOURS, buildProjectLabelColourMap, colourForLabelStable, readableTextForBg } from "./OnboardLabelsV2";
import type { ReferenceImage } from "./OnboardReferencesV2";
import { BoxEditor, detectionsToBoxes, stripTransientBoxFlags, type EditableBox, type MaskShape, type Validation } from "../BoxEditor";
import { ReferenceImageEditor } from "./ReferenceImageEditor";
import { LabelJobCard, type LabelJobState } from "./LabelJobCard";
import { AugmentationsCard } from "./AugmentationsCard";
import { DerivedDatasetsBar } from "./DerivedDatasets";
import { OverviewPanel } from "./OverviewPanel";
import { DatasetHealthModal } from "./DatasetHealthModal";
import { DeleteLabelModal } from "./DeleteLabelModal";
import { ClearAllAnnotationsModal } from "./ClearAllAnnotationsModal";
import { OpenverseInlinePanel } from "./OpenverseInlinePanel";
import { ReviewModeV2 } from "./ReviewModeV2";
import {
  ANNOT_WORKER_ENABLED,
  ANNOT_WORKER_BUFFER_THRESHOLD,
  BINARY_WIRE_ENABLED,
  aggregateLabelStatsInWorker,
  parseSingleAnnotationInWorker,
  parseViewportBatchInWorker,
  parseViewportBatchMsgpackInWorker,
  type ParsedRow,
} from "./workers/useAnnotationsWorker";
import {
  IDB_CACHE_ENABLED,
  getCachedAnnotation,
  getCachedAnnotationBatch,
  putCachedAnnotation,
  scheduleLruEviction,
} from "../../lib/annotationCache";
import { subscribePressure } from "../../lib/memoryPressure";
import {
  ProjectStore,
  STORE_V2_ENABLED,
  useImport,
  type StoreImport,
} from "./store/projectStore";
import { VideoFrameModal } from "./VideoFrameModal";
import { extractVideoFrames, MAX_VIDEO_BYTES } from "../../lib/videoFrames";
import { ProjectSettingsV2 } from "./ProjectSettingsV2";
import { ExportModal } from "../ExportModal";
import { Tooltip } from "../Tooltip";
import { apiFetch } from "../../lib/apiFetch";
import { usePlan } from "../PlanPill";
import { patchProjectMeta, readProjectMeta } from "../../lib/projectMetaCache";
import { containsProfanity } from "../profanity";
import { PixelKitLoader } from "./PixelKitLoader";
import { resizeForUpload, isImageFile } from "../../lib/resize";
import { lookupUsers } from "../../lib/userCache";
import { ScrollToTop } from "../components/ScrollToTop";
import { useIdle } from "../../lib/useIdle";

// Auto-derived dataset behaviour, mirrored from the engine: a dataset
// with reference images behaves as "specific" (reference/embedding
// scoring), without them as "general" (plain text-prompt detection).
// There is no user-facing dataset-type control any more — this value
// is read-only UI state that internal code paths still branch on
// (e.g. suppressing the "No reference embeddings" warning).
type DatasetTypeValue = {
  type: "general" | "specific";
  reason?: string | null;
  source?: string | null;
};

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// Pipeline router for the imports flow. The portable engine is SAM3-only
// (the GroundingDINO "v2" import endpoints no longer exist), so charlie is
// the default; NEXT_PUBLIC_PIPELINE stays as an escape hatch.
const PIPELINE = (process.env.NEXT_PUBLIC_PIPELINE ?? "charlie").toLowerCase();
const IMPORTS_BASE = PIPELINE === "charlie" ? "/api/charlie/imports" : "/api/v2/imports";

// Feature flag: when on, skip the per-import synthetic ImportDetection[]
// allocation we used to drive the chip rail. Chips are now driven
// from m.labelStats (Record<label, count>) directly. On big projects
// this removes ~225k object allocations on cold open. Behind a flag
// while the rest of the codebase migrates off detections.length /
// detections-derived chip rendering for tile placeholders.
const NO_SYNTH_DETS = process.env.NEXT_PUBLIC_NO_SYNTH_DETS === "1";

// Annotation target per label (shown in the UI; enforced by the backend).
const ANNOTS_PER_LABEL = 5;

// Map a Project's max input size (px longest edge) to a JPEG byte budget for
// the client-side upload resize. The 1500 default keeps the historical 50KB
// budget EXACTLY, so any project that never touches the setting uploads
// byte-identically to before. Larger ceilings get a proportionally larger
// budget so the extra pixels aren't crushed straight back out by the byte cap
// (more storage is the opt-in cost of a higher-quality Project).
function uploadBytesForMaxSize(maxSide: number): number {
  // Byte budget per upload, scaled to the chosen resolution. Larger ceilings
  // get a generous budget so a full-resolution image (e.g. 4K) keeps good JPEG
  // quality instead of being squeezed — the resize never drops below maxSide,
  // so a too-tight budget would only hurt quality, but a 4K Project opted into
  // crisp originals, so give it the room.
  if (maxSide <= 1500) return 50 * 1024;
  if (maxSide <= 2048) return 400 * 1024;
  if (maxSide <= 3072) return 1024 * 1024;
  return Math.round(2.5 * 1024 * 1024);
}

// Whimsy strings used by the image-processing progress card. Same
// purpose as LABEL_PHRASES inside LabelJobCard, gives the user a
// rotating reassurance line while the upload / resize / safety-check
// queue chews through a batch.
const PROCESSING_PHRASES = [
  "Compressing and uploading…",
  "Running safety checks…",
  "Decoding image bytes…",
  "Rotating per EXIF…",
  "Generating thumbnails…",
  "Caching blurhashes…",
  "Stashing pixels for the labeller…",
  "Almost there, last few bytes…",
];
const AUGMENT_PHRASES = [
  "Warping perspectives just a touch…",
  "Pasting onto a fresh background…",
  "Adding a dash of motion blur…",
  "Rotating with theatrical flair…",
  "Sprinkling sensor noise…",
  "Nudging the colour cast around…",
  "Cutting objects out with SAM…",
  "Composing new scenes for the model…",
  "Boosting your dataset's variety…",
  "Inventing plausible variations…",
  "Re-rendering with a fresh palette…",
  "Cropping creatively…",
  "Layering domain shifts onto pixels…",
];
// Copy a File's bytes into a fresh ArrayBuffer-backed File right before
// upload. Safari can lose a Blob's backing memory between resize and the
// FormData read (canvas-toBlob detach, drag-source GC), so we re-read
// every file immediately before the POST. Chrome doesn't need it but the
// one byte-copy is cheap and removes a whole class of intermittent
// upload failures. Module-level twin of the per-image path's `reread`.
async function rereadFileBytes(file: File): Promise<File> {
  try {
    const buf = await file.arrayBuffer();
    if (buf.byteLength > 0) {
      return new File([buf], file.name, { type: file.type || "application/octet-stream" });
    }
  } catch {
    /* fall through to FileReader */
  }
  try {
    const buf = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const r = reader.result;
        if (r instanceof ArrayBuffer && r.byteLength > 0) resolve(r);
        else reject(new Error("FileReader produced empty/invalid buffer"));
      };
      reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
      reader.readAsArrayBuffer(file);
    });
    return new File([buf], file.name, { type: file.type || "application/octet-stream" });
  } catch {
    return file;
  }
}

// Target input shapes for the project. The training "Image size" options
// (model_registry imgsz choices) are kept equal to this list so the two
// pickers always offer the same sizes. 480x480 is included for st_yoloxn.
const INPUT_SHAPES = [
  "96x96", "128x128", "160x160", "192x192", "224x224", "256x256",
  "320x320", "480x480", "512x512", "640x640",
];
// Dataset view sections. State is OWNED BY THE SHELL (app/page.tsx —
// the Explorer tree's third level drives it); this view receives the
// active section + a setter via props. Exported so the page can type
// its state; keep in sync with ExplorerPane's DatasetSection.
export type ProjectTab = "overview" | "references" | "dataset" | "augmentations";

// Parse a createdAt value (ms epoch number, ms epoch string, or ISO
// timestamp) into ms-since-epoch. Returns NaN for unparseable values
// so the comparator can fall back to a stable secondary sort.
function parseCreatedAtMs(v: number | string | null | undefined): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  const asNum = Number(v);
  if (Number.isFinite(asNum)) return asNum;
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : NaN;
}

// Sort comparator: newest-first by createdAt, with id-derived
// timestamp as the fallback (locally-uploaded imports use
// `imp_<ms>_<rand>`) and id-string lexicographic as the last resort
// so the order stays stable.
// Placeholder detections (synthesised from /initial or /overview's
// n_detections + label_set) all have box=[0,0,0,0] and mask=null.
// Real detections from /annotations or per-image /annotations/{id}
// have non-zero box coords + an actual mask. We use this to decide
// merge precedence: if cur is a placeholder and fresh is too, prefer
// fresh (it's from a more recent fetch, the older one might be a
// stale sidecar). If cur is real, keep it (real geometry beats
// placeholder).
function hasRealDetections(arr: unknown): boolean {
  if (!Array.isArray(arr) || arr.length === 0) return false;
  for (const d of arr) {
    if (!d || typeof d !== "object") continue;
    const det = d as { box?: number[]; mask?: unknown };
    if (Array.isArray(det.box) && (det.box[2] > 0 || det.box[3] > 0)) return true;
    if (det.mask) return true;
  }
  return false;
}

function compareImportedMediaDesc(a: { id: string; createdAt?: number | string | null }, b: { id: string; createdAt?: number | string | null }): number {
  // DESCENDING by createdAt, newest upload at top-left of the
  // gallery, oldest at the bottom. handleImportFiles pre-assigns
  // timestamps in REVERSE drop order within each batch (first-
  // dropped file gets the highest timestamp), so:
  //   - the whole new batch sits above existing tiles
  //   - within the batch, first-dropped lands at top-left
  //   - the upload chain (which runs in drop order) processes
  //     top-left → bottom-right
  //   - labelling iterates DESC and starts at the same tile
  // All four directions stay aligned without anyone seeing the
  // gallery "fill in from the bottom".
  const ta = parseCreatedAtMs(a.createdAt);
  const tb = parseCreatedAtMs(b.createdAt);
  if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb - ta;
  if (Number.isFinite(ta) && !Number.isFinite(tb)) return -1;
  if (!Number.isFinite(ta) && Number.isFinite(tb)) return 1;
  const idMs = (id: string) => {
    const m = /^imp_(\d+)_/.exec(id);
    return m ? Number(m[1]) : NaN;
  };
  const ia = idMs(a.id);
  const ib = idMs(b.id);
  if (Number.isFinite(ia) && Number.isFinite(ib) && ia !== ib) return ib - ia;
  return a.id < b.id ? 1 : a.id > b.id ? -1 : 0;
}

function hueFor(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

export function ProjectViewV2Stub({
  projectName,
  labels,
  references,
  projectId,
  username,
  userImage = null,
  ownerUsername = null,
  readOnly = false,
  originTab = "workspaces",
  backToProjectId = null,
  firstLoad = null,
  section = "overview",
  onSectionChange,
  onClose,
  onReferencesChange,
}: {
  projectName: string;
  labels: string[];
  references: ReferenceImage[];
  /** Backend project UUID assigned by POST /api/v2/projects. May be
      null if the project create call failed; the page still renders
      from the in-memory references in that case. */
  projectId?: string | null;
  username: string;
  userImage?: string | null;
  /** Project owner's username pulled from the manifest. Drives the
      curator's @handle in the chrome, falls back to `username` (the
      viewer) when null. */
  ownerUsername?: string | null;
  /** True when the viewer is NOT the owner and the project is being
      viewed from the public feed. Disables rename / label edits /
      drop zone / annotations card / bulk delete / per-tile delete
      and collapses the tab strip to "Dataset" only. */
  readOnly?: boolean;
  /** Which tab the user came from when they opened this project.
      Drives the "Back to …" button copy on the loader + header so a
      public-feed origin reads "Back to projects" even when the
      project happens to be the viewer's own. */
  originTab?: "workspaces" | "projects" | "guide" | "pricing" | "terminal";
  /** When set, this dataset was opened from inside a Project (container): the
      back button reads "Back to project" and onClose returns to that Project. */
  backToProjectId?: string | null;
  /** First-load handoff hint from onboarding:
   *  "onboarding" → suppress the full-screen mount loader entirely.
   *                 HomeView is still showing its "Opening project…"
   *                 PixelKit overlay and we don't want a second
   *                 full-screen takeover stacked on top.
   *  null         → normal load, full-screen mount loader behaves as
   *                 it always has. */
  firstLoad?: "onboarding" | null;
  /** Active dataset section. Owned by the shell (app/page.tsx) so the
   *  Explorer tree's third-level rows and this view stay in sync; the
   *  view's own section jumps (Overview cards etc.) go through
   *  onSectionChange. */
  section?: ProjectTab;
  onSectionChange?: (section: ProjectTab) => void;
  onClose: () => void;
  onReferencesChange?: (next: ReferenceImage[]) => void;
}) {
  // The chrome's @handle uses the owner when present, otherwise the
  // viewer's handle (the historical default for own-project view).
  const displayHandle = (ownerUsername || username || "you").trim();
  // Whether the viewer CREATED this dataset. Editors viewing a teammate's
  // dataset can edit its content (not read-only) but are NOT the owner: the
  // chrome must show the OWNER's avatar (not the viewer's), and dataset Settings
  // (rename / cover / delete) stay owner-only. No owner on record → treat as
  // own (the common "open your own dataset" case).
  const isOwnDataset =
    !ownerUsername || ownerUsername.trim().toLowerCase() === (username || "").trim().toLowerCase();
  // Owner's avatar, fetched via lookupUsers (cached client-side for 24h)
  // whenever the viewer isn't the owner — so a teammate's dataset shows their
  // real avatar next to their handle, not a gradient initial or the viewer's.
  const [ownerImage, setOwnerImage] = useState<string | null>(null);
  useEffect(() => {
    if (!ownerUsername || isOwnDataset) {
      setOwnerImage(null);
      return;
    }
    let cancelled = false;
    void lookupUsers([ownerUsername]).then((map) => {
      if (cancelled) return;
      const info = map[ownerUsername.toLowerCase()];
      setOwnerImage(info?.image ?? null);
    });
    return () => { cancelled = true; };
  }, [ownerUsername, isOwnDataset]);
  // After 90s of no mouse/keyboard/touch/scroll/focus, the augment
  // poll suspends (see the deps on its useEffect below). Activity
  // flips back immediately. Cuts ~30 background API calls/min on
  // idle tabs that were eating Vercel observability quota for no
  // user-visible benefit.
  const isIdle = useIdle(90_000);

  const [refs, setRefs] = useState<ReferenceImage[]>(() => {
    // If the parent passed real refs (e.g. fresh from onboarding),
    // those win. Otherwise seed from the per-project localStorage
    // cache so the placeholder grid paints BlurHash gradients
    // instantly on reopen, before the /api/projects/{id} fetch
    // even returns. The hydration effect below replaces them once
    // the manifest lands.
    if (references.length > 0) return references;
    if (!projectId) return references;
    const cached = readProjectMeta(projectId);
    if (!cached?.refTiles?.length) return references;
    return cached.refTiles.map((t) => ({
      file: new File([], t.filename),
      preview: `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(t.filename)}`,
      width: t.width,
      height: t.height,
      boxes: [],
      referenceId: t.id,
      filename: t.filename,
      blurhash: t.blurhash ?? null,
    }));
  });
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log("[v2 view] references prop:", references.length, "items");
    if (references.length > 0) setRefs(references);
  }, [references]);

  // Hydrate references from the backend manifest when opening an
  // existing V2 project (refs prop is empty, projectId is set).
  // Each reference's image is fetched as a Blob so we can wrap it
  // in a File object, ReferenceImage.file is required by the
  // re-process / re-segment paths inside RefImageGrid. Bytes are
  // served by /api/v2/projects/{id}/references/{filename}.
  // refsHydratedRef survives across the StrictMode simulated unmount
  //, that's the property we need so the second effect run sees the
  // claim and skips. NO local `cancelled` flag here: the previous
  // version had one which got set to true by the simulated cleanup,
  // then on the second run the refsHydratedRef guard early-returned
  // and the original async A reached its `if (cancelled) return`
  // and skipped setRefs forever. Refs stayed empty and the grid
  // rendered nothing on reopen. Letting the async write state on
  // an unmounted component is fine, React 18 just dev-warns.
  // How many references the manifest told us to expect, used to
  // size the placeholder tile count while image bytes are
  // streaming in from the backend on reopen.
  // Seed from the workspace meta cache so the placeholder grid
  // renders with the right tile count on first paint, before the
  // manifest fetch resolves. Falls back to 0 for projects the
  // workspace hasn't listed yet (deep-link, fresh device, etc).
  const [expectedRefCount, setExpectedRefCount] = useState<number>(() => {
    if (!projectId) return 0;
    const cached = readProjectMeta(projectId);
    return Math.max(0, cached?.nReferences ?? 0);
  });
  // Lite dataset-stats snapshot from the /initial fetch. Handed to
  // DatasetStatsCard so the stats badge + label distribution paint in
  // the same frame as the first 20 tiles, instead of waiting on a
  // separate /dataset-stats?lite=true round-trip. Cleared when
  // projectId changes so a fresh open doesn't show stale numbers.
  // The "any" shape mirrors DatasetStatsCard's `DatasetStats` type
  // without re-importing it; the card narrows it back on receive.
  const [seedStats, setSeedStats] = useState<unknown | null>(null);

  // /initial fast-path hydration. Fires the moment a project opens and
  // returns project meta + first-20 import tiles + lite stats in a
  // SINGLE document served from disk, no compute, no manifest parse
  // on the request thread. The other hydration effects below
  // (refs/imports/label/annotations) still run but their /overview
  // calls dedup against an in-flight Cloudflare round-trip via
  // apiFetch's coalescer, so this is net-additive: first paint
  // happens in one RTT, the slower follow-ups fill in detail.
  const initialHydratedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    if (initialHydratedRef.current.has(projectId)) return;
    initialHydratedRef.current.add(projectId);
    const pid: string = projectId;
    // AbortController so a project-change mid-hydration cancels the
    // outgoing /initial fetch instead of letting it land into stale
    // state. Tied to the effect's cleanup.
    const ac = new AbortController();
    (async () => {
      try {
        // cache:no-store forces this past the browser's HTTP cache so
        // a destructive action followed by window.location.reload()
        // (which respects the disk cache) doesn't paint stale data
        // from the pre-action /initial response. apiFetch's in-flight
        // dedup still coalesces concurrent /initial calls into one
        // round-trip.
        const r = await apiFetch(
          `/api/v2/projects/${pid}/initial?n=20`,
          { cache: "no-store", signal: ac.signal },
        );
        if (!r.ok) return;
        type InitialResp = {
          name?: string;
          tags?: string[];
          // Snapshot of project tags as of the last label_charlie job.
          // Used by the Start-button's freshLabels heuristic to tell
          // "user just added this tag" apart from "this tag was
          // searched but never matched anywhere." Null on never-
          // labelled projects.
          labelsLastRun?: string[] | null;
          // Detection settings the dataset was last labelled with, so the
          // sliders + "relabel needed" state persist across reloads.
          settingsLastRun?: {
            // Slider trio is only present when the run pinned all three;
            // tiling keys persist independently (a default-slider run with
            // tiling on still records it).
            threshold?: number;
            mask_threshold?: number;
            min_relative_area?: number;
            tile_native?: boolean;
            tile_size?: number;
          } | null;
          label_aliases?: Record<string, string>;
          labelColours?: Record<string, string>;
          private?: boolean;
          // Cached auto-derived dataset behaviour (references → specific),
          // included so internal branches settle with the first frame
          // instead of after a separate /dataset-type round-trip.
          dataset_type?: {
            type?: string;
            reason?: string | null;
            source?: string | null;
          } | null;
          owner?: string | null;
          createdBy?: string | null;
          references?: {
            id: string;
            filename: string;
            originalFilename?: string;
            width?: number;
            height?: number;
            blurhash?: string | null;
            n_detections?: number;
          }[];
          imports?: {
            id: string;
            filename: string;
            originalFilename?: string;
            width?: number;
            height?: number;
            blurhash?: string | null;
            createdAt?: number | string | null;
            source?: { kind?: string; url?: string } | null;
            n_augmentations?: number;
            n_detections?: number;
            label_set?: string[];
            // New: per-label counts. Older backends won't set it
            // and we fall through to the label_set + n_detections
            // synthetic path below.
            label_stats?: Record<string, number>;
            has_edits?: boolean;
            labelledAt?: number | null;
          }[];
          imports_total?: number;
          stats?: unknown;
        };
        const j = (await r.json()) as InitialResp;
        // Project meta, feeds label-aliases hook + ownership flag.
        if (j.label_aliases && typeof j.label_aliases === "object") {
          setLabelAliases(j.label_aliases);
        }
        if (j.labelColours && typeof j.labelColours === "object") {
          setLabelColours(j.labelColours);
        }
        if (Array.isArray(j.tags) && labels.length === 0) {
          setEditLabels(j.tags);
        }
        if (typeof j.private === "boolean") {
          setIsPrivate(j.private);
        }
        // Seed the auto-derived dataset behaviour from the first-paint
        // payload (it used to arrive 100-1000 ms later via a standalone
        // /dataset-type fetch). Null → the dedicated effect fetches it.
        if (
          j.dataset_type
          && (j.dataset_type.type === "general" || j.dataset_type.type === "specific")
        ) {
          const next: DatasetTypeValue = {
            type: j.dataset_type.type,
            reason: typeof j.dataset_type.reason === "string" ? j.dataset_type.reason : "",
            source: typeof j.dataset_type.source === "string" ? j.dataset_type.source : "auto",
          };
          setDatasetType(next);
          patchProjectMeta(pid, { datasetType: next });
        }
        if (typeof (j as { max_input_size?: number }).max_input_size === "number") {
          maxInputSizeRef.current = (j as { max_input_size: number }).max_input_size;
          setMaxInputSize((j as { max_input_size: number }).max_input_size);
        }
        if (Array.isArray(j.labelsLastRun)) {
          setLabelsLastRun(j.labelsLastRun);
        } else if (j.labelsLastRun === null) {
          setLabelsLastRun(null);
        }
        // Pre-fill the detection sliders with the settings the dataset was
        // last labelled with, and seed the relabel baseline to them, so a
        // settings change after a reload is still detected as "relabel".
        if (j.settingsLastRun) {
          const s = j.settingsLastRun;
          // The trio may be absent (run with default sliders) while the
          // tiling keys are present — guard each read.
          const th = typeof s.threshold === "number" ? s.threshold : SAM3_DEFAULTS.threshold;
          const mt = typeof s.mask_threshold === "number" ? s.mask_threshold : SAM3_DEFAULTS.maskThreshold;
          const ma = typeof s.min_relative_area === "number" ? s.min_relative_area : SAM3_DEFAULTS.minRelativeArea;
          setSam3Threshold(th);
          setSam3MaskThreshold(mt);
          setSam3MinRelativeArea(ma);
          setTileNative(!!s.tile_native);
          setLastRunSettings({
            threshold: th,
            maskThreshold: mt,
            minRelativeArea: ma,
            tileNative: !!s.tile_native,
          });
        }
        // Refs metadata, gallery placeholders without detection
        // geometry. /annotations later fills in real boxes.
        const refList = j.references ?? [];
        if (refList.length > 0) {
          setExpectedRefCount(refList.length);
          const newUploadStatus: Record<string, "uploading" | "done" | "failed"> = {};
          const hydrated: ReferenceImage[] = refList.map((ref) => {
            const url = `${API}/api/v2/projects/${pid}/references/${encodeURIComponent(ref.filename)}`;
            newUploadStatus[url] = "done";
            return {
              file: new File([], ref.originalFilename || ref.filename),
              preview: url,
              width: ref.width,
              height: ref.height,
              boxes: [],
              referenceId: ref.id,
              filename: ref.filename,
              blurhash: ref.blurhash ?? null,
            };
          });
          setRefs((cur) => {
            const byId = new Map(hydrated.map((h) => [h.referenceId, h]));
            const merged: ReferenceImage[] = [];
            for (const c of cur) {
              if (c.referenceId && byId.has(c.referenceId)) {
                const fresh = byId.get(c.referenceId)!;
                merged.push({ ...fresh, boxes: c.boxes ?? [] });
                byId.delete(c.referenceId);
              } else {
                merged.push(c);
              }
            }
            for (const h of byId.values()) merged.push(h);
            return merged;
          });
          setRefUploadStatus((cur) => ({ ...newUploadStatus, ...cur }));
        }
        // First-20 imports. Skip the synthetic-detection allocation
        // entirely when the FE is running with NO_SYNTH_DETS=1 - the
        // chip rail reads labelStats directly. When the flag is off
        // we keep the old behaviour (placeholder ImportDetection[])
        // so older readers downstream don't regress.
        const imps = j.imports ?? [];
        const total = j.imports_total ?? imps.length;
        // Surface the true total immediately so the "Dataset N"
        // counter doesn't climb 20 → 100 → 941 as the remainder
        // batches stream in.
        setImportsTotal(total);
        const hydratedImports: ImportedMedia[] = imps.map((imp) => {
          const labelSet = imp.label_set ?? [];
          const nDetections = imp.n_detections ?? 0;
          // Prefer the backend-supplied label_stats when present.
          // Fall back to synthesising one from label_set so an older
          // backend / cached sidecar still renders chips.
          const labelStats: Record<string, number> = imp.label_stats
            ?? Object.fromEntries(labelSet.map((lab) => [lab, 0]));
          const placeholderDetections: ImportDetection[] = NO_SYNTH_DETS
            ? []
            : nDetections > 0
            ? Array.from({ length: nDetections }, (_, i) => {
                const lab = labelSet[i % Math.max(1, labelSet.length)] ?? null;
                return ({
                  box: [0, 0, 0, 0] as [number, number, number, number],
                  mask: null,
                  predLabel: lab,
                  rejected: false,
                } as unknown) as ImportDetection;
              })
            : [];
          return {
            id: imp.id,
            backendId: imp.id,
            file: new File([], imp.originalFilename || imp.filename),
            filename: imp.filename,
            preview: `${API}/api/v2/projects/${pid}/imports/${encodeURIComponent(imp.filename)}`,
            blurhash: imp.blurhash ?? null,
            status: "ready" as const,
            width: imp.width,
            height: imp.height,
            createdAt: imp.createdAt ?? null,
            sourceUrl: imp.source?.url ?? null,
            derivedLabel: (imp as { derivedLabel?: string | null }).derivedLabel ?? null,
            nAugmentations: imp.n_augmentations ?? 0,
            labelStats,
            detectionCount: nDetections,
            detections: placeholderDetections,
            // Persisted cachebuster from the backend so the segmented
            // labelled-preview survives a cold reopen (see backend
            // _tile_overview labelledAt). Undefined for never-labelled
            // imports - the URL stays bare, which is correct (the blank
            // preview is what we want there).
            labelledAt: imp.labelledAt ?? undefined,
          };
        });
        if (hydratedImports.length > 0) {
          setImports((cur) => {
            const byId = new Map(hydratedImports.map((h) => [h.id, h]));
            const merged: ImportedMedia[] = [];
            for (const c of cur) {
              // Match by the in-memory id OR the server id stashed in
              // backendId: a drag-dropped / Openverse / video-frame tile
              // keeps its LOCAL id for life (only backendId holds the
              // server id), so keying solely on id dropped persisted
              // uploads (and any edit made just before the merge) once
              // the server-id-keyed map didn't contain the local id.
              const matchKey = byId.has(c.id)
                ? c.id
                : (c.backendId && byId.has(c.backendId) ? c.backendId : null);
              if (matchKey) {
                const fresh = byId.get(matchKey)!;
                merged.push({
                  ...fresh,
                  // Keep the in-memory id + backendId so state keyed on the
                  // local id (in-flight edits, importsById) stays valid.
                  id: c.id,
                  backendId: c.backendId,
                  // Prefer cur ONLY when it has real (non-placeholder)
                  // detection geometry OR has been user-edited.
                  // Otherwise adopt fresh, protects against stale
                  // sidecar placeholders (e.g. /initial returning
                  // n_detections=0 because the file was built before
                  // the labelling pass ran) winning over the fresher
                  // /overview placeholders that have correct counts +
                  // labels.
                  detections: (hasRealDetections(c.detections) || c.editedBoxes !== undefined)
                    ? c.detections
                    : fresh.detections,
                  editedBoxes: c.editedBoxes,
                  timings: c.timings,
                  // Don't clobber a known-good blurhash with null -
                  // /initial / /overview sometimes return null while
                  // the backend's async backfill is still running,
                  // which used to paint white placeholders past the
                  // first batch on every cached reload.
                  blurhash: fresh.blurhash ?? c.blurhash ?? null,
                  // Server confirmed this id this session - no longer a
                  // provisional cache ghost.
                  provisional: undefined,
                });
                byId.delete(matchKey);
              } else if (c.provisional || !c.backendId) {
                // Keep unconfirmed cache tiles (a later chunk / the
                // /overview load may confirm them; the post-load prune
                // drops any that stay provisional) and in-flight uploads.
                // /initial only returns the newest 20, so it must NOT
                // drop tiles outside that window.
                merged.push(c);
              }
              // else: previously-confirmed (non-provisional) tile the
              // server no longer lists → deleted elsewhere → drop.
            }
            for (const h of byId.values()) merged.push(h);
            return merged.sort(compareImportedMediaDesc);
          });
        }
        // Mirror to the project-meta cache so a reopen paints from
        // cache before this fetch resolves on the next visit.
        const cacheOwner = j.owner ?? j.createdBy ?? undefined;
        patchProjectMeta(pid, {
          ...(refList.length > 0 ? {
            refTiles: refList.map((ref) => ({
              id: ref.id,
              filename: ref.filename,
              blurhash: ref.blurhash ?? null,
              width: ref.width,
              height: ref.height,
            })),
            nReferences: refList.length,
          } : {}),
          ...(imps.length > 0 ? {
            importTiles: imps.map((imp) => ({
              id: imp.id,
              filename: imp.filename,
              blurhash: imp.blurhash ?? null,
              width: imp.width,
              height: imp.height,
              createdAt: imp.createdAt ?? null,
            })),
            nImages: total,
          } : {}),
          ...(typeof j.private === "boolean" ? { private: j.private } : {}),
          ...(cacheOwner ? { owner: cacheOwner } : {}),
        });
        // Stats card seed, DatasetStatsCard reads this on mount so
        // the badge + label distribution paint without its own fetch.
        if (j.stats) setSeedStats(j.stats);
        // Gate flips here so the gallery's empty-state copy doesn't
        // flash "Add images" before the first 20 tiles paint.
        setImportsReady(true);
      } catch (e) {
        // AbortError is a normal project-change cancellation; don't
        // log the noisy stack trace for it.
        if (e instanceof DOMException && e.name === "AbortError") return;
        console.warn("[v2 initial] failed:", e);
      }
    })();
    return () => {
      // Project-change mid-hydration: abort the outgoing /initial
      // fetch so its setImports / setRefs land into the new project
      // are dropped, the new project's effect will hydrate from
      // scratch.
      ac.abort();
    };
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear the stats seed when projectId changes so a fresh open
  // doesn't paint stale numbers from the previous project.
  useEffect(() => {
    setSeedStats(null);
  }, [projectId]);

  // P2 store reset on project switch. No-op when the flag is off,
  // so the existing useState-only code path is untouched. Sits in
  // its own effect with [projectId] dep so it fires once per
  // open / switch, before any mirror-writes from imports below.
  useEffect(() => {
    if (!STORE_V2_ENABLED) return;
    ProjectStore.reset(projectId ?? null);
  }, [projectId]);
  // The mirror-writes effect lives below (after the imports useState
  // declaration), look for "P2 mirror-writes".

  // Project-switch reset for the legacy useState arrays. The
  // `imports` / `refs` useState seeds only fire at mount; without
  // this effect, switching A → B → A leaves B's entries piled
  // under A's localStorage cache, which paints as duplicate /
  // wrong-project tiles. Skips the initial mount via
  // projectSwitchInitialRef so we don't double-seed.
  const projectSwitchInitialRef = useRef(true);
  useEffect(() => {
    if (projectSwitchInitialRef.current) {
      projectSwitchInitialRef.current = false;
      return;
    }
    // Re-seed imports + refs from the new project's per-project
    // cache (matches the useState initializer logic). Empty when
    // the project isn't yet in the cache, the hydration effects
    // below fill it in.
    if (!projectId) {
      setImports([]);
      setRefs(references);
      setExpectedRefCount(0);
      setImportsTotal(null);
      setImportsReady(false);
      setFilterCountsOverride(null);
      setManifestUpdatedAt(null);
      return;
    }
    setFilterCountsOverride(null);
    setManifestUpdatedAt(null);
    const cached = readProjectMeta(projectId);
    // Re-seed imports from the new project's importTiles cache so the
    // gallery paints instantly on switch-back. /overview reconciles +
    // drops stale ids when its first slice lands (see useState above).
    setImports(
      cached?.importTiles?.length
        ? cached.importTiles
            .map<ImportedMedia>((t) => ({
              id: t.id,
              backendId: t.id,
              file: new File([], t.filename),
              filename: t.filename,
              preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(t.filename)}`,
              blurhash: t.blurhash ?? null,
              status: "ready" as const,
              width: t.width,
              height: t.height,
              createdAt: t.createdAt ?? null,
              // See the useState seed above - provisional until the
              // server confirms this id; ghosts get pruned post-load.
              provisional: true,
            }))
            .sort(compareImportedMediaDesc)
        : [],
    );
    setRefs(
      references.length > 0
        ? references
        : (cached?.refTiles ?? []).map((t) => ({
            file: new File([], t.filename),
            preview: `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(t.filename)}`,
            width: t.width,
            height: t.height,
            boxes: [],
            referenceId: t.id,
            filename: t.filename,
            blurhash: t.blurhash ?? null,
          })),
    );
    setExpectedRefCount(Math.max(0, cached?.nReferences ?? 0));
    setImportsTotal(typeof cached?.nImages === "number" ? cached.nImages : null);
    setImportsReady(false);
    // Drop the hydration guards for this projectId so the /initial
    // + /overview effects below re-fetch and replace the cached
    // seed with fresh server data. Without this, a revisit would
    // be stuck on whatever localStorage had - including the missing
    // entries that prompted the duplicate report.
    initialHydratedRef.current.delete(projectId);
    hydratedRef.current.delete(projectId);
    refsHydratedRef.current.delete(projectId);
    // references intentionally read from closure at switch time -
    // the dedicated effect above (line 275ish) already replays
    // setRefs(references) when the prop itself changes, so adding
    // it here would re-fire this whole reset on every parent
    // re-render.
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  const refsHydratedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    if (refsHydratedRef.current.has(projectId)) return;
    if (refs.length > 0 && refs[0]?.preview && !refs[0].preview.startsWith(`${API}/`)) {
      // Fresh in-memory refs from onboarding (blob: previews), leave
      // them. Cache-seeded refs (preview is an API URL) DO want to be
      // refreshed from the network, so they don't short-circuit here.
      return;
    }
    refsHydratedRef.current.add(projectId);
    // Capture into a local so TS keeps the non-null narrowing across
    // the async closures below.
    const pid: string = projectId;
    (async () => {
      // Two-phase hydration:
      //   1. /overview pulls render-critical metadata only (tile
      //      blurhashes, dims, n_detections, label_set). Fast, KB
      //      sized regardless of project size.
      //   2. /annotations follows in the background to populate
      //      detections + editedBoxes. Embeddings stay server-side
      //      and don't travel to the FE at all.
      // The legacy /api/projects/{id} fetch is the fallback so older
      // backends keep working through the rollout.
      try {
        // imports_limit=0 hits the backend's sidecar fast-path AND
        // skips the per-import tile reduction we don't need here ,
        // this effect only reads the `references` field. The imports
        // hydration effect below does its own `?imports_limit=100`
        // request for the gallery tiles. Cuts ~400 ms off cold loads.
        const ovr = await apiFetch(
          `/api/v2/projects/${projectId}/overview?imports_limit=0`,
          {},
        );
        if (!ovr.ok) {
          console.warn("[v2 hydrate] overview fetch failed:", ovr.status);
          // Fall back to the legacy single-shot manifest endpoint.
          await hydrateLegacy();
          return;
        }
        const overview = (await ovr.json()) as {
          references?: {
            id: string;
            filename: string;
            originalFilename?: string;
            width?: number;
            height?: number;
            blurhash?: string | null;
            n_detections?: number;
          }[];
          filter_counts?: {
            all: number;
            unlabelled: number;
            unrated: number;
            good: number;
            bad: number;
            unsure: number;
          };
          name?: string;
          derived?: { parentProjectId?: string; parentName?: string } | null;
          updatedAt?: string | null;
        };
        // Seed the title from the load when the page mounted without one (a
        // freshly-created / derived project, before its card sidecar exists, so
        // the projectName prop was empty). `cur || name` never clobbers a rename.
        const _ovrName = overview.name;
        if (_ovrName) setProjectTitle((cur) => cur || _ovrName);
        setDerivedInfo(overview.derived ?? null);
        if (overview.filter_counts) {
          setFilterCountsOverride(overview.filter_counts);
        }
        if (overview.updatedAt !== undefined) {
          setManifestUpdatedAt(overview.updatedAt ?? null);
        }
        const list = overview.references ?? [];
        setExpectedRefCount(list.length);
        if (list.length > 0) {
          // Phase 1: tile metadata. Boxes are empty arrays here ,
          // they fill in once /annotations resolves below.
          const newUploadStatus: Record<string, "uploading" | "done" | "failed"> = {};
          const hydrated: ReferenceImage[] = list.map((ref) => {
            const url = `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(ref.filename)}`;
            newUploadStatus[url] = "done";
            return {
              file: new File([], ref.originalFilename || ref.filename),
              preview: url,
              width: ref.width,
              height: ref.height,
              boxes: [],
              referenceId: ref.id,
              filename: ref.filename,
              blurhash: ref.blurhash ?? null,
            };
          });
          setRefs((cur) => {
            const byId = new Map(hydrated.map((h) => [h.referenceId, h]));
            const merged: ReferenceImage[] = [];
            for (const c of cur) {
              if (c.referenceId && byId.has(c.referenceId)) {
                // Preserve any boxes already on the cache-seeded entry
                // so we don't briefly clear them while annotations
                // are mid-fetch.
                const fresh = byId.get(c.referenceId)!;
                merged.push({ ...fresh, boxes: c.boxes ?? [] });
                byId.delete(c.referenceId);
              } else {
                merged.push(c);
              }
            }
            for (const h of byId.values()) merged.push(h);
            return merged;
          });
          setRefUploadStatus((cur) => ({ ...newUploadStatus, ...cur }));
          patchProjectMeta(projectId, {
            refTiles: list.map((ref) => ({
              id: ref.id,
              filename: ref.filename,
              blurhash: ref.blurhash ?? null,
              width: ref.width,
              height: ref.height,
            })),
            nReferences: list.length,
          });
        }

        // Phase 2: annotations. scope=refs pulls ONLY the references'
        // detection geometry, the imports' real box geometry isn't
        // needed for the gallery (the chip rail + count read from the
        // placeholder detections synthesised from /overview's
        // n_detections + label_set, which already account for
        // editedBoxes) and the viewer fetches per-image real data via
        // /annotations/{import_id} on open. Skipping the imports half
        // shaves ~270 KB off this response on a 9000-detection project.
        const annR = await apiFetch(
          `/api/v2/projects/${projectId}/annotations?scope=refs`,
          {},
        );
        if (annR.ok) {
          const ann = (await annR.json()) as {
            references?: Record<string, {
              detections?: { label: string; score: number; box: number[]; mask: MaskShape | null }[];
            }>;
          };
          if (ann.references) {
            setRefs((cur) => cur.map((r) => {
              if (!r.referenceId) return r;
              const a = ann.references![r.referenceId];
              if (!a) return r;
              const dets = (a.detections ?? []).map((d) => ({
                label: d.label,
                score: d.score,
                box_xyxy: d.box,
                mask: d.mask,
              }));
              return { ...r, boxes: detectionsToBoxes(dets, []) };
            }));
          }
        } else {
          console.warn("[v2 hydrate] annotations fetch failed:", annR.status);
        }
      } catch (e) {
        console.warn("[v2 hydrate] failed:", e);
      }

      // Legacy fallback: backend without the new endpoints.
      async function hydrateLegacy() {
        try {
          const r = await apiFetch(`/api/projects/${projectId}`);
          if (!r.ok) return;
          const m = (await r.json()) as {
          references?: {
            id: string;
            filename: string;
            originalFilename?: string;
            width?: number;
            height?: number;
            blurhash?: string | null;
            detections?: {
              label: string;
              score: number;
              box: number[];
              mask: MaskShape | null;
              embedding?: number[];
              embed_version?: number;
            }[];
          }[];
        };
        const list = m.references ?? [];
        setExpectedRefCount(list.length);
        if (list.length === 0) return;
        const hydrated: ReferenceImage[] = [];
        const allHydratedEmbeds: { key: string; box: [number, number, number, number]; embedding: number[] }[] = [];
        const newUploadStatus: Record<string, "uploading" | "done" | "failed"> = {};
        for (const ref of list) {
          const url = `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(ref.filename)}`;
          const file = new File([], ref.originalFilename || ref.filename);
          const detsRaw = ref.detections ?? [];
          const dets = detsRaw.map((d) => ({
            label: d.label,
            score: d.score,
            box_xyxy: d.box,
            mask: d.mask,
          }));
          const boxes = detectionsToBoxes(dets, []);
          detsRaw.forEach((d, i) => {
            const emb = d.embedding;
            if (!Array.isArray(emb) || emb.length === 0) return;
            const b = boxes[i];
            if (!b) return;
            allHydratedEmbeds.push({
              key: cropKey(url, b),
              box: [d.box[0], d.box[1], d.box[2], d.box[3]] as [number, number, number, number],
              embedding: emb,
            });
          });
          hydrated.push({
            file,
            preview: url,
            width: ref.width,
            height: ref.height,
            boxes,
            referenceId: ref.id,
            filename: ref.filename,
            blurhash: ref.blurhash ?? null,
          });
          newUploadStatus[url] = "done";
        }
        if (allHydratedEmbeds.length > 0) {
          setRefEmbeds((prev) => {
            const next = new Map(prev);
            for (const e of allHydratedEmbeds) {
              if (!next.has(e.key)) next.set(e.key, { box: e.box, embedding: e.embedding });
            }
            return next;
          });
        }
        setRefs((cur) => {
          const byId = new Map(hydrated.map((h) => [h.referenceId, h]));
          const merged: ReferenceImage[] = [];
          for (const c of cur) {
            if (c.referenceId && byId.has(c.referenceId)) {
              merged.push(byId.get(c.referenceId)!);
              byId.delete(c.referenceId);
            } else {
              merged.push(c);
            }
          }
          for (const h of byId.values()) merged.push(h);
          return merged;
        });
        setRefUploadStatus((cur) => ({ ...newUploadStatus, ...cur }));
        patchProjectMeta(pid, {
          refTiles: list.map((ref) => ({
            id: ref.id,
            filename: ref.filename,
            blurhash: ref.blurhash ?? null,
            width: ref.width,
            height: ref.height,
          })),
          nReferences: list.length,
        });
          console.log("[v2 refs hydrate] hydrated", list.length, "reference(s) (legacy path)");
        } catch (e) {
          console.warn("[v2 refs hydrate] legacy failed:", e);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);
  // Debug dump: every ref's boxes by label so we can confirm what the
  // chip counter is actually seeing. Fires whenever refs changes.
  useEffect(() => {
    const summary = refs.map((r, i) => ({
      idx: i,
      preview: r.preview.slice(-12),
      boxCount: r.boxes?.length ?? 0,
      labels: (r.boxes ?? []).map((b) => b.label),
    }));
    // eslint-disable-next-line no-console
    console.log("[v2 view] refs summary:", summary);
  }, [refs]);
  // eslint-disable-next-line no-console
  console.log("[v2 view] render with refs:", refs.length, "items, projectId:", projectId);
  const updateRefs = (next: ReferenceImage[]) => {
    setRefs(next);
    onReferencesChange?.(next);
  };

  // Reference quality, leave-one-out cross-validation on the
  // server's ref embeddings. Surfaces "outlier" / "looks like other
  // class" warnings per ref in the grid so the user can spot bad
  // references that are dragging their centroid in the wrong
  // direction. Refetched whenever refs change AND projectId is
  // known, with a debounce to coalesce rapid edits.
  const [refQuality, setRefQuality] = useState<RefQuality>({});
  useEffect(() => {
    if (!projectId) return;
    const refIds = refs.map((r) => r.referenceId).filter(Boolean);
    if (refIds.length === 0) {
      setRefQuality({});
      return;
    }
    const t = window.setTimeout(async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/reference_quality`);
        if (!r.ok) return;
        const data = (await r.json()) as { references?: RefQuality };
        setRefQuality(data.references ?? {});
      } catch (e) {
        console.warn("[v2 ref-quality] fetch failed:", e);
      }
    }, 800);
    return () => window.clearTimeout(t);
  }, [projectId, refs]);

  // ─── Reference embeddings ─────────────────────────────────────────
  // Reference embeddings live in two places:
  //
  //   - The manifest: persisted by the backend at upload time
  //     (v2_upload_reference) and updated by PUT /references/{id}
  //     when the user navigates away from the editor with edits.
  //     These survive across sessions.
  //   - This in-memory `refEmbeds` Map: keyed by cropKey, used to
  //     compute per-label centroids client-side for the imports
  //     overlay. Hydrated from the manifest on mount, then patched
  //     by `flushReferenceEmbeddings` whenever the user finishes
  //     editing a reference.
  //
  // We deliberately DO NOT run an eager "embed-on-every-refs-change"
  // effect any more, it fired during interactive box dragging and
  // hammered the GPU on each pixel-level edit. The flush model keeps
  // the manifest consistent without that overhead: embeddings are
  // computed exactly once per box, when the user is done with it.
  const [refEmbeds, setRefEmbeds] = useState<Map<string, { box: [number, number, number, number]; embedding: number[] }>>(new Map());
  const refEmbedsRef = useRef(refEmbeds);
  useEffect(() => { refEmbedsRef.current = refEmbeds; }, [refEmbeds]);

  // Per-ref flush guard: only one in-flight flush per ref at a time,
  // and we coalesce rapid prev/next clicks so we don't queue up a
  // stack of redundant PUTs.
  const flushInFlightRef = useRef<Set<string>>(new Set());

  const flushReferenceEmbeddings = useMemo(
    () => async (ref: ReferenceImage) => {
      if (!projectId || !ref.referenceId) return;
      const flightKey = `${projectId}::${ref.referenceId}`;
      if (flushInFlightRef.current.has(flightKey)) return;
      flushInFlightRef.current.add(flightKey);
      try {
        const boxes = ref.boxes ?? [];
        // Step 1: identify boxes whose embedding we don't have
        // client-side yet, and POST them to /embed_crops to fill
        // the cache. Skips boxes that are already cached so this
        // is a no-op for refs the user only viewed.
        const missing = boxes
          .map((b, i) => ({ b, i, key: cropKey(ref.preview, b) }))
          .filter(({ key }) => !refEmbedsRef.current.has(key));
        if (missing.length > 0) {
          // Lazy-fetch the original bytes the first time we need to
          // ask the backend to embed crops for this ref. Hydrated
          // refs start with an empty placeholder so the grid pops
          // in fast, only flushes that actually need pixels pay the
          // network round-trip.
          const file = await ensureRefFile(ref);
          if (file && file.size > 0) {
          try {
            const fd = new FormData();
            fd.append("image", file);
            fd.append(
              "boxes",
              JSON.stringify(missing.map(({ b }) => [b.x0, b.y0, b.x1, b.y1])),
            );
            const r = await fetch(`${API}/api/v2/references/embed_crops`, {
              method: "POST",
              body: fd,
            });
            if (r.ok) {
              const data = (await r.json()) as {
                crops: { index?: number; box: number[]; embedding: number[] }[];
              };
              const arr = data.crops ?? [];
              setRefEmbeds((prev) => {
                const next = new Map(prev);
                arr.forEach((c, idx) => {
                  const i = typeof c.index === "number" ? c.index : idx;
                  if (i < 0 || i >= missing.length) return;
                  next.set(missing[i].key, {
                    box: [c.box[0], c.box[1], c.box[2], c.box[3]],
                    embedding: c.embedding,
                  });
                });
                return next;
              });
            }
          } catch (e) {
            console.warn("[v2 flush] embed_crops failed:", e);
          }
          }
        }

        // Step 2: PUT the full detection list (with embeddings
        // pulled from the now-up-to-date refEmbeds map) to the
        // manifest so the backend's persisted copy matches what
        // the user sees. The PUT endpoint also re-embeds any
        // boxes we couldn't embed client-side (no file bytes,
        // network blip, etc) so the manifest never drifts.
        const detections = boxes.map((b) => {
          const stored = refEmbedsRef.current.get(cropKey(ref.preview, b));
          const det: Record<string, unknown> = {
            label: b.label,
            score: b.score,
            box: [b.x0, b.y0, b.x1, b.y1],
            mask: b.mask ?? null,
          };
          if (stored?.embedding && stored.embedding.length > 0) {
            det.embedding = stored.embedding;
          }
          return det;
        });
        try {
          await apiFetch(`/api/v2/projects/${projectId}/references/${ref.referenceId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ detections }),
          });
        } catch (e) {
          console.warn("[v2 flush] PUT failed:", e);
        }
      } finally {
        flushInFlightRef.current.delete(flightKey);
      }
    },
    [projectId],
  );

  // Diff-based label/box catch-up. The previous catch-up keyed on
  // the FE-side embedding cache (always empty on mount) so it
  // PUT-stormed every project open. Now we snapshot each ref's
  // current box JSON ON FIRST SIGHT (treated as "already on the
  // server") and only PUT when the boxes diverge from that
  // baseline, typically a label rename in the BoxEditor that
  // hasn't been flushed via onLeaveImage yet. Stops the bug where
  // user-relabelled refs sat with stale labels on the backend,
  // making the resolver only see one class.
  const lastFlushedRefBoxesRef = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!projectId) return;
    // Initial-sight snapshot, record the current state as already-
    // pushed so we don't flush every ref on mount. Diff-flush below
    // only fires once a ref's boxes actually change vs this baseline.
    let snappedAny = false;
    for (const r of refs) {
      if (!r.referenceId) continue;
      if (!lastFlushedRefBoxesRef.current.has(r.referenceId)) {
        lastFlushedRefBoxesRef.current.set(
          r.referenceId, JSON.stringify(r.boxes ?? []),
        );
        snappedAny = true;
      }
    }
    if (snappedAny) return;
    const id = window.setTimeout(() => {
      for (const r of refs) {
        if (!r.referenceId) continue;
        const serialised = JSON.stringify(r.boxes ?? []);
        const last = lastFlushedRefBoxesRef.current.get(r.referenceId);
        if (last === serialised) continue;
        lastFlushedRefBoxesRef.current.set(r.referenceId, serialised);
        flushReferenceEmbeddings(r);
      }
    }, 600);
    return () => window.clearTimeout(id);
  }, [refs, projectId, flushReferenceEmbeddings]);

  // Per-label centroid: collect every embedding labelled L, average
  // component-wise, then re-L2-normalise so cosine == dot product
  // still holds when matching. One entry per label, not per box.
  // More robust to outliers than picking the single nearest neighbour
  //, a noisy reference box can no longer hijack the label assignment
  // because its contribution is averaged with the rest of its class.
  const refEmbeddings = useMemo(() => {
    // Group by case-insensitive key so "Dog" and "dog" merge.
    const sums = new Map<string, { display: string; sum: number[]; count: number }>();
    refs.forEach((ref) => {
      (ref.boxes ?? []).forEach((b) => {
        if (!b.label) return;
        const e = refEmbeds.get(cropKey(ref.preview, b));
        if (!e) return;
        const key = b.label.trim().toLowerCase();
        let slot = sums.get(key);
        if (!slot) {
          slot = { display: b.label, sum: new Array(e.embedding.length).fill(0), count: 0 };
          sums.set(key, slot);
        }
        for (let i = 0; i < e.embedding.length; i++) slot.sum[i] += e.embedding[i];
        slot.count += 1;
      });
    });
    const out: { label: string; embedding: number[]; sourceCount: number }[] = [];
    for (const slot of sums.values()) {
      if (slot.count === 0) continue;
      // Mean.
      const mean = slot.sum.map((s) => s / slot.count);
      // Re-normalise so the centroid is unit length and cosine
      // similarity stays equivalent to dot product downstream.
      let norm = 0;
      for (const v of mean) norm += v * v;
      norm = Math.sqrt(norm);
      if (norm < 1e-8) continue;
      for (let i = 0; i < mean.length; i++) mean[i] /= norm;
      out.push({ label: slot.display, embedding: mean, sourceCount: slot.count });
    }
    // eslint-disable-next-line no-console
    console.log("[v2 ref-embed] centroids:", out.map((c) => `${c.label}(n=${c.sourceCount})`));
    return out;
  }, [refs, refEmbeds]);
  const refEmbeddingsRef = useRef(refEmbeddings);
  useEffect(() => { refEmbeddingsRef.current = refEmbeddings; }, [refEmbeddings]);

  // ─── Labelling job (deferred labelling) ───────────────────────────
  // Imports now upload to /imports/raw with no per-image inference.
  // The user clicks Start → a `label_charlie` job runs server-side
  // and processes every unlabelled import in one pass. The card
  // animates in, polls /api/projects/{id}/jobs/{jobId} for status,
  // and refetches the manifest on completion so detections appear
  // in the gallery.
  const [labelJob, setLabelJob] = useState<LabelJobState | null>(null);
  const [labelJobStarting, setLabelJobStarting] = useState(false);
  // Plan / usage snapshot. Drives the credit cutoff so AI labelling
  // refuses once the user's credits are exhausted, instead of
  // silently consuming more.
  const planUsage = usePlan();
  const overCreditLimit = !!planUsage?.over.anyLabelLimit;
  // Label-purge job state (background strip launched from the
  // DeleteLabelModal). Declared alongside labelJob so the polling
  // effect a thousand lines down can see it. labelPendingDelete is
  // hoisted into the same place below at the chip-rail removeLabel
  // handler.
  const [purgeJob, setPurgeJob] = useState<LabelJobState | null>(null);
  const [labelPendingDelete, setLabelPendingDelete] = useState<{
    canonical: string;
    display: string;
  } | null>(null);

  // Settings popup, fires from the gear button in the top bar.
  // Owns rename, visibility, cover, label colours, delete. Rename
  // and label-colour saves feed setProjectTitle / setLabelColours
  // (declared further down) so the bar + chips repaint live.
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Overview health card opens the full dataset-stats + embeddings modal.
  const [healthOpen, setHealthOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  // ─── Imports (Drop media) ─────────────────────────────────────────
  // Each dropped image gets shipped to /api/v2/projects/{id}/imports/raw
  // which persists the bytes WITHOUT running detection. The user
  // labels everything in one batch via the labelling job above.
  //
  // Seed from the localStorage importTiles cache so the dataset gallery
  // paints BlurHash-gradient placeholder tiles from the previous open's
  // cache BEFORE the /overview round-trip returns - re-opening a project
  // feels instant instead of a cold reload. /overview is still the
  // source of truth: when its first slice lands, the merge below
  // (match-by-id, drop any local entry whose backendId is absent from
  // the fresh list) reconciles and removes stale ids. The old
  // "come back, see duplicates" symptom this seeding once caused was
  // root-caused to upload-retry creating duplicate backend records - now
  // fixed by the idempotency_key on /imports/raw - so re-seeding is safe.
  const [imports, setImports] = useState<ImportedMedia[]>(() => {
    if (!projectId) return [];
    const cached = readProjectMeta(projectId);
    if (!cached?.importTiles?.length) return [];
    return cached.importTiles
      .map<ImportedMedia>((t) => ({
        id: t.id,
        backendId: t.id,
        file: new File([], t.filename),
        filename: t.filename,
        preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(t.filename)}`,
        blurhash: t.blurhash ?? null,
        status: "ready" as const,
        width: t.width,
        height: t.height,
        createdAt: t.createdAt ?? null,
        // Seeded from cache - unconfirmed until /initial or /overview
        // returns this id. Ghosts (deleted server-side) stay provisional
        // and get pruned after the full /overview load.
        provisional: true,
      }))
      .sort(compareImportedMediaDesc);
  });
  const importsLatestRef = useRef<ImportedMedia[]>(imports);
  useEffect(() => { importsLatestRef.current = imports; }, [imports]);

  // Bulk-import progress for the batched uploader. Non-null while a
  // drag-drop import is in flight so the dataset area can paint a
  // progress bar (done / total) as each batch lands. Cleared when the
  // whole drop finishes.
  const [importProgress, setImportProgress] = useState<{ total: number; done: number } | null>(null);

  // Upload size ceiling (px longest edge) inherited from the dataset's Project,
  // default 1500 = the historical client default. A ref so the upload path
  // reads the latest value without re-rendering; populated from manifest fetches.
  const maxInputSizeRef = useRef<number>(1500);
  // Reactive mirror of the ceiling so the hero can show the image size limit
  // chip (the ref alone can't drive a re-render). Kept in step wherever the ref
  // is set from a manifest fetch.
  const [maxInputSize, setMaxInputSize] = useState<number>(1500);
  // Cover hero banner at the top of the dataset overview (the same cover the
  // workspace card shows); falls back to the brand wash on a load error.
  // bannerLight is sampled from the cover's left region (where the info sits)
  // so the title/meta text + scrim adapt to the image (content-aware), not the
  // theme. CORS on the cover endpoint lets the canvas read pixels.
  const [bannerCoverFailed, setBannerCoverFailed] = useState(false);
  const [bannerLight, setBannerLight] = useState(false);
  // Cache-buster for the hero cover. cover_thumb keeps the same URL across cover
  // swaps/uploads, so we bump this on a cover change to force a re-fetch.
  const [coverBust, setCoverBust] = useState(0);
  // Retry counter for the hero cover. cover_thumb lazy-renders, so the first GET
  // (especially the 1280 hero variant) can transiently fail while it bakes;
  // without a retry the banner stayed blank until a remount — the "go on and off
  // the project to make it load" bug. Reset on cover change so a swap re-tries.
  const [bannerRetry, setBannerRetry] = useState(0);
  // The hero starts on the fast Lanczos render and swaps to the GPU AI-upscaled
  // variant once it's baked + loaded, so a warm-up render never leaves the
  // banner blank. Reset on cover change so a swap re-runs from scratch.
  const [aiCoverReady, setAiCoverReady] = useState(false);
  useEffect(() => { setBannerCoverFailed(false); setBannerRetry(0); setAiCoverReady(false); }, [coverBust]);

  // Smooth-scrolls the page to the dataset gallery the moment new
  // images land. Fires after both the drag-drop import path and the
  // Openverse pull so the user sees the gallery filling in even when
  // their viewport is parked at the top of the page.
  const datasetSectionRef = useRef<HTMLElement | null>(null);
  const scrollToDataset = () => {
    // rAF so the gallery has at least one paint to size itself with
    // the new placeholders, otherwise the scroll lands a few hundred
    // pixels short on slow renders.
    if (typeof window === "undefined") return;
    requestAnimationFrame(() => {
      datasetSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  // Imports the auto-PUT effect needs to look at on its next tick.
  // Populated by updateImport so the 600 ms-debounced effect doesn't
  // have to iterate 1000 entries every time `imports` changes (which
  // happens on every label-job progress tick, every n_augmentations
  // bump, every labelledAt refresh - none of which are user edits).
  // Drained inside the effect after each pass.
  const dirtyImportIdsRef = useRef<Set<string>>(new Set());

  const updateImport = useCallback((
    id: string,
    patch: Partial<ImportedMedia>,
    // hydration: the patch carries SERVER data (annotation fetch /
    // hover-prefetch / viewport batch), not a user edit. Two
    // consequences: it must not mark the import dirty (it isn't a
    // pending user change to PUT back), and it must never overwrite
    // a local editedBoxes copy that already exists - the response
    // may land mid-gesture (placeholder box just added, drag in
    // progress) and stomping it silently killed click-to-detect and
    // box drawing. Checked inside the updater so the decision is
    // made against LIVE state, not the row captured when the fetch
    // was kicked off.
    opts?: { hydration?: boolean },
  ) => {
    // Track edits to editedBoxes for the auto-PUT scoping. Patches
    // that don't touch editedBoxes (labelledAt, nAugmentations,
    // detections-only) don't need to fire a PUT, so they're skipped
    // here.
    if (!opts?.hydration && Object.prototype.hasOwnProperty.call(patch, "editedBoxes")) {
      dirtyImportIdsRef.current.add(id);
    }
    setImports((cur) => cur.map((m) => {
      if (m.id !== id) return m;
      if (
        opts?.hydration &&
        Object.prototype.hasOwnProperty.call(patch, "editedBoxes") &&
        m.editedBoxes !== undefined
      ) {
        // FE already holds an editedBoxes copy this session - it is
        // authoritative (the auto-PUT pushes it up). Take the rest of
        // the hydration payload (detections / masks / timings) only.
        const rest = { ...patch };
        delete rest.editedBoxes;
        return { ...m, ...rest };
      }
      return { ...m, ...patch };
    }));
    // Stable identity: updateImport only touches the setImports setter and
    // dirtyImportIdsRef (both stable). Memoising it is critical — it's passed
    // as `onMediaChange` into DatasetViewer, whose annotation-hydration effects
    // both depend on AND call it. An unstable ref made those effects re-run →
    // setImports → re-render → new ref → re-run, an infinite render loop that
    // only manifested while the viewer modal was open (image clicked).
  }, []);

  // P2 mirror-writes. Diffs the current ImportedMedia[] against the
  // store and pushes upserts / removes. Runs after every setImports
  // (since the dep is the imports state itself). One central mirror
  // is safer than chasing all 22 setImports call sites - any future
  // call site automatically participates.
  //
  // Conversion: only the narrow StoreImport surface goes into the
  // store. File handles, mask geometry, transient flags stay in
  // the legacy useState path. Selector-subscribed components (tile
  // chips) read the slim record; anything needing the full
  // ImportedMedia (viewer, BoxEditor) keeps reading via prop chain
  // until follow-up phases (P3 / P4) bring them onto the store too.
  useEffect(() => {
    if (!STORE_V2_ENABLED) return;
    const slim = imports.map(toStoreImport);
    ProjectStore.bulkUpsert(slim);
    // Drop any ids in the store that have disappeared from local
    // state (post-delete, post-batch-purge). Cheap diff via Set.
    const liveIds = new Set(slim.map((s) => s.id));
    const inStore = ProjectStore.state.orderedIds;
    for (const id of inStore) {
      if (!liveIds.has(id)) ProjectStore.removeImport(id);
    }
  }, [imports]);

  // P4: off-thread labelStats aggregate. bulkUpsert above already
  // maintains the value via per-record deltas (optimistic, runs
  // synchronously on every imports change). This effect debounces
  // a from-scratch authoritative recompute in the worker so a
  // labelling-job poll firing setImports at 5 Hz on a 900-image
  // project doesn't burn the main thread on aggregate maths every
  // tick. The worker's result replaces the delta value when it
  // lands; setLabelStatsAggregate is a no-op when they agree.
  useEffect(() => {
    if (!STORE_V2_ENABLED) return;
    if (!ANNOT_WORKER_ENABLED) return;
    const tid = window.setTimeout(() => {
      const slim = imports.map((m) => ({
        id: m.id,
        labelStats: m.labelStats,
      }));
      aggregateLabelStatsInWorker(slim)
        .then((stats) => ProjectStore.setLabelStatsAggregate(stats))
        .catch((e) => {
          console.warn("[label-stats] worker aggregate failed:", e);
        });
    }, 300);
    return () => window.clearTimeout(tid);
  }, [imports]);

  // Reference-upload progress tracking. The handoff from onboarding
  // is now non-blocking: HomeView returns from v2HandOff immediately
  // after creating the project, which means the V2 page mounts with
  // the full ReferenceImage list still in memory but NOT yet on
  // disk. We POST each reference to /api/v2/projects/{id}/references
  // here in the background so the user gets the project page right
  // away; a banner reports progress; and any imports / job-style
  // actions are gated until every reference settles.
  //
  // States:
  //   "uploading", POST in flight
  //   "done"     , server acknowledged 200
  //   "failed"   , exception or non-2xx response
  // No "pending" intermediate, splitting "mark pending" and "start
  // upload" into separate effects ran the upload effect with a
  // stale ref-snapshot of the status map and pending always read
  // empty. Now both happen in one pass.
  type RefUploadStatus = "uploading" | "done" | "failed";
  const [refUploadStatus, setRefUploadStatus] = useState<Record<string, RefUploadStatus>>({});

  const refUploadStatusRef = useRef(refUploadStatus);
  useEffect(() => { refUploadStatusRef.current = refUploadStatus; });
  // Module-level "in-flight" registry survives React 18 dev-mode
  // StrictMode's double-effect, useRef gets the SAME ref across
  // the simulated unmount/remount, so a per-mount sentinel ends up
  // stuck = true after cleanup and every later POST resolution
  // skips its setState. A registry keyed by `${projectId}::${preview}`
  // is checked + reserved synchronously when a POST is launched so
  // the second simulated effect run sees the URL already claimed
  // and doesn't fire a duplicate request.
  const refUploadInFlightRef = useRef<Set<string>>(new Set());
  // Serialise reference uploads through a single promise chain so
  // they fire in array order, top-left first, top-right next, etc.
  // Previously each ref's POST was kicked off via a bare `(async ...)()`
  // IIFE so they raced; the backend's per-ref logs landed in random
  // order and the user couldn't predict when any given reference
  // would finish. The chain pins ordering without changing the per-
  // call latency since the network bottleneck is the same either way.
  const refUploadChainRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    if (!projectId) {
      console.log("[v2 ref-upload] skip, no projectId yet");
      return;
    }
    // Anything not in the status map yet is brand new, claim it
    // by writing "uploading" and kick off the POST. Skip refs that
    // already have a backend `referenceId`: those came from the
    // single-shot onboarding upload path (HomeView.v2TriggerPipeline)
    // which POSTs directly to /api/v2/projects/{id}/references, so
    // re-POSTing here would create a duplicate manifest entry.
    const newRefs = refs.filter(
      (r) => !(r.preview in refUploadStatusRef.current) && !r.referenceId,
    );
    // Pre-mark already-persisted refs as "done" so the upload
    // banner doesn't count them as pending.
    setRefUploadStatus((cur) => {
      const next = { ...cur };
      let changed = false;
      for (const r of refs) {
        if (r.referenceId && next[r.preview] !== "done") {
          next[r.preview] = "done";
          changed = true;
        }
      }
      return changed ? next : cur;
    });
    if (newRefs.length === 0) {
      console.log(
        "[v2 ref-upload] nothing new ,",
        Object.entries(refUploadStatusRef.current).map(([k, v]) => `${k.slice(-8)}:${v}`).join(", ") || "(empty)",
      );
      return;
    }
    console.log("[v2 ref-upload] starting upload of", newRefs.length, "reference(s) to project", projectId);
    setRefUploadStatus((cur) => {
      const next = { ...cur };
      for (const r of newRefs) next[r.preview] = "uploading";
      return next;
    });
    for (const ref of newRefs) {
      const flightKey = `${projectId}::${ref.preview}`;
      if (refUploadInFlightRef.current.has(flightKey)) {
        // Second StrictMode simulated effect run, already claimed.
        continue;
      }
      refUploadInFlightRef.current.add(flightKey);
      const refSnapshot = ref;
      refUploadChainRef.current = refUploadChainRef.current.then(async () => {
        try {
          const fd = new FormData();
          fd.append("image", refSnapshot.file);
          fd.append(
            "detections",
            JSON.stringify(
              (refSnapshot.boxes ?? []).map((b) => ({
                label: b.label,
                score: b.score,
                box: [b.x0, b.y0, b.x1, b.y1],
                mask: b.mask ?? null,
              })),
            ),
          );
          // Section label: persisted on the manifest entry and used to
          // force-label the detections (or synthesise a whole-image box
          // when the per-image pipeline found nothing). `labels` lets
          // the backend run inline detection when the client sent no
          // precomputed boxes.
          const refSectionLabel = refSnapshot.label?.trim();
          if (refSectionLabel) {
            fd.append("label", refSectionLabel);
            fd.append("labels", JSON.stringify([refSectionLabel]));
          }
          if (refSnapshot.width) fd.append("width", String(refSnapshot.width));
          if (refSnapshot.height) fd.append("height", String(refSnapshot.height));
          const r = await apiFetch(`/api/v2/projects/${projectId}/references`, {
            method: "POST",
            body: fd,
          });
          setRefUploadStatus((cur) => ({ ...cur, [refSnapshot.preview]: r.ok ? "done" : "failed" }));
          if (r.ok) {
            try {
              const data = await r.json() as { reference_id?: string; filename?: string };
              if (data?.reference_id) {
                setRefs((cur) =>
                  cur.map((it) =>
                    it.preview === refSnapshot.preview
                      ? { ...it, referenceId: data.reference_id, filename: data.filename ?? it.filename }
                      : it,
                  ),
                );
              }
            } catch { /* response body parse failure, non-fatal */ }
            console.log("[v2 ref-upload] saved", refSnapshot.preview.slice(-12));
          } else {
            const body = await r.text().catch(() => "");
            console.error("[v2 ref-upload]", `http ${r.status}, ${body}`);
          }
        } catch (e) {
          console.error("[v2 ref-upload] failed:", e);
          setRefUploadStatus((cur) => ({ ...cur, [refSnapshot.preview]: "failed" }));
        } finally {
          refUploadInFlightRef.current.delete(flightKey);
        }
      });
    }
  }, [projectId, refs]);

  const refUploadCounts = useMemo(() => {
    let uploading = 0, done = 0, failed = 0, unstarted = 0;
    for (const r of refs) {
      const s = refUploadStatus[r.preview];
      if (s === "uploading") uploading++;
      else if (s === "done") done++;
      else if (s === "failed") failed++;
      // Refs that already round-tripped to the backend (got a
      // referenceId from either the single-shot onboarding upload
      // or the manifest hydration) are implicitly done, count
      // them that way instead of as "unstarted", otherwise they
      // sit briefly in the unstarted bucket between the setRefs
      // call and the matching setRefUploadStatus and flash the
      // amber "Saving references" banner on every project open.
      else if (r.referenceId) done++;
      else unstarted++; // genuinely new, awaiting upload effect
    }
    return { uploading, done, failed, unstarted, pending: unstarted, total: refs.length };
  }, [refs, refUploadStatus]);
  // "References settled", every reference has either landed on the
  // backend or definitively failed. Used to gate import processing.
  const referencesSettled =
    refUploadCounts.total === 0 ||
    refUploadCounts.uploading + refUploadCounts.unstarted === 0;

  // Single-file-at-a-time import queue. Each accepted media gets
  // Video drop queue. handleImportFiles splits incoming files into
  // images (run straight through the upload pipeline) and videos
  // (queued here so VideoFrameModal can pop and let the user choose
  // a trim range + sample rate before client-side frame extraction).
  // Only the head of the queue is visible at any time; the modal
  // advances to the next entry on confirm / cancel.
  const [videoQueue, setVideoQueue] = useState<File[]>([]);
  const [videoExtracting, setVideoExtracting] = useState<{ done: number; total: number } | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  const handleImportFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (!referencesSettled) {
      // Defensive double-guard, the drop zone is also disabled
      // visually while uploads are in flight. This catches any
      // programmatic invocation that sneaks past the UI gate.
      console.log("[v2 import] reference uploads in progress, ignoring drop");
      return;
    }
    // Credit cutoff. Uploads themselves consume credits at
    // 1/800 each, so once the user's monthly allowance is gone
    // we stop here rather than burn through what's left. Same
    // toast hook as labelling so the user gets a clear reason.
    if (overCreditLimit) {
      setVideoError("Out of credits this period. Upgrade to keep uploading.");
      window.setTimeout(() => setVideoError(null), 5000);
      return;
    }
    const dropped = Array.from(files);
    // Split by MIME type up front WITHOUT reading any bytes. The old
    // path read EVERY dropped file's ArrayBuffer here (to dodge a
    // Safari Blob-detach bug); on a thousands-of-images drop that spiked
    // memory into the gigabytes and blocked the first placeholder paint
    // for several seconds, which is exactly what froze the page ("not
    // responding"). File objects are cheap disk-backed handles until
    // read, so we keep the originals and read each one lazily, just
    // before its upload, inside the bounded workers below.
    const droppedImages = dropped.filter(isImageFile);
    const droppedVideos = dropped.filter((f) => f.type.startsWith("video/"));
    const skippedCount = dropped.length - droppedImages.length - droppedVideos.length;
    if (skippedCount > 0) {
      console.warn(`[v2 import] skipped ${skippedCount} non-image/video file(s)`);
    }

    // Safari ONLY: it invalidates a drag-event Blob's backing once the
    // drop microtask completes (WebKitBlobResource error 4), so there we
    // still capture bytes into fresh ArrayBuffer-backed Files right here.
    // Every other browser keeps the original handles (no bytes resident)
    // so the drop stays instant no matter how many files land.
    const isSafari =
      typeof navigator !== "undefined" &&
      /^((?!chrome|android|crios|fxios|edg).)*safari/i.test(navigator.userAgent);
    const candidates: File[] = isSafari
      ? await Promise.all(droppedImages.map((f) => rereadFileBytes(f)))
      : droppedImages;
    const videos: File[] = isSafari
      ? await Promise.all(droppedVideos.map((f) => rereadFileBytes(f)))
      : droppedVideos;
    // Surface oversize videos as a transient error chip instead of
    // letting the browser OOM mid-decode. Cap value lives in
    // lib/videoFrames so both V1 + V2 share it.
    const tooBig = videos.filter((v) => v.size > MAX_VIDEO_BYTES);
    const okVideos = videos.filter((v) => v.size <= MAX_VIDEO_BYTES);
    if (tooBig.length > 0) {
      const limitMb = Math.round(MAX_VIDEO_BYTES / (1024 * 1024));
      const names = tooBig
        .map((v) => `${v.name} (${(v.size / (1024 * 1024)).toFixed(1)} MB)`)
        .join(", ");
      setVideoError(`Video too large, ${limitMb} MB limit: ${names}`);
      window.setTimeout(() => setVideoError(null), 6000);
    }
    if (okVideos.length > 0) {
      setVideoQueue((prev) => [...prev, ...okVideos]);
    }
    if (candidates.length === 0) return;

    // Phase 1: spawn placeholder tiles IMMEDIATELY, one per
    // candidate file. No await, no resize, no preview yet. The user
    // sees the count instantly + a shimmer grid that fills in as
    // each file's downsized preview arrives. Without this, dropping
    // ~50 phone-camera images blocked the UI for several seconds
    // while resizeForUpload chewed through them.
    // Pre-assign timestamps in REVERSE drop order so under the
    // gallery's DESC sort, the FIRST-dropped tile lands at the top-
    // left of this batch and the WHOLE batch sits above any
    // existing tiles (every value > Date.now() at the moment of
    // drop, so they all out-rank older imports). The upload chain
    // still processes in drop order, so the user sees previews
    // fill in top-left → bottom-right within the batch.
    const baseTs = Date.now();
    const placeholders: ImportedMedia[] = candidates.map((f, i) => ({
      id: `imp_${baseTs}_${i}_${Math.random().toString(36).slice(2, 6)}`,
      file: f,
      preview: "",
      status: "processing",
      createdAt: baseTs + (candidates.length - i),
      // Stable per-upload key so the BE can dedupe a retry-after-
      // success (network blip between request-completion and
      // response-reception) instead of writing a second record.
      idempotencyKey: typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `idem_${baseTs}_${i}_${Math.random().toString(36).slice(2, 10)}`,
    }));
    setImports((cur) => [...placeholders, ...cur].sort(compareImportedMediaDesc));
    // Bump the dataset total optimistically so the "Dataset N" counter
    // updates live as images land, not just after a page refresh. Without
    // this, a fresh project's importsTotal stays 0 and `totalImports ??
    // items.length` resolves to 0 (0 is non-nullish). NSFW/failed uploads
    // that get pulled out via removeImport decrement it back. Left as-is
    // when the total is unknown (null) - the display falls back to
    // items.length there, which is already correct.
    setImportsTotal((n) => (n != null ? n + placeholders.length : n));
    scrollToDataset();

    // Phase 2: resize (capped at 6 concurrent canvas.toBlob calls so
    // they don't serialise internally and stall), then ship images in
    // BATCHES to /imports/raw_batch. One manifest write per batch
    // instead of one per image is the dominant win when importing
    // thousands of images (the old per-image path was O(n^2) on the
    // manifest write + n HTTP round-trips). Gallery order comes from
    // the pre-assigned createdAt timestamps, not upload arrival order.
    const RESIZE_CONCURRENCY = 6;
    let _slots = RESIZE_CONCURRENCY;
    const _waiters: Array<() => void> = [];
    const _acquire = () =>
      new Promise<void>((res) => {
        if (_slots > 0) { _slots--; res(); }
        else { _waiters.push(res); }
      });
    const _release = () => {
      const next = _waiters.shift();
      if (next) { next(); } else { _slots++; }
    };
    const resizeOne = async (i: number): Promise<File> => {
      await _acquire();
      try {
        // No per-tile state write here. The placeholder stays a white
        // tile until its upload confirms, then loads the server
        // thumbnail via backendId (DatasetThumb prefers the server URL
        // whenever backendId is set). Writing a blob preview per image
        // was an O(n^2) setImports storm plus thousands of object URLs
        // held in memory on a big drop, a major freeze source.
        return await resizeForUpload(
          candidates[i],
          maxInputSizeRef.current,
          uploadBytesForMaxSize(maxInputSizeRef.current),
        ).catch(() => candidates[i]);
      } finally {
        _release();
      }
    };

    if (!projectId) {
      // Onboarding: no project yet. Resize + mark ready locally; the
      // hand-off effect re-uploads everything (via the per-image path)
      // once the project is actually created. Kept on the single path
      // since these batches are tiny (a few reference-stage images).
      candidates.forEach((_f, i) => {
        const placeholder = placeholders[i];
        void (async () => {
          try {
            const resized = await resizeOne(i);
            // Onboarding tiles have no backendId yet (project not created),
            // so they render from this local blob preview until the
            // hand-off effect re-uploads them. Cheap here: these batches
            // are tiny (a few reference-stage images), unlike a bulk drop.
            updateImport(placeholder.id, { file: resized, preview: URL.createObjectURL(resized) });
            await processImport({ ...placeholder, file: resized, preview: "" });
          } catch (e) {
            console.error("[v2 import-queue] swallowed error:", e);
          }
        })();
      });
      return;
    }

    // Project exists: batch upload with a progress bar. Runs in the
    // background (no await) so the drop handler returns immediately and
    // the gallery stays interactive while images stream in.
    void (async () => {
      const BATCH_SIZE = 40;
      const BATCH_CONCURRENCY = 3;
      const total = candidates.length;
      let doneCount = 0;
      const bumpDone = (n: number) => {
        doneCount += n;
        setImportProgress((p) => (p ? { ...p, done: doneCount } : p));
      };
      setImportProgress({ total, done: 0 });

      // Fallback for a batch that errored or hit a backend without the
      // raw_batch route (e.g. FE deployed ahead of the server): upload
      // each item via the per-image endpoint so the import never stalls.
      const fallbackPerImage = async (items: { ph: ImportedMedia; file: File }[]) => {
        for (const { ph, file } of items) {
          try { await processImport({ ...ph, file, preview: "" }); }
          catch (e) { console.error("[v2 import-batch] per-image fallback failed:", e); }
        }
      };

      const uploadBatch = async (idxs: number[]) => {
        const items = await Promise.all(
          idxs.map(async (i) => ({ ph: placeholders[i], file: await resizeOne(i) })),
        );
        const fd = new FormData();
        for (const { ph, file } of items) {
          const fresh = await rereadFileBytes(file);
          fd.append("image", fresh);
          fd.append("created_at_ms", typeof ph.createdAt === "number" ? String(ph.createdAt) : "");
          fd.append("idempotency_key", ph.idempotencyKey ?? "");
        }
        let r: Response | null = null;
        try {
          r = await apiFetch(
            `/api/v2/projects/${projectId}/imports/raw_batch`,
            { method: "POST", body: fd },
          );
        } catch (e) {
          console.warn("[v2 import-batch] request threw, falling back to per-image:", e);
        }
        if (!r || !r.ok) {
          await fallbackPerImage(items);
          bumpDone(items.length);
          return;
        }
        const data = (await r.json().catch(() => ({}))) as {
          results?: {
            status?: string;
            import_id?: string;
            filename?: string;
            width?: number;
            height?: number;
            blurhash?: string | null;
            error?: string;
          }[];
        };
        const results = data.results ?? [];
        // Apply the WHOLE batch's outcomes in ONE setImports pass. The
        // old code called updateImport/removeImport per item, and each
        // of those maps over the entire imports array, so a big drop was
        // O(n^2) state churn on the main thread (a primary freeze). Here
        // we build the patch/remove sets once and walk the array once.
        const patches = new Map<string, Partial<ImportedMedia>>();
        const removeIds = new Set<string>();
        let rejectedCount = 0;
        items.forEach(({ ph }, k) => {
          const res = results[k];
          if (!res) {
            patches.set(ph.id, { status: "failed", error: "no result" });
            return;
          }
          // Per-item NSFW rejection: pull the tile entirely (matches the
          // single endpoint's 451 handling) so a blocked image doesn't
          // sit as a failed ghost.
          if (res.status === "rejected") {
            removeIds.add(ph.id);
            rejectedCount++;
            return;
          }
          if (res.status === "failed") {
            patches.set(ph.id, { status: "failed", error: res.error || "upload failed" });
            return;
          }
          patches.set(ph.id, {
            status: "ready",
            backendId: res.import_id,
            filename: res.filename,
            width: res.width,
            height: res.height,
            blurhash: res.blurhash ?? null,
            detections: [],
            // Point preview at the server image now, the viewer renders from
            // media.preview directly (the grid tile uses backendId, but the
            // viewer doesn't). Without this the just-uploaded image opens to a
            // blank/blurhash screen until /overview re-hydrates on reopen.
            preview: res.filename
              ? `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(res.filename)}`
              : ph.preview,
          });
        });
        setImports((cur) => {
          const out: ImportedMedia[] = [];
          for (const m of cur) {
            if (removeIds.has(m.id)) {
              if (m.preview.startsWith("blob:")) URL.revokeObjectURL(m.preview);
              continue;
            }
            const p = patches.get(m.id);
            out.push(p ? { ...m, ...p } : m);
          }
          return out;
        });
        if (rejectedCount > 0) {
          setImportsTotal((n) => (n != null ? Math.max(0, n - rejectedCount) : n));
        }
        bumpDone(items.length);
      };

      // Chunk into batches and run BATCH_CONCURRENCY at a time.
      const batches: number[][] = [];
      for (let i = 0; i < candidates.length; i += BATCH_SIZE) {
        batches.push(
          Array.from(
            { length: Math.min(BATCH_SIZE, candidates.length - i) },
            (_v, k) => i + k,
          ),
        );
      }
      let bi = 0;
      const runner = async () => {
        while (bi < batches.length) {
          const mine = batches[bi++];
          try {
            await uploadBatch(mine);
          } catch (e) {
            console.error("[v2 import-batch] swallowed:", e);
            bumpDone(mine.length);
          }
        }
      };
      try {
        await Promise.all(
          Array.from({ length: Math.min(BATCH_CONCURRENCY, batches.length) }, runner),
        );
      } finally {
        setStatsRefreshSignal((n) => n + 1);
        setImportProgress(null);
      }
    })();
  };

  const processImport = async (m: ImportedMedia) => {
    // Deferred-labelling flow: upload bytes only, no per-image SAM3
    // call. The user runs a single labelling job from the project
    // page (Start button → label_charlie job) which iterates over
    // unlabelled imports server-side. /imports/raw persists the
    // image + an empty detections list with labelled=false; the job
    // flips that flag and writes detections back when it processes
    // the entry.
    if (!projectId) {
      // No project yet, keep the local preview but mark ready with
      // no detections. The hand-off effect will create the project
      // and re-upload via /raw once it does.
      updateImport(m.id, { status: "ready", detections: [] });
      return;
    }

    // Fully copy the file's bytes into a fresh File so a subsequent
    // retry doesn't share the underlying (possibly-detached) blob
    // source. Decoupled from canvas/bitmap lifetime entirely. Tries
    // arrayBuffer() first; falls back to FileReader for Safari edge
    // cases where arrayBuffer() returns an unusable buffer on a
    // detached blob.
    const reread = async (file: File): Promise<File> => {
      try {
        const buf = await file.arrayBuffer();
        if (buf.byteLength > 0) {
          return new File([buf], file.name, { type: file.type || "application/octet-stream" });
        }
      } catch {
        /* fall through to FileReader */
      }
      try {
        const buf = await new Promise<ArrayBuffer>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            const r = reader.result;
            if (r instanceof ArrayBuffer && r.byteLength > 0) resolve(r);
            else reject(new Error("FileReader produced empty/invalid buffer"));
          };
          reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
          reader.readAsArrayBuffer(file);
        });
        return new File([buf], file.name, { type: file.type || "application/octet-stream" });
      } catch {
        return file;
      }
    };

    // Safari's File/Blob can lose its backing memory between the
    // time `resizeForUpload` returns the File and the time FormData
    // tries to read it (canvas-toBlob detach, drag-source GC, etc.),
    // so we always copy bytes into a fresh ArrayBuffer-backed File
    // immediately before the POST. Chrome doesn't need this but it's
    // cheap (one byte copy per upload) and removes a whole class of
    // intermittent failures.
    const attempt = async (file: File): Promise<Response> => {
      const fresh = await reread(file);
      const fd = new FormData();
      fd.append("image", fresh);
      // FE-supplied timestamp: handleImportFiles pre-assigns these
      // in reverse drop order so within a batch, first-dropped >
      // last-dropped, which makes the gallery's DESC sort + the
      // upload chain's drop-order processing both walk top-left
      // → bottom-right. Backend falls back to its own datetime.now()
      // if the field's missing or unparseable.
      if (typeof m.createdAt === "number") {
        fd.append("created_at_ms", String(m.createdAt));
      }
      // Idempotency key - same value on both first-attempt and retry
      // so the BE can short-circuit and return the existing record
      // when the first POST landed but the response was lost.
      if (m.idempotencyKey) {
        fd.append("idempotency_key", m.idempotencyKey);
      }
      return apiFetch(
        `/api/v2/projects/${projectId}/imports/raw`,
        { method: "POST", body: fd },
      );
    };

    let r: Response;
    try {
      r = await attempt(m.file);
    } catch (e) {
      // attempt() already rereads bytes once internally; this second
      // pass exists for transient network blips (e.g. connection
      // reset, Safari "Load failed" with no body).
      console.warn("[v2 import-raw] first attempt threw, retrying:", e);
      try {
        r = await attempt(m.file);
      } catch (e2) {
        console.error("[v2 import-raw] retry failed:", e2, "file:", {
          name: m.file.name,
          size: m.file.size,
          type: m.file.type || "<empty>",
        });
        updateImport(m.id, { status: "failed", error: e2 instanceof Error ? e2.message : String(e2) });
        return;
      }
    }

    if (!r.ok) {
      const body = await r.text().catch(() => "");
      // 451 = NSFW rejected. Don't leave a "failed" placeholder
      // sitting in the gallery for safety violations, pull it
      // out entirely. Other 4xx/5xx codes keep the failed state
      // so the user can see the issue + retry.
      if (r.status === 451) {
        console.warn("[v2 import-raw] nsfw rejected", m.file.name, body);
        removeImport(m.id);
        return;
      }
      console.error("[v2 import-raw] http", r.status, body);
      // Include the first chunk of the response body in the failed
      // state so the gallery tile surfaces what actually went wrong
      // (e.g. "image too large" / "unsupported format") instead of
      // just "http 400".
      const detail = body.slice(0, 120).trim();
      updateImport(m.id, {
        status: "failed",
        error: detail ? `http ${r.status}, ${detail}` : `http ${r.status}`,
      });
      return;
    }

    const pd = (await r.json()) as {
      import_id?: string;
      filename?: string;
      width?: number;
      height?: number;
      blurhash?: string | null;
    };
    // filename MUST be set on the FE state or the augmentations
    // card's previewSources filter (which requires `m.filename`)
    // strips this entry out, leaving the user staring at
    // "Loading dataset…" until /overview hydrates separately.
    updateImport(m.id, {
      status: "ready",
      backendId: pd.import_id,
      filename: pd.filename,
      width: pd.width,
      height: pd.height,
      blurhash: pd.blurhash ?? null,
      detections: [],
      // Point preview at the server image so the viewer (which renders from
      // media.preview, not backendId) shows it immediately instead of a blank
      // screen until reopen.
      preview: pd.filename
        ? `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(pd.filename)}`
        : m.preview,
    });
    // Stats card needs to recount imports + repaint the
    // "Unlabelled" badge after every new upload, not just after
    // labelling. The bump is cheap (drives the lite-stats fetch
    // which hits the disk sidecar) and matches the user's
    // expectation that the badge updates live as images land.
    setStatsRefreshSignal((n) => n + 1);
  };

  // Number of imports that haven't been through the labelling job
  // yet, OR have been actively cleared by the user (editedBoxes set
  // to []). Drives whether the Start button is shown / enabled.
  // Counting cleared images here means clicking "Clear all" in the
  // viewer re-arms the Start button, the tile flips to Unlabelled
  // and the button copy reads "new images" automatically.
  // Signal priority:
  //   1. editedBoxes when explicitly set by the user - ground truth.
  //   2. detectionCount from /overview - BE-supplied, populated on
  //      first paint and survives the gallery's NO_SYNTH_DETS path
  //      that empties `detections` in state to save memory.
  //   3. labelStats - non-empty when the image has any chip-rail
  //      label, used as a backup signal for legacy hydrated rows.
  //   4. detections.length as a last resort.
  // Without #2 + #3 the count flipped to "unlabelled" any time the
  // viewer's 30 s mask-strip TTL ran or the FE was running with
  // NO_SYNTH_DETS=1, which surfaced as the Start-labelling button
  // showing "all images" / "new images" copy even on fully-labelled
  // datasets.
  const hasEffectiveBoxes = (m: ImportedMedia): boolean => {
    if (m.editedBoxes !== undefined) return (m.editedBoxes?.length ?? 0) > 0;
    if (typeof m.detectionCount === "number") return m.detectionCount > 0;
    if (m.labelStats && Object.keys(m.labelStats).length > 0) return true;
    return (m.detections?.length ?? 0) > 0;
  };
  const unlabelledImportCount = useMemo(
    () => imports.filter((m) => m.status === "ready" && !hasEffectiveBoxes(m)).length,
    [imports],
  );
  // Number of imports that DO have effective boxes, auto detections
  // that survived or user edits. Used by the Start-button copy logic
  // to pick between "all images" (fresh dataset) and "new images"
  // (some already labelled, others need labelling).
  const labelledImportCount = useMemo(
    () => imports.filter((m) => m.status === "ready" && hasEffectiveBoxes(m)).length,
    [imports],
  );
  // (detectedLabelSet + freshLabels are computed below, after the
  // editLabels state declaration, so the useMemo closure can read
  // the current label set without a forward reference.)

  // Number of imports still in the upload / resize / safety-check
  // pipeline (status === "processing"). When > 0, the Start labelling
  // gate is closed and a progress card mirrors the labelling card so
  // the user sees what's in flight before they can run the next step.
  const processingImportCount = useMemo(
    () => imports.filter((m) => m.status === "processing").length,
    [imports],
  );
  // Total of the most recent processing batch, used as the
  // denominator on the progress card so "9 / 12" feels stable while
  // tiles flip ready one-by-one. Bumps up whenever the processing
  // count grows past the recorded total (new batch landing on top of
  // an in-flight one), resets to zero once the queue drains.
  const [processingBatchTotal, setProcessingBatchTotal] = useState(0);
  useEffect(() => {
    if (processingImportCount === 0) {
      if (processingBatchTotal !== 0) setProcessingBatchTotal(0);
    } else if (processingImportCount > processingBatchTotal) {
      setProcessingBatchTotal(processingImportCount);
    }
  }, [processingImportCount, processingBatchTotal]);
  const processingDone = Math.max(0, processingBatchTotal - processingImportCount);

  const startLabellingJob = async () => {
    if (!projectId) return;
    if (labelJob && labelJob.status === "running") return;
    // Credit cutoff. Hits as soon as the user's monthly credit
    // allowance is exhausted (labels + uploads + storage weighted).
    // Surface a transient toast so the click isn't silent and the
    // user understands why nothing happened.
    if (overCreditLimit) {
      setVideoError("Out of credits this period. Upgrade to keep labelling.");
      window.setTimeout(() => setVideoError(null), 5000);
      return;
    }
    // A fresh label OR changed detection settings warrants a full
    // re-pass even when every image already has detections (the new
    // label hasn't been searched for, or the user moved the confidence
    // / area sliders). Otherwise the original gate (must have
    // unlabelled images) still applies.
    const forceRelabel = freshLabels.length > 0 || settingsChanged;
    if (!forceRelabel && unlabelledImportCount === 0) return;
    setLabelJobStarting(true);
    // Flush any pending per-image edits to the server BEFORE scheduling.
    // The auto-PUT effect that persists editedBoxes is 600ms-debounced,
    // so clearing a tile's labels and immediately clicking Start raced
    // it: the job scheduled against a stale manifest where the cleared
    // image still looked labelled, and the backend 400'd with "no
    // unlabelled images". Draining the dirty set here (awaited) makes
    // the server agree with what the user sees before the job runs.
    try {
      const dirtyIds = Array.from(dirtyImportIdsRef.current);
      if (dirtyIds.length > 0 && projectId) {
        const byId = new Map(imports.map((m) => [m.id, m]));
        await Promise.all(
          dirtyIds.map(async (did) => {
            const m = byId.get(did);
            if (!m || !m.backendId || m.editedBoxes === undefined) return;
            const hasInflight = m.editedBoxes.some(
              (b) => b.detecting || b.classifying || b.segmenting,
            );
            if (hasInflight) return;
            const cleaned = m.editedBoxes.map(stripTransientBoxFlags);
            try {
              await apiFetch(
                `/api/v2/projects/${projectId}/imports/${m.backendId}`,
                {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ editedBoxes: cleaned }),
                },
              );
              lastEditedRef.current.set(m.id, JSON.stringify(cleaned));
              dirtyImportIdsRef.current.delete(did);
            } catch (e) {
              console.warn("[label-job] pre-flush edit PUT failed:", e);
            }
          }),
        );
      }
    } catch (e) {
      console.warn("[label-job] pre-flush failed:", e);
    }
    // Flush the project tag list too. addLabel only mutates local state;
    // the PUT that persists tags is 600ms-debounced and the backend reads
    // tags from the manifest, so clicking Start right after typing a new
    // label raced it - the job ran against the OLD tags and produced zero
    // detections for the new label (a wasted full pass). Await the tags
    // PUT here when dirty so the manifest carries the label first.
    try {
      const tagsPayload = JSON.stringify({ labels: editLabels, aliases: labelAliases, colours: labelColours });
      if (tagsPayload !== labelsSavedRef.current) {
        const tr = await apiFetch(`/api/projects/${projectId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tags: editLabels, label_aliases: labelAliases, labelColours }),
        });
        if (tr.ok) labelsSavedRef.current = tagsPayload;
      }
    } catch (e) {
      console.warn("[label-job] pre-flush tags failed:", e);
    }
    try {
      const r = await apiFetch(`/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: projectId,
          kind: "label_charlie",
          user: "hamish",  // backend's job system requires a user; FE sets it elsewhere via apiFetch but kind=label_charlie doesn't care
          // SAM3 knobs from the Annotations card. Backend treats
          // missing values as "use defaults", sending them
          // unconditionally keeps the user's last pick locked in.
          params: {
            sam3_threshold: sam3Threshold,
            sam3_mask_threshold: sam3MaskThreshold,
            sam3_min_relative_area: sam3MinRelativeArea,
            // Native-resolution tiling for large frames (Downscale vs
            // Tile control). Tile size is left to the backend default —
            // the inference-native value.
            tile_native: tileNative,
            // When set, backend processes every image (not just the
            // pending ones), replacing auto detections so the new
            // label surfaces across the dataset. User-edited boxes
            // stay untouched.
            force_relabel: forceRelabel,
          },
        }),
      });
      if (!r.ok) {
        console.warn("[label-job] schedule failed:", r.status);
        return;
      }
      const d = (await r.json()) as { jobId?: string };
      if (d.jobId) {
        setLabelJob({
          jobId: d.jobId,
          status: "running",
          index: 0,
          // When force-relabelling, every image is in scope so the
          // progress denominator should be the dataset total, not the
          // (possibly zero) unlabelled count. Use importsTotal (the real
          // manifest count) when known rather than imports.length, which
          // is only the rows hydrated into state so far - on a large,
          // partially-loaded project the latter under-counted and the job
          // card showed a wrong total/ETA until the first poll corrected it.
          total: forceRelabel ? (importsTotal ?? imports.length) : unlabelledImportCount,
          startedAt: Date.now(),
        });
        // Optimistically snapshot the labels we just submitted for
        // labelling. The backend persists this at job-end and the
        // next /overview poll would pick it up anyway, but mirroring
        // it here means the button copy / forceRelabel decision is
        // already correct if the user clicks again before the
        // refresh lands.
        setLabelsLastRun(editLabels.slice());
        // Mirror for detection settings: the dataset is now (being)
        // labelled with the current sliders, so clear the "settings
        // changed" state until the user moves a slider again.
        setLastRunSettings({
          threshold: sam3Threshold,
          maskThreshold: sam3MaskThreshold,
          minRelativeArea: sam3MinRelativeArea,
          tileNative,
        });
      }
    } catch (e) {
      console.warn("[label-job] start failed:", e);
    } finally {
      setLabelJobStarting(false);
    }
  };

  // Tracks the serialised form of editedBoxes that the SERVER
  // already has for each import. The auto-PUT effect (below) only
  // fires PUTs when an import's local editedBoxes diverges from
  // this snapshot. Hydration paths (syncAnnotations + initial
  // /overview merge) write into this map so server-loaded data
  // never gets re-PUT'd back at the server, which used to fan out
  // dozens of v2_update_import calls on every project open and
  // trigger the per-edit auto-augment hook unprovoked.
  const lastEditedRef = useRef<Map<string, string>>(new Map());
  // Set of import-ids the auto-PUT effect has already observed at
  // least once. Lets us distinguish "import just appeared in state"
  // (from a hydration path that didn't yet populate lastEditedRef)
  // from "user edited an import we've been watching" so a delete
  // mid-labelling job doesn't get swallowed as a hydration artifact.
  const observedImportIdsRef = useRef<Set<string>>(new Set());

  // Splice the latest /annotations payload into FE state. Used by
  // both the per-tick live update during a labelling job AND the
  // on-done refetch, the per-image detection box content + the
  // labelled_preview cache-buster ride on the same call.
  const syncAnnotations = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    try {
      // cache:no-store, without this the browser may serve a stale
      // /annotations response captured a few seconds back, freezing
      // the dataset gallery on the previous image's detections.
      // scope=imports drops the refs half we don't read here.
      const annR = await apiFetch(
        `/api/v2/projects/${projectId}/annotations?scope=imports`,
        { cache: "no-store" },
      );
      if (!annR.ok) return;
      const annData = (await annR.json()) as {
        imports?: Record<string, {
          detections?: WireDetection[];
          editedBoxes?: EditableBox[] | null;
          timings?: Record<string, number | null>;
        }>;
      };
      if (!annData.imports) return;
      const annMap = annData.imports;
      setImports((cur) =>
        cur.map((m) => {
          if (!m.backendId) return m;
          const a = annMap[m.backendId];
          if (!a) return m;
          const nextDetections = (a.detections ?? []).map(unwrapWireDetection);
          const hadDetections = (m.detections?.length ?? 0) > 0;
          const hasDetections = nextDetections.length > 0;
          // Rising edge of "no detections → has detections" means a
          // labelling pass just landed for this image. Bump
          // labelledAt so the labelled_preview URL changes once and
          // the browser refetches.
          const relabelled = !hadDetections && hasDetections;
          const labelledAt = relabelled ? Date.now() : m.labelledAt;
          // editedBoxes resolution. Normally we keep whatever the
          // server tells us (or fall back to the FE copy), but the
          // server doesn't reset editedBoxes during a labelling job
          // (it treats user-edited boxes as sacred). That means a
          // stale per-image "Clear all" (which set editedBoxes: [])
          // would survive a subsequent label-all and freeze the
          // editor at zero boxes even though detections came back.
          // When this image was just relabelled and the FE-held
          // editedBoxes is empty, drop it so the fresh detections
          // win in the editorBoxes priority rule.
          // Preserve unsaved local edits. If this import is dirty (the
          // user changed boxes and the 600ms auto-PUT hasn't landed yet),
          // keep the FE copy: taking the server's stale editedBoxes here
          // would silently revert the edit AND, via the lastEditedRef
          // baseline below, poison the autosave dedup so it skips the real
          // PUT and the change is lost for good. Mirrors the verdicts
          // path's verdictsDirtyRef guard. A box mid-gesture (detecting /
          // classifying / segmenting) counts as dirty too: the dirty set
          // is drained on every 600ms auto-PUT tick, so between a drain
          // and the gesture's final commit the flag alone under-reports.
          const isDirty =
            dirtyImportIdsRef.current.has(m.id) ||
            (Array.isArray(m.editedBoxes) &&
              m.editedBoxes.some((b) => b.detecting || b.classifying || b.segmenting));
          let nextEditedBoxes: EditableBox[] | undefined;
          if (isDirty) {
            nextEditedBoxes = m.editedBoxes;
          } else {
            nextEditedBoxes = Array.isArray(a.editedBoxes)
              ? a.editedBoxes.map(stripTransientBoxFlags)
              : m.editedBoxes;
            if (
              relabelled &&
              Array.isArray(nextEditedBoxes) &&
              nextEditedBoxes.length === 0
            ) {
              nextEditedBoxes = undefined;
            }
          }
          // Re-baseline lastEditedRef ONLY when we took the server copy,
          // so the auto-PUT doesn't fire a redundant v2_update_import
          // (which would kick an augmentation regen the user never asked
          // for). For a dirty import we keep the prior baseline so the
          // autosave still sees the edit as pending and PUTs it.
          if (nextEditedBoxes !== undefined && !isDirty) {
            lastEditedRef.current.set(
              m.id,
              JSON.stringify(nextEditedBoxes ?? []),
            );
          }
          return {
            ...m,
            detections: nextDetections,
            editedBoxes: nextEditedBoxes,
            labelledAt,
          };
        }),
      );
    } catch (e) {
      console.warn("[label-job] annotations sync failed:", e);
    }
  }, [projectId]);

  // Tracks the last progress.index we already synced annotations for,
  // so a second tick with the same index doesn't re-fetch needlessly.
  // Survives the polling effect via useRef.
  const lastSyncedIndexRef = useRef<{ jobId: string; index: number } | null>(null);

  // Latest labelJob mirrored into a ref so the polling tick reads
  // current progress without re-binding the effect on every state
  // update. Without this, putting `labelJob` directly in the polling
  // effect's deps tore down + rebuilt the interval ~once per second
  //, fragile under any unrelated parent re-render (modal mount,
  // intersection observer firings) which could land between teardown
  // and the immediate-tick that the rebuild fires, leaving the user
  // staring at a job that visibly stopped advancing.
  const labelJobRef = useRef<LabelJobState | null>(null);
  useEffect(() => {
    labelJobRef.current = labelJob;
  }, [labelJob]);

  // Polling effect identity. Only changes when the actual job changes
  // (different jobId, or status flipped out of "running"). Progress
  // updates (index/total/currentImage) leave this stable so the
  // 1-second interval keeps firing without React tearing it down.
  const pollingJobId =
    labelJob && labelJob.status === "running" && labelJob.jobId
      ? labelJob.jobId
      : null;

  useEffect(() => {
    if (!pollingJobId || !projectId) return;
    const jobId = pollingJobId;
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      // Read latest job state from the ref so we compare against
      // current progress, not the stale closure of the effect's
      // initial bind. If the active job changed under us (cancel +
      // re-start), exit, the effect re-runs with the new jobId.
      const cur = labelJobRef.current;
      if (!cur || cur.jobId !== jobId || cur.status !== "running") return;
      try {
        const r = await apiFetch(
          `/api/projects/${projectId}/jobs/${jobId}`,
          { cache: "no-store" },
        );
        if (!r.ok) return;
        const d = (await r.json()) as {
          status?: string;
          progress?: { index?: number; total?: number; image?: string } | null;
        };
        if (cancelled) return;
        const c = labelJobRef.current;
        if (!c || c.jobId !== jobId) return;
        const status = (d.status || "").toLowerCase();
        const idx = Number(d.progress?.index ?? 0);
        const total = Number(d.progress?.total ?? c.total);
        const currentImage = d.progress?.image ?? null;
        if (status === "done" || status === "succeeded" || status === "completed") {
          setLabelJob({ ...c, status: "done", index: total, total, currentImage: null });
          await syncAnnotations();
          // Labelling finished, counts / label distribution / health
          // factors are now stale on the dataset stats card.
          setStatsRefreshSignal((n) => n + 1);
          // Backend's label_charlie hook also kicks off an auto-
          // augment_generate when the project has a saved
          // augmentationConfig. The augment job can finish FAST
          // (one new image with light camera dials runs in <1s)
          // and slip between the two augment-job polls we have
          // running. Schedule a few delayed /overview refreshes to
          // catch the n_augmentations bump regardless of which
          // poll did or didn't see the transition. Cheap (3 GETs)
          // and covers the case the user reported: augmentation
          // sparkle missing on a freshly-labelled tile until full
          // refresh.
          const refreshOverviewSoon = async () => {
            if (!projectId) return;
            try {
              const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
              if (!r.ok) return;
              const ov = await r.json() as {
                imports?: { id: string; n_augmentations?: number }[];
                filter_counts?: {
                  all: number; unlabelled: number; unrated: number;
                  good: number; bad: number; unsure: number;
                };
              };
              // Refresh the gallery filter pill counts (e.g. the
              // "Unlabelled N" chip). filterCountsOverride is the BE's
              // full-dataset count; the job just relabelled everything,
              // so without this the chip kept the pre-labelling number
              // until a manual page refresh.
              if (ov.filter_counts) setFilterCountsOverride(ov.filter_counts);
              const next = new Map((ov.imports ?? []).map((i) => [i.id, i.n_augmentations ?? 0]));
              setImports((cur) => {
                let mutated = false;
                const out = cur.map((m) => {
                  if (!m.backendId) return m;
                  const n = next.get(m.backendId);
                  if (n === undefined || n === m.nAugmentations) return m;
                  mutated = true;
                  return { ...m, nAugmentations: n };
                });
                return mutated ? out : cur;
              });
            } catch { /* ignore */ }
          };
          // Fire immediately for the filter-count refresh, then stagger
          // 1.5s / 4s / 10s - typical auto-augment jobs finish inside the
          // first window; the later windows pick up the long-tail
          // (background swap on many images).
          void refreshOverviewSoon();
          window.setTimeout(refreshOverviewSoon, 1500);
          window.setTimeout(refreshOverviewSoon, 4000);
          window.setTimeout(refreshOverviewSoon, 10000);
          // Force-refresh every just-labelled tile's labelled_preview
          // URL by stamping a fresh labelledAt. The rising-edge
          // bump in syncAnnotations covers most cases, but the
          // backend re-bakes the preview at the end of each image
          // pass, pushing labelledAt for every import the job
          // touched guarantees the browser refetches the freshly
          // segmented cover photo (without it we'd serve the
          // pre-label cached version until the user navigated).
          const labelledNow = Date.now();
          setImports((cur) => cur.map((it) => (
            it.backendId ? { ...it, labelledAt: labelledNow } : it
          )));
        } else if (status === "failed" || status === "error") {
          setLabelJob({ ...c, status: "failed", currentImage: null });
        } else if (status === "cancelled" || status === "canceled") {
          setLabelJob({ ...c, status: "cancelled", currentImage: null });
        } else {
          // Still running. Push the progress update + sync annotations
          // any time the backend's index advanced past our last sync.
          if (
            idx !== c.index
            || total !== c.total
            || currentImage !== (c.currentImage ?? null)
          ) {
            setLabelJob({ ...c, index: idx, total, currentImage });
          }
          const last = lastSyncedIndexRef.current;
          if (
            idx > 0
            && (last == null
              || last.jobId !== jobId
              || last.index < idx)
          ) {
            lastSyncedIndexRef.current = { jobId, index: idx };
            await syncAnnotations();
          }
        }
      } catch (e) {
        console.warn("[label-job] poll error:", e);
      }
    };
    const id = window.setInterval(tick, 1000);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollingJobId, projectId, syncAnnotations]);

  // Independent poller for an active purge_label job. Reads
  // /api/projects/{id}/jobs/{jobId} every second, mirrors
  // progress + status into purgeJob, and on completion strips any
  // residual references to the label from local state so the chips
  // / boxes the FE was holding don't reanimate the term.
  const purgePollingId =
    purgeJob && purgeJob.status === "running" && purgeJob.jobId
      ? purgeJob.jobId
      : null;
  useEffect(() => {
    if (!purgePollingId || !projectId) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const r = await apiFetch(
          `/api/projects/${projectId}/jobs/${purgePollingId}`,
          { cache: "no-store" },
        );
        if (!r.ok) return;
        const d = (await r.json()) as {
          status?: string;
          progress?: { index?: number; total?: number; image?: string } | null;
        };
        if (cancelled) return;
        const status = (d.status || "").toLowerCase();
        const idx = Number(d.progress?.index ?? 0);
        const total = Number(d.progress?.total ?? 0);
        const image = d.progress?.image ?? null;
        if (status === "done" || status === "succeeded" || status === "completed") {
          setPurgeJob((cur) => cur ? { ...cur, status: "done", index: total || cur.total, total: total || cur.total, currentImage: null } : cur);
          // Manifest is fresh; pull the latest annotations so the
          // viewer + dataset gallery drop any boxes we just purged.
          void syncAnnotations();
          setStatsRefreshSignal((n) => n + 1);
        } else if (status === "failed" || status === "error") {
          setPurgeJob((cur) => cur ? { ...cur, status: "failed", currentImage: null } : cur);
        } else if (status === "cancelled" || status === "canceled") {
          setPurgeJob((cur) => cur ? { ...cur, status: "cancelled", currentImage: null } : cur);
        } else {
          setPurgeJob((cur) => cur ? { ...cur, index: idx, total: total || cur.total, currentImage: image } : cur);
        }
      } catch {
        // ignore, next tick will retry
      }
    };
    const id = window.setInterval(tick, 1000);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [purgePollingId, projectId, syncAnnotations]);

  // Resume an in-flight labelling job when the user navigates back to
  // the project. The backend keeps the job alive across page reloads,
  // so we ask /api/projects/{id}/jobs/active on mount and re-attach
  // the polling effect if a `label_charlie` job is still running.
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch(`/api/projects/${projectId}/jobs/active`);
        if (!r.ok) return;
        const d = (await r.json()) as
          | {
              id?: string;
              kind?: string;
              status?: string;
              n_images?: number;
              startedAt?: string | number | null;
              progress?: { index?: number; total?: number; image?: string } | null;
            }
          | null;
        if (cancelled || !d) return;
        if (
          d.kind === "label_charlie"
          && (d.status === "running" || d.status === "queued")
          && d.id
        ) {
          setLabelJob((cur) => {
            if (cur && cur.jobId === d.id) return cur;
            // Backend startedAt is ISO; convert to ms for the FE ETA.
            const startedAt = d.startedAt
              ? new Date(d.startedAt).getTime()
              : Date.now();
            return {
              jobId: d.id!,
              status: "running",
              index: Number(d.progress?.index ?? 0),
              total: Number(d.progress?.total ?? d.n_images ?? 0),
              startedAt,
              currentImage: d.progress?.image ?? null,
            };
          });
        }
        // Re-attach to an in-flight label purge, same shape so we
        // reuse the LabelJobCard chrome below.
        if (
          d.kind === "purge_label"
          && (d.status === "running" || d.status === "queued")
          && d.id
        ) {
          setPurgeJob((cur) => {
            if (cur && cur.jobId === d.id) return cur;
            const startedAt = d.startedAt
              ? new Date(d.startedAt).getTime()
              : Date.now();
            return {
              jobId: d.id!,
              status: "running",
              index: Number(d.progress?.index ?? 0),
              total: Number(d.progress?.total ?? d.n_images ?? 0),
              startedAt,
              currentImage: d.progress?.image ?? null,
            };
          });
        }
      } catch {
        // best-effort, if the endpoint fails the user can re-trigger
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const removeImport = (id: string) => {
    const m = importsLatestRef.current.find((x) => x.id === id);
    if (!m) return;
    // Only revoke object URLs we created locally (preview is set
    // to URL.createObjectURL on upload). Backend-served imports
    // use a regular https URL, calling revoke on those is a no-op
    // but cleaner to skip.
    if (m.preview.startsWith("blob:")) URL.revokeObjectURL(m.preview);
    // Optimistic: drop from state immediately so the tile fades
    // out without waiting for the backend. Also decrement the
    // total counter - the gallery uses `Math.min(displayLimit,
    // totalImports) - visible.length` to decide how many skeleton
    // placeholders to render, so without this decrement deleting
    // a tile left a white skeleton in its slot until the user
    // hit refresh.
    setImports((cur) => cur.filter((x) => x.id !== id));
    setImportsTotal((n) => (n != null ? Math.max(0, n - 1) : n));
    if (!m.backendId || !projectId) return;
    // Fire the DELETE. Previously fire-and-forget with a
    // .catch() that ONLY logged rejections - non-2xx responses
    // (401, 404, 500) didn't fire .catch(), so a failed DELETE
    // silently lost the deletion and the entry came back on the
    // next /overview fetch. Now we explicitly check r.ok and
    // restore the entry on failure.
    (async () => {
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/imports/${m.backendId}`,
          { method: "DELETE" },
        );
        // 404 / 410 mean the backend has no such import - which is the
        // exact end-state a delete wants. Treat it as success and keep
        // the tile removed. This is what makes ghost tiles (stale-cache
        // ids the server never had) actually deletable instead of
        // bouncing back. Only genuinely transient failures (5xx,
        // network) restore the tile so the user can retry.
        if (!r.ok && r.status !== 404 && r.status !== 410) {
          const body = await r.text().catch(() => "");
          console.warn(`[v2 import-delete] http ${r.status} on ${m.backendId} - ${body.slice(0, 200)}`);
          // Restore so the user can retry instead of losing the
          // tile from the UI while the backend still has it.
          setImports((cur) =>
            cur.some((x) => x.id === id) ? cur : [m, ...cur].sort(compareImportedMediaDesc),
          );
          setImportsTotal((n) => (n != null ? n + 1 : n));
        }
      } catch (e) {
        console.warn("[v2 import-delete] network error:", e);
        setImports((cur) =>
          cur.some((x) => x.id === id) ? cur : [m, ...cur].sort(compareImportedMediaDesc),
        );
        setImportsTotal((n) => (n != null ? n + 1 : n));
      }
    })();
  };

  // Bulk delete used by the Dataset's Select mode. One round-trip
  // for the whole batch, the backend takes a single manifest lock,
  // unlinks every file, and re-derives the cover in one pass.
  const removeImportsBatch = useCallback(async (ids: string[]) => {
    if (!projectId || ids.length === 0) return;
    // Local revoke + optimistic state update first so the grid
    // empties immediately. If the POST fails the user can refresh
    // to recover, this matches the fire-and-forget pattern the
    // single-delete handler uses.
    let droppedCount = 0;
    setImports((cur) => {
      const dropSet = new Set(ids);
      for (const m of cur) {
        if (dropSet.has(m.id) && m.preview.startsWith("blob:")) {
          URL.revokeObjectURL(m.preview);
        }
      }
      const next = cur.filter((x) => !dropSet.has(x.id));
      droppedCount = cur.length - next.length;
      return next;
    });
    // Decrement the total so the gallery doesn't render skeleton
    // slots for the deleted images (same bug as removeImport).
    if (droppedCount > 0) {
      setImportsTotal((n) => (n != null ? Math.max(0, n - droppedCount) : n));
    }
    const backendIds = ids
      .map((id) => importsLatestRef.current.find((m) => m.id === id)?.backendId)
      .filter((b): b is string => !!b);
    if (backendIds.length === 0) return;
    try {
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/imports/delete_batch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids: backendIds }),
        },
      );
      if (!r.ok) console.warn("[v2 import-delete-batch] failed:", r.status);
    } catch (e) {
      console.warn("[v2 import-delete-batch] error:", e);
    }
    // Re-fetch stats AFTER the DELETE round-trip lands. The
    // length-change useEffect already fires before the request
    // returns, but the backend manifest hasn't seen the deletion
    // yet at that moment, /dataset-stats would echo the pre-
    // delete numbers. Bumping again here forces a fresh fetch
    // against the post-delete manifest so the card actually
    // updates.
    setStatsRefreshSignal((n) => n + 1);
  }, [projectId]);

  // Mirror the imports list into the workspace meta cache so the
  // next mount paints the right tiles immediately and a refresh
  // doesn't restore deleted images. Without this the cache stayed
  // frozen at hydration time and any add/delete went stale on
  // remount. Filtering to backendId-bearing entries skips in-flight
  // uploads that haven't persisted yet.
  useEffect(() => {
    if (!projectId) return;
    const persisted = imports.filter((m) => m.backendId);
    if (persisted.length === 0 && imports.length === 0) {
      // First-mount empty state, leave the cache to the hydration
      // effect so we don't wipe it before /overview lands.
      return;
    }
    // Merge with existing cache so a transient null blurhash
    // (e.g. /overview served from a sidecar built before the
    // async blurhash backfill landed) doesn't wipe a previously-
    // good hash. Symptom this fixes: reopening a project showed
    // valid tiles for the first ~20 and white placeholders for
    // the rest - the persist clobber kept overwriting good
    // hashes with null on every render of an in-flight hydration.
    const prev = readProjectMeta(projectId);
    const prevById = new Map(
      (prev?.importTiles ?? []).map((t) => [t.id, t]),
    );
    patchProjectMeta(projectId, {
      importTiles: persisted.map((m) => {
        const old = prevById.get(m.backendId!);
        return {
          id: m.backendId!,
          filename: m.filename ?? old?.filename ?? "",
          // Prefer the live blurhash when present; fall back to
          // the cached one when not. Never persist null over a
          // known-good hash.
          blurhash: m.blurhash ?? old?.blurhash ?? null,
          width: m.width ?? old?.width,
          height: m.height ?? old?.height,
          createdAt: m.createdAt ?? old?.createdAt ?? null,
        };
      }),
      nImages: persisted.length,
    });
  }, [imports, projectId]);

  // Hydrate imports from the project manifest on first mount /
  // projectId change. Fetches each persisted image as a Blob so we
  // can wrap it in a File (required for downstream FormData uploads
  // in click-to-detect / segment-box / classify-box flows).
  // hydratedRef survives StrictMode's simulated unmount so the
  // second effect run sees the claim and skips. No `cancelled`
  // flag, same story as the references hydration: the simulated
  // cleanup would set cancelled=true and the original async A
  // would skip its setImports forever.
  // Imports hydration shares the same /overview + /annotations
  // round-trip as references, refs do the actual fetches and write
  // both stores. This effect now ONLY surfaces the import tile
  // metadata as soon as /overview lands, so the dataset gallery
  // paints its placeholders without waiting on /annotations. The
  // refs hydration effect upstream handles the annotations splice
  // for both refs AND imports.
  //
  // Kept as a separate effect from refs so it can run independently
  // when the user lands directly on a project that has no
  // references but does have imports.
  const hydratedRef = useRef<Set<string>>(new Set());
  // Flips true once the /overview fetch has completed (or failed) for
  // the current project. Tabs that gate empty-state copy on "is the
  // dataset really empty?" wait on this so they don't claim "Add
  // images" while the imports list is still loading from the server.
  const [importsReady, setImportsReady] = useState(false);
  // Backend's `imports_total` for the project, the count the user
  // expects to see next to "Dataset". Tracked separately from
  // `imports.length` because the actual state fills in progressively
  // (/initial returns 20, then /overview returns 100, then the
  // remainder fills in the rest). Without this the header counter
  // climbed from 20 → 100 → 941 over several seconds, which the user
  // reported as "the number takes 5-10 s to update".
  const [importsTotal, setImportsTotal] = useState<number | null>(null);
  useEffect(() => {
    if (!projectId) return;
    if (hydratedRef.current.has(projectId)) return;
    hydratedRef.current.add(projectId);
    (async () => {
      type ImpRow = {
        id: string;
        filename: string;
        originalFilename?: string;
        width?: number;
        height?: number;
        blurhash?: string | null;
        createdAt?: number | string | null;
        source?: { kind?: string; url?: string } | null;
        n_augmentations?: number;
        n_detections?: number;
        label_set?: string[];
        has_edits?: boolean;
        label_stats?: Record<string, number>;
        labelledAt?: number | null;
      };
      type OverviewSlice = {
        imports?: ImpRow[];
        imports_total?: number;
        imports_offset?: number;
        updatedAt?: string | null;
        filter_counts?: FilterCountsOverride;
      };
      // Two-phase fetch:
      //   1. First 100 imports, covers the visible viewport on big
      //      projects AND fits a typical small project (<100 images)
      //      in a single roundtrip so the pagination overhead doesn't
      //      regress small-project load times.
      //   2. Remaining imports, fetched in the background and
      //      appended to the gallery so infinite-scroll has data
      //      before the user reaches the bottom.
      // The merge logic dedupes by id so a slow second batch can't
      // overwrite edits the user made on a first-batch tile.
      const FIRST_BATCH = 100;
      // Retry helper, the user hit an intermittent case on the
      // large project where this fetch silently bailed (non-ok or
      // network blip) and the gallery froze at /initial's 20 tiles.
      // Three tries with linear back-off gets us past transient
      // Cloudflare warm-ups + Vast.ai cold-starts. cache:no-store so
      // browser HTTP cache can't serve a stale truncated response
      // from a previous session.
      const fetchOverview = async (
        path: string,
        label: string,
      ): Promise<OverviewSlice | null> => {
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const r = await apiFetch(path, { cache: "no-store" });
            if (r.ok) return (await r.json()) as OverviewSlice;
            console.warn(`[v2 imports hydrate] ${label} returned ${r.status} (attempt ${attempt + 1}/3)`);
          } catch (e) {
            console.warn(`[v2 imports hydrate] ${label} threw (attempt ${attempt + 1}/3):`, e);
          }
          if (attempt < 2) await new Promise((res) => setTimeout(res, 400 * (attempt + 1)));
        }
        return null;
      };
      const firstSliceMaybe = await fetchOverview(
        `/api/v2/projects/${projectId}/overview?imports_limit=${FIRST_BATCH}&imports_offset=0`,
        "first batch",
      );
      if (!firstSliceMaybe) {
        // First-batch fetch failed after retries, flip importsReady
        // anyway so the loader can fade with whatever /initial managed
        // to deliver. Without this the project sat behind the loader
        // for the full 6 s cap.
        setImportsReady(true);
        return;
      }
      const firstSlice: OverviewSlice = firstSliceMaybe;
      if (firstSlice.updatedAt !== undefined) {
        setManifestUpdatedAt(firstSlice.updatedAt ?? null);
      }
      if (firstSlice.filter_counts) {
        setFilterCountsOverride(firstSlice.filter_counts);
      }
      const total = firstSlice.imports_total ?? (firstSlice.imports?.length ?? 0);
      // Mirror to state so the dataset header counter reflects the
      // true total even when /initial returned before this fetch
      // landed with a possibly-updated number.
      setImportsTotal(total);
      try {
        const imps = firstSlice.imports ?? [];
        if (imps.length === 0 && total === 0) {
          // Server authoritatively has zero imports. Drop every
          // provisional cache-seeded tile (ghosts from a stale cache)
          // but keep in-flight uploads that haven't persisted yet.
          setImports((cur) => cur.filter((m) => !m.provisional));
          return;
        }
        const hydrated: ImportedMedia[] = imps.map((imp) => {
          // Synthesize a placeholder detections array sized to the
          // backend's n_detections, label-distributed across label_set.
          // The gallery tile reads detections.length for the box count
          // and predLabel for the chip rail, both work fine with this
          // shape, so the tile paints with full info from /overview
          // alone instead of waiting on /annotations. /annotations
          // later replaces this with the real geometry when the user
          // opens the viewer.
          const labelSet = imp.label_set ?? [];
          const nDetections = imp.n_detections ?? 0;
          // Per-label count map; chip rail reads from this directly.
          const labelStats: Record<string, number> = (imp as {
            label_stats?: Record<string, number>;
          }).label_stats ?? Object.fromEntries(labelSet.map((lab) => [lab, 0]));
          // Build minimal placeholder shapes ONLY when the flag is
          // off - the per-tile chip rendering already reads
          // labelStats and detectionCount instead of mapping
          // detections.length. Saving ~25 ImportDetection objects
          // per image at this hydration step.
          const placeholderDetections: ImportDetection[] = NO_SYNTH_DETS
            ? []
            : nDetections > 0
            ? Array.from({ length: nDetections }, (_, i) => {
                const lab = labelSet[i % Math.max(1, labelSet.length)] ?? null;
                return ({
                  box: [0, 0, 0, 0] as [number, number, number, number],
                  mask: null,
                  predLabel: lab,
                  rejected: false,
                } as unknown) as ImportDetection;
              })
            : [];
          return {
            id: imp.id,
            backendId: imp.id,
            file: new File([], imp.originalFilename || imp.filename),
            filename: imp.filename,
            preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
            blurhash: imp.blurhash ?? null,
            status: "ready" as const,
            width: imp.width,
            height: imp.height,
            createdAt: imp.createdAt ?? null,
            sourceUrl: imp.source?.url ?? null,
            derivedLabel: (imp as { derivedLabel?: string | null }).derivedLabel ?? null,
            nAugmentations: imp.n_augmentations ?? 0,
            labelStats,
            detectionCount: nDetections,
            // Hint at /annotations status so the gallery doesn't show
            // "Unlabelled" before the real detections arrive, empty
            // arrays from /overview count as "we don't know yet".
            detections: placeholderDetections,
            // Persisted cachebuster so the segmented labelled-preview
            // survives a cold reopen instead of falling back to the
            // bare URL (stale blank preview). See backend _tile_overview.
            labelledAt: imp.labelledAt ?? undefined,
          };
        });
        setImports((cur) => {
          // Merge: keep existing detections / editedBoxes if the
          // refs effect's annotations splice already populated them
          // ahead of this fetch landing. Match by id. Entries that
          // exist locally but NOT in the backend's fresh list are
          // dropped, that's the case where the user deleted an
          // image in another tab (or in this tab before the cache
          // was up to date), and we don't want a ghost entry to come
          // back on remount. In-flight uploads (no backendId yet)
          // are kept regardless so they survive until persistence.
          const byId = new Map(hydrated.map((h) => [h.id, h]));
          const merged: ImportedMedia[] = [];
          for (const c of cur) {
            // Match by in-memory id OR the server id in backendId (uploaded
            // tiles keep their local id for life; only backendId holds the
            // server id), so a persisted upload + its in-progress edit are
            // no longer dropped during a slow first-batch hydrate.
            const matchKey = byId.has(c.id)
              ? c.id
              : (c.backendId && byId.has(c.backendId) ? c.backendId : null);
            if (matchKey) {
              const fresh = byId.get(matchKey)!;
              merged.push({
                ...fresh,
                // Keep the in-memory id + backendId so local-id-keyed state
                // (in-flight edits, importsById) stays valid.
                id: c.id,
                backendId: c.backendId,
                // Prefer cur ONLY when it has real (non-placeholder)
                // detection geometry OR has been user-edited.
                // Otherwise adopt fresh so a stale /initial placeholder
                // (n_detections=0 because the sidecar pre-dates the
                // labelling pass) doesn't win over a fresh /overview
                // placeholder with correct counts + labels.
                detections: (hasRealDetections(c.detections) || c.editedBoxes !== undefined)
                  ? c.detections
                  : fresh.detections,
                editedBoxes: c.editedBoxes,
                timings: c.timings,
                // Same blurhash preservation as the /initial merge -
                // never overwrite a known-good hash with null.
                blurhash: fresh.blurhash ?? c.blurhash ?? null,
                // Confirmed by the server this session.
                provisional: undefined,
              });
              byId.delete(matchKey);
            } else if (c.provisional || !c.backendId) {
              // Unconfirmed cache tile (the remainder batch may still
              // confirm it; the post-load prune below drops it if not)
              // or an in-flight upload. This first batch only covers the
              // newest 100, so it must NOT drop tiles beyond that window.
              merged.push(c);
            }
            // else: previously-confirmed tile the server no longer lists
            // → deleted elsewhere → drop so the gallery matches truth.
          }
          for (const h of byId.values()) merged.push(h);
          return merged.sort(compareImportedMediaDesc);
        });
        // When the whole dataset fits in the first batch, this response
        // is authoritative for every import - prune any cache-seeded
        // tile it didn't confirm (a ghost). For larger datasets the
        // prune runs after the remainder batch lands (below).
        if (total <= FIRST_BATCH) {
          setImports((cur) => cur.filter((m) => !m.provisional));
        }
        patchProjectMeta(projectId, {
          importTiles: imps.map((imp) => ({
            id: imp.id,
            filename: imp.filename,
            blurhash: imp.blurhash ?? null,
            width: imp.width,
            height: imp.height,
            createdAt: imp.createdAt ?? null,
          })),
          nImages: total,
        });
      } catch (e) {
        console.warn("[v2 imports hydrate] first-batch merge failed:", e);
      } finally {
        // First-batch paint is done, let the rest of the page settle
        // (drop card, label-job poller etc) without waiting on the
        // remainder fetch below.
        setImportsReady(true);
      }
      // Background fetch for the remainder, if any. Appends to the
      // gallery so infinite scroll has a populated tail when the user
      // reaches it.
      if (total > FIRST_BATCH) {
        const rest = await fetchOverview(
          `/api/v2/projects/${projectId}/overview?imports_offset=${FIRST_BATCH}`,
          "remainder",
        );
        if (!rest) return;
        try {
          const restImps = rest.imports ?? [];
          const hydratedRest: ImportedMedia[] = restImps.map((imp) => {
            const labelSet = imp.label_set ?? [];
            const nDetections = imp.n_detections ?? 0;
            const labelStats: Record<string, number> = (imp as {
              label_stats?: Record<string, number>;
            }).label_stats ?? Object.fromEntries(labelSet.map((lab) => [lab, 0]));
            const placeholderDetections: ImportDetection[] = NO_SYNTH_DETS
              ? []
              : nDetections > 0
              ? Array.from({ length: nDetections }, (_, i) => {
                  const lab = labelSet[i % Math.max(1, labelSet.length)] ?? null;
                  return ({
                    box: [0, 0, 0, 0] as [number, number, number, number],
                    mask: null,
                    predLabel: lab,
                    rejected: false,
                  } as unknown) as ImportDetection;
                })
              : [];
            return {
              id: imp.id,
              backendId: imp.id,
              file: new File([], imp.originalFilename || imp.filename),
              filename: imp.filename,
              preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
              blurhash: imp.blurhash ?? null,
              status: "ready" as const,
              width: imp.width,
              height: imp.height,
              createdAt: imp.createdAt ?? null,
              sourceUrl: imp.source?.url ?? null,
              derivedLabel: (imp as { derivedLabel?: string | null }).derivedLabel ?? null,
              nAugmentations: imp.n_augmentations ?? 0,
              labelStats,
              detectionCount: nDetections,
              detections: placeholderDetections,
            };
          });
          setImports((cur) => {
            // Mirror the first-batch merge: a cache-seeded tile already
            // in `cur` carries NO labelStats / detectionCount (the
            // importTiles cache only stores id/filename/blurhash/size).
            // The old merge just cleared `provisional` on those, so
            // every tile beyond the first batch kept an empty chip rail
            // until the user hovered (which lazy-fetched /annotations).
            // Adopt the fresh per-tile counts here so chips paint on
            // load for the whole gallery, while still preferring any
            // real / user-edited geometry already in memory.
            const freshById = new Map(hydratedRest.map((h) => [h.id, h]));
            const consumed = new Set<string>();
            const out: ImportedMedia[] = [];
            for (const c of cur) {
              const matchKey = freshById.has(c.id)
                ? c.id
                : (c.backendId && freshById.has(c.backendId) ? c.backendId : null);
              if (matchKey) {
                const fresh = freshById.get(matchKey)!;
                consumed.add(matchKey);
                out.push({
                  ...fresh,
                  id: c.id,
                  backendId: c.backendId,
                  detections: (hasRealDetections(c.detections) || c.editedBoxes !== undefined)
                    ? c.detections
                    : fresh.detections,
                  editedBoxes: c.editedBoxes,
                  timings: c.timings,
                  blurhash: fresh.blurhash ?? c.blurhash ?? null,
                  provisional: undefined,
                });
              } else {
                // Not in this remainder slice (first-batch tile or an
                // in-flight upload). Keep as-is; provisional cache
                // ghosts get pruned below.
                out.push(c);
              }
            }
            for (const h of hydratedRest) {
              if (!consumed.has(h.id)) out.push(h);
            }
            // Full dataset is now loaded (first batch + remainder), so
            // any tile still flagged provisional is a ghost the server
            // never confirmed (stale cache) - drop it. In-flight uploads
            // are never provisional, so they survive.
            return out
              .filter((m) => !m.provisional)
              .sort(compareImportedMediaDesc);
          });
          if (restImps.length > 0) {
            patchProjectMeta(projectId, {
              importTiles: [...(firstSlice.imports ?? []), ...restImps].map((imp) => ({
                id: imp.id,
                filename: imp.filename,
                blurhash: imp.blurhash ?? null,
                width: imp.width,
                height: imp.height,
                createdAt: imp.createdAt ?? null,
              })),
              nImages: total,
            });
          }
        } catch (e) {
          console.warn("[v2 imports hydrate] remainder fetch failed:", e);
        }
      }
    })();
  }, [projectId]);

  // Blurhash backfill retry. The backend's _kick_blurhash_backfill
  // runs async - for legacy imports that never had a blurhash
  // computed, the first /overview response after a server-cache
  // miss (e.g. cmd-option-R force-reload, process restart) comes
  // back with blurhash=null for those tiles. The FE renders the
  // solid-fill fallback ("white placeholder") and never refetches,
  // so the user stares at white tiles until a full page reload.
  // This effect schedules a re-fetch of /overview while any tile
  // is missing a blurhash, with backoff capped at 3 tries - by
  // then the encoder loop has had >15s and either filled them in
  // or the backfill genuinely failed for that file.
  const blurhashRetryCountRef = useRef(0);
  useEffect(() => {
    if (!projectId) return;
    if (!importsReady) return;
    const missing = imports.some(
      (m) => m.backendId && !m.blurhash && m.status === "ready",
    );
    if (!missing) {
      blurhashRetryCountRef.current = 0;
      return;
    }
    if (blurhashRetryCountRef.current >= 3) return;
    const attempt = blurhashRetryCountRef.current;
    blurhashRetryCountRef.current = attempt + 1;
    // 3s, 6s, 12s. Lets the executor backfill loop drain even on a
    // big project with hundreds of legacy entries.
    const delay = 3000 * Math.pow(2, attempt);
    const t = window.setTimeout(async () => {
      try {
        // Pull the full first-batch slice so the response actually
        // includes imports (imports_limit=0 skips the per-import
        // payload entirely - useful for meta-only fetches but no
        // good here).
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/overview?imports_limit=100&imports_offset=0`,
          { cache: "no-store" },
        );
        if (!r.ok) return;
        const data = await r.json() as { imports?: { id: string; blurhash?: string | null }[] };
        const byBid = new Map((data.imports ?? []).map((i) => [i.id, i.blurhash ?? null]));
        if (byBid.size === 0) return;
        setImports((cur) => {
          let mutated = false;
          const out = cur.map((m) => {
            if (!m.backendId || m.blurhash) return m;
            const fresh = byBid.get(m.backendId);
            if (!fresh) return m;
            mutated = true;
            return { ...m, blurhash: fresh };
          });
          return mutated ? out : cur;
        });
      } catch { /* keep retrying on next tick */ }
    }, delay);
    return () => window.clearTimeout(t);
  }, [imports, projectId, importsReady]);

  // Persist editedBoxes to the backend whenever the user edits an
  // import's boxes. Debounced via the standard "ref to latest" +
  // microtask pattern so we don't fire a PUT on every keystroke /
  // drag tick.
  //
  // On success we bump `labelledAt` for the matching import, the
  // PUT triggers a server-side re-bake of the labelled_preview JPEG,
  // and the labelledAt timestamp rides on the preview URL as a
  // cache-buster so the browser refetches the fresh image. Without
  // this, click-to-detect added a box but the gallery thumb kept
  // showing the stale segmented cover until a page refresh.
  // (lastEditedRef is declared earlier so syncAnnotations can also
  // write to it on hydration, see the comment beside its `useRef`.)
  useEffect(() => {
    if (!projectId) return;
    const id = window.setTimeout(() => {
      // Drain the dirty-import set: only imports whose editedBoxes
      // changed via updateImport need PUT consideration. On a 1000-
      // image project this avoids walking the full list on every
      // unrelated state change (label-job progress, augment counts,
      // labelledAt cache-buster bumps). Imports that pass through
      // here for first-observation still seed lastEditedRef below
      // via the imports walk fallback.
      const dirtyIds = Array.from(dirtyImportIdsRef.current);
      dirtyImportIdsRef.current.clear();
      // Single map lookup over the full imports list for first-time
      // observation seeding (so we never PUT a hydration-shaped
      // editedBoxes back at the server it just came from). After
      // this seeding pass the dirty set is the source of truth.
      for (const m of imports) {
        if (!m.backendId) continue;
        if (observedImportIdsRef.current.has(m.id)) continue;
        observedImportIdsRef.current.add(m.id);
        if (m.editedBoxes !== undefined) {
          lastEditedRef.current.set(m.id, JSON.stringify(m.editedBoxes));
        }
      }
      const importsById = new Map(imports.map((m) => [m.id, m]));
      for (const dirtyId of dirtyIds) {
        const m = importsById.get(dirtyId);
        if (!m || !m.backendId) continue;
        // We've seen this id before. Skip until editedBoxes is an
        // array we can serialise (undefined still means the user
        // hasn't touched it).
        if (!m.editedBoxes) continue;
        // Defer the PUT entirely while any box on this import is
        // mid-gesture (detecting / classifying / segmenting). The
        // single atomic state commits in BoxEditor's rebuilt
        // click-to-detect + add-box flows leave no transient flags
        // by the time the gesture lands, so a flagged box here means
        // we're catching it BEFORE that commit - and persisting it
        // would either ship a placeholder label ("detecting"/"new")
        // or a half-finished mask. Wait for the next render.
        const hasInflight = m.editedBoxes.some(
          (b) => b.detecting || b.classifying || b.segmenting,
        );
        if (hasInflight) continue;
        // Strip transient UI flags (detecting / classifying /
        // segmenting) before persisting. Belt-and-braces alongside
        // the hasInflight guard - keeps the wire payload pure even
        // if a future code path forgets to clear a flag.
        const cleaned = m.editedBoxes.map(stripTransientBoxFlags);
        const serialised = JSON.stringify(cleaned);
        const last = lastEditedRef.current.get(m.id);
        if (last === serialised) continue;
        lastEditedRef.current.set(m.id, serialised);
        const importId = m.backendId;
        const localId = m.id;
        apiFetch(`/api/v2/projects/${projectId}/imports/${importId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ editedBoxes: cleaned }),
        })
          .then((r) => {
            if (!r.ok) {
              console.warn(`[v2 import-patch] http ${r.status} on ${importId}`);
              return;
            }
            // Force the labelled_preview URL to refresh.
            const now = Date.now();
            setImports((cur) =>
              cur.map((it) =>
                it.id === localId ? { ...it, labelledAt: now } : it,
              ),
            );
            // Manual edit committed → nudge the dataset stats card.
            // Label distribution + health factors change with every
            // box add/remove/relabel; without this bump the card
            // would only refresh on the next augmentations-generated
            // tick (which is firing in the background anyway thanks
            // to the auto-augment-after-edit hook).
            setStatsRefreshSignal((n) => n + 1);
          })
          .catch((e) => console.warn("[v2 import-patch] failed:", e));
      }
    }, 600);
    return () => window.clearTimeout(id);
  }, [imports, projectId]);

  // Renamable project title with 2-step undo.
  const [projectTitle, setProjectTitle] = useState(projectName);
  const [titleEditing, setTitleEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState(projectName);
  const [titleHistory, setTitleHistory] = useState<string[]>([]);
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!titleEditing) return;
    setTitleDraft(projectTitle);
    const t = window.setTimeout(() => {
      titleInputRef.current?.focus();
      titleInputRef.current?.select();
    }, 0);
    return () => window.clearTimeout(t);
  }, [titleEditing, projectTitle]);

  const commitTitleRename = () => {
    const next = titleDraft.trim();
    setTitleEditing(false);
    if (!next || next === projectTitle) return;
    setTitleHistory((h) => [projectTitle, ...h].slice(0, 2));
    setProjectTitle(next);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const meta = e.ctrlKey || e.metaKey;
      if (!meta || e.key.toLowerCase() !== "z" || e.shiftKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (titleHistory.length === 0) return;
      e.preventDefault();
      setTitleHistory((h) => {
        if (h.length === 0) return h;
        const [prev, ...rest] = h;
        setProjectTitle(prev);
        return rest;
      });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [titleHistory]);

  // Editable labels with 2-step undo (covers add, delete, rename).
  type LabelSnapshot = {
    labels: string[];
    refs: ReferenceImage[];
    aliases: Record<string, string>;
  };
  const [editLabels, setEditLabels] = useState<string[]>(() => {
    // Seed from the prop if the parent handed real labels (post-
    // onboarding, deep-link with cached meta). Otherwise pull from
    // the per-project localStorage cache so the chip rail paints
    // INSTANTLY on cold reopen instead of waiting on /initial.
    // Empty array is the fall-through, which the workspace-card
    // hydration overrides as soon as it lands.
    if (labels.length > 0) return labels;
    if (!projectId) return labels;
    try {
      const cached = readProjectMeta(projectId);
      if (cached?.labels && cached.labels.length > 0) return cached.labels;
    } catch { /* ignore localStorage failures */ }
    return labels;
  });
  // Backend's snapshot of the label list used in the last
  // label_charlie job. Null on never-labelled projects; populated
  // on hydrate from /initial + /overview, and optimistically set
  // when the user starts a new job so subsequent button clicks
  // reflect the latest run before the backend's persist round-
  // trips back. Source of truth for freshLabels (below), which
  // used to derive its "searched-for" set from detection results -
  // that broke for labels that were prompted-for but never matched.
  const [labelsLastRun, setLabelsLastRun] = useState<string[] | null>(null);
  // Set of label keys (canonical, lowercased) that already appear in
  // at least one detection or edited box in the dataset. Labels in
  // the project's editLabels list that AREN'T in this set are
  // "fresh", the user just added them and the labelling pass
  // hasn't surfaced them yet. The Start-labelling button copy calls
  // a single fresh label out by name so the user sees the button
  // adapt the moment they add a new tag.
  const detectedLabelSet = useMemo(() => {
    const s = new Set<string>();
    for (const m of imports) {
      const edited = m.editedBoxes;
      if (edited !== undefined) {
        for (const b of edited ?? []) {
          const lab = (b?.label ?? "").trim().toLowerCase();
          if (lab) s.add(lab);
        }
      }
      for (const d of m.detections ?? []) {
        const lab = (d.predLabel ?? "").trim().toLowerCase()
          || (d.gdLabel ?? "").trim().toLowerCase();
        if (lab) s.add(lab);
      }
    }
    return s;
  }, [imports]);
  // labelsLastRun is the authoritative "searched-for" set when the
  // backend has recorded one (i.e. label_charlie has run at least
  // once on this project since the snapshot was introduced). Falls
  // back to detectedLabelSet for legacy projects with no snapshot -
  // that path can still false-positive on labels that were searched
  // but never matched, but only until the next labelling job
  // persists labelsLastRun.
  const freshLabels = useMemo(
    () => {
      if (labelsLastRun != null) {
        const lastRunSet = new Set(
          labelsLastRun.map((l) => l.trim().toLowerCase()),
        );
        return editLabels.filter(
          (lab) => !lastRunSet.has(lab.trim().toLowerCase()),
        );
      }
      return editLabels.filter(
        (lab) => !detectedLabelSet.has(lab.trim().toLowerCase()),
      );
    },
    [editLabels, detectedLabelSet, labelsLastRun],
  );
  // Toast-style transient error surfaced by addLabel / renameLabel
  // when the user types a blocked term. Auto-dismisses after a few
  // seconds so the project page chrome doesn't have to host a
  // permanent error rail.
  const [labelError, setLabelError] = useState<string | null>(null);
  const labelErrorTimerRef = useRef<number | null>(null);
  const flashLabelError = useCallback((msg: string) => {
    setLabelError(msg);
    if (labelErrorTimerRef.current) window.clearTimeout(labelErrorTimerRef.current);
    labelErrorTimerRef.current = window.setTimeout(() => {
      setLabelError(null);
      labelErrorTimerRef.current = null;
    }, 4200);
  }, []);
  useEffect(() => () => {
    if (labelErrorTimerRef.current) window.clearTimeout(labelErrorTimerRef.current);
  }, []);

  // Display-only aliases: maps canonical_lower → display_name. The
  // backend continues to use the canonical label for every detection,
  // ref, and import, renaming via the chip only updates the alias,
  // so existing annotations stay valid and the chip's annotation
  // count keeps matching the canonical key (which fixed the
  // "renamed labels look greyed out" symptom).
  const [labelAliases, setLabelAliases] = useState<Record<string, string>>(() => {
    // Seed from the workspace meta cache so the very first paint
    // already renders chips with the renamed display name. Without
    // this the chips briefly flash the canonical label, then update
    // when the /api/projects/{id} manifest GET resolves and
    // setLabelAliases runs.
    if (typeof window === "undefined" || !projectId) return {};
    const cached = readProjectMeta(projectId);
    return cached?.labelAliases ?? {};
  });
  // Per-label colour overrides ({canonical_lower: "#rrggbb"}). Same
  // store-and-forward shape as labelAliases, set in Settings, mirrored
  // into the workspace project-meta cache so the workspace + public
  // chips repaint without waiting for the next poll.
  const [labelColours, setLabelColours] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined" || !projectId) return {};
    const cached = readProjectMeta(projectId);
    return cached?.labelColours ?? {};
  });
  // Whether the project is private. Drives the padlock badge next to
  // the title (parity with the workspace + public cards) and stays
  // in sync with the Settings popup via the broadcast effect below.
  // Seed from the workspace meta cache so the padlock paints on the
  // first frame when opening a known-private project. Without this
  // we waited on /api/projects/{id} to resolve, which made the
  // padlock pop in ~200 ms after the rest of the page.
  const [isPrivate, setIsPrivate] = useState<boolean>(() => {
    if (typeof window === "undefined" || !projectId) return false;
    return !!readProjectMeta(projectId)?.private;
  });

  // Derived ("child") link, from /overview - drives the derived badge next to
  // the title (mirrors the workspace card). null for normal projects.
  const [derivedInfo, setDerivedInfo] = useState<{ parentProjectId?: string; parentName?: string } | null>(null);
  // A freshly-created derived (child) project has its crops generated in the
  // background, so it starts with zero imports. While that's the case we poll
  // the overview so the gallery fills in, and show a "pulling images" loader
  // instead of an empty grid. `derivedPollDone` ends the loader if no crops ever
  // arrive (rare: a parent with no matching detections).
  const [derivedPollDone, setDerivedPollDone] = useState(false);
  useEffect(() => {
    if (readOnly || !projectId || !derivedInfo || imports.length > 0 || derivedPollDone) return;
    let alive = true;
    let tries = 0;
    const tick = async () => {
      if (!alive) return;
      tries += 1;
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (r.ok) {
          const d = (await r.json()) as {
            imports?: { id: string; filename: string; originalFilename?: string; width?: number; height?: number; blurhash?: string | null; createdAt?: number | string | null; source?: { url?: string } | null; n_augmentations?: number; n_detections?: number; label_stats?: Record<string, number> }[];
          };
          const fresh = d.imports ?? [];
          if (alive && fresh.length > 0) {
            setImports(fresh.map<ImportedMedia>((imp) => ({
              id: imp.id,
              backendId: imp.id,
              file: new File([], imp.originalFilename || imp.filename),
              filename: imp.filename,
              preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
              blurhash: imp.blurhash ?? null,
              status: "ready" as const,
              width: imp.width,
              height: imp.height,
              createdAt: imp.createdAt ?? null,
              sourceUrl: imp.source?.url ?? null,
              derivedLabel: (imp as { derivedLabel?: string | null }).derivedLabel ?? null,
              nAugmentations: imp.n_augmentations ?? 0,
              // Carry the per-label counts so the tile chip rail renders on load
              // (without these, derived crops showed labels only after a hover
              // lazy-loaded their geometry).
              detectionCount: imp.n_detections ?? 0,
              labelStats: imp.label_stats ?? {},
            })).sort(compareImportedMediaDesc));
            return; // imports now present; effect re-runs + bails on the guard
          }
        }
      } catch { /* keep polling */ }
      if (!alive) return;
      if (tries >= 40) { setDerivedPollDone(true); return; } // ~2 min, then give up
      window.setTimeout(tick, 3000);
    };
    const t = window.setTimeout(tick, 2000);
    return () => { alive = false; window.clearTimeout(t); };
  }, [readOnly, projectId, derivedInfo, imports.length, derivedPollDone]);

  // Per-image verdicts from the fast-review modal. Keyed by the
  // import's backendId (falls back to local id for in-flight rows
  // that haven't persisted yet). Hydrated from the manifest's
  // `verdicts` field on mount; mutations PUT a debounced save back.
  const [verdicts, setVerdicts] = useState<Record<string, "good" | "bad" | "unsure">>({});

  // Authoritative gallery-filter chip counts, computed BE-side over
  // the full imports list. Populated by the /overview fetch above;
  // overrides the per-item iteration in DatasetGallery so the chips
  // show their final ALL / UNLABELLED / GOOD / BAD numbers on first
  // paint instead of climbing 20 → 100 → N as batches stream in.
  // Mutations (verdict toggle, label run) re-fetch /overview which
  // refreshes this value.
  type FilterCountsOverride = {
    all: number;
    unlabelled: number;
    unrated: number;
    good: number;
    bad: number;
    unsure: number;
  };
  const [filterCountsOverride, setFilterCountsOverride] =
    useState<FilterCountsOverride | null>(null);

  // Manifest's updatedAt for the active project, refreshed from every
  // /overview response. The IDB annotation cache (P5) gates its
  // reads on this: a cached row whose manifestUpdatedAt doesn't
  // match the current value is treated as a miss, so a mutation
  // anywhere in the manifest invalidates the relevant cache entries
  // without needing per-record cache busting. State (not ref) so
  // the viewer's IDB call sites re-evaluate when the value moves.
  const [manifestUpdatedAt, setManifestUpdatedAt] = useState<string | null>(null);
  // Eviction is one-shot per session. Schedule it now so the cache
  // doesn't grow without bound across long-running tabs.
  useEffect(() => {
    scheduleLruEviction();
  }, []);

  // P7: track which import the viewer modal currently has open so
  // the pressure-eviction pass below doesn't strip its mask
  // polygons out from under the user. The gallery's viewer effect
  // updates this ref on every index change (or null on close).
  const viewerOpenIdRef = useRef<string | null>(null);

  // P7: heap-pressure eviction. Subscribed once per mount; on fire
  // we walk the imports list and null any in-RAM mask polygons
  // EXCEPT the currently-open viewer's. The 30 s navigation TTL
  // already covers steady-state, this is the proactive complement
  // that triggers when the browser is genuinely close to its heap
  // limit. Cheap when no masks are loaded (the .some check fails
  // fast). No-op when the flag is off or performance.memory is
  // missing (Firefox, Safari).
  useEffect(() => {
    const unsubscribe = subscribePressure(0.75, () => {
      const keep = viewerOpenIdRef.current;
      setImports((cur) => {
        let mutated = false;
        const out = cur.map((m) => {
          if (m.id === keep) return m;
          const dets = m.detections;
          if (!dets || dets.length === 0) return m;
          if (!dets.some((d) => d?.mask)) return m;
          mutated = true;
          return {
            ...m,
            detections: dets.map((d) => ({ ...d, mask: null })),
          };
        });
        return mutated ? out : cur;
      });
    });
    return unsubscribe;
  }, []);
  const verdictsSaveTimer = useRef<number | null>(null);
  const verdictsDirtyRef = useRef(false);
  const scheduleVerdictsSave = useCallback((next: Record<string, "good" | "bad" | "unsure">) => {
    if (!projectId) return;
    verdictsDirtyRef.current = true;
    if (verdictsSaveTimer.current) window.clearTimeout(verdictsSaveTimer.current);
    verdictsSaveTimer.current = window.setTimeout(async () => {
      verdictsSaveTimer.current = null;
      try {
        await apiFetch(`/api/projects/${projectId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ verdicts: next }),
        });
        verdictsDirtyRef.current = false;
      } catch (e) {
        console.warn("[v2 verdicts] save failed:", e);
      }
    }, 350);
  }, [projectId]);
  const setVerdict = useCallback((id: string, v: "good" | "bad" | "unsure") => {
    setVerdicts((cur) => {
      // Toggle off when re-clicking the same verdict (parity with V1).
      const next = { ...cur };
      if (next[id] === v) delete next[id];
      else next[id] = v;
      scheduleVerdictsSave(next);
      return next;
    });
    // Invalidate the BE-computed chip counts so the per-item
    // iteration takes over and reflects the user's just-made change
    // immediately. A subsequent /overview load will re-populate the
    // override with authoritative values.
    setFilterCountsOverride(null);
  }, [scheduleVerdictsSave]);

  // Filter pill on the dataset header, controls which gallery rows
  // render. Default "all"; the pills only appear when a filter would
  // actually thin the gallery.
  type VerdictFilter = "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("all");

  // Review modal state. `reviewScope` chooses which slice of items the
  // modal walks; `reviewing` flips the overlay on.
  const [reviewing, setReviewing] = useState(false);
  const [reviewScope, setReviewScope] = useState<"unrated" | "unsure" | "good" | "bad" | "all">("unrated");
  // True while the full-screen reference editor (in RefImageGrid) is
  // open, so the dataset gallery's back-to-top button hides behind it.
  const [refViewerOpen, setRefViewerOpen] = useState(false);

  // Per-image annotation pull used by ReviewModeV2 as the user
  // navigates. /overview + /annotations?scope=imports only carry
  // boxes + labels, mask polygons live on the per-id endpoint, so
  // without this the review canvas would only ever draw rectangles.
  // Dedup by backendId so flicking back and forth doesn't restorm
  // the network with identical requests.
  const reviewAnnoFetchedRef = useRef<Set<string>>(new Set());
  const requestReviewAnnotations = useCallback(async (backendId: string) => {
    if (!projectId || !backendId) return;
    if (reviewAnnoFetchedRef.current.has(backendId)) return;
    reviewAnnoFetchedRef.current.add(backendId);
    try {
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/annotations/${encodeURIComponent(backendId)}`,
      );
      if (!r.ok) return;
      const body = (await r.json()) as {
        detections?: WireDetection[];
        editedBoxes?: EditableBox[] | null;
      };
      const dets = (body.detections ?? []).map(unwrapWireDetection);
      const cleanedEdited = Array.isArray(body.editedBoxes)
        ? body.editedBoxes.map(stripTransientBoxFlags)
        : null;
      setImports((cur) => cur.map((m) => {
        if (m.backendId !== backendId) return m;
        return {
          ...m,
          detections: dets,
          editedBoxes: cleanedEdited ?? m.editedBoxes,
        };
      }));
    } catch {
      reviewAnnoFetchedRef.current.delete(backendId);
    }
  }, [projectId]);
  // Mask polygons aren't part of /overview's tile payload, only the
  // /annotations endpoint carries the full segmentation data. When
  // the user opens review mode we want those polygons drawn over the
  // image, so fire a one-shot annotations fetch the first time the
  // modal is opened on this project mount. syncAnnotations is also
  // run by the label-job poll, so this fires only when the user
  // hasn't run a fresh labelling job in this session.
  const annotationsHydratedRef = useRef(false);
  useEffect(() => {
    if (!reviewing || annotationsHydratedRef.current) return;
    annotationsHydratedRef.current = true;
    void syncAnnotations();
  }, [reviewing, syncAnnotations]);
  // Resolves a canonical label to its user-visible name (alias if
  // one exists, otherwise the canonical itself). Used everywhere
  // the chips / sidebar / popups render a label that the user types
  // in or eyeballs. Pipeline scoring still uses the canonical.
  const displayLabel = (canonical: string): string => {
    if (!canonical) return canonical;
    const k = canonical.trim().toLowerCase();
    return labelAliases[k] || canonical;
  };

  // Reset editLabels from the labels prop ONLY when the project
  // changes (mount / projectId switch). Without this guard, every
  // workspace poll that returned a fresh `labels` array would
  // clobber the user's in-flight rename/add, which was the
  // "labels don't save" bug. Inside a single project the editLabels
  // state is the source of truth; a debounced PUT below pushes
  // them to the manifest.
  const labelInitRef = useRef<string | null>(null);
  useEffect(() => {
    if (labelInitRef.current !== projectId) {
      labelInitRef.current = projectId ?? null;
      setEditLabels(labels);
      // Drop the previous project's snapshot so we don't briefly
      // compute freshLabels against the wrong dataset's history.
      setLabelsLastRun(null);
      // Pull any persisted aliases on project change so the chips
      // render with their custom display names on first paint. The
      // alias map can lag the labels prop by one network round-trip;
      // we update again from the same fetch below.
      //
      // Uses the /overview slim projection (with imports_limit=0 so
      // it hits the sidecar fast-path and skips the per-import tile
      // reduction) instead of the legacy /api/projects/{id} endpoint
      // which returned the whole manifest, multi-megabyte on big
      // projects and 3 s+ on cold reads. Every field we read below
      // (label_aliases, labelColours, tags, private, owner) is in
      // /overview's payload.
      if (projectId) {
        apiFetch(`/api/v2/projects/${projectId}/overview?imports_limit=0`)
          .then((r) => (r.ok ? r.json() : null))
          .then((m) => {
            const freshAliases = (m && typeof m.label_aliases === "object" && m.label_aliases)
              ? (m.label_aliases as Record<string, string>)
              : {};
            const freshColours = (m && typeof m.labelColours === "object" && m.labelColours)
              ? (m.labelColours as Record<string, string>)
              : {};
            setLabelAliases(freshAliases);
            setLabelColours(freshColours);
            // Verdicts (good / bad / unsure) come down with /overview
            // now so the filter pills + review modal know the
            // existing state on first paint. Skip the apply when the
            // local debounced PUT is still in flight, that would
            // clobber the user's most-recent toggle.
            if (
              !verdictsDirtyRef.current
              && m && typeof m.verdicts === "object" && m.verdicts
            ) {
              setVerdicts(m.verdicts as Record<string, "good" | "bad" | "unsure">);
            }
            // Persist into the meta cache so the NEXT mount paints
            // with the latest aliases + colours on the first frame ,
            // otherwise refreshing right after a Settings edit briefly
            // shows the previously-cached values before this fetch
            // lands.
            // Persist tags too so the next cold mount can seed
            // editLabels from localStorage before the /overview
            // round-trip lands - keeps the chip rail painted on
            // first frame instead of empty-then-pop.
            const freshTags = Array.isArray(m?.tags) ? (m!.tags as string[]) : undefined;
            patchProjectMeta(projectId, {
              labelAliases: freshAliases,
              labelColours: freshColours,
              ...(freshTags ? { labels: freshTags } : {}),
            });
            // Public-view first-mount: the parent only had a cache
            // miss for labels (=[]), so without this catch-up the
            // chip rail at the top stays empty and the box-editor's
            // labelColourMap can't resolve detection labels, falling
            // back to id-coloured chips. Skip when local state has
            // edits beyond the manifest so an in-flight rename isn't
            // clobbered.
            if (m && Array.isArray(m.tags) && labels.length === 0) {
              setEditLabels(m.tags as string[]);
            }
            if (m && Array.isArray((m as { labelsLastRun?: unknown }).labelsLastRun)) {
              setLabelsLastRun(
                (m as { labelsLastRun: string[] }).labelsLastRun,
              );
            } else if ((m as { labelsLastRun?: unknown } | null)?.labelsLastRun === null) {
              setLabelsLastRun(null);
            }
            const slr = (m as {
              settingsLastRun?: {
                threshold?: number;
                mask_threshold?: number;
                min_relative_area?: number;
                tile_native?: boolean;
                tile_size?: number;
              } | null;
            } | null)?.settingsLastRun;
            if (slr) {
              // Trio may be absent when the run used default sliders but
              // pinned tiling — mirror the /initial hydration's guards so
              // this poll path can't clobber the sliders with undefined.
              const th = typeof slr.threshold === "number" ? slr.threshold : SAM3_DEFAULTS.threshold;
              const mt = typeof slr.mask_threshold === "number" ? slr.mask_threshold : SAM3_DEFAULTS.maskThreshold;
              const ma = typeof slr.min_relative_area === "number" ? slr.min_relative_area : SAM3_DEFAULTS.minRelativeArea;
              setSam3Threshold(th);
              setSam3MaskThreshold(mt);
              setSam3MinRelativeArea(ma);
              setTileNative(!!slr.tile_native);
              setLastRunSettings({
                threshold: th,
                maskThreshold: mt,
                minRelativeArea: ma,
                tileNative: !!slr.tile_native,
              });
            }
            const priv = !!m?.private;
            setIsPrivate(priv);
            // Keep the auto-derived dataset behaviour in step with
            // /overview (covers refreshes after a Settings edit). Same
            // cached-only value the mount seeds from /initial; null is
            // ignored so a fresher read isn't clobbered by a stale one.
            const dt = (m as { dataset_type?: { type?: string; reason?: string | null; source?: string | null } | null } | null)?.dataset_type;
            if (dt && (dt.type === "general" || dt.type === "specific")) {
              const next: DatasetTypeValue = {
                type: dt.type,
                reason: typeof dt.reason === "string" ? dt.reason : "",
                source: typeof dt.source === "string" ? dt.source : "auto",
              };
              setDatasetType(next);
              patchProjectMeta(projectId, { datasetType: next });
            }
            if (typeof (m as { max_input_size?: number } | null)?.max_input_size === "number") {
              maxInputSizeRef.current = (m as { max_input_size: number }).max_input_size;
              setMaxInputSize((m as { max_input_size: number }).max_input_size);
            }
            // Mirror into the meta cache so the next mount paints
            // the padlock without waiting for this fetch again. Owner
            // is cached too, drives the readOnly decision in
            // /app/page.tsx on subsequent /app/<id> refreshes.
            const cacheOwner = m?.owner ?? m?.createdBy ?? undefined;
            patchProjectMeta(projectId, {
              private: priv,
              ...(cacheOwner ? { owner: cacheOwner } : {}),
            });
          })
          .catch(() => { setLabelAliases({}); setLabelColours({}); });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, labels]);

  // Stay in sync with Settings: the popup broadcasts a meta-changed
  // event on every save, so the padlock + chip colours repaint here
  // immediately without waiting for the next manifest fetch.
  useEffect(() => {
    if (typeof window === "undefined" || !projectId) return;
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail as {
        projectId?: string;
        private?: boolean;
        labelColours?: Record<string, string>;
      } | null | undefined;
      if (!d || d.projectId !== projectId) return;
      if (typeof d.private === "boolean") setIsPrivate(d.private);
      if (d.labelColours) setLabelColours(d.labelColours);
    };
    window.addEventListener("pixelkit-project-meta-changed", handler);
    return () => window.removeEventListener("pixelkit-project-meta-changed", handler);
  }, [projectId]);

  // Bumped from event handlers + lifecycle ticks (job-done, augment-
  // generated, import add/remove) so the DatasetStatsCard re-fetches
  // its /dataset-stats payload without us having to wire a manual
  // refresh button.
  const [statsRefreshSignal, setStatsRefreshSignal] = useState(0);

  // Refresh the seed stats whenever a stats-changing action fires (the same
  // signal the DatasetStatsCard listens to). The hero stat strip reads
  // detections / augmentations / health off seedStats, which is otherwise only
  // set once from /initial — without this its numbers would lag behind the
  // OverviewPanel's live figures until a reload. Skipped on first mount
  // (signal 0) since /initial already seeded it.
  useEffect(() => {
    if (!projectId || statsRefreshSignal === 0) return;
    let alive = true;
    apiFetch(`/api/v2/projects/${projectId}/dataset-stats?lite=true`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d) setSeedStats(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, [projectId, statsRefreshSignal]);

  // Also nudge a refresh whenever the imports count changes, covers
  // fresh uploads, bulk deletes, and any other dataset-size delta
  // that wouldn't otherwise fire a labelling/augmentation event.
  const importsCountRef = useRef(-1);
  useEffect(() => {
    if (importsCountRef.current === imports.length) return;
    importsCountRef.current = imports.length;
    setStatsRefreshSignal((n) => n + 1);
  }, [imports.length]);

  // Highlight state driven by clicks on the variation plot dots.
  //   - `highlightedImportId`  : the import to scroll to + dim others
  //   - `highlightKind`        : "image" or "augmentation", drives a
  //                              short "augmentation" flash badge over
  //                              the cover photo for the latter
  // Both clear after a short window so the page returns to neutral
  // without the user having to click somewhere else.
  const [highlightedImportId, setHighlightedImportId] = useState<string | null>(null);
  const [highlightKind, setHighlightKind] = useState<"image" | "augmentation" | null>(null);
  const highlightTimerRef = useRef<number | null>(null);
  const jumpToImport = useCallback((importId: string, kind: "image" | "augmentation") => {
    setHighlightedImportId(importId);
    setHighlightKind(kind);
    // Tell the DatasetGallery to expand its displayLimit past the
    // target import so the tile actually mounts before we try to
    // scroll to it. Without this, clicking a variation-plot dot
    // for an image hidden behind the "Show more" gate scrolled
    // nowhere.
    try {
      window.dispatchEvent(new CustomEvent("pixelkit-ensure-import-visible", {
        detail: { projectId, importId },
      }));
    } catch { /* ignore */ }
    // Two-frame wait: the first frame commits the displayLimit
    // bump, the second frame lets React render the newly-included
    // tiles. Only THEN does the querySelector + scroll fire.
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        const el = document.querySelector<HTMLElement>(
          `[data-import-id="${CSS.escape(importId)}"]`,
        );
        el?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });
    if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current);
    // Augmentation flash lingers a touch longer so the badge is
    // readable; the image highlight is brief.
    highlightTimerRef.current = window.setTimeout(() => {
      setHighlightedImportId(null);
      setHighlightKind(null);
    }, kind === "augmentation" ? 3800 : 2600);
  }, []);
  useEffect(() => () => {
    if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current);
  }, []);

  // After an augment_generate job completes, the AugmentationsCard
  // broadcasts a pixelkit-augmentations-generated event. Refetch the
  // /overview slim projection so each import's nAugmentations
  // updates, that's what gates the per-tile Augmentations icon, and
  // without this refresh the user wouldn't see it until they
  // refreshed the page.
  useEffect(() => {
    if (typeof window === "undefined" || !projectId) return;
    const handler = async (e: Event) => {
      const d = (e as CustomEvent).detail as { projectId?: string } | null | undefined;
      if (!d || d.projectId !== projectId) return;
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (!r.ok) return;
        const ov = await r.json() as {
          imports?: { id: string; n_augmentations?: number }[];
        };
        const next = new Map((ov.imports ?? []).map((i) => [i.id, i.n_augmentations ?? 0]));
        setImports((cur) => cur.map((m) => {
          if (!m.backendId) return m;
          const n = next.get(m.backendId);
          if (n === undefined || n === m.nAugmentations) return m;
          return { ...m, nAugmentations: n };
        }));
      } catch {
        /* ignore, next manual nav still updates */
      }
      // Augmentations changed → dataset stats card's augmentation
      // count is stale; nudge it to refetch.
      setStatsRefreshSignal((n) => n + 1);
    };
    window.addEventListener("pixelkit-augmentations-generated", handler);
    return () => window.removeEventListener("pixelkit-augmentations-generated", handler);
  }, [projectId]);

  // Belt-and-braces poll: even if the user navigates away from the
  // Augmentations tab during a generate run (so AugmentationsCard
  // never sees the running→done transition), this lightweight loop
  // refreshes per-import nAugmentations the moment the backend's
  // active job clears. Runs at 2 s, slow enough not to be noisy,
  // fast enough that the icon appears almost immediately.
  const augActiveJobRef = useRef<string | null>(null);
  const refreshAugCountsRef = useRef<() => Promise<void>>(async () => {});
  // Mirror of the active augment_generate job, shaped like
  // LabelJobState so we can render the same card chrome below the
  // labelling/processing cards. Stays null when no augment job is in
  // flight.
  const [augmentJob, setAugmentJob] = useState<LabelJobState | null>(null);
  useEffect(() => {
    if (!projectId) return;
    // Sleep mode: pause the 2s augment poll while the user is idle.
    // Activity flips isIdle false → effect re-fires → poll resumes.
    if (isIdle) return;
    let cancelled = false;
    const refreshOverview = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (!r.ok || cancelled) return;
        const ov = await r.json() as { imports?: { id: string; n_augmentations?: number }[] };
        const next = new Map((ov.imports ?? []).map((i) => [i.id, i.n_augmentations ?? 0]));
        // Only commit a new imports array if at least one tile's
        // nAugmentations actually changed. The poll runs every 2 s
        // so without this guard we'd re-render the entire dataset
        // gallery on every tick, visible as a brief dim/refresh
        // flash behind hover modals.
        setImports((cur) => {
          let mutated = false;
          const out = cur.map((m) => {
            if (!m.backendId) return m;
            const n = next.get(m.backendId);
            if (n === undefined || n === m.nAugmentations) return m;
            mutated = true;
            return { ...m, nAugmentations: n };
          });
          return mutated ? out : cur;
        });
      } catch { /* ignore */ }
    };
    refreshAugCountsRef.current = refreshOverview;
    const tick = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augment/job/active`);
        if (!r.ok || cancelled) return;
        const data = await r.json();
        const prev = augActiveJobRef.current;
        const isActive = data && (data.status === "running" || data.status === "queued");
        // Surface progress to the in-page card. Map the backend's
        // {progress: {index, total, image, phase}} payload into the
        // shared LabelJobState shape so we can reuse LabelJobCard.
        if (isActive) {
          const startedMs = data.startedAt
            ? new Date(data.startedAt).getTime()
            : Date.now();
          const freshIndex = Number(data.progress?.index ?? 0);
          // While the run is in flight, bump the stats card's
          // refresh signal every time the index advances. The
          // backend's batch flush (every 25 images) updates the
          // manifest's n_augmentations and rebuilds the
          // dataset-stats sidecar; without this bump the card
          // would only refresh once at job completion. Capped at
          // "advanced by 1" so a stalled job doesn't churn the
          // signal pointlessly.
          setAugmentJob((cur) => {
            if (cur && freshIndex > (cur.index ?? 0)) {
              setStatsRefreshSignal((n) => n + 1);
            }
            return {
              jobId: data.id,
              status: "running",
              index: freshIndex || (cur?.index ?? 0),
              // No fallback to `data.n_images`, augmentations count
              // must NEVER read as image count. The backend sets
              // progress.total to the estimated augmentation total
              // up-front; if it's somehow missing, keep the previous
              // total rather than displaying a misleading image-count
              // denominator.
              total: Number(data.progress?.total ?? cur?.total ?? 0),
              startedAt: cur?.startedAt ?? startedMs,
              currentImage: data.progress?.image ?? cur?.currentImage ?? null,
            };
          });
        }
        if (prev && !isActive) {
          await refreshOverview();
          // Augment job just finished, mark the card as done so
          // it shows the success state for a beat before auto-
          // dismissing.
          setAugmentJob((cur) => cur ? {
            ...cur,
            status: data?.status === "failed" ? "failed"
              : data?.status === "cancelled" ? "cancelled"
              : "done",
            index: cur.total,
            currentImage: null,
          } : cur);
          // Augment job just finished, nudge the dataset stats
          // card too. The AugmentationsCard's own poll usually
          // fires the augmentations-generated event for this, but
          // fast jobs that complete between two ticks slip through
          // its running→done detector. Bumping here belt-and-braces.
          setStatsRefreshSignal((n) => n + 1);
        } else if (!isActive) {
          // Fast-path completion: the BE finished the job before our
          // first poll (1-image datasets often complete sub-second).
          // `prev` is null in this case so the done-handler above
          // doesn't fire. Close the optimistic stamp from
          // AugmentationsCard's onUpdate by treating it as done.
          setAugmentJob((cur) => {
            if (!cur) return cur;
            if (cur.status !== "queued" && cur.jobId !== "pending") return cur;
            return {
              ...cur,
              status: "done",
              total: cur.total || 1,
              index: cur.total || 1,
              currentImage: null,
            };
          });
        }
        augActiveJobRef.current = isActive ? (data?.id ?? prev) : null;
      } catch { /* ignore */ }
    };
    void tick();
    const id = window.setInterval(tick, 2000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [projectId, isIdle]);

  // Debounced PUT, every label edit (add / remove / rename / alias)
  // schedules a save 600 ms later. Coalesces rapid edits so we don't
  // fire 5 PUTs while the user is typing. Sends both `tags` (canonical
  // labels) and `label_aliases` so a single PUT covers both the
  // delete-from-canonical and rename-via-alias paths.
  const labelsSavedRef = useRef<string>(JSON.stringify({ labels, aliases: {} }));
  useEffect(() => {
    if (!projectId) return;
    const current = JSON.stringify({ labels: editLabels, aliases: labelAliases, colours: labelColours });
    if (current === labelsSavedRef.current) return;
    const t = window.setTimeout(async () => {
      try {
        const r = await apiFetch(`/api/projects/${projectId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tags: editLabels,
            label_aliases: labelAliases,
            labelColours,
          }),
        });
        if (r.ok) {
          labelsSavedRef.current = current;
          // Update the workspace project-meta cache so reopening
          // this project from the workspace card shows the new
          // labels on first paint, instead of waiting for the 4 s
          // /api/projects poll to refresh.
          patchProjectMeta(projectId, {
            labels: editLabels,
            labelAliases,
            labelColours,
          });
          // Broadcast to any sibling view (workspace, public feed)
          // that has this project in its list so they patch their
          // local state immediately, no need to wait for the poll
          // cycle to surface the new tags / aliases / colours.
          if (typeof window !== "undefined") {
            window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
              detail: {
                projectId,
                tags: editLabels,
                label_aliases: labelAliases,
                labelColours,
              },
            }));
          }
        } else {
          console.warn("[v2 labels] save failed:", r.status);
        }
      } catch (e) {
        console.warn("[v2 labels] save error:", e);
      }
    }, 600);
    return () => window.clearTimeout(t);
  }, [editLabels, labelAliases, labelColours, projectId]);
  const [labelHistory, setLabelHistory] = useState<LabelSnapshot[]>([]);
  const [adding, setAdding] = useState(false);
  const [addInput, setAddInput] = useState("");
  const addInputRef = useRef<HTMLInputElement | null>(null);
  // Annotations section has its own add state so its input doesn't
  // race with the header input for focus/blur.

  const pushLabelHistory = () => {
    setLabelHistory((h) =>
      [{
        labels: editLabels.slice(),
        refs: refs.slice(),
        aliases: { ...labelAliases },
      }, ...h].slice(0, 2),
    );
  };

  // (labelPendingDelete + purgeJob state declared higher up so the
  // polling effect can read them without a forward reference.)

  const removeLabel = (idx: number) => {
    const canonical = (editLabels[idx] ?? "").trim().toLowerCase();
    if (!canonical) return;
    setLabelPendingDelete({
      canonical,
      display: displayLabel(editLabels[idx] ?? canonical),
    });
  };

  // Schedule the background strip. Called from DeleteLabelModal on
  // confirm. Stamps the local chip strip + alias map instantly so
  // the user sees the chip disappear; the backend job catches up
  // on the manifest + per-image detections asynchronously.
  const confirmPurgeLabel = async (canonical: string) => {
    if (!projectId) return;
    // Optimistic FE strip so the chip disappears the moment the user
    // confirms; the backend job lands the same change on every image
    // in the dataset. pushLabelHistory snapshots the prior list so
    // the user can undo the local part if the job fails to schedule.
    pushLabelHistory();
    setEditLabels((cur) => cur.filter((l) => l.trim().toLowerCase() !== canonical));
    setLabelAliases((cur) => {
      if (!(canonical in cur)) return cur;
      const next = { ...cur };
      delete next[canonical];
      return next;
    });
    setLabelColours((cur) => {
      if (!(canonical in cur)) return cur;
      const next = { ...cur };
      delete next[canonical];
      return next;
    });
    try {
      const r = await apiFetch(`/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: projectId,
          kind: "purge_label",
          user: username || "system",
          params: { label: canonical },
        }),
      });
      if (!r.ok) {
        console.warn("[purge-label] schedule failed:", r.status);
        return;
      }
      const d = (await r.json()) as { jobId?: string };
      if (d.jobId) {
        setPurgeJob({
          jobId: d.jobId,
          status: "running",
          index: 0,
          total: imports.length,
          startedAt: Date.now(),
        });
      }
    } catch (e) {
      console.warn("[purge-label] schedule failed:", e);
    } finally {
      setLabelPendingDelete(null);
    }
  };

  const confirmClearAllAnnotations = async () => {
    if (!projectId) return;
    try {
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/clear_all_annotations`,
        { method: "POST", headers: { "Content-Type": "application/json" } },
      );
      if (!r.ok) {
        console.warn("[clear-all-annotations] failed:", r.status);
        return;
      }
      setImports((cur) =>
        cur.map((m) => ({
          ...m,
          detections: [],
          editedBoxes: undefined,
        })),
      );
      await syncAnnotations();
      setStatsRefreshSignal((n) => n + 1);
    } catch (e) {
      console.warn("[clear-all-annotations] failed:", e);
    } finally {
      setClearAllOpen(false);
    }
  };

  const addLabel = (raw: string) => {
    const next = raw.trim().replace(/[,.]+$/g, "").trim();
    if (!next) return;
    // Profanity gate. The V2 onboarding flow checks the same set
    // before letting the project be created, without the check here,
    // a user could sneak a blocked term past it on the project page's
    // chip rail because the assert_clean check only fires server-side
    // on the periodic PUT, by which point the chip is already painted
    // in the UI. Surface a transient error so the user knows WHY.
    if (containsProfanity(next)) {
      flashLabelError(`"${next}" can't be used as a label, blocked by the profanity filter.`);
      return;
    }
    if (editLabels.some((l) => l.toLowerCase() === next.toLowerCase())) {
      setAddInput("");
      return;
    }
    // Prevent a new canonical from colliding with an existing alias
    // (would break the display map: same display name showing twice).
    const aliasCollision = Object.values(labelAliases).some(
      (disp) => disp.toLowerCase() === next.toLowerCase(),
    );
    if (aliasCollision) {
      setAddInput("");
      return;
    }
    // Pick a palette colour that's not currently visible on any chip.
    // The auto-assigned palette-map collision logic doesn't reserve
    // slots for explicit overrides, so without this step a new label
    // could hash onto the same hex an existing label is already
    // wearing. Persisting the choice as an override locks it in.
    const usedHex = new Set<string>();
    for (const lab of editLabels) {
      usedHex.add(colourForLabel(editLabels, lab, labelColours).toLowerCase());
    }
    const free = LABEL_COLOURS.filter((c) => !usedHex.has(c.toLowerCase()));
    const pickPool = free.length > 0 ? free : LABEL_COLOURS;
    const colour = pickPool[Math.floor(Math.random() * pickPool.length)];
    pushLabelHistory();
    setEditLabels((cur) => [...cur, next]);
    setLabelColours((cur) => ({ ...cur, [next.toLowerCase()]: colour }));
    setAddInput("");
    // Defensive sync. Adding a label by itself doesn't trigger a
    // labelling pass, but it does flip the per-tile label rail's
    // "fresh label" state and prompts the user to click Start
    // labelling. If a background job has already run against the
    // new label (e.g. on a project where label_charlie was kicked
    // by an upload hook), the annotations + stats need to be in
    // sync so the per-image chips and the viewer reflect the new
    // detections. Cheap to fire even when there's nothing new on
    // the backend - both calls short-circuit on no-op.
    void syncAnnotations();
    setStatsRefreshSignal((n) => n + 1);
  };

  const renameLabel = (idx: number, raw: string) => {
    const next = raw.trim();
    const canonical = editLabels[idx];
    if (!canonical || !next) return;
    // Same profanity gate as addLabel, the rename path is just an
    // alias write on the manifest, the backend's assert_clean doesn't
    // catch it until the periodic PUT, and even then the chip on the
    // canvas keeps showing the bad alias until a refresh. Block here.
    if (containsProfanity(next)) {
      flashLabelError(`"${next}" can't be used as a label, blocked by the profanity filter.`);
      return;
    }
    const k = canonical.trim().toLowerCase();
    const currentDisplay = labelAliases[k] || canonical;
    if (next === currentDisplay) return;
    // No collision with another canonical, keeps the alias map a
    // bijection on display names.
    if (editLabels.some((l, i) => i !== idx && l.toLowerCase() === next.toLowerCase())) return;
    if (Object.entries(labelAliases).some(
      ([key, disp]) => key !== k && disp.toLowerCase() === next.toLowerCase()
    )) return;
    pushLabelHistory();
    setLabelAliases((cur) => {
      const out = { ...cur };
      // Setting the alias to the canonical itself is the same as
      // having no alias, drop the entry so the dict stays minimal
      // and the chip falls back to the canonical naturally.
      if (next.toLowerCase() === canonical.toLowerCase()) {
        delete out[k];
      } else {
        out[k] = next;
      }
      return out;
    });
    // editLabels stays unchanged: the canonical is the source of
    // truth for every detection, ref, and import. The display flips
    // via the alias. This is what kills the "renamed labels go
    // grey" bug, the annotation count is still keyed on canonical.
  };

  const undoLastLabelOp = () => {
    setLabelHistory((h) => {
      if (h.length === 0) return h;
      const [prev, ...rest] = h;
      setEditLabels(prev.labels);
      setLabelAliases(prev.aliases);
      updateRefs(prev.refs);
      return rest;
    });
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const meta = e.ctrlKey || e.metaKey;
      if (!meta || e.key.toLowerCase() !== "z" || e.shiftKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (labelHistory.length === 0) return;
      e.preventDefault();
      undoLastLabelOp();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labelHistory]);

  useEffect(() => {
    if (adding) addInputRef.current?.focus();
  }, [adding]);

  // Active section. The SHELL owns this state (the Explorer tree's
  // third-level rows are the nav); the view just renders it and routes
  // its internal section jumps through the setter prop. Read-only
  // public views have no Overview/Augmentations sections, so those
  // section values degrade to Dataset there (the historical landing).
  const tab: ProjectTab =
    readOnly && (section === "overview" || section === "augmentations")
      ? "dataset"
      : section;
  const setTab = useCallback(
    (next: ProjectTab) => onSectionChange?.(next),
    [onSectionChange],
  );
  // Swapping sections always returns to the top of the page. The
  // dataset view scrolls inside the shell's content-area overlay
  // (data-dataset-scroll), not the window — fall back to the window
  // for any standalone mount.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const scroller = document.querySelector<HTMLElement>("[data-dataset-scroll]");
    if (scroller) scroller.scrollTo({ top: 0 });
    else window.scrollTo({ top: 0 });
  }, [tab]);

  // Whenever the user lands on the dataset tab, force a one-shot
  // overview refresh so any augmentations generated while they were
  // elsewhere paint their icon immediately, without depending on
  // the active-job poll catching the running→done transition.
  useEffect(() => {
    if (tab !== "dataset") return;
    refreshAugCountsRef.current?.();
  }, [tab]);

  const [inputSize, setInputSize] = useState<string>("256x256");
  const [shapeHovered, setShapeHovered] = useState(false);
  // Always starts collapsed on project load. User explicitly
  // requested this, opening the drawer should be a deliberate
  // click, not the default. No auto-open effects either: refs
  // hydrating in the background must not flip the drawer open.

  // SAM3 knobs sent on every label-charlie job. Defaults match the
  // backend's SAM3_* module-level values, i.e. opening the card and
  // pressing Start without touching anything yields the same run the
  // backend would do with no params. The Reset button restores these.
  const SAM3_DEFAULTS = {
    threshold: 0.5,        // SAM3_THRESHOLD
    maskThreshold: 0.5,    // SAM3_MASK_THRESHOLD
    minRelativeArea: 0.05, // SAM3_MIN_RELATIVE_AREA (5 %)
  } as const;
  const [sam3Threshold, setSam3Threshold] = useState<number>(SAM3_DEFAULTS.threshold);
  const [sam3MaskThreshold, setSam3MaskThreshold] = useState<number>(SAM3_DEFAULTS.maskThreshold);
  const [sam3MinRelativeArea, setSam3MinRelativeArea] = useState<number>(SAM3_DEFAULTS.minRelativeArea);
  // Native-resolution tiling for large images. Off = classic single pass
  // (SAM3 downscales every frame to its 1500px inference size — small
  // objects on 4K frames shrink below detectability). On = the backend
  // slices big frames into native-resolution tiles and merges. Heavier,
  // so it's an explicit opt-in; tile size stays at the backend's native
  // default (the only value where per-tile resize is a no-op).
  const [tileNative, setTileNative] = useState<boolean>(false);
  const sam3IsDefault =
    sam3Threshold === SAM3_DEFAULTS.threshold
    && sam3MaskThreshold === SAM3_DEFAULTS.maskThreshold
    && sam3MinRelativeArea === SAM3_DEFAULTS.minRelativeArea
    && !tileNative;
  // The detection settings the dataset was last labelled with. Seeded to
  // the slider mount values (defaults) and re-snapshotted whenever a
  // labelling job starts, so `settingsChanged` is true exactly when the
  // user has moved a slider since the last run. That's the signal that a
  // full relabel is worth offering on an already-labelled dataset (where
  // there are no new labels and nothing unlabelled).
  const [lastRunSettings, setLastRunSettings] = useState<{
    threshold: number;
    maskThreshold: number;
    minRelativeArea: number;
    tileNative: boolean;
  }>({
    threshold: SAM3_DEFAULTS.threshold,
    maskThreshold: SAM3_DEFAULTS.maskThreshold,
    minRelativeArea: SAM3_DEFAULTS.minRelativeArea,
    tileNative: false,
  });
  const settingsChanged =
    labelledImportCount > 0
    && (
      sam3Threshold !== lastRunSettings.threshold
      || sam3MaskThreshold !== lastRunSettings.maskThreshold
      || sam3MinRelativeArea !== lastRunSettings.minRelativeArea
      || tileNative !== lastRunSettings.tileNative
    );
  const resetSam3Defaults = () => {
    setSam3Threshold(SAM3_DEFAULTS.threshold);
    setSam3MaskThreshold(SAM3_DEFAULTS.maskThreshold);
    setSam3MinRelativeArea(SAM3_DEFAULTS.minRelativeArea);
    setTileNative(false);
  };
  // Annotations card collapses by default, the user only opens it
  // when they want to tweak labels / SAM3 settings.
  const [clearAllOpen, setClearAllOpen] = useState(false);

  // Auto-derived dataset behaviour, computed and cached server-side in
  // projects/<id>/dataset_type.json (references present → "specific",
  // none → "general"). The label pipeline consumes it (centroid vs kNN
  // scoring); the UI only reads it to steer internal branches — there
  // is no user-facing control or badge for it any more.
  const [datasetType, setDatasetType] = useState<DatasetTypeValue | null>(() => {
    // Seed from cache so the value is present on first paint instead
    // of after a network round-trip to /dataset-type. The fetch below
    // still runs and updates the cache when the cache is empty.
    if (!projectId) return null;
    return readProjectMeta(projectId)?.datasetType ?? null;
  });
  // Fetch dataset-type ONCE per projectId mount (or when the cache
  // is empty). The engine keeps it up to date automatically as
  // references are added/removed; the refs-length effect below
  // mirrors that flip locally without a refetch.
  useEffect(() => {
    if (!projectId) return;
    if (datasetType) return; // cache-seeded → trust it
    if (editLabels.length === 0) { setDatasetType(null); return; }
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API}/api/v2/projects/${projectId}/dataset-type`);
        if (!r.ok) return;
        const data = await r.json();
        if (cancelled) return;
        const t = data?.type === "specific" ? "specific" : "general";
        const next: DatasetTypeValue = {
          type: t,
          reason: String(data?.reason ?? ""),
          source: typeof data?.source === "string" ? data.source : "auto",
        };
        setDatasetType(next);
        patchProjectMeta(projectId, { datasetType: next });
      } catch {
        if (!cancelled) setDatasetType(null);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Adding reference images flips a dataset to "specific" behaviour
  // (the engine derives this automatically from the presence of
  // references). Mirror that flip optimistically so internal branches
  // update the moment a ref lands, without a refetch. Fire ONLY when
  // references are actually ADDED (refs.length grows), not on every
  // datasetType change.
  const prevRefsLenRef = useRef(0);
  useEffect(() => {
    const grew = refs.length > prevRefsLenRef.current;
    prevRefsLenRef.current = refs.length;
    if (!projectId || readOnly) return;
    if (!grew || refs.length === 0) return;
    if (!datasetType) return;
    if (datasetType.type === "specific") return;
    const next: DatasetTypeValue = {
      type: "specific",
      reason: "Reference images present.",
      source: "references",
    };
    setDatasetType(next);
    patchProjectMeta(projectId, { datasetType: next });
  }, [refs.length, datasetType, projectId, readOnly]);

  const dateLabel = useMemo(
    () => new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric" }).format(new Date()),
    [],
  );

  // Subtle entry animation: opacity + a small upward translate on
  // every section. Keyed off a one-frame state flip so the initial
  // render paints the "before" state, the next frame swaps to
  // "after", and the CSS transition fills the gap. No blur, that
  // earlier version made the dataset feel like it was loading
  // when it wasn't.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(false);
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, [projectId, tab]);

  // Force scroll-to-top on project open. Without this the
  // workspace's scroll offset is preserved across the SPA route
  // change (app/page.tsx swaps components without unmounting the
  // <main> the page lives in), so opening a project from low on
  // the workspace list lands the user mid-page, typically inside
  // the imports drop zone with the project header off-screen.
  // `auto` (instant) instead of "smooth" so the user doesn't see
  // the page animate down. Targets the shell's content-area overlay
  // scroller when present (the window as a standalone fallback).
  useEffect(() => {
    if (typeof window === "undefined") return;
    const scroller = document.querySelector<HTMLElement>("[data-dataset-scroll]");
    if (scroller) scroller.scrollTo({ top: 0, behavior: "auto" });
    else window.scrollTo({ top: 0, behavior: "auto" });
  }, [projectId]);

  const fade = (delay = 0): CSSProperties => ({
    opacity: mounted ? 1 : 0,
    transition: `opacity 320ms ease-out ${delay}ms`,
  });
  const rise = (delay = 0): CSSProperties => ({
    opacity: mounted ? 1 : 0,
    transform: mounted ? "translateY(0)" : "translateY(8px)",
    transition: `opacity 320ms ease-out ${delay}ms, transform 360ms cubic-bezier(0.2, 0.7, 0.2, 1) ${delay}ms`,
  });

  // Page-ready gate. The user reported that on a hard refresh the
  // loader faded as soon as /initial returned, then the page sat
  // empty for "a few seconds" while the dataset-stats card and the
  // label chip rail finished populating. On a workspace-nav open
  // both render instantly because the projectMetaCache seeded
  // labels + seedStats + ref/import tiles BEFORE the project view
  // even mounted, so /initial just confirms the cached values and
  // there's no perceptible gap.
  //
  // We bridge the refresh path by holding the loader up until ALL
  // of the first-paint slots have actually landed:
  //   • importsReady, first-batch imports merged into state
  //   • seedStats   , stats card has its lite snapshot
  //   • editLabels  , chip rail has the label canonicals (skip the
  //                    check on empty-label projects, where [] is
  //                    legitimate)
  const seedStatsReady = seedStats !== null;
  const labelsReady = editLabels.length > 0 || labels.length === 0;
  const pageReady = importsReady && seedStatsReady && labelsReady;
  // Loader now runs for public viewers too (they were getting a
  // half-painted page on cold loads). Only suppressed once the page
  // has actually settled.
  const [loaderVisible, setLoaderVisible] = useState<boolean>(!pageReady);
  useEffect(() => {
    if (pageReady) setLoaderVisible(false);
  }, [pageReady]);
  // Hard cap so a flaky network can't pin the loader forever. 6 s
  // gives the cold-sidecar compute path (3-5 s on a 964-image
  // project on first visit after deploy) enough headroom while
  // still bailing well before the user gives up.
  useEffect(() => {
    if (!loaderVisible) return;
    const t = window.setTimeout(() => setLoaderVisible(false), 6000);
    return () => window.clearTimeout(t);
  }, [loaderVisible]);

  // First-load loader variant.
  //   "onboarding" → the user is arriving here straight out of the
  //     create flow. HomeView's "Opening project…" overlay carried
  //     the transition, so no second full-screen takeover here.
  //   null → every other entry (workspace cards, projects feed,
  //     deep-link, refresh). Full-screen blurred mount loader, the
  //     existing behaviour.
  const isFirstLoad = firstLoad === "onboarding";
  const showFullscreenLoader = loaderVisible && !isFirstLoad;

  // Auto-derived behaviour: a dataset with references acts as
  // "specific" (reference scoring). Reads may lag the fetch, so a
  // project that already has refs counts as specific immediately.
  const isSpecific = datasetType?.type === "specific" || refs.length > 0;

  // The view has NO nav column of its own any more — the shell's
  // Explorer tree owns section navigation (third-level rows). Only the
  // mount loader's bail-out button still needs the "Back to …" copy.
  const backTo = backToProjectId
    ? "project"
    : (readOnly || originTab === "projects") ? "projects" : "workspace";

  return (
    // The dataset view fills the shell's content-area overlay (right of
    // the Explorer side bar, between title and status bars). Navigation
    // lives in the shell's tree, so the content takes the full width.
    <main className="min-h-screen bg-[var(--background)]">
      {showFullscreenLoader && (
        // Mount loader covers the CONTENT AREA only (below the 36px
        // title bar, above the 24px status bar, right of the shell
        // side bar via --pk-content-left) — the tree stays reachable
        // while a cold load spins.
        <div
          className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[900] grid place-items-center"
          style={{
            background: "rgb(var(--background-rgb) / 0.92)",
            opacity: pageReady ? 0 : 1,
            transition: "opacity 240ms ease",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
          }}
        >
          {/* Bail-out button, closes the project overlay so the user
              can go back to the workspace if a cold-load is taking
              too long. Skipped when nothing to close (top-level
              `onClose` would no-op). */}
          {onClose && (
            <button
              type="button"
              onClick={() => {
                setLoaderVisible(false);
                onClose();
              }}
              aria-label={`Back to ${backTo}`}
              className="absolute top-4 right-4 rounded-full border border-foreground/15 bg-[var(--surface)]/85 hover:bg-foreground/[0.06] px-3 py-1.5 text-[11px] uppercase tracking-wider font-mono text-foreground/75 hover:text-foreground transition-colors"
              style={{ boxShadow: "var(--shadow-strong)" }}
            >
              ← Back to {backTo}
            </button>
          )}
          <PixelKitLoader size={120} message="Loading project…" />
        </div>
      )}
      {/* Content block. Full width — section navigation lives in the
          shell's Explorer tree, so the old fixed nav column (and its
          mobile drawer) are gone and the content reclaims the space. */}
      <div className="w-full min-w-0 max-w-[2000px]">
      {/* Title section as a cover hero: content overlaid on the dataset cover
          with a content-aware left gradient-blur (white scrim + dark text on a
          light cover, dark scrim + white text on a dark one). */}
      <section className="px-6 lg:px-10 pt-6 pb-2">
        <div className="relative overflow-hidden rounded-3xl border border-foreground/10 pk-cover">
          {projectId && !bannerCoverFailed && (
            <>
              {/* Visible cover — NO crossOrigin so a CORS hiccup never blanks it.
                  Requests the 1280 px hero variant: the fast Lanczos render
                  first, then swaps to the GPU AI-upscaled variant once the
                  hidden preloader below confirms it's baked + loadable. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={aiCoverReady
                  ? `${API}/api/projects/${projectId}/cover_thumb?w=1280&ai=1&v=${coverBust}`
                  : `${API}/api/projects/${projectId}/cover_thumb?w=1280&v=${coverBust}&r=${bannerRetry}`}
                alt=""
                onError={() => {
                  // The AI variant is only shown once the preloader verified it
                  // loads, so an error here is the Lanczos render: backed-off
                  // retry before giving up to the brand wash, so a transient
                  // lazy-render miss doesn't leave the banner blank.
                  if (aiCoverReady) return;
                  if (bannerRetry >= 3) { setBannerCoverFailed(true); return; }
                  const n = bannerRetry;
                  window.setTimeout(() => setBannerRetry(n + 1), 400 * Math.pow(2, n));
                }}
                className="absolute inset-0 h-full w-full object-cover"
              />
              {/* Hidden preloader for the GPU AI-upscaled hero. When it lands we
                  flip to it (already cached, so the swap is seamless). If the GPU
                  / model is unavailable the server falls back to Lanczos for this
                  URL too, so this still resolves — just to the same bytes. */}
              {!aiCoverReady && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`${API}/api/projects/${projectId}/cover_thumb?w=1280&ai=1&v=${coverBust}`}
                  alt=""
                  aria-hidden
                  className="hidden"
                  onLoad={() => setAiCoverReady(true)}
                  onError={() => { /* AI variant unavailable — stay on Lanczos */ }}
                />
              )}
              {/* Hidden crossOrigin copy used only to sample the cover's left
                  luminance for the content-aware text/scrim. Best-effort: if it
                  can't be read, we keep the default dark-scrim + white text. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API}/api/projects/${projectId}/cover_thumb?v=${coverBust}`}
                alt=""
                aria-hidden
                crossOrigin="anonymous"
                onLoad={(e) => {
                  try {
                    const img = e.currentTarget;
                    const cv = document.createElement("canvas");
                    cv.width = 24;
                    cv.height = 24;
                    const ctx = cv.getContext("2d");
                    if (!ctx) return;
                    ctx.drawImage(img, 0, 0, Math.max(1, Math.round(img.naturalWidth * 0.5)), img.naturalHeight, 0, 0, 24, 24);
                    const d = ctx.getImageData(0, 0, 24, 24).data;
                    let s = 0;
                    for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
                    setBannerLight(s / (d.length / 4) > 150);
                  } catch {
                    setBannerLight(false);
                  }
                }}
                className="hidden"
              />
            </>
          )}
          {/* Gradient blur over the left, where the info sits. */}
          <div
            className="absolute inset-0 backdrop-blur-md"
            style={{
              maskImage: "linear-gradient(to right, black 28%, transparent 72%)",
              WebkitMaskImage: "linear-gradient(to right, black 28%, transparent 72%)",
            }}
            aria-hidden
          />
          <div
            className={`absolute inset-0 bg-gradient-to-r ${
              bannerLight ? "from-white/85 via-white/45 to-transparent" : "from-black/80 via-black/45 to-transparent"
            }`}
            aria-hidden
          />
          {/* Content over the cover. Desktop-tool scale: compact hero. */}
          <div className="relative flex min-h-[11rem] flex-wrap items-end justify-between gap-6 p-5 sm:p-6">
            <div className="min-w-0 flex-1">
          {titleEditing && !readOnly ? (
            <input
              ref={titleInputRef}
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") { e.preventDefault(); commitTitleRename(); }
                if (e.key === "Escape") { setTitleEditing(false); }
              }}
              onBlur={() => setTitleEditing(false)}
              aria-label="Rename project"
              className="text-3xl md:text-4xl font-medium tracking-tight leading-[1.3] bg-transparent outline-none border-b border-foreground/30 focus:border-foreground/60 text-[var(--foreground)] pb-3 w-full"
            />
          ) : (
            <div className="flex items-center gap-4 pb-3" style={fade()}>
              <h1
                // pb-2 keeps the descenders (g, y, j, p, q) inside the
                // padding box so overflow-hidden (needed for the
                // text-ellipsis trio) doesn't crop them on light or
                // wide-descender weights. The rename + scale hover
                // affordance only fires when the viewer owns the
                // project; on public read-only view the title is a
                // static label with no cursor / hover surface.
                className={[
                  "text-3xl md:text-4xl font-medium tracking-tight leading-[1.3] pb-2 max-w-full overflow-hidden text-ellipsis whitespace-nowrap w-fit drop-shadow-sm",
                  bannerLight ? "text-zinc-900" : "text-white",
                  readOnly
                    ? "cursor-default"
                    : "cursor-text origin-bottom-left transition-transform duration-200 ease-out hover:scale-[1.02] hover:underline underline-offset-8 decoration-[0.5px] decoration-white/35",
                ].join(" ")}
                onClick={readOnly ? undefined : () => setTitleEditing(true)}
                title={readOnly ? undefined : "Click to rename"}
              >
                {projectTitle}
              </h1>
              {derivedInfo && (
                <a
                  href={derivedInfo.parentProjectId ? `/app/${derivedInfo.parentProjectId}` : undefined}
                  className="shrink-0 text-sky-600 dark:text-sky-400/90 transition-opacity hover:opacity-70"
                  title={derivedInfo.parentName ? `Derived from ${derivedInfo.parentName} - open parent` : "Derived project"}
                  aria-label="Derived project"
                >
                  <svg viewBox="0 0 16 16" className="h-9 w-9" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" role="img">
                    <title>{derivedInfo.parentName ? `Derived from ${derivedInfo.parentName}` : "Derived project"}</title>
                    <circle cx="4" cy="3.2" r="1.7" />
                    <circle cx="4" cy="12.8" r="1.7" />
                    <circle cx="12" cy="12.8" r="1.7" />
                    <path d="M4 4.9V11.1M5.7 12.8H10.3" />
                  </svg>
                </a>
              )}
            </div>
          )}

          {/* Owner + meta chips, project-page style + content-aware over the cover. */}
          <div className="mt-3 flex flex-col gap-2.5" style={fade()}>
            <div className={`flex items-center gap-2 text-sm ${bannerLight ? "text-zinc-800/85" : "text-white/85"}`}>
              {(() => {
                // Show the OWNER's avatar (mine only when it's my dataset), so
                // a teammate's dataset never shows my picture beside their name.
                const avatar = isOwnDataset ? userImage : ownerImage;
                return avatar ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={avatar} alt="" className="h-5 w-5 rounded-full object-cover shrink-0" referrerPolicy="no-referrer" />
                ) : (
                  <span
                    className="h-5 w-5 rounded-full grid place-items-center text-[9px] font-semibold text-white shrink-0"
                    style={{ backgroundImage: `linear-gradient(135deg, hsl(${hueFor(displayHandle)},70%,55%), hsl(${(hueFor(displayHandle) + 60) % 360},70%,55%))` }}
                  >
                    {(displayHandle || "?").charAt(0).toUpperCase()}
                  </span>
                );
              })()}
              <span className={bannerLight ? "font-medium text-zinc-900" : "font-medium text-white"}>@{displayHandle}</span>
              {projectId && (
                <>
                  <span aria-hidden className={bannerLight ? "text-zinc-500/60" : "text-white/40"}>·</span>
                  <span
                    className={`font-mono text-[10px] uppercase tracking-wider ${bannerLight ? "text-zinc-700/70" : "text-white/55"}`}
                    title={`Project ID: ${projectId}`}
                  >
                    {projectId.slice(0, 8)}
                  </span>
                </>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={[
                  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur-md ring-1",
                  isPrivate
                    ? bannerLight
                      ? "bg-orange-500/20 text-orange-900 ring-orange-500/50 shadow-[0_0_16px_rgba(249,115,22,0.45)]"
                      : "bg-orange-500/25 text-orange-50 ring-orange-400/60 shadow-[0_0_18px_rgba(249,115,22,0.6)]"
                    : bannerLight
                      ? "bg-black/10 text-zinc-900 ring-black/10"
                      : "bg-white/15 text-white ring-white/10",
                ].join(" ")}
              >
                {isPrivate ? (
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18" /></svg>
                )}
                {isPrivate ? "Private" : "Public"}
              </span>
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur-md ring-1 ${bannerLight ? "bg-black/10 text-zinc-900 ring-black/10" : "bg-white/15 text-white ring-white/10"}`}>
                Updated {dateLabel}
              </span>
              {/* Image size limit: the max upload resolution (longest edge),
                  inherited from the Project. */}
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur-md ring-1 ${bannerLight ? "bg-black/10 text-zinc-900 ring-black/10" : "bg-white/15 text-white ring-white/10"}`}
                title={`Image size limit — uploads are kept up to ${maxInputSize}px on the longest edge (set by the Project).`}
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M15 3h6v6M21 3l-7 7M9 21H3v-6M3 21l7-7" />
                </svg>
                {maxInputSize.toLocaleString()}px
              </span>
            </div>
          </div>

          <div
            className="mt-4 flex flex-wrap items-center gap-2"
            style={{
              ...fade(),
              position: "relative",
              zIndex: adding ? 60 : undefined,
            }}
          >
            {editLabels.map((lab, i) => {
              const bg = colourForLabel(editLabels, lab, labelColours);
              return readOnly ? (
                <span
                  key={lab}
                  className="inline-flex items-center rounded-full pl-3 pr-3 h-7 text-sm font-medium"
                  style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                >
                  {displayLabel(lab)}
                </span>
              ) : (
                <EditableChip
                  key={lab}
                  label={displayLabel(lab)}
                  colour={bg}
                  onDelete={() => removeLabel(i)}
                  onRename={(next) => renameLabel(i, next)}
                />
              );
            })}
            {readOnly ? null : adding ? (
              <input
                ref={addInputRef}
                value={addInput}
                onChange={(e) => setAddInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") { e.preventDefault(); addLabel(addInput); setAdding(false); }
                  else if (e.key === "Escape") { setAddInput(""); setAdding(false); }
                }}
                onBlur={() => { if (addInput.trim()) addLabel(addInput); setAdding(false); }}
                placeholder="new label"
                className="rounded-full bg-foreground/[0.04] border border-foreground/15 focus:border-foreground/35 focus:bg-foreground/[0.06] outline-none px-3 py-1 text-sm text-[var(--foreground)] placeholder:text-foreground/35 transition-colors"
                style={{ width: "9rem", position: "relative", zIndex: 70 }}
              />
            ) : (
              <button
                type="button"
                onClick={() => setAdding(true)}
                aria-label="Add label"
                title="Add label"
                className={`inline-flex items-center justify-center h-6 w-6 rounded-full transition-colors ${bannerLight ? "border-zinc-900/20 text-zinc-700 hover:text-zinc-900 hover:border-zinc-900/40" : "border-white/30 text-white/70 hover:text-white hover:border-white/50"}`}
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            )}
          </div>

        </div>

        {/* Right column stretches the hero's height so the derived badge
            sits in the top-right corner and the action buttons hold the
            bottom-right. */}
        <div className="flex flex-col items-end justify-between gap-3 self-stretch shrink-0" style={fade()}>
          <div className="flex items-center">
            {derivedInfo && (
              <span
                className={`inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-[11px] font-medium uppercase tracking-wider backdrop-blur-md ring-1 ${bannerLight ? "bg-black/10 text-zinc-900 ring-black/10" : "bg-white/15 text-white ring-white/10"}`}
                title="A cropped child dataset derived from a parent project"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" aria-hidden />
                Derived
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
          {!readOnly && (
            <button
              type="button"
              // Dataset Settings (rename / cover / label colours / delete) are
              // owner-only. An editor on a teammate's dataset can edit content
              // but the Settings button is greyed out + inert.
              disabled={!isOwnDataset}
              onClick={() => { if (projectId && isOwnDataset) setSettingsOpen(true); }}
              className={[
                "inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-semibold backdrop-blur-md shadow-sm transition",
                bannerLight ? "bg-black/80 text-white" : "bg-white/90 text-black",
                isOwnDataset
                  ? bannerLight ? "hover:bg-black/90" : "hover:bg-white"
                  : "cursor-not-allowed opacity-40",
              ].join(" ")}
              title={isOwnDataset ? "Dataset settings" : "Only the dataset's creator can change its settings"}
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
              Settings
            </button>
          )}
          <button
            type="button"
            onClick={() => { if (projectId) setExportOpen(true); }}
            className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-semibold backdrop-blur-md shadow-sm transition ${bannerLight ? "bg-black/80 text-white hover:bg-black/90" : "bg-white/90 text-black hover:bg-white"}`}
            title="Export dataset"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export
          </button>
          </div>
          </div>
          </div>
        </div>
      </section>

      {/* Stat strip — the single dataset-stats row, in the Project-page card
          style. Carries the richer counts (detections / augmentations / a
          clickable health score) that used to live in a second, differently
          styled grid inside OverviewPanel; that duplicate is gone. Numbers come
          from the lite stats snapshot, falling back to summing loaded tiles. */}
      <section className="px-6 lg:px-10 pt-4" style={rise()}>
        {(() => {
          const ss = seedStats as
            | { counts?: { detections?: number; augmentations?: number }; health?: { score?: number } }
            | null;
          const detections = ss?.counts?.detections ?? imports.reduce((a, i) => a + (i.detectionCount ?? 0), 0);
          const augmentations = ss?.counts?.augmentations ?? imports.reduce((a, i) => a + (i.nAugmentations ?? 0), 0);
          const score = ss?.health?.score;
          const healthTone = typeof score === "number" ? (score >= 75 ? "Good" : score >= 45 ? "Fair" : "Needs work") : "";
          const items = [
            { label: "Images", value: importsTotal != null ? importsTotal : imports.length },
            { label: "Detections", value: detections },
            { label: "Labels", value: editLabels.length },
            { label: "Augmentations", value: augmentations },
          ];
          return (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {items.map((s) => (
                <div key={s.label} className="pk-card rounded-2xl px-5 py-5">
                  <div className="text-[2rem] font-bold leading-none tracking-tight tabular-nums text-[var(--foreground)]">
                    {s.value.toLocaleString()}
                  </div>
                  <div className="pk-eyebrow mt-2">{s.label}</div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => setHealthOpen(true)}
                title="Open dataset health"
                className="pk-card pk-card-hover rounded-2xl px-5 py-5 text-left"
              >
                <div className="text-[2rem] font-bold leading-none tracking-tight tabular-nums text-[var(--foreground)]">
                  {typeof score === "number" ? Math.round(score) : "—"}
                </div>
                <div className="pk-eyebrow mt-2" style={{ color: "var(--accent-orange)" }}>
                  {healthTone ? `Health · ${healthTone}` : "Health"}
                </div>
              </button>
            </div>
          );
        })()}
      </section>

      {/* Target input shape */}
      <section className="px-6 lg:px-10 mt-1 pb-6" style={rise()}>
        <div className="flex items-center gap-4 flex-wrap">
          <div
            className="flex items-center gap-3"
            onMouseEnter={() => setShapeHovered(true)}
            onMouseLeave={() => setShapeHovered(false)}
          >
            <span className="text-sm text-foreground/55 shrink-0">Target input shape</span>
            <div
              className="overflow-hidden grid items-center"
              style={{
                gridTemplateAreas: '"stack"',
                maxWidth: shapeHovered ? "720px" : "92px",
                transition: shapeHovered
                  ? "max-width 360ms cubic-bezier(0.25, 0.46, 0.45, 0.94)"
                  : "max-width 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
              }}
            >
              <button
                type="button"
                onClick={() => setShapeHovered(true)}
                style={{
                  gridArea: "stack",
                  opacity: shapeHovered ? 0 : 1,
                  pointerEvents: shapeHovered ? "none" : "auto",
                  transition: shapeHovered
                    ? "opacity 160ms ease-out"
                    : "opacity 260ms ease-out",
                }}
                className="appearance-none bg-transparent border-0 p-0 text-left cursor-pointer"
                aria-label="Expand input-shape picker"
              >
                <span className="inline-flex items-center font-mono text-xs text-foreground/85 px-3 py-1 rounded-full border border-foreground/15 bg-foreground/[0.04] hover:bg-foreground/[0.06] hover:border-foreground/25 whitespace-nowrap transition-colors">
                  {inputSize}
                </span>
              </button>
              <div
                style={{
                  gridArea: "stack",
                  opacity: shapeHovered ? 1 : 0,
                  pointerEvents: shapeHovered ? "auto" : "none",
                  transition: shapeHovered
                    ? "opacity 220ms ease-out 120ms"
                    : "opacity 200ms ease-out",
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
          <ShapeHelpPopover />
        </div>
      </section>

      {/* Section navigation lives in the shell's Explorer tree - no
          in-view nav column or tab strip. */}

      {/* Overview tab: compact stats + recent images / label distribution /
          recent activity, then the Derived datasets strip. Mounted only while
          this tab is active (it is cheap and reuses already-loaded data). */}
      {!readOnly && tab === "overview" && (
        <div className="pk-up">
          <OverviewPanel
            imports={imports}
            importsTotal={importsTotal}
            labels={editLabels}
            labelColours={labelColours}
            projectId={projectId}
            refreshSignal={statsRefreshSignal}
            refs={refs}
            showReferences={!derivedInfo}
            seedStats={seedStats as Parameters<typeof OverviewPanel>[0]["seedStats"]}
            onOpenHealth={() => setHealthOpen(true)}
            onOpenReferences={() => setTab("references")}
            onOpenDataset={() => setTab("dataset")}
            onJumpToImport={(id) => { setTab("dataset"); jumpToImport(id, "image"); }}
          />
          {projectId && <DerivedDatasetsBar projectId={projectId} labels={editLabels} />}
        </div>
      )}
      {/* Dataset health, opened from the Overview health card: one modal with
          health on the left and the (inline, expanded) duplicates review on
          the right - no pop-up within a pop-up. */}
      {!readOnly && (
        <DatasetHealthModal
          open={healthOpen}
          onClose={() => setHealthOpen(false)}
          projectId={projectId}
          labelColours={labelColours}
          refreshSignal={statsRefreshSignal}
        />
      )}

      {/* ─── Dataset tab content ───
          References → uploads → annotations → labelling job →
          dataset gallery. All hidden when the user switches to
          another tab; the JSX stays mounted so existing animations
          + scroll position survive the tab toggle. */}
      <div hidden={tab !== "dataset"} className="pk-up">
      {/* Public read-only credit line: discreet "Powered by PixelKit"
          attribution at the top of the dataset tab so visitors see
          how the curator's annotations were produced. Skipped on the
          owner view to keep their page clean. */}
      {readOnly && (
        <section className="px-6 lg:px-10 pt-3 -mb-1">
          <span className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] font-mono text-foreground/55">
            <span>Powered by PixelKit</span>
          </span>
        </section>
      )}
      </div>
      {/* Reference images: now their own sidebar section. Same collapsible grid.
          (The dataset-stats card + the derived-datasets bar moved to the
          Overview; health/embeddings live in the Overview health modal.) The
          References block is gated to its own tab by closing the dataset <div>
          above and reopening it below, so the Dataset tab is just the gallery
          flow while References keep all their in-component state/handlers. */}
      <div hidden={tab !== "references"} className="pk-up">
      {/* References. Rendered unconditionally and WITHOUT the fade()
          opacity gate, the user expects this section visible on
          first paint, not after the 30ms mount tick + 420ms fade
          that the rest of the page chrome uses for its entrance
          animation. Drawer is default-open (refsOpen = true) so
          the placeholder grid is up immediately and refs slot in
          as their bytes arrive. */}
      <section className="px-6 lg:px-10 pt-3 pb-3">
          <div className="pk-card rounded-2xl px-5 py-3">
          {/* Always-expanded heading (no dropdown): the references page shows
              all of its info directly. */}
          <div className="inline-flex items-center gap-3 text-2xl font-medium tracking-tight text-[var(--foreground)] min-h-[2.5rem]">
            <span className="pk-accent-bar" style={{ height: "1.4rem" }} aria-hidden />
            Reference images
            <span className="ml-1 text-xs font-normal text-foreground/55">
              {refs.length}
              {expectedRefCount > refs.length ? ` / ${expectedRefCount}` : ""}
            </span>
          </div>

          <div>
            <div>
              <p className="mt-1 text-sm text-foreground/65 leading-relaxed">
                Reference images (optional) — add examples to improve label
                matching, especially when classes look similar.
              </p>
              <p className="mt-1 text-sm text-foreground/55">
                Drop each photo into its label section below. The section is the label, so PixelKit knows exactly what each reference shows.
              </p>

              {/* Per-label annotation status chips (annotation count is
                  backend-driven, placeholder at 0/5 for now). */}
              {editLabels.length > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  {editLabels.map((lab) => {
                    // Case + whitespace insensitive compare. cleanLabel
                    // and the open-vocab detector can produce labels that drift
                    // by case ("Dog" vs "dog") or padding, which
                    // strict === would miss and undercount the chip.
                    const labKey = lab.trim().toLowerCase();
                    const annotCount = refs.reduce(
                      (sum, ri) =>
                        sum +
                        (ri.boxes ?? []).filter(
                          (b) => (b.label ?? "").trim().toLowerCase() === labKey,
                        ).length,
                      0,
                    );
                    const annotDone = annotCount >= ANNOTS_PER_LABEL;
                    const bg = colourForLabel(editLabels, lab, labelColours);
                    return (
                      <span
                        key={lab}
                        className="inline-flex items-center gap-1.5 rounded-full pl-3 pr-2.5 h-7 text-sm font-medium"
                        style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                      >
                        {displayLabel(lab)}
                        {annotDone ? (
                          <svg viewBox="0 0 24 24" className="h-3 w-3 opacity-50 shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-label="Complete">
                            <polyline points="5 12 10 17 19 7" />
                          </svg>
                        ) : (
                          <span className="text-[10px] opacity-60 tabular-nums">{annotCount}/{ANNOTS_PER_LABEL}</span>
                        )}
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Per-label reference sections */}
              <RefImageGrid
                refs={refs}
                onChange={updateRefs}
                projectId={projectId}
                labels={editLabels}
                labelColours={labelColours}
                pendingCount={Math.max(0, expectedRefCount - refs.length)}
                onLeaveImage={flushReferenceEmbeddings}
                refQuality={refQuality}
                readOnly={readOnly}
                onViewerOpenChange={setRefViewerOpen}
                onRefDeleted={(refId) => {
                  if (projectId) {
                    apiFetch(`/api/v2/projects/${projectId}/references/${refId}`, { method: "DELETE" }).catch(() => {});
                  }
                  setExpectedRefCount((c) => Math.max(0, c - 1));
                }}
              />
            </div>
          </div>
          </div>
        </section>

      {/* Reference upload progress banner. Visible only while a
          POST is genuinely in flight, `unstarted` doesn't count
          because hydrated refs and just-added refs both pass
          through that state for one render between setRefs and
          setRefUploadStatus, which used to flash the banner on
          every project open. */}
      {refUploadCounts.uploading > 0 && (
        <section className="px-6 lg:px-10 pt-0 pb-3" style={rise()}>
          <div className="rounded-xl border border-amber-300/30 bg-amber-300/[0.05] px-4 py-3 flex items-center gap-3">
            <span className="relative flex h-2 w-2 shrink-0" aria-hidden>
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-300/60 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-300" />
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-[12px] font-medium text-amber-100/90">
                Saving references, {refUploadCounts.done}/{refUploadCounts.total} done
                {refUploadCounts.failed > 0 ? ` · ${refUploadCounts.failed} failed` : ""}
              </div>
              <div className="text-[11px] text-amber-200/55 mt-0.5">
                Image imports and training are paused until centroids are persisted.
              </div>
            </div>
          </div>
        </section>
      )}

      </div>
      {/* Dataset tab (continued): media import, annotations, labelling + the
          gallery. */}
      <div hidden={tab !== "dataset"} className="pk-up">
      {/* Import media, owner only. Public read-only views don't get the drop
          zone or the "Don't have images?" panel. Derived (child) projects don't
          either - their images are auto-cropped from the parent, not uploaded. */}
      {!readOnly && !derivedInfo && (
      <section className="px-6 lg:px-10 pt-3 pb-3" style={rise()}>
        <ImportMediaSection
          onUpload={handleImportFiles}
          projectId={projectId}
          alreadyImportedUrls={imports
            .map((m) => m.sourceUrl)
            .filter((u): u is string => !!u)}
          onOpenverseAdded={async (added) => {
            if (added <= 0 || !projectId) return;
            // Re-fetch /overview so the freshly imported URLs show
            // up in the gallery. Merge by id, preserves any
            // detections / editedBoxes the user might have on
            // existing entries.
            try {
              const ovr = await apiFetch(`/api/v2/projects/${projectId}/overview`);
              if (!ovr.ok) return;
              const overview = (await ovr.json()) as {
                imports?: {
                  id: string;
                  filename: string;
                  originalFilename?: string;
                  width?: number;
                  height?: number;
                  blurhash?: string | null;
                  createdAt?: number | string | null;
                  source?: { kind?: string; url?: string } | null;
                  n_augmentations?: number;
                }[];
              };
              const fresh = overview.imports ?? [];
              setImports((cur) => {
                const byId = new Map(cur.filter((m) => m.backendId).map((m) => [m.backendId!, m]));
                const merged = fresh.map<ImportedMedia>((imp) => {
                  const existing = byId.get(imp.id);
                  if (existing) {
                    return {
                      ...existing,
                      width: imp.width,
                      height: imp.height,
                      blurhash: imp.blurhash ?? existing.blurhash ?? null,
                      createdAt: imp.createdAt ?? existing.createdAt ?? null,
                      sourceUrl: imp.source?.url ?? existing.sourceUrl ?? null,
                      nAugmentations: imp.n_augmentations ?? existing.nAugmentations ?? 0,
                    };
                  }
                  return {
                    id: imp.id,
                    backendId: imp.id,
                    file: new File([], imp.originalFilename || imp.filename),
                    filename: imp.filename,
                    preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
                    blurhash: imp.blurhash ?? null,
                    status: "ready" as const,
                    width: imp.width,
                    height: imp.height,
                    createdAt: imp.createdAt ?? null,
                    sourceUrl: imp.source?.url ?? null,
                    nAugmentations: imp.n_augmentations ?? 0,
                  };
                });
                // Keep in-flight uploads that haven't persisted yet.
                for (const c of cur) {
                  if (!c.backendId) merged.push(c);
                }
                return merged.sort(compareImportedMediaDesc);
              });
              scrollToDataset();
            } catch (e) {
              console.warn("[v2 imports/refresh after openverse]:", e);
            }
          }}
          disabled={!referencesSettled}
          disabledMessage={
            refUploadCounts.uploading > 0 || refUploadCounts.pending > 0
              ? `Saving ${refUploadCounts.done}/${refUploadCounts.total} references, drop available shortly`
              : undefined
          }
        />
      </section>
      )}

      {/* Bulk-import progress. Shown while a drag-drop batch is uploading
          so the user gets clear feedback on a thousands-of-images import
          (the bytes still have to travel; the bar makes the wait legible
          while the gallery fills in behind it). */}
      {!readOnly && importProgress && importProgress.total > 0 && (
        <section className="px-6 lg:px-10 pb-3">
          <div className="rounded-2xl border border-foreground/10 bg-[var(--surface)] shadow-[var(--shadow-soft)] px-5 py-4">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-medium text-[var(--foreground)]">
                {importProgress.done >= importProgress.total ? "Finishing up…" : "Uploading images"}
              </span>
              <span className="tabular-nums text-foreground/55">
                {Math.min(importProgress.done, importProgress.total)} / {importProgress.total}
              </span>
            </div>
            <div className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-foreground/10">
              <div
                className="h-full rounded-full bg-[var(--accent-orange)] transition-[width] duration-300 ease-out motion-reduce:transition-none"
                style={{
                  width: `${Math.round(
                    (Math.min(importProgress.done, importProgress.total) /
                      Math.max(1, importProgress.total)) * 100,
                  )}%`,
                }}
              />
            </div>
            <p className="mt-2 text-[11px] text-foreground/45">
              You can keep working, images appear in the gallery as they finish uploading.
            </p>
            {importProgress.total >= 500 && (
              <p className="mt-1 text-[11px] text-amber-500/90">
                Large import of {importProgress.total.toLocaleString()} images. Keep this tab open until it
                finishes, closing it stops the remaining uploads (already-uploaded images are kept, and
                re-dropping the same files later skips duplicates).
              </p>
            )}
          </div>
        </section>
      )}

      {/* Annotations: detection + auto-labelling controls. Hidden for derived
          (child) projects - their labels/boxes are cropped from the parent and
          can't be auto-labelled here. Also hidden on read-only public views. */}
      {!readOnly && !derivedInfo && (
      <>
      <section className="px-6 lg:px-10 pt-3 pb-3" style={rise()}>
        <div className="rounded-2xl bg-foreground/[0.025] px-5 py-3">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            {/* Static heading (no dropdown): the annotation controls are
                built into the page below. */}
            <div className="flex min-h-[2.5rem] items-center gap-3 flex-wrap">
              <h2 className="flex items-center gap-3 text-2xl font-medium tracking-tight text-[var(--foreground)]">
                <span className="pk-accent-bar" style={{ height: "1.4rem" }} aria-hidden />
                Annotations
              </h2>
              <span className="text-xs text-foreground/55">
                {`${editLabels.length} label${editLabels.length === 1 ? "" : "s"} · ${sam3IsDefault ? "default" : "custom"} detection settings`}
              </span>
            </div>
            {(() => {
              const running = labelJob != null && labelJob.status === "running";
              const processing = processingImportCount > 0;
              // Adaptive idle copy: one-label fresh dataset uses the
              // label's own name; multi-label fresh dataset says "all
              // images"; partial datasets (some already labelled) say
              // "new images". Falls through to "Start labelling" when
              // there are no labels yet (button stays disabled).
              const idleCopy = (() => {
                if (editLabels.length === 0) return "Start labelling";
                // Nothing left to do: every persisted image is
                // labelled and there are no fresh labels that would
                // need a re-pass. Surface that state honestly instead
                // of saying "Start labelling new images" - the button
                // is already disabled here, but the text used to lie.
                if (
                  unlabelledImportCount === 0
                  && freshLabels.length === 0
                  && labelledImportCount > 0
                ) {
                  return settingsChanged
                    ? "Relabel all with new settings"
                    : "All images labelled";
                }
                // Fresh labels (just-added, never detected) take
                // precedence over image-state copy, they describe
                // the more specific action the user just queued up.
                if (freshLabels.length === 1) {
                  return `Start labelling ${displayLabel(freshLabels[0])}`;
                }
                if (freshLabels.length > 1) {
                  return "Start labelling new labels";
                }
                if (labelledImportCount === 0 && editLabels.length === 1) {
                  return `Start labelling ${displayLabel(editLabels[0])}`;
                }
                if (labelledImportCount === 0) return "Start labelling all images";
                return "Start labelling new images";
              })();
              const buttonText = labelJobStarting
                ? "Starting…"
                : running
                ? "Labelling in progress"
                : processing
                ? "Processing images…"
                : idleCopy;
              const buttonKey = labelJobStarting
                ? "starting"
                : running
                ? "running"
                : processing
                ? "processing"
                : `idle:${idleCopy}`;
              return (
                <AnimatedStartButton
                  text={buttonText}
                  buttonKey={buttonKey}
                  running={running}
                  onClick={startLabellingJob}
                  disabled={
                    labelJobStarting
                    || running
                    || processing
                    || editLabels.length === 0
                    // Button stays enabled when there's at least one
                    // unlabelled image OR a freshly-added label that
                    // needs an across-the-board pass. Without the
                    // freshLabels branch a user who added a tag to a
                    // fully-labelled project saw the button copy
                    // change but the button stayed greyed out.
                    || (unlabelledImportCount === 0 && freshLabels.length === 0 && !settingsChanged)
                  }
                  title={
                    editLabels.length === 0
                      ? "Add labels first, labelling needs at least one tag."
                      : processing
                      ? `Wait for ${processingImportCount} image${processingImportCount === 1 ? "" : "s"} to finish processing before labelling.`
                      : freshLabels.length > 0 && unlabelledImportCount === 0
                      ? `Re-label every image to find the new label${freshLabels.length === 1 ? "" : "s"}.`
                      : settingsChanged && unlabelledImportCount === 0
                      ? "Re-label every image with the new detection settings."
                      : unlabelledImportCount === 0
                      ? "No unlabelled images. Drop some onto the dataset."
                      : running
                      ? "A labelling job is already running."
                      : `Start labelling ${unlabelledImportCount} unlabelled image${unlabelledImportCount === 1 ? "" : "s"}.`
                  }
                  fadeOpacity={
                    labelJobStarting || running ? 0.95
                      : (
                        processing
                        || editLabels.length === 0
                        || (unlabelledImportCount === 0 && freshLabels.length === 0 && !settingsChanged)
                        ? 0.4 : 1
                      )
                  }
                />
              );
            })()}
          </div>

          <div>
            <div>
              {/* Labels are managed at the top of the project (the chip rail
                  under the title), so the annotations section no longer repeats
                  a Labels editor. */}
              {/* Detection knobs. The defaults are tuned for general-purpose
                  datasets; the slider rails are clamped to the range
                  where each control still produces sane output. */}
              <div className="mt-4 grid gap-4 border-t border-foreground/[0.06] pt-4 sm:grid-cols-3">
                <Sam3Slider
                  label="Detection confidence"
                  description="How sure the model has to be before keeping a box. Lower picks up more candidates; higher trims false positives."
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  value={sam3Threshold}
                  onChange={setSam3Threshold}
                  format={(v) => `${Math.round(v * 100)}%`}
                />
                <Sam3Slider
                  label="Mask precision"
                  description="Tightness of each detection's mask outline. Higher trims softer pixels for cleaner edges."
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={sam3MaskThreshold}
                  onChange={setSam3MaskThreshold}
                  format={(v) => `${Math.round(v * 100)}%`}
                />
                <Sam3Slider
                  label="Minimum object size"
                  description="Drops same-label fragments smaller than this fraction of the largest match. Set to 0 to keep every hit."
                  min={0}
                  max={0.3}
                  step={0.01}
                  value={sam3MinRelativeArea}
                  onChange={setSam3MinRelativeArea}
                  format={(v) => v === 0 ? "off" : `${Math.round(v * 100)}%`}
                />
              </div>
              {/* Resolution: classic single pass (the model downscales every
                  frame to its fixed inference size, shrinking small objects
                  on big frames below detectability) vs native-resolution
                  tiling. One model pass per tile — explicit opt-in. */}
              <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-foreground/[0.06] pt-4">
                <div className="min-w-0 flex-1">
                  <div className="text-[12px] font-medium text-[var(--foreground)]">Resolution</div>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-foreground/45">
                    {tileNative
                      ? "Large images are sliced into native-resolution tiles (a 4K frame ≈ 7 model passes instead of 1). Catches small objects — slower, uses more compute."
                      : "One pass per image. Large frames are downscaled for the model, so very small objects can be missed."}
                  </p>
                </div>
                <SegmentedControl
                  value={tileNative ? "tile" : "downscale"}
                  onChange={(v) => setTileNative(v === "tile")}
                  options={[
                    { value: "downscale", label: "Downscale" },
                    { value: "tile", label: "Tile · native" },
                  ]}
                />
              </div>
              <div className="mt-3 flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setClearAllOpen(true)}
                  className="text-[11px] uppercase tracking-wider font-mono text-rose-500/80 hover:text-rose-500 transition-colors"
                  title="Wipe every detection + edited box across the whole project"
                >
                  Clear all annotations
                </button>
                <button
                  type="button"
                  onClick={resetSam3Defaults}
                  disabled={sam3IsDefault}
                  className="text-[11px] uppercase tracking-wider font-mono text-foreground/55 hover:text-foreground transition-colors disabled:opacity-30 disabled:cursor-default"
                  title={sam3IsDefault ? "Already at defaults" : "Restore default detection settings"}
                >
                  Default
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Labelling-job card. Sits directly under the Annotations chip
          so the user's eye doesn't have to leave the Start button.
          Wrapped in a height-animating grid so the surrounding
          layout doesn't jump when the card spawns / dismisses. */}
      <section className="px-6 lg:px-10">
        <div
          className="grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
          style={{
            gridTemplateRows: labelJob ? "1fr" : "0fr",
            opacity: labelJob ? 1 : 0,
            marginTop: labelJob ? "0.75rem" : "0",
          }}
        >
          <div className="min-h-0 overflow-hidden">
            {labelJob && (
              <LabelJobCard
                state={labelJob}
                onClose={() => setLabelJob(null)}
                onCancel={async () => {
                  if (!projectId) return;
                  try {
                    await apiFetch(
                      `/api/v2/projects/${projectId}/jobs/${labelJob.jobId}`,
                      { method: "DELETE" },
                    );
                  } catch (e) {
                    console.warn("[label-job] cancel failed:", e);
                  }
                }}
              />
            )}
          </div>
        </div>
      </section>

      {/* Label-purge card. Same chrome as the labelling card but
          the copy describes what the background strip is doing.
          The card survives a browser refresh because the backend
          job persists in the audit log and the active-job poll
          re-attaches on mount. */}
      <section className="px-6 lg:px-10">
        <div
          className="grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
          style={{
            gridTemplateRows: purgeJob ? "1fr" : "0fr",
            opacity: purgeJob ? 1 : 0,
            marginTop: purgeJob ? "0.75rem" : "0",
          }}
        >
          <div className="min-h-0 overflow-hidden">
            {purgeJob && (
              <LabelJobCard
                state={purgeJob}
                onClose={() => setPurgeJob(null)}
                onCancel={async () => {
                  if (!projectId) return;
                  try {
                    await apiFetch(
                      `/api/v2/projects/${projectId}/jobs/${purgeJob.jobId}`,
                      { method: "DELETE" },
                    );
                  } catch (e) {
                    console.warn("[purge-job] cancel failed:", e);
                  }
                }}
                headlines={{
                  running: "Removing label from every image",
                  done: "Label removed",
                  failed: "Label removal failed",
                  cancelled: "Label removal cancelled",
                }}
                phrases={[
                  "Sweeping detections clean…",
                  "Trimming boxes that mention it…",
                  "Refreshing labelled previews…",
                  "Repointing palette + alias map…",
                ]}
                doneMessage="All annotations with this label are gone."
              />
            )}
          </div>
        </div>
      </section>

      {/* Delete-label type-to-confirm modal. Opened by the chip's
          × button; on confirm it schedules the purge job. */}
      {labelPendingDelete && (
        <DeleteLabelModal
          displayName={labelPendingDelete.display}
          onCancel={() => setLabelPendingDelete(null)}
          onConfirm={() => confirmPurgeLabel(labelPendingDelete.canonical)}
        />
      )}

      {clearAllOpen && (
        <ClearAllAnnotationsModal
          onCancel={() => setClearAllOpen(false)}
          onConfirm={confirmClearAllAnnotations}
        />
      )}

      {/* Video trim + sample-rate modal. Mounts on the head of the
          video queue. Cancel pops the queue; Confirm extracts frames
          client-side and feeds them through the existing image
          upload chain via handleImportFiles. */}
      {videoQueue[0] && (
        <VideoFrameModal
          // Key by the head file's identity so each queued video gets a
          // FRESH modal instance. Without this, dropping multiple videos
          // at once reused one instance and carried the previous video's
          // trim (start/end), duration, thumbnails and loaded/error state
          // into the next - so a quick Confirm could extract the wrong
          // frame range from the second video before its metadata loaded.
          key={`${videoQueue[0].name}:${videoQueue[0].size}:${videoQueue[0].lastModified}`}
          file={videoQueue[0]}
          extracting={videoExtracting}
          onCancel={() => {
            if (videoExtracting) return;
            setVideoQueue((prev) => prev.slice(1));
          }}
          onConfirm={async (params) => {
            const head = videoQueue[0];
            if (!head) return;
            const expected = Math.max(
              1,
              Math.floor((params.end - params.start) * params.fps) + 1,
            );
            setVideoExtracting({ done: 0, total: expected });
            let frames: File[] = [];
            try {
              frames = await extractVideoFrames(head, params, (i, total) => {
                setVideoExtracting({ done: i, total });
              });
            } catch (e) {
              const msg = e instanceof Error ? e.message : String(e);
              setVideoError(msg);
              window.setTimeout(() => setVideoError(null), 6000);
              setVideoExtracting(null);
              setVideoQueue((prev) => prev.slice(1));
              return;
            }
            setVideoExtracting(null);
            setVideoQueue((prev) => prev.slice(1));
            if (frames.length > 0) {
              // FileList constructor is browser-specific; the
              // import handler accepts File[] via the FileList |
              // null parameter, TypeScript widens it. Cast through
              // unknown to satisfy strict mode without inventing a
              // DataTransfer dance.
              await handleImportFiles(frames as unknown as FileList);
            }
          }}
        />
      )}

      {/* Transient video-error chip, shown when a video was
          rejected for being too large or failed mid-decode. */}
      {videoError && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[1250] rounded-full border border-rose-500/40 bg-rose-500/[0.12] px-4 py-2 text-[12px] text-rose-700 dark:text-rose-200 shadow-lg backdrop-blur-md">
          {videoError}
        </div>
      )}

      {/* Image processing card, mirrors the labelling card's chrome
          but tracks the upload / resize / safety-check queue. Shows
          when at least one import is in `processing` state and auto-
          dismisses when the batch drains. Sits above the labelling
          card so the user always knows what's blocking Start. */}
      <section className="px-6 lg:px-10">
        <div
          className="grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
          style={{
            gridTemplateRows: processingImportCount > 0 ? "1fr" : "0fr",
            opacity: processingImportCount > 0 ? 1 : 0,
            marginTop: processingImportCount > 0 ? "0.75rem" : "0",
          }}
        >
          <div className="min-h-0 overflow-hidden">
            {processingImportCount > 0 && (
              <LabelJobCard
                state={{
                  jobId: "imports-processing",
                  status: "running",
                  index: processingDone,
                  total: processingBatchTotal,
                  startedAt: undefined,
                  currentImage: null,
                }}
                onClose={() => { /* card auto-dismisses when count hits 0 */ }}
                headlines={{
                  running: "Processing images",
                  done: "All images processed",
                  failed: "Image processing failed",
                  cancelled: "Image processing cancelled",
                }}
                doneMessage="Ready to label."
                phrases={PROCESSING_PHRASES}
              />
            )}
          </div>
        </div>
      </section>

      {/* Augmentation generation card, mirrors the labelling card
          but tracks /augment/job/active. Surfaces progress in-page
          so the user can see the job advance through the dataset
          instead of just the "starting…" toast in the top-right.
          Auto-dismisses 2.4 s after status flips to done. */}
      <section className="px-6 lg:px-10">
        <div
          className="grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
          style={{
            gridTemplateRows: augmentJob ? "1fr" : "0fr",
            opacity: augmentJob ? 1 : 0,
            marginTop: augmentJob ? "0.75rem" : "0",
          }}
        >
          <div className="min-h-0 overflow-hidden">
            {augmentJob && (
              <LabelJobCard
                state={augmentJob}
                onClose={() => setAugmentJob(null)}
                onCancel={async () => {
                  if (!projectId) return;
                  try {
                    await apiFetch(
                      `/api/v2/projects/${projectId}/jobs/${augmentJob.jobId}`,
                      { method: "DELETE" },
                    );
                  } catch (e) {
                    console.warn("[augment-job] cancel failed:", e);
                  }
                }}
                headlines={{
                  running: "Generating augmentations",
                  done: "Augmentations complete",
                  failed: "Augmentation generation failed",
                  cancelled: "Augmentation generation cancelled",
                }}
                doneMessage="New variations added to the dataset."
                phrases={AUGMENT_PHRASES}
              />
            )}
          </div>
        </div>
      </section>
      </>
      )}

      {/* Dataset gallery (V1-style 3-column thumbs, click to open
          viewer). Rendered unconditionally, title shows on first
          paint and the grid fills in as imports stream in from
          the backend. The section ref is the scroll target for
          handleImportFiles + Openverse so dropped images pull the
          page down to the gallery automatically. */}
      <section
        ref={datasetSectionRef}
        className="px-6 lg:px-10 pt-2 pb-4 scroll-mt-20"
      >
        <DatasetGallery
          items={imports}
          totalImports={importsTotal}
          isPulling={!!derivedInfo && !readOnly && imports.length === 0 && !derivedPollDone}
          labels={editLabels}
          hasReferenceEmbeddings={refs.length > 0 || refEmbeddings.length > 0}
          // No references → "general" behaviour, so the "No reference
          // embeddings" warning is suppressed (references are optional).
          isGeneralDataset={!isSpecific}
          onRemove={removeImport}
          onRemoveBatch={removeImportsBatch}
          onMediaChange={updateImport}
          projectId={projectId ?? null}
          labellingFilename={
            labelJob && labelJob.status === "running"
              ? labelJob.currentImage ?? null
              : null
          }
          onAddProjectLabel={(lab) => {
            const trimmed = lab.trim();
            if (!trimmed) return;
            const lower = trimmed.toLowerCase();
            setEditLabels((cur) =>
              cur.some((l) => l.toLowerCase() === lower) ? cur : [...cur, trimmed],
            );
          }}
          labelAliases={labelAliases}
          labelColours={labelColours}
          inputShape={inputSize}
          highlightedImportId={highlightedImportId}
          highlightKind={highlightKind}
          readOnly={readOnly}
          verdicts={verdicts}
          onVerdict={setVerdict}
          verdictFilter={verdictFilter}
          onVerdictFilterChange={setVerdictFilter}
          onStartReview={(scope) => { setReviewScope(scope); setReviewing(true); }}
          filterCountsOverride={filterCountsOverride}
          manifestUpdatedAt={manifestUpdatedAt}
          onViewerOpenIdChange={(id) => { viewerOpenIdRef.current = id; }}
          refViewerOpen={refViewerOpen}
        />
      </section>

      {reviewing && projectId && (
        <ReviewModeV2
          items={(() => {
            // Build the review list off the same image set the
            // gallery shows. Includes only items with a backend id ,
            // verdicts are persisted by import id and an in-flight
            // upload has no stable handle yet.
            const persisted = imports.filter((m) => !!m.backendId && !!m.preview);
            const filtered = persisted.filter((m) => {
              const v = verdicts[m.backendId as string];
              if (reviewScope === "all") return true;
              if (reviewScope === "unrated") return !v;
              if (reviewScope === "good") return v === "good";
              if (reviewScope === "bad") return v === "bad";
              if (reviewScope === "unsure") return v === "unsure";
              return false;
            });
            return filtered.map((m) => ({
              id: m.backendId as string,
              filename: m.filename ?? m.file?.name ?? m.backendId as string,
              preview: m.preview,
              width: m.width,
              height: m.height,
              // Match the dataset viewer's box-source rules so the
              // review canvas paints an identical overlay:
              //   • editedBoxes wins as soon as it EXISTS (including
              //     the explicit empty case where the user has cleared
              //     every box, that should render nothing, not
              //     resurrect the auto detections).
              //   • Otherwise the auto detections are filtered to
              //     skip rejected entries (failed validation / size /
              //     duplicate-cluster pass) which the gallery hides.
              detections: m.editedBoxes !== undefined
                ? m.editedBoxes.map((b) => ({
                    box: [b.x0, b.y0, b.x1, b.y1] as number[],
                    pred_label: b.label ?? null,
                    polygons: b.mask?.polygons ?? null,
                    score: null,
                  }))
                : (m.detections ?? [])
                    .filter((d) => !d.rejected)
                    .map((d) => ({
                      box: d.box ?? null,
                      pred_label: d.predLabel ?? null,
                      polygons: d.mask?.polygons ?? null,
                      score: d.embedSimilarityForLabel ?? d.gdScore ?? null,
                    })),
            }));
          })()}
          verdicts={verdicts}
          onVerdict={setVerdict}
          onClose={() => setReviewing(false)}
          scope={reviewScope}
          projectLabels={editLabels}
          labelAliases={labelAliases}
          labelColours={labelColours}
          onRequestAnnotations={requestReviewAnnotations}
        />
      )}

      {/* Bottom padding to leave room above the Footer once the
          action panels were removed. */}
      <div className="pb-24" />
      </div>

      {/* ─── Augmentations tab content ───
          Just the AugmentationsCard for now, same component the
          dataset tab used to render inline, expanded by default
          since it's the only thing on this tab. */}
      <div hidden={tab !== "augmentations"} className="pk-up">
        <div style={rise()}>
          <AugmentationsCard
            projectId={projectId ?? null}
            previewSourcesReady={importsReady}
            augmentJob={augmentJob}
            onAugmentJobChange={setAugmentJob}
            previewSources={[
              // Imports first so the dataset (where labelled images
              // live) is preferred over references for previews. The
              // backendId requirement was dropping just-uploaded
              // imports from the list, filename is enough for the
              // backend's preview endpoint to find the file.
              ...imports
                .filter((m) => !!m.filename)
                .map((m) => ({
                  source: "import" as const,
                  filename: m.filename as string,
                  preview: m.preview,
                })),
              ...refs
                .filter((r) => !!r.filename)
                .map((r) => ({
                  source: "reference" as const,
                  filename: r.filename as string,
                  preview: r.preview,
                })),
            ]}
          />
        </div>
        <div className="pb-24" />
      </div>

      <Footer />
      </div>

      {exportOpen && projectId && (
        <ExportModal
          projectId={projectId}
          projectName={projectTitle}
          inputShape={inputSize}
          onClose={() => setExportOpen(false)}
        />
      )}

      {settingsOpen && projectId && isOwnDataset && (
        <ProjectSettingsV2
          projectId={projectId}
          projectName={projectTitle}
          initialPrivate={isPrivate}
          labels={editLabels}
          labelAliases={labelAliases}
          labelColours={labelColours}
          references={refs
            .map((r) => ({ filename: r.filename ?? "", preview: r.preview }))
            .filter((r): r is { filename: string; preview: string } => !!r.filename)}
          imports={imports
            .map((m) => ({ filename: m.filename ?? "", preview: m.preview }))
            .filter((m): m is { filename: string; preview: string } => !!m.filename)}
          onClose={() => setSettingsOpen(false)}
          onRenamed={(next) => {
            setProjectTitle(next);
            if (projectId) patchProjectMeta(projectId, { name: next });
          }}
          onLabelColoursChange={(next) => {
            setLabelColours(next);
            if (projectId) patchProjectMeta(projectId, { labelColours: next });
          }}
          onPrivateChange={(next) => {
            setIsPrivate(next);
            if (projectId) patchProjectMeta(projectId, { private: next });
          }}
          onCoverChange={() => {
            if (projectId) patchProjectMeta(projectId, {});
            // A new cover may exist even if a prior load failed; clear the
            // failure latch and bust the cache so the hero re-fetches.
            setBannerCoverFailed(false);
            setCoverBust(Date.now());
          }}
          onDeleted={() => {
            setSettingsOpen(false);
            onClose();
          }}
        />
      )}
      {/* Profanity-filter toast, bottom-centred, auto-dismisses
          after 4.2 s via the flashLabelError timer. Fixed so it
          appears even when the chip rail is below the fold. */}
      {labelError && (
        <div
          role="alert"
          aria-live="polite"
          className="fixed left-1/2 bottom-8 -translate-x-1/2 z-[1200] max-w-sm rounded-lg px-4 py-2.5 text-[12px] font-mono text-red-50 bg-red-700/95 dark:bg-red-800/95 border border-red-300/30"
          style={{
            boxShadow: "var(--shadow-strong)",
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
          }}
        >
          {labelError}
        </div>
      )}
    </main>
  );
}

// ─── Reference image grid ─────────────────────────────────────────────────────
// Flat pool of up to 20 images shared across all labels. The per-label
// annotation counts (0/5) are driven by the backend once wired.

const REF_MAX_POOL = 20;

type RefQualityEntry = {
  self_score: number | null;
  other_score: number | null;
  other_label: string | null;
  quality: number | null;
  warning: "outlier" | "looks like other class" | "only ref for class" | null;
};
type RefQuality = Record<string, Record<string, RefQualityEntry>>;

function RefImageGrid({
  refs,
  onChange,
  projectId = null,
  labels = [],
  labelColours = null,
  pendingCount = 0,
  onLeaveImage,
  refQuality = {},
  readOnly = false,
  onViewerOpenChange,
  onRefDeleted,
}: {
  refs: ReferenceImage[];
  onChange: (next: ReferenceImage[]) => void;
  /** Project the references belong to, threaded to the reference editor so
      its ML calls can load saved images from disk by project_id + filename. */
  projectId?: string | null;
  labels?: string[];
  /** Per-label colour overrides for the filter-chip dots. */
  labelColours?: Record<string, string> | null;
  /** Expected number of references that are still being hydrated
      from the backend. The grid renders this many grey placeholder
      tiles after the real refs so the layout doesn't pop in
      empty-then-full when reopening a project. */
  pendingCount?: number;
  /** Fires whenever the user navigates away from a reference in the
      full-screen editor (prev / next / close). The parent uses this
      to flush any new / edited boxes' embeddings to the manifest so
      reopens skip the on-mount embed catch-up. */
  onLeaveImage?: (ref: ReferenceImage) => void;
  /** Per-ref-id quality map keyed by detection index. Drives the
      "outlier" / "looks like X" warning badge on each thumb. */
  refQuality?: RefQuality;
  /** Public read-only view of someone else's project. Removes the
      drag-and-drop upload zone, the per-tile remove X, and disables
      click-to-edit on each tile, visitors just see the curator's
      references as static thumbs. */
  readOnly?: boolean;
  /** Fires true/false as the full-screen reference editor opens/closes
      so the parent can hide the dataset gallery's back-to-top button
      while the editor overlay is up. */
  onViewerOpenChange?: (open: boolean) => void;
  /** Fires when a persisted reference is removed, with its backend id, so
      the parent can DELETE it server-side (else it resurrects on reload +
      keeps feeding centroids/quality) and drop the expected-count
      placeholder that otherwise spins forever. */
  onRefDeleted?: (referenceId: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [processing, setProcessing] = useState<Set<string>>(new Set());
  const [viewingIdx, setViewingIdx] = useState<number | null>(null);
  const viewing = viewingIdx !== null ? refs[viewingIdx] ?? null : null;
  useEffect(() => {
    onViewerOpenChange?.(viewingIdx !== null);
  }, [viewingIdx, onViewerOpenChange]);
  // Tracks which previews have already been handed to triggerPipeline
  // (across both upload-here and pipeline-recovery code paths) so the
  // recovery effect doesn't double-fire while the call is in flight.
  const triggered = useRef<Set<string>>(new Set());
  // Always points to the latest `refs` so async pipeline callbacks
  // don't overwrite each other with stale snapshots when multiple
  // images are processed concurrently.
  const refsRef = useRef<ReferenceImage[]>(refs);
  useEffect(() => { refsRef.current = refs; }, [refs]);
  const remaining = Math.max(0, REF_MAX_POOL - refs.length);

  // POST the image + labels to /api/v2/references/process. The
  // backend reuses the detector + segmenter models warmed in VRAM by the server's
  // lifespan loader, no separate model load on this path. Detections
  // are turned into EditableBoxes and stored on the matching
  // ReferenceImage's `boxes` field via the parent's onChange.
  const triggerPipeline = async (ref: ReferenceImage) => {
    setProcessing((prev) => new Set([...prev, ref.preview]));
    try {
      const fd = new FormData();
      fd.append("image", ref.file);
      // Section-scoped: when the reference was dropped into a label's
      // section, send only that label and force the detector onto it,
      // so the box is classified by the section rather than guessed
      // among siblings. Falls back to all labels for legacy refs.
      const sectionLabel = ref.label?.trim();
      fd.append("labels", JSON.stringify(sectionLabel ? [sectionLabel] : labels));
      if (sectionLabel) fd.append("force_label", sectionLabel);
      const r = await apiFetch("/api/v2/references/process", {
        method: "POST",
        body: fd,
      });
      if (!r.ok) {
        const body = await r.text().catch(() => "");
        throw new Error(`http ${r.status}, ${body || "no body"}`);
      }
      const data = (await r.json()) as {
        width: number;
        height: number;
        detections: { label: string; score: number; box: number[]; mask: MaskShape | null }[];
      };
      const newBoxes = detectionsToBoxes(
        data.detections.map((d) => ({
          label: d.label,
          score: d.score,
          box_xyxy: d.box,
          mask: d.mask,
        })),
        labels,
      );
      // Don't overwrite boxes that have already been populated ,
      // either by a previous pipeline run or by the user's manual
      // edits in the reference editor. Same reasoning as the
      // matching guard in HomeView.v2TriggerPipeline: a concurrent
      // recovery pipeline finishing while the user is mid-edit
      // would otherwise wipe their click-to-detect / drawn boxes.
      onChange(
        refsRef.current.map((it) => {
          if (it.preview !== ref.preview) return it;
          if (it.boxes !== undefined) {
            return { ...it, width: data.width, height: data.height };
          }
          return { ...it, width: data.width, height: data.height, boxes: newBoxes };
        }),
      );
    } catch (e) {
      console.error("[v2 ref pipeline]", e);
    } finally {
      setProcessing((prev) => {
        const next = new Set(prev);
        next.delete(ref.preview);
        return next;
      });
    }
  };

  const updateBoxes = (preview: string, nextBoxes: EditableBox[]) => {
    onChange(
      refsRef.current.map((it) => (it.preview === preview ? { ...it, boxes: nextBoxes } : it)),
    );
  };

  // Recovery: if any reference arrived without `boxes` (its pipeline
  // didn't finish in HomeView before the user clicked "Open project"),
  // run the pipeline now. Runs once per preview thanks to `triggered`.
  useEffect(() => {
    refs.forEach((ref) => {
      if (ref.boxes !== undefined) return;
      if (triggered.current.has(ref.preview)) return;
      triggered.current.add(ref.preview);
      triggerPipeline(ref);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refs]);

  const addFiles = (files: FileList | File[] | null, sectionLabel?: string) => {
    if (!files || files.length === 0) return;
    const candidates = Array.from(files as ArrayLike<File>)
      .filter(isImageFile)
      .slice(0, remaining);
    if (candidates.length === 0) return;

    // Phase 1: spawn placeholder tiles INSTANTLY. The preview blob
    // URL points at the original file bytes, the browser decodes
    // it locally, no network round-trip, no resize delay. User sees
    // the count + thumbs render the same render tick. The section
    // label is stamped on now so the tile lands in the right section
    // before detection even runs.
    const placeholders: ReferenceImage[] = candidates.map((f) => ({
      file: f,
      preview: URL.createObjectURL(f),
      label: sectionLabel,
    }));
    // Claim each placeholder in `triggered` NOW so the boxes-undefined
    // recovery effect doesn't ALSO fire its pipeline in the window before
    // the resize-then below runs. That double-ran detect+segment (two GPU
    // passes) per added reference; now only the resize-then triggers, on
    // the smaller resized file.
    for (const p of placeholders) triggered.current.add(p.preview);
    onChange([...refs, ...placeholders]);

    // Phase 2: resize each candidate in parallel. As each one
    // finishes we swap `file` to the resized version (smaller bytes
    // for the /references POST + the /process pipeline) WITHOUT
    // touching `preview`, refs identity is keyed by preview URL in
    // a handful of places (refEmbedsRef, refUploadStatus, etc.) so
    // changing it mid-flight would lose those associations. Fires
    // the per-ref pipeline immediately after its own resize, so
    // pipelines can run in parallel even though the tiles spawned
    // on a single tick.
    candidates.forEach((f, i) => {
      const placeholder = placeholders[i];
      resizeForUpload(f)
        .catch(() => f)
        .then((resized) => {
          onChange(
            refsRef.current.map((it) =>
              it.preview === placeholder.preview
                ? { ...it, file: resized }
                : it,
            ),
          );
          triggered.current.add(placeholder.preview);
          triggerPipeline({ ...placeholder, file: resized });
        });
    });
    if (fileRef.current) fileRef.current.value = "";
  };

  const removeAt = (idx: number) => {
    const list = refs.slice();
    const [gone] = list.splice(idx, 1);
    if (gone) {
      try { URL.revokeObjectURL(gone.preview); } catch { /* already revoked */ }
      // Persist the delete server-side. A local-only removal left the
      // reference (bytes + embeddings) on the backend: it resurrected on
      // reload, kept feeding the label centroids / quality, and the stale
      // expected-count showed a permanent loading placeholder.
      if (gone.referenceId) onRefDeleted?.(gone.referenceId);
    }
    onChange(list);
  };

  // Per-label sections: a reference belongs to the label section it was
  // uploaded into. `label` is stamped at drop time; references that
  // predate the field fall back to their first box's label so they
  // still land in the right section. `dragLabel` tracks the section
  // currently under a drag, `pendingUploadLabel` remembers which
  // section opened the shared file picker.
  const [dragLabel, setDragLabel] = useState<string | null>(null);
  const [pendingUploadLabel, setPendingUploadLabel] = useState<string | null>(null);
  const sectionOf = (r: ReferenceImage): string | null => {
    if (r.label && r.label.trim()) return r.label.trim();
    const b = (r.boxes ?? [])[0];
    return b?.label ?? null;
  };
  const knownKeys = new Set(labels.map((l) => l.toLowerCase()));
  const unsortedRefs = refs.filter((r) => {
    const s = sectionOf(r);
    return !s || !knownKeys.has(s.toLowerCase());
  });

  // One reference thumbnail with its quality-warning badge. Shared by
  // every label section and the unsorted bucket. The warning copy +
  // colour are unchanged from the old flat grid, only the host layout
  // moved from a CSS grid cell to a flex-wrapped fixed square.
  const renderThumb = (r: ReferenceImage) => {
    const isProcessing = processing.has(r.preview);
    // Original-array index, needed by setViewingIdx (so the editor's
    // prev/next still walk the full ref list) and by removeAt.
    const i = refs.findIndex((rr) => rr.preview === r.preview);
    const qEntries = r.referenceId ? refQuality[r.referenceId] : undefined;
    let worstWarning: RefQualityEntry | null = null;
    if (qEntries) {
      for (const e of Object.values(qEntries)) {
        if (!e.warning) continue;
        if (worstWarning === null) { worstWarning = e; continue; }
        const order = (w: string | null) =>
          w === "outlier" ? 3 : w === "looks like other class" ? 2 : w === "only ref for class" ? 1 : 0;
        if (order(e.warning) > order(worstWarning.warning)) worstWarning = e;
      }
    }
    const warningTitle = worstWarning?.warning === "outlier"
      ? `Outlier, this reference's embedding doesn't match its own class (self-sim ${worstWarning.self_score?.toFixed(2)} below 0.50). Consider re-cropping or removing.`
      : worstWarning?.warning === "looks like other class"
        ? `Looks like "${worstWarning.other_label}" (sim ${worstWarning.other_score?.toFixed(2)}) more than this section's label (sim ${worstWarning.self_score?.toFixed(2)}). Check it is in the right section.`
        : worstWarning?.warning === "only ref for class"
          ? "Only reference in this section, add a few more so the centroid isn't a single point."
          : null;
    return (
      <div key={r.preview} className="group relative h-24 w-24 rounded-lg overflow-hidden bg-foreground/5 border border-foreground/10">
        <LazyRefImage
          src={r.preview}
          blurhash={r.blurhash}
          isProcessing={isProcessing}
          onClick={readOnly ? () => {} : () => !isProcessing && i >= 0 && setViewingIdx(i)}
        />
        {isProcessing ? (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <svg className="h-7 w-7 animate-spin text-[var(--foreground)]" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
              <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          </div>
        ) : !readOnly ? (
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); if (i >= 0) removeAt(i); }}
            className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-foreground/80 hover:text-foreground hover:bg-[var(--background)] grid place-items-center text-sm opacity-0 group-hover:opacity-100 transition-opacity"
            aria-label="Remove reference"
          >×</button>
        ) : null}
        {worstWarning && warningTitle && !isProcessing && (
          <Tooltip
            label={warningTitle}
            variant="rich"
            side="top"
            align="center"
            width={260}
            className="absolute bottom-1 left-1 right-1 z-10 block"
          >
            <div
              className={[
                "w-full px-1.5 py-0.5 rounded-md text-[9px] font-medium tabular-nums backdrop-blur-sm flex items-center gap-1",
                worstWarning.warning === "outlier"
                  ? "bg-rose-500/35 text-rose-50 border border-rose-300/40"
                  : worstWarning.warning === "looks like other class"
                    ? "bg-amber-500/35 text-amber-50 border border-amber-300/40"
                    : "bg-foreground/10 text-foreground/70 border border-foreground/15",
              ].join(" ")}
            >
              <svg className="h-2.5 w-2.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span className="truncate">
                {worstWarning.warning === "outlier"
                  ? "outlier"
                  : worstWarning.warning === "looks like other class"
                    ? `looks like ${worstWarning.other_label}`
                    : "only ref"}
              </span>
            </div>
          </Tooltip>
        )}
      </div>
    );
  };

  return (
    <div className="mt-4">
      {/* Per-label sections. Each project label is its own drop zone, so
          the class is known the moment a photo lands; the detector only
          localises it. Dropping into a section (or "+ Add photos") tags
          the reference with that label, and the quality badge then flags
          any photo that looks more like another section. */}
      {labels.length === 0 ? (
        <div className="rounded-xl border border-dashed border-foreground/10 px-4 py-6 text-center text-[12px] text-foreground/45">
          Add at least one label above to start uploading references.
        </div>
      ) : (
        <div className="grid gap-3">
          {labels.map((lab) => {
            const k = lab.toLowerCase();
            const c = colourForLabel(labels, lab, labelColours);
            const owned = refs.filter((r) => (sectionOf(r) ?? "").toLowerCase() === k);
            const isDragTarget = dragLabel === lab;
            const canAdd = !readOnly && remaining > 0;
            return (
              <div
                key={lab}
                onDragEnter={(e) => { if (canAdd) { e.preventDefault(); setDragLabel(lab); } }}
                onDragOver={(e) => { if (canAdd) { e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = "copy"; } }}
                onDragLeave={(e) => { if (e.currentTarget === e.target) setDragLabel((cur) => (cur === lab ? null : cur)); }}
                onDrop={(e) => {
                  if (readOnly) return;
                  e.preventDefault();
                  setDragLabel(null);
                  if (remaining <= 0) return;
                  // Safari fallback: dataTransfer.files can be empty even
                  // when items holds the dropped files.
                  const dt = e.dataTransfer;
                  const native = dt.files;
                  const files: File[] = native && native.length > 0 ? Array.from(native) : [];
                  if (files.length === 0 && dt.items && dt.items.length > 0) {
                    for (const item of Array.from(dt.items)) {
                      if (item.kind === "file") { const f = item.getAsFile(); if (f) files.push(f); }
                    }
                  }
                  if (files.length > 0) addFiles(files, lab);
                }}
                className={[
                  "rounded-2xl border p-4 transition-colors",
                  isDragTarget
                    ? "border-foreground/45 bg-foreground/[0.05]"
                    : "border-foreground/[0.08] hover:border-foreground/[0.14]",
                ].join(" ")}
              >
                <div className="flex items-center justify-between gap-3 mb-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: c }} aria-hidden />
                    <span className="text-sm font-medium truncate">{lab}</span>
                    <span className="text-[11px] text-foreground/35 font-mono shrink-0">
                      {owned.length} ref{owned.length === 1 ? "" : "s"}
                    </span>
                  </div>
                  {!readOnly && (
                    <button
                      type="button"
                      onClick={() => { setPendingUploadLabel(lab); fileRef.current?.click(); }}
                      disabled={!canAdd}
                      className="text-xs px-3.5 py-1.5 rounded-full border border-foreground/[0.12] hover:border-foreground/25 hover:bg-foreground/[0.04] disabled:opacity-35 disabled:cursor-not-allowed transition-colors shrink-0"
                    >
                      + Add photos
                    </button>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {owned.length === 0 && (
                    <span className="text-xs text-foreground/35 leading-relaxed py-3">
                      {readOnly
                        ? "No references."
                        : `No references yet. Add or drop 3 to 5 clear photos of ${lab}.`}
                    </span>
                  )}
                  {owned.map((r) => renderThumb(r))}
                </div>
              </div>
            );
          })}

          {/* Unsorted bucket: references whose label is not (or no
              longer) a project label, e.g. uploaded before that label
              was renamed. Open one to re-label it in the editor. */}
          {unsortedRefs.length > 0 && (
            <div className="rounded-2xl border border-amber-300/20 bg-amber-300/[0.03] p-4">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className="text-sm font-medium">Unsorted</span>
                <span className="text-[11px] text-foreground/35 font-mono">{unsortedRefs.length}</span>
                <span className="text-[11px] text-foreground/45">open one to assign it to a label</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {unsortedRefs.map((r) => renderThumb(r))}
              </div>
            </div>
          )}

          {/* Hydration placeholders for references still loading from
              the backend, shown until their bytes + boxes arrive and
              they slot into a section. */}
          {pendingCount > 0 && (
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: pendingCount }).map((_, idx) => (
                <div
                  key={`placeholder_${idx}`}
                  className="relative h-24 w-24 rounded-lg overflow-hidden bg-foreground/[0.03] border border-foreground/10"
                  style={{ animation: "fadeIn 240ms ease-out both" }}
                  aria-hidden
                >
                  <div className="absolute inset-0 grid place-items-center">
                    <svg className="h-6 w-6 animate-spin text-foreground/30" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                      <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                    </svg>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {viewing !== null && viewingIdx !== null && (
        <ReferenceImageEditor
          refImage={viewing}
          labels={labels}
          projectId={projectId}
          onChange={(nextBoxes) => updateBoxes(viewing.preview, nextBoxes)}
          onClose={() => {
            // Snapshot the ref BEFORE setViewingIdx clears it, then
            // fire the flush. The latest box edits are already in
            // the parent's `refs` via the onChange callback path,
            // so we look the ref up by preview rather than reusing
            // the stale `viewing` closure.
            const leaving = refs.find((r) => r.preview === viewing.preview);
            setViewingIdx(null);
            if (leaving) onLeaveImage?.(leaving);
          }}
          onPrev={() => {
            const leaving = refs.find((r) => r.preview === viewing.preview);
            setViewingIdx((i) => (i === null ? null : Math.max(0, i - 1)));
            if (leaving) onLeaveImage?.(leaving);
          }}
          onNext={() => {
            const leaving = refs.find((r) => r.preview === viewing.preview);
            setViewingIdx((i) =>
              i === null ? null : Math.min(refs.length - 1, i + 1),
            );
            if (leaving) onLeaveImage?.(leaving);
          }}
          hasPrev={viewingIdx > 0}
          hasNext={viewingIdx < refs.length - 1}
          index={viewingIdx}
          total={refs.length}
        />
      )}
      {/* Shared hidden file input. Each section's "+ Add photos" stashes
          its label in pendingUploadLabel then opens this picker, so the
          chosen files land in that section. */}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          addFiles(e.target.files, pendingUploadLabel ?? undefined);
          if (fileRef.current) fileRef.current.value = "";
        }}
      />
      {!readOnly && (
        <p className="mt-3 text-[11px] text-foreground/35">
          {remaining > 0
            ? `${remaining} of ${REF_MAX_POOL} reference slots remaining · jpg · png · webp`
            : `Maximum ${REF_MAX_POOL} references reached`}
        </p>
      )}
    </div>
  );
}

// ─── Editable chip ────────────────────────────────────────────────────────────

function EditableChip({
  label, colour, onDelete, onRename,
}: {
  label: string;
  colour: string;
  onDelete: () => void;
  onRename?: (next: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(label);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!editing) return;
    setDraft(label);
    const t = window.setTimeout(() => { inputRef.current?.focus(); inputRef.current?.select(); }, 0);
    return () => window.clearTimeout(t);
  }, [editing, label]);

  const cancel = () => { setDraft(label); setEditing(false); };
  const commit = () => {
    const next = draft.trim();
    setEditing(false);
    if (next && next !== label) onRename?.(next);
  };

  // Chip text + sub-elements inherit from the chip's color so a
  // dark background flips white text + a paler underline / × button.
  // currentColor lets a single contrast pick drive every interior
  // mark.
  const fg = readableTextForBg(colour);
  return (
    <span
      // Shrunk by default (just the label); on hover the chip grows and the
      // remove (x) button expands in. The width + padding animate so the chip
      // visibly increases in size under the cursor.
      className="group inline-flex items-center rounded-full pl-3 pr-3 group-hover:pr-1.5 h-7 text-sm font-medium transition-[padding] duration-150 ease-out motion-reduce:transition-none"
      style={{ backgroundColor: colour, color: fg }}
    >
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); commit(); }
            else if (e.key === "Escape") { e.preventDefault(); cancel(); }
          }}
          onBlur={cancel}
          aria-label={`Rename ${label}`}
          className="bg-transparent outline-none placeholder:opacity-40"
          style={{ width: `${Math.max(2, draft.length)}ch`, color: fg }}
        />
      ) : (
        <span
          role={onRename ? "button" : undefined}
          tabIndex={onRename ? 0 : undefined}
          onClick={() => onRename && setEditing(true)}
          onKeyDown={(e) => { if (onRename && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); setEditing(true); } }}
          className={["select-none", onRename ? "hover:underline underline-offset-2 cursor-text" : "cursor-default"].join(" ")}
          style={onRename ? { textDecorationColor: "currentColor" } : undefined}
        >
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={onDelete}
        aria-label={`Remove ${label}`}
        className="inline-flex h-5 w-0 shrink-0 items-center justify-center overflow-hidden rounded-full opacity-0 transition-all duration-150 ease-out group-hover:ml-1 group-hover:w-5 group-hover:opacity-100 hover:bg-black/20 motion-reduce:transition-none"
        style={{ color: fg }}
      >
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden>
          <path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </button>
    </span>
  );
}


// Relabel-hotkey rows. 1-9 covers the first nine labels; QWERTYUIOP
// catches labels 10-19; ASDFGHJKL catches 20-28. The home row stops
// at L because the next letters (;, ') aren't reliable cross-keyboard.
const RELABEL_TOP_ROW = "qwertyuiop".split("");
const RELABEL_HOME_ROW = "asdfghjkl".split("");
const RELABEL_KEY_LIMIT = 9 + RELABEL_TOP_ROW.length + RELABEL_HOME_ROW.length;

// Display label for slot `i` (0-indexed). Returns null when the slot
// is past RELABEL_KEY_LIMIT, caller skips rendering the kbd chip.
function keyForRelabelSlot(slot: number): string | null {
  if (slot < 0) return null;
  if (slot < 9) return String(slot + 1);
  if (slot < 9 + RELABEL_TOP_ROW.length) return RELABEL_TOP_ROW[slot - 9].toUpperCase();
  if (slot < RELABEL_KEY_LIMIT) return RELABEL_HOME_ROW[slot - 9 - RELABEL_TOP_ROW.length].toUpperCase();
  return null;
}

// Reverse: keyboard event's `key` → slot index. Case-insensitive for
// letters; ignores anything outside the supported rows.
function slotForRelabelKey(key: string): number | null {
  if (/^[1-9]$/.test(key)) return parseInt(key, 10) - 1;
  const low = key.toLowerCase();
  if (low.length !== 1) return null;
  const topIdx = RELABEL_TOP_ROW.indexOf(low);
  if (topIdx >= 0) return 9 + topIdx;
  const homeIdx = RELABEL_HOME_ROW.indexOf(low);
  if (homeIdx >= 0) return 9 + RELABEL_TOP_ROW.length + homeIdx;
  return null;
}

// Three-pill verdict toggle rendered in the image viewer header.
// Same setter Review mode uses, so a flip here also drops the image
// into the matching gallery filter chip live.
function VerdictPills({
  current,
  onToggle,
}: {
  current: "good" | "bad" | "unsure" | null;
  onToggle: (v: "good" | "bad" | "unsure") => void;
}) {
  const pills: { key: "good" | "unsure" | "bad"; label: string; active: string; idle: string }[] = [
    {
      key: "good",
      label: "Good",
      active: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-200",
      idle: "text-foreground/55 hover:text-emerald-700 dark:hover:text-emerald-200",
    },
    {
      key: "unsure",
      label: "Unsure",
      active: "bg-amber-500/20 text-amber-800 dark:text-amber-200",
      idle: "text-foreground/55 hover:text-amber-800 dark:hover:text-amber-200",
    },
    {
      key: "bad",
      label: "Bad",
      active: "bg-rose-500/15 text-rose-700 dark:text-rose-200",
      idle: "text-foreground/55 hover:text-rose-700 dark:hover:text-rose-200",
    },
  ];
  return (
    <div className="inline-flex items-center gap-0.5 rounded-full bg-foreground/[0.04] p-0.5" role="group" aria-label="Set image verdict">
      {pills.map((p) => {
        const active = current === p.key;
        return (
          <button
            key={p.key}
            type="button"
            onClick={() => onToggle(p.key)}
            aria-pressed={active}
            title={active ? `Clear ${p.label.toLowerCase()} verdict` : `Mark image ${p.label.toLowerCase()}`}
            className={[
              "rounded-full px-2.5 py-1 text-[11px] uppercase tracking-wider font-medium transition-colors",
              active ? p.active : p.idle,
            ].join(" ")}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}


// ─── Detection slider ────────────────────────────────────────────────
// Single-row labelled range input used by the Annotations card's
// ─── Adaptive Start labelling button ─────────────────────────────
// Pulls its rendered width from a hidden ghost span so the orange
// pill smoothly reshapes when the copy switches between
// "Start labelling", "Start labelling new images",
// "Start labelling all images" and "Start labelling {labelname}".
// Browsers don't transition `width: auto` so we measure the ghost
// and set an explicit width that the CSS transition can pick up.

function AnimatedStartButton({
  text,
  buttonKey,
  running,
  disabled,
  onClick,
  title,
  fadeOpacity,
}: {
  text: string;
  buttonKey: string;
  running: boolean;
  disabled: boolean;
  onClick: () => void;
  title: string;
  fadeOpacity: number;
}) {
  const ghostRef = useRef<HTMLSpanElement | null>(null);
  const [pixelWidth, setPixelWidth] = useState<number | null>(null);
  // Measure the ghost any time the text changes. useLayoutEffect so
  // the width lands before the browser paints, which avoids a
  // single-frame "auto" → "measured" jump on the first render.
  useLayoutEffect(() => {
    if (!ghostRef.current) return;
    const w = ghostRef.current.offsetWidth;
    if (w > 0) setPixelWidth(w);
  }, [text]);
  return (
    <div className="relative inline-flex items-center">
      {/* Off-screen ghost mirrors the button's chrome (font, padding,
          weight) so its offsetWidth is exactly what the button will
          be once it settles. position:absolute keeps it out of the
          layout flow. */}
      <span
        ref={ghostRef}
        aria-hidden
        className="rounded-full px-5 py-2 text-[13px] font-semibold inline-flex items-center justify-center whitespace-nowrap pointer-events-none"
        style={{
          position: "absolute",
          visibility: "hidden",
          left: 0,
          top: 0,
          height: 0,
          whiteSpace: "nowrap",
        }}
      >
        {text}
      </span>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        title={title}
        className="relative rounded-full px-5 py-2 text-[13px] font-semibold text-black disabled:cursor-not-allowed overflow-hidden whitespace-nowrap inline-flex items-center justify-center"
        style={{
          backgroundColor: running ? "rgba(251,146,60,0.55)" : "#fb923c",
          boxShadow: running
            ? "0 0 14px 1px rgba(249,115,22,0.25)"
            : "0 0 14px 1px rgba(249,115,22,0.35)",
          opacity: fadeOpacity,
          // Pixel width from the ghost lets the browser interpolate
          // between two concrete values. Falls back to "auto" on the
          // very first render so we never paint a 0-px button.
          width: pixelWidth != null ? `${pixelWidth}px` : "auto",
          transition:
            "width 320ms cubic-bezier(0.22, 1, 0.36, 1), opacity 220ms ease, background-color 220ms ease, box-shadow 220ms ease",
        }}
      >
        {running && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "linear-gradient(90deg, transparent 0%, rgb(var(--foreground-rgb) / 0.28) 50%, transparent 100%)",
              animation: "labelBtnShimmer 2.4s linear infinite",
            }}
          />
        )}
        <span
          key={buttonKey}
          className="relative inline-block"
          style={{ animation: "labelBtnTextIn 320ms cubic-bezier(0.2,0.7,0.2,1) both" }}
        >
          {text}
        </span>
        <style>{`
          @keyframes labelBtnTextIn {
            0%   { opacity: 0; transform: translateY(4px); }
            100% { opacity: 1; transform: translateY(0); }
          }
          @keyframes labelBtnShimmer {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `}</style>
      </button>
    </div>
  );
}

// Detection knobs. Renders title + live value on top, then the slider,
// then a quiet help line under it. Same visual rhythm across the
// three knobs so the user reads them as a set.

function Sam3Slider({
  label,
  description,
  min,
  max,
  step,
  value,
  onChange,
  format,
}: {
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] uppercase tracking-wider text-foreground/55">{label}</span>
        <span className="text-[12px] font-mono tabular-nums text-[var(--foreground)]">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1 w-full appearance-none rounded-full bg-foreground/10 accent-[var(--foreground)]"
        aria-label={label}
      />
      <p className="text-[11px] text-foreground/45 leading-snug">{description}</p>
    </div>
  );
}

// ─── Segmented control (local copy) ──────────────────────────────────────────

function SegmentedControl<T extends string>({
  value, onChange, options,
}: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-full border border-[var(--border)] p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={["rounded-full px-3 py-1 text-xs uppercase tracking-wide transition-colors", value === opt.value ? "bg-foreground text-background" : "text-[var(--muted)] hover:text-foreground"].join(" ")}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// ─── Shape help popover ───────────────────────────────────────────────────────

function ShapeHelpPopover() {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);
  const triggerRef = useRef<HTMLSpanElement | null>(null);
  const timerRef = useRef<number | null>(null);
  const positionFromTrigger = () => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Sit just under the trigger, left-aligned to its left edge.
    setAnchor({ x: r.left, y: r.bottom + 8 });
  };
  const onEnter = () => {
    if (timerRef.current) { window.clearTimeout(timerRef.current); timerRef.current = null; }
    timerRef.current = window.setTimeout(() => {
      positionFromTrigger();
      setOpen(true);
      timerRef.current = null;
    }, 200);
  };
  const onLeave = () => {
    if (timerRef.current) { window.clearTimeout(timerRef.current); timerRef.current = null; }
    setOpen(false);
  };
  useEffect(() => () => { if (timerRef.current) window.clearTimeout(timerRef.current); }, []);
  return (
    <span className="relative inline-flex" onMouseEnter={onEnter} onMouseLeave={onLeave}>
      <span ref={triggerRef} role="img" aria-label="Target input shape help" className="grid h-4 w-4 place-items-center rounded-full border border-foreground/20 text-[10px] font-semibold text-foreground/55 cursor-help select-none transition-colors hover:text-foreground hover:border-foreground/40">i</span>
      {open && anchor && typeof window !== "undefined" && createPortal(
        <>
          <div
            className="fixed inset-0 z-[1000] pointer-events-none"
            style={{
              background: "rgb(var(--background-rgb) / 0.55)",
              animation: "shapeHelpDimIn 220ms ease-out both",
            }}
            aria-hidden="true"
          />
          <div
            role="tooltip"
            className="fixed z-[1002] w-80 rounded-2xl border border-foreground/10 p-4 pointer-events-none"
            // Themable tooltip surface, light grey card in light
            // mode, dark surface in dark. Previously pinned to a
            // hard-coded #0c0c0e fallback so the box was always dark.
            style={{
              left: anchor.x,
              top: anchor.y,
              background: "rgb(var(--surface-rgb))",
              boxShadow: "var(--shadow-strong)",
              animation: "shapeHelpPopIn 220ms cubic-bezier(0.2,0.7,0.2,1) both",
            }}
          >
            <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight mb-2">Target input shape</h3>
            <p className="text-xs text-foreground/70 leading-relaxed mb-3">The square tensor size your detector will resize images to before running. Smaller shapes are faster on-device but lose objects whose smallest side gets squeezed below the model&rsquo;s detection floor.</p>
            <p className="text-xs text-foreground/70 leading-relaxed">Pick the shape that matches the runtime you&rsquo;re targeting, the editor uses it to flag boxes that would shrink below that floor at the chosen resolution.</p>
          </div>
          <style>{`
            @keyframes shapeHelpDimIn {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            @keyframes shapeHelpPopIn {
              from { opacity: 0; transform: translateY(-4px) scale(0.97); }
              to { opacity: 1; transform: translateY(0) scale(1); }
            }
          `}</style>
        </>,
        document.body,
      )}
    </span>
  );
}

// ─── Import media section ─────────────────────────────────────────────────────

function ImportMediaSection({
  onUpload,
  projectId,
  alreadyImportedUrls,
  onOpenverseAdded,
  disabled = false,
  disabledMessage,
}: {
  onUpload?: (files: FileList | null) => void;
  /** Project id used by the inline Openverse search panel below.
      When null the panel is hidden, keeps the upload zone usable
      even before the manifest hydrates. */
  projectId?: string | null;
  /** URLs already in this project (from manifest.imports[].source.url).
      Filtered out of every Openverse response so the user can't
      accidentally re-add something already in the dataset. */
  alreadyImportedUrls?: string[];
  /** Called by the inline panel when the user submits a batch of
      Openverse picks. Parent refreshes the gallery state. */
  onOpenverseAdded?: (added: number) => Promise<void> | void;
  /** When true, the drop zone won't fire onUpload and renders with a
      muted, "wait for processing" appearance. Used to gate uploads
      until reference photos finish embedding on the backend. */
  disabled?: boolean;
  /** Message shown over the drop zone when disabled. */
  disabledMessage?: string;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading] = useState(false);

  const handleMediaFiles = (files: FileList | File[] | null) => {
    if (disabled) return;
    // FileList and File[] share `.length` + index access, so the
    // FileList-only callers (e.g. the <input> onChange) continue to
    // work unchanged. Cast keeps TS happy without forcing every
    // downstream caller to accept the union.
    onUpload?.(files as FileList | null);
  };
  // Safari refuses drops on elements whose dragover handler doesn't
  // both preventDefault AND set `dataTransfer.dropEffect`.
  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled && e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    // Safari sometimes hands us an empty `dataTransfer.files` while
    // populating `dataTransfer.items` instead (Photos.app drags,
    // drags from other browser tabs, iCloud-stub files). Fall back
    // to the items API in that case.
    const dt = e.dataTransfer;
    const native = dt.files;
    const files: File[] = native && native.length > 0 ? Array.from(native) : [];
    if (files.length === 0 && dt.items && dt.items.length > 0) {
      const items = Array.from(dt.items);
      console.log(
        "[v2 drop] files empty, falling back to items:",
        items.map((i) => `${i.kind}/${i.type || "<no-type>"}`),
      );
      for (const item of items) {
        if (item.kind === "file") {
          const f = item.getAsFile();
          if (f) files.push(f);
        }
      }
    }
    console.log("[v2 drop] resolved", files.length, "file(s)", {
      types: Array.from(dt.types || []),
      filesLen: native?.length ?? 0,
      itemsLen: dt.items?.length ?? 0,
    });
    if (files.length === 0) return;
    handleMediaFiles(files);
  };

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
      {/* Dropzone. Inner padding matches the collapsible section
          cards (px-5) and the content is wrapped in the same flex-
          chevron layout so the title's left edge lines up with the
          "Reference images" / "Annotations" / "Don't have images?"
          titles. Invisible placeholder where the chevron would be
          keeps the column without adding a fake control. */}
      <div
        onDragEnter={onDragEnter} onDragLeave={onDragLeave} onDragOver={onDragOver} onDrop={onDrop}
        onClick={() => { if (!disabled) fileInputRef.current?.click(); }}
        aria-disabled={disabled}
        className={[
          "relative flex min-w-0 flex-col justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-all px-5 py-5 bg-[var(--surface-2)] sm:flex-1",
          disabled
            ? "cursor-not-allowed opacity-60 border-foreground/15"
            : ["cursor-pointer", dragOver ? "border-foreground/50" : "border-foreground/20 hover:border-foreground/40"].join(" "),
        ].join(" ")}
      >
        <input ref={fileInputRef} type="file" accept="image/*,video/*" multiple className="hidden" disabled={disabled} onChange={(e) => { handleMediaFiles(e.target.files); if (fileInputRef.current) fileInputRef.current.value = ""; }} />

        <div className="flex items-center gap-3 max-w-[80%]">
          <div className="min-w-0 flex-1">
            <div className="text-2xl font-medium tracking-tight text-[var(--foreground)]">
              {disabled
                ? (disabledMessage ?? "Waiting for setup to finish…")
                : (uploading ? "Uploading…" : "Drop media here or click to browse")}
            </div>
            <div className="mt-1.5 text-[12px] text-foreground/45 leading-snug">
              Images jpg · png · webp · gif · heic · avif · bmp · tiff{" "}
              <span className="text-foreground/25">·</span>{" "}
              Videos mp4 · mov · webm · avi · mkv
            </div>
            <p className="mt-1.5 text-[10px] text-foreground/30 leading-relaxed">
              By uploading you agree to our{" "}
              <a href="/acceptable-use" target="_blank" rel="noopener noreferrer" className="text-foreground/45 hover:text-foreground underline underline-offset-2" onClick={(e) => e.stopPropagation()}>Acceptable Use Policy</a>
              {" "}and{" "}
              <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-foreground/45 hover:text-foreground underline underline-offset-2" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
            </p>
          </div>
        </div>

        <div className="absolute pointer-events-none flex items-center" style={{ top: "1.25rem", right: "1.5rem", bottom: "1.25rem" }} aria-hidden="true">
          {/* Photo-glyph icon, theme-aware (surface fill + muted strokes). Capped
              height (max-h-20) + centered so it never overgrows when the box gets
              tall (e.g. text wraps on a narrow viewport) and spills out. */}
          <svg viewBox="0 0 100 80" className="h-full max-h-20 w-auto" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="2" width="96" height="76" rx="9" fill="rgb(var(--surface-rgb))" stroke="rgb(var(--muted-rgb))" strokeWidth="2" />
            <circle cx="24" cy="24" r="9" fill="rgb(var(--muted-rgb))" stroke="none" />
            <polyline points="2 58 26 33 44 48 63 24 98 58" fill="none" stroke="rgb(var(--muted-rgb))" strokeWidth="2" />
          </svg>
        </div>
      </div>

      {/* "Don't have images?" - compact card on the right that opens the
          Openverse importer in a popup. */}
      {projectId && (
        <div className="shrink-0 sm:w-56">
          <OpenverseInlinePanel
            projectId={projectId}
            alreadyImportedUrls={alreadyImportedUrls ?? []}
            onAdded={onOpenverseAdded}
          />
        </div>
      )}
    </div>
  );
}

// ─── Embedding helpers ────────────────────────────────────────────────────────
// Reference embeddings are keyed by `cropKey(preview, box)` so the
// effect can dedupe across re-renders. `colourForLabel` is shared
// between the reference chip rendering and the imports overlay.

function cropKey(preview: string, box: { x0: number; y0: number; x1: number; y1: number }): string {
  return `${preview}|${Math.round(box.x0)},${Math.round(box.y0)},${Math.round(box.x1)},${Math.round(box.y1)}`;
}

// Tile chips + canvas overlays. Goes through the project-scoped
// palette assignment so two labels in the SAME project can never
// share a visually-similar colour, and the same label across the
// workspace + public cards picks the same colour as long as the
// project's tag list is consistent (which it is, all three
// surfaces read the same /api/projects payload).
//
// Falls back to the cross-view stable hash when the project's
// label set isn't in scope (rare, every render site we know about
// passes the project tags).
function colourForLabel(
  labels: string[],
  lab: string | null,
  overrides?: Record<string, string> | null,
): string {
  if (!lab) return "#888888";
  const key = lab.trim().toLowerCase();
  if (overrides && overrides[key]) return overrides[key];
  const map = buildProjectLabelColourMap(labels, overrides);
  return map.get(key) ?? colourForLabelStable(lab);
}

// V1's bbox-size sanity check, ported to V2. Returns whichever box
// edge ends up smallest after the image is letterboxed into the
// model's target input shape. Anything below BOX_FAIL_PX is too
// small to detect reliably; below BOX_WARN_PX is borderline.
const BOX_FAIL_PX = 12;
const BOX_WARN_PX = 24;

function parseInputShape(s: string): { w: number; h: number } {
  const [w, h] = s.split("x").map((n) => parseInt(n, 10));
  return { w: Number.isFinite(w) ? w : 640, h: Number.isFinite(h) ? h : 640 };
}

function scaledMinSide(
  box: { x0: number; y0: number; x1: number; y1: number },
  imgW: number,
  imgH: number,
  inputShape: string,
): number {
  if (!imgW || !imgH) return Infinity;
  const t = parseInputShape(inputShape);
  const s = Math.min(t.w / imgW, t.h / imgH);
  return Math.min(Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0)) * s;
}

function sizeStatusFor(minSide: number): "ok" | "warn" | "fail" {
  if (minSide < BOX_FAIL_PX) return "fail";
  if (minSide < BOX_WARN_PX) return "warn";
  return "ok";
}

// (cosine + nearestLabel were used by the old client-side relabel
// resolver; both are obsolete now that the backend computes label
// centroids and applies the relabel/reject rules itself.)

// ─── Imports (Drop media) ─────────────────────────────────────────────────────

type ImportDetection = {
  box: [number, number, number, number];
  mask: MaskShape | null;
  // Set by the viewer's 30s memory-strip when it nulls `mask`. Lets the
  // first-edit editedBoxes derivation tell a strip (refetch pending) from
  // a genuinely mask-less detection, so it won't persist null masks.
  _maskStripped?: boolean;
  embedding: number[];
  // The open-vocab detector produced this candidate.
  gdLabel: string | null;       // canonical user label (post-canonicalise)
  gdVariant: string | null;     // raw phrase that fired (synonym)
  gdScore: number | null;       // detector confidence
  // The validator is informational only in V2, runs alongside the
  // detector + embedding so the popup can show "what would the
  // validator call this?" as a third opinion, but the backend's
  // label resolver doesn't use it for the final pred_label decision.
  vlmLabel: string | null;
  vlmScore: number | null;
  vlmMs: number | null;
  // Embedding QC. We compute cosine sim against EVERY project-label
  // centroid (not just the nearest), so we can:
  //   1. Re-label when a non-detector label has stronger embedding match
  //   2. Show the user the closest centroid regardless of pred
  //   3. Compute the score for the FINAL chosen label (rejection gate)
  embedNearestLabel: string | null;       // best-matching centroid label
  embedSimilarity: number | null;         // sim to nearest centroid
  embedSimilarityForLabel: number | null; // sim to predLabel (== finalSim)
  // Final pred label and which signal decided it.
  predLabel: string | null;
  predSource:
    | "gd" | "embed" | "embed-vlm" | "vlm" | "embed-strong"
    | "embed-containment" | "containment" | "overlap" | "manual"
    | null;
  // True when the case sits in a genuinely uncertain zone.
  // Surfaced in the UI so the user can scan for detections worth
  // eyeballing.
  ambiguous: boolean;
  // What the validator (or post-pass) did on this detection.
  vlmAction:
    | "confirm" | "tiebreak" | "disagree" | "relabel"
    | "containment-relabel" | "containment-reject"
    | "containment-duplicate" | "containment-partial"
    | "overlap-multi-mask" | "overlap-low-conf"
    | "three-way-disagree" | "confusion-reject"
    | null;
  // top1 - top2 margin in centroid-cosine space (display only;
  // the threshold checks are server-side).
  embedMargin: number | null;
  // Cosine similarity to EVERY project label's references. Lets
  // the pipeline popup show the full label vote, not just the
  // winner, useful for debugging cases where the prediction sits
  // a hair above its rival. Keys are the project's labels,
  // case-preserved as they came back from the resolver.
  embedSims: Record<string, number> | null;
  // Per-encoder sims breakdown so the popup can show the two
  // encoders' contributions separately. Null when the secondary
  // encoder isn't loaded / scored, the resolver falls back to
  // the primary encoder only and keeps the combined `embedSims`
  // populated.
  embedSimsDino: Record<string, number> | null;
  embedSimsSiglip: Record<string, number> | null;
  siglipWeight: number | null;
  // True when the detection failed any rejection gate.
  // Rejected boxes don't render on the canvas by default, they sit
  // in the pipeline popup with a "rejected" marker; hovering the row
  // reveals their box on the canvas so the user can inspect.
  rejected: boolean;
  rejectReason:
    | "gd" | "embed" | "combined" | "containment" | "overlap"
    | "disagree" | "confusion"
    | null;
  cropDataUrl: string;
};

// Image-level pipeline timings. Detector / segmenter / embedder run
// once per image; validator is per-box (its image-level summary is
// the sum). Stored per
// ImportedMedia so the viewer sidebar can show how long each stage
// took for the currently-open image.
type ImportTimings = {
  gdMs: number | null;
  samMs: number | null;
  embedMs: number | null;
  vlmTotalMs: number | null;
};

type ImportedMedia = {
  id: string;
  // Backend's persisted import_id (returned by POST
  // /api/v2/projects/<id>/imports). Used to address PUT/DELETE on
  // future edits / removals; absent on freshly-uploaded imports
  // until the persist call returns.
  backendId?: string;
  file: File;
  // Persisted filename on the backend's imports/<id>.<ext> path.
  // Set on hydrated imports so we can lazy-fetch the original bytes
  // when the user actually needs them (click-to-detect, re-process)
  // instead of pre-downloading every full-size image at hydration.
  filename?: string;
  preview: string;
  // BlurHash placeholder string from the backend manifest. Decoded
  // by react-blurhash so the dataset tile shows a smooth gradient
  // before the real preview JPEG paints. Null on freshly-uploaded
  // imports that haven't round-tripped to the backend yet.
  blurhash?: string | null;
  status: "processing" | "ready" | "failed";
  width?: number;
  height?: number;
  // Per-label counts coming from /overview + /initial. Replaces the
  // synthetic-detection array that used to drive the chip rail.
  // Real ImportDetection[] in `detections` is only populated when
  // the viewer opens the tile and pulls geometry via
  // /annotations/{import_id}.
  labelStats?: Record<string, number>;
  // Total detection count, kept separately so the box-count badge
  // can render with a single read even when labelStats is missing.
  detectionCount?: number;
  // Derived ("new labels") crop only: the parent's original label, shown as a
  // reference while the user assigns a fresh label. Null/absent otherwise.
  derivedLabel?: string | null;
  detections?: ImportDetection[];
  // User edits to the boxes shown in the BoxEditor, when present
  // this overrides the detections-derived view (so add/edit/delete
  // sticks across re-renders). Initialised lazily on first edit by
  // seeding from the inferred detections.
  editedBoxes?: EditableBox[];
  timings?: ImportTimings;
  error?: string;
  // Upload time as ms-since-epoch, used to sort the dataset gallery
  // newest-first across page reloads. Backend may return a string
  // (ISO) or number; we normalise to ms in the sort comparator.
  createdAt?: number | string | null;
  // Origin URL when the import came from /imports/from_urls
  // (Openverse search). Used by the inline search panel to filter
  // already-imported URLs out of new search results.
  sourceUrl?: string | null;
  // Count of augmentation copies persisted server-side. Gates
  // the per-tile Augmentations icon, only shown when > 0.
  nAugmentations?: number;
  // ms-since-epoch the FE last observed this import gaining detections.
  // Appended as a query param on the labelled_preview URL so the
  // browser refetches the freshly-baked JPEG instead of serving the
  // stale "no detections" version it had cached.
  labelledAt?: number;
  // Per-upload idempotency key. Generated once at placeholder
  // creation and sent on every POST attempt to /imports/raw. The
  // backend dedupes by this key so a network blip that triggers
  // the catch-block retry doesn't create a second backend record
  // for the same user action. Absent on cached / hydrated entries
  // (they're already persisted).
  idempotencyKey?: string;
  // True for tiles seeded from the localStorage importTiles cache that
  // the server hasn't confirmed yet THIS session. The hydration merges
  // clear this the moment /initial or /overview returns the matching
  // id; any tile still provisional after the full /overview load is a
  // ghost (deleted server-side, but the stale cache still listed it) and
  // gets dropped. Without this, ghosts lingered as white squares that
  // couldn't be deleted (DELETE 404s) and kept retrying their preview.
  provisional?: boolean;
};

// Narrow ImportedMedia → StoreImport. The store deliberately holds a
// slim record per import so per-id subscribers (tile chips, count
// badges) don't re-render when chunky fields like detections or
// editedBoxes change. Anything chunky stays in the legacy useState
// array until the matching consumer migrates onto the store's
// geometryCache (P3 / P4 work).
function toStoreImport(m: ImportedMedia): StoreImport {
  return {
    id: m.id,
    backendId: m.backendId,
    filename: m.filename,
    preview: m.preview,
    blurhash: m.blurhash,
    status: m.status,
    width: m.width,
    height: m.height,
    createdAt: m.createdAt,
    sourceUrl: m.sourceUrl,
    nAugmentations: m.nAugmentations,
    labelStats: m.labelStats,
    detectionCount: m.detectionCount,
    // hasGeometry is set later by the viewer when /annotations
    // lands; for the mirror-writes path we infer it from whether
    // any detection currently carries a mask.
    hasGeometry: Array.isArray(m.detections) && m.detections.some((d) => d?.mask != null),
    hasEdits: Array.isArray(m.editedBoxes) && m.editedBoxes.length > 0,
    labelledAt: m.labelledAt,
  };
}

// Snake_case wire shape returned by /api/v2/imports/process and
// persisted verbatim via /api/v2/projects/<id>/imports. Defined as
// a top-level type so processImport AND the manifest-hydration path
// can both unwrap it through the same `unwrapWireDetection` helper.
type WireDetection = {
  box: number[];
  mask: MaskShape | null;
  embedding: number[];
  gd_label: string | null;
  gd_variant?: string | null;
  gd_score: number | null;
  vlm_label?: string | null;
  vlm_score?: number | null;
  vlm_ms?: number | null;
  pred_label?: string | null;
  pred_source?:
    | "gd" | "embed" | "embed-vlm" | "vlm" | "embed-strong"
    | "embed-containment" | "containment" | "overlap" | "manual"
    | null;
  embed_nearest_label?: string | null;
  embed_nearest_sim?: number | null;
  embed_sim_for_label?: number | null;
  embed_margin?: number | null;
  rejected?: boolean;
  reject_reason?:
    | "gd" | "embed" | "combined" | "containment" | "overlap"
    | "disagree" | "confusion"
    | null;
  ambiguous?: boolean;
  vlm_action?:
    | "confirm" | "tiebreak" | "disagree" | "relabel"
    | "containment-relabel" | "containment-reject"
    | "containment-duplicate" | "containment-partial"
    | "overlap-multi-mask" | "overlap-low-conf"
    | "three-way-disagree" | "confusion-reject"
    | null;
  embed_sims?: Record<string, number>;
  embed_sims_dino?: Record<string, number>;
  embed_sims_siglip?: Record<string, number>;
  siglip_weight?: number;
  crop_jpg_b64?: string;
};

// Lazy original-bytes fetch. Hydrated imports start with an empty
// File placeholder (we don't pre-download megabytes of imagery on
// project open any more), the click-to-detect / segment-box /
// classify-box flows still need the original pixels though, so the
// first call here fetches and memoises the real File. Subsequent
// calls reuse the cached result. Falls back to the placeholder if
// the network fails so callers don't have to special-case errors.
// Lazy original-bytes fetch for reference images. Same pattern as
// ensureMediaFile, refs hydrate from the manifest with an empty File
// placeholder so the grid paints instantly; this fetches and memoises
// the real File the first time an interactive flow needs the bytes
// (segment-box, classify-box, point-detect, embed-crops flush).
// Falls back to the placeholder on failure so callers don't have to
// special-case errors.
async function ensureRefFile(ref: ReferenceImage): Promise<File> {
  if (ref.file && ref.file.size > 0) return ref.file;
  if (!ref.preview) return ref.file;
  try {
    const r = await fetch(ref.preview);
    if (!r.ok) return ref.file;
    const blob = await r.blob();
    const file = new File(
      [blob],
      ref.file.name || ref.filename || "image",
      { type: blob.type || "image/jpeg" },
    );
    // Mutate in place so subsequent flows that read ref.file get the
    // upgraded version without a state round-trip. The ReferenceImage
    // entry in `refs` shares this object reference.
    ref.file = file;
    return file;
  } catch {
    return ref.file;
  }
}

async function ensureMediaFile(media: ImportedMedia): Promise<File> {
  if (media.file && media.file.size > 0) return media.file;
  if (!media.preview) {
    console.warn(
      "[ensureMediaFile] no preview URL on media, returning empty stub",
      { id: media.id, filename: media.filename },
    );
    return media.file;
  }
  // Use apiFetch so the call carries the backend auth bearer the same
  // way other reads do. Plain `fetch` works when the endpoint is
  // anonymous, but anything behind require_project_owner would 401
  // silently and we'd hand back a 0-byte stub with no clue why.
  try {
    const r = await apiFetch(media.preview);
    if (!r.ok) {
      console.warn(
        `[ensureMediaFile] http ${r.status} ${r.statusText} fetching ${media.preview}`,
      );
      return media.file;
    }
    const blob = await r.blob();
    if (blob.size === 0) {
      console.warn(
        `[ensureMediaFile] empty blob from ${media.preview}`,
      );
      return media.file;
    }
    const file = new File(
      [blob],
      media.file.name || media.filename || "image",
      { type: blob.type || "image/jpeg" },
    );
    // Mutate in place so subsequent flows that read media.file get
    // the upgraded version without needing a state round-trip. The
    // ImportedMedia entry in `imports` shares this object reference,
    // so the upgrade is visible to all callers.
    media.file = file;
    return file;
  } catch (e) {
    console.warn(
      `[ensureMediaFile] fetch threw for ${media.preview}:`, e,
    );
    return media.file;
  }
}

function unwrapWireDetection(d: WireDetection): ImportDetection {
  return {
    box: [d.box[0], d.box[1], d.box[2], d.box[3]],
    mask: d.mask,
    embedding: d.embedding,
    gdLabel: d.gd_label,
    gdVariant: d.gd_variant ?? null,
    gdScore: d.gd_score,
    vlmLabel: d.vlm_label ?? null,
    vlmScore: d.vlm_score ?? null,
    vlmMs: d.vlm_ms ?? null,
    embedNearestLabel: d.embed_nearest_label ?? null,
    embedSimilarity: d.embed_nearest_sim ?? null,
    embedSimilarityForLabel: d.embed_sim_for_label ?? null,
    embedMargin: d.embed_margin ?? null,
    predLabel: d.pred_label ?? d.gd_label ?? null,
    predSource: d.pred_source ?? (d.gd_label ? "gd" : null),
    rejected: !!d.rejected,
    rejectReason: d.reject_reason ?? null,
    ambiguous: !!d.ambiguous,
    vlmAction: d.vlm_action ?? null,
    embedSims: d.embed_sims ?? null,
    embedSimsDino: d.embed_sims_dino ?? null,
    embedSimsSiglip: d.embed_sims_siglip ?? null,
    siglipWeight: typeof d.siglip_weight === "number" ? d.siglip_weight : null,
    cropDataUrl: d.crop_jpg_b64 ? `data:image/jpeg;base64,${d.crop_jpg_b64}` : "",
  };
}

// ─── Dataset gallery (V1-style) ───────────────────────────────────────────────
// Replaces the old "Imports" section. Look + feel mirrors V1's
// ProjectView dataset gallery: 3-col responsive thumbnail grid,
// progressive load, V1-style ImageThumb cards. Clicking a thumb
// opens DatasetViewer (BoxEditor-based modal, also matching V1).

const DATASET_PAGE_SIZE = 15;
// Hard ceiling on how far the gallery auto-expands its render window
// while images are processing. Dropping thousands of images marks them
// all "processing"; without a cap the window grew to that full count and
// rendered thousands of DatasetThumb components at once (and re-rendered
// them all every time the labelling overlay moved to the next image),
// which froze the page. The progress bar communicates overall status;
// the user scrolls to load more beyond this window.
const DATASET_MAX_AUTO_DISPLAY = DATASET_PAGE_SIZE * 8;

function DatasetGallery({
  items,
  labels,
  hasReferenceEmbeddings,
  isGeneralDataset = false,
  isPulling = false,
  onRemove,
  onRemoveBatch,
  onMediaChange,
  projectId,
  labellingFilename = null,
  onAddProjectLabel,
  labelAliases = {},
  labelColours = null,
  inputShape = "256x256",
  highlightedImportId = null,
  highlightKind = null,
  readOnly = false,
  totalImports = null,
  verdicts = {},
  onVerdict,
  verdictFilter = "all",
  onVerdictFilterChange,
  onStartReview,
  filterCountsOverride = null,
  manifestUpdatedAt = null,
  onViewerOpenIdChange,
  refViewerOpen = false,
}: {
  items: ImportedMedia[];
  /** Fired with the currently-open import id whenever the viewer
      modal mounts, navigates, or closes (null on close). The
      outer uses it to skip the open tile when the memory-pressure
      eviction pass nulls in-RAM mask polygons. */
  onViewerOpenIdChange?: (id: string | null) => void;
  /** True while the reference editor overlay is open (owned by
      RefImageGrid); hides the back-to-top button so it doesn't float
      over the editor chrome. */
  refViewerOpen?: boolean;
  /** Manifest's updatedAt for the active project, threaded down to
      the viewer so its IDB cache lookups can gate freshness on it.
      Bumped by every BE write to the manifest (label-job complete,
      verdict save, edit-box PUT) so a stale IDB row never lands in
      state. */
  manifestUpdatedAt?: string | null;
  /** Authoritative chip counts pre-computed by the backend over the
      FULL imports list. When non-null DatasetGallery skips the
      per-item iteration and renders these directly so the ALL /
      UNLABELLED / GOOD / BAD chips show their final numbers on
      first paint instead of climbing as batches stream in. */
  filterCountsOverride?: {
    all: number;
    unlabelled: number;
    unrated: number;
    good: number;
    bad: number;
    unsure: number;
  } | null;
  /** Verdict for each backend-persisted import (good/bad/unsure).
      Used by the filter pills + Review button on the gallery
      header, plus the per-tile coloured ring. */
  verdicts?: Record<string, "good" | "bad" | "unsure">;
  /** Toggle the verdict on a single import. Forwarded to the
      DatasetViewer so the manual good/bad/unsure pills in the
      viewer header can mutate the same map Review mode writes to.
      Toggling the same verdict clears it (parity with Review mode). */
  onVerdict?: (id: string, v: "good" | "bad" | "unsure") => void;
  /** Active filter pill. "all" shows everything; the others narrow
      the gallery to images matching that bucket. */
  verdictFilter?: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
  onVerdictFilterChange?: (next: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure") => void;
  /** Opens the ReviewModeV2 overlay walking the picked scope. */
  onStartReview?: (scope: "unrated" | "good" | "bad" | "unsure" | "all") => void;
  /** Backend's reported imports count for the project. Used to
      drive the "Dataset N" counter + render skeleton slots for
      tiles whose data hasn't been merged into `items` yet (i.e.
      the imports hydration remainder is still in flight). Null
      means we don't know yet, fall back to items.length. */
  totalImports?: number | null;
  labels: string[];
  hasReferenceEmbeddings: boolean;
  /** True when the dataset has no reference images (auto-derived
      "general" behaviour). When set, the "No reference embeddings"
      warning is suppressed: references are optional, so the warning
      would just confuse the user. */
  isGeneralDataset?: boolean;
  /** True while a freshly-created derived (child) project is still having its
      crops generated server-side: shows a "pulling images" loader instead of
      an empty gallery. */
  isPulling?: boolean;
  onRemove: (id: string) => void;
  /** Batch-delete callback. Used by the gallery's Select mode so
      a 50-image purge hits a single backend round-trip instead of
      fanning out 50 DELETEs through onRemove. */
  onRemoveBatch?: (ids: string[]) => Promise<void> | void;
  /** Apply a partial update to a media item. Used by the viewer to
      persist user-drawn boxes, mask edits, and rejected→active
      restorations back to the parent's `imports` array. Pass
      `{hydration: true}` for server-data splices (annotation fetch /
      prefetch) so a response landing mid-gesture can't overwrite the
      user's in-flight editedBoxes. */
  onMediaChange?: (
    id: string,
    patch: Partial<ImportedMedia>,
    opts?: { hydration?: boolean },
  ) => void;
  /** Backend project UUID, needed by the viewer's click-to-detect
      so it can route through /api/v2/imports/detect_point with the
      project's reference centroids and dataset_type. */
  projectId?: string | null;
  /** Filename of the import currently being processed by an active
      label_charlie job. The matching tile gets a "Labelling…" overlay
      so the user can see which one is in flight. */
  labellingFilename?: string | null;
  /** Forwarded to DatasetViewer, called when the user types a
      label in the BoxEditor that isn't already in the project's
      tag list. */
  onAddProjectLabel?: (label: string) => void;
  /** Map from canonical (lowercased) label → user-renamed display
      name. Tile chips honour it so a project-level rename cascades
      to every image's chip immediately without touching the
      underlying detection records. */
  labelAliases?: Record<string, string>;
  /** Per-label colour overrides ({canonical_lower: "#rrggbb"}).
      Propagated to tile chips + viewer + BoxEditor so a colour
      change in Settings repaints everything without a refresh. */
  labelColours?: Record<string, string> | null;
  /** Forwarded to AugmentationsViewer so each tile's hover overlay
      can paint bboxes green/orange/red based on size validation
      against the project's chosen input shape. */
  inputShape?: string;
  /** Currently highlighted import (driven by clicks on the dataset
      stats variation plot). Matching thumb stays at full opacity,
      the rest dim, plus a glow ring for visibility. */
  highlightedImportId?: string | null;
  /** "image" → just highlight. "augmentation" → also flash a small
      "augmentation" badge over the thumb cover so the user knows
      they came here from a plotted augmentation dot. */
  highlightKind?: "image" | "augmentation" | null;
  /** True when the gallery is rendering inside the public read-only
      project view. Suppresses bulk-select, per-tile delete X, and
      makes click-to-open mount the read-only viewer. */
  readOnly?: boolean;
}) {
  const [viewIdx, setViewIdx] = useState<number | null>(null);
  // Hover-prefetch: fire the annotation fetch when the user's pointer
  // enters a tile so the labels are usually already arriving by the
  // time they click. Uses a ref-based in-flight set to avoid duplicate
  // concurrent fetches for the same media id.
  const hoverPrefetchInFlightRef = useRef<Set<string>>(new Set());
  const prefetchAnnotation = useCallback((media: ImportedMedia) => {
    if (!projectId || !media.backendId || !onMediaChange) return;
    const existing = media.detections ?? [];
    if (existing.length > 0 && existing.some((d) => d?.mask)) return;
    if (hoverPrefetchInFlightRef.current.has(media.id)) return;
    hoverPrefetchInFlightRef.current.add(media.id);
    void apiFetch(
      `/api/v2/projects/${projectId}/annotations/${encodeURIComponent(media.backendId)}`,
    )
      .then(async (r) => {
        if (!r.ok) return;
        const body = (await r.json()) as {
          detections?: WireDetection[];
          editedBoxes?: EditableBox[] | null;
          timings?: ImportTimings;
        };
        const dets = (body.detections ?? []).map(unwrapWireDetection);
        const patch: Partial<ImportedMedia> = {
          detections: dets as unknown as ImportDetection[],
        };
        if (Array.isArray(body.editedBoxes)) {
          patch.editedBoxes = body.editedBoxes.map(stripTransientBoxFlags);
        }
        if (body.timings) patch.timings = body.timings as ImportTimings;
        onMediaChange(media.id, patch, { hydration: true });
      })
      .catch(() => {})
      .finally(() => {
        hoverPrefetchInFlightRef.current.delete(media.id);
      });
  }, [projectId, onMediaChange]);
  // Augmentations viewer state, keyed by import_id + the
  // source preview URL the modal renders at the top. Lives in the
  // gallery so the modal mounts once instead of per-tile.
  const [augViewer, setAugViewer] = useState<{ importId: string; sourceUrl: string; filename: string } | null>(null);
  const [displayLimit, setDisplayLimit] = useState(DATASET_PAGE_SIZE);
  // Free-text filter on the image name. Matches the original upload
  // filename (carried on m.file.name when imports hydrate from the
  // backend) and the stored filename, so a user can jump straight to a
  // flight/frame in a big dataset (e.g. "F46" or "frame_0077").
  const [nameQuery, setNameQuery] = useState("");
  // Filter the gallery list by the active verdict pill before any
  // pagination so "showing N of M" reflects the filtered slice. An
  // "unlabelled" tile is one that has no detections yet; "unrated"
  // means there's no fast-review verdict.
  // P7: tell the outer which import the viewer is currently
  // showing so the heap-pressure evict pass can skip it. Closes
  // back to null when the user dismisses the viewer.
  // (Defined just below filteredItems so the dep is in scope.)
  const filteredItems = useMemo(() => {
    // Name search first, so the "shown / total" count + every downstream
    // filter operate on the searched subset.
    const q = nameQuery.trim().toLowerCase();
    const byName = q
      ? items.filter((m) =>
          (m.file?.name || m.filename || "").toLowerCase().includes(q),
        )
      : items;
    if (verdictFilter === "all") return byName;
    return byName.filter((m) => {
      const bid = m.backendId;
      const hasDetections = !!(m.detections && m.detections.length > 0)
        || !!(m.editedBoxes && m.editedBoxes.length > 0);
      if (verdictFilter === "unlabelled") return !hasDetections;
      if (!bid) return false;
      const v = verdicts[bid];
      if (verdictFilter === "unrated") return !v;
      if (verdictFilter === "good") return v === "good";
      if (verdictFilter === "bad") return v === "bad";
      if (verdictFilter === "unsure") return v === "unsure";
      return true;
    });
  }, [items, verdictFilter, verdicts, nameQuery]);
  const visible = filteredItems.slice(0, displayLimit);
  // P7 hook-up - surface the open viewer's import id to the outer
  // pressure-eviction pass so it can skip stripping that one tile's
  // masks while the user is actively looking at them.
  useEffect(() => {
    if (!onViewerOpenIdChange) return;
    if (viewIdx == null) {
      onViewerOpenIdChange(null);
      return;
    }
    const m = filteredItems[viewIdx];
    onViewerOpenIdChange(m?.id ?? null);
  }, [viewIdx, filteredItems, onViewerOpenIdChange]);
  // Counts each pill would show (after applying every filter except
  // its own), so the user can see at a glance whether any bucket has
  // images. Drives pill visibility, buckets with 0 items are hidden
  // (parity with V1's filter row).
  const filterCounts = useMemo(() => {
    // Prefer the BE-pre-computed override when available - it scans
    // the FULL imports list (not just whatever's currently merged
    // into the prop), so the chips read their final numbers
    // immediately even while the imports remainder is still
    // streaming in.
    if (filterCountsOverride) return filterCountsOverride;
    const c = { all: 0, unlabelled: 0, unrated: 0, good: 0, bad: 0, unsure: 0 };
    for (const m of items) {
      c.all++;
      const hasDetections = !!(m.detections && m.detections.length > 0)
        || !!(m.editedBoxes && m.editedBoxes.length > 0);
      if (!hasDetections) c.unlabelled++;
      const v = m.backendId ? verdicts[m.backendId] : undefined;
      if (m.backendId && !v && hasDetections) c.unrated++;
      if (v === "good") c.good++;
      else if (v === "bad") c.bad++;
      else if (v === "unsure") c.unsure++;
    }
    return c;
  }, [items, verdicts, filterCountsOverride]);
  // Effective gallery length, backend total when known, else
  // whatever's currently in state. Drives the infinite-scroll cap so
  // scrolling can render skeleton slots ahead of the imports
  // remainder fetch landing.
  const effectiveLength = verdictFilter === "all"
    ? (totalImports ?? items.length)
    : filteredItems.length;
  // Sentinel for infinite scroll. Sits just under the gallery; when
  // it enters the viewport we bump displayLimit by another page.
  // Replaces the explicit "Show more" button so the user can keep
  // scrolling without clicking.
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (effectiveLength <= displayLimit) return;
    const el = loadMoreRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setDisplayLimit((n) => Math.min(effectiveLength, n + DATASET_PAGE_SIZE));
            break;
          }
        }
      },
      // rootMargin pre-fetches one viewport before the sentinel is
      // actually visible so the next batch is on screen by the time
      // the user reaches it, feels seamless on fast scrolls.
      { rootMargin: "600px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [effectiveLength, displayLimit]);

  // Auto-expand displayLimit whenever there are in-flight processing
  // tiles, so a bulk add reveals the freshly-dropped tiles instead of
  // hiding them behind the scroll gate. CAPPED at DATASET_MAX_AUTO_
  // DISPLAY: dropping thousands of images marks them all "processing",
  // and growing the window to that full count rendered thousands of
  // tiles at once (then re-rendered them all on every labelling-overlay
  // step) which froze the page. The cap keeps the window bounded; the
  // progress bar shows overall status and the user can scroll for more.
  // Never shrinks below where the user already scrolled.
  useEffect(() => {
    let processingCount = 0;
    for (const m of items) {
      if (m.status === "processing") processingCount++;
    }
    if (processingCount === 0) return;
    const want = Math.min(processingCount + DATASET_PAGE_SIZE, DATASET_MAX_AUTO_DISPLAY);
    setDisplayLimit((cur) => Math.max(cur, want));
  }, [items]);

  // Auto-expand displayLimit when the parent fires a jump-to-import
  // event for an item that's hidden behind the "Show more" gate.
  // Without this the parent's scrollIntoView call finds no matching
  // [data-import-id="..."] and silently no-ops, leaving the user
  // staring at the spot they clicked from instead of jumping.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail as { importId?: string; projectId?: string } | null;
      if (!d?.importId) return;
      // Match on backendId, that's what the variation plot sends
      // and what every tile writes into data-import-id.
      const idx = items.findIndex((m) => m.backendId === d.importId);
      if (idx < 0) return;
      setDisplayLimit((cur) => (idx >= cur ? idx + 1 : cur));
    };
    window.addEventListener("pixelkit-ensure-import-visible", handler);
    return () => window.removeEventListener("pixelkit-ensure-import-visible", handler);
  }, [items]);

  // Bulk-select mode. While true each tile renders a checkbox in
  // its top-left corner and clicks toggle selection instead of
  // opening the viewer. The floating action bar pinned to the
  // bottom of the viewport carries the Delete + Cancel actions.
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  // IDs currently fading out. They stay in the DOM with opacity → 0
  // for ~280ms so the deletion feels smooth instead of snapping.
  const [deletingIds, setDeletingIds] = useState<Set<string>>(() => new Set());
  const DELETE_ANIM_MS = 320;
  const exitSelection = useCallback(() => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  }, []);
  const toggleSelected = useCallback((id: string) => {
    setSelectedIds((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  const selectAllVisible = useCallback(() => {
    setSelectedIds((cur) => {
      const next = new Set(cur);
      // Only operate on persisted backend rows, in-flight uploads
      // don't have a backendId yet and can't be deleted server-side.
      for (const m of items) if (m.backendId) next.add(m.id);
      return next;
    });
  }, [items]);
  const selectedCount = selectedIds.size;
  const deleteSelected = useCallback(async () => {
    if (selectedCount === 0 || !onRemoveBatch) return;
    setBulkBusy(true);
    // Start the fade. The tiles stay mounted but transition to
    // opacity 0 + a small inward scale; once the animation has had
    // time to play we fire the actual removal so the items vanish
    // cleanly from the DOM.
    const idsSnapshot = Array.from(selectedIds);
    setDeletingIds(new Set(idsSnapshot));
    await new Promise((resolve) => setTimeout(resolve, DELETE_ANIM_MS));
    try {
      await onRemoveBatch(idsSnapshot);
      exitSelection();
    } finally {
      setBulkBusy(false);
      setDeletingIds(new Set());
    }
  }, [selectedCount, selectedIds, onRemoveBatch, exitSelection]);

  // No card wrapper, title and grid sit directly on the page like
  // the references / annotations sections so the dataset feels like
  // part of the project layout rather than a contained widget.
  // Pills shown in the header. Only buckets with images render, an
  // empty "Unsure" bucket would just be a dead pill. "All" is always
  // visible; the others appear once any image lands in them.
  const filterPills: Array<{ key: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure"; label: string }> = [
    { key: "all", label: "All" },
    { key: "unlabelled", label: "Unlabelled" },
    { key: "unrated", label: "Unrated" },
    { key: "good", label: "Good" },
    { key: "bad", label: "Bad" },
    { key: "unsure", label: "Unsure" },
  ];
  const visiblePills = filterPills.filter(({ key }) =>
    key === "all" ? true : filterCounts[key] > 0
  );
  const showReview = !readOnly && filterCounts.all > 0;

  return (
    <>
      <div className="flex items-center justify-between gap-3 mb-1 flex-wrap">
        <h2 className="text-2xl font-medium tracking-tight text-[var(--foreground)] shrink-0">
          Dataset
          <span className="ml-3 text-sm text-foreground/40 font-mono tabular-nums">
            {/* When the user has filtered, show "shown / total" so the
                pill effect on the gallery is legible. Otherwise the
                backend total wins, it's correct from first paint
                while the imports remainder fetch is still streaming. */}
            {verdictFilter === "all" && !nameQuery.trim()
              ? (totalImports ?? items.length)
              : `${filteredItems.length} / ${totalImports ?? items.length}`}
          </span>
        </h2>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Search the dataset by image name (original upload filename).
              Pure client-side filter over the already-loaded import list. */}
          {items.length > 0 && (
            <div className="relative">
              <svg
                viewBox="0 0 24 24"
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-foreground/40"
                fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden
              >
                <circle cx="11" cy="11" r="7" />
                <path d="M21 21l-4.3-4.3" />
              </svg>
              <input
                type="text"
                value={nameQuery}
                onChange={(e) => setNameQuery(e.target.value)}
                placeholder="Search image names"
                aria-label="Search images by name"
                className="w-48 rounded-full border border-foreground/10 bg-foreground/[0.025] pl-8 pr-7 py-1.5 text-[12px] text-[var(--foreground)] outline-none transition-colors focus:border-foreground/30 placeholder:text-foreground/40"
              />
              {nameQuery && (
                <button
                  type="button"
                  onClick={() => setNameQuery("")}
                  aria-label="Clear search"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-foreground/40 hover:text-foreground transition-colors"
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                    <path d="M18 6 6 18M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          )}

          {/* Filter pills, labelled "Filter" and enclosed in their
              own rounded chip so the affordance reads clearly as a
              "narrow the gallery" control, distinct from the Review
              action button on the right. Hidden when only "All"
              would render (single-pill row is pointless). */}
          {!readOnly && visiblePills.length > 1 && (
            <div className="flex items-center gap-2 rounded-full border border-foreground/10 bg-foreground/[0.025] pl-3 pr-1.5 py-1">
              <span className="inline-flex items-center gap-1.5 text-[12px] font-medium text-foreground/55">
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
                </svg>
                Filter
              </span>
              <div className="flex items-center gap-1 flex-wrap">
                {visiblePills.map(({ key, label }) => {
                  const active = verdictFilter === key;
                  const count = filterCounts[key];
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onVerdictFilterChange?.(key)}
                      className={[
                        "rounded-full px-3 py-1 text-[12px] font-medium border transition-colors",
                        active
                          ? "bg-foreground text-background border-[var(--foreground)]"
                          : "border-transparent text-foreground/65 hover:bg-foreground/[0.06] hover:text-foreground",
                      ].join(" ")}
                    >
                      {label}
                      <span className={["ml-1 tabular-nums", active ? "opacity-70" : "opacity-50"].join(" ")}>
                        {count}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Review button, opens the fast-review modal walking the
              picked scope. The label flips to match the active pill
              so the user can read "Review unrated" / "Review good"
              etc.; "all" pill defaults the scope to "unrated". */}
          {showReview && onStartReview && (
            <ReviewLauncher
              counts={{
                unrated: filterCounts.unrated,
                good: filterCounts.good,
                bad: filterCounts.bad,
                unsure: filterCounts.unsure,
                all: filterCounts.all,
              }}
              activeFilter={verdictFilter}
              onStartReview={onStartReview}
            />
          )}

          {!readOnly && !hasReferenceEmbeddings && !isGeneralDataset && (
            <span className="text-[11px] text-amber-700 dark:text-amber-200/80 font-mono uppercase tracking-wider">
              No reference embeddings
            </span>
          )}
          {!readOnly && onRemoveBatch && items.some((m) => m.backendId) && (
            selectionMode ? (
              <button
                type="button"
                onClick={exitSelection}
                className="rounded-full border border-foreground/15 px-3.5 py-1.5 text-[12px] font-medium text-foreground/70 hover:border-foreground/35 hover:text-foreground transition-colors"
              >
                Done
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setSelectionMode(true)}
                className="inline-flex items-center gap-1.5 rounded-full border border-foreground/15 px-3.5 py-1.5 text-[12px] font-medium text-foreground/70 hover:border-foreground/35 hover:text-foreground transition-colors"
                title="Bulk delete"
              >
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                  <path d="M10 11v6M14 11v6" />
                </svg>
                Select
              </button>
            )
          )}
        </div>
      </div>
      <div
        className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6"
        // Dim every tile when a stats-driven highlight is active.
        // The matching DatasetThumb pins its own opacity back to 1
        // via the `isHighlighted` prop below.
        style={highlightedImportId ? { opacity: 0.95 } : undefined}
      >
        {visible.map((m, i) => {
          const matched = highlightedImportId != null && m.backendId === highlightedImportId;
          return (
            <DatasetThumb
              key={m.id}
              media={m}
              labels={labels}
              projectId={projectId}
              onOpen={() => {
                if (selectionMode) toggleSelected(m.id);
                else setViewIdx(i);
              }}
              onHover={readOnly ? undefined : () => prefetchAnnotation(m)}
              onRemove={() => onRemove(m.id)}
              isLabelling={
                labellingFilename != null && m.filename === labellingFilename
              }
              labelAliases={labelAliases}
              labelColours={labelColours}
              selectionMode={selectionMode}
              selected={selectedIds.has(m.id)}
              onToggleSelected={() => toggleSelected(m.id)}
              deleting={deletingIds.has(m.id)}
              setAugViewer={setAugViewer}
              isHighlighted={matched}
              isDimmed={highlightedImportId != null && !matched}
              augmentationFlash={matched && highlightKind === "augmentation"}
              readOnly={readOnly}
            />
          );
        })}
        {/* Skeleton placeholders for slots whose data hasn't been
            merged into `items` yet (imports hydration remainder
            still streaming in). Only relevant when no filter is
            active, under a verdict filter every visible tile is
            already known to match, so padding the grid with
            skeletons just shows ghost tiles that aren't real. */}
        {verdictFilter === "all" && totalImports != null && visible.length < Math.min(displayLimit, totalImports) && (
          Array.from({
            length: Math.min(displayLimit, totalImports) - visible.length,
          }).map((_, i) => (
            <div
              key={`ds-skel-${visible.length + i}`}
              aria-hidden
              className="relative rounded-xl border border-foreground/10 bg-foreground/[0.02] overflow-hidden"
            >
              <div className="h-[200px] bg-[var(--surface)]" />
              <div className="p-3 space-y-2">
                <div className="h-3.5 w-2/3 rounded bg-foreground/[0.04]" />
                <div className="h-3 w-1/3 rounded bg-foreground/[0.04]" />
              </div>
            </div>
          ))
        )}
      </div>

      {isPulling && visible.length === 0 && (
        <div className="mt-6 flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-foreground/15 bg-foreground/[0.02] px-6 py-16 text-center">
          <span className="h-7 w-7 animate-spin rounded-full border-2 border-foreground/15 border-t-[var(--accent-orange)]" aria-hidden />
          <div className="text-[13px] font-medium text-foreground/70">Pulling cropped images from the parent project…</div>
          <div className="text-[12px] text-foreground/50">This can take a moment for large datasets; images appear as they are cropped.</div>
        </div>
      )}

      {/* Sentinel for the infinite-scroll observer. Uses totalImports
          instead of items.length so scrolling extends the grid past
          the already-merged tiles, the displayLimit bump then
          renders more skeleton slots that fill in as the data
          arrives. Without this, on a 941-image project the user
          scrolled to tile 100 and the sentinel reported "no more
          tiles" because items.length was still 100 even though the
          remainder fetch was about to deliver 841 more. */}
      {effectiveLength > displayLimit && (
        <div
          ref={loadMoreRef}
          className="mt-6 flex justify-center text-[11px] uppercase tracking-wider font-mono text-foreground/35"
        >
          Loading more…
        </div>
      )}

      {/* Render the viewer through a React portal so it mounts as a
          direct child of <body>. Without this, the viewer's
          `fixed inset-0` is contained by the dataset section's
          animation wrapper (which has a `transform` from the rise()
          helper), CSS rules say any transformed ancestor becomes
          the containing block for fixed-position descendants, so the
          modal ends up sized to the section's `max-w-6xl` width
          instead of the viewport. The portal escapes that. */}
      {viewIdx !== null && filteredItems[viewIdx] && typeof window !== "undefined" &&
        createPortal(
          // viewIdx is the index within `filteredItems` (the verdict-
          // filtered list the gallery actually renders). Pass that
          // same array to the viewer so clicking the first Bad tile
          // opens the first Bad image, not items[0] from the full
          // unfiltered list; prev/next then walk through the active
          // filter only, matching the user's mental model of "I'm
          // reviewing the bad ones".
          <DatasetViewer
            items={filteredItems}
            index={viewIdx}
            labels={labels}
            onClose={() => setViewIdx(null)}
            onPrev={() => setViewIdx((i) => (i === null ? null : Math.max(0, i - 1)))}
            onNext={() => setViewIdx((i) => (i === null ? null : Math.min(filteredItems.length - 1, i + 1)))}
            onMediaChange={onMediaChange}
            projectId={projectId}
            onAddProjectLabel={onAddProjectLabel}
            labelAliases={labelAliases}
            labelColours={labelColours}
            readOnly={readOnly}
            verdicts={verdicts}
            onVerdict={onVerdict}
            onDelete={!readOnly ? onRemove : undefined}
            manifestUpdatedAt={manifestUpdatedAt}
            inputShape={inputShape}
          />,
          document.body,
        )
      }

      {/* Back-to-top, only mounted when neither viewer modal is
          open. The button is meant to return the user to the top
          of the gallery scroll, but the image-viewer and
          augmentations-viewer modals both cover the gallery -
          the floating affordance over their chrome was confusing. */}
      {viewIdx === null && !augViewer && !refViewerOpen && <ScrollToTop />}

      {augViewer && projectId && (
        <AugmentationsViewer
          projectId={projectId}
          importId={augViewer.importId}
          sourceUrl={augViewer.sourceUrl}
          filename={augViewer.filename}
          inputShape={inputShape}
          onClose={() => setAugViewer(null)}
        />
      )}

      {/* Floating action bar, pinned to the bottom of the viewport
          while Select mode is on. Portalled so it doesn't get
          clipped by the gallery section's animation wrapper. */}
      {/* Always-dark pill, the bar sits at the bottom of the
          viewport over whatever's on screen, so pin the chrome
          to a fixed dark backdrop and use white-on-dark text
          regardless of theme. */}
      {selectionMode && typeof window !== "undefined" &&
        createPortal(
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[400] flex items-center gap-2 rounded-full border border-white/20 bg-black/85 backdrop-blur-md px-3 py-2 shadow-[var(--shadow-strong)]">
            <span className="text-xs text-white/80 px-2 tabular-nums">
              {selectedCount} selected
            </span>
            <button
              type="button"
              onClick={selectAllVisible}
              disabled={bulkBusy}
              className="rounded-full border border-white/20 bg-white/[0.06] px-3 py-1 text-[11px] text-white/85 hover:border-white/45 hover:text-white transition-colors disabled:opacity-40"
            >
              Select all
            </button>
            <button
              type="button"
              onClick={() => void deleteSelected()}
              disabled={bulkBusy || selectedCount === 0}
              className="rounded-full bg-red-500 text-white px-4 py-1 text-xs font-semibold hover:bg-red-400 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {bulkBusy ? "Deleting…" : `Delete ${selectedCount}`}
            </button>
            <button
              type="button"
              onClick={exitSelection}
              disabled={bulkBusy}
              className="rounded-full border border-white/20 px-3 py-1 text-[11px] text-white/65 hover:border-white/45 hover:text-white transition-colors disabled:opacity-40"
            >
              Cancel
            </button>
          </div>,
          document.body,
        )
      }
    </>
  );
}

// Module-scoped FIFO queue factory. Each named queue serialises a
// distinct class of image loads so they can't interfere with each
// other (dataset thumbs vs ref tiles vs anything else added later).
// Each tile that uses a queue, once it's near the viewport, asks for
// a slot and waits its turn. The queue drains at most
// `concurrency` tiles at a time so the backend isn't asked to serve
// N images in parallel.
//
// IntersectionObserver still gates ENROLLMENT, so even with hundreds
// of items only the visible / near-viewport ones contend for slots.
function makeImageLoadQueue(concurrency: number) {
  const queue: Array<() => void> = [];
  let inFlight = 0;
  return function acquire(): Promise<() => void> {
    return new Promise((resolve) => {
      const release = () => {
        inFlight = Math.max(0, inFlight - 1);
        const next = queue.shift();
        if (next) {
          inFlight += 1;
          next();
        }
      };
      if (inFlight < concurrency) {
        inFlight += 1;
        resolve(release);
      } else {
        queue.push(() => resolve(release));
      }
    });
  };
}

// Dataset tiles fetch the backend's ~600 px labelled_preview JPEG,
// which is per-image rate-limited server-side (one bake per image,
// then disk-cached). Running 6 concurrent fetches is safe, the
// backend's per-image render lock dedupes concurrent calls for the
// SAME image, and different images are independent. The old
// concurrency=1 was leaving 5/6 of the browser's HTTP connection
// budget idle; the visible grid now fills in roughly 6× faster on
// first paint.
const acquireDatasetImgSlot = makeImageLoadQueue(6);
// References are file-served originals; same parallelism budget.
const acquireRefImgSlot = makeImageLoadQueue(6);

// Reference-tile image with queued lazy load + BlurHash placeholder
// crossfade. Sits inside RefImageGrid; pulled out so the queue + IO
// state doesn't have to be hand-written into every chip.
function LazyRefImage({
  src,
  blurhash,
  isProcessing,
  onClick,
}: {
  src: string;
  blurhash?: string | null;
  isProcessing: boolean;
  onClick?: () => void;
}) {
  const tileRef = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);
  const [canLoad, setCanLoad] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const releaseRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (inView) return;
    const node = tileRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    // Non-sticky: inView reflects whether the tile is currently within
    // the 1500-px near-margin. The unload effect below uses the
    // false-edge to drop the image bytes so a 964-image gallery's
    // off-screen refs don't keep their <img> elements consuming
    // compositor memory.
    const obs = new IntersectionObserver(
      (entries) => {
        const e = entries[entries.length - 1];
        if (!e) return;
        setInView(e.isIntersecting);
      },
      { rootMargin: "1500px" },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!inView || canLoad || !src) return;
    let cancelled = false;
    acquireRefImgSlot().then((release) => {
      if (cancelled) { release(); return; }
      releaseRef.current = release;
      setCanLoad(true);
    });
    return () => { cancelled = true; };
  }, [inView, canLoad, src]);

  // Off-screen UNLOAD, matches the DatasetThumb pattern.
  useEffect(() => {
    if (inView) return;
    if (!canLoad) return;
    const timer = window.setTimeout(() => {
      setCanLoad(false);
      setLoaded(false);
      if (releaseRef.current) { releaseRef.current(); releaseRef.current = null; }
    }, 600);
    return () => window.clearTimeout(timer);
  }, [inView, canLoad]);

  useEffect(() => () => {
    if (releaseRef.current) { releaseRef.current(); releaseRef.current = null; }
  }, []);

  useEffect(() => {
    setLoaded(false);
    setCanLoad(false);
    if (releaseRef.current) { releaseRef.current(); releaseRef.current = null; }
  }, [src]);

  const release = () => {
    if (releaseRef.current) { releaseRef.current(); releaseRef.current = null; }
  };

  return (
    <div ref={tileRef} className="absolute inset-0">
      {/* Placeholder layer, BlurHash gradient when the backend
          gave us one, otherwise a neutral animated gradient so
          the tile is NEVER empty while the image is loading.
          Stays underneath the real image and crossfades out once
          it loads. */}
      {blurhash ? (
        <BlurhashCanvas
          hash={blurhash}
          width={32}
          height={32}
          punch={1}
          className="absolute inset-0 w-full h-full"
          style={{
            opacity: loaded ? 0 : 1,
            transition: "opacity 250ms ease-out",
          }}
        />
      ) : (
        <div
          className="absolute inset-0 w-full h-full"
          style={{
            opacity: loaded ? 0 : 1,
            transition: "opacity 250ms ease-out",
            background: "linear-gradient(135deg, #1a1a1c 0%, #232328 50%, #1a1a1c 100%)",
            backgroundSize: "200% 200%",
            animation: "shimmer 2s ease-in-out infinite",
          }}
          aria-hidden
        />
      )}
      {canLoad && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt=""
          className={[
            "absolute inset-0 w-full h-full object-cover transition-all duration-300",
            isProcessing ? "scale-105 blur-[1px]" : "cursor-pointer",
          ].join(" ")}
          style={{
            opacity: loaded && !isProcessing ? 1 : isProcessing ? 0.35 : 0,
          }}
          loading="lazy"
          decoding="async"
          onLoad={() => { setLoaded(true); release(); }}
          onError={() => { release(); }}
          onClick={onClick}
        />
      )}
    </div>
  );
}

// V1-style fixed-height thumb. Image fills 200px high, footer has
// filename + detection count + status pill (processing / failed /
// no-detections). Clicking the card opens the viewer.
// ─── Augmentations viewer ───────────────────────────────────────
// Convert a #rrggbb (or #rgb) hex to an rgba string at the given
// opacity. Used by the augmentation hover overlay so polygons get
// a translucent fill in the project's label colour. Falls back to
// the input as-is if it doesn't parse, SVG will then ignore the
// invalid fill and the stroke still reads.
function hexToRgba(hex: string, alpha: number): string {
  let h = (hex || "").trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  if (h.length !== 6 || /[^0-9a-fA-F]/.test(h)) return hex;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ─── Review launcher ─────────────────────────────────────────────
// Compact button + scope dropdown that lives on the dataset header.
// One-click opens the fast-review modal walking the bucket matching
// the active filter pill (or "unrated" when no filter is active);
// the dropdown lets the user pick a different bucket without
// flipping the pill first.

function ReviewLauncher({
  counts,
  activeFilter,
  onStartReview,
}: {
  counts: { unrated: number; good: number; bad: number; unsure: number; all: number };
  activeFilter: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
  onStartReview: (scope: "unrated" | "good" | "bad" | "unsure" | "all") => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  // Default scope tracks the active pill so the primary action
  // matches what the user is looking at. The "All" and "Unlabelled"
  // pills fall back to "unrated" (the most common review entry).
  const defaultScope: "unrated" | "good" | "bad" | "unsure" =
    activeFilter === "good" ? "good"
    : activeFilter === "bad" ? "bad"
    : activeFilter === "unsure" ? "unsure"
    : "unrated";
  const defaultCount = counts[defaultScope];

  const scopes: Array<{ key: "unrated" | "good" | "bad" | "unsure"; label: string }> = [
    { key: "unrated", label: "Unrated" },
    { key: "good", label: "Good" },
    { key: "bad", label: "Bad" },
    { key: "unsure", label: "Unsure" },
  ];

  return (
    <div ref={ref} className="relative inline-flex items-stretch">
      <button
        type="button"
        onClick={() => defaultCount > 0 && onStartReview(defaultScope)}
        disabled={defaultCount === 0}
        className="inline-flex items-center gap-1.5 rounded-l-full bg-foreground text-background border border-[var(--foreground)] px-3.5 py-1.5 text-[12px] font-medium capitalize hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
        title={defaultCount === 0
          ? "No images in this bucket"
          : `Swipe left = bad, right = good. Walk the ${defaultScope} set.`}
      >
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M15 18l-6-6 6-6" opacity="0.45" />
          <path d="M9 18l6-6-6-6" />
        </svg>
        Review {defaultScope}
        <span className="opacity-70 tabular-nums">{defaultCount}</span>
      </button>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex items-center justify-center rounded-r-full bg-foreground text-background border border-l-0 border-[var(--foreground)] px-2.5 py-1.5 hover:opacity-90 transition-opacity"
        title="Pick review scope"
      >
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
          style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 160ms ease" }}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute right-0 top-full mt-1.5 min-w-[10rem] rounded-xl border border-foreground/15 bg-[var(--surface)] shadow-[var(--shadow-strong)] backdrop-blur-md overflow-hidden z-20"
        >
          {scopes.map(({ key, label }) => {
            const n = counts[key];
            const disabled = n === 0;
            return (
              <button
                key={key}
                type="button"
                disabled={disabled}
                onClick={() => { setOpen(false); onStartReview(key); }}
                className={[
                  "w-full flex items-center justify-between gap-3 px-3 py-2 text-[12px] text-left",
                  disabled
                    ? "text-foreground/30 cursor-not-allowed"
                    : "text-[var(--foreground)] hover:bg-foreground/[0.06]",
                ].join(" ")}
                role="option"
                aria-selected={key === defaultScope}
              >
                <span>{label}</span>
                <span className="font-mono tabular-nums opacity-65">{n}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Per-tile modal. Click the augmentations button on a dataset
// thumb to open. Backdrop dims + blurs in, original sits at the
// top, augmented copies tile underneath. Lists are pulled from
// /api/v2/projects/{id}/augmentations/{import_id}.
function AugmentationsViewer({
  projectId,
  importId,
  sourceUrl,
  filename,
  inputShape,
  onClose,
}: {
  projectId: string;
  importId: string;
  sourceUrl: string;
  filename: string;
  /** Project's currently-selected input shape (e.g. "256x256").
      Drives the per-box size-validation colouring on the hover
      overlay, green = OK, orange = borderline, red = too small
      once the image is downscaled into the target shape. */
  inputShape: string;
  onClose: () => void;
}) {
  const [items, setItems] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Cache-bust token. The backend serves augmentation JPEGs with
  // `Cache-Control: max-age=86400`, so even after Update overwrites
  // them on disk the browser keeps showing the old bytes, which
  // misaligns with the freshly-warped annotations.json. Refreshing
  // this token on mount + on the augmentations-generated event
  // forces the browser to re-fetch with a new URL.
  const [cacheBuster, setCacheBuster] = useState<number>(() => Date.now());
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onChanged = (e: Event) => {
      const d = (e as CustomEvent).detail as { projectId?: string } | null;
      if (!d || d.projectId !== projectId) return;
      setCacheBuster(Date.now());
    };
    window.addEventListener("pixelkit-augmentations-generated", onChanged);
    return () => window.removeEventListener("pixelkit-augmentations-generated", onChanged);
  }, [projectId]);
  // Per-copy annotations keyed by filename ("00.jpg" → list of
  // {label, polys, box}). Polygons are already in the augmented
  // image's coordinate system, the runner warps them through the
  // same geometric matrix the image went through, so SVG over the
  // augmented JPEG lines up regardless of perspective / scale /
  // rotation.
  const [augAnnos, setAugAnnos] = useState<{
    width: number;
    height: number;
    copies: Record<string, { label: string; polys: number[][][]; box: number[] }[]>;
  } | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augmentations/${importId}?v=${cacheBuster}`);
        if (!r.ok) throw new Error(`http ${r.status}`);
        const data = await r.json() as { items?: string[] };
        if (!alive) return;
        setItems(data.items ?? []);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { alive = false; };
  }, [projectId, importId, cacheBuster]);

  // Pre-baked per-copy warped polygons. Backend's
  // augment_generate writes one annotations.json per import after
  // each batch finishes; the runner threads every geometric warp
  // through the polygon coords so what we render here always
  // lines up.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augmentations/${importId}/annotations?v=${cacheBuster}`);
        if (!alive) return;
        if (!r.ok) return;
        const data = await r.json() as {
          width?: number;
          height?: number;
          copies?: Record<string, { label?: string; polys?: number[][][]; box?: number[] }[]>;
        };
        if (!alive) return;
        // Normalise, fill in defaults so the renderer doesn't
        // have to keep guarding.
        const normCopies: Record<string, { label: string; polys: number[][][]; box: number[] }[]> = {};
        for (const [name, arr] of Object.entries(data.copies ?? {})) {
          normCopies[name] = (arr || []).map((a) => ({
            label: String(a.label || ""),
            polys: (a.polys ?? []) as number[][][],
            box: (a.box ?? []) as number[],
          }));
        }
        setAugAnnos({
          width: data.width ?? 0,
          height: data.height ?? 0,
          copies: normCopies,
        });
      } catch {
        /* annotations are optional, tiles still render without */
      }
    })();
    return () => { alive = false; };
  }, [projectId, importId, cacheBuster]);

  // Lock the page behind the viewer so wheel scrolling doesn't
  // bleed through to the dataset gallery underneath.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Delete a single augmentation tile. Optimistically strips it
  // from `items` so the grid reflows immediately, then fires DELETE
  // and broadcasts the augmentations-changed event so the dataset
  // gallery's icon updates if this was the last copy.
  const removeAug = async (name: string) => {
    setItems((cur) => (cur ?? []).filter((n) => n !== name));
    setAugAnnos((cur) => {
      if (!cur) return cur;
      const nextCopies = { ...cur.copies };
      delete nextCopies[name];
      return { ...cur, copies: nextCopies };
    });
    try {
      await apiFetch(
        `/api/v2/projects/${projectId}/augmentations/${importId}/${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      // Nudge the dataset gallery to refresh its per-tile icon ,
      // if this was the last copy the n_augmentations dropped to 0
      // and the icon should hide.
      try {
        window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
          detail: { projectId },
        }));
      } catch { /* ignore */ }
    } catch {
      // Best-effort, the optimistic remove stays; user can retry.
    }
  };

  if (typeof window === "undefined") return null;

  return createPortal(
    // Same containment as the image editor: the viewer fills the
    // shell's content area only (title bar / status bar / Explorer
    // side bar stay visible and interactive).
    <div
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[700] overflow-auto"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        // Outer modal stays transparent; the backdrop lives in its
        // own sibling layer below so hover/state churn inside the
        // content can't cause the browser to re-rasterise the blur
        // (which read as a black flicker on some tiles).
        animation: "augViewerFadeIn 240ms ease-out both",
      }}
    >
      {/* Dedicated backdrop layer. `fixed inset-0` so it sits behind
          the content but on top of the page. `pointer-events-none`
          so clicks still fall through to the outer modal's
          click-to-close. Putting the blur on this isolated layer
          (rather than the modal container itself) stops the
          browser re-rasterising it whenever the content above
          re-renders, that was reading as a black flicker on
          hover / delete. */}
      <div
        aria-hidden
        className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] pointer-events-none"
        style={{
          background: "rgb(var(--background-rgb) / 0.78)",
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
          isolation: "isolate",
          willChange: "backdrop-filter",
        }}
      />
      <div
        className="px-6 lg:px-10 py-10 relative"
        onClick={(e) => e.stopPropagation()}
        style={{ animation: "augViewerRiseIn 280ms cubic-bezier(0.2, 0.7, 0.2, 1) both" }}
      >
        <div className="mb-6 min-w-0">
          {/* Eyebrow + close X share one row so the X sits next to the
              "Augmentations" label, not floating down beside the
              filename below. Stronger border-foreground in light mode
              so the chip stays visible on the near-white surface. */}
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs uppercase tracking-wider text-foreground/45">Augmentations</div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="h-7 w-7 grid place-items-center rounded-full border border-foreground/30 dark:border-foreground/15 text-foreground/70 dark:text-foreground/65 hover:border-foreground/55 hover:text-foreground transition-colors shrink-0"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6 6 18" /><path d="m6 6 12 12" />
              </svg>
            </button>
          </div>
          <h2 className="text-2xl font-medium tracking-tight text-[var(--foreground)] truncate">{filename}</h2>
        </div>

        {/* Original at the top, small, centred. The augmentation
            grid below is wider + bigger so the user spends most
            of their attention there. */}
        <div className="flex justify-center mb-8">
          <div className="rounded-2xl overflow-hidden border border-foreground/10 bg-[var(--surface-2)] max-w-sm w-full">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={sourceUrl} alt={filename} className="block w-full h-auto" draggable={false} />
            <div className="px-4 py-2 text-[11px] uppercase tracking-wider font-mono text-foreground/45 text-center">
              Original
            </div>
          </div>
        </div>

        <div className="text-xs uppercase tracking-wider text-foreground/45 mb-3">
          {items === null ? "Loading…" : `${items.length} ${items.length === 1 ? "augmentation" : "augmentations"}`}
        </div>

        {error && (
          <div className="rounded-xl border border-red-400/25 bg-red-500/[0.08] px-3 py-1.5 text-[11px] text-red-200 mb-4">
            {error}
          </div>
        )}

        {items !== null && items.length === 0 && (
          <div className="rounded-2xl border border-dashed border-foreground/10 bg-foreground/[0.02] px-5 py-12 text-center">
            <div className="text-sm text-foreground/65 mb-1">No augmentations generated yet</div>
            <div className="text-[11px] text-foreground/35">
              Set a per-image count + enable some augmentations, then click Update on the Augmentations tab.
            </div>
          </div>
        )}

        {items !== null && items.length > 0 && (
          <div className="grid gap-4 md:grid-cols-2">
            {items.map((name) => (
              <AugmentationTile
                key={name}
                src={`${API}/api/v2/projects/${projectId}/augmentations/${importId}/${encodeURIComponent(name)}?v=${cacheBuster}`}
                name={name}
                annotations={augAnnos?.copies[name] ?? []}
                canvasW={augAnnos?.width ?? 0}
                canvasH={augAnnos?.height ?? 0}
                inputShape={inputShape}
                onDelete={() => removeAug(name)}
              />
            ))}
          </div>
        )}

        <div className="pb-8" />
      </div>
      <style>{`
        @keyframes augViewerFadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes augViewerRiseIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>,
    document.body,
  );
}

// Single augmentation tile with a hover overlay that paints the
// import's polygon outlines + label chips on top. SVG sits inside
// the same aspect-ratio frame as the image so coords scale
// correctly. Annotations come from the original source image so
// they may not perfectly align with geometric warps, close
// enough to read which object is which.
function AugmentationTile({
  src,
  name,
  annotations,
  canvasW,
  canvasH,
  inputShape,
  onDelete,
}: {
  src: string;
  name: string;
  /** Per-copy warped polygons + boxes, already in the augmented
      image's pixel space. canvasW/H is the working dimensions
      the runner saved them at, usually identical to the
      augmented JPEG's natural size. */
  annotations: { polys: number[][][]; box: number[]; label: string }[];
  canvasW: number;
  canvasH: number;
  /** Target input shape (e.g., "256x256"). Each box's colour is
      driven by how its smallest edge survives the downscale into
      this shape: green = OK, orange = borderline, red = too small. */
  inputShape: string;
  /** Optional delete handler, when present, a × button appears
      in the top-right corner of the tile so the user can drop
      augmentations they don't want. */
  onDelete?: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const hasOverlay = canvasW > 0 && canvasH > 0 && annotations.length > 0;
  return (
    <div
      // Same frame chrome as the original-image card up top ,
      // larger rounding, darker fill, so the augmentation grid
      // visually matches the row above it.
      className="rounded-2xl overflow-hidden border border-foreground/10 bg-[var(--surface-2)] relative group"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Inner wrapper sits inside the border so the SVG overlay
          exactly covers the rendered image area, otherwise the 1 px
          border offsets the SVG vs. the img and polygons drift
          ~1 px down-and-right at high scales. */}
      <div className="relative">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={name}
        className="block w-full h-auto"
        loading="lazy"
        decoding="async"
        draggable={false}
      />
      {onDelete && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          aria-label="Delete this augmentation"
          title="Delete this augmentation"
          // Sits over the augmentation thumbnail, keep dark bubble
          // + white icon regardless of theme.
          className="absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-full bg-black/65 backdrop-blur-md text-white/90 hover:bg-red-500/85 hover:text-white transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
          </svg>
        </button>
      )}
      {hasOverlay && (
        <svg
          viewBox={`0 0 ${canvasW} ${canvasH}`}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{
            opacity: hovered ? 1 : 0,
            transition: "opacity 200ms ease-out",
          }}
          preserveAspectRatio="none"
          aria-hidden
        >
          {annotations.map((a, i) => {
            const hasBox = a.box.length === 4;
            // Colour reflects whether the bbox survives the downscale
            // into the project's target input shape: green = OK,
            // orange = borderline, red = too small to detect. Matches
            // the BoxEditor's size-validation conventions.
            const minSide = hasBox
              ? scaledMinSide(
                  { x0: a.box[0], y0: a.box[1], x1: a.box[2], y1: a.box[3] },
                  canvasW,
                  canvasH,
                  inputShape,
                )
              : Infinity;
            const status = sizeStatusFor(minSide);
            const colour =
              status === "fail"
                ? "#ef4444"
                : status === "warn"
                ? "#f59e0b"
                : "#22c55e";
            const fillRgba = hexToRgba(colour, 0.28);
            // Pixel sizes baked into the viewBox space so they
            // render at the same CSS pixel size regardless of
            // source resolution (paired with non-scaling-stroke
            // on the strokes). chipH stays in viewBox units for
            // the rect; the text uses non-scaling effects too so
            // labels look uniform across small/large tiles.
            const chipH = Math.max(16, Math.min(canvasW, canvasH) / 28);
            const chipPad = chipH * 0.45;
            const fontSize = chipH * 0.6;
            const labelWidth = Math.max(
              chipH,
              fontSize * 0.62 * Math.max(1, (a.label || "").length) + chipPad * 2,
            );
            const chipY = hasBox && a.box[1] > chipH ? a.box[1] - chipH : (hasBox ? a.box[1] : 0);
            return (
              <g key={i}>
                {a.polys.map((poly, j) => (
                  <polygon
                    key={j}
                    points={poly.map((pt) => `${pt[0]},${pt[1]}`).join(" ")}
                    fill={fillRgba}
                    stroke={colour}
                    strokeWidth={1.5}
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {hasBox && (
                  <rect
                    x={a.box[0]}
                    y={a.box[1]}
                    width={Math.max(0, a.box[2] - a.box[0])}
                    height={Math.max(0, a.box[3] - a.box[1])}
                    fill="none"
                    stroke={colour}
                    strokeWidth={1.5}
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {hasBox && a.label && (
                  <g>
                    <rect
                      x={a.box[0]}
                      y={chipY}
                      width={labelWidth}
                      height={chipH}
                      rx={chipH * 0.5}
                      ry={chipH * 0.5}
                      fill={colour}
                    />
                    <text
                      x={a.box[0] + chipPad}
                      y={chipY + chipH * 0.7}
                      fill="#0a0a0a"
                      fontSize={fontSize}
                      fontWeight={600}
                      style={{ letterSpacing: "-0.01em" }}
                    >
                      {a.label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      )}
      </div>
      <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider font-mono text-foreground/65 truncate flex items-center justify-between gap-2 bg-[var(--surface)]">
        <span className="truncate">{name}</span>
        {hasOverlay && (
          <span
            className="shrink-0 text-foreground/55 transition-opacity"
            style={{ opacity: hovered ? 0 : 1 }}
          >
            hover for annotations
          </span>
        )}
      </div>
    </div>
  );
}

function DatasetThumb({
  media,
  labels,
  projectId,
  onOpen,
  onHover,
  onRemove,
  isLabelling = false,
  labelAliases = {},
  labelColours = null,
  selectionMode = false,
  selected = false,
  onToggleSelected,
  deleting = false,
  setAugViewer,
  isHighlighted = false,
  isDimmed = false,
  augmentationFlash = false,
  readOnly = false,
}: {
  media: ImportedMedia;
  labels: string[];
  projectId?: string | null;
  onOpen: () => void;
  onHover?: () => void;
  onRemove: () => void;
  /** True when the active label_charlie job is currently processing
      this tile. Drives the "Labelling…" overlay that sits on top of
      the thumbnail until the next image starts (clearing this back
      to false). */
  isLabelling?: boolean;
  /** Canonical-lowercase → display-name map. Applied at chip render
      so a project-level label rename cascades to every image's chip
      without touching the underlying detection records. */
  labelAliases?: Record<string, string>;
  /** Per-label colour overrides ({canonical_lower: "#rrggbb"}).
      Same propagation pattern as labelAliases, chip backgrounds
      repaint immediately when Settings saves a new colour. */
  labelColours?: Record<string, string> | null;
  /** True when the gallery is in bulk-select mode, the tile shows
      a checkbox in the top-left corner and clicks toggle selection
      via onToggleSelected instead of opening the viewer. */
  selectionMode?: boolean;
  selected?: boolean;
  onToggleSelected?: () => void;
  /** True for tiles in the middle of a bulk-delete animation. The
      tile fades opacity → 0 + scales slightly inward; the parent
      removes it from state once the animation has had time to play. */
  deleting?: boolean;
  /** Opens the augmentations viewer modal for this tile.
      Owned by DatasetGallery so the modal renders once at the
      gallery level and isn't re-mounted as the user pages through
      tiles. */
  setAugViewer?: (next: { importId: string; sourceUrl: string; filename: string } | null) => void;
  /** True when this tile is the target of a stats-driven jump. The
      tile stays at full opacity, picks up a glow ring + scale-up
      animation, and pre-empts the gallery's dim wash. */
  isHighlighted?: boolean;
  /** True for every OTHER tile while a highlight is active ,
      drops opacity to ~0.4 so the matched tile is the only thing
      the user's eye lands on. */
  isDimmed?: boolean;
  /** When true (and the tile is highlighted), flash a short
      "augmentation" badge over the cover photo so the user knows
      they came here from a plotted augmentation dot rather than
      the original image itself. */
  augmentationFlash?: boolean;
  /** True when the gallery is in public read-only mode. Hides the
      per-tile delete X so visitors can't remove someone else's
      project images. */
  readOnly?: boolean;
}) {
  const displayLabel = (canonical: string): string => {
    const k = canonical.trim().toLowerCase();
    return labelAliases[k] || canonical;
  };
  // P2: subscribe to the per-id store record when the flag is on so
  // this tile re-renders only when its OWN slim record (labelStats /
  // detectionCount / labelledAt) flips. Flag-off path is identical
  // to before - useImport(null) returns undefined and we fall back
  // to the media prop everywhere. Full geometry (detections,
  // editedBoxes) still flows through the prop until P3 puts it on
  // the store too.
  const storeView = useImport(STORE_V2_ENABLED ? media.id : null);
  // Use the store's labelStats only once it has actually aggregated some (a
  // non-empty map); otherwise fall back to the import's own labelStats from the
  // initial /overview load. Without this, the store's empty `{}` shadowed the
  // loaded stats and the tile chip rail stayed blank until a hover lazy-loaded
  // the geometry.
  const labelStatsSrc =
    STORE_V2_ENABLED && storeView?.labelStats && Object.keys(storeView.labelStats).length > 0
      ? storeView.labelStats
      : media.labelStats;
  const detectionCountSrc =
    STORE_V2_ENABLED && storeView?.detectionCount !== undefined
      ? storeView.detectionCount
      : media.detectionCount;
  const labelledAtSrc =
    STORE_V2_ENABLED && storeView?.labelledAt !== undefined
      ? storeView.labelledAt
      : media.labelledAt;
  // Counts and label-chips reflect ACCEPTED detections only, rejected
  // ones don't make it onto the gallery thumb (matches the canvas
  // behaviour in the viewer where they're hidden by default).
  const acceptedDetections = useMemo(
    () => (media.detections ?? []).filter((d) => !d.rejected),
    [media.detections],
  );
  const rejectedCount = (media.detections?.length ?? 0) - acceptedDetections.length;
  // Box-source selection: user edits win when present. Same priority
  // BoxEditor uses internally so the tile chip rail reflects the
  // user's relabel work, not the resolver's original prediction.
  // Without this, relabelling a box in the viewer changed the
  // BoxEditor but the tile chip kept showing the auto-detected
  // label until a page refresh.
  const editedLabels = useMemo<string[] | null>(() => {
    if (media.editedBoxes === undefined) return null;
    return media.editedBoxes.map((b) => b.label).filter(Boolean);
  }, [media.editedBoxes]);
  // Prefer the compact labelStats record from /overview when it's
  // present AND the FE hasn't received real detection geometry yet.
  // The user-edit path still wins (above) because edits are the
  // ground-truth chip source whenever they exist. Falls through to
  // the legacy synthetic-detection counting when labelStats is
  // unavailable (old backends or non-hydrated entries).
  const labelStatsCount = useMemo(() => {
    if (!labelStatsSrc) return null;
    let s = 0;
    for (const v of Object.values(labelStatsSrc)) s += v;
    return s;
  }, [labelStatsSrc]);
  const detectionCount = editedLabels !== null
    ? editedLabels.length
    : labelStatsSrc && acceptedDetections.length === 0
    ? labelStatsCount ?? detectionCountSrc ?? 0
    : acceptedDetections.length;
  const labelsInImage = useMemo(() => {
    const set = new Set<string>();
    if (editedLabels !== null) {
      editedLabels.forEach((lab) => { if (lab) set.add(lab); });
    } else if (labelStatsSrc && acceptedDetections.length === 0) {
      // Read from labelStats while the tile is in its placeholder
      // (no synthetic detections) state. Once /annotations lands
      // and real geometry replaces the placeholder, the regular
      // acceptedDetections branch takes over.
      for (const lab of Object.keys(labelStatsSrc)) {
        if (lab) set.add(lab);
      }
    } else {
      acceptedDetections.forEach((d) => { if (d.predLabel) set.add(d.predLabel); });
    }
    return Array.from(set);
  }, [acceptedDetections, editedLabels, labelStatsSrc]);
  // Image-level "unsure" flag, any kept detection the resolver
  // (and post-fusion check) couldn't commit to. Drives the corner
  // pill below so the user can spot review-needed tiles at a
  // glance, even before they open the viewer.
  const hasUnsureDetection = useMemo(
    () => acceptedDetections.some((d) => d.ambiguous),
    [acceptedDetections],
  );

  // For hydrated imports we serve a small labelled-preview JPEG
  // (~30 KB, baked once on the backend with darkened bg + tinted
  // segmented objects). For freshly-uploaded imports that haven't
  // round-tripped yet we fall back to the local blob preview.
  // The `labelledAt` query param invalidates the browser cache when
  // detections appear mid-job, without it the tile keeps showing
  // the cached no-detection JPEG until the user hard-refreshes.
  // Retry counter so img.onError can trigger a fresh GET (the
  // url changes via the &retry= param, which forces the browser
  // past its cached error response). Capped at 3 - beyond that
  // the failure is structural and retrying just churns the
  // network.
  const [previewRetry, setPreviewRetry] = useState(0);
  const previewUrl = media.backendId && projectId
    ? `${API}/api/v2/projects/${projectId}/imports/${media.backendId}/labelled_preview${
        labelledAtSrc || previewRetry
          ? `?v=${labelledAtSrc ?? 0}${previewRetry ? `&retry=${previewRetry}` : ""}`
          : ""
      }`
    : media.preview;

  // IntersectionObserver gates ENROLLMENT in the load queue so off-
  // screen tiles never contend for slots. rootMargin=1500 px pre-
  // queues tiles well before the user scrolls to them so the grid
  // feels filled-in even on fast scrolls. NON-sticky now: a tile
  // that scrolls FAR off-screen flips inView back to false, which
  // the unload effect below uses to release the image bytes + slot.
  // Without that, a user scrolling through a 964-image gallery
  // ended up with every image they'd passed still painted on the
  // compositor, the stats card open/close + chip rail fade-in
  // animations dropped frames once a few hundred had loaded.
  const tileRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [inView, setInView] = useState(false);
  const [canLoad, setCanLoad] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const releaseSlotRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    const node = tileRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const obs = new IntersectionObserver(
      (entries) => {
        const e = entries[entries.length - 1];
        if (!e) return;
        setInView(e.isIntersecting);
      },
      { rootMargin: "1500px" },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, []);

  // Off-screen UNLOAD: when the tile has been outside the near-
  // viewport margin for a short while, drop its <img>. Debounced so
  // the user flicking back-and-forth past a tile doesn't churn the
  // load queue. The image bytes are CDN-cached so a re-enter
  // re-paints from browser HTTP cache in ~20-50ms.
  useEffect(() => {
    if (inView) return;
    if (!canLoad) return;
    const timer = window.setTimeout(() => {
      setCanLoad(false);
      setLoaded(false);
      if (releaseSlotRef.current) {
        releaseSlotRef.current();
        releaseSlotRef.current = null;
      }
    }, 600);
    return () => window.clearTimeout(timer);
  }, [inView, canLoad]);

  // Acquire a queue slot once we're near the viewport. The promise
  // resolves with a release fn we call on load / error / unmount so
  // the next tile can start. Cancelled flag covers the case where
  // the tile unmounts before the slot opens, release the slot
  // anyway so the queue keeps draining.
  useEffect(() => {
    if (!inView || canLoad || !previewUrl) return;
    let cancelled = false;
    acquireDatasetImgSlot().then((release) => {
      if (cancelled) {
        release();
        return;
      }
      releaseSlotRef.current = release;
      setCanLoad(true);
    });
    return () => { cancelled = true; };
  }, [inView, canLoad, previewUrl]);

  // Always release the slot on unmount, protects against the user
  // navigating away mid-fetch and leaving the queue stuck.
  useEffect(() => () => {
    if (releaseSlotRef.current) {
      releaseSlotRef.current();
      releaseSlotRef.current = null;
    }
  }, []);

  // Re-arm `loaded` when the URL flips (e.g. labelled-preview gets
  // invalidated by an edit and the next request renders fresh bytes).
  useEffect(() => {
    setLoaded(false);
    setCanLoad(false);
    if (releaseSlotRef.current) {
      releaseSlotRef.current();
      releaseSlotRef.current = null;
    }
  }, [previewUrl]);

  // Cached-image completion check. When the browser already has the
  // labelled_preview bytes in its HTTP cache (most common after a
  // scroll-out-then-back, the 600 ms unload tears down the <img> but
  // the bytes stay in the disk cache), the onLoad event can fire
  // BEFORE React attaches the handler -- leaving `loaded` stuck at
  // false and the cross-fade stuck at opacity 0, so the tile shows
  // the blurhash forever despite the image being right there.
  // After the <img> mounts, peek at .complete + .naturalWidth on the
  // next animation frame; if the browser already painted the bytes,
  // fast-path the loaded state ourselves.
  useEffect(() => {
    if (!canLoad || loaded) return;
    const img = imgRef.current;
    if (!img) return;
    const tick = () => {
      if (img.complete && img.naturalWidth > 0) {
        setLoaded(true);
        releaseSlot();
      }
    };
    // First check synchronously after mount; if not ready, peek again
    // on the next frame (covers the small window where the browser
    // resolves a cache hit just after our render).
    tick();
    const raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [canLoad, loaded, previewUrl]);

  const releaseSlot = () => {
    if (releaseSlotRef.current) {
      releaseSlotRef.current();
      releaseSlotRef.current = null;
    }
  };

  return (
    <div
      // data-import-id lets the parent's jump-to-import handler
      // locate this tile via querySelector + scrollIntoView. backendId
      // is the canonical match the stats card sends.
      data-import-id={media.backendId ?? media.id}
      onPointerEnter={onHover}
      className={[
        "relative rounded-xl border bg-foreground/[0.02] overflow-hidden transition-all",
        selected
          ? "border-red-400/70 ring-2 ring-red-400/40"
          : isHighlighted
          ? "border-foreground/45 ring-2 ring-foreground/25"
          : "border-foreground/10 hover:border-foreground/20",
      ].join(" ")}
      style={{
        // Lighter base shadow in both themes (alpha drops from 0.6
        // → 0.18) so the gallery thumbs sit softly on a light page
        // without the heavy halo every card used to carry.
        boxShadow: isHighlighted
          ? "var(--shadow-strong), 0 0 0 6px rgb(var(--foreground-rgb) / 0.06)"
          : "0 2px 10px -6px rgb(var(--shadow-rgb) / 0.18)",
        transitionDuration: "320ms",
        transitionTimingFunction: "cubic-bezier(0.4, 0, 0.2, 1)",
        opacity: deleting ? 0 : isDimmed ? 0.35 : 1,
        transform: deleting
          ? "scale(0.92)"
          : isHighlighted
          ? "scale(1.02)"
          : "scale(1)",
        pointerEvents: deleting ? "none" : undefined,
      }}
    >
      <button
        type="button"
        onClick={onOpen}
        aria-label={selectionMode ? `Toggle selection of ${media.file.name}` : `Open ${media.file.name}`}
        className="block w-full text-left"
      >
        {/* Consistent slightly-landscape (4:3) thumb area. Height tracks the
            column width so the grid stays tidy as the column count changes
            with screen size. */}
        <div ref={tileRef} className="relative aspect-[4/3] bg-[var(--background)] overflow-hidden">
          {/* Augmentation flash badge, only painted briefly when
              the user lands here from clicking an augmentation dot
              on the variation plot. Fades out alongside the
              highlight via the parent's timer. */}
          {augmentationFlash && (
            <div
              className="absolute inset-0 z-20 grid place-items-center pointer-events-none"
              style={{ animation: "augFlashIn 220ms ease-out both" }}
            >
              <div
                className="rounded-full bg-black/65 text-white px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.18em] backdrop-blur-md"
                style={{ boxShadow: "var(--shadow-strong)" }}
              >
                Augmentation
              </div>
              <style>{`
                @keyframes augFlashIn {
                  0%   { opacity: 0; transform: scale(0.94); }
                  60%  { opacity: 1; transform: scale(1.02); }
                  100% { opacity: 1; transform: scale(1); }
                }
              `}</style>
            </div>
          )}
          {/* Placeholder layer, BlurHash when the manifest has one,
              otherwise a flat dark fill. The animated shimmer
              gradient that used to live here looked nice on a few
              tiles but turned the gallery into a CPU hog when 50+
              tiles rendered at once, every tile was repainting a
              moving gradient on the compositor each frame. The
              static fill is indistinguishable from a "no image
              yet" placeholder and the processing-text overlay
              below carries the in-flight signal. */}
          {media.blurhash ? (
            <BlurhashCanvas
              hash={media.blurhash}
              width={32}
              height={18}
              punch={1}
              className="absolute inset-0 w-full h-full"
            />
          ) : (
            <div
              className="absolute inset-0 w-full h-full bg-[var(--surface)]"
              aria-hidden
            />
          )}
          {canLoad && previewUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              ref={imgRef}
              src={previewUrl}
              alt=""
              // will-change is set ONLY while the cross-fade is in
              // flight. Once loaded we drop it so the compositor can
              // collapse the tile's layer back into the parent, on
              // a 964-image gallery, leaving will-change-transform +
              // will-change-[opacity,filter] permanently on every
              // <img> created one GPU layer per tile, and the
              // cumulative cost crushed every other animation
              // (theme toggle, stats card open) once a few dozen
              // had loaded.
              className={[
                "absolute inset-0 w-full h-full object-cover",
                loaded ? "" : "will-change-transform will-change-[opacity,filter]",
              ].join(" ")}
              style={{
                // Cross-fade with a subtle blur→sharp + scale-settle.
                // Material "standard" curve (0.4, 0, 0.2, 1) feels
                // polished without dragging. Single 420 ms duration
                // keeps all three properties in sync.
                opacity: loaded ? 1 : 0,
                filter: loaded ? "blur(0)" : "blur(8px)",
                transform: loaded ? "scale(1)" : "scale(1.03)",
                transition:
                  "opacity 420ms cubic-bezier(0.4,0,0.2,1)," +
                  "filter 420ms cubic-bezier(0.4,0,0.2,1)," +
                  "transform 420ms cubic-bezier(0.4,0,0.2,1)",
              }}
              loading="lazy"
              decoding="async"
              onLoad={() => { setLoaded(true); releaseSlot(); }}
              onError={() => {
                releaseSlot();
                // Auto-retry: the backend's lazy labelled_preview
                // render can transiently 5xx (e.g. concurrent
                // first-render contention, a brief Cloudflare hop
                // blip) and without a retry the tile stays as a
                // blurhash placeholder forever. Bump retry and let
                // the URL change trigger a fresh GET. Backed off
                // exponentially: 350ms / 900ms / 2.5s.
                setPreviewRetry((n) => {
                  if (n >= 3) return n;
                  const next = n + 1;
                  window.setTimeout(() => {
                    setCanLoad(false);
                    window.setTimeout(() => setCanLoad(true), 30);
                  }, 350 * Math.pow(2.4, n));
                  return next;
                });
              }}
            />
          )}
          {/* No transient bounding-box overlay, the only cover image
              for a tile is the backend's labelled_preview JPEG (which
              shows segmentations, not boxes). Pre-bake state just
              renders the placeholder/blurhash until the preview
              arrives, avoiding a box-flash → segmentation transition
              the user found jarring. */}
          {media.status === "processing" && (
            <div className="absolute inset-0 grid place-items-center bg-black/55">
              <span className="text-sm text-foreground/75 font-mono animate-pulse">processing…</span>
            </div>
          )}
          {isLabelling && (
            <div
              className="absolute inset-0 grid place-items-center bg-black/55 pointer-events-none"
              style={{ animation: "fadeIn 220ms ease-out" }}
            >
              <span className="text-sm text-foreground/85 font-mono uppercase tracking-wider animate-pulse">
                Labelling…
              </span>
            </div>
          )}
          {media.status === "failed" && (
            <div
              className="absolute inset-0 grid place-items-center bg-black/65 px-3 text-center"
              title={media.error || "Upload failed"}
            >
              <div className="grid gap-1">
                <span className="text-sm text-red-300 font-mono">failed</span>
                {media.error && (
                  <span className="text-[10px] text-red-200/80 font-mono break-all line-clamp-3">
                    {media.error}
                  </span>
                )}
              </div>
            </div>
          )}
          {/* Top-left overlay row: annotations-icon placeholder
              (always shown when the tile is ready) + status pill
              (Unlabelled / Unsure) sitting alongside when the
              tile has something to flag. Icon is a corner-bracket
              annotation glyph, a visual cue that this is the
              annotations slot before per-image annotation
              previews land. */}
          {media.status === "ready" && (
            <div className="absolute top-2 left-2 flex items-center gap-1.5">
              {media.backendId && projectId && setAugViewer && (media.nAugmentations ?? 0) > 0 && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setAugViewer({ importId: media.backendId!, sourceUrl: media.preview, filename: media.filename ?? media.file.name }); }}
                  // Bubble overlays the thumbnail photo, so it needs
                  // contrast against ARBITRARY image content rather
                  // than the page background. Keep both the bubble
                  // and the glyph as fixed dark-on-white regardless
                  // of theme, themable foreground here would make
                  // the icon black on a dark bubble in light mode.
                  className="h-7 w-7 rounded-full bg-black/55 backdrop-blur-md grid place-items-center text-white/90 hover:bg-black/80 hover:text-white transition-colors"
                  title={`${media.nAugmentations} augmentation${media.nAugmentations === 1 ? "" : "s"}`}
                  aria-label="Augmentations"
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    {/* Sparkle / wand glyph, reads as
                        "augmentations / generated variations". */}
                    <path d="M12 3v3" />
                    <path d="M12 18v3" />
                    <path d="M5 12H2" />
                    <path d="M22 12h-3" />
                    <path d="M6 6l2 2" />
                    <path d="M18 18l-2-2" />
                    <path d="M6 18l2-2" />
                    <path d="M18 6l-2 2" />
                  </svg>
                </button>
              )}
              {/* Hide the Unlabelled chip until the /annotations
                  fetch has populated detections, otherwise every
                  tile flashes Unlabelled on first load between the
                  /overview and /annotations responses, even for
                  images that already have boxes on disk. */}
              {detectionCount === 0 && media.detections !== undefined && (
                <span className="rounded-full bg-amber-300/85 px-2 py-0.5 text-[10px] font-semibold text-black uppercase tracking-wider">
                  Unlabelled
                </span>
              )}
              {detectionCount > 0 && hasUnsureDetection && (
                <span
                  className="rounded-full bg-amber-400/90 px-2 py-0.5 text-[10px] font-semibold text-black uppercase tracking-wider"
                  title="At least one detection is borderline, open to review"
                >
                  Unsure
                </span>
              )}
            </div>
          )}
        </div>

        {/* Footer: filename + detection count + label chips. */}
        <div className="px-3 py-2.5 border-t border-foreground/[0.06]">
          <div className="flex items-center justify-between gap-2 text-[12px]">
            <span className="truncate text-foreground/85">{media.file.name}</span>
            <span className="font-mono tabular-nums text-foreground/50 shrink-0">
              {detectionCount} {detectionCount === 1 ? "box" : "boxes"}
              {rejectedCount > 0 && (
                <span className="ml-1.5 text-amber-300/70" title={`${rejectedCount} rejected by embedding QC`}>
                  · {rejectedCount} rej
                </span>
              )}
            </span>
          </div>
          {labelsInImage.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {labelsInImage.map((lab) => {
                const bg = colourForLabel(labels, lab, labelColours);
                return (
                  <span
                    key={lab}
                    className="inline-block rounded-full px-1.5 py-0.5 text-[9px] font-semibold"
                    style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                  >
                    {displayLabel(lab)}
                  </span>
                );
              })}
            </div>
          )}
          {/* Derived crop reference: the parent project's original label,
              shown muted so it stays visible while the user assigns a fresh
              label in "new labels" mode (and as provenance in inherit mode). */}
          {media.derivedLabel && (
            <div
              className="mt-1.5 flex items-center gap-1 text-[10px] text-foreground/40"
              title={`Original label in the parent project: ${media.derivedLabel}`}
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3 opacity-70 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="9 10 4 15 9 20" /><path d="M20 4v7a4 4 0 0 1-4 4H5" />
              </svg>
              <span className="truncate">from {displayLabel(media.derivedLabel)}</span>
            </div>
          )}
        </div>
      </button>

      {/* Single-image trash, only visible outside select mode AND
          when the gallery is editable. Read-only public viewers can
          neither delete nor enter select mode, so the entire corner
          stays clean of destructive affordances. */}
      {!selectionMode && !readOnly && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          aria-label="Remove image"
          // Sits over the image, keep dark bubble + white icon in
          // both themes so it stays legible against any thumbnail.
          className="absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-full bg-black/55 text-white/80 hover:bg-black/80 hover:text-white transition-colors z-10"
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
          </svg>
        </button>
      )}

      {/* Selection checkbox, top-right corner during bulk mode.
          Replaces the single-image trash X (which is suppressed in
          select mode), so the destructive affordance lives in the
          same corner regardless of mode. Clicking the tile body or
          the checkbox itself toggles selection. Disabled for
          in-flight uploads since they have no backend id yet. */}
      {selectionMode && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onToggleSelected?.(); }}
          disabled={!media.backendId}
          aria-pressed={selected}
          aria-label={selected ? "Unselect image" : "Select image"}
          className={[
            "absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-full transition-colors z-10",
            selected
              ? "bg-red-500 text-[var(--foreground)]"
              : "bg-black/55 text-foreground/70 hover:bg-black/80 hover:text-foreground",
            !media.backendId ? "opacity-30 cursor-not-allowed" : "",
          ].join(" ")}
        >
          {selected ? (
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="5 12 10 17 19 7" />
            </svg>
          ) : (
            <span className="block h-3.5 w-3.5 rounded-full border-2 border-foreground/55" aria-hidden="true" />
          )}
        </button>
      )}
    </div>
  );
}

// Per-detection pipeline-stat sidebar row. Each stage shows label,
// confidence, and milliseconds. SAM is image-level (one segment call
// per image, applied to all boxes) but the timing line is shown on
// every detection so the user can see the per-stage cost at a glance.
// Convert one inference-time ImportDetection into an EditableBox
// that BoxEditor can render. The validation field drives V1's
// sidebar pills, green "Verified" for confident detections, amber
// "Unsure" for ambiguous ones, red "Rejected" for ones the resolver
// dropped (clicking the red pill flips it to manual-verified, V1's
// existing flow).
function detectionToEditableBox(
  d: ImportDetection,
  viewIdx: number,
  detIdx: number,
): EditableBox {
  // Sidebar-pill policy:
  //   • rejected → editor filters them out before this fn ever runs
  //     (user wants them auto-hidden, only visible in the pipeline
  //     popup), so the rejected branch is intentionally absent.
  //   • ambiguous → orange "unsure" chip (kind: "unsure"). One signal
  //     the user actually wants to act on.
  //   • confident → no pill. The previous green "Verified AI" was
  //     visual noise on the dominant case.
  let validation: Validation | null = null;
  if (d.ambiguous || d.vlmAction === "tiebreak" || d.vlmAction === "disagree") {
    validation = {
      match: true,
      confidence: d.embedSimilarityForLabel ?? 0.5,
      reason: "PixelKit flagged this detection as ambiguous. Worth a second look.",
      source: "auto",
      kind: "unsure",
    };
  }
  return {
    id: `imp_${viewIdx}_${detIdx}`,
    label: d.predLabel ?? "?",
    x0: d.box[0],
    y0: d.box[1],
    x1: d.box[2],
    y1: d.box[3],
    score: d.gdScore,
    mask: d.mask,
    validation,
  };
}

// Map a rejectReason → human-readable explanation. Centralised so
// the kept-list, the rejected-list, and any future surfaces all
// describe the same verdict the same way.
function rejectReasonExplanation(d: ImportDetection): string {
  switch (d.rejectReason) {
    case "gd":
      return "Detection signal too weak to keep this box.";
    case "combined":
      return "Both signals on this detection were too weak to keep it.";
    case "containment":
      return d.vlmAction === "containment-duplicate"
        ? "Same-label nest: a more confident inner box was preferred; this looser outer box was rejected as a duplicate."
        : d.vlmAction === "containment-partial"
          ? "Same-label nest: a more confident outer box was preferred; this inner box was rejected as a partial / occluded fragment."
          : "Same-label nest: a smaller box was nested inside this one and no alternative label cleared the threshold.";
    case "overlap":
      return d.vlmAction === "overlap-low-conf"
        ? "Same-label overlap: a peer had higher combined confidence; this lower-confidence box was rejected."
        : "Same-label overlap: this box's mask had a secondary entity; the overlapping single-mask peer was preferred.";
    case "disagree":
      return "PixelKit's signals disagreed on this detection's label; no consensus to commit to.";
    case "confusion":
      return "PixelKit's signals didn't firmly commit on this detection.";
    case "embed":
    default:
      return "Similarity to your references was too low to keep this detection.";
  }
}

// V1-style image viewer modal, frosted backdrop, top header bar with
// filename + nav hints + close, BoxEditor as the main canvas, and a
// sidebar showing the V2 pipeline breakdown (detector / segmenter / validator / similarity
// scores + per-stage milliseconds).
function DatasetViewer({
  items,
  index,
  labels,
  onClose,
  onPrev,
  onNext,
  onMediaChange,
  projectId,
  onAddProjectLabel,
  labelAliases = {},
  labelColours = null,
  readOnly = false,
  verdicts = {},
  onVerdict,
  onDelete,
  manifestUpdatedAt = null,
  inputShape = "256x256",
}: {
  items: ImportedMedia[];
  index: number;
  labels: string[];
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  /** `{hydration: true}` marks a server-data splice (annotation fetch),
      which never overwrites a local editedBoxes copy - see updateImport. */
  onMediaChange?: (
    id: string,
    patch: Partial<ImportedMedia>,
    opts?: { hydration?: boolean },
  ) => void;
  projectId?: string | null;
  /** Called when a user types a label that isn't in the project's
      existing tag list. Parent adds it to editLabels so the
      vocabulary stays consistent across imports. */
  onAddProjectLabel?: (label: string) => void;
  /** Canonical → display map for label aliases. Threaded into
      BoxEditor so sidebar / canvas chips / picker render the
      renamed name without changing the underlying detection
      records. */
  labelAliases?: Record<string, string>;
  /** Per-label colour overrides for canvas tints + sidebar chips. */
  labelColours?: Record<string, string> | null;
  /** Public read-only view of someone else's project. Disables every
      mutation surface in the viewer: relabel-by-keypress, add box,
      delete box, brush, eraser, label add, click-to-detect. The
      visibility toggles (boxes / labels / masks) stay so visitors can
      still browse the annotations as the curator made them. */
  readOnly?: boolean;
  /** Verdict map for the project, keyed by backend import id. Used by
      the viewer header's good / unsure / bad pill row so the active
      verdict paints filled. */
  verdicts?: Record<string, "good" | "bad" | "unsure">;
  /** Toggle a verdict on the currently-viewed image. Same setter
      Review mode uses, so a verdict given here flows into the
      gallery filter chips and the Review queue immediately. */
  onVerdict?: (id: string, v: "good" | "bad" | "unsure") => void;
  /** Delete the current image entirely (same backend delete as the gallery
      tile trash). The viewer advances to the next image, or closes if it was
      the last one. */
  onDelete?: (id: string) => void;
  /** Manifest updatedAt for the active project. Gates IDB-cache
      reads so a server-side mutation invalidates the cached
      annotation rows transparently. */
  manifestUpdatedAt?: string | null;
  /** Target model input shape (e.g. "96x96"). Drives the red/amber/green
      detection-size colouring; MUST match the value passed to the export
      size-filter, or boxes shown green here get silently dropped on
      export at sub-256 targets. */
  inputShape?: string;
}) {
  const media = items[index];
  // Hovered detection row, drives the
  // BoxEditor's `emphasizedBoxId` so the matching annotation on the
  // canvas dims everything else and highlights itself, mirroring a
  // real mouse-hover on the editor.
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  // Real-mouse-hover on the canvas, fed back from BoxEditor via its
  // onHoverChange callback. Used to wire keyboard shortcuts (1-9 to
  // relabel) to the box currently under the cursor.
  const [hoveredCanvasBoxId, setHoveredCanvasBoxId] = useState<string | null>(null);
  useEffect(() => { setHoveredIdx(null); setHoveredCanvasBoxId(null); }, [index]);

  // Per-image annotation cache strategy:
  //   • Open or page-to-tile: fetch /annotations/{import_id} when
  //     the current media doesn't already carry mask polygons.
  //   • Prefetch ±1 neighbors in parallel so a left/right key press
  //     paints from cached state instantly instead of hitting the
  //     network mid-flick.
  //   • TTL: when the viewer moves OFF a tile, schedule a 30 s
  //     timer that strips the mask polygons + sim dicts back off the
  //     local copy. Memory pressure on a 9000-detection project gets
  //     bounded, the user can flick rapidly without every tile they
  //     touched lingering in state, AND coming back within 30 s
  //     re-paints from local cache without a round-trip.
  // The cache check is `existing.some(d => d?.mask)`, placeholder
  // detections (from /initial / /overview) never carry a mask, so
  // anything WITH a mask is real per-image data.
  const annotationFetchKickedRef = useRef<Set<string>>(new Set());
  const annotationClearTimersRef = useRef<Map<string, number>>(new Map());
  const prevViewerIndexRef = useRef<number | null>(null);
  // Per-image fetch status - drives the labels-loading spinner and
  // the user-visible "is the geometry still coming?" cue. We track
  // both pending (request in flight) and the set of ids we've
  // already hydrated at least once so a navigation back doesn't
  // re-flash the spinner needlessly.
  const [pendingAnnotationIds, setPendingAnnotationIds] = useState<Set<string>>(
    () => new Set(),
  );
  const annotationsHydratedRef = useRef<Set<string>>(new Set());
  // When the manifest version changes (re-label, box edits elsewhere), let
  // already-hydrated ids re-apply with the fresh data — otherwise the
  // hydrate-once guard in apply* would pin stale annotations.
  useEffect(() => { annotationsHydratedRef.current.clear(); }, [manifestUpdatedAt]);

  // P3 simpler variant: when the flag is on, collapse the per-image
  // /annotations/{id} call (one per current + ±2 neighbour) into a
  // single batched /v3/viewport?ids=a,b,c request. Cycling between
  // tiles still paints from cached state; the win is one round-trip
  // instead of up to five on a fresh open.
  const VIEWPORT_V3_ENABLED =
    process.env.NEXT_PUBLIC_VIEWPORT_V3 === "1";

  // Splices a per-image annotation payload back onto the matching
  // ImportedMedia. Shared between the per-image fallback path and
  // the v3 batched path.
  const applyAnnotationPayload = useCallback((
    m: ImportedMedia,
    body: {
      detections?: WireDetection[];
      editedBoxes?: EditableBox[] | null;
      timings?: ImportTimings;
    },
  ) => {
    if (!onMediaChange) return;
    // Already hydrated this id for the current manifest version. Re-applying
    // identical server data just churns setImports — and for mask-less rows
    // (whose "already has masks" short-circuit in fetchPerImageAnnotation never
    // trips) it spins an infinite render loop while the viewer is open. The
    // hydrated set is cleared on manifest-version change and on mask-strip, so
    // genuine updates still apply.
    if (annotationsHydratedRef.current.has(m.id)) return;
    const dets = (body.detections ?? []).map(unwrapWireDetection);
    if (dets.length > 0 && !dets.some((d) => d?.mask)) {
      console.warn(
        `[viewer] /annotations for ${m.backendId} returned ${dets.length} ` +
        `detections but none carry mask polygons - manifest record(s) ` +
        `likely pre-date the mask-bake era.`,
      );
    }
    const patch: Partial<ImportedMedia> = {
      detections: dets,
      timings: body.timings,
    };
    if (Array.isArray(body.editedBoxes)) {
      patch.editedBoxes = body.editedBoxes.map(stripTransientBoxFlags);
    }
    onMediaChange(m.id, patch, { hydration: true });
    annotationsHydratedRef.current.add(m.id);
    // P5 write-through: persist the parsed row to IDB so revisits
    // paint from local storage instead of refetching tens of KB
    // per image. No-op when the flag is off.
    if (IDB_CACHE_ENABLED && projectId && m.backendId) {
      void putCachedAnnotation(
        projectId,
        m.backendId,
        {
          detections: patch.detections as unknown[] ?? [],
          editedBoxes: (patch.editedBoxes as unknown[] | undefined) ?? null,
          timings: body.timings,
        },
        manifestUpdatedAt,
      );
    }
  }, [onMediaChange, projectId, manifestUpdatedAt]);

  // P4: same splice but the row arrives already-parsed from the
  // annotations worker, so the heavy unwrap + strip pass is skipped
  // here. Structurally the ParsedRow shape matches what the inline
  // path produces.
  const applyParsedRow = useCallback((
    m: ImportedMedia,
    row: ParsedRow,
  ) => {
    if (!onMediaChange) return;
    if (annotationsHydratedRef.current.has(m.id)) return;
    if (row.detections.length > 0 && !row.detections.some((d) => d?.mask)) {
      console.warn(
        `[viewer] /v3/viewport for ${m.backendId} returned ${row.detections.length} ` +
        `detections but none carry mask polygons - manifest record(s) ` +
        `likely pre-date the mask-bake era.`,
      );
    }
    const patch: Partial<ImportedMedia> = {
      detections: row.detections as unknown as ImportDetection[],
      timings: row.timings as ImportTimings | undefined,
    };
    if (Array.isArray(row.editedBoxes)) {
      patch.editedBoxes = row.editedBoxes as unknown as EditableBox[];
    }
    onMediaChange(m.id, patch, { hydration: true });
    annotationsHydratedRef.current.add(m.id);
    if (IDB_CACHE_ENABLED && projectId && m.backendId) {
      void putCachedAnnotation(
        projectId,
        m.backendId,
        {
          detections: row.detections,
          editedBoxes: row.editedBoxes,
          timings: row.timings,
        },
        manifestUpdatedAt,
      );
    }
  }, [onMediaChange, projectId, manifestUpdatedAt]);

  // Mark pending / cleared for one or more import ids in a single
  // setState call so the labels-loading spinner toggles atomically
  // around a batched fetch.
  const setPendingForIds = useCallback((ids: string[], pending: boolean) => {
    if (ids.length === 0) return;
    setPendingAnnotationIds((cur) => {
      let next: Set<string> | null = null;
      for (const id of ids) {
        if (pending) {
          if (cur.has(id)) continue;
          if (!next) next = new Set(cur);
          next.add(id);
        } else {
          if (!cur.has(id)) continue;
          if (!next) next = new Set(cur);
          next.delete(id);
        }
      }
      return next ?? cur;
    });
  }, []);

  // Fetch a single per-image annotation, splicing the heavy fields
  // (masks, sim dicts, timings) back onto the matching ImportedMedia.
  // No-op when the media already carries masks (already cached) or
  // a fetch is in flight for this id.
  const fetchPerImageAnnotation = useCallback(async (m: ImportedMedia | undefined) => {
    if (!m || !projectId || !m.backendId || !onMediaChange) return;
    if (annotationFetchKickedRef.current.has(m.id)) return;
    const existing = m.detections ?? [];
    if (existing.length > 0 && existing.some((d) => d?.mask)) return;
    annotationFetchKickedRef.current.add(m.id);
    setPendingForIds([m.id], true);
    // P5: IDB-cache lookup. Returns null on miss / stale / flag-off.
    if (IDB_CACHE_ENABLED && projectId) {
      try {
        const cached = await getCachedAnnotation(
          projectId,
          m.backendId!,
          manifestUpdatedAt,
        );
        if (cached) {
          applyParsedRow(m, {
            detections: cached.detections as ParsedRow["detections"],
            editedBoxes: cached.editedBoxes as ParsedRow["editedBoxes"],
            timings: cached.timings,
          });
          // Cache hit - no network needed.
          annotationFetchKickedRef.current.delete(m.id);
          setPendingForIds([m.id], false);
          return;
        }
      } catch {
        /* fall through to network */
      }
    }
    try {
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/annotations/${encodeURIComponent(m.backendId!)}`,
      );
      if (!r.ok) return;
      const buf = ANNOT_WORKER_ENABLED ? await r.arrayBuffer() : null;
      if (
        ANNOT_WORKER_ENABLED &&
        buf &&
        buf.byteLength >= ANNOT_WORKER_BUFFER_THRESHOLD
      ) {
        try {
          const row = await parseSingleAnnotationInWorker(buf);
          applyParsedRow(m, row);
          return;
        } catch (e) {
          console.warn("[viewer] worker parse failed, inline fallback:", e);
          const body = JSON.parse(new TextDecoder().decode(buf)) as {
            detections?: WireDetection[];
            editedBoxes?: EditableBox[] | null;
            timings?: ImportTimings;
          };
          applyAnnotationPayload(m, body);
          return;
        }
      }
      const body = (buf
        ? (JSON.parse(new TextDecoder().decode(buf)) as unknown)
        : (await r.json())) as {
        detections?: WireDetection[];
        editedBoxes?: EditableBox[] | null;
        timings?: ImportTimings;
      };
      applyAnnotationPayload(m, body);
    } catch {
      /* keep the lighter bulk detections on error */
    } finally {
      annotationFetchKickedRef.current.delete(m.id);
      setPendingForIds([m.id], false);
    }
  }, [projectId, onMediaChange, applyAnnotationPayload, applyParsedRow, setPendingForIds, manifestUpdatedAt]);

  // P3: batched fetch. One round-trip for current + neighbour ids.
  // Each id short-circuits if it already has mask geometry cached
  // OR a fetch is in flight (matching the single-image guard above).
  const fetchViewportBatch = useCallback(async (group: (ImportedMedia | undefined)[]) => {
    if (!projectId || !onMediaChange) return;
    const targets: ImportedMedia[] = [];
    for (const m of group) {
      if (!m || !m.backendId) continue;
      if (annotationFetchKickedRef.current.has(m.id)) continue;
      const existing = m.detections ?? [];
      if (existing.length > 0 && existing.some((d) => d?.mask)) continue;
      targets.push(m);
    }
    if (targets.length === 0) return;
    // P5: try the IDB cache before any network. Targets that hit are
    // applied immediately and dropped from the fetch list. The
    // batched /v3/viewport request goes out only for the misses.
    let networkTargets = targets;
    if (IDB_CACHE_ENABLED && projectId) {
      try {
        const cached = await getCachedAnnotationBatch(
          projectId,
          targets.map((m) => m.backendId!),
          manifestUpdatedAt,
        );
        if (Object.keys(cached).length > 0) {
          const stillMissing: ImportedMedia[] = [];
          for (const m of targets) {
            const row = cached[m.backendId!];
            if (row) {
              applyParsedRow(m, {
                detections: row.detections as ParsedRow["detections"],
                editedBoxes: row.editedBoxes as ParsedRow["editedBoxes"],
                timings: row.timings,
              });
            } else {
              stillMissing.push(m);
            }
          }
          networkTargets = stillMissing;
        }
      } catch {
        /* fall through to full network fetch */
      }
    }
    // All targets were cache hits - done, no round-trip needed.
    if (networkTargets.length === 0) return;
    const fetchIds = networkTargets.map((m) => m.id);
    for (const id of fetchIds) annotationFetchKickedRef.current.add(id);
    setPendingForIds(fetchIds, true);
    try {
      const csv = networkTargets.map((m) => encodeURIComponent(m.backendId!)).join(",");
      // P6 binary wire: ask for msgpack when the flag is on. BE
      // falls back to JSON when Accept doesn't include the binary
      // mime, so a flag-off FE keeps the existing wire format.
      const r = await apiFetch(
        `/api/v3/projects/${projectId}/viewport?ids=${csv}`,
        BINARY_WIRE_ENABLED
          ? { headers: { Accept: "application/msgpack" } }
          : undefined,
      );
      if (!r.ok) return;
      // Detect what we actually got back so a server that doesn't
      // know msgpack yet still works against a flag-on FE.
      const contentType = r.headers.get("content-type") || "";
      const isMsgpackResponse = contentType.includes("application/msgpack");
      if (
        BINARY_WIRE_ENABLED &&
        ANNOT_WORKER_ENABLED &&
        isMsgpackResponse
      ) {
        try {
          const buf = await r.arrayBuffer();
          const parsed = await parseViewportBatchMsgpackInWorker(buf);
          const importsByBid = parsed.imports;
          for (const m of networkTargets) {
            const row = importsByBid[m.backendId!];
            if (!row) continue;
            applyParsedRow(m, row);
          }
          return;
        } catch (e) {
          console.warn("[viewer] msgpack parse failed, falling back:", e);
          // Fall through to the JSON paths below by re-reading the
          // response as JSON. Note: r.arrayBuffer() above consumed
          // the body, so we have no choice but to skip the apply
          // here. The viewer's next nav will retry the fetch.
          return;
        }
      }
      const buf = ANNOT_WORKER_ENABLED ? await r.arrayBuffer() : null;
      if (
        ANNOT_WORKER_ENABLED &&
        buf &&
        buf.byteLength >= ANNOT_WORKER_BUFFER_THRESHOLD
      ) {
        try {
          const parsed = await parseViewportBatchInWorker(buf);
          const importsByBid = parsed.imports;
          for (const m of networkTargets) {
            const row = importsByBid[m.backendId!];
            if (!row) continue;
            applyParsedRow(m, row);
          }
          return;
        } catch (e) {
          console.warn("[viewer] worker parse failed, inline fallback:", e);
          const body = JSON.parse(new TextDecoder().decode(buf)) as {
            imports?: Record<string, {
              detections?: WireDetection[];
              editedBoxes?: EditableBox[] | null;
              timings?: ImportTimings;
            }>;
          };
          const byBackendId = body.imports ?? {};
          for (const m of networkTargets) {
            const row = byBackendId[m.backendId!];
            if (!row) continue;
            applyAnnotationPayload(m, row);
          }
          return;
        }
      }
      const body = (buf
        ? (JSON.parse(new TextDecoder().decode(buf)) as unknown)
        : (await r.json())) as {
        imports?: Record<string, {
          detections?: WireDetection[];
          editedBoxes?: EditableBox[] | null;
          timings?: ImportTimings;
        }>;
      };
      const byBackendId = body.imports ?? {};
      for (const m of networkTargets) {
        const row = byBackendId[m.backendId!];
        if (!row) continue;
        applyAnnotationPayload(m, row);
      }
    } catch {
      /* keep the lighter bulk detections on error */
    } finally {
      for (const id of fetchIds) annotationFetchKickedRef.current.delete(id);
      setPendingForIds(fetchIds, false);
    }
  }, [projectId, onMediaChange, applyAnnotationPayload, applyParsedRow, setPendingForIds, manifestUpdatedAt]);

  useEffect(() => {
    if (!media || !projectId || !media.backendId || !onMediaChange) return;
    // Coming back to a tile within its 30 s TTL, cancel the pending
    // mask-strip so the cached record sticks.
    const currentClear = annotationClearTimersRef.current.get(media.id);
    if (currentClear) {
      window.clearTimeout(currentClear);
      annotationClearTimersRef.current.delete(media.id);
    }

    if (VIEWPORT_V3_ENABLED) {
      // One batched round-trip for current + ±2 neighbours.
      void fetchViewportBatch([
        media,
        items[index + 1],
        items[index - 1],
        items[index + 2],
        items[index - 2],
      ]);
    } else {
      // Current image: fetch if no masks yet.
      void fetchPerImageAnnotation(media);

      // Neighbour prefetch, flicking left/right paints from cache.
      // Widened from ±1 to ±2 so a fast double-arrow press still
      // lands on cached state. Cheap: at most 4 parallel requests in
      // flight per tile-change, all short-circuit on cache hits.
      for (const offset of [1, -1, 2, -2]) {
        void fetchPerImageAnnotation(items[index + offset]);
      }
    }

    // Warm the browser's HTTP cache for neighbour image bytes too.
    // The /annotations fetch above only covers geometry; without
    // these the user still waits on the photo decode when they
    // press right. new Image() + src= is enough to get the browser
    // to queue the fetch in the background - no DOM mount needed.
    for (const offset of [1, -1, 2, -2]) {
      const neigh = items[index + offset];
      if (neigh?.preview && !neigh.preview.startsWith("blob:")) {
        const im = new window.Image();
        im.decoding = "async";
        im.src = neigh.preview;
      }
    }

    // Schedule a 30 s mask-strip for the PREVIOUS image (the one the
    // user is leaving). Skipped on first open (no previous) and when
    // the user returned to the same tile.
    const prevIdx = prevViewerIndexRef.current;
    if (prevIdx !== null && prevIdx !== index) {
      const prev = items[prevIdx];
      if (prev && prev.id !== media.id) {
        const prevId = prev.id;
        // Refresh-the-timer pattern: cancel any existing clear so
        // re-visiting prev resets its 30 s clock.
        const t0 = annotationClearTimersRef.current.get(prevId);
        if (t0) window.clearTimeout(t0);
        const tid = window.setTimeout(() => {
          // Strip masks so memory frees, but keep box + predLabel so
          // the gallery chip rail (which uses .length + .predLabel)
          // stays correct. The viewer's mask-presence check goes
          // false, so a re-open refetches.
          if (!onMediaChange) return;
          onMediaChange(prevId, {
            detections: ((prev.detections ?? []).map((d) => ({
              ...d,
              mask: null,
              _maskStripped: true,
            })) as ImportDetection[]),
          });
          annotationClearTimersRef.current.delete(prevId);
          // Masks were stripped for memory; let the tile re-hydrate (refetch
          // masks) when it's reopened.
          annotationsHydratedRef.current.delete(prevId);
        }, 30_000);
        annotationClearTimersRef.current.set(prevId, tid);
      }
    }
    prevViewerIndexRef.current = index;
  }, [media?.id, media?.backendId, projectId, onMediaChange, items, index, fetchPerImageAnnotation, fetchViewportBatch, VIEWPORT_V3_ENABLED]);

  // Cleanup on unmount: tear down any pending mask-strip timers so a
  // late-firing setTimeout doesn't try to mutate state after the
  // viewer's component tree has gone away. Mask-strip is purely a
  // memory optimisation, losing one on close is fine.
  useEffect(() => () => {
    annotationClearTimersRef.current.forEach((tid) => window.clearTimeout(tid));
    annotationClearTimersRef.current.clear();
  }, []);

  const hasPrev = index > 0;
  const hasNext = index < items.length - 1;

  // Apply a partial update to the current media. Falls back to a
  // no-op when the parent didn't supply onMediaChange (e.g., older
  // call sites) so the viewer still works in read-only mode.
  const updateMedia = (patch: Partial<ImportedMedia>) => {
    if (!media || !onMediaChange) return;
    onMediaChange(media.id, patch);
  };

  // Quick relabel: press a hotkey while the cursor is on a box and
  // the box's label flips to that label. Slots fill the keyboard in
  // legibility order:
  //   1-9                 → first 9 labels
  //   Q W E R T Y U I O P → labels 10-19 (top row, left-to-right)
  //   A S D F G H J K L   → labels 20-28 (home row, left-to-right)
  // After that, additional labels just don't get a shortcut. The
  // legend at the top of the viewer shows which key maps to which
  // label so the user can build muscle memory without guessing.
  // Captured in a ref so the keydown handler re-reads the latest
  // hovered box / labels without re-binding the listener every
  // render.
  const relabelStateRef = useRef({ hoveredCanvasBoxId, labels, media, onMediaChange });
  useEffect(() => {
    relabelStateRef.current = { hoveredCanvasBoxId, labels, media, onMediaChange };
  });
  useEffect(() => {
    // Skip relabel-by-keypress when the viewer is mounted in public
    // read-only mode, visitors shouldn't be able to mutate someone
    // else's labels just by hovering and pressing a key.
    if (readOnly) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      // Only fire on bare presses, modifiers belong to other shortcuts
      // (Cmd-Z, Ctrl-Shift-1, etc).
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const slot = slotForRelabelKey(e.key);
      if (slot === null) return;
      const { hoveredCanvasBoxId: hovId, labels: lbls, media: m, onMediaChange: onMC } = relabelStateRef.current;
      if (hovId === null || slot >= lbls.length || !m || !onMC) return;
      const newLabel = lbls[slot];
      e.preventDefault();
      // Don't materialise editedBoxes from mask-stripped detections (the
      // 30s memory strip nulled them; a re-open refetch is pending). Doing
      // so would persist null masks for EVERY box and ship bbox-only data
      // into the seg export. Skip this press; the refetch restores masks
      // and the next press lands correctly.
      if (m.editedBoxes === undefined && (m.detections ?? []).some((d) => d._maskStripped)) {
        return;
      }
      // Pull current boxes, editedBoxes wins, otherwise derive from
      // detections so the relabel sticks even on first edit.
      // Mirror editorBoxes' PRE-filter index derivation so the box ids
      // (imp_<index>_<originalIdx>) match what BoxEditor rendered and
      // hoveredCanvasBoxId points at. Filtering BEFORE mapping (the old
      // code) produced post-filter ids, so a rejected detection ahead of
      // a kept one made the relabel land on the wrong box (or none).
      const sourceBoxes: EditableBox[] = m.editedBoxes !== undefined
        ? m.editedBoxes
        : (m.detections ?? [])
            .map((d, originalIdx) => ({ d, originalIdx }))
            .filter(({ d }) => !d.rejected)
            .map(({ d, originalIdx }) => detectionToEditableBox(d, index, originalIdx));
      const next = sourceBoxes.map((b) =>
        b.id === hovId ? { ...b, label: newLabel } : b,
      );
      onMC(m.id, { editedBoxes: next });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, readOnly]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && hasPrev) onPrev();
      else if (e.key === "ArrowRight" && hasNext) onNext();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasPrev, hasNext, onClose, onPrev, onNext]);

  // Convert detections → EditableBox for BoxEditor.
  //
  // Two sources, in priority order:
  //   1. media.editedBoxes, user-modified state; once the user has
  //      drawn / dragged / deleted anything we use this verbatim so
  //      their edits stick across re-renders.
  //   2. media.detections derivation, the inference-time view, with
  //      every detection (kept AND rejected) included. Rejected ones
  //      ride V1's red "Rejected" pill in BoxEditor's sidebar so the
  //      user can see them, hover them on the canvas, and click
  //      "Verify" to flip the validation if they disagree.
  //
  // BoxEditor's `key` is stable so internal state (zoom, pan,
  // selection) survives any re-renders.
  const editorBoxes = useMemo<EditableBox[]>(() => {
    // Map detections → editable boxes WHILE PRESERVING THE ORIGINAL
    // INDEX. The DetectionList in the pipeline popup keys hover /
    // select callbacks on each row's pre-filter position; we need
    // the editor box id to match that exact index so emphasizedBoxId
    // lands on the right box. Filtering before mapping with the
    // post-filter index used to produce off-by-one (or more)
    // misalignments whenever a rejected row preceded a kept one ,
    // the popup said "hare" but the canvas highlighted the next
    // accepted box.
    // editedBoxes wins when the FE has it at all, the backend's
    // /annotations endpoint only sends `editedBoxes` once the user
    // has actually touched them (writes flip `editedBoxesSet=True`
    // server-side). A fresh import that hasn't been edited gets
    // `editedBoxes: undefined` here, so we fall through to detections.
    // An explicit "user deleted everything" case persists as
    // `editedBoxes: []` and correctly trumps detections, without
    // this, a deletion would snap back to the auto detections on
    // the next render.
    const baseBoxes: EditableBox[] = media?.editedBoxes !== undefined
      ? media.editedBoxes
      : (media?.detections ?? [])
          .map((d, originalIdx) => ({ d, originalIdx }))
          .filter(({ d }) => !d.rejected)
          .map(({ d, originalIdx }) => detectionToEditableBox(d, index, originalIdx));

    // Hover-to-reveal: when the cursor is on a row in the pipeline
    // popup that maps to a REJECTED detection, layer that one box
    // back into the canvas with a red "rejected" validation. Falls
    // away again the moment the cursor leaves the row, so the
    // default view stays clean. Uses the canonical
    // `imp_<index>_<hoveredIdx>` id so BoxEditor's
    // emphasizedBoxId spotlight already lands on this ghost
    // without any extra wiring.
    if (hoveredIdx === null) return baseBoxes;
    const dets = media?.detections ?? [];
    const hovered = dets[hoveredIdx];
    if (!hovered || !hovered.rejected) return baseBoxes;
    const ghost: EditableBox = {
      id: `imp_${index}_${hoveredIdx}`,
      label: hovered.predLabel ?? hovered.gdLabel ?? "?",
      x0: hovered.box[0],
      y0: hovered.box[1],
      x1: hovered.box[2],
      y1: hovered.box[3],
      score: hovered.gdScore,
      mask: hovered.mask,
      validation: {
        match: false,
        confidence: 0,
        reason: rejectReasonExplanation(hovered),
        source: "auto",
        kind: "vlm",
      },
    };
    return [...baseBoxes, ghost];
  }, [media?.editedBoxes, media?.detections, index, hoveredIdx]);

  // Decode dimensions from the image when the backend hasn't supplied
  // them yet (e.g. status="processing", we still want to show
  // something). Falls back to `(media.width, media.height)` once the
  // pipeline finishes and updates state.
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    if (!media) return;
    if (media.width && media.height) { setDims({ w: media.width, h: media.height }); return; }
    const img = new window.Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = media.preview;
    return () => { img.onload = null; };
  }, [media]);

  // Prefetch the original-image URLs for the ±3 neighbours so
  // clicking Next / Previous never waits on a network fetch.
  // Anything outside that window gets evicted from our Map so we
  // don't hold image refs forever, the browser's HTTP cache may
  // still serve a stale copy, but the FE memory pressure stays
  // bounded (~6 images held at a time).
  const prefetchRef = useRef<Map<string, HTMLImageElement>>(new Map());
  useEffect(() => {
    if (items.length === 0) return;
    const desired = new Set<string>();
    for (let i = index - 3; i <= index + 3; i++) {
      if (i < 0 || i >= items.length) continue;
      const url = items[i]?.preview;
      if (url) desired.add(url);
    }
    for (const url of desired) {
      if (!prefetchRef.current.has(url)) {
        const img = new window.Image();
        // No onload handler, we just want the browser to fetch
        // and cache the bytes. The BoxEditor's <img> on the next
        // / previous click reads from the cache.
        img.src = url;
        prefetchRef.current.set(url, img);
      }
    }
    for (const url of Array.from(prefetchRef.current.keys())) {
      if (!desired.has(url)) {
        prefetchRef.current.delete(url);
      }
    }
  }, [items, index]);
  // Drop all prefetched refs when the viewer unmounts so the next
  // open starts clean.
  useEffect(() => () => { prefetchRef.current.clear(); }, []);

  // Pipeline panel is hidden by default, the BoxEditor takes the
  // full screen so the editor feels identical to V1's `EditModal`.
  // Press `i` or click the info button to slide the panel in from
  // the right when you want to see pipeline details.

  // Size-coding toggle: when on, each box gets its `sizeStatus`
  // computed by scaling the IMAGE so its longest edge is 256 px
  // (the rule the user wanted, same letterbox math V1 uses with
  // a 256×256 target, which is mathematically identical to
  // "longest-edge = 256"). BoxEditor tints chip + outline red /
  // amber / green so the user can see which boxes will survive
  // training-time downsize.
  const [sizeColoringOn, setSizeColoringOn] = useState(true);
  // Manual labelling mode toggle. When on, the BoxEditor's
  // click-to-detect + add-box flows bypass SAM3 entirely - drawn
  // boxes stay as the user's raw rectangles and the label picker
  // pops open for them to type. Mask painting works the same in
  // both modes (it's always manual).
  const [manualMode, setManualMode] = useState(false);
  // `inputShape` is the prop (the project's selected input size), so the
  // red/amber/green size colouring matches the export size-filter exactly.
  // It was previously hardcoded "256x256" here while export filtered at
  // the real input size, so sub-256 targets silently dropped boxes the
  // user saw as green. Only `max(imgW, imgH) → N` matters for the
  // letterbox math; the "xN" half is irrelevant.

  const sizeStatuses = useMemo<Record<string, "ok" | "warn" | "fail">>(() => {
    if (!sizeColoringOn) return {};
    const w = (media as ImportedMedia | undefined)?.width;
    const h = (media as ImportedMedia | undefined)?.height;
    if (!w || !h) return {};
    const out: Record<string, "ok" | "warn" | "fail"> = {};
    for (const b of editorBoxes) {
      out[b.id] = sizeStatusFor(scaledMinSide(b, w, h, inputShape));
    }
    return out;
  }, [sizeColoringOn, editorBoxes, media, inputShape]);

  // Pipeline-info shortcut removed alongside the button - see the
  // comment near the removed button for context.

  if (!media) return null;

  return (
    // Contained in the shell's content area: below the 36px title bar,
    // above the 24px status bar, right of the Explorer side bar (the
    // shell publishes its live edge as --pk-content-left, so a
    // collapsed side bar widens the editor). The editor never covers
    // the app chrome.
    <div
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[400] flex flex-col bg-[var(--background)]"
      role="dialog"
      aria-modal="true"
    >
      {/* Header: filename + meta. V2 chrome, lighter typography,
          single-row layout matching the project page's title block. */}
      <header className="flex items-center justify-between gap-4 px-8 py-4 border-b border-foreground/[0.06] shrink-0">
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="text-base font-light tracking-tight text-[var(--foreground)] truncate">
            {media.file.name}
          </span>
          {dims && (
            <span className="text-xs text-foreground/45 tabular-nums shrink-0">
              {dims.w} × {dims.h}
            </span>
          )}
          <span className="text-xs text-foreground/45 shrink-0">
            {/* Live count of the boxes on the canvas (editorBoxes) so deleting a
                detection updates this immediately, instead of the stale
                persisted media.detections count. */}
            {editorBoxes.length} detection{editorBoxes.length === 1 ? "" : "s"}
          </span>
        </div>
        {/* Quick-relabel legend: hover any box on the canvas, press a
            key, label flips. Mirrors the chip palette so the user's
            colour-coded mental model matches the digit/letter. Caps
            at RELABEL_KEY_LIMIT shortcuts (28: 1-9 + QWERTYUIOP + ASDFGHJKL).
            Anything beyond just renders without a shortcut chip. */}
        {labels.length > 0 && (
          <div className="hidden lg:flex items-center gap-1.5 shrink-0 flex-wrap justify-end max-w-[40%]">
            <span className="text-[10px] text-foreground/35 mr-1">
              hover + press
            </span>
            {labels.map((lab, i) => {
              const key = keyForRelabelSlot(i);
              return (
                <span
                  key={lab}
                  className="inline-flex items-center gap-1.5 rounded-full border border-foreground/10 bg-foreground/[0.03] px-2 py-0.5 text-[11px] tabular-nums"
                >
                  {key && (
                    <kbd className="font-mono text-foreground/80 px-1 py-[1px] rounded bg-foreground/[0.06] leading-none text-[10px]">
                      {key}
                    </kbd>
                  )}
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: colourForLabel(labels, lab, labelColours) }}
                    aria-hidden
                  />
                  <span className="text-foreground/75">{lab}</span>
                </span>
              );
            })}
          </div>
        )}
        <div className="flex items-center gap-4 shrink-0">
          {/* Manual verdict pills, same setter Review mode uses, so
              flipping one here syncs to the gallery filter chips and
              the Review queue immediately. Hidden in read-only view
              and until the import has a backend id to key off. */}
          {!readOnly && onVerdict && media.backendId && (
            <VerdictPills
              current={verdicts[media.backendId] ?? null}
              onToggle={(v) => onVerdict(media.backendId!, v)}
            />
          )}
          {/* Derived "new labels" crop: show the parent's original label as a
              reference so the user knows what this crop was while assigning a
              fresh label. */}
          {media.derivedLabel && (
            <span
              className="inline-flex items-center gap-1.5 rounded-full bg-foreground/[0.06] px-2.5 h-7 text-[11px] font-medium text-foreground/60"
              title={`Original label in the parent project: "${media.derivedLabel}". Assign your own label here.`}
            >
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 opacity-60" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="9 10 4 15 9 20" /><path d="M20 4v7a4 4 0 0 1-4 4H5" />
              </svg>
              <span className="text-foreground/45">from</span>
              <span className="text-foreground/85">{media.derivedLabel}</span>
            </span>
          )}
          <span className="hidden md:inline text-[11px] text-foreground/30">
            ← / → {readOnly ? "" : "· I "}· Esc
          </span>
          <span className="text-xs text-foreground/45 tabular-nums">{index + 1} / {items.length}</span>
          {/* Clear all labels on the current image. Sets editedBoxes
              to [] which:
                - drops every box on the canvas (BoxEditor reads
                  editedBoxes when set, regardless of auto detections)
                - flags the tile Unlabelled in the gallery
                - re-arms the project-level Start labelling button to
                  read "Start labelling new images" via the stub's
                  unlabelledImportCount + labelledImportCount split.
              Hidden in read-only mode + when there are no boxes to
              clear (both detections AND editedBoxes empty). */}
          {!readOnly && onMediaChange && (() => {
            const liveBoxCount = media.editedBoxes !== undefined
              ? (media.editedBoxes?.length ?? 0)
              : ((media.detections?.filter((d) => !d.rejected).length) ?? 0);
            if (liveBoxCount === 0) return null;
            return (
              <button
                type="button"
                onClick={() => onMediaChange(media.id, { editedBoxes: [] })}
                title="Clear every box on this image"
                className="h-8 inline-flex items-center gap-1.5 rounded-full border border-foreground/[0.10] text-foreground/65 hover:border-rose-400/60 hover:text-rose-700 dark:hover:text-rose-300 px-3 text-[11px] uppercase tracking-wider font-mono transition-colors"
              >
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                  <path d="M10 11v6M14 11v6" />
                </svg>
                Clear all
              </button>
            );
          })()}
          {/* Manual labelling toggle. When on, the BoxEditor's
              click-to-detect / add-box flows skip every backend ML
              call (SAM3, classify, resolver) and just commit the
              user's raw rectangle - they pick the label from the
              picker that pops open. Mask painting is always manual
              regardless of this toggle. */}
          {!readOnly && (
            <button
              type="button"
              onClick={() => setManualMode((v) => !v)}
              aria-pressed={manualMode}
              title={manualMode
                ? "Manual mode ON - clicks and drag-boxes skip SAM3"
                : "Switch to manual labelling (no SAM3 auto-detect)"
              }
              className={[
                "h-8 inline-flex items-center gap-1.5 rounded-full border px-3 text-[11px] uppercase tracking-wider font-mono transition-colors",
                manualMode
                  ? "border-orange-500/60 bg-orange-500/15 text-orange-700 dark:text-orange-200"
                  : "border-foreground/[0.10] text-foreground/65 hover:border-foreground/30 hover:text-foreground",
              ].join(" ")}
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M12 19l7-7 3 3-7 7-3-3z" />
                <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                <path d="M2 2l7.586 7.586" />
                <circle cx="11" cy="11" r="2" />
              </svg>
              Manual
            </button>
          )}
          {/* Delete the current image entirely (same as the gallery tile
              trash). Advances to the next image, or closes if it was the last. */}
          {!readOnly && onDelete && media && (
            <button
              type="button"
              onClick={() => {
                const onlyOne = items.length <= 1;
                const wasLast = index >= items.length - 1;
                onDelete(media.id);
                if (onlyOne) onClose();
                else if (wasLast) onPrev();
              }}
              title="Delete this image"
              aria-label="Delete this image"
              className="h-8 inline-flex items-center gap-1.5 rounded-full border border-foreground/[0.10] text-foreground/65 hover:border-rose-400/60 hover:text-rose-700 dark:hover:text-rose-300 px-3 text-[11px] uppercase tracking-wider font-mono transition-colors"
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                <path d="M10 11v6M14 11v6" />
                <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
              </svg>
              Delete
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="h-8 w-8 grid place-items-center rounded-full border border-foreground/[0.08] text-foreground/55 hover:border-foreground/30 hover:text-foreground transition-colors"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" /><path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      </header>

      {/* Main canvas, BoxEditor takes the full remaining viewport
          height. Outer wrapper flex-col with min-h-0 so the canvas
          and the bottom nav bar partition the available space; the
          canvas grows / nav bar stays a fixed height. Card chrome
          and box-shadow mirror V1's EditModal feel. */}
      <div className="relative flex-1 min-h-0 p-5 flex flex-col gap-3">
        {/* Canvas + popup overlay live in this relative wrapper so the
            popup anchors to the canvas, not to the outer container
            (which would otherwise overlap the bottom nav bar). */}
        <div className="relative flex-1 min-h-0">
        <div
          className="h-full overflow-hidden rounded-2xl border border-foreground/[0.06] bg-[var(--surface-2)]"
        >
          {dims ? (
            <BoxEditor
              key={`${media.id}_${index}`}
              imageUrl={media.preview}
              // For large originals (e.g. 4K), load a 2560px display variant
              // first so the viewer paints fast, then BoxEditor swaps to the
              // full image for pixel-sharp zoom. Small images load directly.
              previewUrl={(dims.w > 2560 || dims.h > 2560) ? `${media.preview}?w=2560` : null}
              imageWidth={dims.w}
              imageHeight={dims.h}
              blurhash={media.blurhash ?? null}
              boxes={editorBoxes}
              onChange={(next) => updateMedia({ editedBoxes: next })}
              projectTags={labels}
              labelColours={labelColours}
              displayLabel={(s) => labelAliases[s.trim().toLowerCase()] || s}
              colorMode="palette"
              sizeStatuses={sizeColoringOn ? sizeStatuses : undefined}
              sizeColoringOn={sizeColoringOn}
              onSizeColoringToggle={() => setSizeColoringOn((v) => !v)}
              readOnly={readOnly}
              // Show the labels sidebar spinner while the per-image
              // /annotations fetch is in flight for this tile. The
              // bulk /annotations payload doesn't include mask
              // polygons, so even a tile that lands with boxes
              // already drawn is "still loading geometry" until
              // this fetch resolves.
              loadingLabels={pendingAnnotationIds.has(media.id)}
              onLabelRenamed={(_id, _oldLabel, newLabel) => {
                // Auto-extend the project's vocabulary when the
                // user types a label that isn't already in the tag
                // list. Keeps the dataset / labelling resolver in
                // sync without forcing the user to manually click
                // "Add label" in the annotations chip first.
                if (!onAddProjectLabel) return;
                const trimmed = (newLabel || "").trim();
                if (!trimmed) return;
                const lower = trimmed.toLowerCase();
                const exists = labels.some((l) => l.toLowerCase() === lower);
                if (!exists) onAddProjectLabel(trimmed);
              }}
              // editorBoxes uses ids of shape `imp_<index>_<i>` ,
              // mapping hoveredIdx to that id reuses BoxEditor's
              // built-in spotlight effect (dim others + ring the
              // hovered one) without a parallel rendering layer.
              emphasizedBoxId={hoveredIdx === null ? null : `imp_${index}_${hoveredIdx}`}
              onHoverChange={setHoveredCanvasBoxId}
              onAddBoxDetect={manualMode ? undefined :
                // Charlie's combined segment_and_classify_box does
                // segmenter + resolver in one call. Saves one RTT vs
                // the segment_box + classify_box pair below (which
                // stays as the fallback for non-charlie / non-by-id
                // paths). Skipped when we don't have the import on
                // disk yet - the combined endpoint relies on the
                // by-id image load.
                PIPELINE === "charlie" && media.backendId && projectId && labels.length > 0
                  ? async (b) => {
                      const fd = new FormData();
                      fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                      fd.append("labels", JSON.stringify(labels));
                      fd.append("project_id", projectId);
                      fd.append("import_id", media.backendId!);
                      const r = await apiFetch(
                        "/api/charlie/imports/segment_and_classify_box",
                        { method: "POST", body: fd },
                      );
                      if (!r.ok) {
                        const body = await r.text().catch(() => "");
                        console.warn(`[v2 segment_and_classify_box] http ${r.status} ${body.slice(0, 200)}`);
                        return null;
                      }
                      const d = await r.json();
                      return {
                        mask: d.mask ?? null,
                        label: d.label ?? null,
                        score: d.score ?? null,
                      };
                    }
                  : undefined
              }
              onBoxDrawn={manualMode ? undefined : async (b) => {
                // Charlie: send import_id so the backend loads the
                // original from disk, no CORS-blocked browser fetch.
                // V2 still uploads the image since /api/v2/references/
                // segment_box predates the by-id path.
                const fd = new FormData();
                fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                if (PIPELINE === "charlie" && media.backendId && projectId) {
                  fd.append("project_id", projectId);
                  fd.append("import_id", media.backendId);
                  fd.append("labels", JSON.stringify(labels));
                } else {
                  const file = await ensureMediaFile(media);
                  if (!file || file.size === 0) {
                    console.warn("[v2 segment_box] media bytes unavailable");
                    return null;
                  }
                  fd.append("image", file);
                  if (PIPELINE === "charlie") {
                    fd.append("labels", JSON.stringify(labels));
                    if (projectId) fd.append("project_id", projectId);
                  }
                }
                const segBase = PIPELINE === "charlie" ? "/api/charlie/imports" : "/api/v2/references";
                // apiFetch (not plain fetch) so the NextAuth bearer
                // rides the request - segment_box sits behind
                // enforce_credits → current_user, which 401s on an
                // unauthenticated call and drops the freshly-drawn
                // placeholder box on the floor.
                const r = await apiFetch(`${segBase}/segment_box`, { method: "POST", body: fd });
                if (!r.ok) {
                  const body = await r.text().catch(() => "");
                  console.warn(`[v2 segment_box] http ${r.status} ${body.slice(0, 200)}`);
                  return null;
                }
                const d = await r.json();
                return d.mask ?? null;
              }}
              onClassifyBox={
                manualMode || labels.length === 0
                  ? undefined
                  : async (b) => {
                      const fd = new FormData();
                      fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                      fd.append("labels", JSON.stringify(labels));
                      if (PIPELINE === "charlie" && media.backendId && projectId) {
                        fd.append("project_id", projectId);
                        fd.append("import_id", media.backendId);
                      } else {
                        const file = await ensureMediaFile(media);
                        if (!file || file.size === 0) {
                          console.warn("[v2 classify_box] media bytes unavailable");
                          return null;
                        }
                        fd.append("image", file);
                        if (PIPELINE === "charlie" && projectId) {
                          fd.append("project_id", projectId);
                        }
                      }
                      const segBase = PIPELINE === "charlie" ? "/api/charlie/imports" : "/api/v2/references";
                      // apiFetch so the bearer reaches classify_box's
                      // enforce_credits gate - plain fetch 401s and
                      // the BoxEditor wipes the in-flight box.
                      const r = await apiFetch(`${segBase}/classify_box`, { method: "POST", body: fd });
                      if (!r.ok) {
                        const body = await r.text().catch(() => "");
                        console.warn(`[v2 classify_box] http ${r.status} ${body.slice(0, 200)}`);
                        return null;
                      }
                      const d = await r.json();
                      return { label: d.label ?? null, score: d.score ?? null };
                    }
              }
              onPointDetect={manualMode ? undefined : async (point) => {
                // Charlie: send import_id so the backend loads the
                // original from disk (no FE-side fetch, no CORS).
                // detect_point also does label resolution in the
                // same round-trip for specific projects, so the
                // BoxEditor's onClassifyBox follow-up is optional.
                const fd = new FormData();
                fd.append("point", JSON.stringify([point.x, point.y]));
                if (projectId) fd.append("project_id", projectId);
                if (PIPELINE === "charlie") {
                  fd.append("labels", JSON.stringify(labels));
                  if (media.backendId) {
                    fd.append("import_id", media.backendId);
                  } else {
                    const file = await ensureMediaFile(media);
                    if (!file || file.size === 0) {
                      console.warn("[v2 detect_point] media bytes unavailable");
                      return null;
                    }
                    fd.append("image", file);
                  }
                } else {
                  const file = await ensureMediaFile(media);
                  if (!file || file.size === 0) {
                    console.warn("[v2 detect_point] media bytes unavailable");
                    return null;
                  }
                  fd.append("image", file);
                }
                // apiFetch so detect_point sees the bearer token -
                // without it the endpoint's enforce_credits dep 401s
                // and the BoxEditor removes the placeholder, which
                // surfaced as the "click to detect briefly blinks
                // then disappears" symptom.
                let r: Response;
                try {
                  r = await apiFetch(`${IMPORTS_BASE}/detect_point`, { method: "POST", body: fd });
                } catch (err) {
                  console.error("[detect_point] network error", err);
                  return null;
                }
                if (!r.ok) {
                  const body = await r.text().catch(() => "");
                  console.error(
                    `[detect_point] http ${r.status} ${r.statusText} - ${body.slice(0, 400)}`,
                  );
                  return null;
                }
                try {
                  return await r.json();
                } catch (err) {
                  console.error("[detect_point] response not JSON", err);
                  return null;
                }
              }}
            />
          ) : (
            <div className="h-full grid place-items-center text-foreground/40 text-sm">Loading…</div>
          )}
        </div>

        </div>

        {/* Bottom nav bar, V2-themed pill buttons matching the rest
            of the project page. Sits below the canvas as a flex
            sibling so it never overlaps the image. */}
        <div className="shrink-0 flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={onPrev}
            disabled={!hasPrev}
            aria-label="Previous image"
            className={[
              "h-10 px-5 inline-flex items-center gap-2 rounded-full border text-sm font-medium transition-all",
              hasPrev
                ? "border-foreground/[0.08] bg-foreground/[0.03] text-foreground/85 hover:bg-foreground/[0.06] hover:border-foreground/25"
                : "border-foreground/[0.04] bg-foreground/[0.01] text-foreground/25 cursor-not-allowed",
            ].join(" ")}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="m15 18-6-6 6-6" />
            </svg>
            Previous
          </button>
          <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] text-foreground/35 px-2">
            <kbd className="rounded-md border border-foreground/[0.08] bg-foreground/[0.03] px-1.5 py-0.5 text-[10px] font-mono">←</kbd>
            <kbd className="rounded-md border border-foreground/[0.08] bg-foreground/[0.03] px-1.5 py-0.5 text-[10px] font-mono">→</kbd>
            navigate
          </span>
          <button
            type="button"
            onClick={onNext}
            disabled={!hasNext}
            aria-label="Next image"
            className={[
              "h-10 px-5 inline-flex items-center gap-2 rounded-full border text-sm font-medium transition-all",
              hasNext
                ? "border-foreground/[0.08] bg-foreground/[0.03] text-foreground/85 hover:bg-foreground/[0.06] hover:border-foreground/25"
                : "border-foreground/[0.04] bg-foreground/[0.01] text-foreground/25 cursor-not-allowed",
            ].join(" ")}
          >
            Next
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="m9 6 6 6-6 6" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
