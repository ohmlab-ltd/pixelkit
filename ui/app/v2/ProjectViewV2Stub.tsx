"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { BlurhashCanvas } from "react-blurhash";

import { LABEL_COLOURS, buildProjectLabelColourMap, colourForLabelStable } from "./labelColours";
import { BoxEditor, detectionsToBoxes, stripTransientBoxFlags, withMasksFrom, type EditableBox, type MaskShape, type Validation } from "../BoxEditor";
import { ReferenceImageEditor, type ReferenceImage } from "./ReferenceImageEditor";
import { LabelJobCard, type LabelJobState } from "./LabelJobCard";
import { requestExplorerRefresh } from "../../lib/appNav";
import { AugmentationsCard } from "./AugmentationsCard";
import { DerivedDatasetsBar } from "./DerivedDatasets";
import { OverviewPanel, healthLabel } from "./OverviewPanel";
import { DatasetHealthModal } from "./DatasetHealthModal";
import { DeleteLabelModal } from "./DeleteLabelModal";
import { ClearAllAnnotationsModal } from "./ClearAllAnnotationsModal";
import { OpenverseInlinePanel } from "./OpenverseInlinePanel";
import { ReviewModeV2 } from "./ReviewModeV2";
import { VideoFrameModal } from "./VideoFrameModal";
import { extractCount, extractVideoFrames, isVideoFile, maxImportBytes, VIDEO_ACCEPT } from "../../lib/videoFrames";
import { ProjectSettingsV2 } from "./ProjectSettingsV2";
import { ExportModal } from "../ExportModal";
import { Tooltip } from "../Tooltip";
import { API_BASE, apiFetch } from "../../lib/apiFetch";
import { splitLabels } from "../../lib/labelInput";
import { galleryFilterCounts, isLabelled, matchesGalleryFilter, shiftFilterCounts } from "../../lib/galleryFilters";
import { patchProjectMeta, readProjectMeta } from "../../lib/projectMetaCache";
import { PixelKitLoader } from "./PixelKitLoader";
import { isImageFile } from "../../lib/resize";
import { ScrollToTop } from "../components/ScrollToTop";
import { useIdle } from "../../lib/useIdle";

// Derived by the engine: "specific" with reference images, "general" without. Read-only.
type DatasetTypeValue = {
  type: "general" | "specific";
  reason?: string | null;
  source?: string | null;
};

// Annotation target per label, enforced by the backend.
const ANNOTS_PER_LABEL = 5;

const PROCESSING_PHRASES = [
  "Uploading images…",
  "Saving the originals…",
  "Decoding images…",
  "Applying EXIF orientation…",
  "Generating thumbnails…",
  "Creating blurred placeholders…",
  "Preparing images for labelling…",
  "Finishing the upload…",
];
const AUGMENT_PHRASES = [
  "Warping perspective…",
  "Swapping backgrounds…",
  "Adding motion blur…",
  "Rotating images…",
  "Adding sensor noise…",
  "Shifting colour…",
  "Cutting objects out along their masks…",
  "Placing objects in new scenes…",
  "Writing augmented copies…",
  "Generating variations…",
  "Shifting hue…",
  "Scaling images…",
  "Varying the lighting…",
];
// Copies a File into a fresh buffer before upload. Safari can lose a Blob's
// backing memory before the FormData read.
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

// The engine's detail for a refused request, else `fallback`.
async function refusalText(r: Response, fallback: string): Promise<string> {
  const body = (await r.json().catch(() => null)) as { detail?: unknown } | null;
  return typeof body?.detail === "string" ? body.detail : fallback;
}

// Equal to the model_registry imgsz choices. 480x480 is for st_yoloxn.
const INPUT_SHAPES = [
  "96x96", "128x128", "160x160", "192x192", "224x224", "256x256",
  "320x320", "480x480", "512x512", "640x640",
];
// Dataset view sections, owned by the shell. Matches ExplorerPane's DatasetSection.
export type ProjectTab = "overview" | "references" | "dataset" | "augmentations";

// createdAt as ms epoch number, ms epoch string or ISO timestamp -> ms; NaN if unparseable.
function parseCreatedAtMs(v: number | string | null | undefined): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  const asNum = Number(v);
  if (Number.isFinite(asNum)) return asNum;
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : NaN;
}

// Placeholder detections have box [0,0,0,0] and mask null; real ones do not.
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

// Save payload for an import; also the baseline an unchanged import is recognised by.
function editPayload(boxes: EditableBox[], background?: boolean): string {
  return JSON.stringify({ editedBoxes: boxes, background: !!background });
}

// Newest first by createdAt, then by the ms in an imp_<ms>_<rand> id, then by id.
function compareImportedMediaDesc(a: { id: string; createdAt?: number | string | null }, b: { id: string; createdAt?: number | string | null }): number {
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

export function ProjectViewV2Stub({
  projectName,
  labels,
  references,
  projectId,
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
  /** Backend project UUID; null if the create call failed. */
  projectId?: string | null;
  /** The Project this dataset was opened from; onClose returns to it. */
  backToProjectId?: string | null;
  /** "onboarding" suppresses the full-screen mount loader while HomeView shows its own. */
  firstLoad?: "onboarding" | null;
  /** Active section, owned by the shell. Changes go through onSectionChange. */
  section?: ProjectTab;
  onSectionChange?: (section: ProjectTab) => void;
  onClose: () => void;
  onReferencesChange?: (next: ReferenceImage[]) => void;
}) {
  // True after 90 s without input; suspends the augment poll.
  const isIdle = useIdle(90_000);

  const [refs, setRefs] = useState<ReferenceImage[]>(() => {
    // Without passed refs, seed from the localStorage cache until the manifest arrives.
    if (references.length > 0) return references;
    if (!projectId) return references;
    const cached = readProjectMeta(projectId);
    if (!cached?.refTiles?.length) return references;
    return cached.refTiles.map((t) => ({
      file: new File([], t.filename),
      preview: `${API_BASE}/api/v2/projects/${projectId}/references/${encodeURIComponent(t.filename)}`,
      width: t.width,
      height: t.height,
      boxes: [],
      referenceId: t.id,
      filename: t.filename,
      blurhash: t.blurhash ?? null,
    }));
  });
  useEffect(() => {
    if (references.length > 0) setRefs(references);
  }, [references]);

  // Placeholder tile count for references, seeded from the meta cache.
  const [expectedRefCount, setExpectedRefCount] = useState<number>(() => {
    if (!projectId) return 0;
    const cached = readProjectMeta(projectId);
    return Math.max(0, cached?.nReferences ?? 0);
  });
  // Lite stats from /initial, for OverviewPanel and the page-ready gate.
  const [seedStats, setSeedStats] = useState<unknown | null>(null);

  // /initial: project meta, the first 20 import tiles and lite stats in one request.
  const initialHydratedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    if (initialHydratedRef.current.has(projectId)) return;
    initialHydratedRef.current.add(projectId);
    const pid: string = projectId;
    const ac = new AbortController();
    (async () => {
      try {
        // no-store: a reload after a destructive action must not paint a cached response.
        const r = await apiFetch(
          `/api/v2/projects/${pid}/initial?n=20`,
          { cache: "no-store", signal: ac.signal },
        );
        if (!r.ok) return;
        type InitialResp = {
          name?: string;
          tags?: string[];
          // Tags used by the last label_charlie job; null if never labelled.
          labelsLastRun?: string[] | null;
          // Detection settings of the last labelling run.
          settingsLastRun?: {
            // The three sliders are present only when the run set all three.
            threshold?: number;
            mask_threshold?: number;
            min_relative_area?: number;
            tile_native?: boolean;
            tile_size?: number;
          } | null;
          label_aliases?: Record<string, string>;
          labelColours?: Record<string, string>;
          dataset_type?: {
            type?: string;
            reason?: string | null;
            source?: string | null;
          } | null;
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
            // Per-label counts. Optional; label_set is the fallback.
            label_stats?: Record<string, number>;
            has_edits?: boolean;
            background?: boolean;
            labelledAt?: number | null;
          }[];
          imports_total?: number;
          stats?: unknown;
        };
        const j = (await r.json()) as InitialResp;
        if (j.label_aliases && typeof j.label_aliases === "object") {
          setLabelAliases(j.label_aliases);
        }
        if (j.labelColours && typeof j.labelColours === "object") {
          setLabelColours(j.labelColours);
        }
        if (Array.isArray(j.tags) && labels.length === 0) {
          setEditLabels(j.tags);
        }
        // When absent, the dataset-type effect fetches it.
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
        if (Array.isArray(j.labelsLastRun)) {
          setLabelsLastRun(j.labelsLastRun);
        } else if (j.labelsLastRun === null) {
          setLabelsLastRun(null);
        }
        // The last run's settings are the relabel baseline after a reload.
        if (j.settingsLastRun) {
          const s = j.settingsLastRun;
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
        // Reference placeholders; /annotations fills in the boxes.
        const refList = j.references ?? [];
        if (refList.length > 0) {
          setExpectedRefCount(refList.length);
          const newUploadStatus: Record<string, "uploading" | "done" | "failed"> = {};
          const hydrated: ReferenceImage[] = refList.map((ref) => {
            const url = `${API_BASE}/api/v2/projects/${pid}/references/${encodeURIComponent(ref.filename)}`;
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
        const imps = j.imports ?? [];
        const total = j.imports_total ?? imps.length;
        setImportsTotal(total);
        const hydratedImports: ImportedMedia[] = imps.map((imp) => {
          const labelSet = imp.label_set ?? [];
          const nDetections = imp.n_detections ?? 0;
          const labelStats: Record<string, number> = imp.label_stats
            ?? Object.fromEntries(labelSet.map((lab) => [lab, 0]));
          const placeholderDetections: ImportDetection[] = nDetections > 0
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
            preview: `${API_BASE}/api/v2/projects/${pid}/imports/${encodeURIComponent(imp.filename)}`,
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
            background: !!imp.background,
            detections: placeholderDetections,
            // Cache-buster for the labelled preview; undefined if never labelled.
            labelledAt: imp.labelledAt ?? undefined,
          };
        });
        if (hydratedImports.length > 0) {
          setImports((cur) => {
            const byId = new Map(hydratedImports.map((h) => [h.id, h]));
            const merged: ImportedMedia[] = [];
            for (const c of cur) {
              // Local tiles keep their local id for life; backendId holds the server id.
              const matchKey = byId.has(c.id)
                ? c.id
                : (c.backendId && byId.has(c.backendId) ? c.backendId : null);
              if (matchKey) {
                const fresh = byId.get(matchKey)!;
                merged.push({
                  ...fresh,
                  id: c.id,
                  backendId: c.backendId,
                  detections: (hasRealDetections(c.detections) || c.editedBoxes !== undefined)
                    ? c.detections
                    : fresh.detections,
                  editedBoxes: c.editedBoxes,
                  // The server returns null while its blurhash backfill runs.
                  blurhash: fresh.blurhash ?? c.blurhash ?? null,
                  provisional: undefined,
                });
                byId.delete(matchKey);
              } else if (c.provisional || !c.backendId) {
                // /initial returns only the newest 20, so unconfirmed and in-flight tiles stay.
                merged.push(c);
              }
              // A confirmed tile absent from the server's list is dropped.
            }
            for (const h of byId.values()) merged.push(h);
            return merged.sort(compareImportedMediaDesc);
          });
        }
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
        });
        if (j.stats) setSeedStats(j.stats);
        setImportsReady(true);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        console.warn("[v2 initial] failed:", e);
      }
    })();
    return () => {
      ac.abort();
    };
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setSeedStats(null);
  }, [projectId]);

  // Re-seeds imports and refs on project switch; the useState seeds run only at mount.
  const projectSwitchInitialRef = useRef(true);
  useEffect(() => {
    if (projectSwitchInitialRef.current) {
      projectSwitchInitialRef.current = false;
      return;
    }
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
    setImports(
      cached?.importTiles?.length
        ? cached.importTiles
            .map<ImportedMedia>((t) => ({
              id: t.id,
              backendId: t.id,
              file: new File([], t.filename),
              filename: t.filename,
              preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(t.filename)}`,
              blurhash: t.blurhash ?? null,
              status: "ready" as const,
              width: t.width,
              height: t.height,
              createdAt: t.createdAt ?? null,
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
            preview: `${API_BASE}/api/v2/projects/${projectId}/references/${encodeURIComponent(t.filename)}`,
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
    initialHydratedRef.current.delete(projectId);
    hydratedRef.current.delete(projectId);
    refsHydratedRef.current.delete(projectId);
    // references is left out of the deps: the prop has its own effect.
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Hydrates references from the manifest. No cancelled flag: StrictMode's
  // simulated cleanup would set it, and the guarded second run never calls setRefs.
  const refsHydratedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    if (refsHydratedRef.current.has(projectId)) return;
    if (refs.length > 0 && refs[0]?.preview && !refs[0].preview.startsWith(`${API_BASE}/`)) {
      // In-memory refs from onboarding have blob: previews; cache-seeded ones are refetched.
      return;
    }
    refsHydratedRef.current.add(projectId);
    (async () => {
      // /overview for tile metadata, /annotations for the boxes.
      try {
        // imports_limit=0: this effect reads only references.
        const ovr = await apiFetch(
          `/api/v2/projects/${projectId}/overview?imports_limit=0`,
          {},
        );
        if (!ovr.ok) {
          console.warn("[v2 hydrate] overview fetch failed:", ovr.status);
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
        // Fills an empty title; never overrides a rename.
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
          const newUploadStatus: Record<string, "uploading" | "done" | "failed"> = {};
          const hydrated: ReferenceImage[] = list.map((ref) => {
            const url = `${API_BASE}/api/v2/projects/${projectId}/references/${encodeURIComponent(ref.filename)}`;
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

        // scope=refs: the viewer fetches import geometry per image on open.
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
              const boxes = detectionsToBoxes(dets, []);
              // Loaded, not edited: the reference flush starts from these.
              lastFlushedRefBoxesRef.current.set(r.referenceId, JSON.stringify(boxes));
              return { ...r, boxes };
            }));
          }
        } else {
          console.warn("[v2 hydrate] annotations fetch failed:", annR.status);
        }
      } catch (e) {
        console.warn("[v2 hydrate] failed:", e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);
  const updateRefs = (next: ReferenceImage[]) => {
    setRefs(next);
    onReferencesChange?.(next);
  };

  // Per-reference warnings from the server's leave-one-out check.
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

  // Crop embeddings by cropKey, from embed_crops; a reference PUT sends those it has.
  const [refEmbeds, setRefEmbeds] = useState<Map<string, { box: [number, number, number, number]; embedding: number[] }>>(new Map());
  const refEmbedsRef = useRef(refEmbeds);
  useEffect(() => { refEmbedsRef.current = refEmbeds; }, [refEmbeds]);

  // One in-flight flush per reference.
  const flushInFlightRef = useRef<Set<string>>(new Set());

  const flushReferenceEmbeddings = useMemo(
    () => async (ref: ReferenceImage) => {
      if (!projectId || !ref.referenceId) return;
      const flightKey = `${projectId}::${ref.referenceId}`;
      if (flushInFlightRef.current.has(flightKey)) return;
      flushInFlightRef.current.add(flightKey);
      try {
        const boxes = ref.boxes ?? [];
        const missing = boxes
          .map((b, i) => ({ b, i, key: cropKey(ref.preview, b) }))
          .filter(({ key }) => !refEmbedsRef.current.has(key));
        if (missing.length > 0) {
          // Hydrated refs hold an empty File until pixels are needed.
          const file = await ensureRefFile(ref);
          if (file && file.size > 0) {
          try {
            const fd = new FormData();
            fd.append("image", file);
            fd.append(
              "boxes",
              JSON.stringify(missing.map(({ b }) => [b.x0, b.y0, b.x1, b.y1])),
            );
            const r = await apiFetch("/api/v2/references/embed_crops", {
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

        // The PUT embeds any box sent without an embedding.
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

  // Flushes a reference once its boxes differ from the snapshot taken on first sight.
  const lastFlushedRefBoxesRef = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!projectId) return;
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

  // Labelling job. Imports upload without inference; a label_charlie job labels them in one pass.
  const [labelJob, setLabelJob] = useState<LabelJobState | null>(null);
  const [labelJobStarting, setLabelJobStarting] = useState(false);
  // Label-purge job started from DeleteLabelModal.
  const [purgeJob, setPurgeJob] = useState<LabelJobState | null>(null);
  const [labelPendingDelete, setLabelPendingDelete] = useState<{
    canonical: string;
    display: string;
  } | null>(null);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [healthOpen, setHealthOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  // Imports, seeded from the importTiles cache. /overview reconciles and removes stale ids.
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
        preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(t.filename)}`,
        blurhash: t.blurhash ?? null,
        status: "ready" as const,
        width: t.width,
        height: t.height,
        createdAt: t.createdAt ?? null,
        // Until /initial or /overview lists the id; pruned if still set after the full load.
        provisional: true,
      }))
      .sort(compareImportedMediaDesc);
  });
  const importsLatestRef = useRef<ImportedMedia[]>(imports);
  useEffect(() => { importsLatestRef.current = imports; }, [imports]);

  // Non-null while a drop is uploading.
  const [importProgress, setImportProgress] = useState<{ total: number; done: number } | null>(null);

  // Hidden input behind the header's "Add images" action; same handler as the drop zone.
  const headerFileInputRef = useRef<HTMLInputElement | null>(null);

  const datasetSectionRef = useRef<HTMLElement | null>(null);
  const scrollToDataset = () => {
    // After one frame, so the new placeholders have a size.
    if (typeof window === "undefined") return;
    requestAnimationFrame(() => {
      datasetSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  // Imports with unsaved edits; filled by updateImport, drained by the flush.
  const dirtyImportIdsRef = useRef<Set<string>>(new Set());
  // Last edit time per import (performance.now()) and saves in flight. A sync
  // that started before either must not replace those boxes.
  const lastEditAtRef = useRef<Map<string, number>>(new Map());
  const savesInFlightRef = useRef<Set<string>>(new Set());
  // Set below, once flushEdits exists: saves the dirty imports 600 ms
  // after the latest edit.
  const scheduleFlushRef = useRef<() => void>(() => {});

  const updateImport = useCallback((
    id: string,
    patch: Partial<ImportedMedia>,
    // hydration: server data, not an edit. Never marks dirty or replaces existing editedBoxes.
    opts?: { hydration?: boolean },
  ) => {
    if (!opts?.hydration && Object.prototype.hasOwnProperty.call(patch, "editedBoxes")) {
      dirtyImportIdsRef.current.add(id);
      lastEditAtRef.current.set(id, performance.now());
      scheduleFlushRef.current();
      // The server's filter counts predate this edit.
      setFilterCountsOverride(null);
    }
    setImports((cur) => cur.map((m) => {
      if (m.id !== id) return m;
      if (
        opts?.hydration &&
        Object.prototype.hasOwnProperty.call(patch, "editedBoxes") &&
        m.editedBoxes !== undefined
      ) {
        // The local editedBoxes win; only the masks they lack come from the server.
        const rest = { ...patch };
        delete rest.editedBoxes;
        const editedBoxes = Array.isArray(m.editedBoxes)
          ? withMasksFrom(m.editedBoxes, patch.editedBoxes)
          : m.editedBoxes;
        return { ...m, ...rest, editedBoxes };
      }
      const next = { ...m, ...patch };
      if (next.background && next.editedBoxes?.length) next.background = false;
      return next;
    }));
    // Must stay stable: DatasetViewer's hydration effects depend on it and call it.
  }, []);

  // References from onboarding upload in the background; imports wait until all settle.
  type RefUploadStatus = "uploading" | "done" | "failed";
  const [refUploadStatus, setRefUploadStatus] = useState<Record<string, RefUploadStatus>>({});

  const refUploadStatusRef = useRef(refUploadStatus);
  useEffect(() => { refUploadStatusRef.current = refUploadStatus; });
  // Keyed `${projectId}::${preview}`, claimed synchronously, so StrictMode's
  // second effect run does not POST again.
  const refUploadInFlightRef = useRef<Set<string>>(new Set());
  // Uploads run one at a time, in array order.
  const refUploadChainRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    if (!projectId) return;
    // A ref with a referenceId is already on the server.
    const newRefs = refs.filter(
      (r) => !(r.preview in refUploadStatusRef.current) && !r.referenceId,
    );
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
    if (newRefs.length === 0) return;
    setRefUploadStatus((cur) => {
      const next = { ...cur };
      for (const r of newRefs) next[r.preview] = "uploading";
      return next;
    });
    for (const ref of newRefs) {
      const flightKey = `${projectId}::${ref.preview}`;
      if (refUploadInFlightRef.current.has(flightKey)) {
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
          // label force-labels the detections; labels lets the backend detect when no boxes are sent.
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
      // A referenceId means the server has it.
      else if (r.referenceId) done++;
      else unstarted++;
    }
    return { uploading, done, failed, unstarted, pending: unstarted, total: refs.length };
  }, [refs, refUploadStatus]);
  // Every reference uploaded or failed. Gates imports.
  const referencesSettled =
    refUploadCounts.total === 0 ||
    refUploadCounts.uploading + refUploadCounts.unstarted === 0;

  // Dropped videos, shown one at a time in VideoFrameModal.
  const [videoQueue, setVideoQueue] = useState<File[]>([]);
  const [videoExtracting, setVideoExtracting] = useState<{ done: number; total: number } | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Window-level drop target. handleImportFiles changes every render, so the
  // listeners call it through a ref.
  const importFilesRef = useRef<(files: FileList | null) => Promise<void>>();
  const [dropActive, setDropActive] = useState(false);

  useEffect(() => {
    importFilesRef.current = handleImportFiles;
  });

  const handleImportFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    // Ignored while reference uploads are in progress.
    if (!referencesSettled) return;
    const dropped = Array.from(files);
    // No bytes are read here; each file is read when it uploads.
    const droppedVideos = dropped.filter(isVideoFile);
    const droppedImages = dropped.filter((f) => isImageFile(f) && !isVideoFile(f));
    const skippedCount = dropped.length - droppedImages.length - droppedVideos.length;
    if (skippedCount > 0) {
      console.warn(`[v2 import] skipped ${skippedCount} non-image/video file(s)`);
    }

    // Safari invalidates a dropped Blob once the drop completes (WebKitBlobResource error 4).
    const isSafari =
      typeof navigator !== "undefined" &&
      /^((?!chrome|android|crios|fxios|edg).)*safari/i.test(navigator.userAgent);
    const candidates: File[] = isSafari
      ? await Promise.all(droppedImages.map((f) => rereadFileBytes(f)))
      : droppedImages;
    const videos: File[] = isSafari
      ? await Promise.all(droppedVideos.map((f) => rereadFileBytes(f)))
      : droppedVideos;
    const tooBig = videos.filter((v) => v.size > maxImportBytes(v));
    const okVideos = videos.filter((v) => v.size <= maxImportBytes(v));
    if (tooBig.length > 0) {
      const names = tooBig
        .map((v) => {
          const limitMb = Math.round(maxImportBytes(v) / (1024 * 1024));
          return `${v.name} (${(v.size / (1024 * 1024)).toFixed(1)} MB, ${limitMb} MB limit)`;
        })
        .join(", ");
      setVideoError(`Too large to import: ${names}`);
      window.setTimeout(() => setVideoError(null), 6000);
    }
    if (okVideos.length > 0) {
      setVideoQueue((prev) => [...prev, ...okVideos]);
    }
    if (candidates.length === 0) return;

    // Timestamps in reverse drop order sort the first-dropped file first, above existing tiles.
    const baseTs = Date.now();
    const placeholders: ImportedMedia[] = candidates.map((f, i) => ({
      id: `imp_${baseTs}_${i}_${Math.random().toString(36).slice(2, 6)}`,
      file: f,
      preview: "",
      status: "processing",
      createdAt: baseTs + (candidates.length - i),
      // Lets the server dedupe a retried upload.
      idempotencyKey: typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `idem_${baseTs}_${i}_${Math.random().toString(36).slice(2, 10)}`,
    }));
    setImports((cur) => [...placeholders, ...cur].sort(compareImportedMediaDesc));
    setImportsTotal((n) => (n != null ? n + placeholders.length : n));
    setFilterCountsOverride((o) => o && shiftFilterCounts(o, placeholders, {}, 1));
    scrollToDataset();

    // Original bytes, unscaled, in batches to /imports/raw_batch.

    if (!projectId) {
      // No project yet: the hand-off effect uploads these once it exists.
      candidates.forEach((_f, i) => {
        const placeholder = placeholders[i];
        void (async () => {
          try {
            const original = candidates[i];
            updateImport(placeholder.id, { file: original, preview: URL.createObjectURL(original) });
            await processImport({ ...placeholder, file: original, preview: "" });
          } catch (e) {
            console.error("[v2 import-queue] swallowed error:", e);
          }
        })();
      });
      return;
    }

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

      // For a failed batch or a server without raw_batch.
      const fallbackPerImage = async (items: { ph: ImportedMedia; file: File }[]) => {
        for (const { ph, file } of items) {
          try { await processImport({ ...ph, file, preview: "" }); }
          catch (e) { console.error("[v2 import-batch] per-image fallback failed:", e); }
        }
      };

      const uploadBatch = async (idxs: number[]) => {
        const items = idxs.map((i) => ({ ph: placeholders[i], file: candidates[i] }));
        const fd = new FormData();
        for (const { ph, file } of items) {
          const fresh = await rereadFileBytes(file);
          fd.append("images", fresh);
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
        // One setImports pass for the batch; per-item updateImport calls are O(n^2).
        const patches = new Map<string, Partial<ImportedMedia>>();
        items.forEach(({ ph }, k) => {
          const res = results[k];
          if (!res) {
            patches.set(ph.id, { status: "failed", error: "no result" });
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
            // The viewer renders from preview; the grid tile uses backendId.
            preview: res.filename
              ? `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(res.filename)}`
              : ph.preview,
          });
        });
        setImports((cur) => cur.map((m) => {
          const p = patches.get(m.id);
          return p ? { ...m, ...p } : m;
        }));
        bumpDone(items.length);
      };

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

  // Uploads bytes only; /imports/raw stores labelled=false until the labelling job runs.
  const processImport = async (m: ImportedMedia) => {
    if (!projectId) {
      updateImport(m.id, { status: "ready", detections: [] });
      return;
    }

    const attempt = async (file: File): Promise<Response> => {
      const fresh = await rereadFileBytes(file);
      const fd = new FormData();
      fd.append("image", fresh);
      // Without created_at_ms the server uses its own clock.
      if (typeof m.createdAt === "number") {
        fd.append("created_at_ms", String(m.createdAt));
      }
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
      // One retry for transient network errors.
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
      console.error("[v2 import-raw] http", r.status, body);
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
    // AugmentationsCard's previewSources skips entries without a filename.
    updateImport(m.id, {
      status: "ready",
      backendId: pd.import_id,
      filename: pd.filename,
      width: pd.width,
      height: pd.height,
      blurhash: pd.blurhash ?? null,
      detections: [],
      // The viewer renders from preview, not backendId.
      preview: pd.filename
        ? `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(pd.filename)}`
        : m.preview,
    });
    setStatsRefreshSignal((n) => n + 1);
  };

  const unlabelledImportCount = useMemo(
    () => imports.filter((m) => m.status === "ready" && !isLabelled(m)).length,
    [imports],
  );
  const labelledImportCount = useMemo(
    () => imports.filter((m) => m.status === "ready" && isLabelled(m)).length,
    [imports],
  );

  const processingImportCount = useMemo(
    () => imports.filter((m) => m.status === "processing").length,
    [imports],
  );
  // Denominator for the processing card; grows with the queue, resets when it drains.
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
    // A new label or changed settings relabels every image.
    const forceRelabel = freshLabels.length > 0 || settingsChanged;
    if (!forceRelabel && unlabelledImportCount === 0) return;
    setLabelJobStarting(true);
    // Pending edits and tags are saved first: the job reads the manifest.
    await flushEditsRef.current();
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
          params: {
            sam3_threshold: sam3Threshold,
            sam3_mask_threshold: sam3MaskThreshold,
            sam3_min_relative_area: sam3MinRelativeArea,
            // Tile size is the backend default.
            tile_native: tileNative,
            // Relabels every image; hand-edited boxes are kept.
            force_relabel: forceRelabel,
          },
        }),
      });
      if (!r.ok) {
        flashLabelError(await refusalText(r, `Labelling could not start (${r.status}).`));
        return;
      }
      const d = (await r.json()) as { jobId?: string };
      if (d.jobId) {
        setLabelJob({
          jobId: d.jobId,
          status: "running",
          index: 0,
          // imports.length counts only the rows loaded so far.
          total: forceRelabel ? (importsTotal ?? imports.length) : unlabelledImportCount,
          startedAt: Date.now(),
        });
        // Mirrors what the backend records when the job ends.
        setLabelsLastRun(editLabels.slice());
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

  // The editPayload the server holds per import. The auto-PUT skips imports that match it.
  const lastEditedRef = useRef<Map<string, string>>(new Map());
  // Imports the auto-PUT effect has seen; a first sighting is hydration, not an edit.
  const observedImportIdsRef = useRef<Set<string>>(new Set());

  // Merges /annotations into state during and after a labelling job.
  const syncAnnotations = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    const startedAt = performance.now();
    try {
      // no-store and noDedup: a cached or joined response can predate an edit.
      const annR = await apiFetch(
        `/api/v2/projects/${projectId}/annotations?scope=imports`,
        { cache: "no-store" },
        { noDedup: true },
      );
      if (!annR.ok) return;
      const annData = (await annR.json()) as {
        imports?: Record<string, {
          detections?: WireDetection[];
          editedBoxes?: EditableBox[] | null;
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
          // First detections for this image: a new labelledAt busts the preview cache.
          const relabelled = !hadDetections && hasDetections;
          const labelledAt = relabelled ? Date.now() : m.labelledAt;
          // Unsaved, saving, edited since this sync began or mid-gesture: the local copy wins.
          const isDirty =
            dirtyImportIdsRef.current.has(m.id) ||
            savesInFlightRef.current.has(m.id) ||
            (lastEditAtRef.current.get(m.id) ?? -1) >= startedAt ||
            (Array.isArray(m.editedBoxes) &&
              m.editedBoxes.some((b) => b.detecting || b.classifying || b.segmenting));
          let nextEditedBoxes: EditableBox[] | undefined;
          if (isDirty) {
            nextEditedBoxes = m.editedBoxes;
          } else {
            // This payload carries no masks: keep the ones held here.
            nextEditedBoxes = Array.isArray(a.editedBoxes)
              ? withMasksFrom(a.editedBoxes.map(stripTransientBoxFlags), m.editedBoxes)
              : m.editedBoxes;
            // The server keeps editedBoxes through a job, so an empty set yields to fresh detections.
            if (
              relabelled &&
              !m.background &&
              Array.isArray(nextEditedBoxes) &&
              nextEditedBoxes.length === 0
            ) {
              nextEditedBoxes = undefined;
            }
          }
          // Only a server copy becomes the baseline, so a dirty import stays pending.
          if (nextEditedBoxes !== undefined && !isDirty) {
            lastEditedRef.current.set(m.id, editPayload(nextEditedBoxes ?? [], m.background));
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

  // Last job progress index whose annotations were synced.
  const lastSyncedIndexRef = useRef<{ jobId: string; index: number } | null>(null);

  // The poll reads the job through this ref, so progress updates do not restart the interval.
  const labelJobRef = useRef<LabelJobState | null>(null);
  useEffect(() => {
    labelJobRef.current = labelJob;
  }, [labelJob]);

  // Changes with the job, not with its progress.
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
      const cur = labelJobRef.current;
      if (!cur || cur.jobId !== jobId || cur.status !== "running") return;
      try {
        const r = await apiFetch(
          `/api/projects/${projectId}/jobs/${jobId}`,
          { cache: "no-store" },
        );
        // 404: the engine restarted and no longer knows the job.
        if (r.status === 404) {
          const c = labelJobRef.current;
          if (c && c.jobId === jobId) setLabelJob({ ...c, status: "failed", currentImage: null });
          void syncAnnotations();
          return;
        }
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
          setStatsRefreshSignal((n) => n + 1);
          // The job may start an auto-augment that finishes between polls; these refreshes catch it.
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
          void refreshOverviewSoon();
          window.setTimeout(refreshOverviewSoon, 1500);
          window.setTimeout(refreshOverviewSoon, 4000);
          window.setTimeout(refreshOverviewSoon, 10000);
          // The backend re-bakes every labelled preview, so every tile gets a new labelledAt.
          const labelledNow = Date.now();
          setImports((cur) => cur.map((it) => (
            it.backendId ? { ...it, labelledAt: labelledNow } : it
          )));
        } else if (status === "failed" || status === "error") {
          setLabelJob({ ...c, status: "failed", currentImage: null });
        } else if (status === "cancelled" || status === "canceled") {
          setLabelJob({ ...c, status: "cancelled", currentImage: null });
        } else {
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

  // Polls an active purge_label job; on completion resyncs annotations.
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
        // 404: the engine restarted and no longer knows the job.
        if (r.status === 404) {
          setPurgeJob((cur) => cur ? { ...cur, status: "failed", currentImage: null } : cur);
          void syncAnnotations();
          return;
        }
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
        // The next tick retries.
      }
    };
    const id = window.setInterval(tick, 1000);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [purgePollingId, projectId, syncAnnotations]);

  // Re-attaches to a running label_charlie or purge_label job.
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
        // Best effort.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const removeImport = (id: string) => {
    const m = importsLatestRef.current.find((x) => x.id === id);
    if (!m) return;
    if (m.preview.startsWith("blob:")) URL.revokeObjectURL(m.preview);
    // Optimistic; the tile is restored if the DELETE fails.
    setImports((cur) => cur.filter((x) => x.id !== id));
    setImportsTotal((n) => (n != null ? Math.max(0, n - 1) : n));
    const shown = verdictsLatestRef.current;
    setFilterCountsOverride((o) => o && shiftFilterCounts(o, [m], shown, -1));
    const restoreCounts = () => setFilterCountsOverride((o) => o && shiftFilterCounts(o, [m], shown, 1));
    if (!m.backendId || !projectId) return;
    (async () => {
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/imports/${m.backendId}`,
          { method: "DELETE" },
        );
        // 404 and 410 count as deleted.
        if (!r.ok && r.status !== 404 && r.status !== 410) {
          const body = await r.text().catch(() => "");
          console.warn(`[v2 import-delete] http ${r.status} on ${m.backendId}: ${body.slice(0, 200)}`);
          setImports((cur) =>
            cur.some((x) => x.id === id) ? cur : [m, ...cur].sort(compareImportedMediaDesc),
          );
          setImportsTotal((n) => (n != null ? n + 1 : n));
          restoreCounts();
        }
      } catch (e) {
        console.warn("[v2 import-delete] network error:", e);
        setImports((cur) =>
          cur.some((x) => x.id === id) ? cur : [m, ...cur].sort(compareImportedMediaDesc),
        );
        setImportsTotal((n) => (n != null ? n + 1 : n));
        restoreCounts();
      }
    })();
  };

  // Bulk delete for the Dataset's Select mode, in one request.
  const removeImportsBatch = useCallback(async (ids: string[]) => {
    if (!projectId || ids.length === 0) return;
    // Optimistic; not restored if the POST fails. The tiles come from the
    // rendered list, as React may run the setImports updater later.
    const dropSet = new Set(ids);
    const dropped = importsLatestRef.current.filter((m) => dropSet.has(m.id));
    for (const m of dropped) {
      if (m.preview.startsWith("blob:")) URL.revokeObjectURL(m.preview);
    }
    setImports((cur) => cur.filter((x) => !dropSet.has(x.id)));
    if (dropped.length > 0) {
      setImportsTotal((n) => (n != null ? Math.max(0, n - dropped.length) : n));
      const shown = verdictsLatestRef.current;
      setFilterCountsOverride((o) => o && shiftFilterCounts(o, dropped, shown, -1));
    }
    const backendIds = dropped
      .map((m) => m.backendId)
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
    // Refreshes again once the server has applied the delete.
    setStatsRefreshSignal((n) => n + 1);
  }, [projectId]);

  // Mirrors persisted imports into the meta cache.
  useEffect(() => {
    if (!projectId) return;
    const persisted = imports.filter((m) => m.backendId);
    if (persisted.length === 0 && imports.length === 0) {
      // Empty before hydration: the cache is left alone.
      return;
    }
    // A cached blurhash survives a null from the server while its backfill runs.
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
          blurhash: m.blurhash ?? old?.blurhash ?? null,
          width: m.width ?? old?.width,
          height: m.height ?? old?.height,
          createdAt: m.createdAt ?? old?.createdAt ?? null,
        };
      }),
      nImages: persisted.length,
    });
  }, [imports, projectId]);

  // Hydrates import tiles from /overview. No cancelled flag, for the same
  // StrictMode reason as the reference hydration.
  const hydratedRef = useRef<Set<string>>(new Set());
  // True once /overview has loaded or failed; gates the empty-dataset text.
  const [importsReady, setImportsReady] = useState(false);
  // The server's imports_total; imports fills in progressively.
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
        background?: boolean;
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
      // The first 100 imports, then the rest in the background.
      const FIRST_BATCH = 100;
      // Three tries with linear back-off; no-store keeps the HTTP cache out.
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
      setImportsTotal(total);
      try {
        const imps = firstSlice.imports ?? [];
        if (imps.length === 0 && total === 0) {
          // Nothing on the server: provisional tiles go, in-flight uploads stay.
          setImports((cur) => cur.filter((m) => !m.provisional));
          return;
        }
        const hydrated: ImportedMedia[] = imps.map((imp) => {
          // Placeholder detections, n_detections of them spread over label_set, until /annotations.
          const labelSet = imp.label_set ?? [];
          const nDetections = imp.n_detections ?? 0;
          const labelStats: Record<string, number> = (imp as {
            label_stats?: Record<string, number>;
          }).label_stats ?? Object.fromEntries(labelSet.map((lab) => [lab, 0]));
          const placeholderDetections: ImportDetection[] = nDetections > 0
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
            preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
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
            background: !!imp.background,
            detections: placeholderDetections,
            labelledAt: imp.labelledAt ?? undefined,
          };
        });
        setImports((cur) => {
          const byId = new Map(hydrated.map((h) => [h.id, h]));
          const merged: ImportedMedia[] = [];
          for (const c of cur) {
            // Local tiles keep their local id for life; backendId holds the server id.
            const matchKey = byId.has(c.id)
              ? c.id
              : (c.backendId && byId.has(c.backendId) ? c.backendId : null);
            if (matchKey) {
              const fresh = byId.get(matchKey)!;
              merged.push({
                ...fresh,
                id: c.id,
                backendId: c.backendId,
                detections: (hasRealDetections(c.detections) || c.editedBoxes !== undefined)
                  ? c.detections
                  : fresh.detections,
                editedBoxes: c.editedBoxes,
                blurhash: fresh.blurhash ?? c.blurhash ?? null,
                provisional: undefined,
              });
              byId.delete(matchKey);
            } else if (c.provisional || !c.backendId) {
              // The first batch covers only the newest 100, so unconfirmed and in-flight tiles stay.
              merged.push(c);
            }
            // A confirmed tile absent from the server's list is dropped.
          }
          for (const h of byId.values()) merged.push(h);
          return merged.sort(compareImportedMediaDesc);
        });
        // The whole dataset fits in the first batch, so unconfirmed tiles go now.
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
        setImportsReady(true);
      }
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
            const placeholderDetections: ImportDetection[] = nDetections > 0
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
              preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
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
              background: !!imp.background,
              detections: placeholderDetections,
            };
          });
          setImports((cur) => {
            // Cache-seeded tiles carry no labelStats or detectionCount, so the fresh row is adopted.
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
                  blurhash: fresh.blurhash ?? c.blurhash ?? null,
                  provisional: undefined,
                });
              } else {
                out.push(c);
              }
            }
            for (const h of hydratedRest) {
              if (!consumed.has(h.id)) out.push(h);
            }
            // Everything is loaded: a tile still provisional is not on the server.
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

  // Refetches /overview, up to 3 times, while a tile lacks a blurhash; the server backfills them.
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
    // 3 s, 6 s, 12 s.
    const delay = 3000 * Math.pow(2, attempt);
    const t = window.setTimeout(async () => {
      try {
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

  // The first sighting of an import seeds its save baseline, so a server copy is never PUT back.
  useEffect(() => {
    for (const m of imports) {
      if (!m.backendId) continue;
      if (observedImportIdsRef.current.has(m.id)) continue;
      observedImportIdsRef.current.add(m.id);
      if (m.editedBoxes !== undefined) {
        lastEditedRef.current.set(m.id, editPayload(m.editedBoxes, m.background));
      }
    }
  }, [imports]);

  // Saves dirty imports 600 ms after the latest edit. A failed save stays dirty and
  // retries with backoff; pending saves flush on unmount and on pagehide.
  const importsRef = useRef(imports);
  importsRef.current = imports;
  const flushTimerRef = useRef<number | null>(null);
  const retryDelayRef = useRef(1000);
  const [saveFailing, setSaveFailing] = useState(false);
  const flushEditsRef = useRef<(opts?: { keepalive?: boolean }) => Promise<void>>(async () => {});
  flushEditsRef.current = async (opts) => {
    if (!projectId) return;
    const dirtyIds = Array.from(dirtyImportIdsRef.current);
    dirtyImportIdsRef.current.clear();
    const importsById = new Map(importsRef.current.map((m) => [m.id, m]));
    let failed = false;
    await Promise.all(dirtyIds.map(async (dirtyId) => {
      const m = importsById.get(dirtyId);
      if (!m || !m.backendId || !m.editedBoxes) return;
      // A box mid-gesture is not saved yet; the import stays dirty.
      if (m.editedBoxes.some((b) => b.detecting || b.classifying || b.segmenting)) {
        dirtyImportIdsRef.current.add(dirtyId);
        return;
      }
      const body = editPayload(m.editedBoxes.map(stripTransientBoxFlags), m.background);
      const last = lastEditedRef.current.get(m.id);
      if (last === body) return;
      lastEditedRef.current.set(m.id, body);
      savesInFlightRef.current.add(m.id);
      let ok = false;
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/imports/${m.backendId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body,
          // keepalive outlives a hidden page; browsers cap it at 64 KB.
          keepalive: !!opts?.keepalive && body.length < 60_000,
        });
        ok = r.ok;
        if (!ok) console.warn(`[v2 import-patch] http ${r.status} on ${m.backendId}`);
      } catch (e) {
        console.warn("[v2 import-patch] failed:", e);
      } finally {
        savesInFlightRef.current.delete(m.id);
      }
      if (!ok) {
        // Restores the baseline; the import stays dirty.
        if (lastEditedRef.current.get(m.id) === body) {
          if (last === undefined) lastEditedRef.current.delete(m.id);
          else lastEditedRef.current.set(m.id, last);
        }
        dirtyImportIdsRef.current.add(dirtyId);
        failed = true;
        return;
      }
      // A save re-bakes the labelled preview.
      const now = Date.now();
      setImports((cur) => cur.map((it) => (it.id === m.id ? { ...it, labelledAt: now } : it)));
      setStatsRefreshSignal((n) => n + 1);
    }));
    if (failed) {
      setSaveFailing(true);
      scheduleFlush(retryDelayRef.current);
      retryDelayRef.current = Math.min(retryDelayRef.current * 2, 30_000);
    } else if (dirtyIds.length > 0) {
      setSaveFailing(false);
      retryDelayRef.current = 1000;
    }
  };
  const scheduleFlush = useCallback((delay = 600) => {
    if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
    flushTimerRef.current = window.setTimeout(() => {
      flushTimerRef.current = null;
      void flushEditsRef.current();
    }, delay);
  }, []);
  scheduleFlushRef.current = scheduleFlush;
  useEffect(() => {
    const onHide = () => { void flushEditsRef.current({ keepalive: true }); };
    window.addEventListener("pagehide", onHide);
    return () => {
      window.removeEventListener("pagehide", onHide);
      if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
      void flushEditsRef.current();
    };
  }, []);

  // Numbers title and label undo steps in order, so Ctrl+Z undoes the latest of either.
  const undoSeqRef = useRef(0);

  // Renamable project title with 2-step undo.
  const [projectTitle, setProjectTitle] = useState(projectName);
  const [titleEditing, setTitleEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState(projectName);
  const [titleHistory, setTitleHistory] = useState<{ name: string; seq: number }[]>([]);
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

  // Saves a dataset name as Settings' Rename does; false when the engine refuses it.
  const saveTitle = useCallback(async (name: string): Promise<boolean> => {
    if (!projectId) return false;
    try {
      const r = await apiFetch(`/api/projects/${projectId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) return false;
      const data = (await r.json()) as { name?: string };
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId, name: data.name ?? name },
      }));
      return true;
    } catch {
      return false;
    }
  }, [projectId]);

  const commitTitleRename = () => {
    const next = titleDraft.trim();
    setTitleEditing(false);
    if (!next || next === projectTitle) return;
    const prev = projectTitle;
    const seq = ++undoSeqRef.current;
    setTitleHistory((h) => [{ name: prev, seq }, ...h].slice(0, 2));
    setProjectTitle(next);
    void saveTitle(next).then((ok) => {
      if (ok) return;
      // Refused: the old name stays.
      setProjectTitle(prev);
      setTitleHistory((h) => (h[0]?.seq === seq ? h.slice(1) : h));
    });
  };

  const undoTitleRename = () => {
    const [prev, ...rest] = titleHistory;
    if (!prev) return;
    setTitleHistory(rest);
    setProjectTitle(prev.name);
    void saveTitle(prev.name);
  };

  // Editable labels with 2-step undo of add, delete and rename.
  type LabelSnapshot = {
    labels: string[];
    refs: ReferenceImage[];
    aliases: Record<string, string>;
    // Set when the step was a server-side rename.
    renamed?: { from: string; to: string };
    seq: number;
  };
  const [editLabels, setEditLabels] = useState<string[]>(() => {
    // Without passed labels, seed from the localStorage cache.
    if (labels.length > 0) return labels;
    if (!projectId) return labels;
    try {
      const cached = readProjectMeta(projectId);
      if (cached?.labels && cached.labels.length > 0) return cached.labels;
    } catch { /* ignore localStorage failures */ }
    return labels;
  });
  // Labels of the last label_charlie job; null if never labelled. Set optimistically
  // when a job starts.
  const [labelsLastRun, setLabelsLastRun] = useState<string[] | null>(null);
  // Lowercased labels present in any detection or edited box.
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
  // Labels not yet searched for: absent from labelsLastRun or, without it, from detectedLabelSet.
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
  // Transient toast for a label operation the engine refuses.
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

  // Display names by lowercased canonical label; detections keep the canonical label.
  const [labelAliases, setLabelAliases] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined" || !projectId) return {};
    const cached = readProjectMeta(projectId);
    return cached?.labelAliases ?? {};
  });
  // Colour overrides, lowercased canonical label -> "#rrggbb". Set in Settings.
  const [labelColours, setLabelColours] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined" || !projectId) return {};
    const cached = readProjectMeta(projectId);
    return cached?.labelColours ?? {};
  });
  // Parent of a derived dataset, from /overview; null otherwise.
  const [derivedInfo, setDerivedInfo] = useState<{ parentProjectId?: string; parentName?: string } | null>(null);

  // Window-level drop listeners. Declared after derivedInfo, which gates them.
  useEffect(() => {
    // A derived dataset takes no imports.
    if (derivedInfo) return;
    let depth = 0;
    const hasFiles = (e: DragEvent) =>
      Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth += 1;
      setDropActive(true);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDropActive(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = 0;
      setDropActive(false);
      // Drop zones that handle the event call preventDefault.
      if (e.defaultPrevented) return;
      e.preventDefault();
      // The import card's set: images and videos.
      const files = Array.from(e.dataTransfer?.files ?? []).filter(
        (f) =>
          f.type.startsWith("image/") ||
          f.type.startsWith("video/") ||
          /\.(jpe?g|png|webp|bmp|heic|avif|mp4|mov|webm|mkv|mjpe?g)$/i.test(f.name),
      );
      if (files.length) void importFilesRef.current?.(files as unknown as FileList);
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [derivedInfo]);

  // A new derived dataset starts empty while its crops are made; polls /overview
  // until they arrive. derivedPollDone ends the loader if none do.
  const [derivedPollDone, setDerivedPollDone] = useState(false);
  useEffect(() => {
    if (!projectId || !derivedInfo || imports.length > 0 || derivedPollDone) return;
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
              preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
              blurhash: imp.blurhash ?? null,
              status: "ready" as const,
              width: imp.width,
              height: imp.height,
              createdAt: imp.createdAt ?? null,
              sourceUrl: imp.source?.url ?? null,
              derivedLabel: (imp as { derivedLabel?: string | null }).derivedLabel ?? null,
              nAugmentations: imp.n_augmentations ?? 0,
              detectionCount: imp.n_detections ?? 0,
              labelStats: imp.label_stats ?? {},
            })).sort(compareImportedMediaDesc));
            return;
          }
        }
      } catch { /* keep polling */ }
      if (!alive) return;
      if (tries >= 40) { setDerivedPollDone(true); return; } // About 2 min.
      window.setTimeout(tick, 3000);
    };
    const t = window.setTimeout(tick, 2000);
    return () => { alive = false; window.clearTimeout(t); };
  }, [projectId, derivedInfo, imports.length, derivedPollDone]);

  // Review verdicts by backendId, or by local id before the upload lands.
  const [verdicts, setVerdicts] = useState<Record<string, "good" | "bad" | "unsure">>({});
  const verdictsLatestRef = useRef(verdicts);
  useEffect(() => { verdictsLatestRef.current = verdicts; }, [verdicts]);

  // Server filter counts over the whole dataset; null means count the loaded tiles.
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

  const [manifestUpdatedAt, setManifestUpdatedAt] = useState<string | null>(null);
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
      // The same verdict again clears it.
      const next = { ...cur };
      if (next[id] === v) delete next[id];
      else next[id] = v;
      scheduleVerdictsSave(next);
      return next;
    });
    setFilterCountsOverride(null);
  }, [scheduleVerdictsSave]);
  // Review mode sets a verdict; the key of the one an image has keeps it.
  const setVerdictTo = useCallback((id: string, v: "good" | "bad" | "unsure") => {
    setVerdicts((cur) => {
      if (cur[id] === v) return cur;
      const next = { ...cur, [id]: v };
      scheduleVerdictsSave(next);
      return next;
    });
    setFilterCountsOverride(null);
  }, [scheduleVerdictsSave]);

  type VerdictFilter = "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("all");

  const [reviewing, setReviewing] = useState(false);
  const [reviewScope, setReviewScope] = useState<"unrated" | "unsure" | "good" | "bad" | "all">("unrated");
  // The reference editor is open; hides the gallery's back-to-top button.
  const [refViewerOpen, setRefViewerOpen] = useState(false);

  // Masks for one import, for ReviewModeV2; fetched once per backendId.
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
        // The local copy wins; server masks fill the boxes that lack them.
        const editedBoxes = Array.isArray(m.editedBoxes)
          ? withMasksFrom(m.editedBoxes, cleanedEdited)
          : (cleanedEdited ?? m.editedBoxes);
        return { ...m, detections: dets, editedBoxes };
      }));
    } catch {
      reviewAnnoFetchedRef.current.delete(backendId);
    }
  }, [projectId]);
  // One annotations sync on the first review open, for the masks.
  const annotationsHydratedRef = useRef(false);
  useEffect(() => {
    if (!reviewing || annotationsHydratedRef.current) return;
    annotationsHydratedRef.current = true;
    void syncAnnotations();
  }, [reviewing, syncAnnotations]);
  // The alias if set, else the canonical label.
  const displayLabel = (canonical: string): string => {
    if (!canonical) return canonical;
    const k = canonical.trim().toLowerCase();
    return labelAliases[k] || canonical;
  };

  // Resets editLabels only on a project change, so a poll cannot overwrite an edit in progress.
  const labelInitRef = useRef<string | null>(null);
  useEffect(() => {
    if (labelInitRef.current !== projectId) {
      labelInitRef.current = projectId ?? null;
      setEditLabels(labels);
      setLabelsLastRun(null);
      // Aliases, colours, verdicts, tags and last-run settings from /overview.
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
            // Loaded from the server, so not saved back to it.
            const loadedTags = Array.isArray(m?.tags) ? (m!.tags as string[]) : null;
            labelsSavedRef.current = JSON.stringify({
              labels: labels.length > 0 || !loadedTags ? labels : loadedTags,
              aliases: freshAliases,
              colours: freshColours,
            });
            // An unsaved verdict change wins.
            if (
              !verdictsDirtyRef.current
              && m && typeof m.verdicts === "object" && m.verdicts
            ) {
              setVerdicts(m.verdicts as Record<string, "good" | "bad" | "unsure">);
            }
            const freshTags = Array.isArray(m?.tags) ? (m!.tags as string[]) : undefined;
            patchProjectMeta(projectId, {
              labelAliases: freshAliases,
              labelColours: freshColours,
              ...(freshTags ? { labels: freshTags } : {}),
            });
            // Without labels from the parent, the manifest's tags are used.
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
          })
          .catch(() => { setLabelAliases({}); setLabelColours({}); });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, labels]);

  // Settings broadcasts pixelkit-project-meta-changed on every save.
  useEffect(() => {
    if (typeof window === "undefined" || !projectId) return;
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail as {
        projectId?: string;
        labelColours?: Record<string, string>;
      } | null | undefined;
      if (!d || d.projectId !== projectId) return;
      if (d.labelColours) setLabelColours(d.labelColours);
    };
    window.addEventListener("pixelkit-project-meta-changed", handler);
    return () => window.removeEventListener("pixelkit-project-meta-changed", handler);
  }, [projectId]);

  // Bumped to refetch the stats: seedStats, OverviewPanel's and the health modal's.
  const [statsRefreshSignal, setStatsRefreshSignal] = useState(0);
  // The Overview panel's full-statistics health score, cached per dataset, so the
  // header shows the same number on every tab. A dataset with no images has none.
  const [fullHealthScore, setFullHealthScore] = useState<number | undefined>(undefined);
  useEffect(() => {
    if (!projectId) return;
    try {
      const raw = window.localStorage.getItem(`pk_stats:${projectId}`);
      const cached = raw ? (JSON.parse(raw) as { counts?: { imports?: number }; health?: { score?: number } } | null) : null;
      setFullHealthScore(cached?.counts?.imports !== 0 && typeof cached?.health?.score === "number" ? cached.health.score : undefined);
    } catch {
      setFullHealthScore(undefined);
    }
  }, [projectId]);
  const reportHealthScore = useCallback((score: number | undefined) => {
    if (score !== undefined) setFullHealthScore(score);
  }, []);

  useEffect(() => {
    if (!projectId || statsRefreshSignal === 0) return;
    let alive = true;
    apiFetch(`/api/v2/projects/${projectId}/dataset-stats?lite=true`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d) setSeedStats(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, [projectId, statsRefreshSignal]);

  const importsCountRef = useRef(-1);
  useEffect(() => {
    if (importsCountRef.current === imports.length) return;
    importsCountRef.current = imports.length;
    setStatsRefreshSignal((n) => n + 1);
  }, [imports.length]);

  // Set by a variation-plot click; clears after a few seconds.
  const [highlightedImportId, setHighlightedImportId] = useState<string | null>(null);
  const [highlightKind, setHighlightKind] = useState<"image" | "augmentation" | null>(null);
  const highlightTimerRef = useRef<number | null>(null);
  const jumpToImport = useCallback((importId: string, kind: "image" | "augmentation") => {
    setHighlightedImportId(importId);
    setHighlightKind(kind);
    // DatasetGallery mounts the tile if it lies past displayLimit.
    try {
      window.dispatchEvent(new CustomEvent("pixelkit-ensure-import-visible", {
        detail: { projectId, importId },
      }));
    } catch { /* ignore */ }
    // Two frames: one for the displayLimit change, one for the render.
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        const el = document.querySelector<HTMLElement>(
          `[data-import-id="${CSS.escape(importId)}"]`,
        );
        el?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });
    if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current);
    highlightTimerRef.current = window.setTimeout(() => {
      setHighlightedImportId(null);
      setHighlightKind(null);
    }, kind === "augmentation" ? 3800 : 2600);
  }, [projectId]);
  useEffect(() => () => {
    if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current);
  }, []);

  // pixelkit-augmentations-generated: augmentations were added or removed.
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
        /* ignore */
      }
      setStatsRefreshSignal((n) => n + 1);
    };
    window.addEventListener("pixelkit-augmentations-generated", handler);
    return () => window.removeEventListener("pixelkit-augmentations-generated", handler);
  }, [projectId]);

  // Boxes written by a job this view does not track, such as labelling from the Train card.
  useEffect(() => {
    const handler = (e: Event) => {
      if ((e as CustomEvent).detail?.projectId !== projectId) return;
      void syncAnnotations();
      setStatsRefreshSignal((n) => n + 1);
    };
    window.addEventListener("pixelkit-labels-changed", handler);
    return () => window.removeEventListener("pixelkit-labels-changed", handler);
  }, [projectId, syncAnnotations]);

  // Polls the active augment job every 2 s on every tab, for the job cards
  // here and in AugmentationsCard, and announces its end.
  const augActiveJobRef = useRef<string | null>(null);
  const refreshAugCountsRef = useRef<() => Promise<void>>(async () => {});
  // The active augment_generate job, as a LabelJobState for the job card.
  const [augmentJob, setAugmentJob] = useState<LabelJobState | null>(null);
  useEffect(() => {
    if (!projectId) return;
    if (isIdle) return;
    let cancelled = false;
    const refreshOverview = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (!r.ok || cancelled) return;
        const ov = await r.json() as { imports?: { id: string; n_augmentations?: number }[] };
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
    refreshAugCountsRef.current = refreshOverview;
    const tick = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augment/job/active`);
        if (!r.ok || cancelled) return;
        const data = await r.json();
        const prev = augActiveJobRef.current;
        const isActive = data && (data.status === "running" || data.status === "queued");
        if (isActive) {
          const startedMs = data.startedAt
            ? new Date(data.startedAt).getTime()
            : Date.now();
          const freshIndex = Number(data.progress?.index ?? 0);
          // The backend updates n_augmentations every 25 images; stats refresh as the index advances.
          setAugmentJob((cur) => {
            if (cur && freshIndex > (cur.index ?? 0)) {
              setStatsRefreshSignal((n) => n + 1);
            }
            return {
              jobId: data.id,
              status: "running",
              index: freshIndex || (cur?.index ?? 0),
              // progress.total counts augmentations; n_images is no substitute.
              total: Number(data.progress?.total ?? cur?.total ?? 0),
              startedAt: cur?.startedAt ?? startedMs,
              currentImage: data.progress?.image ?? cur?.currentImage ?? null,
            };
          });
        }
        if (prev && !isActive) {
          setAugmentJob((cur) => cur ? {
            ...cur,
            status: data?.status === "failed" ? "failed"
              : data?.status === "cancelled" ? "cancelled"
              : "done",
            index: cur.total,
            currentImage: null,
          } : cur);
          // Once per job. The listener above refreshes nAugmentations and the
          // stats; an open augmentations viewer reloads its copies.
          window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
            detail: { projectId, jobId: prev },
          }));
        } else if (!isActive) {
          // A job that ended before the first poll; closes AugmentationsCard's optimistic entry.
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

  // Saves labels, aliases and colours 600 ms after the latest change. The
  // values a dataset opens with came from the server, so they count as saved.
  const labelsSavedRef = useRef<string>(
    JSON.stringify({ labels: editLabels, aliases: labelAliases, colours: labelColours }),
  );
  useEffect(() => {
    if (!projectId) return;
    const current = JSON.stringify({ labels: editLabels, aliases: labelAliases, colours: labelColours });
    if (current === labelsSavedRef.current) return;
    const t = window.setTimeout(async () => {
      if (current === labelsSavedRef.current) return;
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
          patchProjectMeta(projectId, {
            labels: editLabels,
            labelAliases,
            labelColours,
          });
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

  const pushLabelHistory = () => {
    const seq = ++undoSeqRef.current;
    setLabelHistory((h) =>
      [{
        labels: editLabels.slice(),
        refs: refs.slice(),
        aliases: { ...labelAliases },
        seq,
      }, ...h].slice(0, 2),
    );
  };

  const removeLabel = (idx: number) => {
    const canonical = (editLabels[idx] ?? "").trim().toLowerCase();
    if (!canonical) return;
    setLabelPendingDelete({
      canonical,
      display: displayLabel(editLabels[idx] ?? canonical),
    });
  };

  // Called from DeleteLabelModal. Once the engine accepts the purge_label job,
  // the label leaves the chips and the job removes it from every image.
  const confirmPurgeLabel = async (canonical: string) => {
    if (!projectId) return;
    try {
      const r = await apiFetch(`/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: projectId,
          kind: "purge_label",
          params: { label: canonical },
        }),
      });
      if (!r.ok) {
        flashLabelError(await refusalText(r, `The label could not be removed (${r.status}).`));
        return;
      }
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
        flashLabelError(await refusalText(r, `Annotations could not be cleared (${r.status}).`));
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

  // Typed or pasted text may hold several labels, split on commas and newlines.
  const addLabel = (raw: string) => {
    const labels = [...editLabels];
    const colours = { ...labelColours };
    for (const next of splitLabels(raw)) {
      if (labels.some((l) => l.toLowerCase() === next.toLowerCase())) continue;
      // A new label may not equal an existing alias.
      if (Object.values(labelAliases).some((disp) => disp.toLowerCase() === next.toLowerCase())) continue;
      // A colour no chip shows, saved as an override: automatic colours do not avoid overrides.
      const usedHex = new Set<string>();
      for (const lab of labels) {
        usedHex.add(colourForLabel(labels, lab, colours).toLowerCase());
      }
      const free = LABEL_COLOURS.filter((c) => !usedHex.has(c.toLowerCase()));
      const pickPool = free.length > 0 ? free : LABEL_COLOURS;
      colours[next.toLowerCase()] = pickPool[Math.floor(Math.random() * pickPool.length)];
      labels.push(next);
    }
    setAddInput("");
    if (labels.length === editLabels.length) return;
    pushLabelHistory();
    setEditLabels(labels);
    setLabelColours(colours);
    void syncAnnotations();
    setStatsRefreshSignal((n) => n + 1);
  };

  // Saves label edits still waiting on the 600 ms debounce, so a late save
  // cannot undo a rename the server has made.
  const flushLabelSave = async () => {
    const current = JSON.stringify({ labels: editLabels, aliases: labelAliases, colours: labelColours });
    if (!projectId || current === labelsSavedRef.current) return;
    const r = await apiFetch(`/api/projects/${projectId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tags: editLabels, label_aliases: labelAliases, labelColours }),
    });
    if (r.ok) labelsSavedRef.current = current;
  };

  // Renames a label on the server, where every box, reference and augmentation
  // follows, then here. Returns false when the server refused.
  const applyRename = async (from: string, to: string): Promise<boolean> => {
    if (!projectId) return false;
    try {
      await flushLabelSave();
      const r = await apiFetch(`/api/projects/${projectId}/labels/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_label: from, new_label: to }),
      });
      if (!r.ok) {
        flashLabelError(await refusalText(r, `Rename failed (${r.status}).`));
        return false;
      }
    } catch {
      flashLabelError("Rename failed: the engine is not reachable.");
      return false;
    }
    const fromKey = from.trim().toLowerCase();
    const rename = (lab: string | undefined) =>
      lab !== undefined && lab.trim().toLowerCase() === fromKey ? to : lab;
    setEditLabels((cur) => cur.map((l) => rename(l) ?? l));
    setLabelAliases((cur) => {
      const out = { ...cur };
      delete out[fromKey];
      delete out[to.toLowerCase()];
      return out;
    });
    setLabelColours((cur) => {
      if (!(fromKey in cur)) return cur;
      const out = { ...cur, [to.toLowerCase()]: cur[fromKey] };
      if (fromKey !== to.toLowerCase()) delete out[fromKey];
      return out;
    });
    updateRefs(refs.map((r) => ({
      ...r,
      label: rename(r.label),
      boxes: r.boxes?.map((b) => ({ ...b, label: rename(b.label) ?? b.label })),
    })));
    setImports((cur) => cur.map((m) => (Array.isArray(m.editedBoxes)
      ? { ...m, editedBoxes: m.editedBoxes.map((b) => ({ ...b, label: rename(b.label) ?? b.label })) }
      : m)));
    await syncAnnotations();
    setStatsRefreshSignal((n) => n + 1);
    return true;
  };

  const renameLabel = async (idx: number, raw: string) => {
    const next = raw.trim();
    const canonical = editLabels[idx];
    if (!canonical || !next) return;
    const k = canonical.trim().toLowerCase();
    if (next === (labelAliases[k] || canonical)) return;
    // Names stay unique: renaming onto another label would merge the two.
    if (editLabels.some((l, i) => i !== idx && l.toLowerCase() === next.toLowerCase())) return;
    if (Object.entries(labelAliases).some(
      ([key, disp]) => key !== k && disp.toLowerCase() === next.toLowerCase()
    )) return;
    const snapshot = {
      labels: editLabels.slice(),
      refs: refs.slice(),
      aliases: { ...labelAliases },
      renamed: { from: canonical, to: next },
    };
    if (await applyRename(canonical, next)) {
      const seq = ++undoSeqRef.current;
      setLabelHistory((h) => [{ ...snapshot, seq }, ...h].slice(0, 2));
    }
  };

  const undoLastLabelOp = async () => {
    const prev = labelHistory[0];
    if (!prev) return;
    setLabelHistory((h) => h.slice(1));
    // A rename is undone on the server too.
    if (prev.renamed) {
      if (!(await applyRename(prev.renamed.to, prev.renamed.from))) return;
    }
    setEditLabels(prev.labels);
    setLabelAliases(prev.aliases);
    updateRefs(prev.refs);
  };

  // Ctrl+Z undoes the latest title or label step.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const meta = e.ctrlKey || e.metaKey;
      if (!meta || e.key.toLowerCase() !== "z" || e.shiftKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      // The viewer, painter and Review own Ctrl+Z while they are open.
      if (document.querySelector('[aria-modal="true"]')) return;
      const title = titleHistory[0];
      const label = labelHistory[0];
      if (!title && !label) return;
      e.preventDefault();
      if (title && (!label || title.seq > label.seq)) undoTitleRename();
      else undoLastLabelOp();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [titleHistory, labelHistory]);

  useEffect(() => {
    if (adding) addInputRef.current?.focus();
  }, [adding]);

  const tab: ProjectTab = section;
  const setTab = useCallback(
    (next: ProjectTab) => onSectionChange?.(next),
    [onSectionChange],
  );
  // The view scrolls inside the shell's [data-dataset-scroll], else the window.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const scroller = document.querySelector<HTMLElement>("[data-dataset-scroll]");
    if (scroller) scroller.scrollTo({ top: 0 });
    else window.scrollTo({ top: 0 });
  }, [tab]);
  // A section change closes the review overlay; DatasetGallery closes its viewers via activeSection.
  useEffect(() => {
    setReviewing(false);
  }, [tab]);

  useEffect(() => {
    if (tab !== "dataset") return;
    refreshAugCountsRef.current?.();
  }, [tab]);

  const [inputSize, setInputSize] = useState<string>("256x256");
  // Collapsed on load; opens only on a click.
  const [detectionSettingsOpen, setDetectionSettingsOpen] = useState(false);

  // Equal to the backend's SAM3_* defaults.
  const SAM3_DEFAULTS = {
    threshold: 0.5,        // SAM3_THRESHOLD
    maskThreshold: 0.5,    // SAM3_MASK_THRESHOLD
    minRelativeArea: 0.05, // SAM3_MIN_RELATIVE_AREA, 5 %
  } as const;
  const [sam3Threshold, setSam3Threshold] = useState<number>(SAM3_DEFAULTS.threshold);
  const [sam3MaskThreshold, setSam3MaskThreshold] = useState<number>(SAM3_DEFAULTS.maskThreshold);
  const [sam3MinRelativeArea, setSam3MinRelativeArea] = useState<number>(SAM3_DEFAULTS.minRelativeArea);
  // Off: SAM3 downscales each frame to 1500 px. On: the backend tiles large frames at native resolution.
  const [tileNative, setTileNative] = useState<boolean>(false);
  const sam3IsDefault =
    sam3Threshold === SAM3_DEFAULTS.threshold
    && sam3MaskThreshold === SAM3_DEFAULTS.maskThreshold
    && sam3MinRelativeArea === SAM3_DEFAULTS.minRelativeArea
    && !tileNative;
  // Settings of the last labelling run; settingsChanged compares the sliders with them.
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
  const [clearAllOpen, setClearAllOpen] = useState(false);

  // Cached server-side in projects/<id>/dataset_type.json. Read-only here.
  const [datasetType, setDatasetType] = useState<DatasetTypeValue | null>(() => {
    if (!projectId) return null;
    return readProjectMeta(projectId)?.datasetType ?? null;
  });
  // Fetched only when the cache has no value.
  useEffect(() => {
    if (!projectId) return;
    if (datasetType) return;
    if (editLabels.length === 0) { setDatasetType(null); return; }
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/dataset-type`);
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

  // Adding references makes the dataset "specific", as the engine does.
  const prevRefsLenRef = useRef(0);
  useEffect(() => {
    const grew = refs.length > prevRefsLenRef.current;
    prevRefsLenRef.current = refs.length;
    if (!projectId) return;
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
  }, [refs.length, datasetType, projectId]);

  // The dataset's last change, from /overview.
  const dateLabel = useMemo(() => {
    const d = manifestUpdatedAt ? new Date(manifestUpdatedAt) : null;
    if (!d || Number.isNaN(d.getTime())) return null;
    return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric" }).format(d);
  }, [manifestUpdatedAt]);

  // Entry animation: one frame in the start state, then the CSS transition.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(false);
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, [projectId, tab]);

  // The shell keeps its scroll offset across project changes.
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
    // none, not translateY(0): a transform would capture fixed overlays inside.
    transform: mounted ? "none" : "translateY(8px)",
    transition: `opacity 320ms ease-out ${delay}ms, transform 360ms cubic-bezier(0.2, 0.7, 0.2, 1) ${delay}ms`,
  });

  // The loader waits for the first imports, the lite stats and the labels.
  const seedStatsReady = seedStats !== null;
  const labelsReady = editLabels.length > 0 || labels.length === 0;
  const pageReady = importsReady && seedStatsReady && labelsReady;
  const [loaderVisible, setLoaderVisible] = useState<boolean>(!pageReady);
  useEffect(() => {
    if (pageReady) setLoaderVisible(false);
  }, [pageReady]);
  // Hidden after 6 s regardless.
  useEffect(() => {
    if (!loaderVisible) return;
    const t = window.setTimeout(() => setLoaderVisible(false), 6000);
    return () => window.clearTimeout(t);
  }, [loaderVisible]);

  const isFirstLoad = firstLoad === "onboarding";
  const showFullscreenLoader = loaderVisible && !isFirstLoad;

  // A dataset with references counts as specific before the fetch agrees.
  const isSpecific = datasetType?.type === "specific" || refs.length > 0;

  // Text for the mount loader's back button.
  const backTo = backToProjectId ? "project" : "workspace";

  return (
    <main className="min-h-screen bg-[var(--background)]">
      {showFullscreenLoader && (
        // The content area only: below the 36px title bar, above the 24px status bar.
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
          {onClose && (
            <button
              type="button"
              onClick={() => {
                setLoaderVisible(false);
                onClose();
              }}
              aria-label={`Back to ${backTo}`}
              className="absolute top-4 right-4 rounded-md border border-[var(--line)] bg-[var(--surface)]/85 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] px-3 py-1.5 text-[11px] uppercase tracking-wider font-mono text-foreground/75 hover:text-foreground transition-colors"
              style={{ boxShadow: "var(--shadow-soft)" }}
            >
              ← Back to {backTo}
            </button>
          )}
          <PixelKitLoader size={120} message="Loading dataset…" />
        </div>
      )}
      <div className="w-full min-w-0 max-w-[2000px]">
      <section className="px-6 lg:px-10 pt-5" style={fade()}>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex min-w-0 flex-1 items-center gap-2.5">
            {titleEditing ? (
              <input
                ref={titleInputRef}
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") { e.preventDefault(); commitTitleRename(); }
                  if (e.key === "Escape") { setTitleEditing(false); }
                }}
                onBlur={() => setTitleEditing(false)}
                aria-label="Rename dataset"
                className="w-full max-w-md border-b border-[var(--line-strong)] bg-transparent pb-0.5 text-[16px] font-medium tracking-tight text-[var(--foreground)] outline-none"
              />
            ) : (
              <h1
                className="max-w-full overflow-hidden text-ellipsis whitespace-nowrap text-[16px] font-medium tracking-tight text-[var(--foreground)] cursor-text hover:underline underline-offset-4 decoration-[0.5px] decoration-[var(--line-strong)]"
                onClick={() => setTitleEditing(true)}
                title="Click to rename"
              >
                {projectTitle}
              </h1>
            )}
            {derivedInfo && (
              <a
                href={derivedInfo.parentProjectId ? `/app/${derivedInfo.parentProjectId}` : undefined}
                className="inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md border border-[var(--line)] px-2 font-mono text-[11px] uppercase tracking-wider text-[var(--fg-dim)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--fg-soft)]"
                title={derivedInfo.parentName ? `Derived from ${derivedInfo.parentName}. Click to open it.` : "A cropped child dataset derived from a parent dataset"}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" aria-hidden />
                Derived
              </a>
            )}
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            {!derivedInfo && (() => {
              const running = labelJob != null && labelJob.status === "running";
              const processing = processingImportCount > 0;
              const idleCopy = (() => {
                if (editLabels.length === 0) return "Label images";
                if (
                  unlabelledImportCount === 0
                  && freshLabels.length === 0
                  && labelledImportCount > 0
                ) {
                  return settingsChanged ? "Relabel all" : "All images labelled";
                }
                if (freshLabels.length === 1) return `Label ${displayLabel(freshLabels[0])}`;
                if (freshLabels.length > 1) return "Label new labels";
                if (labelledImportCount === 0 && editLabels.length === 1) {
                  return `Label ${displayLabel(editLabels[0])}`;
                }
                if (labelledImportCount === 0) return "Label images";
                return "Label new images";
              })();
              const buttonText = labelJobStarting
                ? "Starting…"
                : running
                ? "Labelling…"
                : processing
                ? "Processing…"
                : idleCopy;
              const inert =
                labelJobStarting
                || running
                || processing
                || editLabels.length === 0
                || (unlabelledImportCount === 0 && freshLabels.length === 0 && !settingsChanged);
              return (
                <button
                  type="button"
                  onClick={startLabellingJob}
                  disabled={inert}
                  title={
                    editLabels.length === 0
                      ? "Add labels first. Labelling needs at least one label."
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
                  className={inert ? "pk-btn" : "pk-btn pk-btn-primary"}
                >
                  {buttonText}
                </button>
              );
            })()}
            {!derivedInfo && (
              <>
                <input
                  ref={headerFileInputRef}
                  type="file"
                  accept={`image/*,${VIDEO_ACCEPT}`}
                  multiple
                  className="hidden"
                  disabled={!referencesSettled}
                  onChange={(e) => {
                    handleImportFiles(e.target.files);
                    if (headerFileInputRef.current) headerFileInputRef.current.value = "";
                  }}
                />
                <button
                  type="button"
                  onClick={() => { if (referencesSettled) headerFileInputRef.current?.click(); }}
                  disabled={!referencesSettled}
                  className="pk-btn"
                  title={referencesSettled ? "Add images to this dataset" : "Wait for references to finish saving"}
                >
                  Add images
                </button>
              </>
            )}
            <button
              type="button"
              onClick={() => { if (projectId) setExportOpen(true); }}
              className="pk-btn"
              title="Export dataset"
            >
              Export
            </button>
            <button
              type="button"
              onClick={() => { if (projectId) setSettingsOpen(true); }}
              className="pk-btn px-2"
              aria-label="Dataset settings"
              title="Dataset settings"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
          </div>
        </div>
        <div className="pk-micro mt-2 flex flex-wrap items-center gap-x-2.5 gap-y-1 tabular-nums">
          {projectId && (
            <span title={`Dataset ID: ${projectId}`}>{projectId.slice(0, 8)}</span>
          )}
          {projectId && dateLabel && (
            <span aria-hidden className="text-[var(--fg-faint)]">·</span>
          )}
          {dateLabel && <span>Updated {dateLabel}</span>}
        </div>
      </section>

      {/* Counts from the lite snapshot, else summed over the loaded tiles; health as the Overview panel computes it. */}
      <section className="px-6 lg:px-10 pt-4" style={rise()}>
        {(() => {
          const ss = seedStats as
            | { counts?: { imports?: number; detections?: number; augmentations?: number }; health?: { score?: number } }
            | null;
          const detections = ss?.counts?.detections ?? imports.reduce((a, i) => a + (i.detectionCount ?? 0), 0);
          const augmentations = ss?.counts?.augmentations ?? imports.reduce((a, i) => a + (i.nAugmentations ?? 0), 0);
          // No images, no score.
          const score = ss?.counts?.imports === 0 ? undefined : fullHealthScore ?? ss?.health?.score;
          const healthTone = healthLabel(score);
          const items = [
            { label: "Images", value: importsTotal != null ? importsTotal : imports.length },
            { label: "Detections", value: detections },
            { label: "Labels", value: editLabels.length },
            { label: "Augmentations", value: augmentations },
          ];
          return (
            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-[var(--line)] bg-[var(--line-soft)] sm:grid-cols-3 lg:grid-cols-5">
              {items.map((s) => (
                <div key={s.label} className="bg-[var(--panel-solid)] px-4 py-3">
                  <div className="text-[20px] font-medium leading-none tracking-tight tabular-nums text-[var(--foreground)]">
                    {s.value.toLocaleString()}
                  </div>
                  <div className="pk-micro mt-1.5">{s.label}</div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => setHealthOpen(true)}
                title="Open dataset health"
                className="bg-[var(--panel-solid)] px-4 py-3 text-left outline-none transition-colors hover:bg-[var(--surface-2)] focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-[var(--focus-ring)]"
              >
                <div className="text-[20px] font-medium leading-none tracking-tight tabular-nums text-[var(--foreground)]">
                  {typeof score === "number" ? Math.round(score) : "-"}
                </div>
                <div className="pk-micro mt-1.5">{healthTone ? `Health · ${healthTone}` : "Health"}</div>
              </button>
            </div>
          );
        })()}
      </section>

      <section className="px-6 lg:px-10 pt-3 pb-4" style={rise()}>
        <div
          className="flex flex-wrap items-center gap-1.5"
          style={{ position: "relative", zIndex: adding ? 60 : undefined }}
        >
          <span className="pk-micro mr-1.5">Labels</span>
          {editLabels.map((lab, i) => (
            <EditableChip
              key={lab}
              label={displayLabel(lab)}
              colour={colourForLabel(editLabels, lab, labelColours)}
              onDelete={() => removeLabel(i)}
              onRename={(next) => renameLabel(i, next)}
            />
          ))}
          {adding ? (
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
              className="h-6 rounded-md border border-[var(--line)] bg-transparent px-2 font-mono text-[12px] text-[var(--foreground)] outline-none transition-colors placeholder:text-[var(--fg-faint)] focus:border-[var(--line-strong)]"
              style={{ width: "9rem", position: "relative", zIndex: 70 }}
            />
          ) : (
            <button
              type="button"
              onClick={() => setAdding(true)}
              aria-label="Add label"
              title="Add label"
              className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-[var(--line)] text-[var(--fg-dim)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--foreground)]"
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
          )}
        </div>
      </section>

      {tab === "overview" && (
        <div className="pk-up">
          <OverviewPanel
            imports={imports}
            importsTotal={importsTotal}
            labelColours={labelColours}
            projectId={projectId}
            refreshSignal={statsRefreshSignal}
            refs={refs}
            showReferences={!derivedInfo}
            seedStats={seedStats as Parameters<typeof OverviewPanel>[0]["seedStats"]}
            onOpenHealth={() => setHealthOpen(true)}
            onOpenReferences={() => setTab("references")}
            onOpenDataset={() => setTab("dataset")}
            onJumpToImport={(id, kind) => { setTab("dataset"); jumpToImport(id, kind ?? "image"); }}
            onHealthScore={reportHealthScore}
          />
          {projectId && <DerivedDatasetsBar projectId={projectId} labels={editLabels} />}
        </div>
      )}
      <DatasetHealthModal
        open={healthOpen}
        onClose={() => setHealthOpen(false)}
        projectId={projectId}
        labelColours={labelColours}
        refreshSignal={statsRefreshSignal}
        onJumpToImport={(id, kind) => {
          setHealthOpen(false);
          setTab("dataset");
          jumpToImport(id, kind);
        }}
      />

      {/* Tabs are hidden, not unmounted, so their state survives a tab change. */}
      <div hidden={tab !== "references"} className="pk-up">
      {/* No entrance animation: visible on first paint. */}
      <section className="px-6 lg:px-10 pt-3 pb-3">
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-5 py-4">
          <div className="flex items-baseline gap-2">
            <h2 className="pk-micro text-[var(--fg-soft)]">Reference images</h2>
            <span className="font-mono text-[11px] tabular-nums text-[var(--fg-dim)]">
              {refs.length}
              {expectedRefCount > refs.length ? ` / ${expectedRefCount}` : ""}
            </span>
          </div>

          <div>
            <div>
              <p className="mt-2 text-[13px] text-foreground/65 leading-relaxed">
                Reference images are optional. Add examples to improve label
                matching, especially when classes look similar.
              </p>
              <p className="mt-1 text-[13px] text-foreground/55">
                Drop each photo into its label section below. The section is the label, so PixelKit knows exactly what each reference shows.
              </p>

              {/* Reference boxes per label, against ANNOTS_PER_LABEL. */}
              {editLabels.length > 0 && (
                <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
                  {editLabels.map((lab) => {
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
                        className="inline-flex h-6 items-center gap-1.5 rounded-md border border-[var(--line)] px-2 font-mono text-[12px] text-[var(--fg-soft)]"
                      >
                        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: bg }} aria-hidden />
                        {displayLabel(lab)}
                        {annotDone ? (
                          <svg viewBox="0 0 24 24" className="h-3 w-3 shrink-0 text-[var(--ok)]" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-label="Complete">
                            <polyline points="5 12 10 17 19 7" />
                          </svg>
                        ) : (
                          <span className="text-[10px] tabular-nums text-[var(--fg-dim)]">{annotCount}/{ANNOTS_PER_LABEL}</span>
                        )}
                      </span>
                    );
                  })}
                </div>
              )}

              <RefImageGrid
                refs={refs}
                onChange={updateRefs}
                projectId={projectId}
                labels={editLabels}
                labelColours={labelColours}
                pendingCount={Math.max(0, expectedRefCount - refs.length)}
                onLeaveImage={flushReferenceEmbeddings}
                refQuality={refQuality}
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

      {refUploadCounts.uploading > 0 && (
        <section className="px-6 lg:px-10 pt-0 pb-3" style={rise()}>
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-3 flex items-center gap-3">
            <span className="relative flex h-2 w-2 shrink-0" aria-hidden>
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--warn)] opacity-60" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--warn)]" />
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-[12px] font-medium text-[var(--warn)]">
                Saving references: {refUploadCounts.done}/{refUploadCounts.total} done
                {refUploadCounts.failed > 0 ? ` · ${refUploadCounts.failed} failed` : ""}
              </div>
              <div className="text-[11px] text-foreground/55 mt-0.5">
                Adding images is paused until the references are saved.
              </div>
            </div>
          </div>
        </section>
      )}

      </div>
      <div hidden={tab !== "dataset"} className="pk-up">
      {!derivedInfo && (
      <section className="px-6 lg:px-10 pt-3 pb-3" style={rise()}>
        <ImportMediaSection
          onUpload={handleImportFiles}
          projectId={projectId}
          alreadyImportedUrls={imports
            .map((m) => m.sourceUrl)
            .filter((u): u is string => !!u)}
          onOpenverseAdded={async (added) => {
            if (added <= 0 || !projectId) return;
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
                    preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
                    blurhash: imp.blurhash ?? null,
                    status: "ready" as const,
                    width: imp.width,
                    height: imp.height,
                    createdAt: imp.createdAt ?? null,
                    sourceUrl: imp.source?.url ?? null,
                    nAugmentations: imp.n_augmentations ?? 0,
                  };
                });
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
              ? `Saving references: ${refUploadCounts.done}/${refUploadCounts.total}. You can add images once they are saved.`
              : undefined
          }
        />
      </section>
      )}

      {importProgress && importProgress.total > 0 && (
        <section className="px-6 lg:px-10 pb-3">
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-5 py-4">
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
                className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-300 ease-out motion-reduce:transition-none"
                style={{
                  width: `${Math.round(
                    (Math.min(importProgress.done, importProgress.total) /
                      Math.max(1, importProgress.total)) * 100,
                  )}%`,
                }}
              />
            </div>
            <p className="mt-2 text-[11px] text-foreground/45">
              You can keep working. Images appear in the gallery as they finish uploading.
            </p>
            {importProgress.total >= 500 && (
              <p className="mt-1 text-[11px] text-[var(--warn)]">
                Large import of {importProgress.total.toLocaleString()} images. Keep PixelKit open until it
                finishes. Closing it stops the remaining uploads; images already uploaded are kept.
              </p>
            )}
          </div>
        </section>
      )}

      {/* A derived dataset's boxes come from its parent: no labelling controls. */}
      {!derivedInfo && (
      <>
      <section className="px-6 lg:px-10 pt-3 pb-3" style={rise()}>
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)]">
          <button
            type="button"
            onClick={() => setDetectionSettingsOpen((v) => !v)}
            aria-expanded={detectionSettingsOpen}
            className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left outline-none transition-colors hover:bg-[var(--surface-hover)] focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-[var(--focus-ring)]"
          >
            <span className="flex items-baseline gap-2.5">
              <span className="pk-micro text-[var(--fg-soft)]">Detection settings</span>
              <span className="font-mono text-[11px] text-[var(--fg-dim)]">
                {sam3IsDefault ? "default" : "custom"}
              </span>
            </span>
            <svg
              viewBox="0 0 24 24"
              className="h-3.5 w-3.5 shrink-0 text-[var(--fg-dim)] transition-transform"
              style={{ transform: detectionSettingsOpen ? "rotate(180deg)" : "none" }}
              fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
          {detectionSettingsOpen && (
            <div className="border-t border-[var(--line-soft)] px-4 pb-4">
              <div className="mt-4 grid gap-4 sm:grid-cols-3">
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
              <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-[var(--line-soft)] pt-4">
                <div className="min-w-0 flex-1">
                  <div className="text-[12px] font-medium text-[var(--foreground)]">Resolution</div>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-foreground/45">
                    {tileNative
                      ? "Large images are sliced into native-resolution tiles (a 4K frame ≈ 7 model passes instead of 1). Catches small objects but is slower and uses more compute."
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
              <div className="mt-4 flex items-center justify-between border-t border-[var(--line-soft)] pt-3">
                <button
                  type="button"
                  onClick={() => setClearAllOpen(true)}
                  className="text-[11px] uppercase tracking-wider font-mono text-[var(--bad)] opacity-80 hover:opacity-100 transition-opacity"
                  title="Delete every detection and edited box in this dataset"
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
          )}
        </div>
      </section>

      {/* The grid animates the job card's height. */}
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
                  "Removing the label's detections…",
                  "Removing boxes with this label…",
                  "Refreshing labelled previews…",
                  "Updating label colours and names…",
                ]}
                doneMessage="All annotations with this label were removed."
              />
            )}
          </div>
        </div>
      </section>

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

      {/* Frames are extracted client-side and imported as images. */}
      {videoQueue[0] && (
        <VideoFrameModal
          // A fresh instance per video, so no trim or load state carries over.
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
            setVideoExtracting({ done: 0, total: extractCount(params) });
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
              // handleImportFiles only needs Array.from to accept it.
              await handleImportFiles(frames as unknown as FileList);
            }
          }}
        />
      )}

      {videoError && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[1250] rounded-md border border-[var(--line-strong)] bg-[var(--surface)] px-4 py-2 text-[12px] text-[var(--bad)] shadow-lg backdrop-blur-md">
          {videoError}
        </div>
      )}

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

      {/* Scroll target after an import. */}
      <section
        ref={datasetSectionRef}
        className="px-6 lg:px-10 pt-2 pb-4 scroll-mt-20"
      >
        <DatasetGallery
          items={imports}
          totalImports={importsTotal}
          isPulling={!!derivedInfo && imports.length === 0 && !derivedPollDone}
          labels={editLabels}
          hasReferenceEmbeddings={refs.length > 0}
          // A general dataset gets no "No reference embeddings" warning.
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
          verdicts={verdicts}
          onVerdict={setVerdict}
          verdictFilter={verdictFilter}
          onVerdictFilterChange={setVerdictFilter}
          onStartReview={(scope) => { setReviewScope(scope); setReviewing(true); }}
          filterCountsOverride={filterCountsOverride}
          manifestUpdatedAt={manifestUpdatedAt}
          refViewerOpen={refViewerOpen}
          activeSection={tab}
        />
      </section>

      {/* pointer-events-none: the drop lands on window. */}
      {dropActive && typeof window !== "undefined" &&
        createPortal(
          <div className="pointer-events-none fixed inset-0 z-[1400] grid place-items-center bg-[var(--background)]/70 backdrop-blur-[2px]">
            <div className="rounded-lg border-2 border-dashed border-[var(--accent)] bg-[var(--panel-solid)] px-8 py-6 text-center shadow-[var(--shadow-strong)]">
              <p className="text-sm font-medium text-[var(--foreground)]">Drop to import</p>
              <p className="mt-1 text-xs text-[var(--fg-muted)]">Images and videos join this dataset like any other upload</p>
            </div>
          </div>,
          document.body,
        )}

      {reviewing && projectId && (
        <ReviewModeV2
          items={(() => {
            // Verdicts are keyed by backendId, so only persisted imports are reviewed.
            // A scope takes its gallery pill's filter, so the deck matches the count.
            const persisted = imports.filter((m) => !!m.backendId && !!m.preview);
            const filtered = persisted.filter((m) => matchesGalleryFilter(m, reviewScope, verdicts));
            return filtered.map((m) => ({
              id: m.backendId as string,
              filename: m.filename ?? m.file?.name ?? m.backendId as string,
              preview: m.preview,
              width: m.width,
              height: m.height,
              // As in the viewer: editedBoxes wins even when empty, else the non-rejected detections.
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
          onVerdict={setVerdictTo}
          onClose={() => setReviewing(false)}
          scope={reviewScope}
          projectLabels={editLabels}
          labelAliases={labelAliases}
          labelColours={labelColours}
          onRequestAnnotations={requestReviewAnnotations}
        />
      )}

      <div className="pb-24" />
      </div>

      <div hidden={tab !== "augmentations"} className="pk-up">
        <div style={rise()}>
          <AugmentationsCard
            projectId={projectId ?? null}
            augmentJob={augmentJob}
            onAugmentJobChange={setAugmentJob}
            visible={tab === "augmentations"}
            previewSources={[
              // Imports first. A filename is enough for the preview endpoint.
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

      </div>

      {exportOpen && projectId && (
        <ExportModal
          projectId={projectId}
          projectName={projectTitle}
          inputShape={inputSize}
          onClose={() => setExportOpen(false)}
        />
      )}

      {settingsOpen && projectId && (
        <ProjectSettingsV2
          projectId={projectId}
          projectName={projectTitle}
          labels={editLabels}
          inputShape={inputSize}
          inputShapeOptions={INPUT_SHAPES}
          onInputShapeChange={setInputSize}
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
            requestExplorerRefresh();
          }}
          onLabelColoursChange={(next) => {
            setLabelColours(next);
            if (projectId) patchProjectMeta(projectId, { labelColours: next });
          }}
          onCoverChange={() => {
            // Workspace cards refresh their cover on a meta cache change.
            if (projectId) patchProjectMeta(projectId, {});
          }}
          onDeleted={() => {
            requestExplorerRefresh();
            setSettingsOpen(false);
            onClose();
          }}
        />
      )}
      {/* Above the image viewer until a retry succeeds. */}
      {saveFailing && (
        <div
          role="status"
          aria-live="polite"
          className="fixed left-1/2 bottom-8 -translate-x-1/2 z-[1250] flex items-center gap-2.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-4 py-2.5 text-[12px] font-medium text-[var(--warn)]"
          style={{ boxShadow: "var(--shadow-strong)" }}
        >
          <span className="inline-flex rounded-full h-2 w-2 shrink-0 bg-[var(--warn)]" aria-hidden />
          Some changes have not been saved yet. Retrying…
        </div>
      )}
      {labelError && (
        <div
          role="alert"
          aria-live="polite"
          className="fixed left-1/2 bottom-8 -translate-x-1/2 z-[1200] max-w-sm rounded-lg px-4 py-2.5 text-[12px] font-mono text-red-50 bg-red-800/95 border border-red-300/30"
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

// Reference image grid

// Reference images per project, across all labels.
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
  onViewerOpenChange,
  onRefDeleted,
}: {
  refs: ReferenceImage[];
  onChange: (next: ReferenceImage[]) => void;
  /** Lets the reference editor's ML calls load saved images by project_id and filename. */
  projectId?: string | null;
  labels?: string[];
  /** Per-label colour overrides for the filter-chip dots. */
  labelColours?: Record<string, string> | null;
  /** References still loading, drawn as grey placeholders. */
  pendingCount?: number;
  /** Called when the editor leaves a reference: prev, next or close. */
  onLeaveImage?: (ref: ReferenceImage) => void;
  /** Quality by reference id, then by detection index. */
  refQuality?: RefQuality;
  /** Called as the full-screen editor opens and closes. */
  onViewerOpenChange?: (open: boolean) => void;
  /** Called with the backend id of a removed reference, for the server DELETE. */
  onRefDeleted?: (referenceId: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [processing, setProcessing] = useState<Set<string>>(new Set());
  const [viewingIdx, setViewingIdx] = useState<number | null>(null);
  const viewing = viewingIdx !== null ? refs[viewingIdx] ?? null : null;
  useEffect(() => {
    onViewerOpenChange?.(viewingIdx !== null);
  }, [viewingIdx, onViewerOpenChange]);
  // Previews already sent to triggerPipeline.
  const triggered = useRef<Set<string>>(new Set());
  // Latest refs, so concurrent pipeline callbacks do not overwrite each other.
  const refsRef = useRef<ReferenceImage[]>(refs);
  useEffect(() => { refsRef.current = refs; }, [refs]);
  const remaining = Math.max(0, REF_MAX_POOL - refs.length);

  // Detects a reference's boxes through /api/v2/references/process.
  const triggerPipeline = async (ref: ReferenceImage) => {
    setProcessing((prev) => new Set([...prev, ref.preview]));
    try {
      const fd = new FormData();
      fd.append("image", ref.file);
      // A reference in a label section is detected for that label only.
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
      // Boxes already set, by a pipeline run or by hand, are kept.
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

  // Runs the pipeline once for each reference without boxes.
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

    const placeholders: ReferenceImage[] = candidates.map((f) => ({
      file: f,
      preview: URL.createObjectURL(f),
      label: sectionLabel,
    }));
    // Claimed now, so the recovery effect does not run the pipeline as well.
    for (const p of placeholders) triggered.current.add(p.preview);
    onChange([...refs, ...placeholders]);
    for (const p of placeholders) triggerPipeline(p);
    if (fileRef.current) fileRef.current.value = "";
  };

  const removeAt = (idx: number) => {
    const list = refs.slice();
    const [gone] = list.splice(idx, 1);
    if (gone) {
      try { URL.revokeObjectURL(gone.preview); } catch { /* already revoked */ }
      if (gone.referenceId) onRefDeleted?.(gone.referenceId);
    }
    onChange(list);
  };

  // A reference belongs to the section it was dropped in, else to its first box's label.
  // pendingUploadLabel is the section that opened the shared file picker.
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

  const renderThumb = (r: ReferenceImage) => {
    const isProcessing = processing.has(r.preview);
    // Index in refs, for the editor's prev and next and for removeAt.
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
      ? `Outlier: this reference's embedding doesn't match its own class (self-sim ${worstWarning.self_score?.toFixed(2)}, below 0.50). Re-crop or remove it.`
      : worstWarning?.warning === "looks like other class"
        ? `Looks like "${worstWarning.other_label}" (sim ${worstWarning.other_score?.toFixed(2)}) more than this section's label (sim ${worstWarning.self_score?.toFixed(2)}). Check it is in the right section.`
        : worstWarning?.warning === "only ref for class"
          ? "Only reference in this section. Add a few more so the centroid isn't a single point."
          : null;
    return (
      <div key={r.preview} className="group relative h-24 w-24 rounded-lg overflow-hidden bg-foreground/5 border border-foreground/10">
        <LazyRefImage
          src={r.preview}
          blurhash={r.blurhash}
          isProcessing={isProcessing}
          onClick={() => !isProcessing && i >= 0 && setViewingIdx(i)}
        />
        {isProcessing ? (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <svg className="h-7 w-7 animate-spin text-[var(--foreground)]" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
              <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          </div>
        ) : (
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); if (i >= 0) removeAt(i); }}
            className="absolute top-1 right-1 h-6 w-6 rounded-md bg-black/70 text-white/85 hover:text-white hover:bg-black/85 grid place-items-center text-sm opacity-0 group-hover:opacity-100 transition-opacity"
            aria-label="Remove reference"
          >×</button>
        )}
        {worstWarning && warningTitle && !isProcessing && (
          <Tooltip
            label={warningTitle}
            width={260}
            className="absolute bottom-1 left-1 right-1 z-10 block"
          >
            <div
              className={[
                "w-full px-1.5 py-0.5 rounded-md text-[9px] font-medium tabular-nums backdrop-blur-sm flex items-center gap-1",
                worstWarning.warning === "outlier"
                  ? "bg-black/60 text-rose-200 border border-white/10"
                  : worstWarning.warning === "looks like other class"
                    ? "bg-black/60 text-amber-200 border border-white/10"
                    : "bg-black/60 text-white/70 border border-white/10",
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
      {/* Each label section is a drop zone that tags its references with the label. */}
      {labels.length === 0 ? (
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-6 text-center text-[12px] text-foreground/45">
          Add at least one label above to start uploading references.
        </div>
      ) : (
        <div className="grid gap-3">
          {labels.map((lab) => {
            const k = lab.toLowerCase();
            const c = colourForLabel(labels, lab, labelColours);
            const owned = refs.filter((r) => (sectionOf(r) ?? "").toLowerCase() === k);
            const isDragTarget = dragLabel === lab;
            const canAdd = remaining > 0;
            return (
              <div
                key={lab}
                onDragEnter={(e) => { if (canAdd) { e.preventDefault(); setDragLabel(lab); } }}
                onDragOver={(e) => { if (canAdd) { e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = "copy"; } }}
                onDragLeave={(e) => { if (e.currentTarget === e.target) setDragLabel((cur) => (cur === lab ? null : cur)); }}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragLabel(null);
                  if (remaining <= 0) return;
                  // Safari can leave dataTransfer.files empty while items holds the files.
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
                  "rounded-lg border p-4 transition-colors",
                  isDragTarget
                    ? "border-foreground/45 bg-foreground/[0.05]"
                    : "border-[var(--line)] hover:border-[var(--line-strong)]",
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
                  <button
                    type="button"
                    onClick={() => { setPendingUploadLabel(lab); fileRef.current?.click(); }}
                    disabled={!canAdd}
                    className="text-[13px] px-3.5 py-1.5 rounded-md border border-[var(--line)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] disabled:opacity-35 disabled:cursor-not-allowed transition-colors shrink-0"
                  >
                    + Add photos
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {owned.length === 0 && (
                    <span className="text-xs text-foreground/35 leading-relaxed py-3">
                      No references yet. Add or drop 3 to 5 clear photos of {lab}.
                    </span>
                  )}
                  {owned.map((r) => renderThumb(r))}
                </div>
              </div>
            );
          })}

          {/* References whose label is not a project label. */}
          {unsortedRefs.length > 0 && (
            <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] p-4">
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
            // Looked up in refs: the viewing closure predates the latest edits.
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
      <p className="mt-3 text-[11px] text-foreground/35">
        {remaining > 0
          ? `${remaining} of ${REF_MAX_POOL} reference slots remaining · jpg · png · webp`
          : `Maximum ${REF_MAX_POOL} references reached`}
      </p>
    </div>
  );
}

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

  return (
    <span className="group inline-flex h-6 items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent px-2 font-mono text-[12px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--foreground)]">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: colour }} aria-hidden />
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
          className="bg-transparent font-mono text-[12px] text-[var(--foreground)] outline-none placeholder:opacity-40"
          style={{ width: `${Math.max(2, draft.length)}ch` }}
        />
      ) : (
        <span
          role={onRename ? "button" : undefined}
          tabIndex={onRename ? 0 : undefined}
          onClick={() => onRename && setEditing(true)}
          onKeyDown={(e) => { if (onRename && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); setEditing(true); } }}
          className={["select-none", onRename ? "hover:underline underline-offset-2 cursor-text" : "cursor-default"].join(" ")}
        >
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={onDelete}
        aria-label={`Remove ${label}`}
        className="inline-flex h-4 w-0 shrink-0 items-center justify-center overflow-hidden rounded-sm text-[var(--fg-dim)] opacity-0 transition-all duration-150 ease-out group-hover:w-4 group-hover:opacity-100 hover:text-[var(--bad)] motion-reduce:transition-none"
      >
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden>
          <path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </button>
    </span>
  );
}


// Relabel hotkeys: 1-9, then QWERTYUIOP for labels 10-19 and ASDFGHJKL for 20-28.
// The row stops at L: ; and ' differ between keyboard layouts.
const RELABEL_TOP_ROW = "qwertyuiop".split("");
const RELABEL_HOME_ROW = "asdfghjkl".split("");
const RELABEL_KEY_LIMIT = 9 + RELABEL_TOP_ROW.length + RELABEL_HOME_ROW.length;

// Key for a 0-based slot; null past RELABEL_KEY_LIMIT.
function keyForRelabelSlot(slot: number): string | null {
  if (slot < 0) return null;
  if (slot < 9) return String(slot + 1);
  if (slot < 9 + RELABEL_TOP_ROW.length) return RELABEL_TOP_ROW[slot - 9].toUpperCase();
  if (slot < RELABEL_KEY_LIMIT) return RELABEL_HOME_ROW[slot - 9 - RELABEL_TOP_ROW.length].toUpperCase();
  return null;
}

// KeyboardEvent.key -> slot, case-insensitive; null outside the rows.
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

// Verdict toggle in the image viewer header.
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
      active: "bg-[var(--surface-hover)] text-[var(--ok)]",
      idle: "text-foreground/55 hover:text-[var(--ok)]",
    },
    {
      key: "unsure",
      label: "Unsure",
      active: "bg-[var(--surface-hover)] text-[var(--warn)]",
      idle: "text-foreground/55 hover:text-[var(--warn)]",
    },
    {
      key: "bad",
      label: "Bad",
      active: "bg-[var(--surface-hover)] text-[var(--bad)]",
      idle: "text-foreground/55 hover:text-[var(--bad)]",
    },
  ];
  return (
    <div className="inline-flex items-center gap-0.5 rounded-md bg-foreground/[0.04] p-0.5" role="group" aria-label="Set image verdict">
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
              "rounded px-2.5 py-1 text-[11px] uppercase tracking-wider font-medium transition-colors",
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
        <span className="pk-micro">{label}</span>
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

function SegmentedControl<T extends string>({
  value, onChange, options,
}: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-md border border-[var(--line)] p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={["rounded px-3 py-1 text-xs uppercase tracking-wide transition-colors", value === opt.value ? "bg-[var(--surface-hover)] text-[var(--foreground)]" : "text-foreground/60 hover:text-foreground"].join(" ")}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function ImportMediaSection({
  onUpload,
  projectId,
  alreadyImportedUrls,
  onOpenverseAdded,
  disabled = false,
  disabledMessage,
}: {
  onUpload?: (files: FileList | null) => void;
  /** For the Openverse panel, which is hidden when null. */
  projectId?: string | null;
  /** Source URLs already imported; Openverse results exclude them. */
  alreadyImportedUrls?: string[];
  /** Called after an Openverse batch is added. */
  onOpenverseAdded?: (added: number) => Promise<void> | void;
  /** Blocks uploads while the references are saving. */
  disabled?: boolean;
  /** Message shown over the drop zone when disabled. */
  disabledMessage?: string;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleMediaFiles = (files: FileList | File[] | null) => {
    if (disabled) return;
    // A File[] works wherever onUpload reads a FileList.
    onUpload?.(files as FileList | null);
  };
  // Safari refuses a drop unless dragover calls preventDefault and sets dropEffect.
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
    // Safari can leave dataTransfer.files empty while items holds the files.
    const dt = e.dataTransfer;
    const native = dt.files;
    const files: File[] = native && native.length > 0 ? Array.from(native) : [];
    if (files.length === 0 && dt.items && dt.items.length > 0) {
      const items = Array.from(dt.items);
      for (const item of items) {
        if (item.kind === "file") {
          const f = item.getAsFile();
          if (f) files.push(f);
        }
      }
    }
    if (files.length === 0) return;
    handleMediaFiles(files);
  };

  return (
    <div
      onDragEnter={onDragEnter} onDragLeave={onDragLeave} onDragOver={onDragOver} onDrop={onDrop}
      onClick={() => { if (!disabled) fileInputRef.current?.click(); }}
      aria-disabled={disabled}
      className={[
        "relative flex min-w-0 flex-wrap items-center justify-between gap-x-6 gap-y-3 rounded-md border px-5 py-4 transition-colors",
        disabled
          ? "cursor-not-allowed border-[var(--line)] bg-[var(--panel)] opacity-60"
          : dragOver
            ? "cursor-pointer border-[var(--accent)] bg-[var(--accent-dim)]"
            : "cursor-pointer border-[var(--line)] bg-[var(--panel)] hover:border-[var(--line-strong)]",
      ].join(" ")}
    >
      <input ref={fileInputRef} type="file" accept={`image/*,${VIDEO_ACCEPT}`} multiple className="hidden" disabled={disabled} onChange={(e) => { handleMediaFiles(e.target.files); if (fileInputRef.current) fileInputRef.current.value = ""; }} />

      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium text-[var(--foreground)]">
          {disabled ? (disabledMessage ?? "Waiting for setup to finish…") : "Drop images or click to browse"}
        </div>
        <div className="mt-1 font-mono text-[11px] text-[var(--fg-dim)]">
          jpg · png · webp · gif · heic · avif · bmp · tiff · mp4 · mov · webm · mkv · mjpeg
        </div>
      </div>

      <div
        className="flex shrink-0 items-center gap-2"
        // Keeps these clicks from opening the file picker.
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={() => { if (!disabled) fileInputRef.current?.click(); }}
          disabled={disabled}
          className="pk-btn"
        >
          Browse
        </button>
        {projectId && (
          <OpenverseInlinePanel
            projectId={projectId}
            alreadyImportedUrls={alreadyImportedUrls ?? []}
            onAdded={onOpenverseAdded}
          />
        )}
      </div>
    </div>
  );
}

function cropKey(preview: string, box: { x0: number; y0: number; x1: number; y1: number }): string {
  return `${preview}|${Math.round(box.x0)},${Math.round(box.y0)},${Math.round(box.x1)},${Math.round(box.y1)}`;
}

// The override, else the project palette, where no two labels share a colour, else a stable hash.
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

// Smallest box side after letterboxing to the input shape, in px: below
// BOX_FAIL_PX is too small to detect, below BOX_WARN_PX borderline.
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

// Imports

type ImportDetection = {
  box: [number, number, number, number];
  mask: MaskShape | null;
  // Set when the viewer's 30 s memory strip nulls mask, so a null mask is not saved.
  _maskStripped?: boolean;
  embedding: number[];
  // SAM 3's label and confidence.
  gdLabel: string | null;
  gdScore: number | null;
  // Cosine similarity of the crop to predLabel's references.
  embedSimilarityForLabel: number | null;
  predLabel: string | null;
  // In the uncertain zone; flagged in the UI.
  ambiguous: boolean;
  // What the validator or post-pass did.
  vlmAction:
    | "confirm" | "tiebreak" | "disagree" | "relabel"
    | "containment-relabel" | "containment-reject"
    | "containment-duplicate" | "containment-partial"
    | "overlap-multi-mask" | "overlap-low-conf"
    | "three-way-disagree" | "confusion-reject"
    | null;
  // Failed a rejection gate; drawn only while its popup row is hovered.
  rejected: boolean;
  rejectReason:
    | "gd" | "embed" | "combined" | "containment" | "overlap"
    | "disagree" | "confusion"
    | null;
};

type ImportedMedia = {
  id: string;
  // Server import_id; absent until the upload returns.
  backendId?: string;
  file: File;
  // Server filename under imports/.
  filename?: string;
  preview: string;
  // Null until the server has computed it.
  blurhash?: string | null;
  status: "processing" | "ready" | "failed";
  width?: number;
  height?: number;
  // Per-label counts; detections fill in only when the viewer opens the image.
  labelStats?: Record<string, number>;
  // Total detections, available without labelStats.
  detectionCount?: number;
  // Derived crops only: the parent's label, shown while a new one is assigned.
  derivedLabel?: string | null;
  detections?: ImportDetection[];
  // When set, used instead of the detections. Seeded from them on the first edit.
  editedBoxes?: EditableBox[];
  error?: string;
  // Upload time: ms epoch or ISO string.
  createdAt?: number | string | null;
  // Origin URL of an Openverse import.
  sourceUrl?: string | null;
  // Augmentation copies on the server.
  nAugmentations?: number;
  // Checked and empty: counts as labelled, labelling skips it, training uses it as a
  // negative. A drawn box clears it.
  background?: boolean;
  // ms epoch; a cache-buster on the labelled preview URL.
  labelledAt?: number;
  // Sent on every upload attempt so the server dedupes retries.
  idempotencyKey?: string;
  // Seeded from the cache and not yet confirmed by the server this session.
  provisional?: boolean;
};

// A detection as the server sends it.
type WireDetection = {
  box: number[];
  mask: MaskShape | null;
  embedding: number[];
  gd_label: string | null;
  gd_score: number | null;
  pred_label?: string | null;
  embed_sim_for_label?: number | null;
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
};

// Fetches and memoises a hydrated reference's bytes, which start as an empty
// File. Returns the empty File on failure.
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
    // In place: the entry in refs is this same object.
    ref.file = file;
    return file;
  } catch {
    return ref.file;
  }
}

// The same for an import.
async function ensureMediaFile(media: ImportedMedia): Promise<File> {
  if (media.file && media.file.size > 0) return media.file;
  if (!media.preview) {
    console.warn(
      "[ensureMediaFile] no preview URL on media, returning empty stub",
      { id: media.id, filename: media.filename },
    );
    return media.file;
  }
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
    // In place: the entry in imports is this same object.
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
    gdScore: d.gd_score,
    embedSimilarityForLabel: d.embed_sim_for_label ?? null,
    predLabel: d.pred_label ?? d.gd_label ?? null,
    rejected: !!d.rejected,
    rejectReason: d.reject_reason ?? null,
    ambiguous: !!d.ambiguous,
    vlmAction: d.vlm_action ?? null,
  };
}

// Dataset gallery

const DATASET_PAGE_SIZE = 15;
// Cap on the render window's automatic growth while images are processing.
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
  totalImports = null,
  verdicts = {},
  onVerdict,
  verdictFilter = "all",
  onVerdictFilterChange,
  onStartReview,
  filterCountsOverride = null,
  manifestUpdatedAt = null,
  refViewerOpen = false,
  activeSection = "dataset",
}: {
  items: ImportedMedia[];
  /** A change closes the viewers, which are portalled to body. */
  activeSection?: ProjectTab;
  /** The reference editor is open; hides the back-to-top button. */
  refViewerOpen?: boolean;
  /** A change makes the viewer fetch annotations again. */
  manifestUpdatedAt?: string | null;
  /** Server counts over the whole dataset; used instead of counting items when set. */
  filterCountsOverride?: {
    all: number;
    unlabelled: number;
    unrated: number;
    good: number;
    bad: number;
    unsure: number;
  } | null;
  /** Verdict per backendId. */
  verdicts?: Record<string, "good" | "bad" | "unsure">;
  /** The same verdict again clears it. */
  onVerdict?: (id: string, v: "good" | "bad" | "unsure") => void;
  verdictFilter?: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
  onVerdictFilterChange?: (next: "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure") => void;
  /** Opens the ReviewModeV2 overlay on the scope. */
  onStartReview?: (scope: "unrated" | "good" | "bad" | "unsure" | "all") => void;
  /** The server's import count; tiles not yet in items render as skeletons.
      Null falls back to items.length. */
  totalImports?: number | null;
  labels: string[];
  hasReferenceEmbeddings: boolean;
  /** Suppresses the "No reference embeddings" warning. */
  isGeneralDataset?: boolean;
  /** A new derived dataset whose crops are still being made. */
  isPulling?: boolean;
  onRemove: (id: string) => void;
  /** Deletes several imports in one request. */
  onRemoveBatch?: (ids: string[]) => Promise<void> | void;
  /** Patches an import. hydration marks server data, which never replaces editedBoxes. */
  onMediaChange?: (
    id: string,
    patch: Partial<ImportedMedia>,
    opts?: { hydration?: boolean },
  ) => void;
  projectId?: string | null;
  /** The image the label_charlie job is on; its tile shows an overlay. */
  labellingFilename?: string | null;
  /** Called for a label typed in the BoxEditor that the project lacks. */
  onAddProjectLabel?: (label: string) => void;
  /** Display names by lowercased canonical label. */
  labelAliases?: Record<string, string>;
  /** Colour overrides, lowercased canonical label -> "#rrggbb". */
  labelColours?: Record<string, string> | null;
  /** "WxH", for the box size check. */
  inputShape?: string;
  /** Shown at full opacity with a ring while the rest dim. */
  highlightedImportId?: string | null;
  /** "augmentation" also flashes an augmentation badge. */
  highlightKind?: "image" | "augmentation" | null;
}) {
  const [viewIdx, setViewIdx] = useState<number | null>(null);
  // Annotations are fetched on hover, one request per id at a time.
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
        };
        const dets = (body.detections ?? []).map(unwrapWireDetection);
        const patch: Partial<ImportedMedia> = {
          detections: dets as unknown as ImportDetection[],
        };
        if (Array.isArray(body.editedBoxes)) {
          patch.editedBoxes = body.editedBoxes.map(stripTransientBoxFlags);
        }
        onMediaChange(media.id, patch, { hydration: true });
      })
      .catch(() => {})
      .finally(() => {
        hoverPrefetchInFlightRef.current.delete(media.id);
      });
  }, [projectId, onMediaChange]);
  const [augViewer, setAugViewer] = useState<{ importId: string; sourceUrl: string; filename: string } | null>(null);
  useEffect(() => {
    setViewIdx(null);
    setAugViewer(null);
  }, [activeSection]);
  const [displayLimit, setDisplayLimit] = useState(DATASET_PAGE_SIZE);
  // Matches the original upload name or the stored filename.
  const [nameQuery, setNameQuery] = useState("");
  // Filtered before pagination, by the predicates the pill counts use.
  const filteredItems = useMemo(() => {
    const q = nameQuery.trim().toLowerCase();
    const byName = q
      ? items.filter((m) =>
          (m.file?.name || m.filename || "").toLowerCase().includes(q),
        )
      : items;
    if (verdictFilter === "all") return byName;
    return byName.filter((m) => matchesGalleryFilter(m, verdictFilter, verdicts));
  }, [items, verdictFilter, verdicts, nameQuery]);
  const visible = filteredItems.slice(0, displayLimit);
  // Count per pill over all items; a pill at 0 is hidden.
  const filterCounts = useMemo(
    () => filterCountsOverride ?? galleryFilterCounts(items, verdicts),
    [items, verdicts, filterCountsOverride],
  );
  // The server total when known, so skeletons render ahead of the loaded items.
  const effectiveLength = verdictFilter === "all"
    ? (totalImports ?? items.length)
    : filteredItems.length;
  // Infinite-scroll sentinel: in view, it adds a page.
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
      { rootMargin: "600px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [effectiveLength, displayLimit]);

  // Grows the window to show processing tiles, up to DATASET_MAX_AUTO_DISPLAY; never shrinks it.
  useEffect(() => {
    let processingCount = 0;
    for (const m of items) {
      if (m.status === "processing") processingCount++;
    }
    if (processingCount === 0) return;
    const want = Math.min(processingCount + DATASET_PAGE_SIZE, DATASET_MAX_AUTO_DISPLAY);
    setDisplayLimit((cur) => Math.max(cur, want));
  }, [items]);

  // Grows the window to include an import the parent jumps to.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail as { importId?: string; projectId?: string } | null;
      if (!d?.importId) return;
      // A backendId, as in each tile's data-import-id.
      const idx = items.findIndex((m) => m.backendId === d.importId);
      if (idx < 0) return;
      setDisplayLimit((cur) => (idx >= cur ? idx + 1 : cur));
    };
    window.addEventListener("pixelkit-ensure-import-visible", handler);
    return () => window.removeEventListener("pixelkit-ensure-import-visible", handler);
  }, [items]);

  // In select mode a click toggles selection; a bottom bar holds the bulk actions.
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  // Tiles fading out before their removal.
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
      // Filtered, persisted rows only: a bulk delete must not reach hidden images.
      for (const m of filteredItems) if (m.backendId) next.add(m.id);
      return next;
    });
  }, [filteredItems]);
  const selectedCount = selectedIds.size;
  // Selected images without boxes, which can be marked background together.
  const boxlessSelected = useMemo(() => items.filter((m) =>
    selectedIds.has(m.id) && m.backendId && !m.background
    && !(m.editedBoxes !== undefined ? m.editedBoxes.length : (m.detectionCount ?? m.detections?.length ?? 0)),
  ), [items, selectedIds]);
  const markSelectedBackground = useCallback(() => {
    if (!onMediaChange) return;
    for (const m of boxlessSelected) onMediaChange(m.id, { editedBoxes: [], background: true });
    exitSelection();
  }, [boxlessSelected, onMediaChange, exitSelection]);
  const deleteSelected = useCallback(async () => {
    if (selectedCount === 0 || !onRemoveBatch) return;
    setBulkBusy(true);
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
  const showReview = filterCounts.all > 0;

  return (
    <>
      <div className="flex items-center justify-between gap-3 mb-1 flex-wrap">
        <h2 className="pk-micro shrink-0 text-[var(--fg-soft)]">
          Dataset
          <span className="ml-2 tabular-nums text-[var(--fg-dim)]">
            {/* "shown / total" while filtered. */}
            {verdictFilter === "all" && !nameQuery.trim()
              ? (totalImports ?? items.length)
              : `${filteredItems.length} / ${totalImports ?? items.length}`}
          </span>
        </h2>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Client-side, over the loaded imports only. */}
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
                className="w-48 rounded-md border border-[var(--line)] bg-foreground/[0.025] pl-8 pr-7 py-1.5 text-[12px] text-[var(--foreground)] outline-none transition-colors focus:border-[var(--line-strong)] placeholder:text-foreground/40"
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

          {visiblePills.length > 1 && (
            <div className="flex items-center gap-2 rounded-md border border-[var(--line)] bg-foreground/[0.025] pl-3 pr-1.5 py-1">
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
                        "rounded px-3 py-1 text-[12px] font-medium border transition-colors",
                        active
                          ? "bg-[var(--surface-hover)] text-[var(--foreground)] border-[var(--line-strong)]"
                          : "border-transparent text-foreground/60 hover:bg-foreground/[0.06] hover:text-foreground",
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

          {/* With the "all" pill active, review starts on unrated. */}
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

          {!hasReferenceEmbeddings && !isGeneralDataset && (
            <span className="text-[11px] text-[var(--warn)] font-mono uppercase tracking-wider">
              No reference embeddings
            </span>
          )}
          {onRemoveBatch && items.some((m) => m.backendId) && (
            selectionMode ? (
              <button
                type="button"
                onClick={exitSelection}
                className="rounded-md border border-[var(--line)] px-3.5 py-1.5 text-[13px] font-medium text-foreground/70 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-foreground transition-colors"
              >
                Done
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setSelectionMode(true)}
                className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-3.5 py-1.5 text-[13px] font-medium text-foreground/70 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] hover:text-foreground transition-colors"
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
        // The highlighted thumb sets its own opacity back to 1.
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
              onHover={() => prefetchAnnotation(m)}
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
            />
          );
        })}
        {/* Skeletons for tiles not loaded yet; none under a filter. */}
        {verdictFilter === "all" && totalImports != null && visible.length < Math.min(displayLimit, totalImports) && (
          Array.from({
            length: Math.min(displayLimit, totalImports) - visible.length,
          }).map((_, i) => (
            <div
              key={`ds-skel-${visible.length + i}`}
              aria-hidden
              className="relative rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden"
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
        <div className="mt-6 flex flex-col items-center justify-center gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-6 py-16 text-center">
          <span className="h-7 w-7 animate-spin rounded-full border-2 border-foreground/15 border-t-[var(--accent)]" aria-hidden />
          <div className="text-[13px] font-medium text-foreground/70">Pulling cropped images from the parent dataset…</div>
          <div className="text-[12px] text-foreground/50">This can take a moment for large datasets; images appear as they are cropped.</div>
        </div>
      )}

      {effectiveLength > displayLimit && (
        <div
          ref={loadMoreRef}
          className="mt-6 flex justify-center text-[11px] uppercase tracking-wider font-mono text-foreground/35"
        >
          Loading more…
        </div>
      )}

      {/* Portalled: a transformed ancestor would be the containing block for fixed elements. */}
      {viewIdx !== null && filteredItems[viewIdx] && typeof window !== "undefined" &&
        createPortal(
          // viewIdx indexes filteredItems, so prev and next stay within the filter.
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
            verdicts={verdicts}
            onVerdict={onVerdict}
            onDelete={onRemove}
            manifestUpdatedAt={manifestUpdatedAt}
            inputShape={inputShape}
          />,
          document.body,
        )
      }

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

      {/* Portalled out of the animated section. */}
      {selectionMode && typeof window !== "undefined" &&
        createPortal(
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[400] flex items-center gap-2 rounded-md border border-white/20 bg-black/85 backdrop-blur-md px-3 py-2 shadow-[var(--shadow-strong)]">
            <span className="text-xs text-white/80 px-2 tabular-nums">
              {selectedCount} selected
            </span>
            <button
              type="button"
              onClick={selectAllVisible}
              disabled={bulkBusy}
              className="rounded-md border border-white/20 bg-white/[0.06] px-3 py-1 text-[11px] text-white/85 hover:border-white/45 hover:text-white transition-colors disabled:opacity-40"
            >
              Select all
            </button>
            {onMediaChange && boxlessSelected.length > 0 && (
              <button
                type="button"
                onClick={markSelectedBackground}
                disabled={bulkBusy}
                title="Nothing to label in these: train on them as background (negative) examples"
                className="rounded-md border border-white/20 bg-white/[0.06] px-3 py-1 text-[11px] text-white/85 hover:border-white/45 hover:text-white transition-colors disabled:opacity-40"
              >
                Mark {boxlessSelected.length} background
              </button>
            )}
            <button
              type="button"
              onClick={() => void deleteSelected()}
              disabled={bulkBusy || selectedCount === 0}
              className="rounded-md bg-red-500 text-white px-4 py-1 text-xs font-semibold hover:bg-red-400 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {bulkBusy ? "Deleting…" : `Delete ${selectedCount}`}
            </button>
            <button
              type="button"
              onClick={exitSelection}
              disabled={bulkBusy}
              className="rounded-md border border-white/20 px-3 py-1 text-[11px] text-white/65 hover:border-white/45 hover:text-white transition-colors disabled:opacity-40"
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

// FIFO queue of image loads, at most concurrency at a time. Tiles join only near the viewport.
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

// 6 at a time: the server bakes each labelled preview once, under a per-image lock.
const acquireDatasetImgSlot = makeImageLoadQueue(6);
const acquireRefImgSlot = makeImageLoadQueue(6);

// Reference tile image: queued lazy load over a BlurHash placeholder.
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
    const node = tileRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    // Not sticky: leaving the 1500 px margin lets the unload effect drop the image.
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

  // Unloads an off-screen image.
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
      {/* BlurHash, else a shimmer; fades out when the image loads. */}
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
            background: "linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 50%, var(--surface) 100%)",
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

// #rrggbb or #rgb -> rgba(); an unparseable input is returned unchanged.
function hexToRgba(hex: string, alpha: number): string {
  let h = (hex || "").trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  if (h.length !== 6 || /[^0-9a-fA-F]/.test(h)) return hex;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Review launcher. The main button reviews the active pill's bucket, else unrated;
// the dropdown picks another.

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
        className="pk-btn rounded-r-none capitalize"
        title={defaultCount === 0
          ? "No images in this bucket"
          : `Swipe left for bad, right for good. Reviews the ${defaultScope} images.`}
      >
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M15 18l-6-6 6-6" opacity="0.45" />
          <path d="M9 18l6-6-6-6" />
        </svg>
        Review {defaultScope}
        <span className="tabular-nums text-[var(--fg-dim)]">{defaultCount}</span>
      </button>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="pk-btn rounded-l-none border-l-0 px-2"
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
          className="absolute right-0 top-full mt-1.5 min-w-[10rem] rounded-md border border-[var(--line)] bg-[var(--surface)] shadow-[var(--shadow-strong)] backdrop-blur-md overflow-hidden z-20"
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

// Augmentations viewer: one import's copies, from /api/v2/projects/{id}/augmentations/{import_id}.
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
  /** "WxH", for the box size colouring. */
  inputShape: string;
  onClose: () => void;
}) {
  const [items, setItems] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The server sends augmentation JPEGs with max-age=86400; a new token refetches them.
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
  // Copies that are listed but failed to load, as after an interrupted generation.
  const [missing, setMissing] = useState<Set<string>>(new Set());
  useEffect(() => { setMissing(new Set()); }, [importId, cacheBuster]);
  const [resuming, setResuming] = useState(false);
  // Reruns generation with the saved config, as the Augmentations tab's Update does.
  const resumeGeneration = async () => {
    if (resuming) return;
    setResuming(true);
    try {
      const cr = await apiFetch(`/api/v2/projects/${projectId}/augment/config`);
      if (!cr.ok) throw new Error(`http ${cr.status}`);
      const cfg = await cr.json() as { perImageMode?: string; config?: Record<string, unknown> };
      if (!cfg?.config || (cfg.perImageMode ?? "off") === "off") {
        throw new Error("no saved augmentation config");
      }
      const r = await apiFetch(`/api/v2/projects/${projectId}/augment/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ perImageMode: cfg.perImageMode, config: cfg.config }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      try {
        window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", { detail: { projectId } }));
      } catch { /* ignore */ }
    } catch (e) {
      console.warn("[augmentations] resume failed:", e);
    } finally {
      setResuming(false);
    }
  };

  // Annotations by copy filename, already warped into the augmented image's coordinates.
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

  // Optimistic; the event refreshes the gallery's augmentation icon.
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
      try {
        window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
          detail: { projectId },
        }));
      } catch { /* ignore */ }
    } catch {
      // Best effort; the tile stays removed.
    }
  };

  if (typeof window === "undefined") return null;

  return createPortal(
    // Fills the content area; closed by Back or a section change, not an outside click.
    <div className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[700] overflow-auto bg-[var(--background)]">
      <div className="px-6 lg:px-10 py-6 relative">
        <div className="mb-6 min-w-0">
          <button
            type="button"
            onClick={onClose}
            className="pk-btn mb-4"
          >
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M19 12H5" /><path d="m12 19-7-7 7-7" />
            </svg>
            Back
          </button>
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="pk-micro">Augmentations</div>
              <h2 className="text-[16px] font-medium tracking-tight text-[var(--foreground)] truncate">{filename}</h2>
            </div>
            {missing.size > 0 && (
              <button
                type="button"
                onClick={() => void resumeGeneration()}
                disabled={resuming}
                className="pk-btn shrink-0"
                title="Some copies were never written (generation was interrupted). Re-run generation with the saved settings."
              >
                {resuming ? "Resuming…" : "Resume generation"}
              </button>
            )}
          </div>
        </div>

        <div className="flex justify-center mb-8">
          <div className="rounded-lg overflow-hidden border border-[var(--line)] bg-[var(--surface-2)] max-w-sm w-full">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={sourceUrl} alt={filename} className="block w-full h-auto" draggable={false} />
            <div className="px-4 py-2 pk-micro text-center">
              Original
            </div>
          </div>
        </div>

        <div className="pk-micro tabular-nums mb-3">
          {items === null ? "Loading…" : `${items.length} ${items.length === 1 ? "augmentation" : "augmentations"}`}
        </div>

        {error && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)] mb-4">
            {error}
          </div>
        )}

        {items !== null && items.length === 0 && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-5 py-12 text-center">
            <div className="text-sm text-foreground/65 mb-1">No augmentations generated yet</div>
            <div className="text-[11px] text-foreground/35">
              Set a per-image count and turn on some augmentations, then click Update on the Augmentations tab.
            </div>
          </div>
        )}

        {items !== null && items.length > 0 && (
          <div className="grid gap-4 md:grid-cols-2">
            {items.map((name) => (
              <AugmentationTile
                key={name}
                src={`${API_BASE}/api/v2/projects/${projectId}/augmentations/${importId}/${encodeURIComponent(name)}?v=${cacheBuster}`}
                name={name}
                annotations={augAnnos?.copies[name] ?? []}
                canvasW={augAnnos?.width ?? 0}
                canvasH={augAnnos?.height ?? 0}
                inputShape={inputShape}
                onDelete={() => removeAug(name)}
                onLoadError={() => setMissing((cur) => {
                  if (cur.has(name)) return cur;
                  const next = new Set(cur);
                  next.add(name);
                  return next;
                })}
              />
            ))}
          </div>
        )}

        <div className="pb-8" />
      </div>
    </div>,
    document.body,
  );
}

// Augmentation tile; hovering draws its annotations over the image.
function AugmentationTile({
  src,
  name,
  annotations,
  canvasW,
  canvasH,
  inputShape,
  onDelete,
  onLoadError,
}: {
  src: string;
  name: string;
  /** Warped polygons and boxes in the augmented image's pixels, at canvasW x canvasH. */
  annotations: { polys: number[][][]; box: number[]; label: string }[];
  canvasW: number;
  canvasH: number;
  /** "WxH". A box's colour shows how its smallest side survives the downscale to it. */
  inputShape: string;
  /** Shows a delete button when set. */
  onDelete?: () => void;
  /** Called when the JPEG fails to load; the tile shows "Not generated". */
  onLoadError?: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [failed, setFailed] = useState(false);
  const hasOverlay = !failed && canvasW > 0 && canvasH > 0 && annotations.length > 0;
  return (
    <div
      className="rounded-lg overflow-hidden border border-[var(--line)] bg-[var(--surface-2)] relative group"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Inside the border, so the SVG overlay covers exactly the image. */}
      <div className="relative">
      {failed ? (
        <div className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-1.5 bg-[var(--panel)]">
          <span className="pk-micro">Not generated</span>
          <span className="text-[11px] text-foreground/35">Generation was interrupted before this copy was written.</span>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={name}
          className="block w-full h-auto"
          loading="lazy"
          decoding="async"
          draggable={false}
          onError={() => { setFailed(true); onLoadError?.(); }}
        />
      )}
      {!failed && onDelete && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          aria-label="Delete this augmentation"
          title="Delete this augmentation"
          className="absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-md bg-black/65 backdrop-blur-md text-white/90 hover:bg-red-500/85 hover:text-white transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
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
            // Green, orange or red by sizeStatusFor, as in the BoxEditor.
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
            // Chip sizes are in viewBox units, scaled to the image.
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
}: {
  media: ImportedMedia;
  labels: string[];
  projectId?: string | null;
  onOpen: () => void;
  onHover?: () => void;
  onRemove: () => void;
  /** The label_charlie job is on this image. */
  isLabelling?: boolean;
  /** Display names by lowercased canonical label. */
  labelAliases?: Record<string, string>;
  /** Colour overrides, lowercased canonical label -> "#rrggbb". */
  labelColours?: Record<string, string> | null;
  /** Clicks toggle selection through onToggleSelected. */
  selectionMode?: boolean;
  selected?: boolean;
  onToggleSelected?: () => void;
  /** Fading out before a bulk delete removes it. */
  deleting?: boolean;
  /** Opens the gallery's augmentations viewer. */
  setAugViewer?: (next: { importId: string; sourceUrl: string; filename: string } | null) => void;
  /** Target of a jump: full opacity, a ring and a scale-up. */
  isHighlighted?: boolean;
  /** Another tile is highlighted. */
  isDimmed?: boolean;
  /** Flashes an "augmentation" badge on a highlighted tile. */
  augmentationFlash?: boolean;
}) {
  const displayLabel = (canonical: string): string => {
    const k = canonical.trim().toLowerCase();
    return labelAliases[k] || canonical;
  };
  const labelStatsSrc = media.labelStats;
  const detectionCountSrc = media.detectionCount;
  const labelledAtSrc = media.labelledAt;
  const acceptedDetections = useMemo(
    () => (media.detections ?? []).filter((d) => !d.rejected),
    [media.detections],
  );
  const rejectedCount = (media.detections?.length ?? 0) - acceptedDetections.length;
  // Edited boxes win, as in the BoxEditor.
  const editedLabels = useMemo<string[] | null>(() => {
    if (media.editedBoxes === undefined) return null;
    return media.editedBoxes.map((b) => b.label).filter(Boolean);
  }, [media.editedBoxes]);
  // Then labelStats while no detections are loaded, then the detections.
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
      for (const lab of Object.keys(labelStatsSrc)) {
        if (lab) set.add(lab);
      }
    } else {
      acceptedDetections.forEach((d) => { if (d.predLabel) set.add(d.predLabel); });
    }
    return Array.from(set);
  }, [acceptedDetections, editedLabels, labelStatsSrc]);
  // Any accepted detection is ambiguous; shows the corner pill.
  const hasUnsureDetection = useMemo(
    () => acceptedDetections.some((d) => d.ambiguous),
    [acceptedDetections],
  );

  // The server's labelled preview, else the local preview. v and retry change the
  // URL to get past the browser cache; retries stop at 3.
  const [previewRetry, setPreviewRetry] = useState(0);
  const previewUrl = media.backendId && projectId
    ? `${API_BASE}/api/v2/projects/${projectId}/imports/${media.backendId}/labelled_preview${
        labelledAtSrc || previewRetry
          ? `?v=${labelledAtSrc ?? 0}${previewRetry ? `&retry=${previewRetry}` : ""}`
          : ""
      }`
    : media.preview;

  // Not sticky: leaving the 1500 px margin lets the unload effect drop the image.
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

  // Drops the image after a short time off-screen.
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

  // A slot that opens after cleanup is released at once.
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

  useEffect(() => () => {
    if (releaseSlotRef.current) {
      releaseSlotRef.current();
      releaseSlotRef.current = null;
    }
  }, []);

  useEffect(() => {
    setLoaded(false);
    setCanLoad(false);
    if (releaseSlotRef.current) {
      releaseSlotRef.current();
      releaseSlotRef.current = null;
    }
  }, [previewUrl]);

  // A cached image can load before React attaches onLoad: checked now and on the next frame.
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
      // jumpToImport finds the tile by this, with a backendId.
      data-import-id={media.backendId ?? media.id}
      onPointerEnter={onHover}
      className={[
        "relative rounded-md border bg-[var(--panel)] overflow-hidden transition-all",
        selected
          ? "border-red-400/70 ring-2 ring-red-400/40"
          : isHighlighted
          ? "border-foreground/45 ring-2 ring-foreground/25"
          : "border-foreground/10 hover:border-foreground/20",
      ].join(" ")}
      style={{
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
        <div ref={tileRef} className="relative aspect-[4/3] bg-[var(--background)] overflow-hidden">
          {augmentationFlash && (
            <div
              className="absolute inset-0 z-20 grid place-items-center pointer-events-none"
              style={{ animation: "augFlashIn 220ms ease-out both" }}
            >
              <div
                className="rounded-md bg-black/65 text-white px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.18em] backdrop-blur-md"
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
          {/* BlurHash, else a static fill. */}
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
              // will-change only during the fade: a GPU layer per loaded tile is costly.
              className={[
                "absolute inset-0 w-full h-full object-cover",
                loaded ? "" : "will-change-transform will-change-[opacity,filter]",
              ].join(" ")}
              style={{
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
                // The preview render can fail transiently: up to 3 retries, backing off from 350 ms.
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
          {media.status === "ready" && (
            // Stops short of the remove button at the top right.
            <div className="absolute top-2 left-2 right-11 flex min-w-0 items-center gap-1.5">
              {media.backendId && projectId && setAugViewer && (media.nAugmentations ?? 0) > 0 && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setAugViewer({ importId: media.backendId!, sourceUrl: media.preview, filename: media.filename ?? media.file.name }); }}
                  className="h-7 w-7 shrink-0 rounded-md bg-black/55 backdrop-blur-md grid place-items-center text-white/90 hover:bg-black/80 hover:text-white transition-colors"
                  title={`${media.nAugmentations} augmentation${media.nAugmentations === 1 ? "" : "s"}`}
                  aria-label="Augmentations"
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
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
              {/* Waits for detections, so a tile does not flash Unlabelled while loading. */}
              {detectionCount === 0 && media.detections !== undefined && (
                <span
                  className={`min-w-0 truncate rounded-md bg-black/60 px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider backdrop-blur-sm ${media.background ? "text-white/80" : "text-amber-200"}`}
                  title={media.background ? "Nothing to label here: trained on as a background example" : undefined}
                >
                  {media.background ? "Background" : "Unlabelled"}
                </span>
              )}
              {detectionCount > 0 && hasUnsureDetection && (
                <span
                  className="min-w-0 truncate rounded-md bg-black/60 px-2 py-0.5 font-mono text-[10px] font-medium text-amber-200 uppercase tracking-wider backdrop-blur-sm"
                  title="At least one detection is borderline. Open the image to review it."
                >
                  Unsure
                </span>
              )}
            </div>
          )}
        </div>

        <div className="px-3 py-2.5 border-t border-foreground/[0.06]">
          <div className="flex items-center justify-between gap-2 text-[12px]">
            <span className="truncate text-foreground/85">{media.file.name}</span>
            <span className="font-mono tabular-nums text-foreground/50 shrink-0">
              {detectionCount} {detectionCount === 1 ? "box" : "boxes"}
              {rejectedCount > 0 && (
                <span className="ml-1.5 text-[var(--warn)] opacity-80" title={`${rejectedCount} rejected by embedding QC`}>
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
                    className="inline-flex items-center gap-1 rounded-md border border-[var(--line)] px-1.5 py-0.5 font-mono text-[9px] text-[var(--fg-soft)]"
                  >
                    <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: bg }} aria-hidden />
                    {displayLabel(lab)}
                  </span>
                );
              })}
            </div>
          )}
          {media.derivedLabel && (
            <div
              className="mt-1.5 flex items-center gap-1 text-[10px] text-foreground/40"
              title={`Original label in the parent dataset: ${media.derivedLabel}`}
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3 opacity-70 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="9 10 4 15 9 20" /><path d="M20 4v7a4 4 0 0 1-4 4H5" />
              </svg>
              <span className="truncate">from {displayLabel(media.derivedLabel)}</span>
            </div>
          )}
        </div>
      </button>

      {!selectionMode && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          aria-label="Remove image"
          className="absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-md bg-black/55 text-white/80 hover:bg-black/80 hover:text-white transition-colors z-10"
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
          </svg>
        </button>
      )}

      {/* Disabled for uploads in flight, which have no backendId. */}
      {selectionMode && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onToggleSelected?.(); }}
          disabled={!media.backendId}
          aria-pressed={selected}
          aria-label={selected ? "Unselect image" : "Select image"}
          className={[
            "absolute top-2 right-2 h-7 w-7 grid place-items-center rounded-md transition-colors z-10",
            selected
              ? "bg-red-500 text-white"
              : "bg-black/55 text-white/70 hover:bg-black/80 hover:text-white",
            !media.backendId ? "opacity-30 cursor-not-allowed" : "",
          ].join(" ")}
        >
          {selected ? (
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="5 12 10 17 19 7" />
            </svg>
          ) : (
            <span className="block h-3.5 w-3.5 rounded-full border-2 border-white/55" aria-hidden="true" />
          )}
        </button>
      )}
    </div>
  );
}

// ImportDetection -> EditableBox. Ambiguous detections get an "unsure" validation.
function detectionToEditableBox(
  d: ImportDetection,
  viewIdx: number,
  detIdx: number,
): EditableBox {
  // Rejected detections are filtered out before this; confident ones get no validation.
  let validation: Validation | null = null;
  if (d.ambiguous || d.vlmAction === "tiebreak" || d.vlmAction === "disagree") {
    validation = {
      match: true,
      confidence: d.embedSimilarityForLabel ?? 0.5,
      reason: "PixelKit flagged this detection as ambiguous. Confirm or delete it.",
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

// Text for a detection's rejectReason.
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

// Image viewer: the BoxEditor with its labels panel.
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
  /** hydration marks server data, which never replaces a local editedBoxes. */
  onMediaChange?: (
    id: string,
    patch: Partial<ImportedMedia>,
    opts?: { hydration?: boolean },
  ) => void;
  projectId?: string | null;
  /** Called for a label the project lacks. */
  onAddProjectLabel?: (label: string) => void;
  /** Display names by lowercased canonical label. */
  labelAliases?: Record<string, string>;
  /** Per-label colour overrides for canvas tints + sidebar chips. */
  labelColours?: Record<string, string> | null;
  /** Verdict per backendId. */
  verdicts?: Record<string, "good" | "bad" | "unsure">;
  /** The same setter as Review mode. */
  onVerdict?: (id: string, v: "good" | "bad" | "unsure") => void;
  /** Deletes the image; the viewer shows the next, the previous after the last, or closes. */
  onDelete?: (id: string) => void;
  /** A change lets loaded images take fresh annotations. */
  manifestUpdatedAt?: string | null;
  /** "WxH" for the size colouring; must equal the shape the export size filter uses. */
  inputShape?: string;
}) {
  const media = items[index];
  // Hovered sidebar row; its box is emphasised in the BoxEditor.
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  // Box under the mouse, from BoxEditor's onHoverChange; the relabel hotkeys act on it.
  const [hoveredCanvasBoxId, setHoveredCanvasBoxId] = useState<string | null>(null);
  useEffect(() => { setHoveredIdx(null); setHoveredCanvasBoxId(null); }, [index]);

  // Annotations load for the open image and its neighbours unless they have masks, which
  // only real detections carry. Masks are stripped 30 s after the viewer leaves an image.
  const annotationFetchKickedRef = useRef<Set<string>>(new Set());
  const annotationClearTimersRef = useRef<Map<string, number>>(new Map());
  const prevViewerIndexRef = useRef<number | null>(null);
  // Ids with a fetch in flight, for the loading spinner.
  const [pendingAnnotationIds, setPendingAnnotationIds] = useState<Set<string>>(
    () => new Set(),
  );
  const annotationsHydratedRef = useRef<Set<string>>(new Set());
  // A new manifest version lets hydrated ids take fresh data.
  useEffect(() => { annotationsHydratedRef.current.clear(); }, [manifestUpdatedAt]);

  const applyAnnotationPayload = useCallback((
    m: ImportedMedia,
    body: {
      detections?: WireDetection[];
      editedBoxes?: EditableBox[] | null;
    },
  ) => {
    if (!onMediaChange) return;
    // Once per id and manifest version: re-applying a mask-less row loops renders.
    if (annotationsHydratedRef.current.has(m.id)) return;
    const dets = (body.detections ?? []).map(unwrapWireDetection);
    const patch: Partial<ImportedMedia> = {
      detections: dets,
    };
    if (Array.isArray(body.editedBoxes)) {
      patch.editedBoxes = body.editedBoxes.map(stripTransientBoxFlags);
    }
    onMediaChange(m.id, patch, { hydration: true });
    annotationsHydratedRef.current.add(m.id);
  }, [onMediaChange]);

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

  // No-op when the image has masks or a fetch for it is in flight.
  const fetchPerImageAnnotation = useCallback(async (m: ImportedMedia | undefined) => {
    if (!m || !projectId || !m.backendId || !onMediaChange) return;
    if (annotationFetchKickedRef.current.has(m.id)) return;
    const existing = m.detections ?? [];
    if (existing.length > 0 && existing.some((d) => d?.mask)) return;
    annotationFetchKickedRef.current.add(m.id);
    setPendingForIds([m.id], true);
    try {
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/annotations/${encodeURIComponent(m.backendId!)}`,
      );
      if (!r.ok) return;
      const body = (await r.json()) as {
        detections?: WireDetection[];
        editedBoxes?: EditableBox[] | null;
      };
      applyAnnotationPayload(m, body);
    } catch {
      /* keep the lighter bulk detections on error */
    } finally {
      annotationFetchKickedRef.current.delete(m.id);
      setPendingForIds([m.id], false);
    }
  }, [projectId, onMediaChange, applyAnnotationPayload, setPendingForIds]);

  useEffect(() => {
    if (!media || !projectId || !media.backendId || !onMediaChange) return;
    // A revisit within 30 s cancels the pending mask strip.
    const currentClear = annotationClearTimersRef.current.get(media.id);
    if (currentClear) {
      window.clearTimeout(currentClear);
      annotationClearTimersRef.current.delete(media.id);
    }

    void fetchPerImageAnnotation(media);
    for (const offset of [1, -1, 2, -2]) {
      void fetchPerImageAnnotation(items[index + offset]);
    }

    // Loads the neighbours' image bytes into the HTTP cache.
    for (const offset of [1, -1, 2, -2]) {
      const neigh = items[index + offset];
      if (neigh?.preview && !neigh.preview.startsWith("blob:")) {
        const im = new window.Image();
        im.decoding = "async";
        im.src = neigh.preview;
      }
    }

    // The image being left has its masks stripped after 30 s.
    const prevIdx = prevViewerIndexRef.current;
    if (prevIdx !== null && prevIdx !== index) {
      const prev = items[prevIdx];
      if (prev && prev.id !== media.id) {
        const prevId = prev.id;
        const t0 = annotationClearTimersRef.current.get(prevId);
        if (t0) window.clearTimeout(t0);
        const tid = window.setTimeout(() => {
          // Boxes and labels stay; without masks, a reopen refetches.
          if (!onMediaChange) return;
          onMediaChange(prevId, {
            detections: ((prev.detections ?? []).map((d) => ({
              ...d,
              mask: null,
              _maskStripped: true,
            })) as ImportDetection[]),
          });
          annotationClearTimersRef.current.delete(prevId);
          annotationsHydratedRef.current.delete(prevId);
        }, 30_000);
        annotationClearTimersRef.current.set(prevId, tid);
      }
    }
    prevViewerIndexRef.current = index;
  }, [media, projectId, onMediaChange, items, index, fetchPerImageAnnotation, manifestUpdatedAt]);

  useEffect(() => () => {
    annotationClearTimersRef.current.forEach((tid) => window.clearTimeout(tid));
    annotationClearTimersRef.current.clear();
  }, []);

  const hasPrev = index > 0;
  const hasNext = index < items.length - 1;

  const updateMedia = (patch: Partial<ImportedMedia>) => {
    if (!media || !onMediaChange) return;
    onMediaChange(media.id, patch);
  };

  // A relabel hotkey sets the hovered box's label. The handler reads state through
  // this ref, so the listener is not rebound every render.
  const relabelStateRef = useRef({ hoveredCanvasBoxId, labels, media, onMediaChange });
  useEffect(() => {
    relabelStateRef.current = { hoveredCanvasBoxId, labels, media, onMediaChange };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      // Modified keys belong to other shortcuts.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const slot = slotForRelabelKey(e.key);
      if (slot === null) return;
      const { hoveredCanvasBoxId: hovId, labels: lbls, media: m, onMediaChange: onMC } = relabelStateRef.current;
      if (hovId === null || slot >= lbls.length || !m || !onMC) return;
      const newLabel = lbls[slot];
      e.preventDefault();
      // Mask-stripped detections must not become editedBoxes: their null masks would be saved.
      if (m.editedBoxes === undefined && (m.detections ?? []).some((d) => d._maskStripped)) {
        return;
      }
      // Ids keep the index before filtering, imp_<index>_<originalIdx>, as in editorBoxes.
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
  }, [index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;  // the editor or painter used this key
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && hasPrev) onPrev();
      else if (e.key === "ArrowRight" && hasNext) onNext();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasPrev, hasNext, onClose, onPrev, onNext]);

  // editedBoxes wins even when empty; the server sends it only once edited. Otherwise the
  // non-rejected detections, ids keeping the index before filtering, which the popup rows use.
  const editorBoxes = useMemo<EditableBox[]>(() => {
    const baseBoxes: EditableBox[] = media?.editedBoxes !== undefined
      ? media.editedBoxes
      : (media?.detections ?? [])
          .map((d, originalIdx) => ({ d, originalIdx }))
          .filter(({ d }) => !d.rejected)
          .map(({ d, originalIdx }) => detectionToEditableBox(d, index, originalIdx));

    // A hovered rejected row in the popup adds its box back, marked rejected.
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

  // Decoded from the image when the server has not supplied the size.
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    if (!media) return;
    if (media.width && media.height) { setDims({ w: media.width, h: media.height }); return; }
    const img = new window.Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = media.preview;
    return () => { img.onload = null; };
  }, [media]);

  // Preloads images within 3 of the current one; others leave the map.
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
  useEffect(() => () => { prefetchRef.current.clear(); }, []);

  // Colours each box by its size after letterboxing to inputShape.
  const [sizeColoringOn, setSizeColoringOn] = useState(true);
  // Drawn boxes skip SAM3 and open the label picker.
  const [manualMode, setManualMode] = useState(false);

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

  if (!media) return null;

  return (
    // The content area only: below the 36px title bar, above the 24px status bar.
    <div
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[400] flex flex-col bg-[var(--background)]"
      role="dialog"
      aria-modal="true"
    >
      {/* The groups wrap onto a second row when the content area is narrow. */}
      <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-8 py-4 border-b border-foreground/[0.06] shrink-0">
        <div className="flex items-baseline gap-4 min-w-0 max-w-full">
          <span className="text-base font-light tracking-tight text-[var(--foreground)] truncate">
            {media.file.name}
          </span>
          {dims && (
            <span className="text-xs text-foreground/45 tabular-nums shrink-0">
              {dims.w} × {dims.h}
            </span>
          )}
          <span className="text-xs text-foreground/45 shrink-0">
            {/* editorBoxes, so a deletion shows at once. */}
            {editorBoxes.length} detection{editorBoxes.length === 1 ? "" : "s"}
          </span>
        </div>
        {/* Relabel hotkey legend; labels past RELABEL_KEY_LIMIT have no key. */}
        {labels.length > 0 && (
          <div className="hidden xl:flex items-center gap-1.5 shrink-0 flex-wrap justify-end max-w-[40%]">
            <span className="text-[10px] text-foreground/35 mr-1">
              hover + press
            </span>
            {labels.map((lab, i) => {
              const key = keyForRelabelSlot(i);
              return (
                <span
                  key={lab}
                  className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-foreground/[0.03] px-2 py-0.5 text-[11px] tabular-nums"
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
          {onVerdict && media.backendId && (
            <VerdictPills
              current={verdicts[media.backendId] ?? null}
              onToggle={(v) => onVerdict(media.backendId!, v)}
            />
          )}
          {media.derivedLabel && (
            <span
              className="inline-flex items-center gap-1.5 rounded-md bg-foreground/[0.06] px-2.5 h-7 text-[11px] font-medium text-foreground/60"
              title={`Original label in the parent dataset: "${media.derivedLabel}". Assign your own label here.`}
            >
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 opacity-60" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="9 10 4 15 9 20" /><path d="M20 4v7a4 4 0 0 1-4 4H5" />
              </svg>
              <span className="text-foreground/45">from</span>
              <span className="text-foreground/85">{media.derivedLabel}</span>
            </span>
          )}
          <span className="hidden 2xl:inline text-[11px] text-foreground/30">
            ← / → · Esc
          </span>
          <span className="text-xs text-foreground/45 tabular-nums">{index + 1} / {items.length}</span>
          {/* Clear all sets editedBoxes to []; an image without boxes offers Background instead. */}
          {onMediaChange && (() => {
            const liveBoxCount = media.editedBoxes !== undefined
              ? (media.editedBoxes?.length ?? 0)
              : ((media.detections?.filter((d) => !d.rejected).length) ?? 0);
            if (liveBoxCount === 0) return (
              <button
                type="button"
                onClick={() => onMediaChange(media.id, { editedBoxes: [], background: !media.background })}
                aria-pressed={!!media.background}
                title={media.background
                  ? "Marked background: nothing here to label. Click to unmark."
                  : "Nothing here to label: train on it as a background (negative) example"}
                className={[
                  "h-8 inline-flex items-center gap-1.5 rounded-md border px-3 text-[11px] uppercase tracking-wider font-mono transition-colors",
                  media.background
                    ? "border-accent/60 bg-accent/10 text-[var(--accent)]"
                    : "border-[var(--line)] text-foreground/65 hover:border-[var(--line-strong)] hover:text-foreground",
                ].join(" ")}
              >
                Background
              </button>
            );
            return (
              <button
                type="button"
                onClick={() => onMediaChange(media.id, { editedBoxes: [] })}
                title="Clear every box on this image"
                className="h-8 inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] text-foreground/65 hover:border-[var(--bad)] hover:text-[var(--bad)] px-3 text-[11px] uppercase tracking-wider font-mono transition-colors"
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
          <button
            type="button"
            onClick={() => setManualMode((v) => !v)}
            aria-pressed={manualMode}
            title={manualMode
              ? "Manual mode is on: clicks and drawn boxes skip SAM 3"
              : "Switch to manual labelling (no SAM 3 auto-detect)"
            }
            className={[
              "h-8 inline-flex items-center gap-1.5 rounded-md border px-3 text-[11px] uppercase tracking-wider font-mono transition-colors",
              manualMode
                ? "border-accent/60 bg-accent/10 text-[var(--accent)]"
                : "border-[var(--line)] text-foreground/65 hover:border-[var(--line-strong)] hover:text-foreground",
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
          {onDelete && media && (
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
              className="h-8 inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] text-foreground/65 hover:border-[var(--bad)] hover:text-[var(--bad)] px-3 text-[11px] uppercase tracking-wider font-mono transition-colors"
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
            className="h-8 w-8 grid place-items-center rounded-md border border-[var(--line)] text-foreground/55 hover:border-[var(--line-strong)] hover:text-foreground transition-colors"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" /><path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      </header>

      <div className="relative flex-1 min-h-0 p-5 flex flex-col gap-3">
        {/* Anchors the popup to the canvas, clear of the nav bar. */}
        <div className="relative flex-1 min-h-0">
        <div
          className="h-full overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--surface-2)]"
        >
          {dims ? (
            <BoxEditor
              key={`${media.id}_${index}`}
              imageUrl={media.preview}
              // Above 2560 px a 2560 px variant paints first; BoxEditor then loads the full image.
              previewUrl={(dims.w > 2560 || dims.h > 2560) ? `${media.preview}?w=2560` : null}
              imageWidth={dims.w}
              imageHeight={dims.h}
              blurhash={media.blurhash ?? null}
              boxes={editorBoxes}
              onChange={(next) => updateMedia({ editedBoxes: next })}
              projectTags={labels}
              labelColours={labelColours}
              displayLabel={(s) => labelAliases[s.trim().toLowerCase()] || s}
              sizeStatuses={sizeColoringOn ? sizeStatuses : undefined}
              sizeColoringOn={sizeColoringOn}
              onSizeColoringToggle={() => setSizeColoringOn((v) => !v)}
              loadingLabels={pendingAnnotationIds.has(media.id)}
              onLabelRenamed={(_id, _oldLabel, newLabel) => {
                // A typed label the project lacks is added to it.
                if (!onAddProjectLabel) return;
                const trimmed = (newLabel || "").trim();
                if (!trimmed) return;
                const lower = trimmed.toLowerCase();
                const exists = labels.some((l) => l.toLowerCase() === lower);
                if (!exists) onAddProjectLabel(trimmed);
              }}
              emphasizedBoxId={hoveredIdx === null ? null : `imp_${index}_${hoveredIdx}`}
              onHoverChange={setHoveredCanvasBoxId}
              onAddBoxDetect={manualMode ? undefined :
                // Mask and label in one call; needs the import on the server.
                media.backendId && projectId && labels.length > 0
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
                      // 422: no object in the box, an ordinary miss.
                      if (r.status === 422) return null;
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
                // The engine reads a saved image by import_id; an unsaved one is sent.
                const fd = new FormData();
                fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                fd.append("labels", JSON.stringify(labels));
                if (projectId) fd.append("project_id", projectId);
                if (media.backendId && projectId) {
                  fd.append("import_id", media.backendId);
                } else {
                  const file = await ensureMediaFile(media);
                  if (!file || file.size === 0) {
                    console.warn("[v2 segment_box] media bytes unavailable");
                    return null;
                  }
                  fd.append("image", file);
                }
                const r = await apiFetch("/api/charlie/imports/segment_box", { method: "POST", body: fd });
                // 422: no object in the box, an ordinary miss.
                if (r.status === 422) return null;
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
                      if (projectId) fd.append("project_id", projectId);
                      if (media.backendId && projectId) {
                        fd.append("import_id", media.backendId);
                      } else {
                        const file = await ensureMediaFile(media);
                        if (!file || file.size === 0) {
                          console.warn("[v2 classify_box] media bytes unavailable");
                          return null;
                        }
                        fd.append("image", file);
                      }
                      const r = await apiFetch("/api/charlie/imports/classify_box", { method: "POST", body: fd });
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
                // detect_point also resolves the label for a specific dataset.
                const fd = new FormData();
                fd.append("point", JSON.stringify([point.x, point.y]));
                if (projectId) fd.append("project_id", projectId);
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
                let r: Response;
                try {
                  r = await apiFetch("/api/charlie/imports/detect_point", { method: "POST", body: fd });
                } catch (err) {
                  console.error("[detect_point] network error", err);
                  return null;
                }
                // 422: nothing at that point, an ordinary miss.
                if (r.status === 422) return null;
                if (!r.ok) {
                  const body = await r.text().catch(() => "");
                  console.error(
                    `[detect_point] http ${r.status} ${r.statusText}: ${body.slice(0, 400)}`,
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

        <div className="shrink-0 flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={onPrev}
            disabled={!hasPrev}
            aria-label="Previous image"
            className={[
              "h-9 px-4 inline-flex items-center gap-2 rounded-md border text-[13px] font-medium transition-all",
              hasPrev
                ? "border-[var(--line)] bg-foreground/[0.03] text-foreground/85 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
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
              "h-9 px-4 inline-flex items-center gap-2 rounded-md border text-[13px] font-medium transition-all",
              hasNext
                ? "border-[var(--line)] bg-foreground/[0.03] text-foreground/85 hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
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
