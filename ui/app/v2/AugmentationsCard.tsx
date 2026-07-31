"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "../../lib/apiFetch";
import { LabelJobCard, type LabelJobState } from "./LabelJobCard";

// Whimsical phrase rotator for the augmentation generate card ,
// same cadence as the labelling vibe phrases, content adjusted to
// the augmentation flow so the user knows what they're waiting on.
const AUGMENT_PHRASES = [
  "Rolling random rotations…",
  "Adding cinematic motion blur…",
  "Replacing scenes with new backgrounds…",
  "Dialling in time-of-day lighting…",
  "Generating off-axis viewing angles…",
  "Quantising bit depth like a vintage sensor…",
  "Sampling colour distortion seeds…",
  "Tinting hues into uncharted palettes…",
  "Painting block occlusions on segmentations…",
  "Sprinkling Gaussian noise across pixels…",
  "Crunching pixels for robustness…",
  "Making copies that are juuust different enough…",
  "Building a dataset that doesn't overfit…",
];
const REMOVE_AUGMENT_PHRASES = [
  "Sweeping augmentation copies off disk…",
  "Resetting per-image augmentation counts…",
  "Clearing the augmentations folder…",
  "Wiping warped polygon annotations…",
  "Dropping every generated variant…",
  "Returning the dataset to its source state…",
];

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

export type AugmentPreviewSource = {
  source: "reference" | "import";
  filename: string;
  preview: string;
};

// Augmentations card, visual / state-only for now. Sits under the
// Annotations card on the project view. Backend wiring intentionally
// deferred; clicking Update is a no-op placeholder.
//
// Hierarchy is two levels deep:
//   Category (Occlusion, Distortion, Camera, Domain) → expandable
//     ↳ Sub-augmentation (Random block, Object overlay, ...) → checkbox
//       + the controls relevant to that augmentation + a per-item
//       frequency toggle that only shows up when the sub is enabled.
//
// Each level (top-level "augmentations per image", per-category,
// per-sub) carries its own enable flag + frequency choice so the
// future backend payload is granular without forcing the user to
// edit nested dicts.

type Frequency = "all" | "random";
type PerImage = "off" | "1" | "2" | "3" | "random";

type SubBase = { enabled: boolean; frequency: Frequency };
type SubWithSize = SubBase & { size: number };
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type SubWithImages = SubBase & { images: (File | null)[] };
type ObjectOverlayEntry = {
  id: string;
  label: string;
  previewUrl: string;
};
type SubObjectOverlay = SubBase & {
  // Scale of each overlay's longest edge as a fraction of the
  // target image's longest edge. 0..1, step 0.01. Shared across
  // all overlays so the dial controls "how big" universally.
  scale: number;
  // Up to MAX_OBJECT_OVERLAYS overlays, each segmented from a
  // different uploaded image. Composited one after the other with
  // the 50% cap honoured cumulatively.
  overlays: ObjectOverlayEntry[];
};

const MAX_OBJECT_OVERLAYS = 3;
type BackgroundEntry = {
  id: string;
  previewUrl: string;
};
type SubBackgrounds = SubBase & {
  backgrounds: BackgroundEntry[];
};
const MAX_BACKGROUND_IMAGES = 6;
type SubWithStrength = SubBase & { strength: number };
type SubWithRange = SubBase & {
  scaleMin: number; scaleMax: number;
  rotMin: number; rotMax: number;
};

type AugmentationsState = {
  perImage: PerImage;
  occlusion: {
    enabled: boolean; frequency: Frequency;
    randomBlock: SubWithSize;
    objectOverlay: SubObjectOverlay;
  };
  distortion: {
    enabled: boolean; frequency: Frequency;
    perspectiveWarp: SubWithStrength;
    scaleRotation: SubWithRange;
    hueShift: SubWithStrength;
  };
  camera: {
    enabled: boolean; frequency: Frequency;
    // Camera dials are continuous 0..10 strengths, no per-dial
    // toggles. 0 = identity (no effect). The category-level
    // `enabled` flag still gates the whole group when the user
    // wants to turn camera augs off entirely.
    motionBlur: number;
    noise: number;
    colourDistortion: number;
    chromaticAberration: number;
    bitDepth: number;
    // Newer optics + compression dials. Older saved configs that
    // predate these keys default to 0 via the defaultState helper
    // below.
    lensDistortion: number;
    pixelation: number;
    lowResolution: number;
    lensGlare: number;
  };
  domain: {
    enabled: boolean; frequency: Frequency;
    backgrounds: SubBackgrounds;
    environmental: {
      enabled: boolean; frequency: Frequency;
      dust: SubBase;
      rain: SubBase;
      fog: SubBase;
      snow: SubBase;
    };
    lighting: SubWithStrength;
  };
};

function defaultState(): AugmentationsState {
  const fSub: SubBase = { enabled: false, frequency: "random" };
  return {
    perImage: "off",
    occlusion: {
      enabled: false, frequency: "random",
      randomBlock: { ...fSub, size: 0 },
      objectOverlay: { ...fSub, scale: 0.25, overlays: [] },
    },
    distortion: {
      enabled: false, frequency: "random",
      perspectiveWarp: { ...fSub, strength: 0 },
      scaleRotation: { ...fSub, scaleMin: 1.0, scaleMax: 1.0, rotMin: 0, rotMax: 0 },
      hueShift: { ...fSub, strength: 0 },
    },
    camera: {
      enabled: false, frequency: "random",
      motionBlur: 0,
      noise: 0,
      colourDistortion: 0,
      chromaticAberration: 0,
      bitDepth: 0,
      lensDistortion: 0,
      pixelation: 0,
      lowResolution: 0,
      lensGlare: 0,
    },
    domain: {
      enabled: false, frequency: "random",
      backgrounds: { ...fSub, backgrounds: [] },
      environmental: {
        enabled: false, frequency: "random",
        dust: { ...fSub }, rain: { ...fSub }, fog: { ...fSub }, snow: { ...fSub },
      },
      lighting: { ...fSub, strength: 0 },
    },
  };
}

export function AugmentationsCard({
  projectId = null,
  previewSources: previewSourcesProp = [],
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  previewSourcesReady = true,
  augmentJob = null,
  onAugmentJobChange,
}: {
  /** Backend project UUID, required for the live-preview endpoint
      to know which project's images to load. */
  projectId?: string | null;
  /** References + dataset imports the preview pane can pick a
      random source from. Both are eligible; the user can repick
      via the Random image button. */
  previewSources?: AugmentPreviewSource[];
  /** True once the parent has finished its initial /overview fetch.
      Reserved, empty-state copy now always reads "Loading dataset…"
      regardless of this flag, but keeping the prop so the parent's
      wiring doesn't need to change if we re-introduce the
      conditional copy later. */
  previewSourcesReady?: boolean;
  /** Augment-job state lifted up to ProjectViewV2Stub. The card
      renders the parent's value directly so we don't run two
      independent pollers (which used to disagree on progress when
      one captured the early "starting" snapshot and the other the
      mid-run one). The parent's poller is the single source of
      truth. */
  augmentJob?: LabelJobState | null;
  /** Parent setter so the Update button can stamp a "queued"
      placeholder immediately, the user sees the card the moment
      they click, instead of waiting for the next poll tick. */
  onAugmentJobChange?: (next: LabelJobState | null | ((cur: LabelJobState | null) => LabelJobState | null)) => void;
} = {}) {
  // Independent /overview fetch so the card never depends on the
  // parent's hydration timing. The parent's prop is just a hint ,
  // when both are present we de-dupe by filename and merge so any
  // fresh in-flight uploads from onboarding don't get dropped, but
  // the card stays functional even when the parent's prop is empty.
  //
  // Polls until something lands. Without this, the user can sit on
  // "Loading dataset…" forever if the first /overview happens to
  // race against a not-yet-ready auth session or a slow backend
  // boot, page refresh would be the only escape. The poll backs
  // off after a few attempts so we're not hammering the server.
  const [fallbackSources, setFallbackSources] = useState<AugmentPreviewSource[]>([]);
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    let timer: number | null = null;
    const fetchOnce = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (cancelled) return;
        if (r.ok) {
          const ov = await r.json() as {
            imports?: { id: string; filename: string }[];
            references?: { id: string; filename: string }[];
          };
          const next: AugmentPreviewSource[] = [];
          for (const imp of ov.imports ?? []) {
            if (!imp.filename) continue;
            next.push({
              source: "import",
              filename: imp.filename,
              preview: `${API}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
            });
          }
          for (const ref of ov.references ?? []) {
            if (!ref.filename) continue;
            next.push({
              source: "reference",
              filename: ref.filename,
              preview: `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(ref.filename)}`,
            });
          }
          if (!cancelled) {
            // Always commit, never give up. Even if we just got
            // entries this tick, the user might upload more on the
            // next, so keep polling so the card stays in sync with
            // the dataset without depending on the parent's state.
            setFallbackSources((cur) => {
              if (cur.length === next.length && cur.every((s, i) => s.filename === next[i].filename)) {
                return cur;
              }
              return next;
            });
          }
        }
      } catch { /* ignore, try again on next tick */ }
      if (cancelled) return;
      // Steady 2 s poll. Cheap (/overview is tens of ms) and
      // guarantees the card never gets stuck staring at an empty
      // dataset that the user just populated.
      timer = window.setTimeout(fetchOnce, 2000);
    };
    fetchOnce();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [projectId]);

  // Merge: prefer the parent's entries (they may carry preview
  // blobs for in-flight uploads), but de-dupe by filename so the
  // fallback fills in anything the parent doesn't yet know about.
  const previewSources = useMemo<AugmentPreviewSource[]>(() => {
    const seen = new Set<string>();
    const out: AugmentPreviewSource[] = [];
    for (const s of previewSourcesProp) {
      const k = `${s.source}|${s.filename}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(s);
    }
    for (const s of fallbackSources) {
      const k = `${s.source}|${s.filename}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(s);
    }
    return out;
  }, [previewSourcesProp, fallbackSources]);

  const [state, setState] = useState<AugmentationsState>(defaultState);
  const [updating, setUpdating] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [activeJob, setActiveJob] = useState<{
    id: string;
    status: string;
    progress?: { index?: number; total?: number; image?: string };
    elapsedS?: number;
    error?: string | null;
  } | null>(null);
  const [categoryOpen, setCategoryOpen] = useState<Record<string, boolean>>({
    occlusion: false, distortion: false, camera: false, domain: false,
  });

  // Restore the user's last-applied config on mount so revisiting
  // the Augmentations tab doesn't reset every toggle. The backend's
  // /augment/config endpoint returns the dict last persisted by an
  // Update click; an empty payload means the user hasn't generated
  // anything yet and the defaultState() is the right starting point.
  useEffect(() => {
    if (!projectId || configLoaded) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augment/config`);
        if (!r.ok) return;
        const data = await r.json() as {
          perImageMode?: string;
          config?: Partial<AugmentationsState>;
        };
        if (cancelled) return;
        const cfg = data?.config;
        if (cfg && typeof cfg === "object" && Object.keys(cfg).length > 0) {
          setState((cur) => ({
            ...cur,
            perImage: ((data?.perImageMode ?? "off") as PerImage),
            // Spread saved sub-trees over defaults so a future schema
            // bump (new sub-augmentation) doesn't crash old configs.
            camera: { ...cur.camera, ...(cfg.camera ?? {}) },
            domain: {
              ...cur.domain,
              ...(cfg.domain ?? {}),
              backgrounds: { ...cur.domain.backgrounds, ...((cfg.domain as AugmentationsState["domain"] | undefined)?.backgrounds ?? {}) },
              environmental: { ...cur.domain.environmental, ...((cfg.domain as AugmentationsState["domain"] | undefined)?.environmental ?? {}) },
              lighting: { ...cur.domain.lighting, ...((cfg.domain as AugmentationsState["domain"] | undefined)?.lighting ?? {}) },
            },
            occlusion: {
              ...cur.occlusion,
              ...(cfg.occlusion ?? {}),
              randomBlock: { ...cur.occlusion.randomBlock, ...((cfg.occlusion as AugmentationsState["occlusion"] | undefined)?.randomBlock ?? {}) },
              objectOverlay: { ...cur.occlusion.objectOverlay, ...((cfg.occlusion as AugmentationsState["occlusion"] | undefined)?.objectOverlay ?? {}) },
            },
            distortion: {
              ...cur.distortion,
              ...(cfg.distortion ?? {}),
              perspectiveWarp: { ...cur.distortion.perspectiveWarp, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.perspectiveWarp ?? {}) },
              scaleRotation: { ...cur.distortion.scaleRotation, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.scaleRotation ?? {}) },
              hueShift: { ...cur.distortion.hueShift, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.hueShift ?? {}) },
            },
          }));
          // Auto-open any category that was on in the restored
          // config, feels weird to land on a "Cleared" card and
          // discover settings hiding behind a chevron.
          setCategoryOpen((co) => ({
            occlusion: co.occlusion || !!(cfg.occlusion as AugmentationsState["occlusion"] | undefined)?.enabled,
            distortion: co.distortion || !!(cfg.distortion as AugmentationsState["distortion"] | undefined)?.enabled,
            camera: co.camera || !!(cfg.camera as AugmentationsState["camera"] | undefined)?.enabled,
            domain: co.domain || !!(cfg.domain as AugmentationsState["domain"] | undefined)?.enabled,
          }));
        }
      } catch { /* ignore, start from defaults */ }
      finally { if (!cancelled) setConfigLoaded(true); }
    })();
    return () => { cancelled = true; };
  }, [projectId, configLoaded]);

  // Poll the project's active augment_generate job every 1s while
  // one is running, and on mount in case the user reloaded mid-job.
  // When the job transitions from running → null (server finished),
  // broadcast a window event so the dataset gallery refetches its
  // overview and the per-tile Augmentations icon appears straight
  // away (instead of waiting for the user to navigate the page).
  const lastJobIdRef = useRef<string | null>(null);
  const firedForJobRef = useRef<string | null>(null);
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/augment/job/active`);
        if (!r.ok) return;
        const data = await r.json();
        if (cancelled) return;
        setActiveJob((prev) => {
          const wasActive =
            prev?.status === "running" ||
            prev?.status === "queued" ||
            prev?.status === "done";
          const isStillActive =
            data && (data.status === "running" || data.status === "queued");
          // Fire on running→done OR running→null (job cleared
          // between polls). Track per-job so we never fire twice
          // for the same generate run.
          const finishedJobId = prev?.id ?? lastJobIdRef.current;
          const shouldFire =
            wasActive &&
            !isStillActive &&
            projectId &&
            finishedJobId &&
            firedForJobRef.current !== finishedJobId;
          if (shouldFire) {
            firedForJobRef.current = finishedJobId;
            try {
              window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
                detail: { projectId, jobId: finishedJobId },
              }));
            } catch { /* ignore */ }
          }
          lastJobIdRef.current = data?.id ?? lastJobIdRef.current;
          // Only commit a new state object if something actually
          // changed. Without this guard, every 1 s tick returns a
          // fresh `data` reference, re-renders AugmentationsCard,
          // and (because the augmentation viewer modal lives in
          // the same tree) makes the modal's backdrop-blur recompute
          //, a visible black/normal flash on hover.
          if (
            prev?.id === (data?.id ?? null) &&
            prev?.status === (data?.status ?? null) &&
            (prev?.progress?.index ?? null) === (data?.progress?.index ?? null) &&
            (prev?.progress?.total ?? null) === (data?.progress?.total ?? null) &&
            (prev?.progress?.image ?? null) === (data?.progress?.image ?? null)
          ) {
            return prev;
          }
          return data ?? null;
        });
      } catch {
        // ignore, polling resumes next tick
      }
    };
    void tick();
    const id = window.setInterval(tick, 1000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [projectId]);

  // Kick off a generate job using the current config. When the
  // per-image slider is at "off", the backend treats this as a
  // clear-only request, it wipes every augmentation on disk +
  // resets n_augmentations without scheduling a generate job.
  const onUpdate = async () => {
    if (!projectId || updating) return;
    setUpdating(true);
    setUpdateError(null);
    // Optimistic stamp, paint the progress card the moment the
    // user clicks, instead of waiting up to 2 s for the parent's
    // poll to detect the new job. Status starts as "queued" so the
    // card's labelling-style copy reads sensibly; the first
    // poll-tick refines status + index + total with the truth.
    // perImageMode="off" goes through the same card with the
    // "removing" sentinel jobId so the render below switches copy.
    const isClearOnly = state.perImage === "off";
    if (onAugmentJobChange) {
      onAugmentJobChange({
        jobId: isClearOnly ? "removing" : "pending",
        status: "queued",
        index: 0,
        total: 0,
        startedAt: Date.now(),
        currentImage: null,
      });
    }
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/augment/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          perImageMode: state.perImage,
          config: {
            camera: state.camera,
            domain: state.domain,
            occlusion: state.occlusion,
            distortion: state.distortion,
          },
        }),
      });
      if (!r.ok) {
        throw new Error(`http ${r.status}: ${await r.text()}`);
      }
      // Always fire the augmentations-generated event after a
      // successful Update POST. Listeners (dataset gallery icon
      // refresh + DatasetStatsCard signal bump + augmentation
      // viewer cache buster) re-read the relevant data. Two
      // refreshes, one now (optimistic, may still show stale
      // counts) and one when the job completes (via the active-
      // job poll), beat missing the refresh entirely on FAST
      // jobs that finish between two ticks.
      try {
        window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
          detail: { projectId },
        }));
      } catch { /* ignore */ }
      // For the clear-only path the backend doesn't enqueue a job,
      // so the parent's poller will never flip our optimistic stamp
      // to "done". Do it here so the card shows the success state
      // for a beat before auto-dismissing via LabelJobCard's done
      // timer.
      if (isClearOnly && onAugmentJobChange) {
        onAugmentJobChange((cur) => cur ? { ...cur, status: "done", total: cur.total || 1, index: cur.total || 1 } : cur);
      }
    } catch (e) {
      setUpdateError(e instanceof Error ? e.message : String(e));
      // Clear the optimistic stamp on failure so a stuck "queued"
      // card doesn't sit forever.
      if (onAugmentJobChange) {
        onAugmentJobChange((cur) => cur ? { ...cur, status: "failed" } : cur);
      }
    } finally {
      setUpdating(false);
    }
  };

  const setCat = (
    key: "occlusion" | "distortion" | "camera" | "domain",
    patch: Partial<AugmentationsState["occlusion"]>,
  ) => {
    setState((cur) => ({ ...cur, [key]: { ...cur[key], ...patch } }));
  };

  // One-click randomise. Picks fresh values for every augmentation
  // that's "pure dial" (operates on the pixels of the existing
  // image). Skips the augmentations that require user-supplied
  // assets, background randomisation needs uploaded backgrounds,
  // object overlay needs uploaded foregrounds. The button enables
  // each touched category so the user sees the values land
  // immediately; coming-soon rows (environmental, hue shift on
  // mega-only, etc.) stay off.
  const randomiseAll = () => {
    if (updating || activeJob) return;
    const rnd = (min: number, max: number) => min + Math.random() * (max - min);
    const flip = (p: number) => Math.random() < p;
    setState((cur) => ({
      ...cur,
      // Per-image count: pick 1 / 2 / 3 / random, never "off",
      // otherwise hitting Randomise → Update would clear instead
      // of generating.
      perImage: (["1", "2", "3", "random"] as const)[Math.floor(Math.random() * 4)],
      camera: {
        ...cur.camera,
        enabled: true,
        frequency: flip(0.5) ? "all" : "random",
        motionBlur: Math.round(rnd(0, 6) * 10) / 10,
        noise: Math.round(rnd(0, 5) * 10) / 10,
        colourDistortion: Math.round(rnd(0, 6) * 10) / 10,
        chromaticAberration: Math.round(rnd(0, 4) * 10) / 10,
        bitDepth: Math.round(rnd(0, 4) * 10) / 10,
      },
      distortion: {
        ...cur.distortion,
        enabled: true,
        frequency: flip(0.5) ? "all" : "random",
        perspectiveWarp: {
          ...cur.distortion.perspectiveWarp,
          enabled: true,
          frequency: flip(0.5) ? "all" : "random",
          strength: Math.round(rnd(2, 7) * 10) / 10,
        },
        scaleRotation: (() => {
          // Scale band is 0.7..1.3; pick a centre + spread so the
          // rolled [min, max] stays inside.
          const sCentre = rnd(0.9, 1.1);
          const sSpread = rnd(0.05, 0.18);
          const rCentre = rnd(-15, 15);
          const rSpread = rnd(5, 25);
          const scaleMin = Math.max(0.7, Math.round((sCentre - sSpread) * 100) / 100);
          const scaleMax = Math.min(1.3, Math.round((sCentre + sSpread) * 100) / 100);
          return {
            ...cur.distortion.scaleRotation,
            enabled: true,
            frequency: flip(0.5) ? "all" : "random",
            scaleMin,
            scaleMax,
            rotMin: Math.round(rCentre - rSpread),
            rotMax: Math.round(rCentre + rSpread),
          };
        })(),
        hueShift: {
          ...cur.distortion.hueShift,
          enabled: flip(0.5),
          frequency: flip(0.5) ? "all" : "random",
          strength: Math.round(rnd(2, 7) * 10) / 10,
        },
      },
      occlusion: {
        ...cur.occlusion,
        enabled: true,
        frequency: flip(0.5) ? "all" : "random",
        randomBlock: {
          ...cur.occlusion.randomBlock,
          enabled: flip(0.7),
          frequency: flip(0.5) ? "all" : "random",
          size: Math.round(rnd(0.05, 0.25) * 100) / 100,
        },
        // objectOverlay needs uploaded assets, leave it untouched
        objectOverlay: cur.occlusion.objectOverlay,
      },
      domain: {
        ...cur.domain,
        enabled: true,
        frequency: flip(0.5) ? "all" : "random",
        // backgrounds + environmental skipped, backgrounds needs
        // uploaded images, environmental is coming-soon disabled.
        backgrounds: cur.domain.backgrounds,
        environmental: cur.domain.environmental,
        lighting: {
          ...cur.domain.lighting,
          enabled: true,
          frequency: flip(0.5) ? "all" : "random",
          strength: Math.round(rnd(2, 7) * 10) / 10,
        },
      },
    }));
    // Open every category so the user can immediately see the
    // freshly-rolled values without hunting for chevrons.
    setCategoryOpen({ occlusion: true, distortion: true, camera: true, domain: true });
  };

  // Count of enabled subs across the whole state, surfaced as a
  // small pill next to the title so the user can see at a glance
  // whether anything's currently on.
  const enabledCount = countEnabled(state);

  return (
    // Tab page layout, matches the Dataset-tab section style
    // (max-w-6xl, pt-8, page-level h2, stacked content). No
    // wrapping card chrome since this is a whole tab, not a
    // collapsed module inside another page.
    <section className="px-6 lg:px-10 pt-8 pb-12">
      {/* Page header, title left, Update CTA right. Matches the
          References / Annotations title rows on the Dataset tab. */}
      <div className="flex items-center justify-between gap-4 mb-2">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="text-xl font-medium tracking-tight text-[var(--foreground)]">
            Augmentations
          </h2>
          {enabledCount > 0 && (
            <span className="inline-flex items-center gap-1 rounded-md border border-[var(--line)] px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider tabular-nums text-[var(--fg-muted)]">
              <span className="h-1 w-1 rounded-full bg-[var(--accent)]" aria-hidden />
              {enabledCount} on
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 shrink-0">
          <button
            type="button"
            onClick={() => randomiseAll()}
            disabled={updating || !!activeJob}
            className={[
              "inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] font-medium transition-colors",
              updating || activeJob
                ? "text-[var(--fg-faint)]"
                : "text-[var(--fg-soft)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground",
            ].join(" ")}
            title="Randomise every augmentation setting (skips inputs that need uploaded images, like background and overlay)"
          >
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <rect x="2.5" y="2.5" width="19" height="19" rx="3" />
              <circle cx="8" cy="8" r="1.1" fill="currentColor" />
              <circle cx="16" cy="8" r="1.1" fill="currentColor" />
              <circle cx="12" cy="12" r="1.1" fill="currentColor" />
              <circle cx="8" cy="16" r="1.1" fill="currentColor" />
              <circle cx="16" cy="16" r="1.1" fill="currentColor" />
            </svg>
            Randomise
          </button>
          <button
            type="button"
            onClick={() => void onUpdate()}
            disabled={updating || !!activeJob}
            className={[
              "rounded-md bg-[var(--accent)] px-3 py-1 text-[13px] font-medium text-[var(--accent-contrast)] transition-[filter,opacity]",
              updating || activeJob ? "opacity-50" : "hover:brightness-105",
            ].join(" ")}
            title={
              state.perImage === "off"
                ? "Delete every augmentation for this project"
                : "Generate augmentations for every dataset image"
            }
          >
            {updating
              ? (state.perImage === "off" ? "Clearing…" : "Starting…")
              : activeJob
              ? "Generating…"
              : state.perImage === "off"
              ? "Clear all"
              : "Update"}
          </button>
        </div>
      </div>
      <p className="text-sm text-foreground/50 mb-6">
        Improve model robustness with configurable dataset augmentations.
      </p>

      {/* Progress bar, matches the auto-labelling job card on the
          Dataset tab. Pops in while a generate job is running and
          stays for one final tick after it completes so the user
          can read the final state. */}
      {/* Reuse the LabelJobCard chrome so the augment job has the
          same look-and-feel as the auto-labelling card on the
          Dataset tab. Headlines + phrase pool overridden so the
          copy reads in context. */}
      <div
        className="mb-6 grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
        style={{
          gridTemplateRows: augmentJob ? "1fr" : "0fr",
          opacity: augmentJob ? 1 : 0,
          marginBottom: augmentJob ? "1.5rem" : "0",
        }}
      >
        <div className="min-h-0 overflow-hidden">
          {/* Single source of truth: parent's augmentJob state.
              Stops the dual-poller bug where this card and the
              parent disagreed on progress (one saw the initial null
              snapshot, the other saw the mid-run one). */}
          {augmentJob && (
            <LabelJobCard
              state={augmentJob}
              onClose={() => onAugmentJobChange?.(null)}
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
              headlines={
                augmentJob.jobId === "removing"
                  ? {
                      running: "Removing augmentations",
                      done: "Augmentations cleared",
                      failed: "Couldn't clear augmentations",
                      cancelled: "Removal cancelled",
                    }
                  : {
                      running: "Generating augmentations",
                      done: "Augmentations ready",
                      failed: "Augmentation generation failed",
                      cancelled: "Augmentation generation cancelled",
                    }
              }
              doneMessage={
                augmentJob.jobId === "removing"
                  ? "Augmentations wiped from disk."
                  : "All images augmented. Open the Dataset tab to view them."
              }
              phrases={
                augmentJob.jobId === "removing"
                  ? REMOVE_AUGMENT_PHRASES
                  : AUGMENT_PHRASES
              }
            />
          )}
        </div>
      </div>
      {updateError && (
        <div className="mb-6 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-2 text-[11px] text-[var(--bad)]">
          {updateError}
        </div>
      )}

      <div className="space-y-6">
        {/* Augmentations per image, top-level frequency-cap toggle. */}
        <div>
          <div className="pk-micro mb-3">Augmentations per image</div>
          <SegmentedToggle<PerImage>
                  options={[
                    { value: "off", label: "Off" },
                    { value: "1", label: "1" },
                    { value: "2", label: "2" },
                    { value: "3", label: "3" },
                    { value: "random", label: "Random" },
                  ]}
                  value={state.perImage}
                  onChange={(v) => setState((s) => ({ ...s, perImage: v }))}
                />
              </div>

              {/* Category render order: Camera → Domain →
                  Occlusion → Distortion. */}
              {/* Camera & sensor simulation */}
              <CategoryBlock
                title="Camera & sensor simulation"
                description="Simulate the artefacts of real-world capture so the model isn't tripped by clean training data."
                enabled={state.camera.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.camera.enabled;
                  return {
                    ...s,
                    camera: {
                      ...s.camera,
                      enabled: next,
                      // Camera has no sub-toggles, its "subs" are
                      // continuous dials. Resetting them to 0 on
                      // disable so re-enabling later starts from
                      // identity instead of carrying stale values.
                      ...(next ? {} : {
                        motionBlur: 0,
                        noise: 0,
                        colourDistortion: 0,
                        chromaticAberration: 0,
                        bitDepth: 0,
                      }),
                    },
                  };
                })}
                frequency={state.camera.frequency}
                onFrequency={(f) => setCat("camera", { frequency: f })}
                open={categoryOpen.camera}
                onOpen={() => setCategoryOpen((s) => ({ ...s, camera: !s.camera }))}
              >
                <CameraSensorBlock
                  projectId={projectId}
                  previewSources={previewSources}
                  motionBlur={state.camera.motionBlur}
                  noise={state.camera.noise}
                  colourDistortion={state.camera.colourDistortion}
                  chromaticAberration={state.camera.chromaticAberration}
                  bitDepth={state.camera.bitDepth}
                  lensDistortion={state.camera.lensDistortion}
                  pixelation={state.camera.pixelation}
                  lowResolution={state.camera.lowResolution}
                  lensGlare={state.camera.lensGlare}
                  onChange={(patch) =>
                    setState((s) => ({ ...s, camera: { ...s.camera, ...patch } }))
                  }
                />
              </CategoryBlock>

              {/* Domain randomisation */}
              <CategoryBlock
                title="Domain randomisation"
                description="Vary background, environment, and lighting so the model transfers across deployment conditions."
                enabled={state.domain.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.domain.enabled;
                  return {
                    ...s,
                    domain: {
                      ...s.domain,
                      enabled: next,
                      ...(next ? {} : {
                        backgrounds: { ...s.domain.backgrounds, enabled: false },
                        environmental: {
                          ...s.domain.environmental,
                          enabled: false,
                          dust: { ...s.domain.environmental.dust, enabled: false },
                          rain: { ...s.domain.environmental.rain, enabled: false },
                          fog: { ...s.domain.environmental.fog, enabled: false },
                          snow: { ...s.domain.environmental.snow, enabled: false },
                        },
                        lighting: { ...s.domain.lighting, enabled: false },
                      }),
                    },
                  };
                })}
                frequency={state.domain.frequency}
                onFrequency={(f) => setCat("domain", { frequency: f })}
                open={categoryOpen.domain}
                onOpen={() => setCategoryOpen((s) => ({ ...s, domain: !s.domain }))}
              >
                <SubAugRow
                  label="Background randomisation"
                  description="Replace the original background with one of the supplied images."
                  enabled={state.domain.backgrounds.enabled}
                  onToggle={(v) =>
                    setState((s) => ({
                      ...s,
                      domain: {
                        ...s.domain,
                        enabled: v || s.domain.enabled,
                        backgrounds: { ...s.domain.backgrounds, enabled: v },
                      },
                    }))
                  }
                  frequency={state.domain.backgrounds.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      domain: { ...s.domain, backgrounds: { ...s.domain.backgrounds, frequency: f } },
                    }))
                  }
                >
                  {state.domain.backgrounds.enabled && (
                    <BackgroundRandomisationBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      backgrounds={state.domain.backgrounds.backgrounds}
                      onBackgrounds={(next) =>
                        setState((s) => ({
                          ...s,
                          domain: { ...s.domain, backgrounds: { ...s.domain.backgrounds, backgrounds: next } },
                        }))
                      }
                    />
                  )}
                </SubAugRow>

                {/* Environmental effects, nested group, each item
                    has its own checkbox + frequency. */}
                {/* Environmental effects, gated behind Coming
                    soon while we build the weather particle
                    compositor. State + nested kind rows kept
                    intact so re-enabling is a single deletion
                    of the rightBadge prop. */}
                <SubAugRow
                  label="Environmental effects"
                  description="Add weather and atmospheric particles for outdoor robustness."
                  enabled={false}
                  onToggle={() => { /* disabled */ }}
                  frequency={state.domain.environmental.frequency}
                  onFrequency={() => { /* disabled */ }}
                  rightBadge={
                    <span className="inline-flex items-center gap-1 rounded-md border border-[var(--line)] px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--fg-dim)]">
                      Coming soon
                    </span>
                  }
                />

                <SubAugRow
                  label="Lighting variation"
                  description="Shift brightness, contrast, and warmth to match different times of day."
                  enabled={state.domain.lighting.enabled}
                  onToggle={(v) =>
                    setState((s) => ({
                      ...s,
                      domain: {
                        ...s.domain,
                        enabled: v || s.domain.enabled,
                        lighting: { ...s.domain.lighting, enabled: v },
                      },
                    }))
                  }
                  frequency={state.domain.lighting.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      domain: { ...s.domain, lighting: { ...s.domain.lighting, frequency: f } },
                    }))
                  }
                >
                  {state.domain.lighting.enabled && (
                    <SimpleStrengthBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      strength={state.domain.lighting.strength}
                      onChange={(v) =>
                        setState((s) => ({
                          ...s,
                          domain: { ...s.domain, lighting: { ...s.domain.lighting, strength: v } },
                        }))
                      }
                      buildBody={(_src, strength) => ({ lighting_strength: strength })}
                    />
                  )}
                </SubAugRow>
              </CategoryBlock>

              {/* Occlusion */}
              <CategoryBlock
                title="Occlusion"
                description="Mask out parts of the image so the model learns to recover under partial visibility."
                enabled={state.occlusion.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.occlusion.enabled;
                  return {
                    ...s,
                    occlusion: {
                      ...s.occlusion,
                      enabled: next,
                      // Cascade off, turning the category off
                      // deselects every sub so the preview blocks
                      // unmount. Turning it on leaves subs as the
                      // user left them.
                      ...(next ? {} : {
                        randomBlock: { ...s.occlusion.randomBlock, enabled: false },
                        objectOverlay: { ...s.occlusion.objectOverlay, enabled: false },
                      }),
                    },
                  };
                })}
                frequency={state.occlusion.frequency}
                onFrequency={(f) => setCat("occlusion", { frequency: f })}
                open={categoryOpen.occlusion}
                onOpen={() => setCategoryOpen((s) => ({ ...s, occlusion: !s.occlusion }))}
              >
                <SubAugRow
                  label="Random block occlusion"
                  description="Drops one or more axis-aligned rectangles onto the image, constrained to the segmented regions."
                  enabled={state.occlusion.randomBlock.enabled}
                  onToggle={(v) =>
                    setState((s) => ({
                      ...s,
                      occlusion: {
                        ...s.occlusion,
                        // Enabling a sub-augmentation auto-enables
                        // its parent category, otherwise the user
                        // turns on the sub and nothing happens
                        // because the category gate is still off.
                        enabled: v || s.occlusion.enabled,
                        randomBlock: { ...s.occlusion.randomBlock, enabled: v },
                      },
                    }))
                  }
                  frequency={state.occlusion.randomBlock.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      occlusion: { ...s.occlusion, randomBlock: { ...s.occlusion.randomBlock, frequency: f } },
                    }))
                  }
                >
                  {state.occlusion.randomBlock.enabled && (
                    <RandomBlockOcclusionBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      size={state.occlusion.randomBlock.size}
                      onChange={(v) =>
                        setState((s) => ({
                          ...s,
                          occlusion: { ...s.occlusion, randomBlock: { ...s.occlusion.randomBlock, size: v } },
                        }))
                      }
                    />
                  )}
                </SubAugRow>

                {/* Object overlay, temporarily disabled while we
                    speed up SAM3 + compositing. Toggle is a no-op
                    and the body shows a "Coming soon" placeholder
                    so the user can see the augmentation is in the
                    pipeline. The full ObjectOverlayBlock component
                    + segmentation endpoint stay in the codebase
                    untouched and will be re-enabled in a future
                    pass. */}
                <SubAugRow
                  label="Object overlay"
                  description="PixelKit cuts the named object out of an uploaded image and pastes it onto random dataset images."
                  enabled={false}
                  onToggle={() => { /* disabled */ }}
                  frequency={state.occlusion.objectOverlay.frequency}
                  onFrequency={() => { /* disabled */ }}
                  rightBadge={
                    <span className="inline-flex items-center gap-1 rounded-md border border-[var(--line)] px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--fg-dim)]">
                      Coming soon
                    </span>
                  }
                />
                {/*, preview UI hidden until ObjectOverlayBlock is
                    fast enough to ship. Keep this commented call
                    around so re-enabling is a single uncomment.
                  <ObjectOverlayBlock
                    projectId={projectId}
                    previewSources={previewSources}
                    overlays={state.occlusion.objectOverlay.overlays}
                    scale={state.occlusion.objectOverlay.scale}
                    onOverlays={(next) =>
                      setState((s) => ({
                        ...s,
                        occlusion: { ...s.occlusion, objectOverlay: { ...s.occlusion.objectOverlay, overlays: next } },
                      }))
                    }
                    onScale={(v) =>
                      setState((s) => ({
                        ...s,
                        occlusion: { ...s.occlusion, objectOverlay: { ...s.occlusion.objectOverlay, scale: v } },
                      }))
                    }
                  />
                */}
              </CategoryBlock>

              {/* Distortion */}
              <CategoryBlock
                title="Distortion"
                description="Warp the spatial geometry so the model generalises across viewpoints."
                enabled={state.distortion.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.distortion.enabled;
                  return {
                    ...s,
                    distortion: {
                      ...s.distortion,
                      enabled: next,
                      ...(next ? {} : {
                        perspectiveWarp: { ...s.distortion.perspectiveWarp, enabled: false },
                        scaleRotation: { ...s.distortion.scaleRotation, enabled: false },
                        hueShift: { ...s.distortion.hueShift, enabled: false },
                      }),
                    },
                  };
                })}
                frequency={state.distortion.frequency}
                onFrequency={(f) => setCat("distortion", { frequency: f })}
                open={categoryOpen.distortion}
                onOpen={() => setCategoryOpen((s) => ({ ...s, distortion: !s.distortion }))}
              >
                <SubAugRow
                  label="Perspective warp"
                  description="Simulates off-axis viewing angles."
                  enabled={state.distortion.perspectiveWarp.enabled}
                  onToggle={(v) =>
                    setState((s) => ({
                      ...s,
                      distortion: {
                        ...s.distortion,
                        enabled: v || s.distortion.enabled,
                        perspectiveWarp: { ...s.distortion.perspectiveWarp, enabled: v },
                      },
                    }))
                  }
                  frequency={state.distortion.perspectiveWarp.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      distortion: { ...s.distortion, perspectiveWarp: { ...s.distortion.perspectiveWarp, frequency: f } },
                    }))
                  }
                >
                  {state.distortion.perspectiveWarp.enabled && (
                    <PerspectiveWarpBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      strength={state.distortion.perspectiveWarp.strength}
                      onChange={(v) =>
                        setState((s) => ({
                          ...s,
                          distortion: { ...s.distortion, perspectiveWarp: { ...s.distortion.perspectiveWarp, strength: v } },
                        }))
                      }
                    />
                  )}
                </SubAugRow>

                <SubAugRow
                  label="Scale & rotation"
                  description="Apply uniform scaling and rotation within the configured ranges."
                  enabled={state.distortion.scaleRotation.enabled}
                  onToggle={(v) =>
                    setState((s) => {
                      const cur = s.distortion.scaleRotation;
                      // When enabling, lift identity ranges to a useful
                      // default so the dual-slider thumbs aren't stacked
                      // on top of each other. When disabling, snap back
                      // to identity so re-enabling later starts clean.
                      const isIdentity = cur.scaleMin === 1.0 && cur.scaleMax === 1.0 && cur.rotMin === 0 && cur.rotMax === 0;
                      const ranges = v && isIdentity
                        ? { scaleMin: 0.9, scaleMax: 1.1, rotMin: -10, rotMax: 10 }
                        : (!v
                          ? { scaleMin: 1.0, scaleMax: 1.0, rotMin: 0, rotMax: 0 }
                          : {});
                      return {
                        ...s,
                        distortion: {
                          ...s.distortion,
                          enabled: v || s.distortion.enabled,
                          scaleRotation: { ...cur, enabled: v, ...ranges },
                        },
                      };
                    })
                  }
                  frequency={state.distortion.scaleRotation.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      distortion: { ...s.distortion, scaleRotation: { ...s.distortion.scaleRotation, frequency: f } },
                    }))
                  }
                >
                  {state.distortion.scaleRotation.enabled && (
                    <ScaleRotationBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      scaleMin={state.distortion.scaleRotation.scaleMin}
                      scaleMax={state.distortion.scaleRotation.scaleMax}
                      rotMin={state.distortion.scaleRotation.rotMin}
                      rotMax={state.distortion.scaleRotation.rotMax}
                      onChange={(patch) =>
                        setState((s) => ({
                          ...s,
                          distortion: {
                            ...s.distortion,
                            scaleRotation: { ...s.distortion.scaleRotation, ...patch },
                          },
                        }))
                      }
                    />
                  )}
                </SubAugRow>

                <SubAugRow
                  label="Hue shift"
                  description="Rotate the colour hue by a random offset so the model isn't tied to the dataset's palette."
                  enabled={state.distortion.hueShift.enabled}
                  onToggle={(v) =>
                    setState((s) => ({
                      ...s,
                      distortion: {
                        ...s.distortion,
                        enabled: v || s.distortion.enabled,
                        hueShift: { ...s.distortion.hueShift, enabled: v },
                      },
                    }))
                  }
                  frequency={state.distortion.hueShift.frequency}
                  onFrequency={(f) =>
                    setState((s) => ({
                      ...s,
                      distortion: { ...s.distortion, hueShift: { ...s.distortion.hueShift, frequency: f } },
                    }))
                  }
                >
                  {state.distortion.hueShift.enabled && (
                    <SimpleStrengthBlock
                      projectId={projectId}
                      previewSources={previewSources}
                      strength={state.distortion.hueShift.strength}
                      onChange={(v) =>
                        setState((s) => ({
                          ...s,
                          distortion: { ...s.distortion, hueShift: { ...s.distortion.hueShift, strength: v } },
                        }))
                      }
                      buildBody={(_src, strength) => ({ hue_shift: strength })}
                    />
                  )}
                </SubAugRow>
              </CategoryBlock>

      </div>
    </section>
  );
}

// ─── Building blocks ─────────────────────────────────────────────

function countEnabled(s: AugmentationsState): number {
  // Top-level count, one tick per enabled category, irrespective
  // of how many sub-augmentations / dials are active inside it.
  // Matches the user's mental model that "Camera & sensor" on the
  // chip is one thing, not five.
  let n = 0;
  if (s.occlusion.enabled) n++;
  if (s.distortion.enabled) n++;
  if (s.camera.enabled) n++;
  if (s.domain.enabled) n++;
  return n;
}

// ─── Camera & sensor block ─────────────────────────────────────
// Live-preview pane + four 0..10 dials. The preview pulls a random
// reference / dataset image, ships it through /augment/preview on
// every dial change (debounced) and crossfades the new bake in.
// The Random image button re-picks a source and re-seeds the
// noise pattern so the user sees a different starting point.
function CameraSensorBlock({
  projectId,
  previewSources,
  motionBlur,
  noise,
  colourDistortion,
  chromaticAberration,
  bitDepth,
  lensDistortion,
  pixelation,
  lowResolution,
  lensGlare,
  onChange,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  motionBlur: number;
  noise: number;
  colourDistortion: number;
  chromaticAberration: number;
  bitDepth: number;
  lensDistortion: number;
  pixelation: number;
  lowResolution: number;
  lensGlare: number;
  onChange: (patch: Partial<{
    motionBlur: number;
    noise: number;
    colourDistortion: number;
    chromaticAberration: number;
    bitDepth: number;
    lensDistortion: number;
    pixelation: number;
    lowResolution: number;
    lensGlare: number;
  }>) => void;
}) {
  // Active source for the preview. Picked deterministically on
  // mount (first item) and re-picked on Random-image click.
  const [sourceIdx, setSourceIdx] = useState<number>(() => 0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = previewSources[sourceIdx] ?? null;

  // Resolved preview URL. Empty when there are no sources yet ,
  // the block shows a placeholder card in that case.
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  // Snapshot of the latest request so a slow earlier response
  // doesn't overwrite a faster newer one.
  const requestIdRef = useRef(0);

  const allZero = motionBlur === 0
    && noise === 0
    && colourDistortion === 0
    && chromaticAberration === 0
    && bitDepth === 0
    && lensDistortion === 0
    && pixelation === 0
    && lowResolution === 0
    && lensGlare === 0;

  // Debounced fetch. Snapshots all five inputs at schedule time;
  // any change re-arms the timer and supersedes the pending call.
  useEffect(() => {
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) {
        URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = null;
      }
      setPreviewUrl(null);
      return;
    }
    // When all dials are 0 we just show the source image directly ,
    // skips a backend round-trip every time the user lands back
    // on identity.
    if (allZero) {
      if (lastBlobRef.current) {
        URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = null;
      }
      setPreviewUrl(activeSource.preview);
      setPreviewError(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              motion_blur: motionBlur,
              noise,
              colour_distortion: colourDistortion,
              chromatic_aberration: chromaticAberration,
              bit_depth: bitDepth,
              lens_distortion: lensDistortion,
              pixelation,
              low_resolution: lowResolution,
              lens_glare: lensGlare,
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        // Ignore if a newer request has already replaced us.
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
    // activeSource is a NEW object on every parent re-render (the
    // previewSources prop is rebuilt from refs/imports state and
    // re-filtered/sliced each render), which used to force this
    // effect to re-fire and the debounce to re-arm forever. Depend
    // on scalar fields so a no-op parent render is a no-op here.
  }, [
    projectId,
    activeSource?.source,
    activeSource?.filename,
    allZero, seed,
    motionBlur, noise, colourDistortion, chromaticAberration, bitDepth,
    lensDistortion, pixelation, lowResolution, lensGlare,
  ]);

  // Release the last blob on unmount so we don't leak object URLs.
  useEffect(() => () => {
    if (lastBlobRef.current) {
      URL.revokeObjectURL(lastBlobRef.current);
      lastBlobRef.current = null;
    }
  }, []);

  const pickRandom = useCallback(() => {
    if (previewSources.length === 0) return;
    if (previewSources.length === 1) {
      // Single source, just re-seed so noise/colour-jitter shift.
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    // Pick anything that isn't the current source so the user
    // actually sees a change.
    let next = sourceIdx;
    while (next === sourceIdx) {
      next = Math.floor(Math.random() * previewSources.length);
    }
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [previewSources.length, sourceIdx]);

  const resetAll = () => onChange({
    motionBlur: 0, noise: 0, colourDistortion: 0, chromaticAberration: 0, bitDepth: 0,
    lensDistortion: 0, pixelation: 0, lowResolution: 0, lensGlare: 0,
  });

  return (
    <div className="grid gap-4 md:grid-cols-2 md:items-center">
      {/* Preview pane, left column on md+. No fixed aspect: the
          image renders at its natural ratio so portrait sources
          don't get letterboxed with black bars. Vertically
          centered against the dial stack so a short image + its
          controls sit halfway down the card. */}
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Augmentation preview"
              className="block w-full h-auto"
              draggable={false}
            />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {previewSources.length === 0
                  ? "Loading dataset…"
                  : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={previewSources.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
            title="Pick a new random image"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
              {activeSource ? activeSource.filename : ""}
            </span>
            {!allZero && (
              <button
                type="button"
                onClick={resetAll}
                className="text-[10px] uppercase tracking-wider text-foreground/40 hover:text-foreground"
                title="Reset all dials to 0"
              >
                Reset
              </button>
            )}
          </div>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>

      {/* Dials, right column on md+. Each 0..10, default 0,
          step 0.1 for fine control. Commits only on release so
          dragging doesn't hammer the GPU. */}
      <div className="grid gap-3 content-start">
        <BigDial label="Motion blur" value={motionBlur}
          onCommit={(v) => onChange({ motionBlur: v })} />
        <BigDial label="Noise" value={noise}
          onCommit={(v) => onChange({ noise: v })} />
        <BigDial label="Colour distortion" value={colourDistortion}
          onCommit={(v) => onChange({ colourDistortion: v })} />
        <BigDial label="Chromatic aberration" value={chromaticAberration}
          onCommit={(v) => onChange({ chromaticAberration: v })} />
        <BigDial label="Bit depth" value={bitDepth}
          onCommit={(v) => onChange({ bitDepth: v })} />
        <BigDial label="Lens distortion" value={lensDistortion}
          onCommit={(v) => onChange({ lensDistortion: v })} />
        <BigDial label="Lens glare" value={lensGlare}
          onCommit={(v) => onChange({ lensGlare: v })} />
        <BigDial label="Low resolution" value={lowResolution}
          onCommit={(v) => onChange({ lowResolution: v })} />
        <BigDial label="Pixelation" value={pixelation}
          onCommit={(v) => onChange({ pixelation: v })} />
      </div>
    </div>
  );
}

// ─── Random block occlusion block ──────────────────────────────
// Mirrors CameraSensorBlock visually but only ships block_size +
// show_outlines to /augment/preview. Pool restricted to imports
// (segmentation lives on the dataset side, references don't carry
// detections).
function RandomBlockOcclusionBlock({
  projectId,
  previewSources,
  size,
  onChange,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  size: number;
  onChange: (v: number) => void;
}) {
  const eligible = useMemo(
    () => previewSources.filter((s) => s.source === "import"),
    [previewSources],
  );
  const [sourceIdx, setSourceIdx] = useState<number>(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              // Camera dials stay at 0, this preview is about
              // showing the block occlusion + outlines only.
              block_size: size,
              show_outlines: true,
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
    // activeSource is a fresh object reference on every parent
    // render (previewSources gets rebuilt from refs / imports
    // state). Depend on stable scalar fields so the debounce
    // doesn't re-arm on every no-op render.
  }, [projectId, activeSource?.source, activeSource?.filename, size, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  return (
    <div className="grid gap-4 md:grid-cols-2 md:items-center w-full">
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Random block occlusion preview"
              className="block w-full h-auto"
              draggable={false}
            />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0
                  ? "Add dataset images with detections to see a preview"
                  : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
            title="Pick a new random image"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>
      <div className="grid gap-3 content-start">
        <BigDial
          label="Block size"
          value={Math.min(size, 40)}
          onCommit={onChange}
          min={0}
          max={40}
          step={1}
          suffix="%"
        />
      </div>
    </div>
  );
}

// ─── Perspective warp block ────────────────────────────────────
// Same shell as Camera & Random-block; one 0..10 strength dial,
// one Random image button. References + dataset both eligible
// since this preview doesn't need detections.
// ─── Generic single-strength preview block ─────────────────────
// Shared shell used by Perspective warp, Hue shift and Lighting
// variation, anywhere the augmentation is a single 0..10 dial
// against a live preview image. Builds the request body via a
// caller-supplied `buildBody` callback so each block can pick
// which backend param it drives.
function SimpleStrengthBlock({
  projectId,
  previewSources,
  strength,
  onChange,
  dialLabel = "Strength",
  buildBody,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  strength: number;
  onChange: (v: number) => void;
  dialLabel?: string;
  buildBody: (source: AugmentPreviewSource, strength: number, seed: number) => Record<string, unknown>;
}) {
  const eligible = previewSources;
  const [sourceIdx, setSourceIdx] = useState(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    if (strength <= 0) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(activeSource.preview);
      setPreviewError(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              ...buildBody(activeSource, strength, seed),
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
    // buildBody is intentionally not a dep, callers either keep
    // it stable or accept that strength/seed change drives the
    // refetch (which it always does).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, activeSource?.source, activeSource?.filename, strength, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  return (
    <div className="grid gap-4 md:grid-cols-2 md:items-center w-full">
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt={`${dialLabel} preview`} className="block w-full h-auto" draggable={false} />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0 ? "Add images to see a preview" : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>
      <div className="grid gap-3 content-start">
        <BigDial label={dialLabel} value={strength} onCommit={onChange} />
      </div>
    </div>
  );
}

function PerspectiveWarpBlock({
  projectId,
  previewSources,
  strength,
  onChange,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  strength: number;
  onChange: (v: number) => void;
}) {
  const eligible = previewSources;
  const [sourceIdx, setSourceIdx] = useState(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    if (strength <= 0) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(activeSource.preview);
      setPreviewError(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              perspective_warp: strength,
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
  }, [projectId, activeSource?.source, activeSource?.filename, strength, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  return (
    <div className="grid gap-4 md:grid-cols-2 md:items-center w-full">
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Perspective warp preview" className="block w-full h-auto" draggable={false} />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0 ? "Add images to see a preview" : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>
      <div className="grid gap-3 content-start">
        <BigDial label="Strength" value={strength} onCommit={onChange} />
      </div>
    </div>
  );
}

// ─── Scale & rotation block ────────────────────────────────────
// Two DualSliders feed (min, max) ranges for scale and rotation;
// backend samples one value from each range per preview using the
// seed so the preview is stable across slider drags.
function ScaleRotationBlock({
  projectId,
  previewSources,
  scaleMin,
  scaleMax,
  rotMin,
  rotMax,
  onChange,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  scaleMin: number;
  scaleMax: number;
  rotMin: number;
  rotMax: number;
  onChange: (patch: Partial<{
    scaleMin: number; scaleMax: number; rotMin: number; rotMax: number;
  }>) => void;
}) {
  const eligible = previewSources;
  const [sourceIdx, setSourceIdx] = useState(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  const isIdentity = scaleMin === 1.0 && scaleMax === 1.0 && rotMin === 0 && rotMax === 0;

  useEffect(() => {
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    if (isIdentity) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(activeSource.preview);
      setPreviewError(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              scale_min: scaleMin,
              scale_max: scaleMax,
              rot_min: rotMin,
              rot_max: rotMax,
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 200);
    return () => window.clearTimeout(t);
  }, [projectId, activeSource?.source, activeSource?.filename, isIdentity, scaleMin, scaleMax, rotMin, rotMax, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  return (
    <div className="grid gap-4 md:grid-cols-2 md:items-center w-full">
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Scale & rotation preview" className="block w-full h-auto" draggable={false} />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0 ? "Add images to see a preview" : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>
      <div className="grid gap-3 content-start">
        <p className="text-[11px] leading-snug text-foreground/55">
          Each augmented image is randomly scaled and rotated within
          the ranges below, every copy picks a fresh value.
        </p>
        <DualSlider
          label="Scale range"
          min={0.7}
          max={1.3}
          step={0.05}
          values={[Math.max(0.7, scaleMin), Math.min(1.3, scaleMax)]}
          onChange={([mn, mx]) => onChange({ scaleMin: mn, scaleMax: mx })}
          format={(v) => `${v.toFixed(2)}×`}
        />
        <DualSlider
          label="Rotation range"
          min={-45}
          max={45}
          step={1}
          values={[rotMin, rotMax]}
          onChange={([mn, mx]) => onChange({ rotMin: mn, rotMax: mx })}
          format={(v) => `${v}°`}
        />
      </div>
    </div>
  );
}

// Canvas-based resize + JPEG compress so background uploads stay
// under ~50 KB on the wire (longest edge capped at 1024 px). Keeps
// alpha-bearing files as PNG only when shrinking; otherwise we ladder
// JPEG quality down until we fit the target.
const BG_MAX_EDGE = 1024;
const BG_MAX_BYTES = 50 * 1024;
async function compressBackgroundFile(file: File): Promise<File> {
  if (!file.type.startsWith("image/")) return file;
  const bitmap = await (async () => {
    try {
      return await createImageBitmap(file);
    } catch {
      const url = URL.createObjectURL(file);
      try {
        const img = new Image();
        img.src = url;
        await new Promise<void>((res, rej) => {
          img.onload = () => res();
          img.onerror = () => rej(new Error("decode failed"));
        });
        return img as unknown as ImageBitmap;
      } finally {
        URL.revokeObjectURL(url);
      }
    }
  })();
  const w0 = (bitmap as ImageBitmap).width;
  const h0 = (bitmap as ImageBitmap).height;
  const longest = Math.max(w0, h0);
  const scale = longest > BG_MAX_EDGE ? BG_MAX_EDGE / longest : 1;
  const w = Math.max(1, Math.round(w0 * scale));
  const h = Math.max(1, Math.round(h0 * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return file;
  ctx.drawImage(bitmap as CanvasImageSource, 0, 0, w, h);
  for (const q of [0.85, 0.75, 0.65, 0.55, 0.45]) {
    const blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob(res, "image/jpeg", q),
    );
    if (!blob) continue;
    if (blob.size <= BG_MAX_BYTES || q === 0.45) {
      const base = file.name.replace(/\.[^.]+$/, "");
      return new File([blob], `${base}.jpg`, { type: "image/jpeg" });
    }
  }
  return file;
}

// ─── Background randomisation block ─────────────────────────────
// User uploads up to MAX_BACKGROUND_IMAGES backgrounds; the live
// preview takes a random dataset image (must have detections),
// keeps the pixels inside the detection polygons, and replaces
// everything outside with one of the uploaded backgrounds.
// Backgrounds upload straight to the backend, no label modal,
// no segmentation. Backend resizes and stores them; this block
// references them by id.
function BackgroundRandomisationBlock({
  projectId,
  previewSources,
  backgrounds,
  onBackgrounds,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  backgrounds: BackgroundEntry[];
  onBackgrounds: (next: BackgroundEntry[]) => void;
}) {
  // Dataset-only: references don't carry detections so the
  // foreground / background split has no anchor.
  const eligible = useMemo(
    () => previewSources.filter((s) => s.source === "import"),
    [previewSources],
  );
  const [sourceIdx, setSourceIdx] = useState(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  // Upload state. No modal, just file picker → POST → append.
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const bgKey = useMemo(() => backgrounds.map((b) => b.id).join("|"), [backgrounds]);

  useEffect(() => {
    if (!projectId || !activeSource || backgrounds.length === 0) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              background_ids: backgrounds.map((b) => b.id),
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, activeSource?.source, activeSource?.filename, bgKey, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  // Upload one file and return the new BackgroundEntry. Doesn't
  // touch state directly, the caller owns appending so multi-file
  // selections don't race against a stale `backgrounds` closure
  // (each iteration would otherwise overwrite the previous append
  // because setState is async).
  const uploadOneRaw = async (f: File): Promise<BackgroundEntry | null> => {
    if (!projectId) return null;
    const compressed = await compressBackgroundFile(f);
    const fd = new FormData();
    fd.append("image", compressed);
    const r = await apiFetch(
      `/api/v2/projects/${projectId}/augment/background/upload`,
      { method: "POST", body: fd },
    );
    if (!r.ok) throw new Error(`http ${r.status}: ${await r.text()}`);
    const data = (await r.json()) as { background_id: string };
    return {
      id: data.background_id,
      previewUrl: `${API}/api/v2/projects/${projectId}/augment/backgrounds/${data.background_id}`,
    };
  };

  const uploadMany = async (files: File[]) => {
    if (!projectId) return;
    setUploading(true);
    setUploadError(null);
    try {
      // Snapshot the current backgrounds once; we accumulate into a
      // local copy and onBackgrounds once at the end so React's
      // batched state update commits the full set.
      const acc = backgrounds.slice();
      for (const f of files) {
        if (acc.length >= MAX_BACKGROUND_IMAGES) break;
        try {
          const entry = await uploadOneRaw(f);
          if (entry) acc.push(entry);
        } catch (e) {
          setUploadError(e instanceof Error ? e.message : String(e));
        }
      }
      if (acc.length > MAX_BACKGROUND_IMAGES) acc.length = MAX_BACKGROUND_IMAGES;
      onBackgrounds(acc);
    } finally {
      setUploading(false);
    }
  };

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    await uploadMany(files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files || []).filter((f) => f.type.startsWith("image/"));
    await uploadMany(files);
  };

  const removeAt = async (idx: number) => {
    const entry = backgrounds[idx];
    if (!entry) return;
    onBackgrounds(backgrounds.filter((_, i) => i !== idx));
    if (projectId) {
      try {
        await apiFetch(
          `/api/v2/projects/${projectId}/augment/backgrounds/${entry.id}`,
          { method: "DELETE" },
        );
      } catch { /* best effort */ }
    }
  };

  return (
    <div
      className={[
        "grid gap-4 md:grid-cols-2 w-full",
        backgrounds.length > 0 ? "md:items-center" : "md:items-start",
      ].join(" ")}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={onFile}
        className="hidden"
      />

      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {backgrounds.length === 0 ? (
            <div className="grid place-items-center py-16 px-4 text-center">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                Upload a background to start
              </span>
            </div>
          ) : previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Background randomisation preview" className="block w-full h-auto" draggable={false} />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0 ? "Add dataset images with detections to see a preview" : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0 || backgrounds.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>

      <div className="grid gap-2 content-start">
        {backgrounds.length > 0 && (
          <div className="grid grid-cols-3 gap-2">
            {backgrounds.map((entry, idx) => (
              <div
                key={entry.id}
                className="relative aspect-square rounded-md overflow-hidden border border-[var(--line)] bg-[var(--panel)]"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={entry.previewUrl}
                  alt={`Background ${idx + 1}`}
                  className="absolute inset-0 w-full h-full object-cover"
                  draggable={false}
                />
                <button
                  type="button"
                  onClick={() => void removeAt(idx)}
                  aria-label="Remove background"
                  className="absolute top-1 right-1 h-5 w-5 grid place-items-center rounded-md bg-black/65 text-white/80 hover:bg-black/85 hover:text-white text-xs"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        {backgrounds.length < MAX_BACKGROUND_IMAGES && (
          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); } }}
            onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={[
              "rounded-md border bg-[var(--panel)] px-4 text-center text-sm transition-colors cursor-pointer outline-none",
              // Empty state: stretch tall so the drop zone matches
              // the height of the "Upload a background to start"
              // preview placeholder on the left. Once at least one
              // background exists the zone shrinks back to a
              // standard utility row.
              backgrounds.length === 0
                ? "py-16 grid place-items-center"
                : "py-6",
              dragOver
                ? "border-[var(--accent)] bg-[var(--surface-hover)] text-[var(--foreground)]"
                : "border-[var(--line)] text-[var(--fg-muted)] hover:border-[var(--line-strong)] hover:text-foreground",
              uploading ? "opacity-60 cursor-wait" : "",
            ].join(" ")}
          >
            <div>
            {uploading
              ? "Uploading…"
              : backgrounds.length === 0
                ? "+ Upload background image"
                : "+ Add another background"}
            <div className="text-[10px] uppercase tracking-wider text-foreground/35 mt-1">
              or drop one here · up to {MAX_BACKGROUND_IMAGES}
            </div>
            </div>
          </div>
        )}
        {uploadError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {uploadError}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Object overlay block ──────────────────────────────────────
// Same preview shell as Camera & Random-block, plus an upload
// affordance. On file pick we open a label-prompt modal; on
// confirm we POST the file + label to the SAM3-backed segment
// endpoint which returns an overlay_id. The live preview then
// composites that overlay onto a random dataset image, with a
// scale dial driving the longest-edge size.
//
// Currently NOT rendered, the augmentation is gated behind a
// "Coming soon" pill while we speed up SAM3 + compositing. The
// function stays in the file (and is referenced once below via
// `void ObjectOverlayBlock`) so re-enabling later is a single
// uncomment in the SubAugRow up above.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function ObjectOverlayBlock({
  projectId,
  previewSources,
  overlays,
  scale,
  onOverlays,
  onScale,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  overlays: ObjectOverlayEntry[];
  scale: number;
  onOverlays: (next: ObjectOverlayEntry[]) => void;
  onScale: (v: number) => void;
}) {
  // Dataset images only, references don't carry detections so
  // the 50%-coverage constraint can't bind against them.
  const eligible = useMemo(
    () => previewSources.filter((s) => s.source === "import"),
    [previewSources],
  );
  const [sourceIdx, setSourceIdx] = useState(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = eligible[sourceIdx] ?? null;

  // Upload + label-prompt modal state. `pendingFile` is set when
  // the user picks a file; the modal reads it and clears it on
  // confirm/cancel. `replaceIndex` tracks whether the new overlay
  // should append to the list (null) or replace a specific slot.
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingPreview, setPendingPreview] = useState<string | null>(null);
  const [labelDraft, setLabelDraft] = useState("");
  const [segmenting, setSegmenting] = useState(false);
  const [segError, setSegError] = useState<string | null>(null);
  const [replaceIndex, setReplaceIndex] = useState<number | null>(null);

  // Augmentation preview state.
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  // Fetch the composite. Skipped until we have a project, a source,
  // and at least one segmented overlay.
  const overlayKey = useMemo(() => overlays.map((o) => o.id).join("|"), [overlays]);
  useEffect(() => {
    if (!projectId || !activeSource || overlays.length === 0 || scale <= 0) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
    const myReq = ++requestIdRef.current;
    const t = window.setTimeout(async () => {
      setPreviewBusy(true);
      setPreviewError(null);
      try {
        const r = await apiFetch(
          `/api/v2/projects/${projectId}/augment/preview`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: activeSource.source,
              filename: activeSource.filename,
              overlay_ids: overlays.map((o) => o.id),
              overlay_scale: scale,
              seed,
            }),
          },
        );
        if (!r.ok) throw new Error(`http ${r.status}`);
        const blob = await r.blob();
        if (myReq !== requestIdRef.current) return;
        const url = URL.createObjectURL(blob);
        if (lastBlobRef.current) URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = url;
        setPreviewUrl(url);
      } catch (e) {
        if (myReq !== requestIdRef.current) return;
        setPreviewError(e instanceof Error ? e.message : String(e));
      } finally {
        if (myReq === requestIdRef.current) setPreviewBusy(false);
      }
    }, 140);
    return () => window.clearTimeout(t);
    // overlayKey carries the dependency on the overlays list, using
    // it instead of `overlays` itself keeps reference-equality
    // stable across re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, activeSource?.source, activeSource?.filename, overlayKey, scale, seed]);

  useEffect(() => () => {
    if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
  }, []);

  // Modal preview lifecycle, own blob URL released when the
  // pending file is cleared.
  useEffect(() => {
    if (!pendingFile) {
      if (pendingPreview) URL.revokeObjectURL(pendingPreview);
      setPendingPreview(null);
      return;
    }
    const url = URL.createObjectURL(pendingFile);
    setPendingPreview(url);
    return () => URL.revokeObjectURL(url);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingFile]);

  const pickRandom = useCallback(() => {
    if (eligible.length === 0) return;
    if (eligible.length === 1) {
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
    let next = sourceIdx;
    while (next === sourceIdx) next = Math.floor(Math.random() * eligible.length);
    setSourceIdx(next);
    setSeed(Math.floor(Math.random() * 1_000_000));
  }, [eligible.length, sourceIdx]);

  const openPicker = (idx: number | null = null) => {
    setReplaceIndex(idx);
    fileInputRef.current?.click();
  };
  const startFlowWithFile = (f: File | null) => {
    if (!f) return;
    setLabelDraft("");
    setSegError(null);
    setPendingFile(f);
  };
  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    startFlowWithFile(e.target.files?.[0] ?? null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Drag-and-drop on the upload affordance. Accepts the first
  // image-typed file the OS hands over. Visual state via
  // `dragOver` so the dashed border lights up while the user
  // hovers a file over the box.
  const [dragOver, setDragOver] = useState(false);
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files || []);
    const first = files.find((f) => f.type.startsWith("image/")) ?? files[0] ?? null;
    startFlowWithFile(first);
  };

  const submitSegment = async () => {
    if (!projectId || !pendingFile) return;
    const lab = labelDraft.trim();
    if (!lab) {
      setSegError("Type a label first.");
      return;
    }
    setSegmenting(true);
    setSegError(null);
    try {
      const fd = new FormData();
      fd.append("image", pendingFile);
      fd.append("label", lab);
      const r = await apiFetch(
        `/api/v2/projects/${projectId}/augment/object_overlay/segment`,
        { method: "POST", body: fd },
      );
      if (!r.ok) throw new Error(`http ${r.status}: ${await r.text()}`);
      const data = (await r.json()) as { overlay_id: string; label?: string };
      const id = data.overlay_id;
      const entry: ObjectOverlayEntry = {
        id,
        label: data.label || lab,
        previewUrl: `${API}/api/v2/projects/${projectId}/augment/overlays/${id}`,
      };
      const next = overlays.slice();
      if (replaceIndex != null && replaceIndex >= 0 && replaceIndex < next.length) {
        next[replaceIndex] = entry;
      } else {
        // Append, but never exceed the per-block cap.
        next.push(entry);
        if (next.length > MAX_OBJECT_OVERLAYS) next.length = MAX_OBJECT_OVERLAYS;
      }
      onOverlays(next);
      setPendingFile(null);
      setLabelDraft("");
      setReplaceIndex(null);
    } catch (e) {
      setSegError(e instanceof Error ? e.message : String(e));
    } finally {
      setSegmenting(false);
    }
  };

  return (
    <div
      className={[
        "grid gap-4 md:grid-cols-2 w-full",
        // Top-aligned when there's no overlay yet so the upload
        // box on the right sits level with the "Upload an image
        // to start" placeholder on the left. Vertically centred
        // once any overlay exists, matching Camera & sensor.
        overlays.length > 0 ? "md:items-center" : "md:items-start",
      ].join(" ")}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={onFile}
        className="hidden"
      />

      {/* Preview pane, composite of selected dataset image + overlays. */}
      <div className="space-y-3">
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] overflow-hidden relative">
          {overlays.length === 0 ? (
            <div className="grid place-items-center py-16 px-4 text-center">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                Upload an image to start
              </span>
            </div>
          ) : previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Object overlay preview"
              className="block w-full h-auto"
              draggable={false}
            />
          ) : (
            <div className="grid place-items-center py-16">
              <span className="text-xs uppercase tracking-wider text-foreground/35">
                {eligible.length === 0
                  ? "Add dataset images to see a preview"
                  : "Loading…"}
              </span>
            </div>
          )}
          {previewBusy && (
            <div className="absolute top-2 right-2 grid place-items-center h-6 w-6 rounded-full bg-foreground/70">
              <svg className="h-3.5 w-3.5 animate-spin text-background" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
                <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <button
            type="button"
            onClick={pickRandom}
            disabled={eligible.length === 0 || overlays.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Random image
          </button>
          <span className="text-[10px] uppercase tracking-wider font-mono text-foreground/35 truncate max-w-[16rem]">
            {activeSource ? activeSource.filename : ""}
          </span>
        </div>
        {previewError && (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
            {previewError}
          </div>
        )}
      </div>

      {/* Right column: per-overlay row, "+ Add" until cap, scale dial. */}
      <div className="grid gap-2 content-start">
        {overlays.map((entry, idx) => (
          <div
            key={entry.id}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 flex items-center gap-3"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={entry.previewUrl}
              alt={entry.label}
              className="h-10 w-10 rounded-md object-contain bg-[var(--panel)]"
            />
            <div className="min-w-0 flex-1">
              <div className="pk-micro">
                Overlay {idx + 1}
              </div>
              <div className="text-sm text-foreground/90 truncate">{entry.label}</div>
            </div>
            <button
              type="button"
              onClick={() => openPicker(idx)}
              className="text-[10px] uppercase tracking-wider text-foreground/55 hover:text-foreground"
            >
              Replace
            </button>
            <button
              type="button"
              onClick={() => onOverlays(overlays.filter((_, i) => i !== idx))}
              className="text-[10px] uppercase tracking-wider text-foreground/40 hover:text-[var(--bad)]"
            >
              Remove
            </button>
          </div>
        ))}
        {overlays.length < MAX_OBJECT_OVERLAYS && (
          <div
            role="button"
            tabIndex={0}
            onClick={() => openPicker(null)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(null); } }}
            onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={[
              "rounded-md border bg-[var(--panel)] px-4 py-6 text-center text-sm transition-colors cursor-pointer outline-none",
              dragOver
                ? "border-[var(--accent)] bg-[var(--surface-hover)] text-[var(--foreground)]"
                : "border-[var(--line)] text-[var(--fg-muted)] hover:border-[var(--line-strong)] hover:text-foreground",
            ].join(" ")}
          >
            {overlays.length === 0 ? "+ Upload overlay image" : "+ Add another overlay"}
            <div className="text-[10px] uppercase tracking-wider text-foreground/35 mt-1">
              or drop one here · up to {MAX_OBJECT_OVERLAYS}
            </div>
          </div>
        )}
        <BigDial
          label="Scale"
          value={scale}
          onCommit={onScale}
          min={0.05}
          max={1}
          step={0.01}
          suffix="×"
        />
      </div>

      {/* Label-prompt modal. Backdrop blur+darken; image preview;
          label input; SAM3 confirm. */}
      {pendingFile && pendingPreview && typeof window !== "undefined" &&
        createPortal(
          <div
            className="fixed inset-0 z-[700] grid place-items-center px-4"
            style={{ background: "var(--backdrop)", backdropFilter: "blur(8px)" }}
            onClick={() => { if (!segmenting) setPendingFile(null); }}
            role="dialog"
            aria-modal="true"
          >
            <div
              className="w-full max-w-md rounded-md pk-glass p-5 grid gap-4"
              onClick={(e) => e.stopPropagation()}
              style={{ animation: "objectOverlayPopIn 220ms cubic-bezier(0.2, 0.7, 0.2, 1) both" }}
            >
              <div className="flex items-baseline justify-between">
                <h3 className="pk-micro" style={{ color: "var(--foreground)" }}>Segment object</h3>
                <button
                  type="button"
                  onClick={() => setPendingFile(null)}
                  disabled={segmenting}
                  className="text-foreground/45 hover:text-foreground text-xl leading-none disabled:opacity-40"
                  aria-label="Close"
                >×</button>
              </div>
              <div className="rounded-md overflow-hidden bg-[var(--panel)] max-h-[50vh] grid place-items-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={pendingPreview} alt="" className="block max-h-[50vh] w-auto h-auto" />
              </div>
              <div className="grid gap-1.5">
                <label htmlFor="object-overlay-label" className="pk-micro">
                  What object should PixelKit cut out?
                </label>
                <input
                  id="object-overlay-label"
                  autoFocus
                  value={labelDraft}
                  onChange={(e) => setLabelDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !segmenting && labelDraft.trim()) {
                      e.preventDefault();
                      void submitSegment();
                    }
                  }}
                  placeholder="e.g. cat, red sports car, traffic cone"
                  className="rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] focus:outline-none focus:border-[var(--line-strong)]"
                />
              </div>
              {segError && (
                <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11px] text-[var(--bad)]">
                  {segError}
                </div>
              )}
              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setPendingFile(null)}
                  disabled={segmenting}
                  className="rounded-md border border-[var(--line)] px-3.5 py-1.5 text-[13px] font-medium text-[var(--fg-soft)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground disabled:opacity-40"
                >Cancel</button>
                <button
                  type="button"
                  onClick={() => void submitSegment()}
                  disabled={segmenting || !labelDraft.trim()}
                  className="rounded-md bg-[var(--accent)] text-[var(--accent-contrast)] px-4 py-1.5 text-[13px] font-medium hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed"
                >{segmenting ? "Segmenting…" : "Segment"}</button>
              </div>
              <style>{`
                @keyframes objectOverlayPopIn {
                  from { opacity: 0; transform: translateY(8px) scale(0.97); }
                  to   { opacity: 1; transform: translateY(0) scale(1); }
                }
              `}</style>
            </div>
          </div>,
          document.body,
        )
      }
    </div>
  );
}

// Full-width dial with the value pinned to the right. Used inside
// Camera & sensor (0..10 strength) and Random block occlusion
// (0..60 % coverage). Bigger than DialInput so it dominates the
// right column.
//
// The committed value only fires onCommit on mouse-up / touch-end
// / blur / Enter, dragging just updates the visual position
// locally. That keeps the GPU bake (one round-trip per commit)
// off the dragging path so the slider stays buttery.
function BigDial({
  label,
  value,
  onCommit,
  min = 0,
  max = 10,
  step = 0.1,
  suffix = "",
}: {
  label: string;
  value: number;
  onCommit: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
}) {
  // Local visual state. Resets to the parent value when the parent
  // commits a change (e.g. Random image or Reset clears the dials).
  const [draft, setDraft] = useState<number>(value);
  const draggingRef = useRef(false);
  useEffect(() => {
    if (!draggingRef.current) setDraft(value);
  }, [value]);

  const commit = (raw: string | number) => {
    const n = typeof raw === "string" ? Number(raw) : raw;
    if (!Number.isFinite(n)) return;
    const clamped = Math.max(min, Math.min(max, n));
    setDraft(clamped);
    if (clamped !== value) onCommit(clamped);
  };

  // Active when the dial is off its rest position. Most dials are
  // unidirectional with rest=min=0, but the lens-distortion dial is
  // bidirectional (-10..+10, rest = 0). Comparing to 0 directly
  // works for both — value > min and value !== 0 coincide when
  // min === 0.
  const active = draft !== 0;
  const decimals = step < 1 ? Math.min(2, Math.ceil(-Math.log10(step))) : 0;
  return (
    <div className={[
      "rounded-md px-4 py-3 transition-colors",
      active ? "bg-[var(--surface-hover)]" : "bg-[var(--panel)]",
    ].join(" ")}>
      <div className="flex items-center justify-between mb-1">
        <span className={[
          "font-mono text-[11px] uppercase tracking-[0.12em] transition-colors",
          active ? "text-foreground/85" : "text-[var(--fg-dim)]",
        ].join(" ")}>{label}</span>
        <span className="text-sm text-foreground/85 tabular-nums font-mono">{draft.toFixed(decimals)}{suffix}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={draft}
        // Live update of the visual position only, no parent
        // notification yet, so the GPU pipeline stays idle.
        onChange={(e) => setDraft(Number(e.target.value))}
        // Commit on release / keyboard step. mouseup/touchend
        // cover the drag path; change fires on keyboard arrows;
        // blur is a safety net if the input loses focus mid-drag.
        onPointerDown={() => { draggingRef.current = true; }}
        onPointerUp={(e) => { draggingRef.current = false; commit((e.target as HTMLInputElement).value); }}
        onPointerCancel={() => { draggingRef.current = false; setDraft(value); }}
        onKeyUp={(e) => commit((e.target as HTMLInputElement).value)}
        onBlur={(e) => { draggingRef.current = false; commit(e.target.value); }}
        className="w-full accent-[var(--accent)]"
        aria-label={label}
      />
    </div>
  );
}

// CategoryBlock, collapsible top-level category. Header has a
// checkbox (the whole-category enable), the title + description,
// the frequency toggle (visible only when the category is on),
// and a chevron that expands/collapses the contained sub-rows.
function CategoryBlock({
  title,
  description,
  enabled,
  onToggle,
  frequency,
  onFrequency,
  open,
  onOpen,
  children,
}: {
  title: string;
  description?: string;
  enabled: boolean;
  onToggle: () => void;
  frequency: Frequency;
  onFrequency: (f: Frequency) => void;
  open: boolean;
  onOpen: () => void;
  children: React.ReactNode;
}) {
  // Auto-expand when the category gets enabled, feels broken
  // otherwise since users tick the checkbox and nothing visible
  // happens until they also click the chevron. The grid-rows
  // transition below handles the actual animation.
  const wasEnabledRef = useRef(enabled);
  useEffect(() => {
    if (enabled && !wasEnabledRef.current && !open) onOpen();
    wasEnabledRef.current = enabled;
  }, [enabled, open, onOpen]);

  const handleToggle = () => {
    onToggle();
  };

  return (
    <div
      className={[
        "rounded-md border border-[var(--line)] transition-colors",
        enabled ? "bg-[var(--surface-hover)]" : "bg-[var(--panel)]",
      ].join(" ")}
    >
      <div className="flex items-center gap-3 px-5 py-4">
        <Checkbox checked={enabled} onChange={handleToggle} ariaLabel={`${title} enabled`} />
        <button
          type="button"
          onClick={onOpen}
          className="flex-1 text-left flex items-center gap-3 group"
          aria-expanded={open}
        >
          <div className="min-w-0 flex-1">
            <div className="text-base font-medium text-[var(--foreground)]">{title}</div>
            {description && (
              <div className="text-xs text-foreground/45 mt-0.5 truncate">{description}</div>
            )}
          </div>
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4 text-foreground/45 group-hover:text-foreground/80 shrink-0"
            fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            style={{
              transform: open ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 220ms cubic-bezier(0.2,0.7,0.2,1)",
            }}
            aria-hidden
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        {enabled && (
          <FrequencyToggle value={frequency} onChange={onFrequency} compact />
        )}
      </div>
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="px-5 pb-5 pt-5 border-t border-[var(--line-soft)] space-y-3">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

// SubAugRow, an individual augmentation under a category. Checkbox
// on the left, label + description in the middle, controls (passed
// via `children`) take the remaining width when the sub is enabled,
// and the frequency toggle pins to the right.
function SubAugRow({
  label,
  description,
  enabled,
  onToggle,
  frequency,
  onFrequency,
  children,
  rightBadge,
}: {
  label: string;
  description?: string;
  enabled: boolean;
  onToggle: (v: boolean) => void;
  frequency: Frequency;
  onFrequency: (f: Frequency) => void;
  children?: React.ReactNode;
  /** Optional pill rendered where the frequency toggle would
      otherwise sit. Used to flag rows that are temporarily
      "Coming soon", the toggle becomes a no-op stub and the
      badge replaces the active-row controls. */
  rightBadge?: React.ReactNode;
}) {
  const isComingSoon = !!rightBadge;
  return (
    <div className={[
      // Standard flat sub-panel: hairline border + panel step, same
      // surface recipe as every other in-flow card.
      "rounded-md border border-[var(--line)] bg-[var(--panel)] transition-colors",
      isComingSoon ? "opacity-70" : "",
    ].join(" ")}>
      <div className="flex items-start gap-3 px-4 py-3">
        <div className="pt-0.5">
          <Checkbox
            checked={enabled}
            onChange={() => { if (!isComingSoon) onToggle(!enabled); }}
            ariaLabel={`${label} enabled`}
          />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-[var(--foreground)]">{label}</div>
          {description && (
            <div className="text-[11px] text-foreground/45 mt-0.5">{description}</div>
          )}
        </div>
        {rightBadge ? (
          <div className="shrink-0">{rightBadge}</div>
        ) : (
          enabled && (
            <div className="shrink-0">
              <FrequencyToggle value={frequency} onChange={onFrequency} compact />
            </div>
          )
        )}
      </div>
      {enabled && !isComingSoon && (
        <div className="px-4 pb-4 pt-3 border-t border-[var(--line-soft)]">
          {children}
        </div>
      )}
    </div>
  );
}

// NestedToggleRow, a smaller row used for the environmental-effect
// subitems (dust, rain, fog, snow). No description, no body, just
// a checkbox + label + per-item frequency.
// Currently unused, Environmental effects (its only caller)
// sits behind a Coming-soon pill. Kept around for re-enabling
// later, same pattern as ImageUploadGrid + DialInput.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function NestedToggleRow({
  label,
  enabled,
  onToggle,
  frequency,
  onFrequency,
}: {
  label: string;
  enabled: boolean;
  onToggle: (v: boolean) => void;
  frequency: Frequency;
  onFrequency: (f: Frequency) => void;
}) {
  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-md border border-[var(--line)] bg-[var(--panel)]">
      <Checkbox checked={enabled} onChange={() => onToggle(!enabled)} ariaLabel={`${label} enabled`} />
      <span className="text-sm text-foreground/85 flex-1">{label}</span>
      {enabled && <FrequencyToggle value={frequency} onChange={onFrequency} compact />}
    </div>
  );
}

// FrequencyToggle, All / Random / Separate three-way pill. `compact`
// drops the padding when sitting in tight rows next to a checkbox.
function FrequencyToggle({
  value,
  onChange,
  compact = false,
}: {
  value: Frequency;
  onChange: (v: Frequency) => void;
  compact?: boolean;
}) {
  const items: { value: Frequency; label: string }[] = [
    { value: "all", label: "All" },
    { value: "random", label: "Random" },
  ];
  return (
    <div className={[
      "inline-flex rounded-md border border-[var(--line)] bg-[var(--surface)]",
      compact ? "p-0.5" : "p-1",
    ].join(" ")}>
      {items.map((it) => {
        const active = value === it.value;
        return (
          <button
            key={it.value}
            type="button"
            onClick={(e) => { e.stopPropagation(); onChange(it.value); }}
            className={[
              "rounded-[4px] px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider transition-colors",
              active ? "bg-[var(--surface-hover)] text-[var(--foreground)]" : "text-foreground/55 hover:text-foreground",
            ].join(" ")}
          >
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

// SegmentedToggle, generic single-row segmented control. Used for
// the per-image augmentation count.
function SegmentedToggle<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-[var(--line)] bg-[var(--surface)] p-1">
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={[
              "rounded-[4px] px-4 py-1 text-[12px] font-medium uppercase tracking-wider transition-colors",
              active ? "bg-[var(--surface-hover)] text-[var(--foreground)]" : "text-foreground/55 hover:text-foreground",
            ].join(" ")}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function Checkbox({
  checked,
  onChange,
  ariaLabel,
}: {
  checked: boolean;
  onChange: () => void;
  ariaLabel: string;
}) {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onChange(); }}
      aria-checked={checked}
      role="checkbox"
      aria-label={ariaLabel}
      className={[
        "h-5 w-5 rounded-md border grid place-items-center transition-colors shrink-0",
        checked
          ? "bg-[var(--accent)] border-[var(--accent)] text-[var(--accent-contrast)]"
          : "border-foreground/20 bg-foreground/[0.02] text-transparent hover:border-foreground/40",
      ].join(" ")}
    >
      <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <polyline points="5 12 10 17 19 7" />
      </svg>
    </button>
  );
}

// Smaller dial used by SubAugRows that don't need the live-preview
// shell. Currently unused now that every dial-driven sub has its
// own SimpleStrengthBlock or bespoke block; kept so future
// sub-augs (or a future "compact" mode) can reuse it.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function DialInput({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3 w-full">
      <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--fg-dim)] w-20 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-[var(--accent)]"
      />
      <span className="text-sm text-[var(--foreground)] tabular-nums w-8 text-right">{value}</span>
    </div>
  );
}

// DualSlider, two thumbs constrained so min ≤ max. Implemented as
// two overlaid native range inputs to keep keyboard accessibility +
// styling consistent with the single-value Dial above.
function DualSlider({
  label,
  min,
  max,
  step,
  values,
  onChange,
  format,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  values: [number, number];
  onChange: (v: [number, number]) => void;
  format: (v: number) => string;
}) {
  const [a, b] = values;
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1.5">
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--fg-dim)]">{label}</span>
        <span className="text-[11px] text-foreground/55 tabular-nums">
          {format(a)} <span className="text-foreground/30">→</span> {format(b)}
        </span>
      </div>
      <div className="relative h-6 flex items-center">
        <div className="absolute left-0 right-0 h-1 rounded-full bg-foreground/10" />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={a}
          onChange={(e) => {
            const next = Math.min(Number(e.target.value), b);
            onChange([next, b]);
          }}
          className="absolute inset-0 w-full pointer-events-none appearance-none bg-transparent dual-range"
          style={{ zIndex: a > max - (max - min) * 0.05 ? 3 : 2 }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={b}
          onChange={(e) => {
            const next = Math.max(Number(e.target.value), a);
            onChange([a, next]);
          }}
          className="absolute inset-0 w-full pointer-events-none appearance-none bg-transparent dual-range"
          style={{ zIndex: 3 }}
        />
      </div>
      <style>{`
        .dual-range::-webkit-slider-thumb {
          pointer-events: auto;
          -webkit-appearance: none;
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 9999px;
          background: var(--accent);
          border: 2px solid var(--background);
          box-shadow: 0 0 0 1px var(--accent-dim);
          cursor: pointer;
        }
        .dual-range::-moz-range-thumb {
          pointer-events: auto;
          height: 16px;
          width: 16px;
          border-radius: 9999px;
          background: var(--accent);
          border: 2px solid var(--background);
          box-shadow: 0 0 0 1px var(--accent-dim);
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}

// ImageUploadGrid, five fixed placeholder squares. Clicking any
// empty slot opens the file picker; clicking a filled slot opens it
// too so the user can replace. The X button on a filled slot clears.
// Currently unused, Background randomisation uses a server-backed
// uploader instead. Kept around as it's the existing pattern for
// other future sub-augs that might want local-only File refs.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function ImageUploadGrid({
  images,
  onChange,
}: {
  images: (File | null)[];
  onChange: (next: (File | null)[]) => void;
}) {
  const fileRefs = useRef<(HTMLInputElement | null)[]>([]);
  const blobUrls = useRef<Map<number, string>>(new Map());

  const setAt = (i: number, file: File | null) => {
    const next = images.slice();
    next[i] = file;
    onChange(next);
    if (file) {
      // Cache a fresh blob URL so the thumbnail renders without
      // an extra round-trip, we don't have a backend URL yet.
      const url = URL.createObjectURL(file);
      const prev = blobUrls.current.get(i);
      if (prev) URL.revokeObjectURL(prev);
      blobUrls.current.set(i, url);
    } else {
      const prev = blobUrls.current.get(i);
      if (prev) URL.revokeObjectURL(prev);
      blobUrls.current.delete(i);
    }
  };

  const onFile = (i: number) => (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setAt(i, f);
    e.target.value = "";
  };

  return (
    <div className="grid grid-cols-5 gap-2">
      {images.map((img, i) => {
        const url = img ? blobUrls.current.get(i) ?? URL.createObjectURL(img) : null;
        if (img && url && !blobUrls.current.has(i)) blobUrls.current.set(i, url);
        return (
          <div key={i} className="relative aspect-square">
            <input
              ref={(el) => { fileRefs.current[i] = el; }}
              type="file"
              accept="image/*"
              onChange={onFile(i)}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileRefs.current[i]?.click()}
              className={[
                "absolute inset-0 rounded-md border grid place-items-center transition-colors overflow-hidden",
                img
                  ? "border-[var(--line)] bg-[var(--panel)]"
                  : "border-[var(--line)] bg-[var(--panel)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
              ].join(" ")}
              title={img ? `${img.name}, click to replace` : "Click to upload"}
            >
              {img && url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={url} alt="" className="absolute inset-0 w-full h-full object-cover" />
              ) : (
                <svg viewBox="0 0 24 24" className="h-5 w-5 text-foreground/40" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 5v14M5 12h14" />
                </svg>
              )}
            </button>
            {img && (
              <button
                type="button"
                onClick={() => setAt(i, null)}
                aria-label="Remove image"
                className="absolute top-1 right-1 h-5 w-5 grid place-items-center rounded-md bg-black/65 text-white/80 hover:bg-black/85 hover:text-white text-xs"
              >
                ×
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
