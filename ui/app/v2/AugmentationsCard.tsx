"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { API_BASE, apiFetch } from "../../lib/apiFetch";
import { LabelJobCard, type LabelJobState } from "./LabelJobCard";

// Rotating status lines for the generate and remove jobs.
const AUGMENT_PHRASES = [
  "Rotating images…",
  "Adding motion blur…",
  "Swapping backgrounds…",
  "Varying the lighting…",
  "Warping perspective…",
  "Reducing bit depth…",
  "Distorting colour…",
  "Shifting hue…",
  "Adding block occlusions…",
  "Adding sensor noise…",
  "Scaling images…",
  "Writing augmented copies…",
  "Updating warped annotations…",
];
const REMOVE_AUGMENT_PHRASES = [
  "Deleting augmented copies…",
  "Resetting per-image augmentation counts…",
  "Clearing the augmentations folder…",
  "Removing warped annotations…",
  "Removing generated copies…",
  "Keeping the original images…",
];

type AugmentPreviewSource = {
  source: "reference" | "import";
  filename: string;
  preview: string;
};

// Augmentation settings. Each category and sub-augmentation has its own
// enable flag and frequency.

type Frequency = "all" | "random";
type PerImage = "off" | "1" | "2" | "3" | "random";

type SubBase = { enabled: boolean; frequency: Frequency };
type SubWithSize = SubBase & { size: number };
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
  };
  distortion: {
    enabled: boolean; frequency: Frequency;
    perspectiveWarp: SubWithStrength;
    scaleRotation: SubWithRange;
    hueShift: SubWithStrength;
  };
  camera: {
    enabled: boolean; frequency: Frequency;
    // Strengths 0..10, 0 for no effect. The category's `enabled` gates them all.
    motionBlur: number;
    noise: number;
    colourDistortion: number;
    chromaticAberration: number;
    bitDepth: number;
    // A saved config without these keys keeps the default 0.
    lensDistortion: number;
    pixelation: number;
    lowResolution: number;
    lensGlare: number;
  };
  domain: {
    enabled: boolean; frequency: Frequency;
    backgrounds: SubBackgrounds;
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
      lighting: { ...fSub, strength: 0 },
    },
  };
}

export function AugmentationsCard({
  projectId = null,
  previewSources: previewSourcesProp = [],
  augmentJob = null,
  onAugmentJobChange,
  visible = true,
}: {
  /** Project UUID, for the preview endpoint. */
  projectId?: string | null;
  /** References and imports that the preview picks a random image from. */
  previewSources?: AugmentPreviewSource[];
  /** Augment job state, polled by the parent. */
  augmentJob?: LabelJobState | null;
  /** Lets Update set a "queued" placeholder before the next poll. */
  onAugmentJobChange?: (next: LabelJobState | null | ((cur: LabelJobState | null) => LabelJobState | null)) => void;
  /** The tab is showing; the full image list loads only then. */
  visible?: boolean;
} = {}) {
  // Every image from /overview, fetched each time the tab shows, merged with
  // `previewSources`, which may hold only the first page.
  const [fallbackSources, setFallbackSources] = useState<AugmentPreviewSource[]>([]);
  useEffect(() => {
    if (!projectId || !visible) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview`);
        if (cancelled || !r.ok) return;
        const ov = await r.json() as {
          imports?: { id: string; filename: string }[];
          references?: { id: string; filename: string }[];
        };
        if (cancelled) return;
        const next: AugmentPreviewSource[] = [];
        for (const imp of ov.imports ?? []) {
          if (!imp.filename) continue;
          next.push({
            source: "import",
            filename: imp.filename,
            preview: `${API_BASE}/api/v2/projects/${projectId}/imports/${encodeURIComponent(imp.filename)}`,
          });
        }
        for (const ref of ov.references ?? []) {
          if (!ref.filename) continue;
          next.push({
            source: "reference",
            filename: ref.filename,
            preview: `${API_BASE}/api/v2/projects/${projectId}/references/${encodeURIComponent(ref.filename)}`,
          });
        }
        setFallbackSources((cur) => {
          if (cur.length === next.length && cur.every((s, i) => s.filename === next[i].filename)) {
            return cur;
          }
          return next;
        });
      } catch { /* The parent's sources still work. */ }
    })();
    return () => { cancelled = true; };
  }, [projectId, visible]);

  // The parent's entries first, as they can carry blobs of uploads in flight.
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
  // A queued or running job, from the parent's poll, locks Update and Randomise.
  const jobActive = augmentJob?.status === "queued" || augmentJob?.status === "running";
  const [categoryOpen, setCategoryOpen] = useState<Record<string, boolean>>({
    occlusion: false, distortion: false, camera: false, domain: false,
  });

  // Loads the config saved by the last Update. An empty config keeps the defaults.
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
            // Saved sub-trees over the defaults, so missing keys keep their defaults.
            camera: { ...cur.camera, ...(cfg.camera ?? {}) },
            domain: {
              ...cur.domain,
              ...(cfg.domain ?? {}),
              backgrounds: { ...cur.domain.backgrounds, ...((cfg.domain as AugmentationsState["domain"] | undefined)?.backgrounds ?? {}) },
              lighting: { ...cur.domain.lighting, ...((cfg.domain as AugmentationsState["domain"] | undefined)?.lighting ?? {}) },
            },
            occlusion: {
              ...cur.occlusion,
              ...(cfg.occlusion ?? {}),
              randomBlock: { ...cur.occlusion.randomBlock, ...((cfg.occlusion as AugmentationsState["occlusion"] | undefined)?.randomBlock ?? {}) },
            },
            distortion: {
              ...cur.distortion,
              ...(cfg.distortion ?? {}),
              perspectiveWarp: { ...cur.distortion.perspectiveWarp, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.perspectiveWarp ?? {}) },
              scaleRotation: { ...cur.distortion.scaleRotation, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.scaleRotation ?? {}) },
              hueShift: { ...cur.distortion.hueShift, ...((cfg.distortion as AugmentationsState["distortion"] | undefined)?.hueShift ?? {}) },
            },
          }));
          // Opens each category enabled in the saved config.
          setCategoryOpen((co) => ({
            occlusion: co.occlusion || !!(cfg.occlusion as AugmentationsState["occlusion"] | undefined)?.enabled,
            distortion: co.distortion || !!(cfg.distortion as AugmentationsState["distortion"] | undefined)?.enabled,
            camera: co.camera || !!(cfg.camera as AugmentationsState["camera"] | undefined)?.enabled,
            domain: co.domain || !!(cfg.domain as AugmentationsState["domain"] | undefined)?.enabled,
          }));
        }
      } catch { /* Defaults stay. */ }
      finally { if (!cancelled) setConfigLoaded(true); }
    })();
    return () => { cancelled = true; };
  }, [projectId, configLoaded]);

  // Starts a generate job. With perImage "off" the server only deletes
  // the existing augmentations.
  const onUpdate = async () => {
    if (!projectId || updating) return;
    setUpdating(true);
    setUpdateError(null);
    // Placeholder until the parent's poll sees the job. jobId "removing"
    // marks a clear-only run.
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
      // Fired here and again when the job ends: a fast job can finish between polls.
      try {
        window.dispatchEvent(new CustomEvent("pixelkit-augmentations-generated", {
          detail: { projectId },
        }));
      } catch { /* ignore */ }
      // A clear-only run has no job to poll, so it is marked done here.
      if (isClearOnly && onAugmentJobChange) {
        onAugmentJobChange((cur) => cur ? { ...cur, status: "done", total: cur.total || 1, index: cur.total || 1 } : cur);
      }
    } catch (e) {
      setUpdateError(e instanceof Error ? e.message : String(e));
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

  // Random values for the dial-only augmentations. Backgrounds need uploads
  // and are left alone.
  const randomiseAll = () => {
    if (updating || jobActive) return;
    const rnd = (min: number, max: number) => min + Math.random() * (max - min);
    const flip = (p: number) => Math.random() < p;
    setState((cur) => ({
      ...cur,
      // Never "off", which would make Update clear instead of generate.
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
          // Scale stays within 0.7..1.3.
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
          size: Math.round(rnd(5, 25)),
        },
      },
      domain: {
        ...cur.domain,
        enabled: true,
        frequency: flip(0.5) ? "all" : "random",
        lighting: {
          ...cur.domain.lighting,
          enabled: true,
          frequency: flip(0.5) ? "all" : "random",
          strength: Math.round(rnd(2, 7) * 10) / 10,
        },
      },
    }));
    setCategoryOpen({ occlusion: true, distortion: true, camera: true, domain: true });
  };

  const enabledCount = countEnabled(state);

  return (
    <section className="px-6 lg:px-10 pt-8 pb-12">
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
            disabled={updating || jobActive}
            className={[
              "inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-2.5 py-1 text-[13px] font-medium transition-colors",
              updating || jobActive
                ? "text-[var(--fg-faint)]"
                : "text-[var(--fg-soft)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground",
            ].join(" ")}
            title="Randomise every augmentation setting (skips inputs that need uploaded images, like backgrounds)"
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
            disabled={updating || jobActive}
            className={[
              "rounded-md bg-[var(--accent)] px-3 py-1 text-[13px] font-medium text-[var(--accent-contrast)] transition-[filter,opacity]",
              updating || jobActive ? "opacity-50" : "hover:brightness-105",
            ].join(" ")}
            title={
              state.perImage === "off"
                ? "Delete every augmentation for this dataset"
                : "Generate augmentations for every dataset image"
            }
          >
            {updating
              ? (state.perImage === "off" ? "Clearing…" : "Starting…")
              : jobActive
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

      {/* Job progress, in a LabelJobCard. */}
      <div
        className="mb-6 grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out"
        style={{
          gridTemplateRows: augmentJob ? "1fr" : "0fr",
          opacity: augmentJob ? 1 : 0,
          marginBottom: augmentJob ? "1.5rem" : "0",
        }}
      >
        <div className="min-h-0 overflow-hidden">
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
                  ? "Augmentations removed."
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

              <CategoryBlock
                title="Camera & sensor simulation"
                description="Simulate the artefacts of real-world capture so a model trained on clean images still works on camera footage."
                enabled={state.camera.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.camera.enabled;
                  return {
                    ...s,
                    camera: {
                      ...s.camera,
                      enabled: next,
                      // Disabling zeroes these dials.
                      ...(next ? {} : {
                        motionBlur: 0,
                        noise: 0,
                        colourDistortion: 0,
                        chromaticAberration: 0,
                        bitDepth: 0,
                        lensDistortion: 0,
                        pixelation: 0,
                        lowResolution: 0,
                        lensGlare: 0,
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

              <CategoryBlock
                title="Domain randomisation"
                description="Vary background and lighting so the model transfers across deployment conditions."
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
                        lighting: { ...s.domain.lighting, enabled: false },
                      }),
                    },
                  };
                })}
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

              <CategoryBlock
                title="Occlusion"
                description="Mask out parts of the image so the model learns to detect partly hidden objects."
                enabled={state.occlusion.enabled}
                onToggle={() => setState((s) => {
                  const next = !s.occlusion.enabled;
                  return {
                    ...s,
                    occlusion: {
                      ...s.occlusion,
                      enabled: next,
                      // Disabling the category disables its sub-augmentations.
                      ...(next ? {} : {
                        randomBlock: { ...s.occlusion.randomBlock, enabled: false },
                      }),
                    },
                  };
                })}
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
                        // Enabling a sub-augmentation enables its category.
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
              </CategoryBlock>

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
                      // Enabling from identity widens the ranges; disabling resets them.
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

function countEnabled(s: AugmentationsState): number {
  // Enabled categories, not sub-augmentations.
  let n = 0;
  if (s.occlusion.enabled) n++;
  if (s.distortion.enabled) n++;
  if (s.camera.enabled) n++;
  if (s.domain.enabled) n++;
  return n;
}

// Camera and sensor dials, 0..10, with a preview from /augment/preview.
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
  const [sourceIdx, setSourceIdx] = useState<number>(() => 0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));
  const activeSource = previewSources[sourceIdx] ?? null;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  // Latest request id. A response to an older request is dropped.
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

  // Debounced preview request.
  useEffect(() => {
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) {
        URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = null;
      }
      setPreviewUrl(null);
      return;
    }
    // All dials at 0: the source image itself.
    if (allZero) {
      if (lastBlobRef.current) {
        URL.revokeObjectURL(lastBlobRef.current);
        lastBlobRef.current = null;
      }
      setPreviewUrl(activeSource.preview);
      setPreviewError(null);
      return;
    }
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
    // activeSource is a new object on every parent render, so its fields
    // are the dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    projectId,
    activeSource?.source,
    activeSource?.filename,
    allZero, seed,
    motionBlur, noise, colourDistortion, chromaticAberration, bitDepth,
    lensDistortion, pixelation, lowResolution, lensGlare,
  ]);

  useEffect(() => () => {
    if (lastBlobRef.current) {
      URL.revokeObjectURL(lastBlobRef.current);
      lastBlobRef.current = null;
    }
  }, []);

  const pickRandom = useCallback(() => {
    if (previewSources.length === 0) return;
    if (previewSources.length === 1) {
      // One source: a new seed only.
      setSeed(Math.floor(Math.random() * 1_000_000));
      return;
    }
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
      {/* Preview at the image's own aspect ratio. */}
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

      {/* Dials commit on release. */}
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

// Random block occlusion. Imports only, as references carry no detections.
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
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
    if (!projectId || !activeSource) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
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
    // activeSource is a new object on every parent render, so its fields
    // are the dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

// One 0..10 dial with a preview, for hue shift and lighting. `buildBody`
// gives the request parameters.
function SimpleStrengthBlock({
  projectId,
  previewSources,
  strength,
  onChange,
  buildBody,
}: {
  projectId: string | null;
  previewSources: AugmentPreviewSource[];
  strength: number;
  onChange: (v: number) => void;
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
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
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
    // buildBody is left out: strength and seed drive the refetch.
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
            <img src={previewUrl} alt="Strength preview" className="block w-full h-auto" draggable={false} />
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
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
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

// Scale and rotation ranges. The server samples each range with the seed.
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
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
          the ranges below. Every copy gets new values.
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

// Up to MAX_BACKGROUND_IMAGES uploaded backgrounds, referenced by id. The
// preview keeps the pixels inside the detections and fills the rest from one.
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
  // Imports only, as references carry no detections.
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

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const bgKey = useMemo(() => backgrounds.map((b) => b.id).join("|"), [backgrounds]);

  useEffect(() => {
    // Every run, even one that returns early, supersedes the request before it.
    const myReq = ++requestIdRef.current;
    if (!projectId || !activeSource || backgrounds.length === 0) {
      if (lastBlobRef.current) { URL.revokeObjectURL(lastBlobRef.current); lastBlobRef.current = null; }
      setPreviewUrl(null);
      return;
    }
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

  // Uploads one file. The caller appends all the entries in one update.
  const uploadOneRaw = async (f: File): Promise<BackgroundEntry | null> => {
    if (!projectId) return null;
    const fd = new FormData();
    fd.append("image", f);
    const r = await apiFetch(
      `/api/v2/projects/${projectId}/augment/background/upload`,
      { method: "POST", body: fd },
    );
    if (!r.ok) throw new Error(`http ${r.status}: ${await r.text()}`);
    const data = (await r.json()) as { background_id: string };
    return {
      id: data.background_id,
      previewUrl: `${API_BASE}/api/v2/projects/${projectId}/augment/backgrounds/${data.background_id}`,
    };
  };

  const uploadMany = async (files: File[]) => {
    if (!projectId) return;
    setUploading(true);
    setUploadError(null);
    try {
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

// Slider that calls onCommit on release, key up or blur, not while dragging.
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
  // Follows `value` except while dragging.
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

  // Highlighted when off 0.
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
        onChange={(e) => setDraft(Number(e.target.value))}
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

// Collapsible category with an enable checkbox and, when enabled, a
// frequency toggle.
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
  // Camera only: the generator reads the other categories' choice per
  // sub-augmentation.
  frequency?: Frequency;
  onFrequency?: (f: Frequency) => void;
  open: boolean;
  onOpen: () => void;
  children: React.ReactNode;
}) {
  // Opens when the category is enabled.
  const wasEnabledRef = useRef(enabled);
  useEffect(() => {
    if (enabled && !wasEnabledRef.current && !open) onOpen();
    wasEnabledRef.current = enabled;
  }, [enabled, open, onOpen]);

  return (
    <div
      className={[
        "rounded-md border border-[var(--line)] transition-colors",
        enabled ? "bg-[var(--surface-hover)]" : "bg-[var(--panel)]",
      ].join(" ")}
    >
      <div className="flex items-center gap-3 px-5 py-4">
        <Checkbox checked={enabled} onChange={onToggle} ariaLabel={`${title} enabled`} />
        <button
          type="button"
          onClick={onOpen}
          className="min-w-0 flex-1 text-left flex items-center gap-3 group"
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
        {enabled && frequency && onFrequency && (
          <div className="shrink-0">
            <FrequencyToggle value={frequency} onChange={onFrequency} />
          </div>
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

// One sub-augmentation: checkbox, label and frequency, with its controls
// as children while enabled.
function SubAugRow({
  label,
  description,
  enabled,
  onToggle,
  frequency,
  onFrequency,
  children,
}: {
  label: string;
  description?: string;
  enabled: boolean;
  onToggle: (v: boolean) => void;
  frequency: Frequency;
  onFrequency: (f: Frequency) => void;
  children?: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] transition-colors">
      <div className="flex items-start gap-3 px-4 py-3">
        <div className="pt-0.5">
          <Checkbox
            checked={enabled}
            onChange={() => onToggle(!enabled)}
            ariaLabel={`${label} enabled`}
          />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-[var(--foreground)]">{label}</div>
          {description && (
            <div className="text-[11px] text-foreground/45 mt-0.5">{description}</div>
          )}
        </div>
        {enabled && (
          <div className="shrink-0">
            <FrequencyToggle value={frequency} onChange={onFrequency} />
          </div>
        )}
      </div>
      {enabled && (
        <div className="px-4 pb-4 pt-3 border-t border-[var(--line-soft)]">
          {children}
        </div>
      )}
    </div>
  );
}

// All or Random.
function FrequencyToggle({
  value,
  onChange,
}: {
  value: Frequency;
  onChange: (v: Frequency) => void;
}) {
  const items: { value: Frequency; label: string }[] = [
    { value: "all", label: "All" },
    { value: "random", label: "Random" },
  ];
  return (
    <div className="inline-flex rounded-md border border-[var(--line)] bg-[var(--surface)] p-0.5">
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

// Two overlaid range inputs. The lower thumb never passes the upper.
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
