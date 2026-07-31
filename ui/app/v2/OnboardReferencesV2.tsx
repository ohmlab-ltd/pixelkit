"use client";

import { useEffect, useRef, useState } from "react";

import { LABEL_COLOURS } from "./OnboardLabelsV2";
import { ReferenceImageEditor } from "./ReferenceImageEditor";

import type { EditableBox } from "../BoxEditor";
import { isImageFile } from "@/lib/resize";

// One reference image picked by the user during onboarding. We keep
// the original File alongside an object URL so the thumbnail can
// render without re-encoding, and so the orchestrator can hand the
// blobs straight to the upload endpoint when we wire the backend.
//
// `width` / `height` / `boxes` are populated after the V2 pipeline
// finishes on the image. `boxes` carries any user edits made in the
// reference image editor (V2's BoxEditor wrapper).
export type ReferenceImage = {
  file: File;
  preview: string; // object URL
  width?: number;
  height?: number;
  boxes?: EditableBox[];
  // The label section this reference belongs to in the per-label
  // reference UI. Set at upload time from the section the user dropped
  // it into, so the class is known up front (the detector only
  // localises, it doesn't classify among siblings). Hydrated refs that
  // predate this field fall back to their first box's label.
  label?: string;
  // Set once the reference has been persisted to the backend's
  // manifest (POST /api/v2/projects/{id}/references). Drives:
  //  - the post-onboarding background uploader to skip refs that
  //    are already on the server (no duplicate POSTs)
  //  - the editor's edit-flush PUT, which targets
  //    /api/v2/projects/{id}/references/{referenceId}
  // Undefined for refs the user just dropped that haven't round-
  // tripped yet, those still flow through the upload path.
  referenceId?: string;
  filename?: string;
  // BlurHash placeholder string from the backend (4×3 components,
  // ~30 chars). FE decodes via react-blurhash into a colour
  // gradient that fills the tile before the real image bytes
  // arrive. Null when the backend hasn't encoded one yet.
  blurhash?: string | null;
};

export type ReferenceMap = Record<string, ReferenceImage[]>;

const MIN_PER_LABEL = 3;
const MAX_PER_LABEL = 5;

// Stage B of the V2 create flow.
//
// Header morphs from Stage A's "What do you want to detect?" into the
// project name (passed as prop). Skip / Done from Stage A fade out.
// Label chips animate up to sit just below the title, the chip for
// the *current* label is highlighted, the others are dimmed.
//
// One label is configured at a time. The user uploads 3-5 reference
// images that show clear, varied views of the object. Done advances
// to the next label; Skip leaves this label without references and
// moves on. After the last label the orchestrator's `onComplete` is
// invoked with the full map of references.
export function OnboardReferencesV2({
  projectName,
  labels,
  onComplete,
}: {
  projectName: string;
  labels: string[];
  onComplete: (refs: ReferenceMap) => void;
  // `onClose` reserved for a future "back to Stage A" action; the
  // current flow only exits via Done / Skip per label, so the prop
  // is not consumed here yet.
  onClose?: () => void;
}) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [refs, setRefs] = useState<ReferenceMap>({});
  const [mounted, setMounted] = useState(false);
  const [editingRefIdx, setEditingRefIdx] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  // No labels → skip this stage entirely.
  useEffect(() => {
    if (labels.length === 0) onComplete({});
  }, [labels, onComplete]);

  useEffect(() => {
    const t = window.setTimeout(() => setMounted(true), 30);
    return () => window.clearTimeout(t);
  }, []);

  // Revoke object URLs on unmount so we don't leak browser memory if
  // the user backs out of the flow without uploading.
  useEffect(() => {
    return () => {
      for (const list of Object.values(refs)) {
        for (const r of list) URL.revokeObjectURL(r.preview);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (labels.length === 0) return null;

  const currentLabel = labels[currentIdx];
  const currentRefs = refs[currentLabel] ?? [];
  const remainingSlots = Math.max(0, MAX_PER_LABEL - currentRefs.length);
  const meetsMinimum = currentRefs.length >= MIN_PER_LABEL;
  const isLast = currentIdx === labels.length - 1;

  const onUploadFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const accepted: ReferenceImage[] = [];
    let kept = remainingSlots;
    for (const f of Array.from(files)) {
      if (kept <= 0) break;
      if (!isImageFile(f)) continue;
      accepted.push({ file: f, preview: URL.createObjectURL(f) });
      kept -= 1;
    }
    if (accepted.length) {
      setRefs({
        ...refs,
        [currentLabel]: [...currentRefs, ...accepted],
      });
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const removeRef = (idx: number) => {
    const list = currentRefs.slice();
    const [gone] = list.splice(idx, 1);
    if (gone) URL.revokeObjectURL(gone.preview);
    setRefs({ ...refs, [currentLabel]: list });
  };

  const advance = () => {
    if (isLast) {
      onComplete(refs);
    } else {
      setCurrentIdx(currentIdx + 1);
    }
  };

  return (
    <>
    <div
      className="pk-backdrop fixed inset-0 z-[300] overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-label={`Reference images for ${currentLabel}`}
    >
      <div className="mx-auto max-w-3xl px-6 pt-16 pb-24">
        {/* Title, animates up from where Stage A's title was. The
            text is swapped to the project name once the user clicks
            Done in Stage A. */}
        <h1
          className={[
            "text-4xl md:text-5xl font-medium tracking-tight text-[var(--foreground)] transition-all duration-700 ease-out",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-2",
          ].join(" ")}
        >
          {projectName}
        </h1>

        {/* Label chips, current one highlighted, others dimmed. */}
        <div
          className={[
            "mt-4 flex flex-wrap items-center gap-2 transition-all duration-700 ease-out delay-100",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-1",
          ].join(" ")}
        >
          {labels.map((lab, i) => {
            const isCurrent = i === currentIdx;
            const isDone = (refs[lab] ?? []).length >= MIN_PER_LABEL;
            return (
              <span
                key={lab}
                className={[
                  "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium text-black transition-all duration-300",
                  isCurrent ? "ring-2 ring-white ring-offset-2 ring-offset-black scale-105" : "",
                  !isCurrent && !isDone ? "opacity-45" : "",
                ].join(" ")}
                style={{ backgroundColor: LABEL_COLOURS[i % LABEL_COLOURS.length] }}
              >
                {lab}
                {isDone && (
                  <span aria-hidden className="text-[10px] opacity-70">✓</span>
                )}
              </span>
            );
          })}
          <span className="ml-2 text-[11px] text-foreground/40 font-mono">
            {currentIdx + 1} / {labels.length}
          </span>
        </div>

        <div
          className={[
            "mt-10 transition-all duration-500 ease-out",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2",
          ].join(" ")}
          // Re-key by current label so the inner content re-mounts and
          // re-animates as the user moves between labels.
          key={currentLabel}
        >
          <div className="text-[11px] uppercase tracking-wider text-foreground/40 font-mono">
            Reference images
          </div>
          <h2 className="mt-1 text-2xl font-medium tracking-tight text-[var(--foreground)]">
            Upload {MIN_PER_LABEL}–{MAX_PER_LABEL} pictures of{" "}
            <span style={{ color: LABEL_COLOURS[currentIdx % LABEL_COLOURS.length] }}>
              {currentLabel}
            </span>
          </h2>
          <p className="mt-2 text-sm text-foreground/60 leading-relaxed">
            Varied photos work best, different angles, lighting, and contexts.
            Make labels as accurate as possible, every single object you want
            detected should be marked.
          </p>

          {/* Drop zone */}
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={remainingSlots <= 0}
            className={[
              "mt-6 w-full rounded-2xl border-2 border-dashed transition-all p-8 text-center",
              remainingSlots > 0
                ? "border-foreground/15 hover:border-foreground/30 hover:bg-foreground/[0.02] cursor-pointer"
                : "border-foreground/5 cursor-not-allowed opacity-50",
            ].join(" ")}
          >
            <div className="text-base text-foreground/85">
              {remainingSlots > 0
                ? "Click or drop images here"
                : `Maximum ${MAX_PER_LABEL} images reached`}
            </div>
            {remainingSlots > 0 && (
              <div className="mt-1 text-xs text-foreground/45">
                {currentRefs.length} of {MAX_PER_LABEL} used &nbsp;·&nbsp; jpg · png · webp
              </div>
            )}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => onUploadFiles(e.target.files)}
          />

          {/* Thumbnails */}
          {currentRefs.length > 0 && (
            <div className="mt-4 grid grid-cols-3 sm:grid-cols-5 gap-2">
              {currentRefs.map((r, i) => {
                const hasBoxes = (r.boxes ?? []).length > 0;
                return (
                  <div
                    key={i}
                    className="group relative aspect-square rounded-lg overflow-hidden bg-foreground/5 border border-foreground/10 cursor-pointer"
                    onClick={() => setEditingRefIdx(i)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setEditingRefIdx(i); } }}
                    aria-label="Edit reference image"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={r.preview}
                      alt=""
                      className="w-full h-full object-cover"
                    />
                    {hasBoxes && (
                      <div className="absolute bottom-1 left-1 pointer-events-none">
                        <span className="inline-flex items-center rounded-full bg-black/55 text-white px-1.5 py-0.5 text-[9px] font-mono backdrop-blur-sm">
                          {r.boxes!.length}✓
                        </span>
                      </div>
                    )}
                    <div className="absolute inset-0 grid place-items-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30 pointer-events-none">
                      <span className="inline-flex items-center gap-1 rounded-full bg-white/90 text-black px-2 py-0.5 text-[10px] font-semibold shadow">
                        Edit
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); removeRef(i); }}
                      className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-white hover:bg-black/90 grid place-items-center text-sm opacity-0 group-hover:opacity-100 transition-opacity"
                      aria-label="Remove reference"
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="mt-14 flex items-center justify-between">
          <span className="text-sm text-foreground/45">
            {meetsMinimum
              ? `${currentRefs.length} reference${currentRefs.length === 1 ? "" : "s"} ready`
              : `${MIN_PER_LABEL - currentRefs.length} more needed for confident matching`}
          </span>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={advance}
              className="px-5 py-2.5 text-sm text-foreground/55 hover:text-foreground/90 transition-colors"
            >
              Skip
            </button>
            <button
              type="button"
              onClick={advance}
              disabled={!meetsMinimum}
              className="rounded-full bg-orange-500 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isLast ? "Open project" : "Next label"}
            </button>
          </div>
        </div>
      </div>
    </div>

    {editingRefIdx !== null && currentRefs[editingRefIdx] && (
      <ReferenceImageEditor
        refImage={currentRefs[editingRefIdx]}
        labels={[currentLabel]}
        onChange={(nextBoxes) => {
          const updated = currentRefs.map((r, i) =>
            i === editingRefIdx ? { ...r, boxes: nextBoxes } : r,
          );
          setRefs({ ...refs, [currentLabel]: updated });
        }}
        onClose={() => setEditingRefIdx(null)}
        onPrev={() => setEditingRefIdx((idx) => (idx === null ? null : Math.max(0, idx - 1)))}
        onNext={() => setEditingRefIdx((idx) => (idx === null ? null : Math.min(currentRefs.length - 1, idx + 1)))}
        hasPrev={editingRefIdx > 0}
        hasNext={editingRefIdx < currentRefs.length - 1}
        index={editingRefIdx}
        total={currentRefs.length}
      />
    )}
    </>
  );
}
