"use client";

import { useEffect, useRef, useState } from "react";

import { BoxEditor, type EditableBox } from "../BoxEditor";
import { LABEL_COLOURS } from "./labelColours";
import { apiFetch } from "@/lib/apiFetch";

// One reference image. width, height and boxes are set after the pipeline
// runs on it; boxes include edits from this editor.
export type ReferenceImage = {
  file: File;
  preview: string; // blob: URL, or the engine URL once saved
  width?: number;
  height?: number;
  boxes?: EditableBox[];
  // Label section the reference was dropped into; if unset, its first box's label.
  label?: string;
  // Set once saved by POST /api/v2/projects/{id}/references.
  referenceId?: string;
  filename?: string;
  // BlurHash placeholder, 4 x 3 components; null until the backend encodes one.
  blurhash?: string | null;
};

// Reference image editor: BoxEditor on the /api/v2/references endpoints.
export function ReferenceImageEditor({
  refImage,
  labels,
  projectId,
  onChange,
  onClose,
  onPrev,
  onNext,
  hasPrev,
  hasNext,
  index,
  total,
}: {
  refImage: ReferenceImage;
  labels: string[];
  // Set for a saved reference, which the backend loads by project_id and
  // filename. Undefined during onboarding.
  projectId?: string | null;
  onChange: (next: EditableBox[]) => void;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  hasPrev: boolean;
  hasNext: boolean;
  index: number;
  total: number;
}) {
  const [boxes, setBoxes] = useState<EditableBox[]>(refImage.boxes ?? []);
  // The reference's section label, forced onto detected and classified
  // boxes. Null for a reference without a section.
  const sectionLabel = refImage.label?.trim() || null;
  const [dims, setDims] = useState<{ w: number; h: number } | null>(
    refImage.width && refImage.height
      ? { w: refImage.width, h: refImage.height }
      : null,
  );

  // Set once this image's boxes are edited.
  const userTouchedRef = useRef(false);

  // Keyed on the preview URL: the parent rebuilds refImage on every
  // update, and a reset then would drop an in-flight draw.
  useEffect(() => {
    userTouchedRef.current = false;
    setBoxes(refImage.boxes ?? []);
    setDims(
      refImage.width && refImage.height
        ? { w: refImage.width, h: refImage.height }
        : null,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refImage.preview]);

  // Takes boxes from the parent until the user edits.
  useEffect(() => {
    if (userTouchedRef.current) return;
    setBoxes(refImage.boxes ?? []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refImage.boxes]);

  // Reads the size from the decoded image when the pipeline gave none.
  useEffect(() => {
    if (dims) return;
    const img = new window.Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = refImage.preview;
    return () => { img.onload = null; };
  }, [dims, refImage.preview]);

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
  }, [onClose, onPrev, onNext, hasPrev, hasNext]);

  // Keys 1-9 relabel the hovered box to labels[digit - 1].
  const [hoveredCanvasBoxId, setHoveredCanvasBoxId] = useState<string | null>(null);
  // Manual mode: drawn boxes skip segment_box, classify_box and detect_point.
  const [manualMode, setManualMode] = useState(false);
  const relabelStateRef = useRef({ hoveredCanvasBoxId, labels, boxes });
  useEffect(() => {
    relabelStateRef.current = { hoveredCanvasBoxId, labels, boxes };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key;
      if (!/^[1-9]$/.test(k)) return;
      const slot = parseInt(k, 10) - 1;
      const state = relabelStateRef.current;
      if (state.hoveredCanvasBoxId === null || slot >= state.labels.length) return;
      e.preventDefault();
      const newLabel = state.labels[slot];
      const next = state.boxes.map((b) =>
        b.id === state.hoveredCanvasBoxId ? { ...b, label: newLabel } : b,
      );
      userTouchedRef.current = true;
      setBoxes(next);
      onChange(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onChange]);

  const apply = (next: EditableBox[]) => {
    userTouchedRef.current = true;
    setBoxes(next);
    onChange(next);
  };

  // Saved references start with an empty File. This fetches the bytes on
  // first use and stores them in refImage.file.
  const ensureFile = async (): Promise<File> => {
    if (refImage.file && refImage.file.size > 0) return refImage.file;
    if (!refImage.preview) return refImage.file;
    try {
      // apiFetch would put the engine origin in front of a blob: URL.
      const fetchFn = refImage.preview.startsWith("blob:") ? fetch : apiFetch;
      const r = await fetchFn(refImage.preview);
      if (!r.ok) return refImage.file;
      const blob = await r.blob();
      const file = new File(
        [blob],
        refImage.file?.name || refImage.filename || "image",
        { type: blob.type || "image/jpeg" },
      );
      refImage.file = file;
      return file;
    } catch {
      return refImage.file;
    }
  };

  // Body for the reference ML calls. The image is attached only when local
  // bytes exist. The backend loads a saved one by project_id and filename.
  const refFormData = async (): Promise<FormData> => {
    const fd = new FormData();
    const file = await ensureFile();
    if (file && file.size > 0) fd.append("image", file);
    if (projectId) fd.append("project_id", projectId);
    if (refImage.filename) fd.append("filename", refImage.filename);
    return fd;
  };

  return (
    // Fills the shell's content area, leaving the app chrome visible.
    <div
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[400] backdrop-blur-xl flex flex-col"
      style={{ background: "rgb(var(--background-rgb) / 0.8)" }}
      role="dialog"
      aria-modal="true"
    >
      {/* Editor card */}
      <div
        className="flex-1 mx-12 mt-12 mb-4 min-h-0 rounded-2xl border border-foreground/10 bg-[var(--background)] overflow-hidden flex flex-col"
        style={{ boxShadow: "0 24px 80px -12px rgb(var(--shadow-rgb) / 0.7), 0 0 0 1px rgb(var(--foreground-rgb) / 0.04) inset" }}
      >
        <div className="flex items-center justify-between gap-4 px-6 py-3 border-b border-foreground/10">
          <div className="text-sm text-foreground/55 shrink-0">
            Reference editor
            {dims && (
              <span className="ml-3 font-mono text-xs text-foreground/35">
                {dims.w}×{dims.h}
              </span>
            )}
          </div>
          {/* Quick-relabel legend, keys 1-9 */}
          {labels.length > 0 && (
            <div className="hidden lg:flex items-center gap-1.5 shrink-0 flex-wrap justify-center max-w-[60%]">
              <span className="text-[10px] font-mono text-foreground/35 uppercase tracking-wider mr-1">
                hover + press
              </span>
              {labels.slice(0, 9).map((lab, i) => (
                <span
                  key={lab}
                  className="inline-flex items-center gap-1 rounded-full border border-foreground/15 bg-foreground/[0.04] px-1.5 py-0.5 text-[10px] tabular-nums"
                >
                  <kbd className="font-mono text-foreground/85 px-1 py-[1px] rounded bg-foreground/10">
                    {i + 1}
                  </kbd>
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: LABEL_COLOURS[i % LABEL_COLOURS.length] }}
                    aria-hidden
                  />
                  <span className="text-foreground/75">{lab}</span>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-center gap-3 shrink-0">
            <span className="hidden md:inline text-[11px] text-foreground/30">
              ← / → · Esc
            </span>
            {/* Manual annotation toggle */}
            <button
              type="button"
              onClick={() => setManualMode((v) => !v)}
              aria-pressed={manualMode}
              title={manualMode
                ? "Manual mode is on: drawn boxes skip auto-detect and you type the label"
                : "Switch to manual annotation (draw boxes by hand, no auto-detect)"}
              className={[
                "h-8 inline-flex items-center gap-1.5 rounded-full border px-3 text-[11px] uppercase tracking-wider font-mono transition-colors",
                manualMode
                  ? "border-orange-500/60 bg-orange-500/15 text-orange-200"
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
            <button
              type="button"
              onClick={onClose}
              className="text-sm text-foreground/65 hover:text-foreground transition-colors"
            >
              Close
            </button>
          </div>
        </div>
        <div className="flex-1 min-h-0 p-3">
          {dims ? (
            <BoxEditor
              // Remounts per image, so editor state does not carry over.
              key={refImage.preview}
              imageUrl={refImage.preview}
              imageWidth={dims.w}
              imageHeight={dims.h}
              boxes={boxes}
              onChange={apply}
              projectTags={labels}
              onHoverChange={setHoveredCanvasBoxId}
              onBoxDrawn={manualMode ? undefined : async (b) => {
                const fd = await refFormData();
                fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                const r = await apiFetch("/api/v2/references/segment_box", {
                  method: "POST",
                  body: fd,
                });
                if (!r.ok) return null;
                const d = await r.json();
                return d.mask ?? null;
              }}
              onClassifyBox={
                manualMode || labels.length === 0
                  ? undefined
                  : async (b) => {
                      const fd = await refFormData();
                      fd.append("box", JSON.stringify([b.x0, b.y0, b.x1, b.y1]));
                      fd.append("labels", JSON.stringify(labels));
                      const r = await apiFetch("/api/v2/references/classify_box", {
                        method: "POST",
                        body: fd,
                      });
                      if (!r.ok) return null;
                      const d = await r.json();
                      // A section reference keeps its label, with the model's score.
                      return { label: sectionLabel ?? d.label ?? null, score: d.score ?? null };
                    }
              }
              onPointDetect={manualMode ? undefined : async (point) => {
                const fd = await refFormData();
                fd.append("point", JSON.stringify([point.x, point.y]));
                if (sectionLabel) fd.append("force_label", sectionLabel);
                const r = await apiFetch("/api/v2/references/detect_point", {
                  method: "POST",
                  body: fd,
                });
                if (!r.ok) return null;
                const data = await r.json();
                // A section reference keeps its label.
                if (sectionLabel && data && typeof data === "object") data.label = sectionLabel;
                return data;
              }}
            />
          ) : (
            <div className="h-full grid place-items-center text-foreground/40 text-sm">
              Loading…
            </div>
          )}
        </div>
      </div>

      {/* Prev and next */}
      <div className="flex flex-col items-center gap-2 pb-6 pt-2 select-none">
        <div className="flex items-center justify-center gap-6">
          <NavArrow direction="prev" disabled={!hasPrev} onClick={onPrev} />
          <span className="font-mono tabular-nums text-sm text-foreground/50 min-w-[4rem] text-center">
            {index + 1} / {total}
          </span>
          <NavArrow direction="next" disabled={!hasNext} onClick={onNext} />
        </div>
        <span className="flex items-center gap-1.5 text-[11px] text-foreground/40">
          Use
          <kbd className="font-mono px-1.5 py-[1px] rounded bg-foreground/10 text-foreground/70">←</kbd>
          <kbd className="font-mono px-1.5 py-[1px] rounded bg-foreground/10 text-foreground/70">→</kbd>
          arrow keys to move between images ·
          <kbd className="font-mono px-1.5 py-[1px] rounded bg-foreground/10 text-foreground/70">Esc</kbd>
          to close
        </span>
      </div>
    </div>
  );
}

function NavArrow({
  direction,
  disabled,
  onClick,
}: {
  direction: "prev" | "next";
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={direction === "prev" ? "Previous image" : "Next image"}
      className={[
        "h-14 w-14 grid place-items-center rounded-full border border-foreground/15 bg-foreground/[0.04] text-foreground/75 transition-all",
        disabled
          ? "opacity-25 cursor-not-allowed"
          : "hover:bg-foreground/[0.10] hover:border-foreground/35 hover:text-foreground hover:scale-105 active:scale-95",
      ].join(" ")}
    >
      <svg
        viewBox="0 0 24 24"
        className="h-6 w-6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {direction === "prev" ? <path d="m15 18-6-6 6-6" /> : <path d="m9 6 6 6-6 6" />}
      </svg>
    </button>
  );
}
