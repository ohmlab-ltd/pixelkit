"use client";

import { useEffect, useRef, useState } from "react";

import { BoxEditor, type EditableBox } from "../BoxEditor";
import type { ReferenceImage } from "./OnboardReferencesV2";
import { LABEL_COLOURS } from "./OnboardLabelsV2";
import { apiFetch } from "@/lib/apiFetch";

// Full-screen reference image editor wrapping V1's BoxEditor with the
// stateless V2 reference endpoints — the onboarding mirror of the
// project page's DatasetViewer. The card itself is inset from the
// edges (smaller than full screen) on a theme-aware blurred backdrop
// (white-ish in light mode, dark in dark mode via --background-rgb),
// and large prev/next buttons sit centred below the card so the user
// can flick through the reference pool without leaving the editor.
// Esc closes; Left / Right arrow keys navigate. A Manual toggle in the
// header switches drawn boxes from auto-detect (SAM2 + classify) to
// hand-drawn rectangles where the user types the label themselves,
// matching the project editor's manual annotation mode.
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
  // Project the reference belongs to, when saved. Lets the backend load
  // the reference image from disk by project_id + filename, so click-to-
  // detect / segment / classify work on existing-project references even
  // when the browser can't re-upload the stored bytes. Undefined during
  // onboarding (refs are still local Files and upload directly).
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
  // The label section this reference belongs to. A reference is "a photo
  // of <label>", so boxes detected or classified here are forced onto
  // the section label rather than guessed among siblings. Null for
  // references with no section (legacy / general), which keep the old
  // classify-among-all-labels behaviour. (Label order is left untouched
  // so the canvas palette stays colour-stable across references.)
  const sectionLabel = refImage.label?.trim() || null;
  const [dims, setDims] = useState<{ w: number; h: number } | null>(
    refImage.width && refImage.height
      ? { w: refImage.width, h: refImage.height }
      : null,
  );

  // True once the user has edited boxes on the CURRENT image (drawn,
  // dragged, deleted, relabelled, or has a click-to-detect in flight).
  // Late-arriving parent writes (the upload pipeline finishing, a
  // quality refresh) must not clobber their work - or an in-progress
  // gesture, which lives in `boxes` as a placeholder box.
  const userTouchedRef = useRef(false);

  // Reset local state whenever the image being viewed CHANGES - keyed
  // on the preview URL (stable per image, same identity BoxEditor's
  // `key` uses), NOT the refImage object. The parent rebuilds the
  // object on every refs-array update (sibling uploads completing,
  // referenceId arriving, the onChange echo itself), and resetting on
  // those wiped an in-flight draw / click-to-detect placeholder.
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

  // Adopt boxes that arrive from the parent AFTER mount (the detect +
  // embed pipeline finishing for a reference opened while still
  // processing) - but only until the user touches the canvas; their
  // copy is authoritative after that.
  useEffect(() => {
    if (userTouchedRef.current) return;
    setBoxes(refImage.boxes ?? []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refImage.boxes]);

  // Fall back to reading dimensions from the browser-decoded image
  // when the pipeline didn't supply them (e.g. user removed image
  // before pipeline finished).
  useEffect(() => {
    if (dims) return;
    const img = new window.Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = refImage.preview;
    return () => { img.onload = null; };
  }, [dims, refImage.preview]);

  // Esc closes; Left / Right navigate when not focused in an input.
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

  // Quick-relabel: press 1-9 while hovering a box to flip its label
  // to labels[digit-1]. Mirrors the dataset viewer's behaviour so
  // muscle memory carries between onboarding and live projects.
  const [hoveredCanvasBoxId, setHoveredCanvasBoxId] = useState<string | null>(null);
  // Manual annotation mode. When ON, drawn boxes / click-to-detect skip
  // the backend ML calls (segment_box, classify_box, detect_point) and
  // the user gets a raw rectangle + the label picker to type into —
  // same toggle the project page's DatasetViewer exposes.
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

  // Lazy-fetch the original bytes when the editor needs them. Refs
  // hydrate from manifest with an empty File placeholder so the grid
  // renders fast, this fills it in the moment a draw / classify /
  // point flow actually fires. Mutates refImage.file in place so
  // subsequent calls reuse the same File without a re-fetch.
  const ensureFile = async (): Promise<File> => {
    if (refImage.file && refImage.file.size > 0) return refImage.file;
    if (!refImage.preview) return refImage.file;
    try {
      // blob: URLs are local and don't need auth; server CDN URLs do.
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

  // Build the multipart body shared by all three reference ML calls.
  // The image is only attached when we actually have local bytes (a
  // freshly dropped reference). For a saved reference the backend
  // re-loads it from disk by project_id + filename, which avoids the
  // "stale reference image" 400 when the browser can't re-fetch the
  // CDN-hosted bytes.
  const refFormData = async (): Promise<FormData> => {
    const fd = new FormData();
    const file = await ensureFile();
    if (file && file.size > 0) fd.append("image", file);
    if (projectId) fd.append("project_id", projectId);
    if (refImage.filename) fd.append("filename", refImage.filename);
    return fd;
  };

  return (
    // Contained in the shell's content area (below the title bar,
    // above the status bar, right of the Explorer side bar via
    // --pk-content-left) — same containment as the dataset image
    // editor, the app chrome stays visible.
    <div
      className="fixed top-9 bottom-6 right-0 left-[var(--pk-content-left,0px)] z-[400] backdrop-blur-xl flex flex-col"
      // Theme-aware scrim: --background-rgb resolves to a light tone in
      // light mode and a dark tone in dark mode, so the blur reads
      // white-on-light and dark-on-dark instead of always-black.
      style={{ background: "rgb(var(--background-rgb) / 0.8)" }}
      role="dialog"
      aria-modal="true"
    >
      {/* Editor card, inset from the edges, leaves room for nav. */}
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
          {/* Quick-relabel legend, same shape as the dataset viewer's
              header. Hover a box, press a digit, label flips. Caps at
              9 so the legend stays compact. */}
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
            {/* Manual annotation toggle — clone of the project editor's.
                When ON, drawn boxes / clicks skip the reference ML
                endpoints and become hand-drawn rectangles the user
                labels via the picker. */}
            <button
              type="button"
              onClick={() => setManualMode((v) => !v)}
              aria-pressed={manualMode}
              title={manualMode
                ? "Manual mode ON — drawn boxes skip auto-detect; type the label yourself"
                : "Switch to manual annotation (draw boxes by hand, no auto-detect)"}
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
              // Force a remount when navigating between images so
              // BoxEditor's internal selection / drawing / mask state
              // doesn't leak across siblings.
              key={refImage.preview}
              imageUrl={refImage.preview}
              imageWidth={dims.w}
              imageHeight={dims.h}
              boxes={boxes}
              onChange={apply}
              projectTags={labels}
              colorMode="palette"
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
                      // Section reference: the class is known, so force
                      // the section label and keep the model's score.
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
                // Section reference: force the known class onto the result.
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

      {/* Bottom nav, large prev / next centred under the card, with a
          keyboard hint beneath so the arrow-key shortcut is discoverable. */}
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
