"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { buildProjectLabelColourMap, colourForLabelStable, readableTextForBg } from "../v2/OnboardLabelsV2";

// Project-card label row that NEVER wraps onto a second line.
// Renders the prefix of `tags` that fits the available width, falling
// back to a "+N" overflow chip once the next tag won't fit. Hovering
// the overflow chip pops a label list out to the right of the cursor
// and dims the rest of the page; the host card lifts above the dim
// via an inline z-index that the parent applies based on
// onOverflowOpenChange.
//
// Used by both the workspace card (HomeView) and the public-projects
// card (ProjectsView) so the visual treatment stays consistent.
// Colour assignment is via colourForLabelStable so the same label
// string keeps the same colour across every view + project.
export function ProjectTagsRow({
  tags,
  labelAliases = {},
  colourOverrides,
  onOverflowOpenChange,
}: {
  tags: string[];
  labelAliases?: Record<string, string>;
  /** Per-label colour overrides keyed by canonical-lower label. When
      set, the user-picked colour wins over the project's stable
      hash palette so the workspace + public chips match the in-app
      viewer chips. */
  colourOverrides?: Record<string, string> | null;
  /** Fires when the hover popup opens / closes so the host card
      can bump its own z-index above the dim overlay. */
  onOverflowOpenChange?: (open: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLCanvasElement | null>(null);

  // disp() returns the display name for a canonical label by
  // applying the alias map (case-insensitive lookup).
  const disp = useMemo(
    () => (t: string) => labelAliases[t.trim().toLowerCase()] || t,
    [labelAliases],
  );

  // Project-scoped palette assignment, guarantees no two chips on
  // this card share a visually-similar colour. Same project +
  // same labels yields the same map across the workspace, public,
  // and project surfaces.
  const colourMap = useMemo(
    () => buildProjectLabelColourMap(tags, colourOverrides),
    [tags, colourOverrides],
  );
  const colourFor = (label: string): string => {
    const key = label.trim().toLowerCase();
    return (colourOverrides && colourOverrides[key])
      || colourMap.get(key)
      || colourForLabelStable(label);
  };

  const [visibleCount, setVisibleCount] = useState(tags.length);
  const [hoverOpen, setHoverOpen] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    onOverflowOpenChange?.(hoverOpen);
  }, [hoverOpen, onOverflowOpenChange]);

  // Width-aware fit pass. Uses canvas measureText against the
  // chip's font so we never have to mount + measure off-DOM nodes.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!measureRef.current) measureRef.current = document.createElement("canvas");
    const ctx = measureRef.current.getContext("2d");
    if (!ctx) {
      setVisibleCount(tags.length);
      return;
    }
    // Match the chip's actual rendered font. uppercase + tracking-
    // wider don't change measureText (CSS letter-spacing isn't
    // applied to canvas), so we approximate by inflating the
    // measured width.
    const CHIP_FONT = '600 10px ui-monospace, "Geist Mono", monospace';
    const LETTER_SPACING_FACTOR = 1.18; // ~0.05em + uppercase
    const CHIP_PAD_X = 16; // px-2 (= 0.5rem each side) at 10px
    const CHIP_GAP = 6;    // gap-1.5
    const OVERFLOW_CHIP_W = 38; // "+99" worst case
    const MAX_CHIP_W = 160; // matches `max-w-[10rem]`
    ctx.font = CHIP_FONT;

    const update = () => {
      const w = el.offsetWidth;
      if (w <= 0) return;
      let used = 0;
      let count = 0;
      for (let i = 0; i < tags.length; i++) {
        const tw = Math.min(
          MAX_CHIP_W,
          ctx.measureText(disp(tags[i])).width * LETTER_SPACING_FACTOR,
        );
        const chipW = tw + CHIP_PAD_X;
        const g = i > 0 ? CHIP_GAP : 0;
        // After placing chip i, are there more chips? If yes,
        // reserve room for the "+N" overflow chip.
        const remaining = tags.length - count - 1;
        const reserve = remaining > 0 ? CHIP_GAP + OVERFLOW_CHIP_W : 0;
        if (used + g + chipW + reserve > w) break;
        used += g + chipW;
        count++;
      }
      setVisibleCount(count);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [tags, disp]);

  // Close on scroll, popup is position-fixed against the cursor
  // so a scroll would orphan it.
  useEffect(() => {
    if (!hoverOpen) return;
    const close = () => setHoverOpen(false);
    window.addEventListener("scroll", close, { passive: true, capture: true });
    return () => window.removeEventListener("scroll", close, true);
  }, [hoverOpen]);

  const visible = tags.slice(0, visibleCount);
  const overflow = tags.slice(visibleCount);
  const overflowCount = overflow.length;

  return (
    <>
      <div
        ref={containerRef}
        className="flex flex-nowrap items-center gap-1.5 overflow-hidden min-h-[20px]"
      >
        {visible.map((t) => {
          const bg = colourFor(t);
          return (
            <span
              key={t}
              className="rounded-full px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider truncate max-w-[10rem] shrink-0"
              style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
            >
              {disp(t)}
            </span>
          );
        })}
        {overflowCount > 0 && (
          <span
            className="rounded-full border border-foreground/15 bg-foreground/[0.06] px-2 py-0.5 text-[10px] leading-normal text-foreground/80 hover:text-foreground hover:border-foreground/30 transition-colors shrink-0 cursor-default"
            onMouseEnter={(e) => {
              setCursor({ x: e.clientX, y: e.clientY });
              setHoverOpen(true);
            }}
            onMouseMove={(e) => setCursor({ x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setHoverOpen(false)}
            onClick={(e) => e.stopPropagation()}
          >
            +{overflowCount}
          </span>
        )}
      </div>
      {hoverOpen && typeof window !== "undefined" && createPortal(
        <>
          <div
            className="fixed inset-0 z-[1000] bg-black/55 pointer-events-none"
            style={{ animation: "tagsDimIn 220ms ease-out both" }}
            aria-hidden
          />
          <div
            className="fixed z-[1002] pointer-events-none"
            style={{
              left: cursor.x + 14,
              top: cursor.y + 14,
              animation: "tagsPopIn 220ms cubic-bezier(0.2,0.7,0.2,1) both",
            }}
          >
            <div className="flex flex-wrap gap-1.5 max-w-[22rem] rounded-2xl border border-foreground/15 bg-[var(--surface)]/95 backdrop-blur-md p-3 shadow-[0_8px_40px_-8px_rgb(var(--shadow-rgb) / 0.85)]">
              {overflow.map((t) => {
                const bg = colourFor(t);
                return (
                  <span
                    key={t}
                    className="rounded-full px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider truncate max-w-[12rem]"
                    style={{ backgroundColor: bg, color: readableTextForBg(bg) }}
                  >
                    {disp(t)}
                  </span>
                );
              })}
            </div>
          </div>
          <style>{`
            @keyframes tagsDimIn {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            @keyframes tagsPopIn {
              from { opacity: 0; transform: translateY(-4px) scale(0.97); }
              to { opacity: 1; transform: translateY(0) scale(1); }
            }
          `}</style>
        </>,
        document.body,
      )}
    </>
  );
}
