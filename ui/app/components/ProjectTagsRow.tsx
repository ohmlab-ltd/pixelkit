"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { buildProjectLabelColourMap, colourForLabelStable, readableTextForBg } from "../v2/labelColours";

// Single-line label row; a "+N" chip lists the overflow on hover.
export function ProjectTagsRow({
  tags,
  labelAliases = {},
  colourOverrides,
  onOverflowOpenChange,
}: {
  tags: string[];
  labelAliases?: Record<string, string>;
  /** Colours keyed by lower-case label; they win over the palette. */
  colourOverrides?: Record<string, string> | null;
  /** Lets the host card raise its z-index above the dim overlay. */
  onOverflowOpenChange?: (open: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLCanvasElement | null>(null);

  const disp = useMemo(
    () => (t: string) => labelAliases[t.trim().toLowerCase()] || t,
    [labelAliases],
  );

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

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!measureRef.current) measureRef.current = document.createElement("canvas");
    const ctx = measureRef.current.getContext("2d");
    if (!ctx) {
      setVisibleCount(tags.length);
      return;
    }
    // Chips inherit the row's font family and weight, at 10px, upper case,
    // with tracking-wider (0.05em). Canvas applies neither the case nor the
    // tracking, so both are added here.
    const style = window.getComputedStyle(el);
    const CHIP_FONT = `${style.fontStyle} ${style.fontWeight} 10px ${style.fontFamily}`;
    const CHIP_TRACKING = 0.5; // px per character
    const CHIP_PAD_X = 16; // px-2, both sides
    const CHIP_GAP = 6;    // gap-1.5
    const OVERFLOW_CHIP_W = 38; // "+99" worst case
    const MAX_CHIP_W = 160; // max-w-[10rem], padding included
    ctx.font = CHIP_FONT;

    const update = () => {
      const w = el.offsetWidth;
      if (w <= 0) return;
      let used = 0;
      let count = 0;
      for (let i = 0; i < tags.length; i++) {
        const text = disp(tags[i]).toUpperCase();
        const chipW = Math.min(
          MAX_CHIP_W,
          ctx.measureText(text).width + CHIP_TRACKING * text.length + CHIP_PAD_X,
        );
        const g = i > 0 ? CHIP_GAP : 0;
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
    // The first pass can run before the web font has loaded.
    let live = true;
    document.fonts?.ready.then(() => { if (live) update(); });
    return () => {
      live = false;
      ro.disconnect();
    };
  }, [tags, disp]);

  // Closed on scroll: the popup is placed at the cursor's viewport position.
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
