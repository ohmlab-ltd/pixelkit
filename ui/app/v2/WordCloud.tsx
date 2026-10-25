"use client";

import { useEffect, useMemo, useState } from "react";

import { WORDS } from "@/lib/words";

// Ambient word backdrop for the labels stage of the create flow.

type Item = {
  word: string;
  top: number;     // percent from top of container
  left: number;    // percent from left of container
  size: number;    // rem
  rotation: number; // degrees
};

const WORD_COUNT = 28;

// Words fade out one after another, EXIT_STAGGER_MS apart, each over FADE_MS.
const EXIT_STAGGER_MS = 18;
const FADE_MS = 640;
// Until the last word has faded out.
export const WORD_CLOUD_EXIT_MS = (WORD_COUNT - 1) * EXIT_STAGGER_MS + FADE_MS;

// Exclusion zones in percent of the wrapper: a top band for the header and
// title, a centre rect for the labels card.
const TOP_BAND_MAX = 22;
const CENTRE_TOP_MIN = 32;
const CENTRE_TOP_MAX = 70;
const CENTRE_LEFT_MIN = 18;
const CENTRE_LEFT_MAX = 82;

function pickItems(): Item[] {
  const items: Item[] = [];
  const used = new Set<string>();
  let attempts = 0;
  while (items.length < WORD_COUNT && attempts < WORD_COUNT * 15) {
    attempts++;
    const w = WORDS[Math.floor(Math.random() * WORDS.length)];
    if (used.has(w)) continue;

    // 4-96% keeps words off the edges.
    const top = 4 + Math.random() * 92;
    const left = 4 + Math.random() * 92;

    if (top < TOP_BAND_MAX) continue;
    if (
      top >= CENTRE_TOP_MIN &&
      top <= CENTRE_TOP_MAX &&
      left >= CENTRE_LEFT_MIN &&
      left <= CENTRE_LEFT_MAX
    ) continue;

    used.add(w);
    items.push({
      word: w,
      top,
      left,
      size: 1 + Math.random() * 2.5,         // 1 to 3.5 rem
      rotation: (Math.random() - 0.5) * 22,  // -11 to +11 deg
    });
  }
  return items;
}

export function WordCloud({ show }: { show: boolean }) {
  const items = useMemo(() => pickItems(), []);
  const [entered, setEntered] = useState(false);
  const [spotlight, setSpotlight] = useState(-1);

  // A tick late: CSS transitions only run on a change after first paint.
  useEffect(() => {
    if (show) {
      const t = window.setTimeout(() => setEntered(true), 30);
      return () => window.clearTimeout(t);
    }
    setEntered(false);
  }, [show]);

  useEffect(() => {
    if (!show) {
      setSpotlight(-1);
      return;
    }
    let timer: number | undefined;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      const idx = Math.floor(Math.random() * items.length);
      setSpotlight(idx);
      const focusDur = 1000 + Math.random() * 1000; // 1-2 s
      timer = window.setTimeout(() => {
        if (cancelled) return;
        setSpotlight(-1);
        const gap = 180 + Math.random() * 520; // 0.18-0.7 s
        timer = window.setTimeout(tick, gap);
      }, focusDur);
    };
    // Starts after the entry animation.
    timer = window.setTimeout(tick, 600);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [show, items.length]);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none select-none">
      {items.map((it, i) => {
        const isFocused = i === spotlight;
        const exitDelay = i * EXIT_STAGGER_MS;
        return (
          <span
            key={i}
            className="absolute font-light tracking-tight"
            style={{
              top: `${it.top}%`,
              left: `${it.left}%`,
              fontSize: `${it.size}rem`,
              color: "var(--foreground)",
              transform: `translate(-50%, -50%) rotate(${it.rotation}deg) scale(${
                entered ? (isFocused ? 1.18 : 1) : 0.9
              })`,
              filter: `blur(${isFocused ? 0 : 9}px)`,
              opacity: entered ? (isFocused ? 0.82 : 0.18) : 0,
              transitionProperty: "opacity, filter, transform",
              transitionDuration: isFocused ? "380ms" : `${FADE_MS}ms`,
              transitionTimingFunction: "ease-out",
              transitionDelay: entered ? "0ms" : `${exitDelay}ms`,
              willChange: "filter, opacity, transform",
            }}
          >
            {it.word}
          </span>
        );
      })}
    </div>
  );
}
