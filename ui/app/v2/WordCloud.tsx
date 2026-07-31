"use client";

import { useEffect, useMemo, useState } from "react";

import { WORDS } from "@/lib/words";

// Ambient backdrop for the V2 labels stage. Picks a handful of object
// names from `lib/words.txt`, scatters them across the available
// surface (the V2 wrapper, which spans the full viewport above the
// footer), and gently spotlights one at a time, un-blurring it for a
// random 1-2 second window before the next one is picked. Mount-in
// fades each word with a small stagger; unmount fades them all out
// uniformly. All transitions are CSS-driven (transform / opacity /
// filter) so the effect stays cheap on the GPU.

type Item = {
  word: string;
  top: number;     // percent from top of container
  left: number;    // percent from left of container
  size: number;    // rem
  rotation: number; // degrees
};

const WORD_COUNT = 28;

// Exclusion zones, keep words clear of the active UI so the
// overlay reads as ambient background, not visual clutter.
//
// Numbers are percentages of the wrapper (which spans the full
// viewport width and 100vh − 6rem in height). The wrapper's top
// edge sits at the page top because of the negative top margin we
// use in HomeView, so y=0 here lines up with the very top of the
// viewport.
//
//   - Top band (0–22%): the sticky TopNav header bar plus the
//     workspace title section that holds the project name + creator
//     metadata.
//   - Centre rect (32–70% top × 18–82% left): the V2 body card ,
//     "What do you want to detect?" headline, chip input, and the
//     Skip / Done buttons. The horizontal range is generous so the
//     centred max-w-2xl panel stays clear on wide screens too.
const TOP_BAND_MAX = 22;
const CENTRE_TOP_MIN = 32;
const CENTRE_TOP_MAX = 70;
const CENTRE_LEFT_MIN = 18;
const CENTRE_LEFT_MAX = 82;

// Pick a fresh set of words + positions on each mount. Only called
// once per mount via useMemo so the layout is stable across re-renders.
function pickItems(): Item[] {
  const items: Item[] = [];
  const used = new Set<string>();
  // Cap attempts so a stuck rejection loop can't hang the render.
  // With ~50% of the surface clear, WORD_COUNT * 15 gives plenty of
  // headroom for the rejection sampling.
  let attempts = 0;
  while (items.length < WORD_COUNT && attempts < WORD_COUNT * 15) {
    attempts++;
    const w = WORDS[Math.floor(Math.random() * WORDS.length)];
    if (used.has(w)) continue;

    // 4-96 keeps words from clipping at the edges. translate(-50%,-50%)
    // anchors at centre, so values near 50% put the word in the
    // visible centre band.
    const top = 4 + Math.random() * 92;
    const left = 4 + Math.random() * 92;

    // Reject anything that falls inside an exclusion zone, the
    // sample is re-rolled on the next iteration.
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
      size: 1 + Math.random() * 2.5,         // 1 - 3.5rem
      rotation: (Math.random() - 0.5) * 22,  // -11 to +11 deg
    });
  }
  return items;
}

export function WordCloud({ show }: { show: boolean }) {
  const items = useMemo(() => pickItems(), []);
  const [entered, setEntered] = useState(false);
  const [spotlight, setSpotlight] = useState(-1);

  // Flip `entered` one tick after `show` so the entry transition runs
  // (CSS transitions only fire on a class change after first paint).
  // When `show` flips false we drop entered → words fade out together,
  // and the parent unmounts us after the matching delay.
  useEffect(() => {
    if (show) {
      const t = window.setTimeout(() => setEntered(true), 30);
      return () => window.clearTimeout(t);
    }
    setEntered(false);
  }, [show]);

  // Random-spotlight loop. Picks a word, holds it focused for 1-2 s,
  // releases it for a brief gap, then advances. Cleared whenever the
  // overlay is dismissed so we don't keep ticking after exit.
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
    // Hold off until the entry animation has had a moment to land.
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
        const enterDelay = i * 18; // ms, small stagger on the way in
        return (
          <span
            key={i}
            className="absolute font-light tracking-tight"
            style={{
              top: `${it.top}%`,
              left: `${it.left}%`,
              fontSize: `${it.size}rem`,
              // Monochrome on both themes: foreground colour at the
              // current theme so words paint as soft black on white
              // in light mode and soft white on black in dark.
              color: "var(--foreground)",
              // Default state stays heavily blurred + plain size. When
              // the spotlight lands on a word it un-blurs and scales
              // up a touch so the focus reads at a glance.
              transform: `translate(-50%, -50%) rotate(${it.rotation}deg) scale(${
                entered ? (isFocused ? 1.18 : 1) : 0.9
              })`,
              filter: `blur(${isFocused ? 0 : 9}px)`,
              opacity: entered ? (isFocused ? 0.82 : 0.18) : 0,
              transitionProperty: "opacity, filter, transform",
              transitionDuration: isFocused ? "380ms" : "640ms",
              transitionTimingFunction: "ease-out",
              transitionDelay: entered ? "0ms" : `${enterDelay}ms`,
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
